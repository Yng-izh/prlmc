"""v9 objective: per-token knowledge protection on top of the v7 targeted recovery.

The v7 protection targets are aggregate (mean answer CE hinge + first-token KL
anchor). The observed fusion damage is entity/date substitution inside the
answer body (e.g. "Bob Dylan" -> "George Harrison", 1998 -> 2002), which a
mean-CE hinge barely sees when the answer has several tokens and only one is
flipped.

This module extends `natural_merge_v4_objective` with:

1. full per-token Selection reference distributions over the whole answer
   (`sel_token_log_probs`), stored CPU fp16, one row per answer token;
2. a per-token degraded-token penalty: for every answer position where the
   Selection reference predicted the gold token (argmax == gold), the fused
   stream pays a hinge `relu(ce_t - sel_ce_t - margin)` plus a KL pull toward
   the Selection distribution at that position;
3. Selection-wrong positions carry no imitation term, so recovery of discarded
   knowledge is unobstructed (the targeted recovery CE + Full teacher stay
   exactly as in v7).

Everything else (recovery gate, cohorts, EOS handling, optimizer, frozen
reader) is inherited from the v7 objective unchanged.
"""

import json
import contextlib
import torch
import torch.nn.functional as F

from natural_merge_v2_train import answer_logits, realized_residual
from natural_merge_v4_objective import balanced_coefficients, objective_terms
import natural_merge_v4_objective  # noqa: F401  (sealed objective module, inherited helpers)


V8_TRAINING_OVERRIDES = dict(
    token_protection_weight=.5,
    token_protection_margin=.30,
    token_kl_weight=.25,
    objective="v8: context-balanced targeted recovery content CE + eligible Full content KL"
              " + protection content hinge + first-token Selection KL anchor"
              " + per-token degraded-token protection (Selection-correct positions only:"
              " relu(ce_t - sel_ce_t - 0.30) hinge at weight 0.5 and Selection KL at weight 0.25)"
              " + separate EOS CE",
)


def answer_logits_batch(model, qas, device):
    """`answer_logits` for a whole group in ONE forward.

    Right padding is exact for a causal reader: no real token can attend a pad, and
    the pads' own outputs are never read, so per-sample logits are bit-for-bit what
    the single-sample calls produce. The memory prefix is repeated per sample inside
    `cat_memory_and_hiddens` (it does `self.memory[idx].unsqueeze(0).repeat(len(
    hidden_states), 1, 1)`), so one forward really does serve the whole batch, and a
    summed-loss backward gives the same gradient as N separate backwards.
    """
    import torch
    width = max(len(qa["prompt_ids"]) + len(qa["answer_ids"]) for qa in qas)
    ids = torch.zeros(len(qas), width, dtype=torch.long, device=device)
    for row, qa in enumerate(qas):
        seq = qa["prompt_ids"] + qa["answer_ids"]
        ids[row, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    logits = model(input_ids=ids, use_cache=False, return_dict=True).logits
    out = []
    for row, qa in enumerate(qas):
        prompt = len(qa["prompt_ids"])
        answer = len(qa["answer_ids"])
        out.append((logits[row, prompt-1:prompt-1+answer].float(),
                    torch.tensor(qa["answer_ids"], dtype=torch.long, device=device)))
    return out


def _group_logits(model, qas, device, start, chunk):
    """Batched when the group has more than one QA, otherwise the sealed path."""
    group = qas[start:start+chunk]
    if len(group) > 1:
        return answer_logits_batch(model, group, device)
    return [answer_logits(model, group[0], device)]


def token_protection_terms(logits, gold, sel_log_probs, sel_correct,
                           margin=.30):
    """Degraded-token hinge + Selection KL over Selection-correct positions.

    `logits` covers prompt+answer rows as in `answer_logits`; positions are the
    first len(gold)-1 rows (answer content). Returns scalar tensors (zero when
    Selection had no correct token in this answer).
    """
    content = logits[:-1]
    ce_t = F.cross_entropy(content.float(), gold[:-1], reduction="none")
    sel = sel_log_probs.to(content.device).float()
    sel_ce_t = -sel.gather(1, gold[:-1].unsqueeze(1)).squeeze(1)
    hinge_t = F.relu(ce_t - sel_ce_t - margin)
    # F.kl_div(log_target=False) expects *probabilities* as target; passing
    # log-probs (negative values) produces log(negative) = NaN. Feed exp()
    # instead; a zero target contributes zero, so fp16-underflowed mass is safe.
    kl_t = F.kl_div(F.log_softmax(content.float(), -1), sel.exp(),
                    reduction="none").sum(-1)
    mask = sel_correct.to(content.device)
    denom = mask.sum().clamp(min=1).float()
    hinge = (hinge_t * mask).sum() / denom
    kl = (kl_t * mask).sum() / denom
    return hinge, kl, int(mask.sum()), int(mask.numel())


def qa_cohorts(origins, qas, index_list, window_blocks, teacher_seen=None):
    """Provenance-exact cohort label per QA, with the local teacher's evidence gated.

    `NativeOrigins.cohorts` additionally demands that every resident origin appear
    in `qas`, i.e. one supervised QA per resident block -- up to 120 forwards per
    event on the 50-block stream. The labels only need the provenance record the
    origins object already holds, so they are computed from the same rule here and
    the coverage of the resident pool is reported instead of assumed.

    Three outcomes:

    protect  the origin is still resident (kept carrier or untouched tail), so a
             Selection forward over the same pool is a valid anchor and no local
             teacher is needed at all;
    recover  the origin was discarded by this merge AND the local full teacher
             really holds it, so the KL is distillation and not noise fitting;
    skip     the origin is gone (truncated by an earlier merge) or the teacher
             could not see it. Supervising against a blind teacher teaches
             guessing, so such a QA gets no term in this event.

    `teacher_seen` maps qa id -> bool; a missing key means "assume visible", which
    is the pre-existing behaviour for single-window events whose teacher memory is
    the uncompressed stream.
    """
    blocks = list(origins.blocks)
    if len(blocks) < window_blocks:
        raise ValueError("incomplete provenance window")
    if len({q["source_index"] for q in qas}) != len(qas):
        raise ValueError("one QA per original context required")
    seen = dict(teacher_seen or {})
    known = set(blocks)
    carriers = {blocks[i] for i in index_list}
    tail = set(blocks[window_blocks:])
    current = set(blocks[:window_blocks])
    discarded = current - carriers
    labels, skipped = [], []
    for q in qas:
        origin = q["source_index"]
        visible = bool(seen.get(q["id"], True))
        if origin in carriers:
            labels.append(("protect", "retained_carrier"))
        elif origin in tail:
            labels.append(("protect", "untouched_tail"))
        elif origin in discarded and visible:
            labels.append(("recover", "discarded_this_round"))
        elif origin in discarded:
            labels.append(("skip", "teacher_evidence_missing"))
            skipped.append(q["id"])
        else:
            labels.append(("skip", "previously_discarded_out_of_residence"))
            skipped.append(q["id"])
    covered = len({q["source_index"] for q in qas if q["source_index"] in known})
    return labels, dict(resident_origins=len(known), covered_origins=covered,
                        coverage=covered / max(1, len(known)), skipped=list(skipped),
                        skip_teacher_missing=sum(1 for _, r in labels
                                                 if r == "teacher_evidence_missing"),
                        skip_out_of_residence=sum(1 for _, r in labels
                                                  if r == "previously_discarded_out_of_residence"))


def full_targets(manager, qas, chunk=1):
    """Full-teacher targets + per-token Selection reference from ONE forward.

    `natural_merge_v4_objective.full_targets` and `selection_token_targets` each ran
    their own `answer_logits` over the same teacher memory for the same QA. Both
    read the same `content_logits`, so they are computed together here: the output
    is byte-identical to the two-pass version (the v7 teacher keeps raw-dtype
    log-softmax, the per-token reference keeps the fp32 one) at half the forwards.
    """
    result = {}
    with torch.no_grad():
        chunk = max(1, int(chunk or 1))
        pairs = []
        for start in range(0, len(qas), chunk):
            pairs.extend(_group_logits(manager.model, qas, manager._memory.device,
                                       start, chunk))
        for qa, (logits, gold) in zip(qas, pairs):
            content_logits, content_gold = logits[:-1], gold[:-1]
            # v7 teacher: unchanged dtype path, unchanged numbers.
            teacher_log_probs = F.log_softmax(content_logits, -1)
            # v9 per-token Selection reference: fp32 before the log-softmax, as before.
            log_probs = F.log_softmax(content_logits.float(), -1)
            correct = content_logits.argmax(-1) == content_gold
            result[qa["id"]] = dict(
                gold_ce=float(F.cross_entropy(content_logits, content_gold)),
                eos_ce=float(F.cross_entropy(logits[-1:], gold[-1:])),
                log_probs=teacher_log_probs.to(device="cpu", dtype=torch.float16),
                sel_token_log_probs=log_probs.to(device="cpu", dtype=torch.float16),
                sel_token_correct=correct.cpu(),
                sel_token_ce=log_probs.gather(
                    1, content_gold.unsqueeze(1)).squeeze(1).cpu(),
                context_sha256=qa["context_sha256"],
                prompt_ids=list(qa["prompt_ids"]), answer_ids=list(qa["answer_ids"]))
            del logits, gold, content_logits, content_gold, teacher_log_probs, log_probs
    return result


def train_event(manager, merge, optimizer, qas, targets, variant, origins,
                scaler=None, plan=None, adapter=None):
    """v9 train_event: v8 loop with the v9 structure wired in.

    The sealed v4/v7 objective file is not modified; this duplicates the v7
    body verbatim except for the marked v8 lines.
    """
    from methods.prlmc.probes import (_pool_in_place, _compacted_mask,
        active_mask_override, differentiable_memory)
    from methods.prlmc.consolidate import _select_block_indices, select_bridge_aware
    from natural_merge_runtime import install
    if plan is None:
        raise ValueError("v9 train_event requires an explicit plan")
    allowed = plan["variants"]
    if variant not in allowed or not qas or len({q["id"] for q in qas}) != len(qas):
        raise ValueError("known variant and unique nonempty training QAs required")
    for qa in qas:
        teacher = targets.get(qa["id"], {})
        if any(teacher.get(k) != qa[k] for k in ("context_sha256", "prompt_ids", "answer_ids")):
            raise ValueError("Full answer-only teacher provenance mismatch")
    geometry = plan.get("geometry") or dict(budget_blocks=20, window_blocks=19, target_blocks=9)
    m = manager
    window, target = geometry["window_blocks"]*m.block, geometry["target_blocks"]*m.block
    if len(origins.blocks)*m.block != m.used:
        raise ValueError("native-origin count differs from active memory")
    start = m.total-m.used
    source = m._memory[:, start:start+window].contiguous()
    ages = m.window_block_ages(start, window) if plan.get("geometry") else None
    # ---- v9: freeze mask, archive pins and the two-stage selector ----------
    # The supervised event must pick carriers with the same rule the evaluation
    # stream compacts with, otherwise the write-once lock is trained against a
    # window the reader never sees.
    if hasattr(m, "fused_window"):
        frozen, pinned = m.fused_window(start, window)
        m.pinned_blocks = pinned
    else:
        frozen, pinned = None, None
    from methods.prlmc.consolidate import select_window_carrier_blocks
    picked = select_window_carrier_blocks(
        window_memory=source,
        target_blocks=target // m.block,
        block_size=m.block,
        block_ages=ages,
        condensed_slot_mask=None,
    )
    if pinned:
        picked = build_pinned_teacher_selection(pinned, picked, target // m.block)
    indices = torch.as_tensor(picked, device=source.device, dtype=torch.long)
    # -----------------------------------------------------------------------
    index_list = indices.cpu().tolist()
    window_blocks = window // m.block
    teacher_seen = {q["id"]: targets[q["id"]].get("teacher_evidence", True) for q in qas}
    cohorts, provenance_coverage = qa_cohorts(origins, qas, index_list, window_blocks,
                                              teacher_seen=teacher_seen)
    # A skipped QA must not contribute anything: not the shared CE, not the
    # teacher KL, and not the Selection reference it would otherwise anchor.
    active = [i for i, (cohort, _) in enumerate(cohorts) if cohort != "skip"]
    if len(active) != len(qas):
        qas = [qas[i] for i in active]
        cohorts = [cohorts[i] for i in active]
    recover_ids = [i for i, (cohort, _) in enumerate(cohorts) if cohort == "recover"]
    protect_count = len(qas)-len(recover_ids)
    if not recover_ids or not protect_count:
        # A cohort can legitimately vanish: the supervised set is small (<= QA_CAP),
        # a merge may keep every supervised window QA (recover empty), or a deep
        # event may have no realisable no-compression teacher. `balanced_coefficients`
        # already normalises with max(count, 1), so the surviving cohort just carries
        # the loss; record the degradation instead of aborting a 704-update run.
        print(f"[cohort] degraded event: recover={len(recover_ids)}"
              f" protect={protect_count} of {len(qas)} QAs")
    kwargs = dict(target=target, block=m.block, carrier_indices=indices)
    if getattr(merge, "bridge_blocks", None) is not None:
        kwargs.update(frozen_slots=frozen, pinned_blocks=pinned)
    carriers = source.reshape(m.num_layers, window_blocks, m.block, m.hidden).index_select(1, indices).flatten(1, 2)
    mask = _compacted_mask(m, m.used-window+target, m._memory.device)
    full_ce = [targets[q["id"]]["gold_ce"] for q in qas]
    selection_ce = []
    selection_first = []
    with torch.no_grad(), _pool_in_place(m, m._memory.device, carriers, window, target) as injection:
        with active_mask_override(m.model, mask):
            chunk = max(1, int(plan.get("qa_batch", 1) or 1))
            for start in range(0, len(qas), chunk):
                injection.reinject()
                for logits, gold in _group_logits(m.model, qas, m._memory.device,
                                                  start, chunk):
                    selection_ce.append(float(F.cross_entropy(logits[:-1], gold[:-1])))
                    selection_first.append(F.softmax(logits[0].float(), -1).cpu())
    if not torch.isfinite(torch.tensor(full_ce+selection_ce)).all():
        raise ValueError("nonfinite reference CE")
    eligible = [cohort == "recover" and full_ce[i] <= plan["teacher_max_gold_ce"]
                for i, (cohort, _) in enumerate(cohorts)]
    eligible_count = sum(eligible)
    coefficients = balanced_coefficients(qas, cohorts, eligible, full_ce, selection_ce, plan)
    optimizer.zero_grad(set_to_none=True)
    # The no_grad call used to compute the same operator on the same input as the
    # graph-bearing call further down; `merge` has no dropout, so the two results
    # are identical and one forward is pure duplication.
    rebuilt = merge(source, **kwargs)
    from tools.v9_geometry_50block import DISTILL_LAYERS as _TAPS
    _pre_windows = {layer: source[layer].detach() for layer in _TAPS}
    _post_windows = {layer: rebuilt[layer] for layer in _TAPS}
    source_host = source.detach().cpu() if source.is_cuda else source.detach()
    del source, carriers
    leaf = rebuilt.detach().requires_grad_(True)
    # The slots this event writes as bridges are only marked in the manager at
    # `install`, which runs after the backward. Hand them to the adapter now, so
    # the supervised read sees the fusion this event actually produced instead of
    # only the bridges of previous rounds.
    if adapter is not None and hasattr(adapter, "set_event_bridge"):
        fusion = getattr(merge, "last_fusion", None) or {}
        offsets = fusion.get("bridge_slot_offsets") or []
        if offsets:
            base = m.total - (m.used - window + target)
            index = torch.as_tensor([base + int(o) for o in offsets], dtype=torch.long,
                                    device=m._memory.device)
            adapter.set_event_bridge(index[(index >= 0) & (index < m.total)])
        else:
            adapter.set_event_bridge(None)
    tpw = plan["token_protection_weight"]
    tkw = plan["token_kl_weight"]
    rows = []; recovery_sum = kd_sum = protect_sum = eos_sum = token_sum = total = 0.
    token_stats = dict(correct_positions=0, total_positions=0)
    chunk = max(1, int(plan.get("qa_batch", 1) or 1))
    readout_weight = float(plan.get("readout_weight", 0.05))
    readout_term = torch.zeros((), device=m._memory.device)
    if _pre_windows and len(qas) > 0:
        query_hidden = _query_hidden_states(m.model, qas[0], m._memory.device)
        readout_term = compute_local_window_readout_loss(
            m.model, _pre_windows, _post_windows, query_hidden)
        print(json.dumps({"readout_loss": float(readout_term.detach()),
                          "readout_weight": readout_weight}), flush=True)
    with _pool_in_place(m, m._memory.device, leaf, window, target) as injection:
        with active_mask_override(m.model, mask):
            for start in range(0, len(qas), chunk):
                group = list(range(start, min(start + chunk, len(qas))))
                pool = injection.reinject()
                with differentiable_memory(m.model, pool):
                    # Only the QA read uses the reader adapter; the write path stays
                    # native. The activation has to survive the backward pass:
                    # gradient checkpointing recomputes this forward inside backward,
                    # and a toggled-off adapter saves a different tensor set there.
                    if adapter is not None and getattr(adapter, "stage_active", True):
                        adapter.on()
                    if len(group) > 1:
                        pairs = answer_logits_batch(m.model, [qas[j] for j in group],
                                                    m._memory.device)
                    else:
                        pairs = [answer_logits(m.model, qas[group[0]], m._memory.device)]
                losses = []
                for k, i in enumerate(group):
                    qa = qas[i]
                    logits, gold = pairs[k]
                    teacher = targets[qa["id"]]["log_probs"] if eligible[i] else None
                    anchor = None
                    if coefficients[i]["protect"] > 0:
                        anchor = F.kl_div(F.log_softmax(logits[0].float(), -1),
                                          selection_first[i].to(logits.device), reduction="sum")
                    ce, kd, recovery, distillation, protection, eos_ce, eos = objective_terms(
                        logits, gold, selection_ce[i], teacher, coefficients[i], anchor,
                        eos_scheme=plan.get("eos_scheme", "legacy"),
                        eos_token_weight=plan.get("eos_token_weight", .2))
                    # ---- v8: per-token degraded-token protection ----
                    sel_lp = targets[qa["id"]]["sel_token_log_probs"]
                    sel_ok = targets[qa["id"]]["sel_token_correct"]
                    tp_hinge, tp_kl, n_ok, n_tot = token_protection_terms(
                        logits, gold, sel_lp, sel_ok, plan["token_protection_margin"])
                    token_term = tpw*tp_hinge + tkw*tp_kl
                    # --------------------------------------------------
                    loss = recovery + distillation + protection + eos + token_term
                    if not torch.isfinite(loss): raise ValueError("nonfinite v9 objective")
                    losses.append(loss)
                    recovery_sum += float(recovery.detach()); kd_sum += float(distillation.detach())
                    protect_sum += float(protection.detach()); eos_sum += float(eos.detach())
                    token_sum += float(token_term.detach()); total += float(loss.detach())
                    token_stats["correct_positions"] += n_ok
                    token_stats["total_positions"] += n_tot
                    rows.append(dict(id=qa["id"], source_index=qa["source_index"],
                        context_sha256=qa["context_sha256"], cohort=cohorts[i][0], origin_status=cohorts[i][1],
                        full_ce=full_ce[i], selection_ce=selection_ce[i], shared_ce=float(ce.detach()),
                        weight=coefficients[i]["weight"], eos_ce=float(eos_ce.detach()), eos_term=float(eos.detach()), teacher_eligible=eligible[i], teacher_kl=float(kd.detach()),
                        recovery_term=float(recovery.detach()), protection_term=float(protection.detach()),
                        protection_anchor=float(anchor.detach()) if anchor is not None else 0.,
                        token_hinge=float(tp_hinge.detach()), token_kl=float(tp_kl.detach()),
                        answer_token_count=len(qa["content_ids"])))
                # One backward for the whole group: the losses are summed, so the
                # gradient is the sum of the per-QA gradients, exactly as before.
                group_loss = sum(losses)
                (scaler.scale(group_loss) if scaler else group_loss).backward()
                if adapter is not None:
                    adapter.off()      # recomputation happened inside backward
                del pool, pairs, losses, group_loss, logits, gold
                del ce, kd, recovery, distillation, protection, eos_ce, eos, loss
    if leaf.grad is None or not torch.isfinite(leaf.grad).all():
        raise ValueError("missing/nonfinite frozen-reader memory gradient")
    source = source_host.to(m._memory.device)
    # One backward through the shared merge graph: the pool gradient and the
    # readout distillation term are two outputs of the same forward, so they
    # must be backpropped together or the graph is freed by the first call.
    if readout_weight > 0 and readout_term.requires_grad:
        torch.autograd.backward([rebuilt, readout_weight * readout_term],
                                [leaf.grad, torch.ones((), device=leaf.grad.device,
                                                       dtype=leaf.grad.dtype)])
    else:
        torch.autograd.backward(rebuilt, leaf.grad)
    del rebuilt, leaf
    if scaler: scaler.unscale_(optimizer)
    adapter_grads = None
    if adapter is not None:
        params = [p for p in adapter.parameters() if p.requires_grad]
        grads = [p.grad for p in params if p.grad is not None]
        if len(grads) != len(params):
            raise ValueError("reader adapter received no gradient")
        norms = [float(g.norm()) for g in grads]
        if not any(n > 0. for n in norms) and getattr(adapter, "stage_active", True):
            # Stage 1 deliberately keeps the adapter frozen and structurally off, so
            # an all-zero gradient there is the contract, not a failure.
            raise ValueError("reader adapter gradient is identically zero")
        torch.nn.utils.clip_grad_norm_(params, plan["gradient_clip"])
        adapter_grads = dict(tensors=len(params), nonzero=sum(n > 0. for n in norms),
                             grad_norm=float(sum(n * n for n in norms) ** .5))
    gradients = {n:p.grad for n,p in merge.named_parameters() if p.grad is not None}
    if not gradients or any(not torch.isfinite(g).all() for g in gradients.values()):
        raise ValueError("missing/nonfinite module gradient")
    gradient_norms = {n:float(g.norm()) for n,g in gradients.items()}
    norm = float(torch.nn.utils.clip_grad_norm_(merge.parameters(), plan["gradient_clip"]))
    if not torch.isfinite(torch.tensor(norm)): raise ValueError("nonfinite module gradient norm")
    if scaler: scaler.step(optimizer); scaler.update()
    else: optimizer.step()
    with torch.no_grad():
        current = merge(source, **kwargs)
        carriers = source.reshape(m.num_layers, window_blocks, m.block, m.hidden).index_select(1, indices).flatten(1, 2)
        stored = realized_residual(current, carriers); fusion = dict(merge.last_fusion)
        # V10 replaces whole native slots instead of adding a bounded residual,
        # so the residual ratio is recorded for telemetry only (no gate).
        # v9 M4: a slot frozen by an earlier round must never be written again.
        carrier_frozen = None
        if frozen is not None:
            carrier_frozen = (frozen.to(m._memory.device).reshape(-1, m.block)
                              .index_select(0, indices.to(m._memory.device)).reshape(-1))
        bridge_offsets = getattr(merge, "last_fusion", {}).get("bridge_slot_offsets") or []
        if bridge_offsets and carrier_frozen is not None:
            offsets = torch.as_tensor(bridge_offsets, dtype=torch.long,
                                      device=carrier_frozen.device)
            if int(offsets.max()) >= carrier_frozen.numel():
                raise ValueError("bridge slot offsets fall outside the carrier axis")
            hit = carrier_frozen[offsets]
            if bool(hit.any()):
                raise ValueError("frozen bridge slot was written again")
        install(m, current, window, selected_blocks=indices if plan.get("geometry") else None,
                fused_offsets=bridge_offsets)
    origins.compact(index_list, window_blocks)
    if not rows:
        raise ValueError("train_event produced no supervised QA rows")
    means = {}
    for cohort in ("recover", "protect"):
        subset = [r for r in rows if r["cohort"] == cohort]
        # An event may end up single-cohort: a merge can keep every supervised
        # window QA (recover empty), or a deep event may have no realisable
        # no-compression teacher. Report zeros for the absent cohort instead of
        # dividing by an empty list -- this is the aggregation half of the same
        # condition `train_event` warns about above.
        if not subset:
            means[cohort] = dict(count=0, shared_ce=0.0, selection_ce=0.0,
                                 shared_minus_selection_ce=0.0)
            continue
        means[cohort] = dict(count=len(subset), shared_ce=sum(r["shared_ce"] for r in subset)/len(subset),
            selection_ce=sum(r["selection_ce"] for r in subset)/len(subset),
            shared_minus_selection_ce=sum(r["shared_ce"]-r["selection_ce"] for r in subset)/len(subset))
    return dict(objective=total, gold_ce=sum(r["shared_ce"] for r in rows)/len(rows),
        weighted_gold_ce=recovery_sum, teacher_kl=kd_sum/plan["teacher_weight"],
        protection_hinge=protect_sum, token_protection=token_sum,
        token_correct_positions=token_stats["correct_positions"],
        token_total_positions=token_stats["total_positions"],
        protection_anchor=sum(r["protection_anchor"] for r in rows)/len(rows), eos_loss=eos_sum,
        teacher_eligible_count=eligible_count,
        teacher_weight=plan["teacher_weight"], original_qa_count=len(qas), qas=rows,
        cohorts=means, carrier_indices=index_list, retained_native_origins=list(origins.blocks),
        gradient_norms_before_clip=gradient_norms, grad_norm_before_clip=norm,
        provenance=provenance_coverage,
        fusion=fusion, realized_residual=stored, reader_adapter=adapter_grads,
        window_blocks=window_blocks, target_blocks=target // m.block,
        selection_reference="same pre-event student history; current window uses identical native carriers",
        qa_measurement="content-only CE/KL/teacher eligibility; per-token Selection-correct protection;"
                       " separate EOS; context-balanced loss before update; stored budget after update")
import torch
import torch.nn.functional as F

from tools.v9_geometry_50block import DISTILL_LAYERS


def _query_hidden_states(model, qa, device):
    """Layer taps for one QA query, in the same forward shape the reader uses."""
    import torch
    ids = torch.tensor([qa["prompt_ids"] + qa["answer_ids"]], dtype=torch.long, device=device)
    out = model(input_ids=ids, use_cache=False, return_dict=True, output_hidden_states=True)
    hidden = out.hidden_states  # [embeddings, layer1, ...]
    return {layer: hidden[layer + 1].float() for layer in DISTILL_LAYERS}


def compute_local_window_readout_loss(model, pre_windows, post_windows, query_hidden,
                                      query_tokens=8, slot_stride=8):
    """Match the reader's attention readout from the compacted window to the
    uncompacted one at the distillation taps.

    MemoryLLM is GQA (32 query heads, 8 KV heads): K/V are expanded to the query
    head count before the readout. The post window keeps its graph so the term
    trains the merge operator, so slots are strided and math stays in the model
    dtype to bound memory.
    """
    import torch
    import torch.nn.functional as F

    cfg = model.config
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_heads) or n_heads)
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // n_heads))
    repeat = max(1, n_heads // n_kv)
    layers = model.model.layers

    device = query_hidden[DISTILL_LAYERS[0]].device
    total = torch.zeros((), device=device, dtype=torch.float32)

    for layer in DISTILL_LAYERS:
        block = layers[layer]
        attn = block.self_attn
        model_dtype = block.input_layernorm.weight.dtype

        def project(memory):
            normed = block.input_layernorm(memory.unsqueeze(0))
            k = F.linear(normed, attn.k_proj.weight)
            v = F.linear(normed, attn.v_proj.weight)
            k = k.view(1, -1, n_kv, head_dim).transpose(1, 2)
            v = v.view(1, -1, n_kv, head_dim).transpose(1, 2)
            if repeat > 1:
                k = k.repeat_interleave(repeat, dim=1)
                v = v.repeat_interleave(repeat, dim=1)
            return k.float(), v.float()

        query = F.linear(query_hidden[layer].to(model_dtype), attn.q_proj.weight)
        query = query.view(1, -1, n_heads, head_dim).transpose(1, 2)[:, :, -query_tokens:, :].float()
        scale = head_dim ** -0.5

        with torch.no_grad():
            k_pre, v_pre = project(pre_windows[layer][::slot_stride])
            weights = F.softmax(torch.matmul(query, k_pre.transpose(-1, -2)) * scale, dim=-1)
            readout_pre = torch.matmul(weights, v_pre)

        k_post, v_post = project(post_windows[layer][::slot_stride])
        weights_post = F.softmax(torch.matmul(query, k_post.transpose(-1, -2)) * scale, dim=-1)
        readout_post = torch.matmul(weights_post, v_post)

        cosine = 1.0 - F.cosine_similarity(readout_post.reshape(1, -1),
                                           readout_pre.reshape(1, -1), dim=-1).mean()
        l2 = F.smooth_l1_loss(readout_post, readout_pre)
        total = total + (cosine + 0.5 * l2)

    return total / len(DISTILL_LAYERS)


def build_pinned_teacher_selection(pinned_blocks, default_selected_blocks, target_blocks=9):
    """Force the evidence blocks to survive a 19->9 merge.

     are the resident block indices holding gold evidence for the
    current event (the manager's fused/pinned window). They are kept first and
    the V10 selection fills the remaining budget, so the teacher pool never
    loses the block a supervised question depends on.
    """
    pinned = sorted({int(b) for b in (pinned_blocks or [])})
    merged = pinned[:target_blocks]
    for idx in default_selected_blocks:
        if len(merged) >= target_blocks:
            break
        if int(idx) not in merged:
            merged.append(int(idx))
    return sorted(merged)
