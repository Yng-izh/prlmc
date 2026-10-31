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

import contextlib
import torch
import torch.nn.functional as F

from natural_merge_v2_train import answer_logits, realized_residual
from natural_merge_v4_objective import (balanced_coefficients, full_targets
                                        as v7_full_targets, objective_terms)
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


def selection_token_targets(manager, qas):
    """Per-token Selection reference for the whole answer content.

    Stored per QA: `sel_token_log_probs` [T, V] CPU fp16 and
    `sel_token_correct` [T] bool (argmax == gold). The fused stream must
    preserve what Selection already got right; Selection-wrong positions are
    left to the recovery losses.
    """
    result = {}
    with torch.no_grad():
        for qa in qas:
            logits, gold = answer_logits(manager.model, qa, manager._memory.device)
            content_logits, content_gold = logits[:-1], gold[:-1]
            log_probs = F.log_softmax(content_logits.float(), -1)
            correct = content_logits.argmax(-1) == content_gold
            result[qa["id"]] = dict(
                sel_token_log_probs=log_probs.to(device="cpu", dtype=torch.float16),
                sel_token_correct=correct.cpu(),
                sel_token_ce=log_probs.gather(1, content_gold.unsqueeze(1)).squeeze(1).cpu(),
                context_sha256=qa["context_sha256"],
                prompt_ids=list(qa["prompt_ids"]), answer_ids=list(qa["answer_ids"]))
    return result


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


def full_targets(manager, qas):
    """v7 Full-teacher targets + per-token Selection reference in one record."""
    targets = v7_full_targets(manager, qas)
    for qa_id, extra in selection_token_targets(manager, qas).items():
        targets[qa_id].update(extra)
    return targets


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
    _bridge = int(getattr(merge, "bridge_blocks", 0))
    if _bridge or pinned:
        indices = torch.as_tensor(select_bridge_aware(
            source, target, m.block, m._window_importance(start, window), None,
            m.select_weights, block_ages=ages, pinned=pinned,
            bridge_blocks=_bridge)["index"], device=source.device, dtype=torch.long)
    else:
        indices = _select_block_indices(source, target, m.block,
            m._window_importance(start, window), None, m.select_weights, block_ages=ages)
    # -----------------------------------------------------------------------
    index_list = indices.cpu().tolist()
    window_blocks = window // m.block
    cohorts = origins.cohorts(qas, index_list, window_blocks)
    recover_ids = [i for i, (cohort, _) in enumerate(cohorts) if cohort == "recover"]
    protect_count = len(qas)-len(recover_ids)
    if not recover_ids or not protect_count:
        raise ValueError("both recovery and preservation cohorts required")
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
            for qa in qas:
                injection.reinject()
                logits, gold = answer_logits(m.model, qa, m._memory.device)
                selection_ce.append(float(F.cross_entropy(logits[:-1], gold[:-1])))
                selection_first.append(F.softmax(logits[0].float(), -1).cpu())
    if not torch.isfinite(torch.tensor(full_ce+selection_ce)).all():
        raise ValueError("nonfinite reference CE")
    eligible = [cohort == "recover" and full_ce[i] <= plan["teacher_max_gold_ce"]
                for i, (cohort, _) in enumerate(cohorts)]
    eligible_count = sum(eligible)
    coefficients = balanced_coefficients(qas, cohorts, eligible, full_ce, selection_ce, plan)
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad(): value = merge(source, **kwargs)
    source_host = source.detach().cpu() if source.is_cuda else source.detach()
    del source, carriers
    leaf = value.detach().requires_grad_(True)
    tpw = plan["token_protection_weight"]
    tkw = plan["token_kl_weight"]
    rows = []; recovery_sum = kd_sum = protect_sum = eos_sum = token_sum = total = 0.
    token_stats = dict(correct_positions=0, total_positions=0)
    with _pool_in_place(m, m._memory.device, leaf, window, target) as injection:
        with active_mask_override(m.model, mask):
            for i, qa in enumerate(qas):
                pool = injection.reinject()
                with differentiable_memory(m.model, pool):
                    # Only the QA read uses the reader adapter; the write path
                    # that produced this memory never did.
                    with (adapter.active() if adapter is not None else contextlib.nullcontext()):
                        logits, gold = answer_logits(m.model, qa, m._memory.device)
                teacher = targets[qa["id"]]["log_probs"] if eligible[i] else None
                anchor = None
                if coefficients[i]["protect"] > 0:
                    anchor = F.kl_div(F.log_softmax(logits[0].float(), -1),
                                      selection_first[i].to(logits.device), reduction="sum")
                ce, kd, recovery, distillation, protection, eos_ce, eos = objective_terms(
                    logits, gold, selection_ce[i], teacher, coefficients[i], anchor)
                # ---- v8: per-token degraded-token protection ----
                sel_lp = targets[qa["id"]]["sel_token_log_probs"]
                sel_ok = targets[qa["id"]]["sel_token_correct"]
                tp_hinge, tp_kl, n_ok, n_tot = token_protection_terms(
                    logits, gold, sel_lp, sel_ok, plan["token_protection_margin"])
                token_term = tpw*tp_hinge + tkw*tp_kl
                # --------------------------------------------------
                loss = recovery + distillation + protection + eos + token_term
                if not torch.isfinite(loss): raise ValueError("nonfinite v9 objective")
                (scaler.scale(loss) if scaler else loss).backward()
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
                del pool, logits, ce, kd, recovery, distillation, protection, eos_ce, eos, loss
    if leaf.grad is None or not torch.isfinite(leaf.grad).all():
        raise ValueError("missing/nonfinite frozen-reader memory gradient")
    source = source_host.to(m._memory.device)
    rebuilt = merge(source, **kwargs); torch.autograd.backward(rebuilt, leaf.grad)
    del rebuilt, leaf, value
    if scaler: scaler.unscale_(optimizer)
    adapter_grads = None
    if adapter is not None:
        params = [p for p in adapter.parameters() if p.requires_grad]
        grads = [p.grad for p in params if p.grad is not None]
        if len(grads) != len(params):
            raise ValueError("reader adapter received no gradient")
        norms = [float(g.norm()) for g in grads]
        if not any(n > 0. for n in norms):
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
        if stored["stored_residual_rms_ratio"] > plan["global_rms_budget"]+1e-6:
            raise ValueError("actual stored C/D budget exceeded")
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
    means = {}
    for cohort in ("recover", "protect"):
        subset = [r for r in rows if r["cohort"] == cohort]
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
        fusion=fusion, realized_residual=stored, reader_adapter=adapter_grads,
        window_blocks=window_blocks, target_blocks=target // m.block,
        selection_reference="same pre-event student history; current window uses identical native carriers",
        qa_measurement="content-only CE/KL/teacher eligibility; per-token Selection-correct protection;"
                       " separate EOS; context-balanced loss before update; stored budget after update")
