"""Context-balanced content supervision and independent true EOS loss."""
import torch
from collections import Counter
import torch.nn.functional as F

from prepare_natural_merge_v4 import TRAIN_V4
from natural_merge_v2_train import answer_logits, teacher_kl, realized_residual
from natural_merge_v3_objective import NativeOrigins as BaseOrigins
from prepare_natural_merge_v4 import event_qas


class NativeOrigins(BaseOrigins):
    def cohorts(self, qas, indices, window=19):
        unique = {q["source_index"]:q for q in qas}
        rows = super().cohorts(list(unique.values()), indices, window)
        mapping = dict(zip(unique, rows))
        return [mapping[q["source_index"]] for q in qas]


def full_targets(manager, qas):
    result = {}
    with torch.no_grad():
        for qa in qas:
            logits, gold = answer_logits(manager.model, qa, manager._memory.device)
            content_logits, content_gold = logits[:-1], gold[:-1]
            result[qa["id"]] = dict(gold_ce=float(F.cross_entropy(content_logits, content_gold)),
                eos_ce=float(F.cross_entropy(logits[-1:], gold[-1:])),
                log_probs=F.log_softmax(content_logits, -1).to(device="cpu", dtype=torch.float16),
                context_sha256=qa["context_sha256"], prompt_ids=list(qa["prompt_ids"]),
                answer_ids=list(qa["answer_ids"]))
    return result


def recurrent_targets(manager, merge, initial, episode, round_index, plan, training=None):
    from natural_merge_runtime import reset, MergeStream
    from prepare_natural_merge_v4 import TRAIN_V4
    training = TRAIN_V4 if training is None else training
    reset(manager,initial); stream=MergeStream(manager,merge,"Full"); result={}
    for i,write in enumerate(episode["writes"]):
        if i in training["recurrent_event_before_writes"]:
            result[i]=full_targets(manager,event_qas(plan,round_index,episode,"recurrent",i))
        with torch.no_grad():
            if stream.write(write,i) is None:raise ValueError("Full teacher capacity exceeded")
    if sorted(result)!=training["recurrent_event_before_writes"]:raise ValueError("incomplete teacher events")
    return result


def recovery_raw_weight(cohort, full_ce, selection_ce, plan=None):
    """Un-normalised recovery weight for one QA (C10).

    The legacy v4 rule (`1 + clamp(CE_Selection - CE_Full, 0, 2)` for the
    recovery cohort, 0 for everything else) is kept bit-for-bit when the plan
    carries no `recovery_gate`, because `TRAIN_V3`'s formula string is sealed by
    deep-equality assertions in the archived plans and must never move.

    A plan that does carry `recovery_gate` (v7) switches to the targeted rule:

        gap = CE_Selection - CE_Full
        raw = 1 + scale*min(gap, gap_cap)   if recover and CE_Full <= ce_full_max
                                            and gap > gap_min
        raw = background                    otherwise

    The `background` anchor is deliberately non-zero: discarded context still
    receives a small recovery pull instead of exactly none. `teacher_max_gold_ce`
    (2.0) is a *different* gate -- it decides who gets a KL teacher -- so the
    2.0 < CE_Full <= 2.5 band carries a recovery weight but no distillation.
    """
    from prepare_natural_merge_v4 import TRAIN_V4
    plan = TRAIN_V4 if plan is None else plan
    gate = plan.get("recovery_gate")
    if gate is None:
        return 1 + min(2., max(0., selection_ce - full_ce)) if cohort == "recover" else 0.
    gap = selection_ce - full_ce
    if cohort == "recover" and full_ce <= gate["ce_full_max"] and gap > gate["gap_min"]:
        return 1 + gate["scale"] * min(gap, gate["gap_cap"])
    return gate["background"]


def balanced_coefficients(qas, cohorts, eligible, full_ce, selection_ce, plan=None):
    from prepare_natural_merge_v4 import TRAIN_V4
    plan = TRAIN_V4 if plan is None else plan
    counts=Counter(q["source_index"] for q in qas)
    context_cohort={q["source_index"]:c[0] for q,c in zip(qas,cohorts)}
    nr=sum(c=="recover" for c in context_cohort.values())
    np=sum(c=="protect" for c in context_cohort.values())
    raw=[recovery_raw_weight(c[0], f, s, plan) for c,f,s in zip(cohorts,full_ce,selection_ce)]
    # The recover cohort is normalised to mean one per event. The legacy rule
    # gives the protect cohort raw 0, so summing over every QA and summing over
    # the recover cohort alone are the same number there; for v7 the 0.25
    # background anchor is absolute and must not be rescaled by that mean.
    average=sum(w/counts[q["source_index"]] for w,q,c in zip(raw,qas,cohorts)
                if c[0]=="recover")/max(nr,1)
    weights=[w/max(average,1e-12) if c[0]=="recover" else w
             for w,c in zip(raw,cohorts)]
    # `recovery` is the CE coefficient. Legacy plans keep it recover-only; a
    # plan with a background anchor applies it to the protect cohort too, which
    # is the documented v7 behaviour change (protect now carries 0.25*CE).
    recovery_scope = (lambda cohort: cohort == "recover") if plan.get("recovery_gate") is None \
        else (lambda cohort: True)
    teacher_counts=Counter(q["source_index"] for q,e in zip(qas,eligible) if e)
    nt=len(teacher_counts)
    return [dict(weight=w,
                 recovery=w/counts[q["source_index"]]/max(nr,1) if recovery_scope(c[0]) else 0.,
                 protect=1/counts[q["source_index"]]/max(np,1) if c[0]=="protect" else 0.,
                 teacher=1/teacher_counts[q["source_index"]]/nt if el else 0.,
                 eos=1/counts[q["source_index"]]/len(counts))
            for q,c,w,el in zip(qas,cohorts,weights,eligible)]


def make_optimizer(module, plan=None):
    from prepare_natural_merge_v4 import TRAIN_V4
    plan = TRAIN_V4 if plan is None else plan
    gates, matrices = [], []
    gate_names, matrix_names = [], []
    for name, parameter in module.named_parameters():
        if (name == "gamma_logit" or name.startswith("slot_gate.") or name.startswith("rel_gate.")
                or name == "native_gain"):
            gates.append(parameter); gate_names.append(name)
        else:
            matrices.append(parameter); matrix_names.append(name)
    if not gates or not matrices:
        raise ValueError("both optimizer groups required")
    return torch.optim.AdamW([
        dict(params=matrices, lr=plan["matrix_learning_rate"], name="matrices", parameter_names=matrix_names),
        dict(params=gates, lr=plan["gate_learning_rate"], name="gates", parameter_names=gate_names,
             weight_decay=0.)],
        weight_decay=plan["weight_decay"])


# Continuation-pollution first tokens: if the model treats the answer end as a
# prompt boundary it will start a new "Context:"/"Question(s):" turn instead of
# stopping. Penalizing their margin against the gold final token trains the EOS
# decision, not just the token identity.
BAD_CONTINUATION_IDS = (2014, 14924, 36349)  # " Context", " Question", " Questions"


def objective_terms(logits, gold, reference_ce, teacher, coefficients, anchor=None):
    ce = F.cross_entropy(logits[:-1], gold[:-1])
    eos_ce = F.cross_entropy(logits[-1:], gold[-1:])
    vocab = logits.shape[-1]
    bad_ids = [i for i in BAD_CONTINUATION_IDS if i < vocab]
    if bad_ids:
        hallucination_penalty = (torch.logsumexp(logits[-1, bad_ids], dim=0)
                                 - logits[-1, gold[-1]])
    else:
        hallucination_penalty = torch.zeros((), dtype=logits.dtype, device=logits.device)
    eos_margin = F.relu(hallucination_penalty + 2.0)
    kd = teacher_kl(logits[:-1], teacher) if teacher is not None else ce.new_zeros(())
    recovery = coefficients["recovery"]*ce
    distillation = TRAIN_V4["teacher_weight"]*coefficients["teacher"]*kd
    hinge = F.relu(ce-reference_ce-TRAIN_V4["protection_margin"])
    if anchor is not None:
        # The hinge is one-sided: for a protect-cohort sample that is still at or
        # below the Selection CE it contributes exactly zero, so nothing stops
        # the shared read from drifting while it is still "good enough". The
        # first-token KL is non-negative everywhere and pins the answer onset
        # distribution to the Selection reference at every event.
        hinge = hinge + TRAIN_V4["protection_anchor_weight"]*anchor
    protection = coefficients["protect"]*TRAIN_V4["protection_weight"]*hinge
    eos = TRAIN_V4["eos_weight"]*coefficients["eos"]*(eos_ce + 0.5*eos_margin)
    return ce, kd, recovery, distillation, protection, eos_ce, eos


def train_event(manager, merge, optimizer, qas, targets, variant, origins, scaler=None, plan=None):
    from methods.prlmc.probes import (_pool_in_place, _compacted_mask,
        active_mask_override, differentiable_memory)
    from methods.prlmc.consolidate import _select_block_indices
    from natural_merge_runtime import install
    from prepare_natural_merge_v4 import TRAIN_V4
    if plan is None:
        # Backwards-compatible dispatch: the v6 trainer calls this without a
        # plan and must keep receiving the sealed v6 variant record.
        if variant.endswith("V7"):
            from prepare_natural_merge_v7 import TRAIN_V7 as plan
        elif variant.endswith("V6"):
            from prepare_natural_merge_v6 import TRAIN_V6 as plan
        else:
            plan = TRAIN_V4
    allowed = plan["variants"]
    if variant not in allowed or not qas or len({q["id"] for q in qas}) != len(qas):
        raise ValueError("known variant and unique nonempty training QAs required")
    for qa in qas:
        teacher = targets.get(qa["id"], {})
        if any(teacher.get(k) != qa[k] for k in ("context_sha256", "prompt_ids", "answer_ids")):
            raise ValueError("Full answer-only teacher provenance mismatch")
    # Merge geometry from the plan (C8 item 4): `None` keeps the sealed v4 19->9
    # window bit-for-bit, v7 supplies 13->9 so the supervised operator sees the
    # same window width the evaluation stream compacts.
    geometry = plan.get("geometry") or dict(budget_blocks=20, window_blocks=19, target_blocks=9)
    m = manager
    window, target = geometry["window_blocks"]*m.block, geometry["target_blocks"]*m.block
    if len(origins.blocks)*m.block != m.used:
        raise ValueError("native-origin count differs from active memory")
    start = m.total-m.used
    source = m._memory[:, start:start+window].contiguous()
    ages = m.window_block_ages(start, window) if plan.get("geometry") else None
    indices = _select_block_indices(source, target, m.block,
        m._window_importance(start, window), None, m.select_weights, block_ages=ages)
    index_list = indices.cpu().tolist()
    window_blocks = window // m.block
    cohorts = origins.cohorts(qas, index_list, window_blocks)
    recover_ids = [i for i, (cohort, _) in enumerate(cohorts) if cohort == "recover"]
    protect_count = len(qas)-len(recover_ids)
    if not recover_ids or not protect_count:
        raise ValueError("both recovery and preservation cohorts required")
    kwargs = dict(target=target, block=m.block, carrier_indices=indices)
    carriers = source.reshape(m.num_layers, window_blocks, m.block, m.hidden).index_select(1, indices).flatten(1, 2)
    mask = _compacted_mask(m, m.used-window+target, m._memory.device)
    full_ce = [targets[q["id"]]["gold_ce"] for q in qas]
    selection_ce = []
    # First-token distribution of the Selection reference, kept on CPU in fp32:
    # the anchor below needs a target that does not move with the student.
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
    coefficients = balanced_coefficients(qas,cohorts,eligible,full_ce,selection_ce,plan)
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad(): value = merge(source, **kwargs)
    source_host = source.detach().cpu() if source.is_cuda else source.detach()
    del source, carriers
    leaf = value.detach().requires_grad_(True)
    rows = []; recovery_sum = kd_sum = protect_sum = eos_sum = total = 0.
    with _pool_in_place(m, m._memory.device, leaf, window, target) as injection:
        with active_mask_override(m.model, mask):
            for i, qa in enumerate(qas):
                pool = injection.reinject()
                with differentiable_memory(m.model, pool):
                    logits, gold = answer_logits(m.model, qa, m._memory.device)
                teacher = targets[qa["id"]]["log_probs"] if eligible[i] else None
                anchor = None
                if coefficients[i]["protect"] > 0:
                    anchor = F.kl_div(F.log_softmax(logits[0].float(), -1),
                                      selection_first[i].to(logits.device), reduction="sum")
                ce,kd,recovery,distillation,protection,eos_ce,eos = objective_terms(
                    logits,gold,selection_ce[i],teacher,coefficients[i],anchor)
                loss = recovery+distillation+protection+eos
                if not torch.isfinite(loss): raise ValueError("nonfinite C/D objective")
                (scaler.scale(loss) if scaler else loss).backward()
                recovery_sum += float(recovery.detach()); kd_sum += float(distillation.detach())
                protect_sum += float(protection.detach()); eos_sum += float(eos.detach()); total += float(loss.detach())
                rows.append(dict(id=qa["id"], source_index=qa["source_index"],
                    context_sha256=qa["context_sha256"], cohort=cohorts[i][0], origin_status=cohorts[i][1],
                    full_ce=full_ce[i], selection_ce=selection_ce[i], shared_ce=float(ce.detach()),
                    weight=coefficients[i]["weight"], eos_ce=float(eos_ce.detach()), eos_term=float(eos.detach()), teacher_eligible=eligible[i], teacher_kl=float(kd.detach()),
                    recovery_term=float(recovery.detach()), protection_term=float(protection.detach()),
                    protection_anchor=float(anchor.detach()) if anchor is not None else 0.,
                    answer_token_count=len(qa["content_ids"])))
                del pool, logits, ce, kd, recovery, distillation, protection, eos_ce, eos, loss
    if leaf.grad is None or not torch.isfinite(leaf.grad).all():
        raise ValueError("missing/nonfinite frozen-reader memory gradient")
    source = source_host.to(m._memory.device)
    rebuilt = merge(source, **kwargs); torch.autograd.backward(rebuilt, leaf.grad)
    del rebuilt, leaf, value
    if scaler: scaler.unscale_(optimizer)
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
        # v7 settles the ages of the supervised merge as well, so the training
        # selector sees the same generational state the evaluation selector
        # does; the sealed v4 path keeps passing None (no age change).
        install(m, current, window,
                selected_blocks=indices if plan.get("geometry") else None)
    origins.compact(index_list, window_blocks)
    means = {}
    for cohort in ("recover", "protect"):
        subset = [r for r in rows if r["cohort"] == cohort]
        means[cohort] = dict(count=len(subset), shared_ce=sum(r["shared_ce"] for r in subset)/len(subset),
            selection_ce=sum(r["selection_ce"] for r in subset)/len(subset),
            shared_minus_selection_ce=sum(r["shared_ce"]-r["selection_ce"] for r in subset)/len(subset))
    return dict(objective=total, gold_ce=sum(r["shared_ce"] for r in rows)/len(rows),
        weighted_gold_ce=recovery_sum, teacher_kl=kd_sum/plan["teacher_weight"],
        protection_hinge=protect_sum,
        protection_anchor=sum(r["protection_anchor"] for r in rows)/len(rows), eos_loss=eos_sum, teacher_eligible_count=eligible_count,
        teacher_weight=plan["teacher_weight"], original_qa_count=len(qas), qas=rows,
        cohorts=means, carrier_indices=index_list, retained_native_origins=list(origins.blocks),
        gradient_norms_before_clip=gradient_norms, grad_norm_before_clip=norm,
        fusion=fusion, realized_residual=stored,
        selection_reference="same pre-event student history; current window uses identical native carriers",
        qa_measurement="content-only CE/KL/teacher eligibility; separate EOS; context-balanced loss before update; stored budget after update")
