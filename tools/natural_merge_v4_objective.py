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


def recurrent_targets(manager, merge, initial, episode, round_index, plan):
    from natural_merge_runtime import reset, MergeStream
    reset(manager,initial); stream=MergeStream(manager,merge,"Full"); result={}
    for i,write in enumerate(episode["writes"]):
        if i in TRAIN_V4["recurrent_event_before_writes"]:
            result[i]=full_targets(manager,event_qas(plan,round_index,episode,"recurrent",i))
        with torch.no_grad():
            if stream.write(write,i) is None:raise ValueError("Full teacher capacity exceeded")
    if sorted(result)!=TRAIN_V4["recurrent_event_before_writes"]:raise ValueError("incomplete teacher events")
    return result


def balanced_coefficients(qas, cohorts, eligible, full_ce, selection_ce):
    counts=Counter(q["source_index"] for q in qas)
    context_cohort={q["source_index"]:c[0] for q,c in zip(qas,cohorts)}
    nr=sum(c=="recover" for c in context_cohort.values())
    np=sum(c=="protect" for c in context_cohort.values())
    raw=[1+min(2.,max(0.,s-f)) if c[0]=="recover" else 0.
         for c,f,s in zip(cohorts,full_ce,selection_ce)]
    average=sum(w/counts[q["source_index"]] for w,q in zip(raw,qas))/max(nr,1)
    weights=[w/max(average,1e-12) for w in raw]
    teacher_counts=Counter(q["source_index"] for q,e in zip(qas,eligible) if e)
    nt=len(teacher_counts)
    return [dict(weight=w,
                 recovery=w/counts[q["source_index"]]/max(nr,1) if c[0]=="recover" else 0.,
                 protect=1/counts[q["source_index"]]/max(np,1) if c[0]=="protect" else 0.,
                 teacher=1/teacher_counts[q["source_index"]]/nt if el else 0.,
                 eos=1/counts[q["source_index"]]/len(counts))
            for q,c,w,el in zip(qas,cohorts,weights,eligible)]


def make_optimizer(module):
    gates, matrices = [], []
    gate_names, matrix_names = [], []
    for name, parameter in module.named_parameters():
        if name == "gamma_logit" or name.startswith("slot_gate."):
            gates.append(parameter); gate_names.append(name)
        else:
            matrices.append(parameter); matrix_names.append(name)
    if not gates or not matrices:
        raise ValueError("both optimizer groups required")
    return torch.optim.AdamW([
        dict(params=matrices, lr=TRAIN_V4["matrix_learning_rate"], name="matrices", parameter_names=matrix_names),
        dict(params=gates, lr=TRAIN_V4["gate_learning_rate"], name="gates", parameter_names=gate_names)],
        weight_decay=TRAIN_V4["weight_decay"])


def objective_terms(logits, gold, reference_ce, teacher, coefficients):
    ce = F.cross_entropy(logits[:-1], gold[:-1])
    eos_ce = F.cross_entropy(logits[-1:], gold[-1:])
    kd = teacher_kl(logits[:-1], teacher) if teacher is not None else ce.new_zeros(())
    recovery = coefficients["recovery"]*ce
    distillation = TRAIN_V4["teacher_weight"]*coefficients["teacher"]*kd
    protection = coefficients["protect"]*TRAIN_V4["protection_weight"]*F.relu(ce-reference_ce-TRAIN_V4["protection_margin"])
    eos = TRAIN_V4["eos_weight"]*coefficients["eos"]*eos_ce
    return ce, kd, recovery, distillation, protection, eos_ce, eos


def train_event(manager, merge, optimizer, qas, targets, variant, origins, scaler=None):
    from methods.prlmc.probes import (_pool_in_place, _compacted_mask,
        active_mask_override, differentiable_memory)
    from methods.prlmc.consolidate import _select_block_indices
    from natural_merge_runtime import install
    if variant not in TRAIN_V4["variants"] or not qas or len({q["id"] for q in qas}) != len(qas):
        raise ValueError("known variant and unique nonempty training QAs required")
    for qa in qas:
        teacher = targets.get(qa["id"], {})
        if any(teacher.get(k) != qa[k] for k in ("context_sha256", "prompt_ids", "answer_ids")):
            raise ValueError("Full answer-only teacher provenance mismatch")
    m = manager; window, target = 19*m.block, 9*m.block
    if len(origins.blocks)*m.block != m.used:
        raise ValueError("native-origin count differs from active memory")
    start = m.total-m.used
    source = m._memory[:, start:start+window].contiguous()
    indices = _select_block_indices(source, target, m.block,
        m._window_importance(start, window), None, m.select_weights)
    index_list = indices.cpu().tolist()
    cohorts = origins.cohorts(qas, index_list)
    recover_ids = [i for i, (cohort, _) in enumerate(cohorts) if cohort == "recover"]
    protect_count = len(qas)-len(recover_ids)
    if not recover_ids or not protect_count:
        raise ValueError("both recovery and preservation cohorts required")
    kwargs = dict(target=target, block=m.block, carrier_indices=indices)
    carriers = source.reshape(m.num_layers, 19, m.block, m.hidden).index_select(1, indices).flatten(1, 2)
    mask = _compacted_mask(m, m.used-window+target, m._memory.device)
    full_ce = [targets[q["id"]]["gold_ce"] for q in qas]
    selection_ce = []
    with torch.no_grad(), _pool_in_place(m, m._memory.device, carriers, window, target) as injection:
        with active_mask_override(m.model, mask):
            for qa in qas:
                injection.reinject()
                logits, gold = answer_logits(m.model, qa, m._memory.device)
                selection_ce.append(float(F.cross_entropy(logits[:-1], gold[:-1])))
    if not torch.isfinite(torch.tensor(full_ce+selection_ce)).all():
        raise ValueError("nonfinite reference CE")
    eligible = [cohort == "recover" and full_ce[i] <= TRAIN_V4["teacher_max_gold_ce"]
                for i, (cohort, _) in enumerate(cohorts)]
    eligible_count = sum(eligible)
    coefficients = balanced_coefficients(qas,cohorts,eligible,full_ce,selection_ce)
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
                ce,kd,recovery,distillation,protection,eos_ce,eos = objective_terms(
                    logits,gold,selection_ce[i],teacher,coefficients[i])
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
    norm = float(torch.nn.utils.clip_grad_norm_(merge.parameters(), TRAIN_V4["gradient_clip"]))
    if not torch.isfinite(torch.tensor(norm)): raise ValueError("nonfinite module gradient norm")
    if scaler: scaler.step(optimizer); scaler.update()
    else: optimizer.step()
    with torch.no_grad():
        current = merge(source, **kwargs)
        carriers = source.reshape(m.num_layers, 19, m.block, m.hidden).index_select(1, indices).flatten(1, 2)
        stored = realized_residual(current, carriers); fusion = dict(merge.last_fusion)
        if stored["stored_residual_rms_ratio"] > TRAIN_V4["global_rms_budget"]+1e-6:
            raise ValueError("actual stored C/D budget exceeded")
        install(m, current, window)
    origins.compact(index_list)
    means = {}
    for cohort in ("recover", "protect"):
        subset = [r for r in rows if r["cohort"] == cohort]
        means[cohort] = dict(count=len(subset), shared_ce=sum(r["shared_ce"] for r in subset)/len(subset),
            selection_ce=sum(r["selection_ce"] for r in subset)/len(subset),
            shared_minus_selection_ce=sum(r["shared_ce"]-r["selection_ce"] for r in subset)/len(subset))
    return dict(objective=total, gold_ce=sum(r["shared_ce"] for r in rows)/len(rows),
        weighted_gold_ce=recovery_sum, teacher_kl=kd_sum/TRAIN_V4["teacher_weight"],
        protection_hinge=protect_sum, eos_loss=eos_sum, teacher_eligible_count=eligible_count,
        teacher_weight=TRAIN_V4["teacher_weight"], original_qa_count=len(qas), qas=rows,
        cohorts=means, carrier_indices=index_list, retained_native_origins=list(origins.blocks),
        gradient_norms_before_clip=gradient_norms, grad_norm_before_clip=norm,
        fusion=fusion, realized_residual=stored,
        selection_reference="same pre-event student history; current window uses identical native carriers",
        qa_measurement="content-only CE/KL/teacher eligibility; separate EOS; context-balanced loss before update; stored budget after update")
