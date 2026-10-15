"""Common C/D objective and recurrent native-origin bookkeeping."""
import torch
import torch.nn.functional as F

from prepare_natural_merge_v3 import TRAIN_V3
from natural_merge_v2_train import answer_logits, full_targets, recovery_weights, teacher_kl, realized_residual


class NativeOrigins:
    """One immutable original context index per native carrier block.

    Fusion does not turn a discarded context into an intact native carrier.
    At recurrent events, original contexts absent from the current native
    survivors remain recovery QAs. The untouched newest block is protected too.
    """
    def __init__(self):
        self.blocks = []

    def append(self, source_index):
        if source_index in self.blocks:
            raise ValueError("duplicate native write origin")
        self.blocks.append(source_index)

    def cohorts(self, qas, indices, window=19):
        if len(self.blocks) < window:
            raise ValueError("incomplete provenance window")
        retained = {self.blocks[i] for i in indices} | set(self.blocks[window:])
        current = set(self.blocks[:window])
        if len({q["source_index"] for q in qas}) != len(qas):
            raise ValueError("one QA per original context required")
        if not set(self.blocks).issubset({q["source_index"] for q in qas}):
            raise ValueError("native origins missing from QA prefix")
        return [("protect" if q["source_index"] in retained else "recover",
                 "current_window" if q["source_index"] in current else
                 "untouched_tail" if q["source_index"] in retained else "previously_discarded") for q in qas]

    def compact(self, indices, window=19):
        self.blocks = [self.blocks[i] for i in indices] + self.blocks[window:]


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
        dict(params=matrices, lr=TRAIN_V3["matrix_learning_rate"], name="matrices", parameter_names=matrix_names),
        dict(params=gates, lr=TRAIN_V3["gate_learning_rate"], name="gates", parameter_names=gate_names)],
        weight_decay=TRAIN_V3["weight_decay"])


def objective_terms(logits, gold, reference_ce, cohort, weight, teacher, recover_count,
                    protect_count, teacher_count):
    ce = F.cross_entropy(logits, gold)
    zero = logits.new_zeros(())
    kd = teacher_kl(logits, teacher) if cohort == "recover" and teacher is not None else zero
    recovery = weight * ce / max(recover_count, 1) if cohort == "recover" else zero
    distillation = TRAIN_V3["teacher_weight"] * kd / max(teacher_count, 1)
    protection = (TRAIN_V3["protection_weight"] * F.relu(ce - reference_ce - TRAIN_V3["protection_margin"])
                  / max(protect_count, 1)) if cohort == "protect" else zero
    return ce, kd, recovery, distillation, protection


def train_event(manager, merge, optimizer, qas, targets, variant, origins, scaler=None):
    from methods.prlmc.probes import (_pool_in_place, _compacted_mask,
        active_mask_override, differentiable_memory)
    from methods.prlmc.consolidate import _select_block_indices
    from natural_merge_runtime import install
    if variant not in TRAIN_V3["variants"] or not qas or len({q["id"] for q in qas}) != len(qas):
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
                selection_ce.append(float(F.cross_entropy(logits, gold)))
    if not torch.isfinite(torch.tensor(full_ce+selection_ce)).all():
        raise ValueError("nonfinite reference CE")
    weights = [0.]*len(qas)
    for i, weight in zip(recover_ids, recovery_weights([full_ce[i] for i in recover_ids],
                                                      [selection_ce[i] for i in recover_ids])):
        weights[i] = weight
    eligible = [cohort == "recover" and full_ce[i] <= TRAIN_V3["teacher_max_gold_ce"]
                for i, (cohort, _) in enumerate(cohorts)]
    eligible_count = sum(eligible)
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad(): value = merge(source, **kwargs)
    source_host = source.detach().cpu() if source.is_cuda else source.detach()
    del source, carriers
    leaf = value.detach().requires_grad_(True)
    rows = []; recovery_sum = kd_sum = protect_sum = total = 0.
    with _pool_in_place(m, m._memory.device, leaf, window, target) as injection:
        with active_mask_override(m.model, mask):
            for i, qa in enumerate(qas):
                pool = injection.reinject()
                with differentiable_memory(m.model, pool):
                    logits, gold = answer_logits(m.model, qa, m._memory.device)
                teacher = targets[qa["id"]]["log_probs"] if eligible[i] else None
                ce, kd, recovery, distillation, protection = objective_terms(logits, gold,
                    selection_ce[i], cohorts[i][0], weights[i], teacher,
                    len(recover_ids), protect_count, eligible_count)
                loss = recovery+distillation+protection
                if not torch.isfinite(loss): raise ValueError("nonfinite C/D objective")
                (scaler.scale(loss) if scaler else loss).backward()
                recovery_sum += float(recovery.detach()); kd_sum += float(distillation.detach())
                protect_sum += float(protection.detach()); total += float(loss.detach())
                rows.append(dict(id=qa["id"], source_index=qa["source_index"],
                    context_sha256=qa["context_sha256"], cohort=cohorts[i][0], origin_status=cohorts[i][1],
                    full_ce=full_ce[i], selection_ce=selection_ce[i], shared_ce=float(ce.detach()),
                    weight=weights[i], teacher_eligible=eligible[i], teacher_kl=float(kd.detach()),
                    recovery_term=float(recovery.detach()), protection_term=float(protection.detach()),
                    answer_token_count=len(qa["answer_ids"])))
                del pool, logits, ce, kd, recovery, distillation, protection, loss
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
    norm = float(torch.nn.utils.clip_grad_norm_(merge.parameters(), TRAIN_V3["gradient_clip"]))
    if not torch.isfinite(torch.tensor(norm)): raise ValueError("nonfinite module gradient norm")
    if scaler: scaler.step(optimizer); scaler.update()
    else: optimizer.step()
    with torch.no_grad():
        current = merge(source, **kwargs)
        carriers = source.reshape(m.num_layers, 19, m.block, m.hidden).index_select(1, indices).flatten(1, 2)
        stored = realized_residual(current, carriers); fusion = dict(merge.last_fusion)
        if stored["stored_residual_rms_ratio"] > TRAIN_V3["global_rms_budget"]+1e-6:
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
        weighted_gold_ce=recovery_sum, teacher_kl=kd_sum/TRAIN_V3["teacher_weight"],
        protection_hinge=protect_sum, teacher_eligible_count=eligible_count,
        teacher_weight=TRAIN_V3["teacher_weight"], original_qa_count=len(qas), qas=rows,
        cohorts=means, carrier_indices=index_list, retained_native_origins=list(origins.blocks),
        gradient_norms_before_clip=gradient_norms, grad_norm_before_clip=norm,
        fusion=fusion, realized_residual=stored,
        selection_reference="same pre-event student history; current window uses identical native carriers",
        qa_measurement="answer-only loss before update; stored budget measured after update")
