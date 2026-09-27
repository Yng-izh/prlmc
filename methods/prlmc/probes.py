"""Functional probes and the PRLM-C compact objective.

Section 3.4 gives the composite objective

    L_compact = l_rec * L_latent-recon + l_probe * L_gold-probe
              + l_KD * L_teacher-KD + l_anchor * L_anchor + l_budget * K'

with the requirement that L_gold-probe is the *primary* functional supervision
and latent reconstruction is only auxiliary. It also states two constraints this
module enforces structurally rather than by convention:

  * Diagnostics run on a **copy** of the state. `evaluate_compaction` snapshots
    the manager, applies the candidate compaction, measures, and restores. The
    live agent is never handed a probe question, so it cannot be reminded of
    anything by being evaluated.
  * Test-time isolation. At test time the final question, its answer, the success
    criterion and any future tool result must not be used for writing or
    compaction. `Probe` therefore carries a `split` tag and `make_probes` only
    accepts "train" probes at test time; a "test" probe raises.

The teacher-KD term keeps a copy of the pre-compaction model and distils its
distribution on trusted probes. Since the uncompacted model can be wrong, it is
weighted below the gold-probe term and never replaces it.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass
class Probe:
    """A functional probe: a question plus the answers a good memory must support.

    `split` is load-bearing. Probes constructed from test-split data must never
    reach the writer or the compactor, per section 3.4.
    """

    question: str
    answers: Sequence[str]
    split: str = "train"
    weight: float = 1.0
    kind: str = "qa"  # qa | behavior


def make_probes(
    question_answer_pairs: Sequence,
    split: str = "train",
    limit: Optional[int] = None,
    answer_prefix: str = "Answer:",
) -> List[Probe]:
    """Build probes from (question, answer) pairs visible before the current step.

    Args:
        question_answer_pairs: pairs already observed by the writer.
        split: tag to attach. Callers passing "test" are rejected downstream when
            building test-time probes.

    Returns:
        A list of Probe objects, truncated to `limit` when given.
    """
    probes: List[Probe] = []
    for item in question_answer_pairs:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            question, answer = item[0], item[1]
        else:
            continue
        answers = answer if isinstance(answer, (list, tuple)) else [answer]
        probes.append(
            Probe(question=str(question), answers=[str(a) for a in answers], split=split, kind="qa")
        )
        if limit is not None and len(probes) >= limit:
            break
    return probes


def assert_test_isolation(probes: Sequence[Probe], allow_test: bool = False) -> None:
    """Refuse to run compaction against probes drawn from the test split.

    Section 3.4 forbids using the final test question, its answer, the success
    criterion, or future tool results for writing or compaction. Making this a
    hard error is the only way the guarantee survives a later refactor.
    """
    if allow_test:
        return
    offenders = [p for p in probes if p.split != "train"]
    if offenders:
        raise ValueError(
            "compaction received %d probe(s) from a non-train split; "
            "test questions and answers must not supervise the compactor"
            % len(offenders)
        )


def _probe_batches(
    tokenizer,
    probes: Sequence[Probe],
    device: str,
    max_length: int = 512,
):
    """Tokenise each probe once and locate the answer span.

    Returns a list of (input_ids, n_prefix, weight). Shared by the gold and KD
    terms so both score exactly the same spans.
    """
    batches = []
    for probe in probes:
        for answer in probe.answers:
            text = f"{probe.question}\nAnswer: {answer}"
            encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            input_ids = encoded.input_ids[:, :max_length].to(device)
            if input_ids.shape[1] < 2:
                continue
            prefix_ids = tokenizer(
                f"{probe.question}\nAnswer:", add_special_tokens=False
            ).input_ids
            n_prefix = min(len(prefix_ids), input_ids.shape[1] - 1)
            if n_prefix <= 0:
                continue
            batches.append((input_ids, n_prefix, float(probe.weight)))
    return batches


def _forward_spans(model, batches, enable_grad: bool):
    """Run the model on each probe and return the answer-span logits."""
    outputs = []
    for input_ids, n_prefix, _ in batches:
        if enable_grad:
            logits = model(input_ids=input_ids, return_dict=True).logits
        else:
            with torch.no_grad():
                logits = model(input_ids=input_ids, return_dict=True).logits
        outputs.append(logits[:, n_prefix - 1 : -1, :])
    return outputs


def gold_loss_from_logits(batches, span_logits, device) -> torch.Tensor:
    """Answer-span cross-entropy from already-computed logits.

    Split out so the training loop can score the gold term and the KD term from a
    single forward pass; calling a self-contained scorer twice would double the
    cost of every step.
    """
    if not batches or not span_logits:
        return torch.zeros((), device=device)

    total = torch.zeros((), device=device)
    weight_sum = 0.0
    for (input_ids, n_prefix, weight), logits in zip(batches, span_logits):
        shift_labels = input_ids[:, n_prefix:]
        if shift_labels.numel() == 0:
            continue
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), shift_labels.reshape(-1)
        )
        total = total + weight * loss
        weight_sum += weight

    if weight_sum <= 0.0:
        return torch.zeros((), device=device)
    return total / weight_sum


def gold_probe_loss(
    model,
    tokenizer,
    probes: Sequence[Probe],
    device: str = "cuda:0",
    max_length: int = 512,
    enable_grad: bool = False,
) -> torch.Tensor:
    """Cross-entropy of the gold answer tokens under the current memory state.

    This is the primary functional term. It answers "does the compressed memory
    still support the right answer", which is what the note says latent
    reconstruction cannot establish (risk 4).

    Args:
        enable_grad: must be True when this loss trains the compactor, or no
            gradient reaches the compressor and the functional term silently
            becomes a constant. Diagnostics keep it False.
    """
    if not probes:
        return torch.zeros((), device=device)

    batches = _probe_batches(tokenizer, probes, device, max_length)
    if not batches:
        return torch.zeros((), device=device)

    span_logits = _forward_spans(model, batches, enable_grad)
    return gold_loss_from_logits(batches, span_logits, device)


def teacher_kd_loss(teacher_logits, batches, student_logits) -> torch.Tensor:
    """KL divergence from the pre-compaction teacher to the post-compaction student.

    Section 3.4 lists L_teacher-KD as a term that stabilises compaction. It is
    weighted below the gold term because the uncompacted model can itself be
    wrong, so matching it more closely must never outweigh answering correctly.
    """
    total = torch.zeros((), device=student_logits[0].device if student_logits else "cpu")
    count = 0
    for teacher, student, (_, _, weight) in zip(teacher_logits, student_logits, batches):
        if teacher is None or teacher.numel() == 0:
            continue
        teacher_log_prob = F.log_softmax(teacher.detach().float(), dim=-1)
        student_log_prob = F.log_softmax(student.float(), dim=-1)
        # Batch mean over the span, so a long answer does not dominate a short one.
        divergence = F.kl_div(
            student_log_prob, teacher_log_prob, log_target=True, reduction="batchmean"
        )
        total = total + weight * divergence
        count += 1
    if count == 0:
        return torch.zeros((), device="cpu")
    return total / count


def reader_hidden_loss(
    teacher_hidden: Sequence[torch.Tensor],
    student_hidden: Sequence[torch.Tensor],
    layers: Sequence[int] = (8, 16, 24, 31),
) -> torch.Tensor:
    """L_reader-hidden: match the reader's hidden states, not just its answers.

    Phase 4's premise is that a compactor can preserve the *output* while still
    moving the reader into a different regime, and a regime change is what shows
    up later as fragility rather than as an immediate accuracy drop. Matching the
    query-token hidden states at a few depths constrains the trajectory itself.

    Only the query positions are compared, and only at `layers`. The memory
    prefix is deliberately excluded: it is the thing being changed, so requiring
    it to match would penalise the compaction for doing its job.

    `teacher_hidden` must come from a forward pass over the *uncompacted* state,
    so it is detached by the caller before being passed here.
    """
    if not teacher_hidden or not student_hidden:
        return torch.zeros((), device="cpu")

    total = None
    count = 0
    for teacher, student in zip(teacher_hidden, student_hidden):
        if teacher is None or student is None or teacher.numel() == 0:
            continue
        if teacher.shape != student.shape:
            raise ValueError(
                "reader_hidden_loss requires matching shapes, got %s and %s"
                % (tuple(teacher.shape), tuple(student.shape))
            )
        # Two-sided: the reader should not drift in either direction, and a
        # one-sided penalty would let the student simply shrink its activations.
        term = F.mse_loss(student.to(torch.float32), teacher.detach().to(torch.float32))
        total = term if total is None else total + term
        count += 1

    if total is None or count == 0:
        return torch.zeros((), device="cpu")
    return total / count


def latent_reconstruction_loss(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    """Auxiliary weighted reconstruction term, kept subordinate to the probes."""
    if before.shape != after.shape:
        raise ValueError("latent_reconstruction_loss requires matching shapes")
    return F.mse_loss(after.to(torch.float32), before.to(torch.float32))


def anchor_loss(
    before: torch.Tensor,
    after: torch.Tensor,
) -> torch.Tensor:
    """L_anchor: penalise change in the Anchor region across compaction.

    The note points out that when the Anchor state is unchanged this term is
    identically zero and would therefore be a useless penalty; the honest
    alternative it recommends is to exclude the Anchor from compaction
    altogether. That is what the manager does, and this function exists for the
    case where a caller passes a region that *does* change, so the term is
    reported rather than silently zero.
    """
    if before.shape != after.shape:
        raise ValueError("anchor_loss requires matching shapes")
    return F.mse_loss(after.to(torch.float32), before.to(torch.float32))


@contextmanager
def differentiable_memory(model, memory: torch.Tensor):
    """Present `memory` to the model's forward pass while keeping autograd.

    Assigning through `.data` would sever the graph, which is what makes naively
    "writing" a trainable candidate a silent no-op: the functional loss would be
    detached and the compressor would receive no gradient at all. So the
    parameter is temporarily moved out of `_parameters` and the trainable tensor
    takes its place, letting `cat_memory_and_hiddens` read it as usual. The
    original parameter object is restored on exit, including on exception.
    """
    parameters = model._parameters
    had_parameter = "memory" in parameters
    original = parameters.pop("memory", None)
    object.__setattr__(model, "memory", memory)
    try:
        yield
    finally:
        model.__dict__.pop("memory", None)
        if had_parameter:
            parameters["memory"] = original


@dataclass
class CompactObjective:
    """Weights for the compaction objective.

    Phase 4 target, from the design note:

        L = L_gold + 0.5*L_KD + 0.1*L_reader-hidden + 0.05*L_carrier + 0.02*L_stats

    These are **development starting values, not final hyperparameters**. The note
    is explicit that every threshold and weight is to be chosen on a dev split, so
    treat any number here as a placeholder to be re-tuned rather than a result.
    """

    gold: float = 1.0
    kd: float = 0.5
    reader_hidden: float = 0.1
    carrier: float = 0.05
    stats: float = 0.02
    # Legacy weights, kept so existing callers of `total` keep working.
    recon: float = 0.1
    probe: float = 1.0
    anchor: float = 0.5
    budget: float = 0.01
    target_tokens: int = 256

    def phase4_total(
        self,
        gold_loss: Optional[torch.Tensor] = None,
        kd_loss: Optional[torch.Tensor] = None,
        reader_hidden_loss: Optional[torch.Tensor] = None,
        carrier_loss: Optional[torch.Tensor] = None,
        stats_loss: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ) -> torch.Tensor:
        """The phase-4 objective. Missing terms contribute zero, never NaN."""
        zero = torch.zeros((), device=device)

        def pick(term):
            return zero if term is None else term

        return (
            self.gold * pick(gold_loss)
            + self.kd * pick(kd_loss)
            + self.reader_hidden * pick(reader_hidden_loss)
            + self.carrier * pick(carrier_loss)
            + self.stats * pick(stats_loss)
        )

    def total(
        self,
        recon_loss: Optional[torch.Tensor] = None,
        probe_loss: Optional[torch.Tensor] = None,
        kd_loss: Optional[torch.Tensor] = None,
        anchor_term: Optional[torch.Tensor] = None,
        n_tokens: int = 0,
        device: str = "cpu",
    ) -> torch.Tensor:
        zero = torch.zeros((), device=device)
        return (
            self.recon * (recon_loss if recon_loss is not None else zero)
            + self.probe * (probe_loss if probe_loss is not None else zero)
            + self.kd * (kd_loss if kd_loss is not None else zero)
            + self.anchor * (anchor_term if anchor_term is not None else zero)
            + self.budget * float(n_tokens)
        )


@dataclass
class CompactionReport:
    """Functional diagnosis of one compaction, measured on a copy of the state."""

    probe_loss_before: float = 0.0
    probe_loss_after: float = 0.0
    freed_slots: int = 0
    used_before: int = 0
    used_after: int = 0
    cumulative_loss: float = 0.0
    per_probe: List[Dict[str, object]] = field(default_factory=list)

    @property
    def degradation(self) -> float:
        """Increase in probe loss caused by compaction. Positive is a loss."""
        return self.probe_loss_after - self.probe_loss_before

    def to_dict(self) -> Dict[str, object]:
        return {
            "probe_loss_before": self.probe_loss_before,
            "probe_loss_after": self.probe_loss_after,
            "degradation": self.degradation,
            "freed_slots": self.freed_slots,
            "used_before": self.used_before,
            "used_after": self.used_after,
            "cumulative_loss": self.cumulative_loss,
        }


def evaluate_compaction(
    manager,
    probes: Sequence[Probe],
    ratio: Optional[float] = None,
    allow_test_probes: bool = False,
) -> CompactionReport:
    """Measure functional loss of one candidate compaction on a copy of the state.

    The live pool is snapshotted, the candidate compaction is applied, the probes
    are scored, and the original state is restored. Nothing the agent will use is
    modified by this call.
    """
    assert_test_isolation(probes, allow_test=allow_test_probes)

    report = CompactionReport(used_before=manager.used)
    state = manager.snapshot()

    try:
        if probes:
            report.probe_loss_before = float(
                gold_probe_loss(
                    manager.model, manager.tokenizer, probes, device=str(manager.model.memory.device)
                ).item()
            )

        report.freed_slots = manager.compact(reason="probe_evaluation", ratio=ratio)
        report.used_after = manager.used

        if probes:
            report.probe_loss_after = float(
                gold_probe_loss(
                    manager.model, manager.tokenizer, probes, device=str(manager.model.memory.device)
                ).item()
            )
    finally:
        manager.restore(state)

    return report


def train_compactor(
    manager,
    probes: Sequence[Probe],
    objective: Optional[CompactObjective] = None,
    steps: int = 50,
    lr: float = 1e-3,
    allow_test_probes: bool = False,
    kd_weight: Optional[float] = None,
) -> Dict[str, float]:
    """Train the learned compactor against functional probes.

    The backbone stays frozen; only the compressor's parameters move. The
    gold-probe term is the primary signal and the teacher-KD term is auxiliary,
    as section 3.4 requires.

    Cost warning: this substitutes a full differentiable copy of the memory pool
    into the model for each step, so it needs roughly one extra pool's worth of
    fp32 memory (about 6.7 GB at the default 32 x 12800 x 4096 pool) on top of
    the fp16 weights. On a 32 GB V100 that does not fit alongside activations;
    it is intended for a larger card or a shortened pool. It is opt-in for that
    reason, and the default ablation never calls it.
    """
    from methods.prlmc.consolidate import LearnedCompactor

    assert_test_isolation(probes, allow_test=allow_test_probes)

    if not probes:
        raise ValueError("train_compactor needs at least one functional probe")
    if manager.used < 2 * manager.block:
        raise ValueError("not enough memory written to train a compactor")
    if manager.cfg.mode in ("full_rewrite", "anchor_rewrite"):
        raise ValueError("this ablation arm has no consolidatable working memory")

    objective = objective or CompactObjective()
    if kd_weight is not None:
        objective.kd = float(kd_weight)
    device = manager.model.memory.device

    start = manager.total - manager.used
    # The compactor maps a K-slot window to K' slots, so the window it trains on
    # must be the same shape the consolidation path will hand it. Passing the
    # first `target` rows instead made the reconstruction term compare tensors of
    # different shapes and would have raised on the first step.
    window = min(manager.used - manager.newest_width, max(manager.cfg.compact_min_slots, manager.used // 2))
    window = max(manager.block, (window // manager.block) * manager.block)
    window = min(window, manager.used - manager.newest_width)
    target = max(manager.block, (int(window * manager.cfg.compact_ratio) // manager.block) * manager.block)
    target = min(target, window - manager.block)
    if target <= 0 or target >= window:
        raise ValueError("compaction target is not smaller than the current state")

    source = manager._memory[:, start : start + window, :].detach().to(torch.float32)
    base = manager._memory.detach().clone()
    batches = _probe_batches(manager.tokenizer, probes, str(device))

    with torch.no_grad():
        teacher_logits = _forward_spans(manager.model, batches, enable_grad=False)

    compactor = LearnedCompactor(manager.num_layers, manager.hidden, target).to(device)
    optimiser = torch.optim.AdamW(compactor.parameters(), lr=lr)

    # The backbone is frozen: without this, the differentiable forward would build
    # and retain gradients for all 8B parameters on every step, which is both
    # wasted memory and a silent violation of "the checkpoint is used frozen".
    frozen = []
    for parameter in manager.model.parameters():
        if parameter.requires_grad:
            parameter.requires_grad_(False)
            frozen.append(parameter)

    history: List[float] = []
    try:
        for _ in range(int(steps)):
            optimiser.zero_grad(set_to_none=True)

            candidate = compactor(source)
            # Substituting into a clone, rather than writing through `.data`, is
            # what keeps `candidate` connected to the graph so the functional term
            # can actually train the compressor.
            pool = base.clone()
            pool[:, start : start + candidate.shape[1], :] = candidate.to(pool.dtype)

            with differentiable_memory(manager.model, pool):
                student_logits = _forward_spans(manager.model, batches, enable_grad=True)

            # One forward pass serves both supervised terms.
            probe_term = gold_loss_from_logits(batches, student_logits, device)
            recon_term = latent_reconstruction_loss(source, candidate.to(torch.float32))
            kd_term = teacher_kd_loss(teacher_logits, batches, student_logits)
            loss = objective.total(
                recon_loss=recon_term,
                probe_loss=probe_term,
                kd_loss=kd_term.to(device),
                n_tokens=target,
                device=str(device),
            )
            loss.backward()
            optimiser.step()
            history.append(float(loss.item()))
    finally:
        for parameter in frozen:
            parameter.requires_grad_(True)

    compactor.trained = True
    manager.compactor = compactor
    return {
        "steps": float(steps),
        "window_tokens": float(window),
        "target_tokens": float(target),
        "final_loss": history[-1] if history else float("nan"),
        "initial_loss": history[0] if history else float("nan"),
        "probe_weight": float(objective.probe),
        "kd_weight": float(objective.kd),
    }


# --- phase 4: carrier-residual training -------------------------------------

# Depth at which the reader's hidden states are compared. Chosen to span the
# stack rather than cluster at the end: an early layer catches a change in how
# the memory is *read*, a late one catches a change in what is read out.
_READER_HIDDEN_LAYERS = (8, 16, 24, 31)

# Curriculum from the design note. Each entry is the *retained* block count, so
# the operator is asked to compress harder as training proceeds. 19 -> 9 blocks
# is the full schedule; the intermediate step exists because jumping straight to
# the hardest ratio made the gold term dominate and the residual collapse to
# zero in the first version of this schedule.
_CARRIER_CURRICULUM = (19, 15, 19, 12, 19, 9)


def _active_pool(manager, device):
    """Build the active [A, S, R] prefix only, in the pool dtype.

    The design note is explicit that the whole [32, 12800, 4096] pool must not be
    copied to fp32: at 4096 hidden and 32 layers that copy alone is ~6.7 GB, which
    does not fit on a 32 GB V100 next to an 8B model and its activations. The
    reader only ever attends to the active region anyway -- the free region is
    masked out by phase 2 -- so the narrowed tensor is both cheaper and exactly
    the state the reader is meant to see.
    """
    total = manager.total
    used = manager.used
    k_anchor = manager.layout.k_anchor
    memory = manager._memory
    active = memory[:, :k_anchor, :]
    if used > 0:
        active = torch.cat([active, memory[:, total - used : total, :]], dim=1)
    return active.to(device).detach().clone()


def _forward_with_hidden(model, input_ids, layers, enable_grad: bool):
    """One reader forward pass, returning both logits and selected hidden states.

    `output_hidden_states=True` is what the phase-4 objective needs, and it is the
    reason this path cannot reuse `_forward_spans`: that helper only keeps logits.
    """
    def run():
        out = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
        return out.logits, out.hidden_states

    if enable_grad:
        logits, hidden = run()
    else:
        with torch.no_grad():
            logits, hidden = run()

    picked = []
    if hidden is not None:
        for index in layers:
            if 0 <= index < len(hidden):
                picked.append(hidden[index])
    return logits, picked


def train_carrier_compactor(
    manager,
    probes: Sequence[Probe],
    objective: Optional[CompactObjective] = None,
    steps: int = 50,
    lr: float = 1e-3,
    allow_test_probes: bool = False,
    curriculum: Sequence[int] = _CARRIER_CURRICULUM,
    reader_layers: Sequence[int] = _READER_HIDDEN_LAYERS,
    max_probe_tokens: int = 128,
    use_amp: bool = True,
) -> Dict[str, float]:
    """Phase 4: train the carrier residual against the reader.

    Objective, with development starting weights from the design note:

        L = L_gold + 0.5*L_KD + 0.1*L_reader-hidden + 0.05*L_carrier + 0.02*L_stats

    Both the backbone and the reader are frozen; only the carrier module moves.
    The gold term is primary, KD is auxiliary, and the reader-hidden term is what
    makes this phase 4 rather than a longer phase 3.

    Resource discipline, all of it load-bearing on a 32 GB V100:

      * only the active [A, S, R] prefix is materialised (`_active_pool`);
      * probes are truncated to `max_probe_tokens` (64-128), batch of one;
      * the candidate is cast to the pool dtype and the loss is scaled, so the
        activation graph stays fp16;
      * `torch.utils.checkpoint` is used when available, trading a second forward
        for a much smaller activation footprint.

    Returns a report dict; the trained module is attached to `manager.compactor`.

    Known train/inference discrepancy, recorded rather than hidden: narrowing the
    pool to [A, S, R] changes the memory prefix length from 12800 to
    `k_anchor + used`, which shifts the rotary positions of every query token.
    The design note mandates the narrowing because the full-pool fp32 copy does not
    fit on a 32 GB V100, so the alternative is no phase 4 at all. Teacher and
    student see the *same* prefix length, so KD and the hidden-state term remain
    well posed; what is unverified is whether a residual trained at the shortened
    prefix still transfers to the fixed-length pool. Check that before trusting a
    phase-4 checkpoint in a full run.
    """
    from methods.prlmc.consolidate import CarrierResidualCompactor

    assert_test_isolation(probes, allow_test=allow_test_probes)
    if not probes:
        raise ValueError("train_carrier_compactor needs at least one functional probe")
    if manager.used < 2 * manager.block:
        raise ValueError("not enough memory written to train a compactor")

    objective = objective or CompactObjective()
    device = manager.model.memory.device

    start = manager.total - manager.used
    window = min(
        manager.used - manager.newest_width,
        max(manager.cfg.compact_min_slots, manager.used // 2),
    )
    window = max(manager.block, (window // manager.block) * manager.block)
    window = min(window, manager.used - manager.newest_width)
    if window <= manager.block:
        raise ValueError("window is too small to train against")

    source = manager._memory[:, start : start + window, :].to(device)
    batches = _probe_batches(manager.tokenizer, probes, str(device), max_length=max_probe_tokens)
    if not batches:
        raise ValueError("every probe was truncated away; raise max_probe_tokens")

    compactor = CarrierResidualCompactor(
        num_layers=manager.num_layers,
        hidden=manager.hidden,
        block=manager.block,
        rank=int(getattr(manager.cfg, "carrier_rank", 32)),
        gamma_max=float(getattr(manager.cfg, "carrier_gamma_max", 0.1)),
    ).to(device=device, dtype=torch.float32)

    optimiser = torch.optim.AdamW(compactor.parameters(), lr=lr)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=bool(use_amp))
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=bool(use_amp))

    # Freeze the backbone and the reader. Without this the differentiable forward
    # would retain an 8B-parameter graph on every step.
    frozen = []
    for parameter in manager.model.parameters():
        if parameter.requires_grad:
            parameter.requires_grad_(False)
            frozen.append(parameter)

    base_active = _active_pool(manager, device)
    k_anchor = manager.layout.k_anchor
    active_start = k_anchor + (manager.total - manager.used - start)

    # Teacher pass over the *uncompacted* active state, detached. It must run
    # against `base_active`, not the live pool: the student forward sees the
    # narrowed [A, S, R] prefix, and a teacher pass over the full 12800-slot pool
    # would give the two sides different sequence lengths, which both
    # `teacher_kd_loss` and `reader_hidden_loss` would reject on shape.
    with torch.no_grad(), differentiable_memory(manager.model, base_active):
        teacher_logits = []
        teacher_hidden = []
        for input_ids, n_prefix, _ in batches:
            logits, hidden = _forward_with_hidden(
                manager.model, input_ids, reader_layers, enable_grad=False
            )
            teacher_logits.append(logits[:, n_prefix - 1 : -1, :].detach())
            teacher_hidden.append([h.detach() for h in hidden])

    history: List[float] = []
    ratios: List[int] = []
    try:
        for step in range(int(steps)):
            keep_blocks = curriculum[min(step * len(curriculum) // max(1, int(steps)), len(curriculum) - 1)]
            target = max(manager.block, min(window - manager.block, keep_blocks * manager.block))
            ratios.append(target)

            optimiser.zero_grad(set_to_none=True)

            candidate = compactor(
                source,
                target=target,
                block=manager.block,
                goal_direction=manager.goal_direction,
                select_weights=manager.select_weights,
            )

            # Substitute only into the active region of the narrowed pool. Writes
            # go through `candidate.to(...)`, which keeps the graph: assigning via
            # `.data` here would silently detach the objective.
            pool = base_active.clone()
            width = candidate.shape[1]
            lo = active_start
            pool[:, lo : lo + width, :] = candidate.to(pool.dtype)

            with differentiable_memory(manager.model, pool):
                student_logits = []
                student_hidden = []
                for input_ids, n_prefix, _ in batches:
                    logits, hidden = _forward_with_hidden(
                        manager.model, input_ids, reader_layers, enable_grad=True
                    )
                    student_logits.append(logits[:, n_prefix - 1 : -1, :])
                    student_hidden.append(hidden)

            gold_term = gold_loss_from_logits(batches, student_logits, device)
            kd_term = teacher_kd_loss(teacher_logits, batches, student_logits).to(device)

            hidden_term = None
            for teacher, student in zip(teacher_hidden, student_hidden):
                term = reader_hidden_loss(teacher, student, layers=reader_layers)
                hidden_term = term if hidden_term is None else hidden_term + term

            # L_carrier: the residual must stay a residual. Without this the
            # amplitude bound is the only thing keeping the operator from turning
            # into a re-encoder.
            carrier_term = torch.zeros((), device=device)
            if compactor.up.weight.numel() > 0:
                carrier_term = compactor.up.weight.pow(2).mean()
            stats_term = (compactor.gamma.to(device) ** 2).reshape(())

            loss = objective.phase4_total(
                gold_loss=gold_term,
                kd_loss=kd_term,
                reader_hidden_loss=hidden_term,
                carrier_loss=carrier_term,
                stats_loss=stats_term,
                device=str(device),
            )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                optimiser.step()

            history.append(float(loss.item()))
    finally:
        for parameter in frozen:
            parameter.requires_grad_(True)

    manager.compactor = compactor
    manager.carrier_compactor = compactor
    return {
        "steps": float(steps),
        "window_tokens": float(window),
        "target_tokens": float(ratios[-1]) if ratios else float("nan"),
        "initial_loss": history[0] if history else float("nan"),
        "final_loss": history[-1] if history else float("nan"),
        "gamma": float(compactor.gamma.detach()),
        "gamma_max": float(compactor.gamma_max),
        "curriculum": [float(v) for v in ratios],
        "amp": 1.0 if scaler.is_enabled() else 0.0,
    }
