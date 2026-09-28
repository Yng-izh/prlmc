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

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


def _mem(label, device):
    """Phase-boundary memory trace, enabled by PRLMC_MEM_TRACE=1.

    Kept because the E6a training step failed with ~31 GiB already live *before*
    the first student forward, which no amount of checkpointing can fix: the leak
    is upstream of the optimizer step and has to be located by measurement.
    """
    if not os.environ.get("PRLMC_MEM_TRACE"):
        return
    if not torch.cuda.is_available():
        return
    print(
        "[mem] %-26s alloc=%.2fGB reserved=%.2fGB"
        % (
            label,
            torch.cuda.memory_allocated(device) / 1e9,
            torch.cuda.memory_reserved(device) / 1e9,
        ),
        flush=True,
    )


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
    prompt: Optional[str] = None
    """The exact text the reader sees before the answer, if it is not the default.

    A probe scored through a different prompt than inference is not a probe of the
    memory. FactConsolidation is the sharp case: every question carries a long
    "you are a knowledge management system" instruction, and the model answers in
    that format but not in a bare one, so a bare `Question\\nAnswer:` prompt scores
    0/16 even with the full uncompacted pool. That reads as a broken memory and is
    really a broken probe, which would make a before/after test vacuous (0 on both
    sides). Callers pass the template-rendered prefix here.
    """


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
        # `probe.prompt` already contains the question, so it is used verbatim;
        # otherwise fall back to the bare form for callers that have no template.
        stem = probe.prompt if probe.prompt else f"{probe.question}\nAnswer:"
        for answer in probe.answers:
            text = f"{stem} {answer}"
            encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            input_ids = encoded.input_ids[:, :max_length].to(device)
            if input_ids.shape[1] < 2:
                continue
            prefix_ids = tokenizer(stem, add_special_tokens=False).input_ids
            if len(prefix_ids) >= max_length:
                # Clamping here does not shorten the prompt, it moves the answer
                # span into the middle of it. The rendered `rag_agent` prompt is
                # ~163 tokens, so any `max_length` below that scored an instruction
                # word rather than the answer and every accuracy/gold number was
                # independent of the memory pool. See EXPERIMENTS.md, E6a.
                raise ValueError(
                    "_probe_batches: max_length=%d truncates the %d-token prompt; "
                    "the answer span would be mid-instruction"
                    % (max_length, len(prefix_ids))
                )
            n_prefix = len(prefix_ids)
            if n_prefix <= 0 or n_prefix >= input_ids.shape[1]:
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


def _checkpoint_func():
    """Checkpoint function for the fork's per-layer branch.

    `transformers` normally installs `self._gradient_checkpointing_func` on the
    module it is called on. `gradient_checkpointing_enable()` is invoked on
    `MemoryLLM`, but the fork's decoder loop reads it from the inner `LlamaModel`
    (`self.model._gradient_checkpointing_func`). `use_reentrant=False` is required
    because the loop passes the layer's arguments positionally and the reentrant
    implementation would warn/fail on the non-tensor `prefix_token_length` and
    `memory_attention_bias` values.
    """
    import torch.utils.checkpoint as checkpoint

    def func(function, *args, **kwargs):
        return checkpoint.checkpoint(
            function, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
        )

    return func


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


@contextmanager
def active_mask_override(model, mask):
    """Temporarily publish `mask` as the reader's active-slot mask.

    `_compacted_pool` rearranges the memory tensor, but the model also holds a
    `memory_active_mask` buffer that decides which pool slots the reader may
    attend to. If that buffer is left describing the *pre-compaction* layout, the
    freed slots stay active and still contain the blocks the compaction just
    dropped, so the student would read the carriers and the originals side by
    side and the probe would score the uncompacted memory. Syncing the mask is
    what makes the measurement a measurement of the compaction.
    """
    if mask is None:
        yield
        return
    buffers = model._buffers
    had = "memory_active_mask" in buffers
    original = buffers.get("memory_active_mask")
    buffers["memory_active_mask"] = mask
    try:
        yield
    finally:
        if had:
            buffers["memory_active_mask"] = original
        else:
            buffers.pop("memory_active_mask", None)


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


def _forward_with_hidden(
    model,
    input_ids,
    layers,
    enable_grad: bool,
    pool: Optional[torch.Tensor] = None,
    use_checkpoint: bool = False,
):
    """One reader forward pass, returning both logits and selected hidden states.

    `output_hidden_states=True` is what the phase-4 objective needs, and it is the
    reason this path cannot reuse `_forward_spans`: that helper only keeps logits.

    `pool` installs an explicit memory tensor for the duration of the call, which
    is how the student sees the compressed state.

    Checkpointing is per *layer* and is arranged by the caller, not here: see
    `train_carrier_compactor`, which enables the fork's
    `_gradient_checkpointing_func` branch. Wrapping this whole forward in
    `torch.utils.checkpoint` does not reduce peak memory (`use_reentrant=False`
    recomputes the entire forward on backward), so no outer wrapper is used.

    The hidden states returned are the model's own, one per requested layer. They
    are already query-length: `MemoryLLM.forward` appends `hidden_states` at the
    top of each layer iteration and trims the layer output back to
    `input_ids.shape[1]` (modeling_memoryllm.py:2032), so the memory prefix never
    enters `hidden_states` and `reader_hidden_loss` needs no extra slicing.
    """
    def run(ids, mem):
        if mem is None:
            out = model(input_ids=ids, output_hidden_states=True, return_dict=True)
        else:
            with differentiable_memory(model, mem):
                out = model(input_ids=ids, output_hidden_states=True, return_dict=True)
        return out.logits, out.hidden_states

    if enable_grad:
        logits, hidden = run(input_ids, pool)
    else:
        with torch.no_grad():
            logits, hidden = run(input_ids, pool)

    picked = []
    if hidden is not None:
        for index in layers:
            if 0 <= index < len(hidden):
                picked.append(hidden[index])
    return logits, picked


def _compacted_mask(manager, new_used, device):
    """Active-slot mask for a pool whose distinct suffix is `new_used` wide.

    Returns None when the run does not use the active mask, so a config with
    `active_memory_mask: false` keeps reading the pool exactly as before.
    """
    if not getattr(manager, "active_memory_mask", False):
        return None
    total = manager.layout.total
    k_anchor = manager.layout.k_anchor
    mask = torch.zeros(total, dtype=torch.bool, device=device)
    mask[:k_anchor] = True
    if new_used > 0:
        mask[total - new_used :] = True
    return mask


def _compacted_pool(manager, device, candidate, window_used, target):
    """The reader's pool after `window_used` slots are compacted into `target`.

    Reproduces `compact_with_report`'s placement exactly (layout.py:1105-1120),
    and it must be a *rearrangement*, never a shortening.

    Two properties are load-bearing:

    1. The pool keeps its config width (`num_tokens * num_blocks`). This fork
       derives `cache_position` from the config constants, so a shortened pool
       makes `rotary_emb` emit cos/sin for 12801 positions against a shorter q and
       `apply_rotary_pos_emb` raises `The size of tensor a (12) must match the
       size of tensor b (10252)`.
    2. Carriers land flush against the trailing end, with the protected newest
       block immediately after them. Substituting in place at `start` would leave
       the freed slots inside the distinct suffix, so the reader would attend to
       the carriers *and* the blocks they replaced.

    The freed slots at the front are the base pool's existing filler, which the
    active mask removes from attention. They are not re-tiled here: the filler's
    only role is to avoid reading uninitialised rows when the mask is off, and the
    base already provides that.
    """
    base = manager._memory.to(device).detach()
    total = manager.total
    used = manager.used
    new_used = used - window_used + target
    new_start = total - new_used
    remainder = base[:, total - (used - window_used) :, :]
    pool = torch.cat([base[:, :new_start, :], candidate.to(base.dtype), remainder], dim=1)
    if pool.shape[1] != total:
        raise ValueError(
            "_compacted_pool width %d != pool width %d" % (pool.shape[1], total)
        )
    return pool


class _InjectSlice(torch.autograd.Function):
    """Write `candidate` into `pool[:, start:start+target, :]` in place, differentiably.

    A plain `pool[...] = candidate` cannot be used here. `pool` is the frozen 8B
    model's `memory` parameter and does **not** require grad, and PyTorch only
    records an in-place assignment when it lands on a tensor that already carries
    a graph. Assigning into a non-requiring tensor silently drops the graph, so
    `candidate` would receive no gradient at all and the compactor would never
    train. (Verified on the host: `torch.autograd.grad(out, candidate)` raises
    "One of the differentiated Tensors does not require grad".)

    Doing the write inside a custom `Function` restores it: the forward performs
    the in-place write under no-grad, and the backward routes the output gradient
    for exactly the written slice back to `candidate`. The returned tensor is the
    same object as `pool`, so no second pool is allocated — the entire point,
    since `_compacted_pool`'s `torch.cat` would allocate another 3.36 GB here.

    The dtype cast lives inside `forward` because ops inside a custom `Function`
    are not tracked; the fp16 storage therefore stays fp16 while `candidate` keeps
    its fp32 gradient.
    """

    @staticmethod
    def forward(ctx, pool, candidate, start, target):
        ctx.start = int(start)
        ctx.target = int(target)
        pool[:, ctx.start : ctx.start + ctx.target, :] = candidate.to(pool.dtype)
        return pool

    @staticmethod
    def backward(ctx, grad_output):
        start, target = ctx.start, ctx.target
        return (
            None,
            grad_output[:, start : start + target, :].clone(),
            None,
            None,
        )


class _PoolInjection:
    """Live in-place injection of a compacted candidate into `model.memory`.

    Writes the candidate at the inference placement and restores the displaced
    slots on exit. `reinject()` re-runs the write and returns a *fresh* pool
    tensor carrying its own `_InjectSlice` node.

    The fresh node per probe is the point. One shared node would force every
    probe's backward to pass `retain_graph=True`, which keeps all reader graphs
    alive at once and defeats per-layer checkpointing. A fresh node lets each
    probe free its graph immediately, while gradients accumulate on the candidate
    leaf — which is where they are summed anyway.
    """

    def __init__(self, base, candidate, new_start, target):
        self.base = base
        self.candidate = candidate
        self.new_start = new_start
        self.target = target
        self.saved = base[:, new_start : new_start + target, :].clone()

    def reinject(self):
        return _InjectSlice.apply(
            self.base, self.candidate, self.new_start, self.target
        )

    def restore(self):
        with torch.no_grad():
            self.base[
                :, self.new_start : self.new_start + self.target, :
            ] = self.saved


@contextmanager
def _pool_in_place(manager, device, candidate, window_used, target):
    """Rearrange the window in place, yielding a live `_PoolInjection`.

    Same layout as `_compacted_pool`, but without the `torch.cat`. That `cat`
    allocates a second full-width pool: 3.36 GB at the E6a shape
    (`32 x 12800 x 4096` fp16), which is most of why the student forward OOM'd.
    Only `target` slots actually change, so the copy is almost entirely waste.

    The layout is contiguous by construction (layout.py:1105-1120):

        [ filler ... | candidate (target) | remainder (used - window) ]

    with `new_start = total - (used - window + target)`. The remainder's source
    range *is* its destination range, so it is already in place and needs no
    copy. Only `[new_start, new_start + target)` has to be written, via
    `_InjectSlice` so the graph survives.

    The displaced slots are saved and restored on exit, including on exception,
    so `model.memory` is unchanged after training.
    """
    # Deliberately the `nn.Parameter` rather than `manager._memory` (which
    # returns `.data`): `.data` bypasses autograd bookkeeping.
    base = manager.model.memory
    total = manager.total
    used = manager.used
    new_used = used - window_used + target
    new_start = total - new_used
    injection = _PoolInjection(base, candidate, new_start, target)
    try:
        yield injection
    finally:
        injection.restore()


def compressed_pool(
    manager,
    compactor,
    window_blocks: int,
    target_blocks: int,
    device=None,
) -> torch.Tensor:
    """The pool that compacting `window_blocks` into `target_blocks` actually yields.

    Deliberately mirrors the training loop's substitution rather than calling
    `compact_with_report`: the probe has to score the same object training
    optimises, and going through `manager.compact` would take a different path
    (block alignment, `newest_width` protection, the `native_filler_only`
    filler rewrite) and could differ from the trained operator's input.
    """
    device = manager.model.memory.device if device is None else device
    start = manager.total - manager.used
    window = int(window_blocks) * manager.block
    target = int(target_blocks) * manager.block
    if window > manager.used:
        # `_compacted_pool` derives `new_start` from `used - window + target`, so a
        # window wider than the pool makes that arithmetic negative and the
        # placement silently produces an over-wide pool. The failure it used to
        # raise ("width 13056 != 12800") pointed at the placement rather than at
        # the real cause, which is always a window the caller never wrote.
        raise ValueError(
            "compressed_pool: window %d slots exceeds the %d slots written"
            % (window, manager.used)
        )
    source = manager._memory[:, start : start + window, :].to(device)
    # The importance weights are not optional here. `_select_block_indices` scores
    # importance at weight 0.4, and without them it falls back to the
    # dispersion-based stand-in -- a *different* set of carriers than
    # `compact_with_report` picks for the same source. A screen that derives the
    # dropped blocks from the weighted scorer and then measures the unweighted one
    # is comparing two different operators, and every probe whose fact sat in a
    # dropped block would still be readable in the scored pool.
    weights = manager._window_importance(start, window)
    candidate = compactor(
        source,
        weights,
        target=target,
        block=manager.block,
        goal_direction=manager.goal_direction,
        select_weights=manager.select_weights,
    )
    pool = _compacted_pool(manager, device, candidate, window, target)
    mask = _compacted_mask(manager, manager.used - window + target, device)
    return pool, mask


@torch.no_grad()
def score_candidate(
    manager,
    compactor,
    probes: Sequence[Probe],
    window_blocks: int,
    target_blocks: int,
    max_probe_tokens: int = 256,
) -> Dict[str, float]:
    """Teacher-forced answer accuracy and gold loss of one candidate compaction.

    Accuracy is the measure the acceptance criteria are written in ("probe
    accuracy before vs after"), and it is a stricter test than the loss: a
    candidate can lower the mean loss while still flipping the greedy answer on
    individual probes, which is exactly the failure the criteria guard against.
    """
    device = manager.model.memory.device
    batches = _probe_batches(manager.tokenizer, probes, str(device), max_length=max_probe_tokens)
    if not batches:
        return {"accuracy": float("nan"), "gold_loss": float("nan"), "n": 0.0}

    pool, mask = compressed_pool(manager, compactor, window_blocks, target_blocks, device=device)

    hits = 0
    total = 0
    loss_sum = 0.0
    with active_mask_override(manager.model, mask), differentiable_memory(manager.model, pool):
        for input_ids, n_prefix, _ in batches:
            logits = manager.model(input_ids=input_ids, return_dict=True).logits
            span = logits[:, n_prefix - 1 : -1, :]
            labels = input_ids[:, n_prefix:]
            if labels.numel() == 0:
                continue
            loss_sum += float(
                F.cross_entropy(
                    span.reshape(-1, span.shape[-1]), labels.reshape(-1)
                )
            )
            predicted = span.argmax(dim=-1)
            hits += int(bool((predicted == labels).all(dim=-1).item()))
            total += 1

    if total == 0:
        return {"accuracy": float("nan"), "gold_loss": float("nan"), "n": 0.0}
    return {
        "accuracy": hits / total,
        "gold_loss": loss_sum / total,
        "n": float(total),
    }


def _stage_weights(stage: str) -> CompactObjective:
    """Two-stage weights from the design note.

    Stage A first teaches "do not break the reader": KD and the hidden-state term
    are the whole objective and the gold term is off, so the operator is pushed
    toward compressed-reader behaviour equal to uncompressed-reader behaviour.

    Stage B then adds the functional term. Solving representation compatibility
    before information selection is what the project's own findings argue for, and
    starting with all five terms at once makes the two fight from step one.
    """
    objective = CompactObjective()
    name = (stage or "B").upper()
    if name == "A":
        objective.gold = 0.0
        objective.kd = 1.0
        objective.reader_hidden = 0.1
        objective.carrier = 0.1
        objective.stats = 0.02
    elif name == "B":
        objective.gold = 1.0
        objective.kd = 0.5
        objective.reader_hidden = 0.1
        objective.carrier = 0.05
        objective.stats = 0.02
    else:
        raise ValueError("stage must be 'A' or 'B', got %r" % (stage,))
    return objective


def train_carrier_compactor(
    manager,
    probes: Sequence[Probe],
    objective: Optional[CompactObjective] = None,
    steps: int = 50,
    lr: float = 1e-3,
    allow_test_probes: bool = False,
    curriculum: Sequence[int] = _CARRIER_CURRICULUM,
    reader_layers: Sequence[int] = _READER_HIDDEN_LAYERS,
    max_probe_tokens: int = 256,
    use_amp: bool = False,
    window_blocks: Optional[int] = None,
    target_blocks: Optional[int] = None,
    stage: str = "B",
    use_checkpoint: bool = True,
    compactor: Optional[object] = None,
    ckpt_out: Optional[str] = None,
) -> Dict[str, float]:
    """Phase 4: train the carrier residual against the reader.

    Objective, with development starting weights from the design note:

        L = L_gold + 0.5*L_KD + 0.1*L_reader-hidden + 0.05*L_carrier + 0.02*L_stats

    Both the backbone and the reader are frozen; only the carrier module moves.
    The gold term is primary, KD is auxiliary, and the reader-hidden term is what
    makes this phase 4 rather than a longer phase 3.

    Resource discipline, all of it load-bearing on a 32 GB V100:

      * the pool keeps its config width and the freed region is excluded from
        attention by the active mask, rather than the pool being narrowed;
      * probes are capped at `max_probe_tokens`, which must exceed the ~163
        token rendered prompt or the answer span is scored mid-instruction;
        batch of one;
      * the candidate is cast to the pool dtype and the loss is scaled, so the
        activation graph stays fp16;
      * `torch.utils.checkpoint` is used when available, trading a second forward
        for a much smaller activation footprint.

    Returns a report dict; the trained module is attached to `manager.compactor`.

    On the earlier "known train/inference discrepancy": it is falsified, and the
    narrowing it described is not possible with this modeling fork. `MemoryLLM`
    derives `cache_position` from the config constants, so a narrowed pool makes
    `rotary_emb` emit cos/sin for 12801 positions against a shorter q and the
    forward raises. Training and inference therefore both run the fixed-width
    pool and the rotary offsets agree by construction; see `_compacted_pool`.
    """
    from methods.prlmc.consolidate import CarrierResidualCompactor

    assert_test_isolation(probes, allow_test=allow_test_probes)
    if not probes:
        raise ValueError("train_carrier_compactor needs at least one functional probe")
    if manager.used < 2 * manager.block:
        raise ValueError("not enough memory written to train a compactor")

    objective = objective or _stage_weights(stage)
    device = manager.model.memory.device

    start = manager.total - manager.used
    if window_blocks is not None:
        # Explicit override, used by the single-window overfit probe: the whole
        # point there is to hold the compaction shape fixed at a real 19 -> 9.
        window = int(window_blocks) * manager.block
        if window > manager.used:
            raise ValueError(
                "window_blocks=%d needs %d slots but only %d are written"
                % (int(window_blocks), window, manager.used)
            )
    else:
        window = min(
            manager.used - manager.newest_width,
            max(manager.cfg.compact_min_slots, manager.used // 2),
        )
        window = max(manager.block, (window // manager.block) * manager.block)
        window = min(window, manager.used - manager.newest_width)
    if window <= manager.block:
        raise ValueError("window is too small to train against")

    source = manager._memory[:, start : start + window, :].to(device)
    # Same reason as in `compressed_pool`: the carrier picker scores importance at
    # weight 0.4, so training without the weights would optimise a different
    # carrier set than the one inference actually places.
    train_weights = manager._window_importance(start, window)
    batches = _probe_batches(manager.tokenizer, probes, str(device), max_length=max_probe_tokens)
    if not batches:
        raise ValueError("every probe was truncated away; raise max_probe_tokens")

    # A caller-supplied module is trained in place. This is what lets a
    # before/after comparison score the *same* operator: rebuilding a fresh
    # compactor for the "before" measurement would compare two different random
    # inits and attribute the difference to training.
    if compactor is None:
        compactor = CarrierResidualCompactor(
            num_layers=manager.num_layers,
            hidden=manager.hidden,
            block=manager.block,
            rank=int(getattr(manager.cfg, "carrier_rank", 32)),
            gamma_max=float(getattr(manager.cfg, "carrier_gamma_max", 0.1)),
        )
    # The backbone is fp16, so a float32 module here would promote every activation
    # it touches and double the reader's footprint. Its *parameters* stay fp32
    # (they are cast per use inside `forward`), which is what keeps AdamW's update
    # precise for a small module.
    compactor = compactor.to(device=device)

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

    # Per-layer gradient checkpointing, and it is the difference between fitting
    # and OOM. A whole-forward `torch.utils.checkpoint` (the `use_checkpoint` path
    # below) does *not* help: with `use_reentrant=False` the backward recomputes
    # the entire forward, so peak activation memory is unchanged. The fork has a
    # real per-layer checkpoint branch in its decoder loop
    # (modeling_memoryllm.py:2003, gated on `gradient_checkpointing and training`),
    # which stores only each layer's input and recomputes one layer at a time.
    #
    # Requiring `training` is why the model is switched to train mode here. That is
    # safe in this fork: `attention_dropout` is 0.0 and the decoder stack has no
    # hidden/MLP dropout, so `.train()` adds no stochasticity.
    #
    # It is also safe with respect to `differentiable_memory`: the memory prefix is
    # concatenated into `hidden_states` *before* the decoder loop, and the loop
    # passes `memory_attention_bias` explicitly as a checkpoint argument. The
    # backward recompute therefore never re-reads `model.memory`.
    checkpointing_was = bool(getattr(manager.model, "gradient_checkpointing", False))
    training_was = manager.model.training
    inner_checkpointing_was = None
    if use_checkpoint:
        manager.model.gradient_checkpointing_enable()
        # The fork's decoder loop reads the flag off the *inner* `LlamaModel`
        # (`self.model.gradient_checkpointing`), while
        # `gradient_checkpointing_enable()` sets it on the outer `MemoryLLM`. Both
        # the flag and the callable must therefore be propagated explicitly, or the
        # branch never fires and the training step OOMs exactly as before.
        inner = getattr(manager.model, "model", None)
        if inner is not None:
            inner_checkpointing_was = bool(getattr(inner, "gradient_checkpointing", False))
            inner.gradient_checkpointing = True
            inner._gradient_checkpointing_func = _checkpoint_func()
        # `training` must be true for the branch as well. Safe in this fork:
        # `attention_dropout` is 0.0 and the decoder stack has no hidden/MLP dropout,
        # so `.train()` adds no stochasticity.
        manager.model.train()

    # Teacher pass over the *uncompacted* pool: the live one, at its config width.
    #
    # Full width is forced by the fork, not chosen for fidelity: `cache_position`
    # comes from the config constants, so a narrowed pool makes `rotary_emb`
    # emit cos/sin for 12801 positions against a shorter q and the forward
    # raises. Teacher and student therefore both run at the same width, which also
    # keeps the rotary offsets identical to the ones the reader sees at inference.
    _mem("before teacher", device)
    with torch.no_grad(), differentiable_memory(manager.model, manager._memory.to(device).detach()):
        teacher_logits = []
        teacher_hidden = []
        for input_ids, n_prefix, _ in batches:
            logits, hidden = _forward_with_hidden(
                manager.model, input_ids, reader_layers, enable_grad=False
            )
            teacher_logits.append(logits[:, n_prefix - 1 : -1, :].detach())
            teacher_hidden.append([h.detach() for h in hidden])
    _mem("after teacher", device)

    history: List[float] = []
    ratios: List[int] = []
    # Component trajectory. A falling total loss says nothing about *which* term
    # moved, and the acceptance criteria are stated per term (KD down, reader
    # hidden down), so they have to be recorded rather than inferred.
    components: Dict[str, List[float]] = {"gold": [], "kd": [], "hidden": [], "carrier": [], "stats": []}
    try:
        for step in range(int(steps)):
            if target_blocks is not None:
                keep_blocks = int(target_blocks)
            else:
                keep_blocks = curriculum[min(step * len(curriculum) // max(1, int(steps)), len(curriculum) - 1)]
            target = max(manager.block, min(window - manager.block, keep_blocks * manager.block))
            ratios.append(target)

            optimiser.zero_grad(set_to_none=True)
            _mem("step %d start" % step, device)

            # The compacted values are built WITHOUT a graph first. The
            # compactor's own graph is ~3.0 GB at this shape, and holding it
            # across the reader phase is the E6a OOM: reader checkpointing already
            # stores 32 full-width layer inputs (~3.4 GB), and the two together
            # left no room for the backward's recompute working set.
            #
            # `forward` is a deterministic function of `source` and the
            # (still unchanged) parameters, so the graph rebuilt after the reader
            # finishes produces the same tensor, and the gradient pushed through
            # it equals the one a single-pass forward would have produced.
            with torch.no_grad():
                candidate_val = compactor(
                    source,
                    train_weights,
                    target=target,
                    block=manager.block,
                    goal_direction=manager.goal_direction,
                    select_weights=manager.select_weights,
                )

            # Substitute the compacted result using the *inference* placement
            # (layout.py:1105-1120), not a naive overwrite of the window's first
            # slots. Done in place: `_compacted_pool`'s `torch.cat` allocates a
            # second full-width pool (3.36 GB here), which is most of why the
            # student forward OOM'd.
            # Gradient isolation boundary. The reader must read a *connected*
            # tensor (the whole point is to train the compactor through the
            # reader), but the tensor whose `.grad` accumulates across probes must
            # be the small one. Detaching the full pool into a leaf instead makes
            # autograd allocate a `[32, 12800, 4096]` fp16 `pool.grad` — 3.36 GB
            # held live for the whole probe loop, on top of the live compactor
            # graph. That sum is what OOM'd the 32 GB V100.
            #
            # `candidate` is `[32, 2304, 4096]` fp16 (604 MB). Because the only
            # path from `candidate` into the model output is the injected slice,
            # d(loss)/d(candidate) is exactly the slice of d(loss)/d(pool) that
            # the injected slots received. Accumulating that on `cand_leaf` and
            # pushing it through `candidate` once per step is therefore identical
            # to backpropagating the full pool, at a sixth of the memory.
            cand_leaf = candidate_val.detach().requires_grad_(True)
            # The freed slots stay physically present, so the mask is what stops
            # the reader from attending to the blocks the compaction dropped.
            mask = _compacted_mask(manager, manager.used - window + target, device)

            step_terms = {"gold": 0.0, "kd": 0.0, "hidden": 0.0}
            n_probes = 0
            with _pool_in_place(manager, device, cand_leaf, window, target) as injection:
                _mem("step %d before student" % step, device)
                with active_mask_override(manager.model, mask):
                    for index, (input_ids, n_prefix, _) in enumerate(batches):
                        # A *fresh* injection node per probe. A single shared node
                        # would force `retain_graph=True` on every probe's
                        # backward, keeping all 16 reader graphs alive at once and
                        # defeating per-layer checkpointing. A fresh node frees
                        # each reader graph immediately; the gradient accumulates
                        # on `cand_leaf`, which is where it is summed anyway.
                        pool = injection.reinject()
                        logits, hidden = _forward_with_hidden(
                            manager.model,
                            input_ids,
                            reader_layers,
                            enable_grad=True,
                            pool=pool,
                            use_checkpoint=use_checkpoint,
                        )
                        student_logits = [logits[:, n_prefix - 1 : -1, :]]
                        student_hidden = [hidden]

                        probe_gold = gold_loss_from_logits(
                            [batches[index]], student_logits, device
                        )
                        probe_kd = teacher_kd_loss(
                            [teacher_logits[index]], [batches[index]], student_logits
                        ).to(device)
                        probe_hidden = reader_hidden_loss(
                            teacher_hidden[index], student_hidden[0], layers=reader_layers
                        )

                        # The carrier and stats terms do not depend on the probe,
                        # so they enter the objective once per probe and are
                        # normalised by the probe count below, keeping their
                        # weight independent of it. They reach the compactor
                        # directly, not through `pool`, so they never touch
                        # `pool.grad`.
                        carrier_term = torch.zeros((), device=device)
                        if compactor.up.weight.numel() > 0:
                            carrier_term = compactor.up.weight.pow(2).mean()
                        stats_term = (compactor.gamma.to(device) ** 2).reshape(())

                        probe_loss = objective.phase4_total(
                            gold_loss=probe_gold,
                            kd_loss=probe_kd,
                            reader_hidden_loss=probe_hidden,
                            carrier_loss=carrier_term,
                            stats_loss=stats_term,
                            device=str(device),
                        )
                        # Backward stops at `cand_leaf`: the reader graph is freed
                        # here, so peak activation memory is one probe's worth, not
                        # sixteen. The compactor graph is *not* reachable from this
                        # call, so it survives untouched for the single backward
                        # below — that is what makes the plain form correct and
                        # lets the compactor's own layer checkpoints do their job.
                        if scaler.is_enabled():
                            scaler.scale(probe_loss).backward()
                        else:
                            probe_loss.backward()
                        del probe_loss, student_logits, student_hidden, logits, hidden, pool

                        step_terms["gold"] += float(probe_gold.detach())
                        step_terms["kd"] += float(probe_kd.detach())
                        step_terms["hidden"] += float(probe_hidden.detach())
                        n_probes += 1

                # One backward through the compactor, after the reader graphs are
                # gone. `cand_leaf` holds the summed d(loss)/d(candidate) over all
                # probes. Rebuilding the forward here is what keeps the 8B reader
                # graph and the compactor graph from ever being live at the same
                # time -- that coexistence, not the attention scores, was the E6a
                # OOM. `forward` reads only `source` and the parameters, both
                # unchanged since the no-grad pass, so the rebuilt tensor is
                # bit-identical and this backward is the exact gradient the single
                # stacked backward would have produced.
                if cand_leaf.grad is not None:
                    candidate = compactor(
                        source,
                        train_weights,
                        target=target,
                        block=manager.block,
                        goal_direction=manager.goal_direction,
                        select_weights=manager.select_weights,
                    )
                    torch.autograd.backward([candidate], [cand_leaf.grad])
                    del candidate

                if os.environ.get("PRLMC_GRAD_TRACE") == "1":
                    gnorm = "None" if cand_leaf.grad is None else "%.4e" % float(cand_leaf.grad.norm())
                    parts = []
                    for name, parameter in compactor.named_parameters():
                        grad = parameter.grad
                        parts.append(
                            "%s=%s" % (name, "None" if grad is None else "%.4e" % float(grad.norm()))
                        )
                    print(
                        "[grad] step %d cand_leaf=%s connected=%s scale=%.1f | %s"
                        % (
                            step,
                            gnorm,
                            bool(candidate_val.requires_grad),
                            float(scaler.get_scale()),
                            " ".join(parts),
                        ),
                        flush=True,
                    )
            del cand_leaf, candidate_val, injection

            step_terms["carrier"] = float(carrier_term.detach())
            step_terms["stats"] = float(stats_term.detach())

            if scaler.is_enabled():
                scaler.step(optimiser)
                scaler.update()
            else:
                optimiser.step()

            scale = max(1, n_probes)
            history.append(
                objective.gold * step_terms["gold"] / scale
                + objective.kd * step_terms["kd"] / scale
                + objective.reader_hidden * step_terms["hidden"] / scale
                + objective.carrier * step_terms["carrier"]
                + objective.stats * step_terms["stats"]
            )
            components["gold"].append(step_terms["gold"] / scale)
            components["kd"].append(step_terms["kd"] / scale)
            components["hidden"].append(step_terms["hidden"] / scale)
            components["carrier"].append(step_terms["carrier"])
            components["stats"].append(step_terms["stats"])
    finally:
        for parameter in frozen:
            parameter.requires_grad_(True)
        if use_checkpoint:
            if not checkpointing_was:
                manager.model.gradient_checkpointing_disable()
            inner = getattr(manager.model, "model", None)
            if inner is not None and inner_checkpointing_was is not None:
                inner.gradient_checkpointing = inner_checkpointing_was
            manager.model.train(training_was)

    if ckpt_out:
        torch.save({"state_dict": compactor.state_dict()}, ckpt_out)

    manager.compactor = compactor
    manager.carrier_compactor = compactor
    return {
        "steps": float(steps),
        "window_tokens": float(window),
        "target_tokens": float(ratios[-1]) if ratios else float("nan"),
        "initial_loss": history[0] if history else float("nan"),
        "final_loss": history[-1] if history else float("nan"),
        "components": components,
        "state_report": compactor.state_report(),
        "gamma": float(compactor.gamma.detach()),
        "gamma_max": float(compactor.gamma_max),
        "curriculum": [float(v) for v in ratios],
        "amp": 1.0 if scaler.is_enabled() else 0.0,
    }
