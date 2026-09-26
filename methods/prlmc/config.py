"""PRLM-C hyperparameters.

Everything the memory manager needs lives here, so an ablation is a config diff
rather than a code diff. The benchmark's agent config carries the same mapping
under the "prlmc" key; unknown keys are ignored so older configs keep loading.

Defaults implement the note's first version: row-wise memory tokens, read-only
Anchor slots, append-then-consolidate working memory, geometric compaction. The
learned writer and the learned compressor are opt-in, because both need a
training run and the note requires the geometric baseline to exist first
(section 3.4).

Thresholds are explicitly *not* tuned: the note says the values mentioned in the
original discussion are examples, and that every threshold must be chosen on a
dev split (sections 3.3 and 5.2). The defaults below are placeholders.
"""

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional


def _coerce(value: Any, name: str) -> Any:
    """Coerce a YAML scalar to the declared type of field `name`.

    An explicit table rather than eval over the annotation string: the
    annotations include Dict/Optional forms that are not evaluable in this scope,
    and silently falling back would let a string "256" reach arithmetic.
    """
    if name in _INT_FIELDS:
        return int(value)
    if name in _FLOAT_FIELDS:
        return float(value)
    if name in _BOOL_FIELDS:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if name in _OPTIONAL_INT_FIELDS:
        return None if value is None else int(value)
    if name == "projection_lambdas":
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        return float(value)
    if name == "importance_weights":
        return {str(k): float(v) for k, v in dict(value or {}).items()}
    if name == "select_weights":
        return {str(k): float(v) for k, v in dict(value or {}).items()}
    if name in _STR_FIELDS:
        return str(value)
    return value


_INT_FIELDS = frozenset({
    "k_anchor", "basis_max_rank", "additive_width", "projection_rank",
    "trigger_consecutive", "rank_max", "compact_period", "compact_min_slots",
})
_FLOAT_FIELDS = frozenset({
    "address_threshold", "novelty_ema_beta", "basis_energy", "novelty_eps",
    "tau_k", "tau_n", "tau_k_exit", "tau_n_exit", "tau_alloc",
    "rank_eta", "compact_ratio",
})
_BOOL_FIELDS = frozenset({"compact_probe", "verbose", "active_memory_mask", "native_filler_only"})
_OPTIONAL_INT_FIELDS = frozenset({"k_max", "seed"})
_STR_FIELDS = frozenset({
    "mode", "mode_code", "anchor_source", "anchor_text", "write_mode",
    "gate_mode", "projection", "trigger", "compact_mode", "dtype", "device",
    "stats_dir",
})


@dataclass
class PRLMCConfig:
    # ---- ablation identity -------------------------------------------------
    # mode is the *mechanism* switch used by the write path.
    mode: str = "ours"
    # mode_code is the section 5.1 arm label (B5/B6/B7/B8/Ours). Kept separate
    # because several arms share a mechanism switch but differ in their defaults,
    # and a results table must be able to name the arm it ran.
    mode_code: str = "OURS"

    # ---- slot layout (section 3.1) ----------------------------------------
    # k_anchor is the number of read-only slots holding the initial goal and hard
    # constraints. 256 is one MemoryLLM memory block, which is the smallest unit
    # the injection forward can produce.
    k_anchor: int = 256
    # Working slots are carved out of the remainder and split dynamically into a
    # consolidated region S and a residual region R.
    k_max: Optional[int] = None  # resolved from the backbone at load time
    anchor_source: str = "first_chunk"  # first_chunk | explicit | none
    anchor_text: Optional[str] = None

    # ---- residual writing (section 3.2) -----------------------------------
    # append  : R grows by new slots, the gate scales the incoming block
    # additive: R keeps a fixed width and updates addressed slots in place
    write_mode: str = "append"
    gate_mode: str = "novelty"  # ones | novelty | learned
    # Cosine above which an incoming slot is treated as a correction of an
    # existing one rather than a new fact.
    address_threshold: float = 0.85
    additive_width: int = 2048

    # ---- novelty (section 3.2) --------------------------------------------
    novelty_ema_beta: float = 0.9
    basis_max_rank: int = 256
    basis_energy: float = 0.95
    novelty_eps: float = 1e-8
    # Below this novelty the candidate is judged explainable by the existing
    # working subspace, so it updates an addressed block instead of allocating
    # new slots. This is what makes `used` mean "how much independent content is
    # stored" rather than "how many writes happened".
    tau_alloc: float = 0.5

    # ---- Anchor projection ablation (section 3.2) -------------------------
    projection: str = "none"  # none | hard | soft
    # Scalar lambda for the soft variant; a list is accepted and broadcast.
    projection_lambdas: float = 0.5
    projection_rank: int = 128

    # ---- capacity signals (section 3.3) -----------------------------------
    tau_k: float = 0.80
    tau_n: float = 0.50
    tau_k_exit: float = 0.60
    tau_n_exit: float = 0.30
    trigger_consecutive: int = 5
    rank_eta: float = 0.95
    rank_max: int = 256

    # ---- compaction (section 3.4) -----------------------------------------
    trigger: str = "occupancy_novelty_rank"
    # none | recent_blocks | select_blocks | svd_tokens | weighted_svd_tokens |
    # learned | carrier_residual
    # `recent_blocks` and `select_blocks` are native block selection: they copy
    # whole 256-slot blocks the reader already produced and never synthesise a
    # vector. `svd_tokens` is retained only as a failed baseline.
    compact_mode: str = "svd_tokens"
    compact_ratio: float = 0.5
    # Weights of the `select_blocks` score. Recency-led by default; see
    # consolidate._DEFAULT_SELECT_WEIGHTS for why.
    select_weights: Dict[str, float] = field(default_factory=lambda: {
        "importance": 0.4,
        "recency": 0.3,
        "goal": 0.2,
        "diversity": 0.1,
    })
    compact_period: int = 10
    compact_min_slots: int = 256
    # Phase 2: mask the free region out of the reader's attention instead of
    # trusting the filler to stay harmless. Active keys are [A, S, R]; the free
    # region [k_anchor, total - used) gets the dtype minimum as an additive
    # bias, so SDPA assigns it exactly zero weight. Requires the Phase 2 patch
    # in `modeling_memoryllm.py`; with an unpatched file the flag is inert and
    # the pool is read exactly as before.
    active_memory_mask: bool = False
    # Phase 2 interim fallback: restrict free-region filler to uncompressed
    # native blocks, never compactor output. Repeated native tokens still
    # perturb the attention probability mass, so this is a stopgap for runs
    # where the active mask is off; with the mask on it is belt-and-braces.
    native_filler_only: bool = False
    # Diagnostic probes are evaluated on a copy of the state, never on the live
    # one, so the running agent is not reminded of the probe questions.
    compact_probe: bool = True

    # ---- importance weights (section 3.4) ---------------------------------
    importance_weights: Dict[str, float] = field(default_factory=lambda: {
        "goal": 1.0,
        "reuse": 0.5,
        "recency": 0.5,
        "confidence": 0.5,
        "surprise": 0.5,
        "emphasis": 1.0,
    })

    # ---- runtime ----------------------------------------------------------
    dtype: str = "float16"
    device: str = "cuda:0"
    seed: Optional[int] = None
    stats_dir: Optional[str] = None
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.write_mode not in ("append", "additive"):
            raise ValueError(f"unknown write_mode: {self.write_mode}")
        if self.gate_mode not in ("ones", "novelty", "learned"):
            raise ValueError(f"unknown gate_mode: {self.gate_mode}")
        if self.projection not in ("none", "hard", "soft"):
            raise ValueError(f"unknown projection: {self.projection}")
        if self.compact_mode not in (
            "none",
            "recent_blocks",
            "select_blocks",
            "svd_tokens",
            "weighted_svd_tokens",
            "learned",
            "carrier_residual",
        ):
            raise ValueError(f"unknown compact_mode: {self.compact_mode}")
        if self.anchor_source not in ("first_chunk", "explicit", "none"):
            raise ValueError(f"unknown anchor_source: {self.anchor_source}")
        if not 0.0 < self.compact_ratio <= 1.0:
            raise ValueError("compact_ratio must be in (0, 1]")
        if self.anchor_source == "explicit" and not self.anchor_text:
            raise ValueError("anchor_source=explicit requires anchor_text")
        # Hysteresis is only meaningful if the exit thresholds sit below the
        # entry ones; silently accepting the reverse would make the trigger
        # chatter every update.
        if self.tau_k_exit > self.tau_k:
            raise ValueError("tau_k_exit must be <= tau_k")
        if self.tau_n_exit > self.tau_n:
            raise ValueError("tau_n_exit must be <= tau_n")

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "PRLMCConfig":
        """Build a config from a YAML mapping, ignoring unknown keys.

        Unknown keys are ignored rather than rejected so that a config written for
        a later version still loads; a typo therefore degrades to the default
        instead of raising, which is why `describe()` echoes the effective values
        into the run log.
        """
        if not raw:
            return cls()

        known = {f.name for f in fields(cls)}
        kwargs: Dict[str, Any] = {}
        for key, value in raw.items():
            if key not in known:
                continue
            kwargs[key] = _coerce(value, key)
        return cls(**kwargs)

    def describe(self) -> Dict[str, Any]:
        """Effective config, for the run log. Makes silent defaults visible."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def projection_lambda_vector(self, length: int, device, dtype):
        """Broadcast projection_lambdas to a length-`length` vector."""
        import torch

        if isinstance(self.projection_lambdas, (list, tuple)):
            values: List[float] = [float(v) for v in self.projection_lambdas]
        else:
            values = [float(self.projection_lambdas)] * length
        if not values:
            values = [1.0] * length
        if len(values) < length:
            values = values + [values[-1]] * (length - len(values))
        return torch.tensor(values[:length], device=device, dtype=dtype)
