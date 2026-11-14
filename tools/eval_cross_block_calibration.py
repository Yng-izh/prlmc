#!/usr/bin/env python3
"""Experiment 10 (V14): does the SOURCE of the neighbours buy anything?

    Full              19 blocks, uncompressed                     reference ceiling
    Selection         9 carrier blocks, verbatim                  capacity baseline
    Graft             V10 native graft, 1,944 + 360               the arm AnchorOnly beat
    AnchorOnly        the frozen AnchorResidual at 360 anchors     the 56/158 baseline
    V11Evict          V11Evict's window + the SAME module trained with --cross off
    V11EvictDonor     V11Evict, graft slots taken from ANOTHER document
    V13               the cross-block conditioned module, k = 4 kind-blind neighbours
    V13Donor          V13, graft slots taken from ANOTHER document
    V14               the SAME window as V13, 4 neighbours = 2 of the query's own
                      core/graft kind + 2 from the other side
    V14Donor          V14, graft slots taken from ANOTHER document

    V15               the layered sink-gated relation module on the SAME window as V13
    V15Donor          V15 on the donor window
    V16               the band-aligned, margin-gated module on the SAME window as V13
    V16Donor          V16 on the donor window
    V13Permuted<seed> inference-time attribution arm (one per `--v13-permute-seeds`
                      seed): V13's own window, weights and self-calibration branch, with
                      the precomputed cross-block context rows shuffled inside
                      (source_block, slot_type). No retraining, no re-selection.

    V17               key anchoring + value release (methods/prlmc/deep_fusion.py) on the
                      SAME window as V13: the key-visible part of the residual is capped,
                      the part the key cannot see is released up to `value_ratio`
    V17Value<pp>      one per `--v17-value-ratios`: V17's weights with ONLY the runtime
                      value ceiling changed (0.25 -> V17Value25). Inference time only.
                      The sweep ALWAYS contains the ceiling the checkpoint actually
                      clips at (`value_ratio_effective`, i.e. `max_ratio` when
                      `value_ratio` is None): the arm is added to the list if missing
    V17NoAttn         `--v17-no-attn`: mechanism four (graft attention damping) turned
                      OFF. The trainer damps the window before it measures the neighbour
                      context, so a checkpoint with non-neutral constants damps the
                      window again inside `deep_fusion.calibrate`; this arm neutralises
                      tau / bias / norm_quantile on a copy and adds no damping of its
                      own, so `V17 - V17NoAttn` is what that mechanism is worth
    V17NoBand         `--v17-band-off`: the same checkpoint with the three-band gate off
    V17Attn           `--v17-attention`: graft-only key damping applied ONCE by this
                      driver. Only built when the checkpoint's own constants are
                      neutral (an older checkpoint, or a deliberate switch-on): a
                      checkpoint that already damps inside `calibrate` would make this
                      arm a duplicate of `V17`, so it is WARNED about and SKIPPED -- the
                      control for mechanism four there is `--v17-no-attn`
    V13Density        `--v17-window-density`: V13's module on a window re-quotaed per
                      block by redundancy (same 1,620 core slots, same capacity)
    V17Density        V17's module on that same density window, so the module and the
                      quota rule are each measured with the other held fixed

The V17 arms are inference-time only where they are labelled so: they deep-copy the
checkpoint's module and replace ONE runtime attribute, so the checkpoint on disk and the
`V17` arm itself are never mutated. Every new arm is summarised, damaged and paired like
the existing ones; a knob that leaves the calibrated window identical is recorded per row
(`window_equals_v17`) and warned about rather than reported as a gain.

Two claims this driver refuses to make silently
-----------------------------------------------
* a `sim_floor` below the p10 of the run's first neighbour-similarity distribution
  filters almost nothing, so that arm is evidence NEITHER way. The run says so on stderr
  and records `sim_floor_filters_nothing` in the summary;
* with `band_mode="three"` and the factory layer table (12, 16, 20, 24, 28) no window
  layer falls in band 0, so that gate never sees a gradient and the "three-band" gate is
  a two-band one here. `band0_observable` and `gate_parameters_live` record that instead
  of leaving a reader to count three zeros in the checkpoint.

Neither raises: both are warnings plus a recorded field, because a floor that filters
nothing and a band that never trains are findings about the arm, not crashes.

V13 and V14 read the IDENTICAL window: `built["D"]`'s window is calibrated twice, by two
modules -- same 2,304 slots, same 1,620 / 684 split, same graft content, same rank,
ceilings, regulariser, loss, updates, cross learning rate and initialisation. The only
thing that differs is which slots each context row is built from, so `V14 - V13` measures
the neighbour rule and nothing else.

The questions, in the order they have to be answered
----------------------------------------------------
1. `V14 > V13` -- is the complement side worth anything over the kind-blind top-k that
   already reaches 67/158? The plan's bar is +4 to +5 questions net; below that the
   change is noise and the neighbour source was not the bottleneck.
2. `V14 > V13Donor` and `V14 > V14Donor` -- the gain still has to come from the CONTENT of
   the graft slots. Every mechanism in this project that passed (1) and failed (2) moved
   answers without carrying knowledge. `V13 > V11Evict` and `V11Evict > V11EvictDonor`
   are reported unchanged, so this run re-answers the earlier tests too.
3. Damage on the rows the uncompressed window already answers must not grow: a mechanism
   that buys questions by losing others is a redistribution, not a gain.

The permutation arms answer a different question: does V13's gain depend on WHICH slot
receives WHICH neighbour context, or would any context of the same provenance do? The
context is permuted AFTER it is computed, inside (source_block, slot_type) groups, so the
marginal distribution -- neighbour count, similarity range, block spread, context RMS --
is preserved and only the pairing is destroyed. Recorded for them: total correct, how many
of the rows V13 newly rescued over the plain calibration come back wrong, the damage on
Full's 47 protected rows, and the fraction of slots whose context actually moved (a
permutation that moved nothing is refused, not reported as "no effect").

The composition V13 never reported is now recorded per row, for every cross arm:
`neighbour_type_cross_fraction` (share of the selected neighbours from the other side of
the core/graft divide), `core_to_graft_neighbour_fraction`,
`graft_to_core_neighbour_fraction`, and `cross_weight_mass_mean` (how much of the context
weight the complement side actually received -- a complement arm whose cross mass sits
near zero has silently degenerated into "top-2 of my own kind").

Two notes on what this driver cannot do
---------------------------------------
There is no second dataset in this file. The independent natural multi-hop set the plan
asks for is a separate run against a different `--task`/`--data-dir`; the protocol block
records `task`, `data_dir` and their sha256 so two runs can be compared safely.
`V11Evict` here is NOT the checkpoint from experiment 7 -- that one was deleted together
with the V11 code -- it is the same module retrained with `--cross off` inside this
round's framework, which is the cleaner comparison anyway (one variable differs).

What is checked before a number is read
--------------------------------------
* the four pre-existing arms must reproduce the reference report row for row;
* every compressed arm holds 2,304 slots split 1,620 / 684, chronologically ordered,
  every slot a verbatim copy of the source position it claims;
* the calibration must move both halves, and a `--v11-evict` checkpoint must really be
  cross-block conditioned (`--v11` must really not be);
* `--v14` must be a complement-mode checkpoint and `--v11-evict` a top-k one, with the
  same neighbour count, or the two arms would not differ in the neighbour source alone;
* the donor arm's window must differ from the true one;
* V14's calibrated window must differ from V13's: a complement rule that changes no slot
  changed nothing, and must not be reported as "no gain";
* the cross-block context must be built from OTHER blocks -- the neighbour spread is
  recorded per row, and a context with only same-block neighbours is a bug, not a result.

Usage
-----
    python tools/eval_cross_block_calibration.py --dry-run \
        --data-dir /data/.../LongBench/extracted/data --model-path /dev/shm/s1d_reader_fp16

    python tools/eval_cross_block_calibration.py \
        --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
        --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
        --model-path /dev/shm/s1d_reader_fp16 \
        --checkpoint runs/natural_merge_v9/full_v10.pt \
        --anchor runs/anchor_residual/anchor_only_rank64.pt \
        --v11 runs/cross_block/plain.pt \
        --v11-evict runs/cross_block/cross_lr.pt \
        --v14 runs/cross_block/complement.pt \
        --verify-against reports/anchor_fusion/eval_anchor_fusion_hotpotqa.json \
        --cuda-device 0 --out reports/cross_block/eval_v14_hotpotqa.json
"""
import argparse
import copy
import inspect
import math
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_graft_recovery as gr                          # noqa: E402
import eval_learned_fusion as ef                          # noqa: E402
import eval_replace_reorder as er                         # noqa: E402
import eval_source_compaction as es                       # noqa: E402
import native_eval as nv                                  # noqa: E402
from io_utils import atomic_json, sha                     # noqa: E402
from runtime import load_runtime, reset                   # noqa: E402
from methods.prlmc.anchor_residual import (               # noqa: E402
    SCHEMA as ANCHOR_SCHEMA, AnchorResidual,
    compact_window as compact_window_anchor)
from methods.prlmc import band_relation as br             # noqa: E402
from methods.prlmc import hierarchical_relation as hr       # noqa: E402
from methods.prlmc.cross_block_calibration import (       # noqa: E402
    SCHEMA as CALIB_SCHEMA, CrossBlockResidual, calibrate, cross_block_context,
    permute_context, relevance_graft_caps, slot_relevance)
from methods.prlmc.evict_window import (                  # noqa: E402
    SCHEMA as JOINT_SCHEMA, JointConsolidation)

# V13 reuses the V11 driver: the calibration module is what changed.
FULL_POOL_SCHEMA = CALIB_SCHEMA
FullPoolResidual = CrossBlockResidual
from methods.prlmc.reader_utility import ReaderScorer     # noqa: E402
from methods.prlmc.source_compaction import split_source_anchors  # noqa: E402

BLOCK_TOKENS = gr.BLOCK_TOKENS
SLOTS_PER_BLOCK = gr.SLOTS_PER_BLOCK
TARGET_SLOTS = gr.TARGET_SLOTS
FULL, SELECTION, GRAFT = ef.FULL, ef.SELECTION, ef.GRAFT
ANCHOR_ONLY = "AnchorOnly"
# The slots of the V11 driver, relabelled for V13: the plain full-pool calibration is
# now the arm to BEAT (it is `V11Evict` retrained here with --cross off), and the
# cross-block conditioned module is the treatment (`V13`).
V11 = "V11Evict"
V11_DONOR = "V11EvictDonor"
V11_EVICT = "V13"
V11_EVICT_DONOR = "V13Donor"
# V14: the same window, the same modules, the same training budget -- only the neighbour
# rule differs (2 of the query's own kind + 2 from the other side of the core/graft
# divide, instead of the 4 most similar cross-block slots whatever kind they are).
V14 = "V14"
V14_DONOR = "V14Donor"
# V15: the same window again, calibrated by the layered sink-gated relation module.
V15 = "V15"
V15_DONOR = "V15Donor"
# V16: band-aligned injection + post-projection margin gate, on V13's window again.
V16 = "V16"
V16_DONOR = "V16Donor"
# The V13 line's own successor: same module class/schema, different context rule
# (span-value neighbours, in-block diffusion, CSLS seats) and a tighter graft ceiling.
# Both checkpoints load through the same loader; each rebuilds its rule from its config.
V13_PLUS = "V13Plus"
V13_PLUS_DONOR = "V13PlusDonor"
# V17: key anchoring + value release (`methods/prlmc/deep_fusion.py`). Every V17 arm
# calibrates the SAME window as V13 (`--v11-evict`), so `V17 - V13` moves the module and
# nothing else. The derived arms are inference-time only -- they deep-copy the
# checkpoint's module and turn ONE runtime knob (value ceiling, band gate, graft
# attention), so no weight is rewritten and the main arm is never polluted.
DEEP_SCHEMA = "deep_fusion/1"
V17 = "V17"
V17_NO_BAND = "V17NoBand"
V17_ATTN = "V17Attn"
# Mechanism four OFF: the trainer damps the window before it measures the neighbour
# context, so `deep_fusion.calibrate` damps a rebuilt checkpoint again by itself. The
# only way to read a checkpoint with that mechanism switched off is to neutralise its
# three constants on a copy -- which is what this arm is.
V17_NO_ATTN = "V17NoAttn"
# The density-quota windows: the same 1,620 core slots, redistributed across the carrier
# blocks by redundancy instead of the fixed 180 per block. Two arms share that window, so
# the only difference between them is the module -- exactly as V13 and V17 share the
# fixed-quota window.
V13_DENSITY = "V13Density"
V17_DENSITY = "V17Density"
VALUE_PREFIX = "V17Value"
# The V17 arms whose names are fixed (the value-ceiling sweep is named per ratio).
DEEP_ARMS = (V17, V17_NO_BAND, V17_NO_ATTN, V17_ATTN)
DENSITY_ARMS = (V13_DENSITY, V17_DENSITY)
ARMS = (FULL, SELECTION, GRAFT, ANCHOR_ONLY, V11, V11_DONOR, V11_EVICT, V11_EVICT_DONOR,
        V13_PLUS, V13_PLUS_DONOR, V14, V14_DONOR, V15, V15_DONOR, V16, V16_DONOR,
        V17, V17_NO_BAND, V17_NO_ATTN, V17_ATTN, V13_DENSITY, V17_DENSITY)
# Inference-time attribution arms for V13's cross-block correspondence: same window, same
# module, same self-calibration branch, context rows shuffled inside (source_block,
# slot_type). They are named after their seed and appended to `ARMS` at run time.
PERMUTE_PREFIX = "V13Permuted"
# Attribution arms for the neighbour CHOICE: V13's weights, window, projection and
# neighbour count are untouched; only the rule that picks the four slots changes.
NEIGHBOR_ARMS = {"random": "V13RandNeighbor", "bottom": "V13BottomNeighbor"}


def cap_arm(scale):
    """Arm name for a uniform inference-time graft ceiling: 0.38 -> V13Cap038."""
    return f"V13Cap{int(round(float(scale) * 100)):03d}"


def parse_scales(raw):
    if not raw:
        return []
    return [float(part) for part in str(raw).replace(" ", "").split(",") if part]


def permute_arm(seed):
    return f"{PERMUTE_PREFIX}{int(seed)}"


def parse_seeds(raw):
    """`--v13-permute-seeds 0,1,2` -> [0, 1, 2]; empty means the arms are not built."""
    if not raw:
        return []
    seeds = [int(part) for part in str(raw).replace(" ", "").split(",") if part != ""]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"--v13-permute-seeds repeats a seed: {raw}")
    return seeds


def value_arm(ratio):
    """Arm name for an inference-time value ceiling: 0.25 -> V17Value25, 0.34 -> V17Value34.

    The percentage is `ratio * 100` rounded HALF UP to a whole percent (round-half-up,
    i.e. 四舍五入), not Python's `round`: `round` sends an exact half to the even digit,
    so 0.125 would be 12 % and 0.135 would be 14 %, which is not what a two-digit
    percentage in an arm name is read as. The `1e-9` absorbs the binary representation
    error of the product itself (0.345 * 100 is 34.499999999999996 in float64, and
    四舍五入 of that number is 35). Two ratios that land on the same name are refused
    rather than letting one arm silently overwrite the other.
    """
    return f"{VALUE_PREFIX}{int(math.floor(float(ratio) * 100 + 0.5 + 1e-9)):02d}"


def unique_value_arms(ratios, flag="--v17-value-ratios"):
    """`{arm: ratio}` for a sweep, refusing two ratios that share one arm name."""
    named = {}
    for ratio in ratios:
        arm = value_arm(ratio)
        if arm in named:
            raise ValueError(f"{flag}: {named[arm]} and {float(ratio)} both round to "
                             f"{arm}: one arm would silently replace the other")
        named[arm] = float(ratio)
    return named


def with_baseline_ratio(ratios, module, flag="--v17-value-ratios", tol=1e-6):
    """The user's sweep PLUS the ceiling this checkpoint actually clips at. (C)

    `value_ratio=None` means "the parent's `max_ratio`", and the trainer's `--max-ratio`
    default is 0.34 -- not the constructor's 0.5. A sweep of 0.25 / 0.30 / 0.40 read
    against that module is therefore compared with a ceiling no arm in the table holds,
    and every "the sweep beat the checkpoint" line would be an artefact of an unlisted
    number. The effective ceiling is added to the sweep, under the same naming rule, so
    the baseline appears in the table like every other arm.
    """
    effective = getattr(module, "value_ratio_effective", None)
    if effective is None:
        raise ValueError("the V17 module carries no `value_ratio_effective`: the sweep "
                         "cannot be anchored to the ceiling this checkpoint clips at, and "
                         "guessing it is exactly the error this guard exists for")
    explicit = getattr(module, "value_ratio", None)
    effective = float(effective)
    source = "module" if explicit is not None else "max_ratio"
    ratios = [float(ratio) for ratio in ratios]
    listed = any(abs(ratio - effective) <= float(tol) for ratio in ratios)
    added = None
    if not listed:
        added = effective
        ratios = ratios + [effective]
        print(f"NOTE: {flag}: the checkpoint clips the value side at {effective:.6g} "
              f"(source: {source}); {effective:.6g} was added to the sweep as "
              f"{value_arm(effective)} so the baseline is an arm in the table and not an "
              f"unlisted number.", file=sys.stderr)
        unique_value_arms(ratios, flag)
    return dict(
        ratios=ratios, arms=unique_value_arms(ratios, flag),
        value_ratio_effective=effective, value_ratio_effective_source=source,
        baseline_arm=value_arm(effective), baseline_listed=bool(listed),
        added_arm=(value_arm(added) if added is not None else None),
        note=("`value_ratio_effective` is what the value side really clips at: "
              "`value_ratio` when the module sets one ('module'), `max_ratio` otherwise "
              "('max_ratio', the trainer's --max-ratio default 0.34). A sweep that omits "
              "it compares every arm against a ceiling the checkpoint does not use"))


def parse_ratios(raw, flag="--v17-value-ratios"):
    """`0.25,0.30,0.40` -> [0.25, 0.3, 0.4].

    The ceiling is the share of a slot's RMS the VALUE-visible half of the residual may
    spend, so a ratio of 0 or above 1 is refused rather than clamped: a silently clamped
    sweep reports the same arm three times.
    """
    ratios = []
    for ratio in parse_scales(raw):
        if not 0.0 < float(ratio) <= 1.0:
            raise ValueError(f"{flag} takes ceilings in (0, 1]; got {ratio}")
        if float(ratio) in ratios:
            raise ValueError(f"{flag} repeats a ratio: {raw}")
        ratios.append(float(ratio))
    return ratios
# The baseline quota, which is what the reference report's arms were built at.
BASELINE_QUOTA = dict(core_per_block=216, graft_per_block=40)
CORE_TOTAL = 9 * 180
GRAFT_TOTAL = 9 * 76
SCHEMA_RESULTS = "cross_block_calibration_results/1"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_anchor_module(path, hidden):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != ANCHOR_SCHEMA + "/checkpoint":
        raise ValueError(f"{path} is not a {ANCHOR_SCHEMA} checkpoint")
    config = dict(payload["config"])
    if int(config.get("hidden_size", -1)) != int(hidden):
        raise ValueError(f"{path} was built for hidden {config.get('hidden_size')}")
    module = AnchorResidual(**config)
    module.load_state_dict(payload["state_dict"], strict=True)
    return module.eval().requires_grad_(False), config


def deep_fusion():
    """The V17 module, imported on demand.

    It is imported inside the function, not at module scope, so a run that does not pass
    `--v17` never touches `methods/prlmc/deep_fusion.py` -- a broken or half-written file
    there cannot change what the existing arms do.
    """
    try:
        from methods.prlmc import deep_fusion as df
    except Exception as error:                       # noqa: BLE001 - reported below
        raise ValueError("this run asks for a V17 arm but methods/prlmc/deep_fusion.py "
                         f"cannot be imported: {error}") from error
    return df


def module_class_for_schema(schema, path):
    """Which residual class a checkpoint's recorded schema selects.

    V13/V13Plus write `cross_block_calibration/1`; V17 writes `deep_fusion/1` and carries
    the extra constructor keys in its own `config()`. Choosing the class from the schema
    (rather than from the CLI flag) is what lets one loader read both lines.
    """
    if schema == FULL_POOL_SCHEMA + "/checkpoint":
        return FullPoolResidual
    if schema == DEEP_SCHEMA + "/checkpoint":
        return deep_fusion().DeepFusionResidual
    raise ValueError(f"{path} is neither a {FULL_POOL_SCHEMA} nor a {DEEP_SCHEMA} "
                     f"checkpoint (schema {schema!r})")


def load_v11(path, hidden):
    """A V11 checkpoint: the module and the geometry the window has to be rebuilt with."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    module_class = module_class_for_schema(payload.get("schema"), path)
    geometry = dict(payload.get("geometry") or {})
    for field in ("cross", "core_per_block", "graft_per_block", "utility_layers",
                  "probe_ids"):
        if field not in geometry:
            raise ValueError(f"{path} records no {field}: the window cannot be rebuilt "
                             f"from it")
    if int(geometry["core_per_block"]) + int(geometry["graft_per_block"]) != SLOTS_PER_BLOCK:
        raise ValueError(f"{path}: the recorded quota does not fill a block")
    config = dict(payload["config"])
    if "use_cross" not in config:
        # This round's checkpoints were written before the field was recorded. The state
        # dict says it unambiguously: only a cross-block module carries `cross_proj`.
        config["use_cross"] = any(str(key).startswith("cross_proj")
                                  for key in payload["state_dict"])
    if "use_evicted" not in config:
        # Same inference for the second path: only a module with it carries `evict_proj`.
        config["use_evicted"] = any(str(key).startswith("evict_proj")
                                    for key in payload["state_dict"])
    module = module_class(**config)
    module.load_state_dict(payload["state_dict"], strict=True)
    return module.eval().requires_grad_(False), geometry, dict(payload["config"]), payload


def _as_module(result, path):
    """`load_deep` may return the module, `(module, config)` or `(module, config, geom)`.

    The spec fixes the call, not the tuple, so the return is unwrapped here instead of
    guessed at every call site: the module is always the first element.
    """
    module = result[0] if isinstance(result, (tuple, list)) else result
    if not hasattr(module, "forward") or not hasattr(module, "config"):
        raise ValueError(f"{path}: load_deep did not return a residual module "
                         f"({type(module).__name__})")
    return module


def apply_layer_norm_weights(module, payload, path):
    """Write the checkpoint's trained per-layer regulariser weights into the module. (D)

    The module's `layer_norm_weights` buffer is deliberately non-persistent (a persistent
    one would add a state_dict key and break strict loading of every earlier checkpoint),
    so the trainer copies the trained numbers into the checkpoint separately. Dropping
    them on the floor here would evaluate the arm with the neutral all-ones weights and
    then report a regulariser that was never trained; a length mismatch is therefore an
    error, not a footnote. They only enter the REPORT (`info["layer_norm_weights"]`,
    `info["regulariser"]`), so writing them changes no calibrated slot and no weight of
    the module itself.
    """
    geometry = dict(payload.get("geometry") or {})
    recorded = geometry.get("layer_norm_weights")
    recorded = dict(recorded) if isinstance(recorded, dict) else {}
    values = payload.get("layer_norm_weights")
    if values is None:
        return dict(values=None, enabled=False, source=None, persisted=recorded.get(
            "persisted"),
            note="the checkpoint carries no `layer_norm_weights`: the per-layer "
                 "regulariser was not trained on this run (--layer-norm-weights off), and "
                 "the module keeps its neutral all-ones buffer")
    values = [float(value) for value in values]
    buffer = dict(module.named_buffers()).get("layer_norm_weights")
    if buffer is None:
        # Only a module built before §2.3 of `cross_block_calibration` lacks it; the
        # fallback keeps the arm honest instead of skipping the weights.
        module.register_buffer("layer_norm_weights",
                               torch.ones(len(values), dtype=torch.float32),
                               persistent=False)
        buffer = module.layer_norm_weights
    if int(buffer.numel()) != len(values):
        raise ValueError(f"{path}: the checkpoint carries {len(values)} "
                         f"layer_norm_weights but the module's buffer holds "
                         f"{int(buffer.numel())}: writing them would either fail or "
                         f"silently resize the regulariser")
    tensor = torch.as_tensor(values, dtype=buffer.dtype)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{path}: layer_norm_weights is not finite ({values})")
    with torch.no_grad():
        buffer.copy_(tensor.to(buffer.device))
    if not torch.equal(buffer.detach().cpu().float(), torch.as_tensor(values).float()):
        raise ValueError(f"{path}: the layer_norm_weights buffer did not take the "
                         f"checkpoint's values")
    enabled = recorded.get("enabled")
    if enabled is None:
        # No geometry block: `enabled` can still be read off the numbers themselves,
        # because all-ones IS the neutral value the buffer starts at.
        enabled = bool(any(value != 1.0 for value in values))
    return dict(values=values, enabled=bool(enabled), source="checkpoint",
                persisted=recorded.get("persisted"),
                note="read back from the checkpoint into the module's non-persistent "
                     "buffer; the regulariser reports with these weights, the calibrated "
                     "slots are unchanged by them")


def load_deep_arm(path, hidden, rank=None, max_ratio=None):
    """`--v17`: a DeepFusionResidual rebuilt from its own `config()`.

    The module is built by `deep_fusion.load_deep` (the checkpoint's own loader, strict);
    the geometry is read from the same payload, because the window this arm calibrates
    has to be rebuilt by the builder the checkpoint was trained with.
    """
    df = deep_fusion()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != DEEP_SCHEMA + "/checkpoint":
        raise ValueError(f"{path} is not a {DEEP_SCHEMA} checkpoint "
                         f"(schema {payload.get('schema')!r})")
    module = _as_module(df.load_deep(path, hidden, rank=rank, max_ratio=max_ratio), path)
    config = dict(payload.get("config") or {})
    if int(config.get("hidden_size", -1)) != int(hidden):
        raise ValueError(f"{path} was built for hidden {config.get('hidden_size')}, "
                         f"the reader is {hidden}")
    geometry = dict(payload.get("geometry") or {})
    module = module.to(torch.float32).eval().requires_grad_(False)
    weights = apply_layer_norm_weights(module, payload, path)
    return dict(module=module, config=config, geometry=geometry, payload=payload,
                path=path, basis=None, layer_norm_weights=weights)


def module_needs_key_basis(module, config=None):
    """Does this module read `module.key_basis` in `forward`?"""
    for holder in (module, config or {}):
        value = (getattr(holder, "key_channel", None) if not isinstance(holder, dict)
                 else holder.get("key_channel"))
        if value is not None:
            return str(value) == "project"
    return False


def install_deep_basis(v17):
    """`deep_fusion.install_key_basis`: (module, basis) or basis-only, whichever it takes."""
    df = deep_fusion()
    function = getattr(df, "install_key_basis", None)
    if function is None:
        raise ValueError("methods/prlmc/deep_fusion.py exports no install_key_basis: the "
                         "key-channel='project' module cannot be run without a basis")
    accepted = [parameter for parameter in inspect.signature(function).parameters.values()
                if parameter.kind in (parameter.POSITIONAL_ONLY,
                                      parameter.POSITIONAL_OR_KEYWORD,
                                      parameter.VAR_POSITIONAL)]
    if len(accepted) >= 2:
        function(v17["module"], v17["basis"])
    else:
        returned = function(v17["basis"])
        if returned is not None:
            v17["basis"] = returned
        v17["module"].key_basis = v17["basis"]
    if getattr(v17["module"], "key_basis", None) is None:
        raise ValueError("install_key_basis did not leave a basis on the module: nothing "
                         "would be key-anchored, so the arm would not be V17")
    return v17


def _first_present(mapping, names):
    for name in names:
        if mapping and name in mapping and mapping[name] is not None:
            return name, mapping[name]
    return None, None


def graft_attention_params(config, geometry):
    """The damping constants the checkpoint recorded, and where they were found.

    The spec puts `graft_attention_calibration(tau=..., bias=..., norm_quantile=...)` in
    `deep_fusion` but does not fix where the trainer records the three numbers, so this
    reads `config` first, then `geometry` (top level and the `graft`/`attention` blocks).
    Nothing is invented: a checkpoint that records nothing yields tau=1, bias=0,
    quantile=0, and the arm then reports that it damped nothing instead of pretending.
    """
    blocks = [("config", config), ("geometry", geometry)]
    for name in ("graft", "attention", "graft_attention"):
        if geometry and isinstance(geometry.get(name), dict):
            blocks.append((f"geometry.{name}", dict(geometry[name])))
    found, values, where = [], {}, {}
    for tau_names, bias_names, quantile_names in (
            (("graft_tau", "tau", "attn_tau"), ("graft_bias", "bias", "attn_bias"),
             ("graft_norm_quantile", "norm_quantile", "attn_norm_quantile")),):
        for label, block in blocks:
            name, value = _first_present(block, tau_names)
            if name is not None:
                values["tau"], where["tau"] = float(value), f"{label}.{name}"
            name, value = _first_present(block, bias_names)
            if name is not None:
                values["bias"], where["bias"] = float(value), f"{label}.{name}"
            name, value = _first_present(block, quantile_names)
            if name is not None:
                values["norm_quantile"], where["norm_quantile"] = float(value), \
                    f"{label}.{name}"
    values.setdefault("tau", 1.0)
    values.setdefault("bias", 0.0)
    values.setdefault("norm_quantile", 0.0)
    found = dict(values, recorded=where,
                 non_default=bool(abs(values["tau"] - 1.0) > 0.0
                                  or abs(values["bias"]) > 0.0
                                  or values["norm_quantile"] > 0.0))
    return found


def runtime_knob(module, attribute, value):
    """A deep copy of `module` with one inference-time attribute replaced.

    The arms exist to move ONE runtime knob on an unchanged checkpoint; the copy is what
    keeps `V17` itself and every other arm out of it. Nothing on disk and no parameter of
    the original module is touched.
    """
    tuned = copy.deepcopy(module)
    current = getattr(tuned, attribute, None)
    if isinstance(current, torch.Tensor):
        # A registered buffer (e.g. per-slot-type ceilings) keeps being a tensor; a bare
        # float would either break `forward` or silently lose the per-type split.
        setattr(tuned, attribute, torch.as_tensor(value, dtype=current.dtype,
                                                  device=current.device).expand_as(
                                                      current).clone())
    else:
        setattr(tuned, attribute, value)
    return tuned


def joint_for(geometry, scorer, probe_cache, hidden):
    """Rebuild the exact window builder a checkpoint was trained with."""
    layers = tuple(int(x) for x in geometry["utility_layers"])
    bank_key = (layers, tuple(tuple(ids) for ids in geometry["probe_ids"]))
    bank = probe_cache.get(bank_key)
    if bank is None:
        bank = [scorer.probe_bank(ids) for ids in geometry["probe_ids"]]
        probe_cache[bank_key] = bank
    return JointConsolidation(
        scorer, bank, core_per_block=int(geometry["core_per_block"]),
        graft_per_block=int(geometry["graft_per_block"]), policy="evict",
        mmr_penalty=float(geometry.get("mmr_penalty", 0.5)),
        novelty_weight=float(geometry.get("novelty_weight", 0.5)))


def joint_density_for(geometry, scorer, probe_cache, hidden, absorb_cos=0.0,
                      min_core=140, max_core=256):
    """The SAME builder, but the 1,620 core slots are allocated by block redundancy.

    `quota_mode="density"` and `absorb_cos` are the second owner's additions to
    `JointConsolidation`. They are looked up on the live signature rather than passed
    blindly: a run against a tree where they do not exist yet must say so, not quietly
    rebuild the fixed-quota window and label it "density".
    """
    joint = joint_for(geometry, scorer, probe_cache, hidden)
    accepted = set(inspect.signature(JointConsolidation.__init__).parameters)
    required = ["quota_mode", "min_core", "max_core"]
    if float(absorb_cos) > 0.0:
        required.append("absorb_cos")
    missing = [name for name in required if name not in accepted]
    if missing:
        raise ValueError("--v17-window-density needs JointConsolidation to accept "
                         f"{missing}; methods/prlmc/evict_window.py does not have them "
                         f"yet, so the fixed-quota window cannot be re-quotaed here")
    optional = dict(quota_mode="density", min_core=int(min_core), max_core=int(max_core),
                    absorb_cos=float(absorb_cos))
    if "density_sample" in accepted:
        optional["density_sample"] = 64
    if float(absorb_cos) > 0.0 and "absorb_alpha_max" in accepted:
        optional["absorb_alpha_max"] = 0.5
    density_joint = JointConsolidation(
        scorer, joint.probes, core_per_block=int(geometry["core_per_block"]),
        graft_per_block=int(geometry["graft_per_block"]), policy="evict",
        mmr_penalty=float(geometry.get("mmr_penalty", 0.5)),
        novelty_weight=float(geometry.get("novelty_weight", 0.5)), **optional)
    if str(getattr(density_joint, "quota_mode", "fixed")) != "density":
        raise ValueError("the operator did not take quota_mode='density'")
    return density_joint


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def set_quota(merge, core_per_block, graft_per_block, block=SLOTS_PER_BLOCK):
    if core_per_block + graft_per_block != block:
        raise ValueError("the baseline quota does not fill a block")
    merge.core_slots_per_block = int(core_per_block)
    merge.graft_slots_per_block = int(graft_per_block)
    if (int(merge.core_slots_per_block) != int(core_per_block)
            or int(merge.graft_slots_per_block) != int(graft_per_block)):
        raise ValueError("the operator did not take the baseline quota")


def check_v11(window, meta, telemetry, source, name, allow_absorbed=False):
    """Capacity, split, order, provenance, and what each half is made of.

    `allow_absorbed` exists for one caller: a density window built with `absorb_cos > 0`
    writes evicted slots INTO a retained neighbour, so those slots are deliberately not
    verbatim copies any more. The relaxation is exact rather than blanket -- every slot
    that differs must be covered by an absorption the builder recorded -- and the default
    is off, so every existing arm keeps the strict check.
    """
    capacity = window.shape[1]
    if capacity != TARGET_SLOTS:
        raise ValueError(f"{name}: {capacity} slots, expected {TARGET_SLOTS}")
    core = int((meta["slot_type"] == 0).sum())
    graft = int((meta["slot_type"] == 1).sum())
    expected_core = telemetry["core_total"]
    expected_graft = telemetry["graft_total"]
    if core != expected_core or graft != expected_graft:
        raise ValueError(f"{name}: {core}/{graft} slots, telemetry says "
                         f"{expected_core}/{expected_graft}")
    if core + graft != capacity:
        raise ValueError(f"{name}: the split does not fill the budget")
    positions = meta["source_position"]
    if int(positions.numel()) != capacity:
        raise ValueError(f"{name}: one source position per slot")
    if not torch.equal(positions, torch.sort(positions).values):
        raise ValueError(f"{name}: the output is not in global chronological order")
    dropped = set(int(block) for block in telemetry["dropped_blocks"])
    carriers = set(int(block) for block in telemetry["carrier_blocks"])
    graft_blocks = set(int(x) for x in meta["source_block"][meta["slot_type"] == 1].tolist())
    if graft_blocks - dropped:
        raise ValueError(f"{name}: a graft slot came from a carrier block")
    core_blocks = set(int(x) for x in meta["source_block"][meta["slot_type"] == 0].tolist())
    if core_blocks - carriers:
        raise ValueError(f"{name}: a core slot came from a dropped block")
    if not torch.isfinite(window).all():
        raise ValueError(f"{name}: the window is not finite")

    index = positions.to(window.device)
    copied = source.index_select(1, index)
    unfolded = meta["cluster_size"] == 1
    absorbed_slots = 0
    if not torch.equal(window[:, unfolded], copied[:, unfolded]):
        if not allow_absorbed:
            raise ValueError(f"{name}: an unfolded slot is not a verbatim copy of its "
                             f"source")
        differs = (window[:, unfolded] != copied[:, unfolded]).any(dim=0)
        absorbed_slots = int(differs.sum())
        merge = telemetry.get("merge") or {}
        recorded = int(telemetry.get("absorbed_pairs")
                       or merge.get("absorbed_pairs") or 0)
        if absorbed_slots > recorded:
            raise ValueError(f"{name}: {absorbed_slots} unfolded slots are not verbatim "
                             f"copies of their source, but only {recorded} absorptions "
                             f"were recorded: the difference is unaccounted drift")
    folded = ~unfolded
    if bool(folded.any()):
        base = copied[:, folded]
        moved = window[:, folded]
        base_rms = base.float().square().mean(dim=-1).sqrt()
        moved_rms = moved.float().square().mean(dim=-1).sqrt()
        drift = float(((moved_rms - base_rms).abs() / base_rms.clamp_min(1e-6)).max())
        if drift > 1e-2:
            raise ValueError(f"{name}: a folded slot lost its representative's RMS "
                             f"(max relative drift {drift:.2e})")
    per_block = telemetry["graft_per_dropped_block"]
    if sum(per_block) != expected_graft:
        raise ValueError(f"{name}: the per-block graft counts do not add up")
    merge = telemetry["merge"]
    return dict(
        core_total=core, graft_total=graft, capacity=capacity,
        folded_slots=int(folded.sum()), merge_policy=telemetry["policy"],
        merged_pairs=int(merge["merged_pairs"]), avoided_evictions=int(merge["evicted"]),
        blocks_with_no_merge=int(merge["blocks_with_no_merge"]),
        blocks_where_quota_came_from_eviction=int(
            merge["blocks_where_quota_came_from_eviction"]),
        mean_cluster_size=float(telemetry["cluster_size_core_mean"]),
        empty_dropped_blocks=int(telemetry["empty_dropped_blocks"]),
        min_per_dropped_block=int(min(per_block)), max_per_dropped_block=int(max(per_block)),
        novelty_mean=float(telemetry["novelty_mean"]),
        absorbed_slots=absorbed_slots, verbose=merge)


def check_calibration(calibrated, window, meta, info, name):
    """Both halves must move, and neither may move out of the native norm range."""
    core = meta["slot_type"] == 0
    graft = meta["slot_type"] == 1
    if torch.equal(calibrated[:, core], window[:, core]):
        raise ValueError(f"{name}: the calibration did not touch a single core slot, so "
                         f"this arm does not test the full-pool module")
    if torch.equal(calibrated[:, graft], window[:, graft]):
        raise ValueError(f"{name}: the calibration did not touch a single graft slot")
    ratio = (calibrated.float().square().mean(dim=-1).sqrt()
             / window.float().square().mean(dim=-1).sqrt().clamp_min(1e-6))
    if float((ratio - 1.0).abs().max()) > 2e-2:
        raise ValueError(f"{name}: the RMS restore did not hold "
                         f"(max drift {float((ratio - 1.0).abs().max()):.2e})")
    return dict(core_relative_delta=info["core_relative_delta"],
                graft_relative_delta=info["graft_relative_delta"],
                relative_delta=info["relative_delta"],
                folded_slots=info["folded_slots"], gate=info["gate"],
                # Without these three the report cannot tell a mechanism that was
                # trained and did nothing from one that was never applied.
                use_cross=info.get("use_cross"),
                neighbor_mode=info.get("neighbor_mode"),
                k_neighbors=info.get("k_neighbors"),
                cross_proj_norm=info.get("cross_proj_norm"),
                cross_context=info.get("cross_context"),
                evicted_context=info.get("evicted_context"),
                relation=info.get("relation"),
                # V16's band diagnostics. Without these `_band_gate` finds no rows and
                # reports {n: 0} for every band -- an empty table that looks like "no
                # gate" rather than "the number was never carried out of the module".
                gate_bands=info.get("gate_bands"),
                band_proj_norm=info.get("band_proj_norm"),
                layer_proj_norm=info.get("layer_proj_norm"),
                cross_injection_rms=info.get("cross_injection_rms"),
                shallow_cross_leak=info.get("shallow_cross_leak"), check=True)


# ---------------------------------------------------------------------------
# V17: the deep-fusion arms
# ---------------------------------------------------------------------------
def _scalars(value):
    """A runtime knob as JSON-safe numbers: a float, or the list a small tensor holds."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        flat = [round(float(x), 6) for x in value.detach().float().reshape(-1).tolist()]
        return flat[0] if len(flat) == 1 else flat
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return str(value)


def neutralise_graft_attention(module):
    """A copy of `module` whose own damping constants are set to their neutral values.

    Only plain numeric attributes are touched, never a Parameter or a buffer: the arms
    move runtime knobs, they never move weights. Two arms use this, in opposite
    directions: `V17Attn` applies the checkpoint's tau/bias ONCE by hand and needs the
    copy's own constants neutral so the damping cannot happen twice, and `V17NoAttn`
    needs them neutral so that it happens NOT AT ALL (the checkpoint damps inside
    `deep_fusion.calibrate`, the neutral copy does not).
    """
    tuned = copy.deepcopy(module)
    neutralised = []
    for name in list(vars(tuned)):
        value = getattr(tuned, name)
        if isinstance(value, (torch.Tensor, torch.nn.Module)):
            continue
        if not isinstance(value, (int, float)):
            continue
        low = name.lower()
        # `graft_tau` / `graft_bias` / `graft_norm_quantile` are the three constants
        # `deep_fusion.calibrate` reads back off the module; a checkpoint that records
        # them turns damping on inside its own calibrate.
        if "tau" in low:
            setattr(tuned, name, type(value)(1.0))
            neutralised.append(name)
        elif "bias" in low:
            setattr(tuned, name, type(value)(0.0))
            neutralised.append(name)
        elif "quantile" in low:
            setattr(tuned, name, type(value)(0.0))
            neutralised.append(name)
    return tuned, neutralised


def graft_attention_self_damping(module, params=None):
    """Will `deep_fusion.calibrate` damp this window by itself, and with which numbers?

    `df.calibrate` reads `graft_tau` / `graft_bias` / `graft_norm_quantile` OFF THE
    MODULE and damps when they are non-neutral, so the module's own attributes are the
    authority here, exactly as they are there. The checkpoint's recorded numbers
    (`graft_attention_params`: config first, then geometry) are the fallback for a module
    that does not carry the attributes at all. Both are reported, because "who damped
    the window" is the one thing these arms must not blur.
    """
    recorded = dict(params or {})
    constants, source = {}, "module"
    for name, key, default in (("graft_tau", "tau", 1.0), ("graft_bias", "bias", 0.0),
                               ("graft_norm_quantile", "norm_quantile", 0.0)):
        value = getattr(module, name, None)
        if value is None:
            source = "checkpoint"
            value = recorded.get(key, default)
        constants[name] = float(default if value is None else value)
    non_neutral = bool(constants["graft_tau"] != 1.0 or constants["graft_bias"] != 0.0
                       or constants["graft_norm_quantile"] > 0.0)
    return dict(source=source, constants=constants, non_neutral=non_neutral,
                recorded=dict(tau=recorded.get("tau"), bias=recorded.get("bias"),
                              norm_quantile=recorded.get("norm_quantile")),
                note="`non_neutral` mirrors the test inside deep_fusion.calibrate: "
                     "tau != 1 or bias != 0 or norm_quantile > 0 damps the window "
                     "before the neighbour context is measured")


def decide_attention_arm(args, v17):
    """Does `--v17-attention` still name an arm that is not `V17` under another name? (B)

    It does not, when the checkpoint carries the damping itself: `deep_fusion.calibrate`
    damps the window before it measures the context, so an explicitly damped arm would
    re-calibrate a window that differs from `V17`'s by exactly one extra damping pass
    applied by this driver instead of by the module. Reporting that as "the worth of
    graft damping" would be wrong twice over (it is not what training did, and it damps
    twice), so the arm is skipped with a WARNING and the run points at `--v17-no-attn`.
    """
    decision = dict(requested=bool(getattr(args, "v17_attention", False)), arm=V17_ATTN,
                    built=False, self_damping=None,
                    note="not requested")
    if not decision["requested"]:
        return decision
    state = graft_attention_self_damping(v17["module"], v17.get("attention"))
    decision["self_damping"] = state
    if state["non_neutral"]:
        args.v17_attention = False
        constants = state["constants"]
        decision["note"] = (
            "SKIPPED: the checkpoint damps the graft keys inside deep_fusion.calibrate, "
            "so an arm that damps the window again here would be V17 with the damping "
            "applied twice, not a treatment. The control for mechanism four is "
            "--v17-no-attn (V17 - V17NoAttn)")
        print("WARNING: --v17-attention SKIPPED: this checkpoint already carries "
              f"non-neutral damping constants (tau={constants['graft_tau']}, "
              f"bias={constants['graft_bias']}, "
              f"norm_quantile={constants['graft_norm_quantile']}, read from "
              f"{state['source']}), so deep_fusion.calibrate damps the window itself and "
              "this arm would be V17 under another name. The arm that turns mechanism "
              "four OFF is --v17-no-attn.", file=sys.stderr)
        return decision
    decision.update(built=True, note=(
        "the checkpoint's own constants are neutral, so the arm damps the window once, "
        "here, with the tau/bias it recorded"))
    return decision


def band_honesty(v17):
    """How many of the three band gates can have learned anything. (E, second warning)"""
    if v17 is None:
        return dict(band_mode=None, band_layout=None, band0_observable=None,
                    gate_parameters_live=None,
                    note="no --v17 arm in this run, so no band gate was read")
    df = deep_fusion()
    module = v17["module"]
    layers = getattr(module, "layer_ids", None) or v17["config"].get("layer_ids")
    layout = df.band_layout(tuple(int(layer) for layer in (layers or (12, 16, 20, 24, 28))))
    band_mode = str(getattr(module, "band_mode", "off"))
    record = dict(
        band_mode=band_mode,
        layers=[int(layer) for layer in layout["layers"]],
        bands=[int(band) for band in layout["bands"]],
        band_names=[str(name) for name in layout["band_names"]],
        zeroed_layers=[int(layer) for layer in layout["zeroed_layers"]],
        bands_present=[int(band) for band in layout["bands_present"]],
        bands_absent=[int(band) for band in layout["bands_absent"]],
        band0_observable=bool(layout["band0_observable"]),
        gate_parameters_live=[int(band) for band in layout["gate_parameters_live"]],
        note="`gate_parameters_live` are the bands whose gate can receive a gradient on "
             "this window's layer table; a band with no layer is masked to zero by "
             "`band_open`, so its 0.0 is the initial value and not a learned zero")
    if band_mode == "three" and not record["band0_observable"]:
        print("WARNING: band_mode='three' but no layer of this window "
              f"({record['layers']}) falls in band 0, so band 0's gate never sees a "
              f"gradient: only {record['gate_parameters_live']} gates are live and the "
              "'three-band gate' is a two-band one on this table. Read "
              "`gate_parameters_live` before reading the three gates in the checkpoint.",
              file=sys.stderr)
    return record


def honesty_report(samples, v11_evict=None, v17=None):
    """The two warnings of (E), each with the number that triggered it.

    Both are recorded rather than raised: a floor that filters nothing and a band that
    never trains are findings about the arm, and a run that dies would hide them.
    """
    context = None
    for sample in samples:
        report = ((sample.get("evict") or {}).get("calibration") or {}).get(
            "cross_context")
        if report:
            context = report
            break
    quantiles = (context or {}).get("neighbour_similarity_quantiles") or {}
    p10 = quantiles.get("p10")
    rows = []
    for label, holder in (("--v11-evict", v11_evict), ("--v17", v17)):
        if not holder:
            continue
        value = (holder.get("config") or {}).get("sim_floor")
        if value is None:
            continue
        floor = float(value)
        row = dict(checkpoint=label, sim_floor=floor,
                   neighbour_similarity_p10=(None if p10 is None else float(p10)),
                   filters_nothing=None)
        if floor > 0.0 and p10 is not None:
            row["filters_nothing"] = bool(float(p10) > floor)
            if row["filters_nothing"]:
                print(f"WARNING: {label} carries sim_floor={floor:.6g} but the first "
                      f"cross-block context of this run has a p10 neighbour similarity "
                      f"of {float(p10):.6g}: the floor sits BELOW almost every candidate, "
                      "so it filters almost nothing and this arm cannot be used as "
                      "evidence that a lower-bound filter does or does not work.",
                      file=sys.stderr)
        elif floor > 0.0:
            print(f"WARNING: {label} carries sim_floor={floor:.6g} but no neighbour "
                  "similarity quantiles were reported, so whether the floor filters "
                  "anything is MISSING rather than false.", file=sys.stderr)
        rows.append(row)
    measured = [row["filters_nothing"] for row in rows
                if row["filters_nothing"] is not None]
    filters_nothing = (True if any(measured) else False) if measured else None
    return dict(
        sim_floor=dict(
            filters_nothing=filters_nothing, rows=rows, arms=rows,
            neighbour_similarity_quantiles=(dict(quantiles) if quantiles else None),
            note="`filters_nothing` is None when no checkpoint of this run records a "
                 "positive sim_floor, or when the run produced no similarity quantiles: "
                 "None means 'not measured', never 'the floor works'"),
        band=band_honesty(v17))


def calibrate_deep(v17, window, meta, name, source=None, extra=None):
    """`deep_fusion.calibrate` on one window, checked and annotated like every arm.

    `v17` is the arm's own dict, so a derived arm can pass a copy of the module with one
    runtime knob turned without the checkpoint's module (or `V17`) ever being mutated.
    """
    df = deep_fusion()
    module = v17["module"]
    basis = v17.get("basis")
    calibrated, info = df.calibrate(window, meta, module, source=source, basis=basis)
    calibration = check_calibration(calibrated, window, meta, info, name)
    calibration["deep"] = dict(
        key_channel=(None if getattr(module, "key_channel", None) is None
                     else str(module.key_channel)),
        value_ratio=_scalars(getattr(module, "value_ratio", None)),
        band_mode=(None if getattr(module, "band_mode", None) is None
                   else str(module.band_mode)),
        key_ratio=_scalars(getattr(module, "key_ratio", None)),
        # The damp report the module chose to run itself, if any: `V17Attn` has to be
        # distinguishable from a checkpoint that already damps in its own calibrate.
        damp=info.get("v17"), **(extra or {}))
    return dict(window=window, calibrated=calibrated, check=None,
                calibration=calibration)


def _per_block_quota(meta, telemetry, carriers):
    """Core slots per carrier block, counted from the window itself.

    The count is taken from `meta` rather than read out of the telemetry so a builder that
    reports the quota under another key cannot make this look like a density window when
    the slots were allocated the fixed way.
    """
    counts = []
    core = meta["slot_type"] == 0
    blocks = meta["source_block"][core]
    for block in carriers:
        counts.append(int((blocks == int(block)).sum()))
    recorded = (telemetry or {}).get("per_block_quota")
    if recorded is not None and len(list(recorded)) == len(counts):
        recorded = [int(x) for x in recorded]
        if recorded != counts:
            raise ValueError("the telemetry's per-block quota does not match the window "
                             f"({recorded} vs {counts}): the report would be lying")
    return counts


def _density_telemetry(telemetry):
    """The window-builder telemetry a density arm has to carry into the report.

    Only the density-relevant keys are kept (the full telemetry is large and repeats the
    fixed keys every compressed arm already records), plus `quota_mode` and the per-block
    quota, which are the two numbers the comparison is about.
    """
    telemetry = telemetry or {}
    keep = {}
    for key in ("quota_mode", "per_block_quota", "density", "core_total", "graft_total",
                "total", "graft_share", "absorbed_pairs", "absorbed_mean_alpha",
                "absorbed_mean_cos", "carrier_blocks", "dropped_blocks",
                "graft_per_dropped_block", "min_core", "max_core", "density_sample"):
        if key in telemetry:
            keep[key] = _scalars(telemetry[key]) if not isinstance(
                telemetry[key], (list, tuple, dict)) else telemetry[key]
    merge = telemetry.get("merge")
    if isinstance(merge, dict):
        keep["merge"] = {key: merge.get(key) for key in
                         ("merged_pairs", "evicted", "blocks_with_no_merge",
                          "mean_cluster_size", "policy")}
    return keep


def check_anchor(output, parts, anchor_window, name):
    offsets = parts["anchor_offsets"]
    keep = torch.ones(output.shape[1], dtype=torch.bool)
    keep[offsets] = False
    if not torch.equal(output[:, keep], anchor_window[:, keep]):
        raise ValueError(f"{name}: the anchor module changed a core slot")
    if torch.equal(output.index_select(1, offsets), anchor_window.index_select(1, offsets)):
        raise ValueError(f"{name}: the anchor module changed nothing")
    return dict(anchor_slots=int(offsets.numel()), check=True)


# ---------------------------------------------------------------------------
# one sample
# ---------------------------------------------------------------------------
def build_v11(joint, module, source, picked, block, name):
    """One window plus its calibrated twin, both checked."""
    window, meta, telemetry = joint.build(source, picked, block)
    window_check = check_v11(window, meta, telemetry, source, name)
    calibrated, info = calibrate(window, meta, module, source=source)
    calibration_check = check_calibration(calibrated, window, meta, info, name)
    return dict(window=window, calibrated=calibrated, meta=meta,
                telemetry=telemetry, check=window_check,
                calibration=calibration_check)


def _chunks(prepared, window_blocks):
    chunks = [prepared["write_ids"][index:index + BLOCK_TOKENS]
              for index in range(0, len(prepared["write_ids"]), BLOCK_TOKENS)]
    return chunks[:window_blocks]


def run_sample(model, tokenizer, manager, merge, anchor_module, v11, v11_evict,
               prepared, donor_graft, donor_graft_evict, donor_id, args, v14=None,
               v15=None, v16=None, v13_plus=None, v17=None, density=None):
    """One write; the baseline window and both V11 windows; then every arm's answer."""
    chunks = _chunks(prepared, args.window_blocks)
    answers = prepared["answers"]
    fingerprints, windows = {}, {}
    next_donor_evict = None
    with torch.no_grad():
        es.write_blocks(manager, merge, chunks[:args.window_blocks])
        state = gr.snapshot(model, manager)
        source, ages, picked = gr.select_carriers(manager, args.window_blocks,
                                                  args.target_blocks)
        frozen, pinned = gr.window_locks(manager, args.window_blocks)
        # Arm A: the baseline quota, native grafting, then the frozen AnchorResidual.
        set_quota(merge, BASELINE_QUOTA["core_per_block"],
                  BASELINE_QUOTA["graft_per_block"])
        output, telemetry = gr.compress_graft(manager, merge, source, picked,
                                              args.window_blocks, args.target_blocks,
                                              frozen, pinned)
        parts = split_source_anchors(telemetry, manager.block, args.window_blocks,
                                     device=source.device)
        anchor_window, anchor_info = compact_window_anchor(output, parts, anchor_module)
        windows[GRAFT] = output
        windows[ANCHOR_ONLY] = anchor_window
        baseline = dict(anchor=check_anchor(output, parts, anchor_window, ANCHOR_ONLY),
                        core_slots=int(output.shape[1]) - int(parts["anchor_offsets"].numel()),
                        graft_slots=int(parts["anchor_offsets"].numel()),
                        relative_delta=float(anchor_info["relative_delta"]))
        # Arm B and arm D: two windows, built by two recorded builders.
        built = dict(B=build_v11(v11["joint"], v11["module"], source, picked,
                                 manager.block, V11))
        if v11_evict is not None:
            built["D"] = build_v11(v11_evict["joint"], v11_evict["module"], source,
                                   picked, manager.block, V11_EVICT)
        # Arm C: arm B's window with the graft slots taken from the donor document.
        graft_offsets = torch.nonzero(built["B"]["meta"]["slot_type"] == 1,
                                      as_tuple=False).flatten()
        # `shape[1]` is the slot count; `shape[-1]` is the hidden width. Comparing the
        # hidden width with 684 was never true, so arm C was silently skipped and the
        # run answered every question except the one it exists for.
        if donor_graft is not None and int(donor_graft.shape[1]) == int(
                graft_offsets.numel()):
            donor_window = built["B"]["window"].clone()
            donor_window[:, graft_offsets] = donor_graft.to(built["B"]["window"].dtype)
            if torch.equal(donor_window, built["B"]["window"]):
                raise ValueError("the donor window equals the true window: the graft "
                                 "swap did not reach the memory")
            donor_calibrated, donor_info = calibrate(donor_window, built["B"]["meta"],
                                                     v11["module"])
            # Same key set as arm B so the record below can read it uniformly. Arm C
            # shares B's geometry and B's window check; only the graft vectors and the
            # calibration residual differ, and both are recorded.
            built["C"] = dict(window=donor_window,
                              calibrated=donor_calibrated,
                              check=built["B"]["check"],
                              calibration=check_calibration(
                                  donor_calibrated, donor_window, built["B"]["meta"],
                                  donor_info, V11_DONOR))
        # Arm E: the eviction line's own counterfactual. Same builder as arm D, same
        # calibration module, same 684 slots -- only whose dropped blocks they came
        # from changes. Arm C cannot answer this for D: the fold and the eviction
        # select different slots, so a donor taken from the folded window would
        # confound content with slot identity.
        next_donor_evict = None
        if "D" in built:
            evict_offsets = torch.nonzero(built["D"]["meta"]["slot_type"] == 1,
                                          as_tuple=False).flatten()
            next_donor_evict = built["D"]["window"].index_select(
                1, evict_offsets).detach().clone()
            if donor_graft_evict is not None and int(donor_graft_evict.shape[1]) == int(
                    evict_offsets.numel()):
                donor_window_evict = built["D"]["window"].clone()
                donor_window_evict[:, evict_offsets] = donor_graft_evict.to(
                    built["D"]["window"].dtype)
                if torch.equal(donor_window_evict, built["D"]["window"]):
                    raise ValueError("the eviction donor window equals the true one: the "
                                     "graft swap did not reach the memory")
                evict_donor_calibrated, evict_donor_info = calibrate(
                    donor_window_evict, built["D"]["meta"], v11_evict["module"],
                    source=source)
                built["E"] = dict(
                    window=donor_window_evict, calibrated=evict_donor_calibrated,
                    check=built["D"]["check"],
                    calibration=check_calibration(
                        evict_donor_calibrated, donor_window_evict, built["D"]["meta"],
                        evict_donor_info, V11_EVICT_DONOR))
        windows[V11] = built["B"]["calibrated"]
        if "C" in built:
            windows[V11_DONOR] = built["C"]["calibrated"]
        if "D" in built:
            windows[V11_EVICT] = built["D"]["calibrated"]
        if "E" in built:
            windows[V11_EVICT_DONOR] = built["E"]["calibrated"]
        # Arms for the V13 successor: the SAME window as D, calibrated by the module
        # that carries span-value / diffusion / CSLS in its own config.
        if v13_plus is not None:
            if "D" not in built:
                raise ValueError("--v13-plus needs the cross-block window it shares")
            plus_window, plus_info = calibrate(built["D"]["window"], built["D"]["meta"],
                                               v13_plus["module"], source=source)
            built["L"] = dict(window=built["D"]["window"], calibrated=plus_window,
                              check=built["D"]["check"],
                              calibration=check_calibration(
                                  plus_window, built["D"]["window"], built["D"]["meta"],
                                  plus_info, V13_PLUS))
            windows[V13_PLUS] = built["L"]["calibrated"]
            if "E" in built:
                plus_donor, plus_donor_info = calibrate(built["E"]["window"],
                                                        built["D"]["meta"],
                                                        v13_plus["module"],
                                                        source=source)
                built["M"] = dict(window=built["E"]["window"], calibrated=plus_donor,
                                  check=built["D"]["check"],
                                  calibration=check_calibration(
                                      plus_donor, built["E"]["window"],
                                      built["D"]["meta"], plus_donor_info,
                                      V13_PLUS_DONOR))
                windows[V13_PLUS_DONOR] = built["M"]["calibrated"]
        # Arms H and I: V15 on the SAME windows as D and E. Same slots, same graft
        # content, same budget; the relation rule and the module are what differ.
        if v15 is not None:
            if "D" not in built:
                raise ValueError("--v15 needs the cross-block window it has to share")
            relation_calibrated, relation_info = hr.calibrate(built["D"]["window"],
                                                              built["D"]["meta"],
                                                              v15["module"])
            built["H"] = dict(window=built["D"]["window"],
                              calibrated=relation_calibrated,
                              check=built["D"]["check"],
                              calibration=check_calibration(
                                  relation_calibrated, built["D"]["window"],
                                  built["D"]["meta"], relation_info, V15))
            windows[V15] = built["H"]["calibrated"]
            if "E" in built:
                relation_donor, relation_donor_info = hr.calibrate(
                    built["E"]["window"], built["D"]["meta"], v15["module"])
                built["I"] = dict(window=built["E"]["window"],
                                  calibrated=relation_donor,
                                  check=built["D"]["check"],
                                  calibration=check_calibration(
                                      relation_donor, built["E"]["window"],
                                      built["D"]["meta"], relation_donor_info,
                                      V15_DONOR))
                windows[V15_DONOR] = built["I"]["calibrated"]
        # Arms J and K: V16 on the SAME window as V13 and V15. Band routing and the
        # margin gate are the only things that differ.
        if v16 is not None:
            if "D" not in built:
                raise ValueError("--v16 needs the cross-block window it has to share")
            band_calibrated, band_info = br.calibrate(built["D"]["window"],
                                                      built["D"]["meta"],
                                                      v16["module"])
            built["J"] = dict(window=built["D"]["window"], calibrated=band_calibrated,
                              check=built["D"]["check"],
                              calibration=check_calibration(
                                  band_calibrated, built["D"]["window"],
                                  built["D"]["meta"], band_info, V16))
            windows[V16] = built["J"]["calibrated"]
            if "E" in built:
                band_donor, band_donor_info = br.calibrate(built["E"]["window"],
                                                           built["D"]["meta"],
                                                           v16["module"])
                built["K"] = dict(window=built["E"]["window"], calibrated=band_donor,
                                  check=built["D"]["check"],
                                  calibration=check_calibration(
                                      band_donor, built["E"]["window"],
                                      built["D"]["meta"], band_donor_info, V16_DONOR))
                windows[V16_DONOR] = built["K"]["calibrated"]
        # Inference-time graft-cap arms: V13's weights and window, only the bound the
        # graft residual may spend changes. A uniform sweep first; then the
        # relevance-proportional cap, which gives the graft slots that really bridge to
        # the retained blocks the full 0.50 and squeezes the quota-filler slots to 0.25.
        cap_arms = {}
        if getattr(args, "graft_scales", None) or getattr(args, "relevance_cap", False):
            if "D" not in built or v11_evict is None:
                raise ValueError("the graft-cap arms need the V13 window and module")
            relevance = None
            if getattr(args, "relevance_cap", False):
                relevance = slot_relevance(built["D"]["window"], built["D"]["meta"],
                                           k=v11_evict["module"].k_neighbors)
            for scale in getattr(args, "graft_scales", []):
                arm = cap_arm(scale)
                capped = copy.deepcopy(v11_evict["module"])
                capped.set_graft_cap(float(scale))
                capped_window, capped_info = calibrate(built["D"]["window"],
                                                       built["D"]["meta"], capped)
                built[arm] = dict(window=built["D"]["window"], calibrated=capped_window,
                                  check=built["D"]["check"],
                                  calibration=check_calibration(
                                      capped_window, built["D"]["window"],
                                      built["D"]["meta"], capped_info, arm))
                cap_arms[arm] = dict(kind="uniform", graft_cap=float(scale))
                windows[arm] = built[arm]["calibrated"]
                del capped, capped_window
            if relevance is not None:
                arm = "V13RelevanceCap"
                capped = copy.deepcopy(v11_evict["module"])
                caps = relevance_graft_caps(relevance, built["D"]["meta"]["slot_type"])
                capped.set_graft_cap(caps)
                capped_window, capped_info = calibrate(built["D"]["window"],
                                                       built["D"]["meta"], capped)
                built[arm] = dict(window=built["D"]["window"], calibrated=capped_window,
                                  check=built["D"]["check"],
                                  calibration=check_calibration(
                                      capped_window, built["D"]["window"],
                                      built["D"]["meta"], capped_info, arm))
                graft = (built["D"]["meta"]["slot_type"] > 0.5)
                cap_arms[arm] = dict(kind="relevance",
                                     graft_cap_min=float(caps[graft].min()),
                                     graft_cap_max=float(caps[graft].max()),
                                     graft_cap_mean=float(caps[graft].mean()),
                                     relevance_mean=float(relevance[graft].mean()),
                                     relevance_std=float(relevance[graft].std()))
                windows[arm] = built[arm]["calibrated"]
                del capped, capped_window, caps
        # Neighbour-choice arms: V13's window, weights and projection with the four
        # neighbours picked at random / from the least similar end instead of the top.
        neighbor_arms = {}
        modes = [str(mode) for mode in (getattr(args, "v13_neighbor_modes", None) or [])]
        if modes:
            if "D" not in built or v11_evict is None:
                raise ValueError("the neighbour arms need the V13 window and module")
            for mode in modes:
                if mode not in NEIGHBOR_ARMS:
                    raise ValueError(f"unknown neighbour mode {mode!r}")
                arm = NEIGHBOR_ARMS[mode]
                swapped, swap_report = cross_block_context(
                    built["D"]["window"], built["D"]["meta"],
                    k=v11_evict["module"].k_neighbors, return_report=True,
                    selection=mode, seed=args.seed,
                    span_value=v11_evict["module"].span_value,
                    context_diffusion=v11_evict["module"].context_diffusion,
                    csls=v11_evict["module"].csls)
                swapped_window, swapped_info = calibrate(
                    built["D"]["window"], built["D"]["meta"], v11_evict["module"],
                    context=swapped)
                swapped_info["neighbour_swap"] = swap_report
                built[arm] = dict(window=built["D"]["window"],
                                  calibrated=swapped_window,
                                  check=built["D"]["check"],
                                  calibration=check_calibration(
                                      swapped_window, built["D"]["window"],
                                      built["D"]["meta"], swapped_info, arm))
                neighbor_arms[arm] = swap_report
                windows[arm] = built[arm]["calibrated"]
                del swapped, swapped_window
        # Permutation arms: the SAME window, the SAME trained module and the SAME
        # self-calibration branch as arm D; only the mapping slot -> context row is
        # shuffled, inside (source_block, slot_type) groups so the batch a slot draws
        # from is untouched. Nothing is retrained and nothing is re-selected.
        permuted_arms = {}
        seeds = list(getattr(args, "permute_seeds", []) or [])
        if seeds:
            if "D" not in built or v11_evict is None:
                raise ValueError("the permutation arms need the V13 window and module")
            base_context, _ = v11_evict["module"].neighbours(built["D"]["window"],
                                                             built["D"]["meta"],
                                                             return_report=True)
            for seed in seeds:
                arm = permute_arm(seed)
                moved, move_info = permute_context(base_context, built["D"]["meta"], seed)
                if torch.equal(moved, base_context):
                    raise ValueError(f"{arm}: the permutation left every context row in "
                                     f"place, so this arm is arm D under another name")
                arm_window, arm_info = calibrate(built["D"]["window"], built["D"]["meta"],
                                                 v11_evict["module"], context=moved)
                built[arm] = dict(window=built["D"]["window"], calibrated=arm_window,
                                  check=built["D"]["check"],
                                  calibration=check_calibration(
                                      arm_window, built["D"]["window"], built["D"]["meta"],
                                      arm_info, arm))
                permuted_arms[arm] = move_info["moved_fraction"]
                windows[arm] = built[arm]["calibrated"]
                del moved, arm_window
            del base_context
        # Arms F and G: the SAME windows as D and E, calibrated by the complement-mode
        # module. Both modules see the identical slot vectors, so the difference between
        # V13 and V14 is the neighbour rule and nothing else -- not the window, not the
        # graft content, not the number of parameters that moved.
        if v14 is not None:
            if "D" not in built:
                raise ValueError("--v14 needs the cross-block window it has to share")
            v14_calibrated, v14_info = calibrate(built["D"]["window"], built["D"]["meta"],
                                                 v14["module"])
            built["F"] = dict(window=built["D"]["window"], calibrated=v14_calibrated,
                              check=built["D"]["check"],
                              calibration=check_calibration(
                                  v14_calibrated, built["D"]["window"],
                                  built["D"]["meta"], v14_info, V14))
            windows[V14] = built["F"]["calibrated"]
            if "E" in built:
                v14_donor, v14_donor_info = calibrate(built["E"]["window"],
                                                      built["D"]["meta"], v14["module"])
                built["G"] = dict(window=built["E"]["window"],
                                  calibrated=v14_donor,
                                  check=built["D"]["check"],
                                  calibration=check_calibration(
                                      v14_donor, built["E"]["window"], built["D"]["meta"],
                                      v14_donor_info, V14_DONOR))
                windows[V14_DONOR] = built["G"]["calibrated"]
        # Arms N..R: V17 (key anchoring + value release) on the SAME window as V13, and
        # the inference-time ablations of one runtime knob each. `V17` itself runs the
        # checkpoint's own module; every derived arm runs a deep copy with one attribute
        # replaced, so the checkpoint's module is never mutated and `V17` stays valid for
        # the arms that are built after it.
        v17_arms = {}
        if v17 is not None:
            if "D" not in built:
                raise ValueError("--v17 needs the cross-block window it has to share")
            built["N"] = calibrate_deep(v17, built["D"]["window"], built["D"]["meta"],
                                        V17, source=source)
            built["N"]["check"] = built["D"]["check"]
            windows[V17] = built["N"]["calibrated"]
            v17_arms[V17] = dict(knob="checkpoint",
                                 note="the module exactly as the checkpoint wrote it")
            # `--v17-value-ratios`: the value-visible ceiling is a runtime attribute of
            # the module, so the same weights can be read at 25 / 30 / 40 % of the slot
            # RMS without touching a parameter or rewriting the checkpoint.
            for ratio in list(getattr(args, "v17_value_ratios", None) or []):
                arm = value_arm(ratio)
                tuned = dict(v17, module=runtime_knob(v17["module"], "value_ratio",
                                                      float(ratio)))
                built[arm] = calibrate_deep(tuned, built["D"]["window"],
                                            built["D"]["meta"], arm, source=source)
                built[arm]["check"] = built["D"]["check"]
                windows[arm] = built[arm]["calibrated"]
                v17_arms[arm] = dict(knob="value_ratio", value_ratio=float(ratio),
                                     note="inference-time ceiling; weights untouched")
                del tuned
            # `--v17-band-off`: the three-band gate is a runtime mode, so the ablation is
            # the same module with `band_mode="off"`.
            if getattr(args, "v17_band_off", False):
                arm = V17_NO_BAND
                tuned = dict(v17, module=runtime_knob(v17["module"], "band_mode", "off"))
                built[arm] = calibrate_deep(tuned, built["D"]["window"],
                                            built["D"]["meta"], arm, source=source)
                built[arm]["check"] = built["D"]["check"]
                windows[arm] = built[arm]["calibrated"]
                v17_arms[arm] = dict(
                    knob="band_mode", band_mode="off",
                    note="ablation: the band gate is off, every other weight is the "
                         "checkpoint's -- `V17 - V17NoBand` is that gate's worth")
                del tuned
            # `--v17-no-attn`: mechanism four turned OFF. The trainer damps the training
            # window before it measures the neighbour context, and that damping is in the
            # trained weights; a rebuilt checkpoint carries the same constants, so
            # `deep_fusion.calibrate` damps the evaluation window by itself. Neutralising
            # the three constants on a COPY and adding no damping here is therefore the
            # only arm in which the damping mechanism does not run at all -- `V17 -
            # V17NoAttn` is what it is worth. No weight, no checkpoint and no `V17`
            # attribute is touched: the copy is thrown away after this row.
            if getattr(args, "v17_no_attn", False):
                arm = V17_NO_ATTN
                neutral_module, neutralised = neutralise_graft_attention(v17["module"])
                tuned = dict(v17, module=neutral_module)
                built[arm] = calibrate_deep(
                    tuned, built["D"]["window"], built["D"]["meta"], arm,
                    source=source,
                    extra=dict(neutralised=neutralised,
                               checkpoint_params=dict(
                                   tau=float(v17["attention"]["tau"]),
                                   bias=float(v17["attention"]["bias"]),
                                   norm_quantile=float(
                                       v17["attention"]["norm_quantile"])),
                               note="mechanism four off: tau/bias/norm_quantile "
                                    "neutralised on a copy and NO explicit damping, so "
                                    "the module's own calibrate damps nothing"))
                built[arm]["check"] = built["D"]["check"]
                windows[arm] = built[arm]["calibrated"]
                v17_arms[arm] = dict(
                    knob="graft_attention_off",
                    tau=1.0, bias=0.0, norm_quantile=0.0,
                    neutralised=neutralised,
                    checkpoint_tau=float(v17["attention"]["tau"]),
                    checkpoint_bias=float(v17["attention"]["bias"]),
                    checkpoint_norm_quantile=float(v17["attention"]["norm_quantile"]),
                    note="ablation: mechanism four is off. `V17 - V17NoAttn` is the "
                         "damping's worth; if this arm's window equals V17's, the "
                         "checkpoint never damped anything to begin with")
                del tuned, neutral_module
            # `--v17-attention`: the graft-only key damping applied ONCE by this driver.
            # Only built when the checkpoint's own constants are neutral (see
            # `decide_attention_arm`, which skipped this arm at load time otherwise): with
            # a self-damping checkpoint the module damps the window inside `calibrate`
            # already, and damping it again here would be V17 with the pass applied twice.
            if getattr(args, "v17_attention", False):
                arm = V17_ATTN
                params = v17["attention"]
                df = deep_fusion()
                damped, damp_report = df.graft_attention_calibration(
                    built["D"]["window"], built["D"]["meta"], v17["basis"],
                    tau=float(params["tau"]), bias=float(params["bias"]),
                    norm_quantile=float(params["norm_quantile"]),
                    # The damping is per window ROW: an identity layer table means one id
                    # per row, in order, which is what a 32-row stack needs.
                    layer_ids=tuple(int(x) for x in (
                        v17["config"].get("layer_ids")
                        or range(int(built["D"]["window"].shape[0])))),
                    return_report=True)
                moved = not torch.equal(damped, built["D"]["window"])
                neutralised = []
                module = v17["module"]
                if (built["N"]["calibration"].get("deep") or {}).get("damp"):
                    module, neutralised = neutralise_graft_attention(module)
                applied = bool((damp_report or {}).get("applied"))
                touched = int((damp_report or {}).get("slots_touched") or 0)
                if not applied or touched == 0:
                    print(f"WARNING: {arm}: graft_attention_calibration touched no "
                          f"slot (tau={params['tau']}, bias={params['bias']}, "
                          f"norm_quantile={params['norm_quantile']}): this arm is V17 "
                          f"under another name. `--v17-no-attn` is the arm that turns "
                          f"mechanism four OFF.", file=sys.stderr)
                tuned = dict(v17, module=module)
                built[arm] = calibrate_deep(
                    tuned, damped, built["D"]["meta"], arm, source=source,
                    extra=dict(graft_attention=damp_report, window_damped=moved,
                               neutralised=neutralised,
                               params=dict(tau=float(params["tau"]),
                                           bias=float(params["bias"]),
                                           norm_quantile=float(params["norm_quantile"]),
                                           recorded=params["recorded"]),
                               note="the window was damped ONCE with the checkpoint's "
                                    "tau/bias before the module ran; the module's own "
                                    "copy of those constants is neutralised on a copy"))
                built[arm]["check"] = built["D"]["check"]
                windows[arm] = built[arm]["calibrated"]
                v17_arms[arm] = dict(knob="graft_attention",
                                     tau=float(params["tau"]),
                                     bias=float(params["bias"]),
                                     norm_quantile=float(params["norm_quantile"]),
                                     window_moved=bool(moved),
                                     note="graft-only key damping, tau/bias as recorded "
                                          "in the checkpoint")
                del damped, tuned
            # Did the knob actually reach the window? A value ceiling that is not
            # binding, a band gate that was already open and a damping constant of
            # (1.0, 0.0) all produce the identical calibrated window. That is a result,
            # not an error, but it has to be recorded or the arm looks like a treatment
            # that did something.
            for name, info in v17_arms.items():
                if name == V17:
                    info["window_equals_v17"] = True
                    continue
                info["window_equals_v17"] = bool(torch.equal(
                    built[name]["calibrated"], built["N"]["calibrated"]))
        # The density-quota window: the SAME 1,620 core slots, redistributed across the
        # nine carriers by redundancy instead of the fixed 180 per block. Two arms read
        # it -- V13's module and V17's -- so the module comparison is repeated on a
        # window whose quota is not the fixed one, and each is paired with its own
        # fixed-quota arm on the same question.
        density_record = None
        if density is not None:
            if v11_evict is None:
                raise ValueError("--v17-window-density needs --v11-evict: the fixed-quota "
                                 "V13 arm is what the density window is compared with")
            density_window, density_meta, density_telemetry = density["joint"].build(
                source, picked, manager.block)
            built["Q"] = dict(window=density_window,
                              meta=density_meta,
                              check=check_v11(density_window, density_meta,
                                              density_telemetry, source, V13_DENSITY,
                                              allow_absorbed=float(
                                                  density["absorb_cos"]) > 0.0))
            quota = _per_block_quota(density_meta, density_telemetry,
                                     density_telemetry["carrier_blocks"])
            if quota and max(quota) == min(quota):
                print(f"WARNING: --v17-window-density returned a uniform quota "
                      f"({quota[0]} core slots in every carrier block): the window is "
                      f"the fixed-quota one under another name.", file=sys.stderr)
            dense_window, dense_info = calibrate(density_window, density_meta,
                                                 v11_evict["module"], source=source)
            built["Q"]["calibrated"] = dense_window
            built["Q"]["calibration"] = check_calibration(
                dense_window, density_window, density_meta, dense_info, V13_DENSITY)
            windows[V13_DENSITY] = dense_window
            density_record = dict(
                telemetry=_density_telemetry(density_telemetry),
                check=built["Q"]["check"], calibration=built["Q"]["calibration"],
                v17_calibration=None,
                quota=dict(per_block=quota, core_total=int(sum(quota)),
                           min=min(quota) if quota else None,
                           max=max(quota) if quota else None,
                           fixed=int(v11_evict["geometry"]["core_per_block"])),
                quota_mode=str((density_telemetry or {}).get("quota_mode", "density")),
                absorb_cos=float(density["absorb_cos"]),
                note="same capacity as the fixed-quota window; only the per-block split "
                     "differs, so `V13Density - V13` is the quota rule and "
                     "`V17Density - V13Density` is the module")
            if v17 is not None:
                built["R"] = calibrate_deep(dict(v17), density_window, density_meta,
                                            V17_DENSITY, source=source)
                built["R"]["check"] = built["Q"]["check"]
                windows[V17_DENSITY] = built["R"]["calibrated"]
                density_record["v17_calibration"] = built["R"]["calibration"]
            else:
                print("WARNING: --v17-window-density without --v17: the density window is "
                      "answered by V13's module only, so V17Density is not in this run.",
                      file=sys.stderr)
            del density_window, dense_window
        next_donor = built["B"]["window"].index_select(1, graft_offsets).detach().clone()
    gr.restore(model, manager, state)

    def prepare(arm):
        def run():
            gr.restore(model, manager, state)
            if arm == SELECTION:
                gr.install_selection(manager, source, picked, args.window_blocks)
            elif arm != FULL:
                er.install_output(manager, windows[arm], picked, args.window_blocks)
            fingerprints[arm] = gr.active_fingerprint(manager)
            fingerprints[arm + "/slots"] = int(manager.used)
        return run

    # The permutation arms are named after their seed and are not in ARMS; every arm that
    # has a window must still be read out, or the arm silently disappears from the table.
    ordered = list(ARMS) + [arm for arm in windows if arm not in ARMS]
    answered = {arm: gr.readout(model, tokenizer, prepared["query_ids"], answers,
                                args.f1_threshold, prepare(arm))
                for arm in ordered if arm in windows or arm in (FULL, SELECTION)}
    gr.restore(model, manager, state)

    if int(fingerprints[FULL + "/slots"]) != args.window_blocks * manager.block:
        raise ValueError("the Full arm did not keep the whole window")
    for arm in ARMS:
        if arm in (FULL, SELECTION):
            continue
        if arm not in windows:
            if (arm in (V11_DONOR, V14, V14_DONOR, V15, V15_DONOR, V16, V16_DONOR,
                        V13_PLUS, V13_PLUS_DONOR)
                    or arm in DEEP_ARMS or arm in DENSITY_ARMS
                    or arm.startswith(PERMUTE_PREFIX) or arm.startswith(VALUE_PREFIX)):
                # A run without the donor document (or without --v14/--v17) has no such
                # arm; it is simply not answered, and `arms_present` records that.
                continue
            raise ValueError(f"{arm} produced no window this run")
        if int(fingerprints[arm + "/slots"]) != TARGET_SLOTS:
            raise ValueError(f"{arm} does not hold the {TARGET_SLOTS}-slot budget")
    for left, right in ((SELECTION, ANCHOR_ONLY), (ANCHOR_ONLY, V11), (V11, GRAFT)):
        if fingerprints[left] == fingerprints[right]:
            raise ValueError(f"{left} and {right} read the same pool")
    if len(prepared) > 1 and V11_DONOR not in windows:
        raise ValueError("the donor arm was not built: the content-causality control "
                         "is missing, so this run cannot answer its own question")
    if len(prepared) > 1 and V11_EVICT in windows and V11_EVICT_DONOR not in windows:
        raise ValueError("the eviction line's donor arm was not built: this run cannot "
                         "attribute arm D's gain to content")
    if V11_DONOR in windows and fingerprints[V11] == fingerprints[V11_DONOR]:
        raise ValueError("the donor arm reads the same pool as V11: the swap did not "
                         "reach the memory")
    if V11_EVICT_DONOR in windows and (
            fingerprints[V11_EVICT] == fingerprints[V11_EVICT_DONOR]):
        raise ValueError("the eviction donor arm reads the same pool as V11Evict")
    if V11_EVICT in windows and fingerprints[V11] == fingerprints[V11_EVICT]:
        raise ValueError("the eviction control reads the same pool as V11: the fold "
                         "moved nothing")
    if V14 in windows and fingerprints[V14] == fingerprints[V11_EVICT]:
        raise ValueError("V14 reads the same pool as V13: the complement neighbour rule "
                         "changed not one slot, so this run cannot speak about it")
    if len(prepared) > 1 and V14 in windows and V14_DONOR not in windows:
        raise ValueError("V14's donor arm was not built: the complement arm's content "
                         "causality cannot be answered by V13's donor")
    if V14_DONOR in windows and fingerprints[V14] == fingerprints[V14_DONOR]:
        raise ValueError("V14's donor arm reads the same pool as V14: the swap did not "
                         "reach the memory")
    if V13_PLUS in windows and fingerprints[V13_PLUS] == fingerprints[V11_EVICT]:
        raise ValueError("V13Plus reads the same pool as V13: the new context rule "
                         "changed nothing")
    if V16 in windows and fingerprints[V16] == fingerprints[V11_EVICT]:
        raise ValueError("V16 reads the same pool as V13: the band injection changed "
                         "nothing, so this run cannot speak about it")
    if len(prepared) > 1 and V16 in windows and V16_DONOR not in windows:
        raise ValueError("V16's donor arm was not built: its content causality cannot be "
                         "answered by V13's donor")
    if V15 in windows and fingerprints[V15] == fingerprints[V11_EVICT]:
        raise ValueError("V15 reads the same pool as V13: the relation module changed "
                         "nothing, so this run cannot speak about it")
    if len(prepared) > 1 and V15 in windows and V15_DONOR not in windows:
        raise ValueError("V15's donor arm was not built: its content causality cannot be "
                         "answered by V13's donor")
    for arm in cap_arms:
        if fingerprints[V11_EVICT] == fingerprints[arm]:
            raise ValueError(f"{arm} reads the same pool as V13: the cap changed nothing")
    for arm in neighbor_arms:
        if fingerprints[V11_EVICT] == fingerprints[arm]:
            raise ValueError(f"{arm} reads the same pool as V13: the neighbour swap "
                             f"changed nothing")
    for arm in permuted_arms:
        if fingerprints[V11_EVICT] == fingerprints[arm]:
            raise ValueError(f"{arm} reads the same pool as V13: the permutation changed "
                             f"no answer, which is not the same as it changing no slot")
        if int(fingerprints[arm + "/slots"]) != TARGET_SLOTS:
            raise ValueError(f"{arm} does not hold the {TARGET_SLOTS}-slot budget")
    # V17 has to be distinguishable from V13, but only the arm itself is refused when the
    # checkpoint turns nothing on: a knob that is on and not binding is a result, and is
    # warned about rather than killed.
    if V17 in windows and V11_EVICT in windows and (
            fingerprints[V17] == fingerprints[V11_EVICT]):
        knobs = (built["N"]["calibration"].get("deep") or {})
        trivial = (str(knobs.get("key_channel") or "off") == "off"
                   and str(knobs.get("band_mode") or "off") == "off"
                   and knobs.get("value_ratio") is None
                   and float(v17["config"].get("key_ratio") or 0.0) == 0.0)
        if trivial:
            raise ValueError("V17 reads the same pool as V13 and every V17 knob is at its "
                             "default (key_channel/value_ratio/band_mode/key_ratio): the "
                             "checkpoint is V13 under another name, not a V17 arm")
        print("WARNING: V17's calibrated window equals V13's although the checkpoint "
              "turns a V17 knob on: that knob is not binding on this window.",
              file=sys.stderr)
    for arm in (V17_NO_BAND, V17_NO_ATTN, V17_ATTN):
        if arm in windows and V17 in windows and fingerprints[arm] == fingerprints[V17]:
            extra = ("the checkpoint's damping constants are already neutral, so switching "
                     "them off changes nothing and there is no mechanism four to measure"
                     if arm == V17_NO_ATTN else
                     "the knob it turns is not binding")
            print(f"WARNING: {arm} calibrated the window exactly as V17 did: {extra}. "
                  f"`{arm} - V17` is a null result, not a missing arm.",
                  file=sys.stderr)

    if V13_DENSITY in windows and V11_EVICT in windows and (
            fingerprints[V13_DENSITY] == fingerprints[V11_EVICT]):
        print("WARNING: the density window calibrated to the same pool as the fixed-quota "
              "window: the quota rule made no difference on this row.", file=sys.stderr)

    record = dict(
        arms=answered, fingerprints=fingerprints, baseline=baseline,
        v11=built["B"]["check"], v11_calibration=built["B"]["calibration"],
        donor=dict(window_check=built["C"]["check"] if "C" in built else None,
                   calibration=built["C"]["calibration"] if "C" in built else None,
                   source_id=donor_id),
        evict=(dict(window_check=built["D"]["check"],
                    calibration=built["D"]["calibration"]) if "D" in built else None),
        evict_donor=(dict(calibration=built["E"]["calibration"], source_id=donor_id)
                     if "E" in built else None),
        v14=(dict(window_check=built["F"]["check"],
                  calibration=built["F"]["calibration"]) if "F" in built else None),
        v14_donor=(dict(calibration=built["G"]["calibration"], source_id=donor_id)
                   if "G" in built else None),
        v15=(dict(window_check=built["H"]["check"],
                  calibration=built["H"]["calibration"]) if "H" in built else None),
        v15_donor=(dict(calibration=built["I"]["calibration"], source_id=donor_id)
                   if "I" in built else None),
        v13_plus=(dict(window_check=built["L"]["check"],
                       calibration=built["L"]["calibration"]) if "L" in built else None),
        v13_plus_donor=(dict(calibration=built["M"]["calibration"], source_id=donor_id)
                        if "M" in built else None),
        v16=(dict(window_check=built["J"]["check"],
                  calibration=built["J"]["calibration"]) if "J" in built else None),
        v16_donor=(dict(calibration=built["K"]["calibration"], source_id=donor_id)
                   if "K" in built else None),
        graft_cap={arm: dict(info=info, calibration=built[arm]["calibration"])
                   for arm, info in cap_arms.items()} or None,
        neighbor_swap={arm: dict(calibration=built[arm]["calibration"],
                                 report=report)
                       for arm, report in neighbor_arms.items()} or None,
        permutation={arm: dict(calibration=built[arm]["calibration"],
                               moved_fraction=moved)
                     for arm, moved in permuted_arms.items()} or None,
        # V17's arms. `v17` is the checkpoint arm itself; `v17_arms` carries one entry per
        # derived arm with the knob it turned and whether that knob reached the window.
        v17=(dict(window_check=built["N"]["check"],
                  calibration=built["N"]["calibration"],
                  module=dict(key_channel=v17["config"].get("key_channel"),
                              key_ratio=_scalars(v17["config"].get("key_ratio")),
                              value_ratio=_scalars(v17["config"].get("value_ratio")),
                              band_mode=v17["config"].get("band_mode"),
                              band_gates=_scalars(v17["config"].get("band_gates")),
                              sim_floor=_scalars(v17["config"].get("sim_floor")),
                              layer_ids=(list(v17["config"]["layer_ids"])
                                         if v17["config"].get("layer_ids")
                                         else "identity: row i is reader layer i"),
                              key_basis_installed=bool(v17.get("basis") is not None)))
             if "N" in built else None),
        v17_arms={arm: dict(info=info, calibration=built[arm]["calibration"])
                  for arm, info in v17_arms.items()} or None,
        density=density_record,
        dropped_blocks=[int(block) for block in parts["dropped_blocks"]],
        selected_blocks=[int(block) for block in telemetry["selected_blocks"]],
        carrier_ages=[int(ages[block]) for block in picked])
    return record, dict(grafts=next_donor, grafts_evict=next_donor_evict,
                        id=prepared["id"])


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def _stats(values):
    values = [float(value) for value in values]
    if not values:
        return dict(n=0)
    return dict(n=len(values), mean=round(sum(values) / len(values), 6),
                min=round(min(values), 6), max=round(max(values), 6),
                median=round(sorted(values)[len(values) // 2], 6))


def _evidence_in_dropped(sample):
    blocks = set(int(block) for block in (sample.get("evidence", {}).get("blocks") or []))
    return bool(blocks) and blocks <= set(sample["dropped_blocks"])


def _count(rows, arm, threshold):
    """Correct rows, as a count rather than a percentage, for the planned thresholds."""
    return int(sum(1 for row in rows if gr._correct(row, arm, threshold)))


def _neighbour_composition(samples, key):
    """Per-row neighbour diagnostics of one arm's calibration record, aggregated.

    This is the block V13 never had: without it, "the top-4 neighbours are all carriers"
    is a hypothesis about the mechanism, not a measurement. `cross_weight_mass_mean` is
    the one that catches a complement arm which selected two of each kind but let the
    own-kind side keep all the weight.
    """
    reports = [(sample.get(key) or {}).get("calibration") or {} for sample in samples]
    reports = [row.get("cross_context") for row in reports]
    reports = [report for report in reports if report]
    if not reports:
        return None
    fields = ("mode", "k", "neighbour_similarity_mean", "neighbour_block_spread_mean",
              "context_rms", "neighbour_graft_fraction", "neighbour_type_cross_fraction",
              "core_to_graft_neighbour_fraction", "graft_to_core_neighbour_fraction",
              "cross_weight_mass_mean", "own_kind_weight_mass_mean")
    summary = {field: _stats([report[field] for report in reports if field in report])
               for field in fields if field not in ("mode", "k")}
    summary["mode"] = sorted({str(report.get("mode")) for report in reports})
    summary["k"] = sorted({int(report.get("k")) for report in reports if "k" in report})
    summary["n"] = len(reports)
    return summary


def permutation_report(graded, threshold, seeds):
    """The three numbers the permutation test exists to produce, per seed.

    `rescued_by_v13`  rows V13 answers and the plain calibration does not;
    `then_lost`       of those, the rows the permuted arm answers WRONG -- the direct
                      question "does V13's gain survive when the pairing is destroyed";
    `damage`          rows Full answers that the permuted arm does not, which is where a
                      redistribution would show up.
    """
    paired = [sample for sample in graded
              if V11 in sample.get("arms", {}) and V11_EVICT in sample.get("arms", {})]

    def correct(sample, arm):
        return gr._correct(sample, arm, threshold)

    rescued = [sample for sample in paired if correct(sample, V11_EVICT)
               and not correct(sample, V11)]
    arms = {}
    for seed in seeds:
        arm = permute_arm(seed)
        rows = [sample for sample in paired if arm in sample.get("arms", {})]
        if not rows:
            continue
        moved = [sample["permutation"][arm]["moved_fraction"] for sample in rows
                 if sample.get("permutation") and arm in sample["permutation"]]
        arms[arm] = dict(
            seed=int(seed),
            correct=int(sum(1 for sample in rows if correct(sample, arm))),
            accuracy=round(100.0 * sum(1 for sample in rows if correct(sample, arm))
                           / len(rows), 2),
            versus_v13=er.pair_summary(rows, arm, V11_EVICT, threshold),
            versus_plain=er.pair_summary(rows, arm, V11, threshold),
            rescued_by_v13=len(rescued),
            rescued_by_v13_then_lost=int(sum(1 for sample in rescued
                                             if not correct(sample, arm))),
            changed_answers=int(sum(1 for sample in rows
                                    if sample["arms"][arm]["prediction"]
                                    != sample["arms"][V11_EVICT]["prediction"])),
            damage=ef._damage(rows, arm, threshold),
            context_moved_fraction=_stats(moved),
            calibration=dict(
                core_relative_delta=_stats([
                    sample["permutation"][arm]["calibration"]["core_relative_delta"]
                    for sample in rows]),
                graft_relative_delta=_stats([
                    sample["permutation"][arm]["calibration"]["graft_relative_delta"]
                    for sample in rows])))
    return dict(
        definition=("inference-time only: the trained V13 module, its weights, the window "
                    "and the self-calibration branch are unchanged; the precomputed "
                    "cross-block context rows are shuffled inside (source_block, "
                    "slot_type) groups, so every slot draws from the same batch of "
                    "contexts as before and only WHICH row it receives changes"),
        seeds=[int(seed) for seed in seeds], arms=arms,
        original=dict(arm=V11_EVICT,
                      correct=int(sum(1 for sample in paired if correct(sample, V11_EVICT))),
                      rescued_by_v13=len(rescued), damage=ef._damage(
                          paired, V11_EVICT, threshold)),
        plain=dict(arm=V11,
                   correct=int(sum(1 for sample in paired if correct(sample, V11)))),
        reading=("a stable drop of 4-6 questions means the exact pairing carries "
                 "information; 0-2 means this run has no evidence that it does. Neither "
                 "is a significance test, and neither shows that the model performs "
                 "cross-block multi-hop reasoning"))


def _band_gate(samples):
    """V16's per-band gate distribution and the injected RMS, aggregated over rows."""
    rows = [((sample.get("v16") or {}).get("calibration") or {}) for sample in samples]
    rows = [row for row in rows if row]
    if not rows:
        return None
    bands = sorted({band for row in rows for band in (row.get("gate_bands") or {})})
    summary = {}
    for band in bands:
        entry = {}
        for field in ("gate_core_mean", "gate_core_std", "gate_graft_mean",
                      "gate_graft_std"):
            entry[field] = _stats([row["gate_bands"][band][field] for row in rows
                                   if band in row.get("gate_bands", {})
                                   and field in row["gate_bands"][band]])
        summary[band] = entry
    return dict(
        n=len(rows), bands=summary,
        band_proj_norm={band: _stats([row["band_proj_norm"][band] for row in rows
                                      if row.get("band_proj_norm")])
                        for band in range(len(rows[0].get("band_proj_norm") or []))},
        layer_proj_norm=_stats([row["layer_proj_norm"] for row in rows
                                if row.get("layer_proj_norm") is not None]),
        cross_injection_rms=_stats([row["cross_injection_rms"] for row in rows
                                    if row.get("cross_injection_rms") is not None]),
        shallow_cross_leak=_stats([row["shallow_cross_leak"] for row in rows
                                   if row.get("shallow_cross_leak") is not None]),
        core_relative_delta=_stats([row["core_relative_delta"] for row in rows
                                    if row.get("core_relative_delta") is not None]),
        graft_relative_delta=_stats([row["graft_relative_delta"] for row in rows
                                     if row.get("graft_relative_delta") is not None]),
        note="gate is omega*eta applied after the projection; shallow_cross_leak is the "
             "largest cross-term value on memory layers 0..9 and must be exactly 0")


def _relation_gate(samples):
    """Per-layer sink occupancy and retained mass of V15, aggregated over rows.

    This is the number the design turns on: a sink that never closes leaves the retained
    mass at 1 and the gate is decorative, whatever the loss did.
    """
    reports = [((sample.get("v15") or {}).get("calibration") or {}).get("relation")
               for sample in samples]
    reports = [report for report in reports if report]
    if not reports:
        return None
    layers = sorted({layer for report in reports
                     for layer in (report.get("per_layer") or {})})
    fields = ("retained_mass", "sink_mass", "neighbour_similarity_mean", "feature_rms")
    return dict(
        n=len(reports),
        retained_mass_mean=_stats([report["retained_mass_mean"] for report in reports
                                   if "retained_mass_mean" in report]),
        sink_mass_mean=_stats([report["sink_mass_mean"] for report in reports
                               if "sink_mass_mean" in report]),
        tau=reports[0].get("tau"), sink_similarity=reports[0].get("sink_similarity"),
        per_layer={layer: {field: _stats(
            [report["per_layer"][layer][field] for report in reports
             if layer in (report.get("per_layer") or {})
             and field in report["per_layer"][layer]])
            for field in fields} for layer in layers},
        note="retained mass is the share of the context weight the sink left for "
             "injection (<= 1 by construction); a value near 1 on every layer means the "
             "gate is decorative and the run is V13 with extra parameters")


def _v17_report(samples):
    """What each V17 arm did to the window, aggregated over rows.

    Two things this table must not hide. First, whether the knob reached the window at
    all: `rows_where_window_equals_v17` counts the rows on which the derived arm produced
    the very same calibrated window as `V17`, which is a null result and not a treatment
    that happened to change nothing in the answers. Second, how the residual was split
    between the key side and the value side (`key_channel`, `key_ratio`, `value_ratio`,
    `band_mode`) and whether the module damped the graft keys itself (`damp`).
    """
    arms = {}
    for sample in samples:
        for arm, record in (sample.get("v17_arms") or {}).items():
            arms.setdefault(arm, []).append((record.get("info") or {},
                                             record.get("calibration") or {}))
    if not arms:
        return None
    summary = {}
    for arm, pairs in sorted(arms.items()):
        infos = [pair[0] for pair in pairs]
        rows = [pair[1] for pair in pairs if pair[1]]
        deep = [row.get("deep") or {} for row in rows]
        damp = [(row.get("damp") or {}) for row in deep]
        attention = [(info.get("graft_attention") or {}) for info in infos]
        summary[arm] = dict(
            n=len(rows), knob=infos[0].get("knob"), note=infos[0].get("note"),
            key_channel=sorted({str(row.get("key_channel")) for row in deep}),
            band_mode=sorted({str(row.get("band_mode")) for row in deep}),
            value_ratio=sorted({str(row.get("value_ratio")) for row in deep}),
            key_ratio=sorted({str(row.get("key_ratio")) for row in deep}),
            rows_where_window_equals_v17=int(
                sum(1 for info in infos if info.get("window_equals_v17") is True)),
            rows_where_window_differs_from_v17=int(
                sum(1 for info in infos if info.get("window_equals_v17") is False)),
            core_relative_delta=_stats([row["core_relative_delta"] for row in rows]),
            graft_relative_delta=_stats([row["graft_relative_delta"] for row in rows]),
            gate=_stats([row["gate"] for row in rows if row.get("gate") is not None]),
            # The module's own damping, when its config turns it on.
            damp_rows=int(sum(1 for report in damp if report)),
            damp_slots_touched=_stats([report["slots_touched"] for report in damp
                                       if isinstance(report, dict)
                                       and report.get("slots_touched") is not None]),
            damp_key_scale_mean=_stats([report["key_scale_mean"] for report in damp
                                        if isinstance(report, dict)
                                        and report.get("key_scale_mean") is not None]),
            # The explicit damping this driver applied for `V17Attn`.
            graft_attention=(dict(
                window_damped_rows=int(sum(1 for info in infos
                                          if info.get("window_damped"))),
                tau=_stats([report["tau"] for report in attention
                            if report.get("tau") is not None]),
                bias=_stats([report["bias"] for report in attention
                             if report.get("bias") is not None]),
                norm_quantile=_stats([report["norm_quantile"] for report in attention
                                      if report.get("norm_quantile") is not None]),
                slots_touched=_stats([report["slots_touched"] for report in attention
                                      if report.get("slots_touched") is not None]),
                key_scale_mean=_stats([report["key_scale_mean"] for report in attention
                                       if report.get("key_scale_mean") is not None]),
                key_norm_before=_stats([report["key_norm_before"] for report in attention
                                        if report.get("key_norm_before") is not None]),
                key_norm_after=_stats([report["key_norm_after"] for report in attention
                                       if report.get("key_norm_after") is not None]),
                logit_proxy=_stats([report["logit_proxy"] for report in attention
                                    if report.get("logit_proxy") is not None]),
                neutralised=sorted({name for info in infos
                                    for name in (info.get("neutralised") or [])}),
                recorded=sorted({str(info.get("params", {}).get("recorded"))
                                 for info in infos if info.get("params")}))
                # `V17NoAttn` neutralises the constants instead of damping, so it has no
                # `graft_attention` report of its own: the arm still has to show WHICH
                # constants it switched off, or the ablation is invisible in the table.
                if any(attention) or any(info.get("neutralised") for info in infos)
                else None))
    return dict(
        definition=("every V17 arm calibrates the window V13 calibrates; the derived arms "
                    "run a deep copy of the checkpoint's module with ONE runtime knob "
                    "changed, so `arm - V17` is that knob and nothing else"),
        arms=summary,
        note="`rows_where_window_equals_v17` == n on a derived arm means the knob was not "
             "binding: the arm is V17 under another name and its paired difference is a "
             "null result, not a gain")


def _window_density(samples):
    """The density-quota windows: per-block quota, redundancy and absorption.

    The quota rule is only interesting if it is visible: `per_block_quota` must satisfy
    `min <= quota <= max` for the widths the run used, must sum to the documented 1,620,
    and must not be the fixed 180 in every block. All three are recorded here rather than
    asserted, because a uniform quota is a finding about the density measure, not a crash.
    """
    rows = [sample.get("density") for sample in samples]
    rows = [row for row in rows if row]
    if not rows:
        return None
    quotas = [(row.get("quota") or {}) for row in rows]
    per_block = [[int(x) for x in (quota.get("per_block") or [])] for quota in quotas]
    flat = [value for block in per_block for value in block]
    telemetry = [(row.get("telemetry") or {}) for row in rows]

    def _collect(key):
        values = [row[key] for row in telemetry if row.get(key) is not None]
        return _stats(values) if values else None

    return dict(
        definition=("the same 2,304-slot window and the same 1,620 core slots; only the "
                    "split across the nine carrier blocks changes, from the fixed quota "
                    "to a redundancy-weighted one, before either module calibrates it"),
        n=len(rows),
        quota_mode=sorted({str(row.get("quota_mode")) for row in rows}),
        fixed_core_per_block=sorted({int(quota["fixed"]) for quota in quotas
                                     if quota.get("fixed") is not None}),
        core_total=sorted({int(quota["core_total"]) for quota in quotas
                           if quota.get("core_total") is not None}),
        per_block_quota=_stats(flat),
        per_block_quota_spread=_stats([max(block) - min(block) for block in per_block
                                       if block]),
        blocks_with_uniform_quota=int(sum(1 for block in per_block
                                          if block and max(block) == min(block))),
        absorb_cos=sorted({float(row.get("absorb_cos") or 0.0) for row in rows}),
        density=_stats([float(value) for row in telemetry
                        for value in (row.get("density") or [])]),
        absorbed_pairs=_collect("absorbed_pairs"),
        absorbed_mean_alpha=_collect("absorbed_mean_alpha"),
        absorbed_mean_cos=_collect("absorbed_mean_cos"),
        note="`density` is the window builder's per-block redundancy; a constant "
             "per_block_quota equal to the fixed quota means the rule did nothing here")


def joint_compare(samples, threshold, v17_meta=None, honesty=None):
    """The summary. `v17_meta` / `honesty` are appended blocks, never replacements.

    `v17_meta` records what the checkpoint itself carried (`value_ratio_effective`,
    `layer_norm_weights`, the band layout) and `honesty` the two warnings of (E), so the
    numbers in this summary cannot be read without the fields that qualify them.
    """
    graded = [sample for sample in samples if sample.get("status") == "graded"]
    located = [sample for sample in graded if sample.get("evidence", {}).get("located")]
    dropped = [sample for sample in located if _evidence_in_dropped(sample)]
    kept = [sample for sample in located if not _evidence_in_dropped(sample)]
    present = [arm for arm in ARMS if arm in graded[0]["arms"]]
    # The permutation arms are named after their seed, so they are not in the static
    # ARMS tuple; they must still be summarised, damaged and paired like every other arm.
    extra = sorted({arm for sample in graded for arm in sample["arms"]} - set(ARMS))
    present += [arm for arm in extra if arm in graded[0]["arms"]]
    controls = [arm for arm in (V11_DONOR, V11_EVICT, V11_EVICT_DONOR, V14, V14_DONOR,
                                V15, V15_DONOR, V16, V16_DONOR, V13_PLUS,
                                V13_PLUS_DONOR, GRAFT) + DEEP_ARMS + DENSITY_ARMS
                + tuple(extra)
                if arm in present]
    # The V17 arms are all calibrated on V13's window (the density line on its own), so
    # they are paired against V13 like every other module arm; the derived arms are also
    # paired against V17 itself, which is what makes each knob's worth a number.
    v17_derived = [arm for arm in (V17_NO_BAND, V17_NO_ATTN, V17_ATTN) + tuple(
        arm for arm in extra if arm.startswith(VALUE_PREFIX)) if arm in present]

    def block(rows):
        return dict(
            n=len(rows),
            arms={arm: gr._arm_summary(rows, arm, threshold) for arm in present},
            versus_anchor_only={arm: er.pair_summary(rows, arm, ANCHOR_ONLY, threshold)
                                for arm in controls},
            versus_v11={arm: er.pair_summary(rows, arm, V11, threshold)
                        for arm in controls if arm in (V11_DONOR, V11_EVICT)},
            versus_v11evict={V11_EVICT_DONOR: er.pair_summary(
                rows, V11_EVICT_DONOR, V11_EVICT, threshold)}
            if V11_EVICT_DONOR in present else {},
            # The one comparison this round exists for: the same window, the same
            # modules' training budget, only the neighbour source differs.
            versus_v13={arm: er.pair_summary(rows, arm, V11_EVICT, threshold)
                        for arm in (V14, V15, V16, V13_PLUS) + DEEP_ARMS
                        if arm in present},
            # `V17 - V17NoBand` is the band gate, `V17ValueXX - V17` is the value ceiling,
            # `V17Attn - V17` is the graft damping: one knob each, nothing else moved.
            versus_v17={arm: er.pair_summary(rows, arm, V17, threshold)
                        for arm in v17_derived} if V17 in present else {},
            # The density line, each arm against ITS OWN fixed-quota counterpart on the
            # same questions: V13Density - V13 is the quota rule, V17Density - V17 is the
            # module read at a different quota, V17Density - V13Density is the module
            # again on the density window.
            versus_fixed_quota={
                V13_DENSITY: er.pair_summary(rows, V13_DENSITY, V11_EVICT, threshold),
                V17_DENSITY: er.pair_summary(rows, V17_DENSITY, V17, threshold),
            } if (V13_DENSITY in present and V17 in present) else (
                {V13_DENSITY: er.pair_summary(rows, V13_DENSITY, V11_EVICT, threshold)}
                if V13_DENSITY in present and V11_EVICT in present else {}),
            versus_density={V17_DENSITY: er.pair_summary(rows, V17_DENSITY, V13_DENSITY,
                                                         threshold)}
            if (V17_DENSITY in present and V13_DENSITY in present) else {},
            versus_v14={V14_DONOR: er.pair_summary(rows, V14_DONOR, V14, threshold)}
            if V14_DONOR in present else {},
            versus_v15={V15_DONOR: er.pair_summary(rows, V15_DONOR, V15, threshold)}
            if V15_DONOR in present else {},
            versus_v16={V16_DONOR: er.pair_summary(rows, V16_DONOR, V16, threshold)}
            if V16_DONOR in present else {},
            versus_v13_plus={V13_PLUS_DONOR: er.pair_summary(rows, V13_PLUS_DONOR,
                                                             V13_PLUS, threshold)}
            if V13_PLUS_DONOR in present else {})

    merge_rows = [sample["v11"] for sample in graded]
    thresholds = dict(
        total=dict(target=62, baseline=_count(graded, ANCHOR_ONLY, threshold),
                   achieved=_count(graded, V11, threshold),
                   # `achieved` above keeps the V11 driver's meaning (the plain
                   # calibration); the treatment counts live here so a V13/V14 run does
                   # not silently report 61 as its own result.
                   achieved_by_arm={arm: _count(graded, arm, threshold)
                                    for arm in (V11, V11_EVICT, V14) if arm in present}),
        dropped_evidence=dict(target=19, n=len(dropped),
                              baseline=_count(dropped, ANCHOR_ONLY, threshold),
                              achieved=_count(dropped, V11, threshold),
                              achieved_by_arm={arm: _count(dropped, arm, threshold)
                                               for arm in (V11, V11_EVICT, V14)
                                               if arm in present}),
        kept_evidence=dict(target=12, n=len(kept),
                           baseline=_count(kept, ANCHOR_ONLY, threshold),
                           achieved=_count(kept, V11, threshold),
                           achieved_by_arm={arm: _count(kept, arm, threshold)
                                            for arm in (V11, V11_EVICT, V14)
                                            if arm in present}),
        content_causality=dict(
            target=5,
            achieved=er.pair_summary(graded, V11, V11_DONOR, threshold).get("net")
            if V11_DONOR in present else None),
        cross_over_plain=dict(
            target=5,
            achieved=er.pair_summary(graded, V11_EVICT, V11, threshold).get("net")
            if V11_EVICT in present else None),
        evict_content_causality=dict(
            target=5,
            achieved=er.pair_summary(graded, V11_EVICT, V11_EVICT_DONOR,
                                     threshold).get("net")
            if V11_EVICT_DONOR in present else None),
        # V14's own three bars, stated before the run: complement over top-k, content
        # causality, and a donor that must not win.
        complement_over_topk=dict(
            target=4,
            achieved=er.pair_summary(graded, V14, V11_EVICT, threshold).get("net")
            if V14 in present else None),
        v14_content_causality=dict(
            target=5,
            achieved=er.pair_summary(graded, V14, V14_DONOR, threshold).get("net")
            if V14_DONOR in present else None),
        # The V15 targets, stated before the run: +6 questions over V13 (73/158) and
        # damage below 8 of Full's 47.
        v15_over_v13=dict(
            target=6,
            achieved=er.pair_summary(graded, V15, V11_EVICT, threshold).get("net")
            if V15 in present else None),
        v15_content_causality=dict(
            target=5,
            achieved=er.pair_summary(graded, V15, V15_DONOR, threshold).get("net")
            if V15_DONOR in present else None),
        v15_damage=dict(
            target=8,
            achieved=(ef._damage(graded, V15, threshold).get("lost")
                      if V15 in present else None)),
        # V16's targets, stated before the run: +5 questions over V13 (72/158), damage
        # down to 8-9 of Full's 47, and content causality kept.
        v16_over_v13=dict(
            target=5,
            achieved=er.pair_summary(graded, V16, V11_EVICT, threshold).get("net")
            if V16 in present else None),
        v16_content_causality=dict(
            target=5,
            achieved=er.pair_summary(graded, V16, V16_DONOR, threshold).get("net")
            if V16_DONOR in present else None),
        v13_plus_over_v13=dict(
            target=5,
            achieved=er.pair_summary(graded, V13_PLUS, V11_EVICT, threshold).get("net")
            if V13_PLUS in present else None),
        v13_plus_content_causality=dict(
            target=5,
            achieved=er.pair_summary(graded, V13_PLUS, V13_PLUS_DONOR,
                                     threshold).get("net")
            if V13_PLUS_DONOR in present else None),
        v13_plus_damage=dict(
            target=9,
            achieved=(ef._damage(graded, V13_PLUS, threshold).get("lost")
                      if V13_PLUS in present else None)),
        v16_damage=dict(
            target=9,
            achieved=(ef._damage(graded, V16, threshold).get("lost")
                      if V16 in present else None)),
        # V17's bars. No number was fixed before the run for these (the plan's 46/158 is
        # below the V13 baseline of 67/158), so the targets are stated as `None` and the
        # reading is left to the paired table: `target=None` means "no bar was declared",
        # never "the bar was met".
        v17_over_v13=dict(
            target=None, bar="no bar was declared before the run; V13 = 67/158",
            achieved=er.pair_summary(graded, V17, V11_EVICT, threshold).get("net")
            if V17 in present else None),
        v17_damage=dict(
            target=None, bar="reported, not tested against a number",
            achieved=(ef._damage(graded, V17, threshold).get("lost")
                      if V17 in present else None)),
        # The three inference-time knobs, each against the checkpoint's own arm.
        v17_band_ablation=dict(
            target=None, bar="`V17 - V17NoBand`: what the three-band gate is worth",
            achieved=er.pair_summary(graded, V17_NO_BAND, V17, threshold).get("net")
            if (V17_NO_BAND in present and V17 in present) else None),
        # The value ratios and V17Attn are read in their own direction (`arm - V17`,
        # negative = the knob costs questions); mechanism four is read the other way
        # round, because `V17NoAttn` is the arm WITHOUT it: `V17 - V17NoAttn` is what the
        # damping is worth.
        v17_attention_off=dict(
            target=None,
            bar="`V17 - V17NoAttn`: what graft attention damping (mechanism four) is "
                "worth, read on the checkpoint that carries it",
            achieved=er.pair_summary(graded, V17, V17_NO_ATTN, threshold).get("net")
            if (V17_NO_ATTN in present and V17 in present) else None),
        v17_attention=dict(
            target=None, bar="`V17Attn - V17`: what graft-only key damping is worth",
            achieved=er.pair_summary(graded, V17_ATTN, V17, threshold).get("net")
            if (V17_ATTN in present and V17 in present) else None),
        v17_value_ratio=dict(
            target=None,
            bar="each V17ValueXX against V17: the value ceiling, weights untouched",
            value_ratio_effective=((v17_meta or {}).get("value_ratio_effective")),
            value_ratio_effective_source=((v17_meta or {}).get(
                "value_ratio_effective_source")),
            baseline_arm=((v17_meta or {}).get("value_ratio_baseline_arm")),
            achieved={arm: er.pair_summary(graded, arm, V17, threshold).get("net")
                      for arm in (arm for arm in extra
                                  if arm.startswith(VALUE_PREFIX) and arm in present)}
            if V17 in present else None),
        # The density line: the quota rule and the module, one at a time.
        density_quota=dict(
            target=None,
            bar="V13Density - V13 and V17Density - V17: the per-block quota rule alone",
            achieved=dict(
                **({V13_DENSITY: er.pair_summary(graded, V13_DENSITY, V11_EVICT,
                                                 threshold).get("net")}
                   if V13_DENSITY in present and V11_EVICT in present else {}),
                **({V17_DENSITY: er.pair_summary(graded, V17_DENSITY, V17,
                                                 threshold).get("net")}
                   if V17_DENSITY in present and V17 in present else {})) or None),
        density_module=dict(
            target=None,
            bar="V17Density - V13Density: the module, on the density window",
            achieved=er.pair_summary(graded, V17_DENSITY, V13_DENSITY,
                                     threshold).get("net")
            if (V17_DENSITY in present and V13_DENSITY in present) else None))

    return dict(
        definition=("the same 19-block window, the same nine carriers, the same 2,304 "
                    "slots, the same 158 rows; each arm moves one variable: the "
                    "selection rule, the capacity split, the calibration, the neighbour "
                    "source, or whose content the graft slots hold"),
        arms_present=present,
        n=len(graded),
        arms={arm: gr._arm_summary(graded, arm, threshold) for arm in present},
        total=block(graded), dropped_evidence_group=block(dropped),
        kept_evidence_group=block(kept),
        damage={arm: ef._damage(graded, arm, threshold) for arm in present},
        permutation=permutation_report(
            graded, threshold,
            [int(arm[len(PERMUTE_PREFIX):]) for arm in present
             if arm.startswith(PERMUTE_PREFIX)]) if any(
                arm.startswith(PERMUTE_PREFIX) for arm in present) else None,
        relation_gate=_relation_gate(graded),
        band_gate=_band_gate(graded),
        neighbour_composition=dict(
            topk=_neighbour_composition(graded, "evict"),
            complement=_neighbour_composition(graded, "v14"),
            note="topk is V13's kind-blind top-4; complement is V14's 2 + 2. "
                 "neighbour_type_cross_fraction is the share of selected neighbours "
                 "from the other side of the core/graft divide, and cross_weight_mass "
                 "is the share of context weight they received"),
        thresholds=thresholds,
        merge_health=dict(
            merged_pairs=_stats([row["merged_pairs"] for row in merge_rows]),
            folded_slots=_stats([row["folded_slots"] for row in merge_rows]),
            blocks_with_no_merge=_stats([row["blocks_with_no_merge"]
                                         for row in merge_rows]),
            blocks_where_quota_came_from_eviction=_stats(
                [row["blocks_where_quota_came_from_eviction"] for row in merge_rows]),
            mean_cluster_size=_stats([row["mean_cluster_size"] for row in merge_rows]),
            novelty_mean=_stats([row["novelty_mean"] for row in merge_rows])),
        calibration=dict(
            core=_stats([sample["v11_calibration"]["core_relative_delta"]
                         for sample in graded]),
            graft=_stats([sample["v11_calibration"]["graft_relative_delta"]
                          for sample in graded]),
            gate=_stats([sample["v11_calibration"]["gate"] for sample in graded])),
        v17=_v17_report(graded),
        # Appended, never folded into `v17`: what the checkpoint carried and what the run
        # cannot claim. The three flat keys are the ones a reader greps for; they are
        # assigned from the same dicts so the two copies cannot disagree.
        v17_checkpoint=v17_meta,
        v17_value_ratio_effective=(v17_meta or {}).get("value_ratio_effective"),
        v17_value_ratio_effective_source=(v17_meta or {}).get(
            "value_ratio_effective_source"),
        honesty=honesty,
        sim_floor_filters_nothing=(((honesty or {}).get("sim_floor") or {}).get(
            "filters_nothing")),
        band0_observable=(((honesty or {}).get("band") or {}).get("band0_observable")),
        gate_parameters_live=(((honesty or {}).get("band") or {}).get(
            "gate_parameters_live")),
        window_density=_window_density(graded))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def build_protocol(args, task, anchor_config, v11_geometry, evict_geometry,
                   v14_geometry=None, permute_seeds=None, v15_geometry=None,
                   v16_geometry=None, v17_geometry=None, v17_attention=None,
                   v17_value_scan=None, attention_decision=None, v17_meta=None):
    return dict(
        experiment="10 (V14): does the SOURCE of the cross-block neighbours buy anything? "
                   "Same window, same budget, same training as V13; only the rule that "
                   "picks the 4 neighbours changes -- kind-blind top-k versus 2 of the "
                   "query's own core/graft kind + 2 from the other side.",
        treatment=("every slot is calibrated against a sparse set of OTHER blocks' slots: "
                   "V13 takes the k = 4 most associated by reader-key cosine; V14 takes "
                   "k_same = 2 of the query's own kind and k_cross = 2 from the other "
                   "side of the core/graft divide, softmaxed per group with equal group "
                   "mass. The context vector is injected into the residual bottleneck by "
                   "a zero-initialised projection; the slot itself is never blended, and "
                   "the same bounded low-rank map moves it, with a tighter ceiling on "
                   "carrier slots (0.15 x rms) than on graft slots (0.50 x rms)"),
        task=task, data_dir=args.data_dir,
        arms=list(ARMS) + [value_arm(ratio) for ratio
                           in (getattr(args, "v17_value_ratios", None) or [])],
        window_blocks=args.window_blocks, target_blocks=args.target_blocks,
        block_tokens=BLOCK_TOKENS, slots_per_block=SLOTS_PER_BLOCK,
        budget=TARGET_SLOTS,
        baseline=dict(arm=ANCHOR_ONLY, quota=BASELINE_QUOTA,
                      core_total=args.target_blocks * BASELINE_QUOTA["core_per_block"],
                      graft_total=args.target_blocks * BASELINE_QUOTA["graft_per_block"],
                      anchor=args.anchor, anchor_sha256=sha(args.anchor),
                      anchor_config=anchor_config),
        v11=dict(arm=V11, path=args.v11, sha256=sha(args.v11), geometry=v11_geometry,
                 core_total=CORE_TOTAL, graft_total=GRAFT_TOTAL),
        band_relation=dict(
            arm=V16, path=args.v16, sha256=sha(args.v16) if args.v16 else None,
            geometry=v16_geometry,
            treatment=("one cross-block top-4 candidate set chosen from the five-layer "
                       "CONSENSUS similarity (keys centred over slots, L2-normalised); "
                       "each band then softmaxes its own similarity over those four and "
                       "aggregates its own raw values; memory layer l reads band "
                       "BAND_OF_LAYER[l] only (L12->L10-14, L16->L15-18, L20->L19-22, "
                       "L24->L23-26, L28->L27-31, L0-9 self branch only), through "
                       "band_proj[band] + layer_proj[layer], both zero-initialised; the "
                       "injection is gated AFTER the projection by "
                       "omega = sigmoid(kappa[type,band]*(margin - tau[type,band])) and "
                       "eta = sigmoid(eta_logit[type,layer]), with nothing normalising "
                       "the product afterwards"),
            targets=dict(questions="V16 >= 72/158 (+5 over V13)",
                         damage="8-9 of Full's 47 lost",
                         content="V16 - V16Donor keeps a clear margin")),
        relation=dict(
            arm=V15, path=args.v15, sha256=sha(args.v15) if args.v15 else None,
            geometry=v15_geometry,
            treatment=("five INDEPENDENT top-4 relations, one per utility layer "
                       "(12/16/20/24/28), each softmaxed with a per-layer sink logit that "
                       "is dropped afterwards so the retained mass can be below 1; the "
                       "feature is the neighbours minus the slot (sum_j w_ij v_j - m_i "
                       "v_i); it enters as a FiLM modulation, b * (1 + tanh(W_s c)) + "
                       "tanh(W_h c), on an RMS-normalised sum of five per-layer "
                       "projections; memory layers 0..9 keep the self branch only"),
            targets=dict(questions="V15 >= 73/158 (+6 over V13)",
                         damage="at most 8 of Full's 47 lost")),
        permutation=dict(
            arms=[permute_arm(seed) for seed in (permute_seeds or [])],
            seeds=list(permute_seeds or []),
            treatment=("inference time only, no retraining: V13's precomputed "
                       "cross-block context rows are shuffled inside (source_block, "
                       "slot_type) groups, so each slot draws from the same batch of "
                       "neighbours it could draw from before and only the slot -> row "
                       "pairing changes"),
            not_done=("random top-4 neighbours, which would change the marginal "
                      "distribution of the context as well as the pairing")),
        controls=dict(
            donor=dict(arm=V11_DONOR,
                       definition="the plain calibration's window with the 684 graft "
                                  "slots taken from the previous row's document"),
            cross_donor=dict(arm=V11_EVICT_DONOR,
                             definition="V13's window with the 684 graft slots taken "
                                        "from the previous row's document: the only "
                                        "thing that changes is whose content the "
                                        "calibrator is conditioned on"),
            complement=dict(arm=V14, path=args.v14,
                            sha256=sha(args.v14) if args.v14 else None,
                            geometry=v14_geometry,
                            definition="V13's window calibrated by the complement-mode "
                                       "module: identical slots, identical graft "
                                       "content, only the neighbour source differs"),
            complement_donor=dict(arm=V14_DONOR,
                                  definition="V14 on the donor window: the complement "
                                             "arm's own content-causality control"),
            evict=dict(arm=V11_EVICT, path=args.v11_evict,
                       sha256=sha(args.v11_evict) if args.v11_evict else None,
                       geometry=evict_geometry,
                       definition="same selection and calibration, carriers cut to 1,620 "
                                  "by plain eviction instead of folding")),
        deep_fusion=dict(
            arm=V17, path=getattr(args, "v17", None),
            sha256=sha(args.v17) if getattr(args, "v17", None) else None,
            geometry=v17_geometry,
            treatment=("the slot residual is split by the key map: the key-visible part "
                       "is capped at key_ratio x rms, the part the key cannot see (the "
                       "value side) at value_ratio x rms, so a slot may move as far as "
                       "the values need while its keys stay anchored; `band_mode='three'` "
                       "additionally gates the three layer bands (lexical 0-10, bridging "
                       "11-21, readout 22-31), band 0 held at exactly zero"),
            arms=dict(
                checkpoint=V17,
                value_ratios={value_arm(ratio): float(ratio)
                              for ratio in (getattr(args, "v17_value_ratios", None) or [])},
                # The ceiling the checkpoint really clips at, and whether it had to be
                # added to the sweep because the user's list did not contain it.
                value_ratio_effective=(v17_value_scan or {}).get("value_ratio_effective"),
                value_ratio_effective_source=(v17_value_scan or {}).get(
                    "value_ratio_effective_source"),
                value_ratio_baseline=(v17_value_scan or {}).get("baseline_arm"),
                value_ratio_baseline_listed=(v17_value_scan or {}).get("baseline_listed"),
                band_off=V17_NO_BAND if getattr(args, "v17_band_off", False) else None,
                attention=V17_ATTN if getattr(args, "v17_attention", False) else None,
                # Mechanism four OFF. This is the only arm in which a checkpoint that
                # damps inside `deep_fusion.calibrate` runs undamped.
                no_attention=V17_NO_ATTN if getattr(args, "v17_no_attn", False) else None),
            inference_time_only=[V17_NO_BAND, V17_NO_ATTN, V17_ATTN],
            attention_decision=attention_decision,
            layer_norm_weights=(v17_meta or {}).get("layer_norm_weights"),
            graft_attention=v17_attention,
            bars=("no bar was fixed before the run: the plan's 46/158 is BELOW the V13 "
                  "baseline of 67/158, so every V17 comparison is read from the paired "
                  "table and from `window_density`, not against a number"),
            note=("the value-ceiling sweep and the ablation arms deep-copy the module and "
                  "replace ONE runtime attribute; no weight is rewritten and the "
                  "checkpoint on disk is never touched")),
        window_density=dict(
            enabled=bool(getattr(args, "v17_window_density", False)),
            arms=([V13_DENSITY] + ([V17_DENSITY] if getattr(args, "v17", None) else [])
                  if getattr(args, "v17_window_density", False) else []),
            quota_mode=("density" if getattr(args, "v17_window_density", False)
                        else "fixed"),
            min_core=140, max_core=256, absorb_cos=float(
                getattr(args, "v17_merge_cos", 0.0) or 0.0),
            treatment=("the same 2,304 slots and the same 1,620 core slots; only the "
                       "per-block allocation changes -- redundancy-weighted, bounded to "
                       "[min_core, max_core] per carrier block, summing to exactly 1,620 "
                       "-- and, when --v17-merge-cos > 0, an evicted slot that is nearly "
                       "parallel to a retained one is absorbed into it instead of dropped"),
            controls=dict(fixed_quota=dict(arm=V11_EVICT,
                                           definition="the same module on the fixed-quota "
                                                      "window, on the same questions"),
                          fixed_quota_v17=dict(arm=V17,
                                               definition="V17 on the fixed-quota window")),
            telemetry="the builder's quota_mode / per_block_quota / density / absorbed_* "
                      "are carried per row and aggregated under summary.window_density"),
        unchanged_conditions=dict(
            data="the same 158 HotpotQA rows", reader="frozen MemoryLLM-8B",
            carriers="the same 9, selected once and shared by every arm",
            window="the same 19-block write, document prefix only",
            capacity="2,304 slots in every compressed arm",
            transport="one selection shared by all 32 layers; chronological output",
            decoding="greedy, unchanged"),
        checks=dict(
            reproduction="Full, Selection, Graft and AnchorOnly must reproduce the "
                         "reference report row for row",
            capacity="every V11 arm holds 2,304 slots split 1,620 / 684",
            provenance="every graft slot from a dropped block, verbatim; every core "
                       "slot from a carrier",
            folding="unfolded slots are verbatim copies; folded slots keep their "
                    "representative's per-layer RMS",
            calibration="both halves must move, and the RMS restore must hold",
            cross_block="the context must be built from other blocks: the neighbour "
                        "block spread is recorded per row",
            donor="the donor arm's window must differ from the true one",
            neighbour_source="--v14 must be complement mode and --v11-evict top-k mode "
                             "with the same k, so the two arms differ in the neighbour "
                             "source alone; V14's calibrated window must differ from "
                             "V13's",
            permutation="the permuted arms must change the cross-block context (a "
                        "permutation that moves nothing is refused), must differ from "
                        "V13 in answers or the arm is refused, and must still hold the "
                        "2,304-slot budget; the self-calibration branch and every weight "
                        "stay untouched",
            deep_fusion="--v17 must hold the 2,304-slot budget; a checkpoint whose every "
                        "knob is at its default is refused as V13 under another name; a "
                        "knob that leaves the calibrated window identical is WARNED about "
                        "and recorded per row (`window_equals_v17`), so a null result "
                        "cannot be read as a gain",
            graft_attention="a checkpoint whose own tau / bias / norm_quantile are "
                            "non-neutral damps the window inside deep_fusion.calibrate, so "
                            "`--v17-attention` is SKIPPED on it (it would be V17 with the "
                            "damping applied twice) and `--v17-no-attn` is the arm that "
                            "turns mechanism four off; the checkpoint's "
                            "`layer_norm_weights` are read back into the module (a length "
                            "mismatch is an error, not a silent drop)",
            value_ratio_sweep="the sweep always contains `value_ratio_effective`: the "
                              "ceiling the checkpoint really clips at is `value_ratio` "
                              "when the module sets one, `max_ratio` (0.34 by default) "
                              "when it is None, and the arm for it is added if the user's "
                              "list omits it",
            window_density="the density window must still hold 2,304 slots with 1,620 "
                           "core, every unabsorbed slot must remain a verbatim copy of "
                           "its source, and the per-block quota is counted from the "
                           "window and cross-checked against the builder's telemetry "
                           "(a mismatch is an error, not a footnote)"),
        thresholds=dict(
            total="V11 >= 62/158 against 56", dropped_evidence="V11 >= 19/40 against 15",
            kept_evidence="V11 >= 12/34 against 13",
            damage="correct rows lost by Full <= 13",
            content_causality="V11 - V11Donor >= 5 questions",
            complement_over_topk="V14 - V13 >= 4 questions net",
            v14_content_causality="V14 - V14Donor >= 5 questions",
            v17_over_v13="no number was fixed before the run (the plan's 46/158 is below "
                         "the V13 baseline of 67/158); read V17 - V13, V17 - V17NoBand, "
                         "V17 - V17NoAttn (mechanism four), V17ValueXX - V17 and "
                         "V17Attn - V17 as paired differences",
            window_density="V13Density - V13 and V17Density - V17 are the quota rule; "
                           "V17Density - V13Density is the module on the density window"),
        screening_caveat=gr.SCREENING_CAVEAT, f1_threshold=args.f1_threshold,
        seed=args.seed, query=gr.QUERY_SUFFIX,
        scoring="official HotpotQA QA F1, F1 >= threshold is correct",
        reader_utility_schema=JOINT_SCHEMA)


def _net_summary(total):
    """Every paired comparison of the run as `arm -> net`, duplicates collapsed.

    An arm can appear in several blocks (V11EvictDonor is in versus_anchor_only and in
    versus_v11evict; every V17 arm is in versus_v13 and the derived ones are also in
    versus_v17). That is why this is an UPDATE and not `dict(**a, **b)`: duplicate keys
    are a `TypeError` there, and that killed one finished run at the printing step. The
    later block wins, and the blocks are ordered so the most specific comparison is the
    one left in the flat map.
    """
    net = {}
    for name in ("versus_anchor_only", "versus_v11", "versus_v11evict", "versus_v13",
                 "versus_v17", "versus_fixed_quota", "versus_density",
                 "versus_v14", "versus_v15", "versus_v16", "versus_v13_plus"):
        for arm, value in (total.get(name) or {}).items():
            net[arm] = value.get("net")
    return net


REFERENCE_ARMS = ((FULL, FULL), (SELECTION, SELECTION), (GRAFT, GRAFT),
                  (ANCHOR_ONLY, ANCHOR_ONLY))


def verify_against_report(args, reference, table, samples):
    """The four pre-existing arms must reproduce the reference report row for row."""
    this_run = {str(sample["id"]) for sample in samples}
    verdict = dict(reference=args.verify_against,
                   reference_schema=reference.get("schema"),
                   reference_samples=len(table), compared=len(this_run),
                   row_count_matches_reference=(len(this_run) == len(table)),
                   reference_not_in_this_run=sorted(set(table) - this_run),
                   this_run_not_in_reference=sorted(this_run - set(table)),
                   matched={reference_arm: ours for reference_arm, ours in REFERENCE_ARMS})
    passed = not verdict["this_run_not_in_reference"]
    for reference_arm, ours in REFERENCE_ARMS:
        same = different = absent = 0
        differing = []
        for sample in samples:
            old = table.get(str(sample["id"]))
            if old is None or reference_arm not in old.get("arms", {}):
                absent += 1
                continue
            if old["arms"][reference_arm]["prediction"] == sample["arms"][ours]["prediction"]:
                same += 1
            else:
                different += 1
                differing.append(dict(id=sample["id"],
                                      old=old["arms"][reference_arm]["prediction"],
                                      new=sample["arms"][ours]["prediction"]))
        verdict[reference_arm] = dict(matched_to=ours, identical=same, different=different,
                                      not_in_reference=absent,
                                      differing=differing[:20])
        passed = passed and different == 0 and absent == 0 and same == len(this_run)
    verdict["passed"] = passed
    return verdict


def evaluate(args):
    facts = gr.load_supporting_facts(args.supporting_facts_glob)
    if not facts:
        print("WARNING: no supporting facts; the evidence grouping will be empty.",
              file=sys.stderr)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    if args.dry_run:
        rows, prepared, excluded = gr.prepare_rows(args, tokenizer, facts)
        print(json.dumps(dict(
            dry_run=True, out=args.out,
            counts=dict(rows=len(rows), eligible=len(prepared), excluded=len(excluded)),
            arms=(list(ARMS) + [permute_arm(seed) for seed in args.permute_seeds]
                  + [value_arm(ratio)
                     for ratio in parse_ratios(args.v17_value_ratios)]),
            readouts_per_sample=(len(ARMS) + len(args.permute_seeds)
                                 + len(parse_ratios(args.v17_value_ratios))),
            v11=args.v11, v11_evict=args.v11_evict, v14=args.v14, v15=args.v15,
            v16=args.v16, v13_plus=args.v13_plus,
            v17=args.v17, v17_value_ratios=parse_ratios(args.v17_value_ratios),
            v17_band_off=args.v17_band_off, v17_no_attn=args.v17_no_attn,
            v17_attention=args.v17_attention,
            # The sweep's baseline arm can only be named once the checkpoint is loaded,
            # so the dry run reports that it will be added rather than inventing its name.
            v17_value_ratio_baseline=("added at run time from the checkpoint's "
                                      "value_ratio_effective if the list omits it"
                                      if args.v17 and args.v17_value_ratios else None),
            v17_window_density=args.v17_window_density,
            v17_merge_cos=args.v17_merge_cos,
            permute_seeds=args.permute_seeds), indent=2,
            ensure_ascii=False))
        return None
    for flag, value, what in (("--checkpoint", args.checkpoint, "the frozen operator"),
                              ("--anchor", args.anchor, "the AnchorOnly module"),
                              ("--v11", args.v11, "the trained V11 module")):
        if not value or not Path(value).is_file():
            raise ValueError(f"this experiment needs {what}: pass {flag}")
    output = Path(args.out)
    if output.exists():
        raise ValueError(f"{output} exists; delete it first or choose another --out")

    nv.seed_everything(args.seed)
    checkpoint_training = torch.load(args.checkpoint, map_location="cpu").get("training")
    model, tokenizer, manager, merge, initial, cfg, saved = load_runtime(
        args.model_path, checkpoint=args.checkpoint, train=False, architecture=None,
        budget_blocks=gr.BUDGET_BLOCKS, strict_code_hash=args.strict_code_hash,
        training=checkpoint_training)
    merge.operator_mode = "native"
    anchor_module, anchor_config = load_anchor_module(args.anchor, manager.hidden)
    anchor_module = anchor_module.to(model.memory.device)

    v11_module, v11_geometry, v11_config, _ = load_v11(args.v11, manager.hidden)
    v11_module = v11_module.to(model.memory.device).float()
    scorer = ReaderScorer(model, tuple(int(x) for x in v11_geometry["utility_layers"]))
    probe_cache = {}
    v11 = dict(module=v11_module, geometry=v11_geometry, config=v11_config,
               joint=joint_for(v11_geometry, scorer, probe_cache, manager.hidden))
    v11_evict = None
    if args.v11_evict:
        evict_module, evict_geometry, evict_config, _ = load_v11(args.v11_evict,
                                                                 manager.hidden)
        evict_module = evict_module.to(model.memory.device).float()
        if not evict_geometry.get("cross"):
            raise ValueError("--v11-evict is not a cross-block conditioned checkpoint: "
                             "the arm that is supposed to test the new mechanism would "
                             "run the old one")
        if v11_geometry.get("cross"):
            raise ValueError("--v11 is cross-block conditioned: it must be the plain "
                             "calibration the treatment is compared against")
        v11_evict = dict(module=evict_module, geometry=evict_geometry,
                         config=evict_config,
                         joint=joint_for(evict_geometry, scorer, probe_cache,
                                         manager.hidden))
    args.graft_scales = parse_scales(args.v13_graft_scales)
    args.relevance_cap = str(args.v13_relevance_cap or "").lower() in ("on", "true", "1")
    if (args.graft_scales or args.relevance_cap) and v11_evict is None:
        raise ValueError("--v13-graft-scales / --v13-relevance-cap need --v11-evict: the "
                         "ceiling belongs to the V13 checkpoint")
    args.v13_neighbor_modes = [mode.strip() for mode
                               in str(args.v13_neighbor_modes or "").split(",")
                               if mode.strip()]
    for mode in args.v13_neighbor_modes:
        if mode not in NEIGHBOR_ARMS:
            raise ValueError(f"--v13-neighbor-modes takes {sorted(NEIGHBOR_ARMS)}, "
                             f"got {mode!r}")
        if v11_evict is None:
            raise ValueError("--v13-neighbor-modes needs --v11-evict: the neighbours "
                             "belong to the V13 checkpoint")
    args.permute_seeds = parse_seeds(args.v13_permute_seeds)
    if args.permute_seeds:
        if v11_evict is None:
            raise ValueError("--v13-permute-seeds needs --v11-evict: the correspondence "
                             "that is being permuted belongs to the V13 checkpoint")
        if v11_evict["module"].neighbor_mode != "topk":
            raise ValueError("the permutation test is defined on V13's kind-blind top-k "
                             "context; this checkpoint carries neighbour mode "
                             f"{v11_evict['module'].neighbor_mode!r}")
    v13_plus = None
    if args.v13_plus:
        if v11_evict is None:
            raise ValueError("--v13-plus needs --v11-evict: that is the arm to beat")
        plus_module, plus_geometry, plus_config, _ = load_v11(args.v13_plus,
                                                              manager.hidden)
        plus_module = plus_module.to(model.memory.device).float()
        if not plus_geometry.get("cross"):
            raise ValueError("--v13-plus is not a cross-block conditioned checkpoint")
        for field in ("core_per_block", "graft_per_block"):
            if plus_geometry.get(field) != v11_evict["geometry"].get(field):
                raise ValueError(f"--v13-plus and --v11-evict record different {field}: "
                                 f"they cannot share one window")
        v13_plus = dict(module=plus_module, geometry=plus_geometry,
                        config=plus_config)
    v16 = None
    if args.v16:
        if v11_evict is None:
            raise ValueError("--v16 needs --v11-evict: V13 is the arm V16 has to beat")
        v16_module, v16_config, v16_geometry = br.load_band(args.v16, manager.hidden)
        v16_module = v16_module.to(model.memory.device).float().eval()
        v16_module.requires_grad_(False)
        for field in ("core_per_block", "graft_per_block"):
            if v16_geometry.get(field) != v11_evict["geometry"].get(field):
                raise ValueError(f"--v16 and --v11-evict record different {field}: they "
                                 f"cannot share one window")
        if int(v16_config.get("rank", -1)) != int(v11_evict["config"].get("rank", -1)):
            raise ValueError("--v16 and --v11-evict were built with different ranks")
        v16 = dict(module=v16_module, config=v16_config, geometry=v16_geometry)
    v15 = None
    if args.v15:
        if v11_evict is None:
            raise ValueError("--v15 needs --v11-evict: V13 is the arm V15 has to beat")
        v15_module, v15_config, v15_geometry = hr.load_relation(args.v15, manager.hidden)
        v15_module = v15_module.to(model.memory.device).float().eval()
        v15_module.requires_grad_(False)
        if not torch.isfinite(v15_module.sink_similarity).all():
            raise ValueError("--v15 carries no measured sink threshold: the checkpoint "
                             "was written before init_sink, and the gate it is supposed "
                             "to test would be undefined")
        for field in ("core_per_block", "graft_per_block"):
            if v15_geometry.get(field) != v11_evict["geometry"].get(field):
                raise ValueError(f"--v15 and --v11-evict record different {field}: they "
                                 f"cannot share one window")
        v15 = dict(module=v15_module, config=v15_config, geometry=v15_geometry)
    v14 = None
    if args.v14:
        if v11_evict is None:
            raise ValueError("--v14 needs --v11-evict: the top-k arm is the baseline the "
                             "complement arm is measured against")
        v14_module, v14_geometry, v14_config, _ = load_v11(args.v14, manager.hidden)
        v14_module = v14_module.to(model.memory.device).float()
        if not v14_geometry.get("cross"):
            raise ValueError("--v14 is not a cross-block conditioned checkpoint")
        if v14_module.neighbor_mode != "complement":
            raise ValueError(f"--v14 carries neighbour mode "
                             f"{v14_module.neighbor_mode!r}: a top-k checkpoint cannot "
                             f"test the complement rule")
        if v11_evict["module"].neighbor_mode != "topk":
            raise ValueError("--v11-evict carries neighbour mode "
                             f"{v11_evict['module'].neighbor_mode!r}: V13 and V14 must "
                             f"differ in the neighbour rule, so the reference arm has to "
                             f"be the kind-blind top-k one")
        if int(v14_module.k_neighbors) != int(v11_evict["module"].k_neighbors):
            raise ValueError(f"the two arms read {v14_module.k_neighbors} and "
                             f"{v11_evict['module'].k_neighbors} neighbours: the "
                             f"comparison has to hold k fixed")
        for field in ("core_per_block", "graft_per_block", "utility_layers"):
            if v14_geometry.get(field) != v11_evict["geometry"].get(field):
                raise ValueError(f"--v14 and --v11-evict record different {field}: they "
                                 f"cannot share one window")
        v14 = dict(module=v14_module, geometry=v14_geometry, config=v14_config)

    # V17: key anchoring + value release, on V13's window. Everything that needs the
    # reader's key map is built here, once: the basis is a property of the frozen reader,
    # not of the checkpoint, so it is never rebuilt per row.
    v17 = None
    # (B)'s verdict, filled in once the checkpoint is loaded; the default says "no such
    # arm" so a run without --v17 records the field instead of omitting it.
    attention_decision = dict(
        requested=bool(args.v17_attention), arm=V17_ATTN, built=False,
        self_damping=None, note="not requested")
    if args.v17:
        if v11_evict is None:
            raise ValueError("--v17 needs --v11-evict: V13 is the arm V17 has to beat and "
                             "the window V17 calibrates")
        v17 = load_deep_arm(args.v17, manager.hidden)
        v17["module"] = v17["module"].to(model.memory.device)
        for field in ("core_per_block", "graft_per_block"):
            recorded, reference = (v17["geometry"].get(field),
                                   v11_evict["geometry"].get(field))
            if recorded is not None and reference is not None and recorded != reference:
                raise ValueError(f"--v17 and --v11-evict record different {field}: they "
                                 f"cannot share one window")
        layers = v17["geometry"].get("utility_layers") or v11_evict["geometry"].get(
            "utility_layers")
        if layers is not None:
            v17["geometry"]["utility_layers"] = [int(x) for x in layers]
        # The checkpoint's own damping constants decide what `--v17-attention` still
        # means (B). This runs BEFORE the basis is built, because a skipped arm needs no
        # basis of its own.
        v17["attention"] = graft_attention_params(v17["config"], v17["geometry"])
        attention_decision = decide_attention_arm(args, v17)
        # The basis is needed to write a residual in the key space at all: key_channel
        # 'project' reads it in `forward`, and the graft damping projects with it too.
        needs_basis = (module_needs_key_basis(v17["module"], v17["config"])
                       or bool(args.v17_attention)
                       or bool(getattr(v17["module"], "graft_tau", 1.0) != 1.0
                               or getattr(v17["module"], "graft_bias", 0.0) != 0.0
                               or getattr(v17["module"], "graft_norm_quantile", 0.0) > 0.0))
        if needs_basis:
            from methods.prlmc import key_safe_projection as ks
            basis, basis_report = ks.build_key_bases(model)
            v17["basis"] = basis.to(model.memory.device)
            v17["basis_report"] = dict(
                {key: value for key, value in (basis_report or {}).items()
                 if key != "ranks"},
                digest=ks.basis_digest(v17["basis"]),
                reason=("key_channel='project'" if module_needs_key_basis(
                    v17["module"], v17["config"]) else
                        "--v17-attention" if args.v17_attention
                        else "the checkpoint damps its own graft keys inside calibrate"))
            install_deep_basis(v17)
    args.v17_value_ratios = parse_ratios(args.v17_value_ratios)
    if args.v17_value_ratios and v17 is None:
        raise ValueError("--v17-value-ratios needs --v17: there is no checkpoint whose "
                         "value ceiling could be swept")
    for flag, value in (("--v17-band-off", args.v17_band_off),
                        ("--v17-no-attn", args.v17_no_attn),
                        ("--v17-attention", args.v17_attention)):
        if value and v17 is None:
            raise ValueError(f"{flag} needs --v17: it is an ablation of that checkpoint")
    # (C): the sweep must contain the ceiling this checkpoint actually clips at, or
    # every "the sweep beat the checkpoint" line is drawn against an unlisted number.
    v17_value_scan = None
    if args.v17_value_ratios:
        v17_value_scan = with_baseline_ratio(args.v17_value_ratios, v17["module"])
        args.v17_value_ratios = v17_value_scan["ratios"]
    # What the checkpoint itself carried, for `identity` and `summary`: the effective
    # value ceiling (with its source), the trained per-layer regulariser weights, the
    # band layout and the damping constants. Recorded even when no derived arm is built.
    v17_meta = None
    if v17 is not None:
        effective = getattr(v17["module"], "value_ratio_effective", None)
        explicit = getattr(v17["module"], "value_ratio", None)
        v17_meta = dict(
            value_ratio=_scalars(explicit),
            value_ratio_effective=(None if effective is None else float(effective)),
            value_ratio_effective_source=(None if effective is None else
                                          ("module" if explicit is not None
                                           else "max_ratio")),
            value_ratio_baseline_arm=(value_arm(float(effective))
                                      if effective is not None else None),
            value_ratio_sweep=([float(ratio) for ratio in args.v17_value_ratios]
                               if args.v17_value_ratios else None),
            value_ratio_sweep_note=((v17_value_scan or {}).get("note") if v17_value_scan
                                    else "no --v17-value-ratios: the checkpoint's own "
                                         "ceiling was not swept"),
            layer_norm_weights=v17["layer_norm_weights"],
            band_mode=(None if getattr(v17["module"], "band_mode", None) is None
                       else str(v17["module"].band_mode)),
            graft_attention=v17["attention"],
            attention_arm=attention_decision,
            max_ratio=_scalars(getattr(v17["module"], "max_ratio", None)),
            key_ratio=_scalars(getattr(v17["module"], "key_ratio", None)),
            key_channel=(None if getattr(v17["module"], "key_channel", None) is None
                         else str(v17["module"].key_channel)),
            sim_floor=_scalars(v17["config"].get("sim_floor")),
            note="`value_ratio_effective` is the ceiling the value side really clips at; "
                 "`layer_norm_weights.values` is None when the checkpoint carries none, "
                 "which means the per-layer regulariser was not trained on that run")
    if args.v17_merge_cos and not args.v17_window_density:
        raise ValueError("--v17-merge-cos needs --v17-window-density: absorption happens "
                         "while the density window is built")
    if args.v17_window_density and v11_evict is None:
        raise ValueError("--v17-window-density needs --v11-evict: the fixed-quota V13 arm "
                         "is what the density window is compared with")
    density = None
    if args.v17_window_density:
        density = dict(absorb_cos=float(args.v17_merge_cos or 0.0),
                       joint=joint_density_for(v11_evict["geometry"], scorer, probe_cache,
                                               manager.hidden,
                                               absorb_cos=float(args.v17_merge_cos or 0.0)))

    rows, prepared, excluded = gr.prepare_rows(args, tokenizer, facts)
    report = dict(
        schema=SCHEMA_RESULTS, status="running",
        counts=dict(rows=len(rows), eligible=len(prepared), excluded=len(excluded)),
        protocol=build_protocol(args, args.task, anchor_config, v11_geometry,
                                v11_evict["geometry"] if v11_evict else None,
                                v14["geometry"] if v14 else None,
                                permute_seeds=args.permute_seeds,
                                v15_geometry=(v15["geometry"] if v15 else None),
                                v16_geometry=(v16["geometry"] if v16 else None),
                                v17_geometry=(v17["geometry"] if v17 else None),
                                v17_attention=(v17["attention"] if v17 else None),
                                v17_value_scan=v17_value_scan,
                                attention_decision=attention_decision,
                                v17_meta=v17_meta),
        identity=dict(
            model_path=args.model_path, checkpoint=args.checkpoint,
            checkpoint_sha256=sha(args.checkpoint),
            checkpoint_code_hash_verified=(saved or {}).get("code_hash_verified"),
            anchor=args.anchor, anchor_sha256=sha(args.anchor),
            v11=args.v11, v11_sha256=sha(args.v11),
            v11_evict=args.v11_evict,
            v11_evict_sha256=sha(args.v11_evict) if args.v11_evict else None,
            v14=args.v14,
            v14_sha256=sha(args.v14) if args.v14 else None,
            v15=args.v15,
            v15_sha256=sha(args.v15) if args.v15 else None,
            v16=args.v16,
            v16_sha256=sha(args.v16) if args.v16 else None,
            v17=args.v17,
            v17_sha256=sha(args.v17) if args.v17 else None,
            v17_arms=dict(
                value_ratios=[float(ratio) for ratio in args.v17_value_ratios],
                # The arm added because the user's list did not contain the checkpoint's
                # own effective ceiling, and whether it had to be added at all.
                value_ratio_baseline=((v17_value_scan or {}).get("baseline_arm")),
                value_ratio_baseline_listed=((v17_value_scan or {}).get("baseline_listed")),
                value_ratio_effective=((v17_value_scan or {}).get("value_ratio_effective")),
                value_ratio_effective_source=((v17_value_scan or {}).get(
                    "value_ratio_effective_source")),
                band_off=bool(args.v17_band_off),
                no_attn=bool(args.v17_no_attn),
                attention=bool(args.v17_attention),
                attention_requested=bool(attention_decision["requested"]),
                attention_skipped=bool(attention_decision["requested"]
                                       and not attention_decision["built"]),
                window_density=bool(args.v17_window_density),
                merge_cos=float(args.v17_merge_cos or 0.0)),
            v17_checkpoint=v17_meta,
            v17_attention_arm=attention_decision,
            v17_value_scan=v17_value_scan,
            deep_fusion_module_sha256=(sha(ROOT / "methods/prlmc/deep_fusion.py")
                                       if args.v17 else None),
            key_basis_digest=(v17["basis_report"]["digest"]
                              if v17 and v17.get("basis_report") else None),
            data_dir=args.data_dir,
            data_sha256=sha(Path(args.data_dir) / (args.task + ".jsonl")),
            supporting_facts_glob=args.supporting_facts_glob,
            entrypoint_sha256=sha(__file__),
            window_module_sha256=sha(ROOT / "methods/prlmc/evict_window.py"),
            calibration_module_sha256=sha(ROOT / "methods/prlmc/cross_block_calibration.py"),
            utility_module_sha256=sha(ROOT / "methods/prlmc/reader_utility.py"),
            residual_module_sha256=sha(ROOT / "methods/prlmc/anchor_residual.py"),
            reader_frozen=True, training_performed=False),
        excluded=excluded, samples=[], training_performed=False,
        optimizer_created=False,
        full_split=(args.max_samples is None and args.sample_ids is None))
    atomic_json(output, report)

    reference, table = ((None, None) if not args.verify_against
                        else ef.load_reference(args.verify_against))
    # The donor of the first row is the last prepared row's window, built once up
    # front: a donor must exist for row 0 and must never be row 0 itself.
    donor_graft, donor_graft_evict, donor_id = None, None, None
    if len(prepared) > 1:
        with torch.no_grad():
            reset(manager, initial)
            es.write_blocks(manager, merge,
                            _chunks(prepared[-1], args.window_blocks))
            warm_source, _, warm_picked = gr.select_carriers(manager, args.window_blocks,
                                                             args.target_blocks)
            warm_window, warm_meta, _ = v11["joint"].build(warm_source, warm_picked,
                                                           manager.block)
            warm_offsets = torch.nonzero(warm_meta["slot_type"] == 1,
                                         as_tuple=False).flatten()
            donor_graft = warm_window.index_select(1, warm_offsets).detach().clone()
            donor_graft_evict = None
            if v11_evict is not None:
                evict_window, evict_meta, _ = v11_evict["joint"].build(
                    warm_source, warm_picked, manager.block)
                evict_offsets = torch.nonzero(evict_meta["slot_type"] == 1,
                                              as_tuple=False).flatten()
                donor_graft_evict = evict_window.index_select(
                    1, evict_offsets).detach().clone()
                del evict_window
            donor_id = prepared[-1]["id"]
            del warm_window, warm_source
        reset(manager, initial)

    started = time.monotonic()
    for position, sample in enumerate(prepared):
        sample_started = time.monotonic()
        nv.seed_everything(args.seed)
        reset(manager, initial)
        result, carried = run_sample(model, tokenizer, manager, merge, anchor_module,
                                     v11, v11_evict, sample, donor_graft,
                                     donor_graft_evict, donor_id, args, v14, v15, v16,
                                     v13_plus, v17, density)
        # `donor_id` is the row whose graft slots this sample read; `carried["id"]` is
        # this sample. Comparing the carried id with the sample id compared a row with
        # itself and fired on every row.
        if donor_id is not None and donor_id == sample["id"]:
            raise ValueError(f"sample {sample['id']} was used as its own donor")
        donor_graft, donor_graft_evict = carried["grafts"], carried["grafts_evict"]
        donor_id = carried["id"]
        record = dict(id=sample["id"], status="graded",
                      available_blocks=sample["available_blocks"],
                      evidence=sample["evidence"],
                      seconds=time.monotonic() - sample_started, **result)
        report["samples"].append(record)
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output, report)
        print(json.dumps(dict(
            sample=position + 1, total=len(prepared), id=sample["id"],
            **{arm: result["arms"][arm]["correct"] for arm in ARMS
               if arm in result["arms"]},
            seconds=round(record["seconds"], 1))), flush=True)

    report["summary"] = joint_compare(
        report["samples"], args.f1_threshold, v17_meta=v17_meta,
        honesty=honesty_report(report["samples"], v11_evict, v17))
    if args.verify_against:
        report["verification_against_reference"] = verify_against_report(
            args, reference, table, report["samples"])
    report["elapsed_seconds"] = time.monotonic() - started
    report["status"] = "complete"
    report["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    atomic_json(output, report)
    print(json.dumps(dict(
        status="complete", out=str(output),
        elapsed_seconds=round(report["elapsed_seconds"], 1),
        arms={arm: round(summary["accuracy"], 2)
              for arm, summary in report["summary"]["arms"].items()},
        # Merging these with `dict(**a, **b)` crashed on V15's run: an arm that appears
        # in two comparison blocks (V11EvictDonor is in both versus_anchor_only and
        # versus_v11evict) is a duplicate keyword, and the whole run died AFTER writing
        # status=complete. Update the dict instead of unpacking it.
        net=_net_summary(report["summary"]["total"]),
        thresholds=report["summary"]["thresholds"]),
        indent=1, ensure_ascii=False))
    return report


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--task", default="hotpotqa", choices=("hotpotqa",))
    p.add_argument("--supporting-facts-glob", dest="supporting_facts_glob", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--checkpoint", default=None,
                   help="the frozen V10 operator (runs/natural_merge_v9/full_v10.pt)")
    p.add_argument("--anchor", default=None,
                   help="the finished AnchorOnly checkpoint (the 56/158 baseline)")
    p.add_argument("--v11", default=None,
                   help="the plain full-pool calibration checkpoint (trained with "
                        "--cross off): the V11Evict arm to beat")
    p.add_argument("--v11-evict", dest="v11_evict", default=None,
                   help="the cross-block conditioned checkpoint (trained with --cross "
                        "on --neighbor-mode topk): the V13 arm")
    p.add_argument("--v13-plus", dest="v13_plus", default=None,
                   help="a CrossBlockResidual checkpoint trained with span-value / "
                        "context-diffusion / csls (the V13 successor); calibrates the "
                        "SAME window as --v11-evict")
    p.add_argument("--v16", default=None,
                   help="the band-aligned, margin-gated module (trained with "
                        "--architecture v16): calibrates the SAME window as --v11-evict")
    p.add_argument("--v15", default=None,
                   help="the layered sink-gated relation module (trained with "
                        "--architecture v15): calibrates the SAME window as --v11-evict")
    p.add_argument("--v13-graft-scales", dest="v13_graft_scales", default=None,
                   help="inference-time graft ceiling sweep on the V13 checkpoint, e.g. "
                        "0.38,0.40,0.42: weights and window untouched, only the bound the "
                        "graft residual may spend changes")
    p.add_argument("--v13-relevance-cap", dest="v13_relevance_cap", default=None,
                   help="inference-time relevance-proportional graft cap: "
                        "0.25 + 0.25 * rank_norm(mean Top-4 cross-block similarity)")
    p.add_argument("--v13-neighbor-modes", dest="v13_neighbor_modes", default=None,
                   help="attribution arms for V13's neighbour choice, comma separated "
                        "from {random,bottom}: same weights, same window, same projection, "
                        "same number of neighbours, only WHICH cross-block slots are "
                        "picked changes (random = uniform over legal cross-block slots, "
                        "bottom = the least similar). Builds V13RandNeighbor / "
                        "V13BottomNeighbor")
    p.add_argument("--v13-permute-seeds", dest="v13_permute_seeds", default=None,
                   help="inference-time attribution test for V13's cross-block "
                        "correspondence: a comma separated seed list (e.g. 0,1,2) builds "
                        "one arm per seed that calibrates V13's window with the "
                        "precomputed context rows shuffled inside (source_block, "
                        "slot_type). Nothing is retrained")
    p.add_argument("--v17", default=None,
                   help="the key-anchored / value-released checkpoint "
                        "(methods/prlmc/deep_fusion.py, schema deep_fusion/1); builds the "
                        "V17 arm and calibrates the SAME window as --v11-evict")
    p.add_argument("--v17-value-ratios", dest="v17_value_ratios", default=None,
                   help="inference-time value-ceiling sweep on the --v17 checkpoint, "
                        "comma separated (e.g. 0.25,0.30,0.40): builds V17Value25 / "
                        "V17Value30 / V17Value40 by replacing ONE runtime attribute of a "
                        "copy of the module. No weight is rewritten. The checkpoint's own "
                        "effective ceiling (`value_ratio_effective`, i.e. max_ratio when "
                        "value_ratio is None) is ADDED to the list when it is missing, as "
                        "V17Value34 for 0.34 (nearest whole percent)")
    p.add_argument("--v17-band-off", dest="v17_band_off", action="store_true",
                   help="ablation V17NoBand: the same checkpoint with band_mode turned "
                        "off at inference time")
    p.add_argument("--v17-no-attn", dest="v17_no_attn", action="store_true",
                   help="arm V17NoAttn: mechanism four (graft attention damping) OFF. The "
                        "checkpoint's tau / bias / norm_quantile are neutralised on a deep "
                        "copy and no damping is applied, so the module's own calibrate "
                        "damps nothing: `V17 - V17NoAttn` is that mechanism's worth. No "
                        "weight, no checkpoint and no V17 attribute is touched")
    p.add_argument("--v17-attention", dest="v17_attention", action="store_true",
                   help="arm V17Attn: graft-only key damping with the tau / bias the "
                        "checkpoint recorded (graft_attention_calibration), applied once "
                        "before the module runs. Only built when the checkpoint's own "
                        "constants are neutral: a checkpoint that damps inside "
                        "deep_fusion.calibrate makes this arm a duplicate of V17, so it is "
                        "WARNED about and SKIPPED -- use --v17-no-attn there")
    p.add_argument("--v17-window-density", dest="v17_window_density",
                   action="store_true",
                   help="rebuild the window with quota_mode='density' (the same 1,620 "
                        "core slots, allocated per block by redundancy) and answer "
                        "V13Density and, with --v17, V17Density on it")
    p.add_argument("--v17-merge-cos", dest="v17_merge_cos", type=float, default=0.0,
                   help="with --v17-window-density: evicted slots within this cosine of a "
                        "retained one are absorbed into it (0 = off)")
    p.add_argument("--v14", default=None,
                   help="the complement-mode checkpoint (trained with --cross on "
                        "--neighbor-mode complement --k-same 2 --k-cross 2): the V14 arm, "
                        "which calibrates the SAME window as --v11-evict")
    p.add_argument("--out", default=None)
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--window-blocks", dest="window_blocks", type=int,
                   default=gr.WINDOW_BLOCKS)
    p.add_argument("--target-blocks", dest="target_blocks", type=int,
                   default=gr.TARGET_BLOCKS)
    p.add_argument("--f1-threshold", dest="f1_threshold", type=float, default=0.5)
    p.add_argument("--sample-ids", dest="sample_ids", default=None)
    p.add_argument("--max-samples", dest="max_samples", type=int, default=None)
    p.add_argument("--verify-against", dest="verify_against", default=None,
                   help="a finished report; Full, Selection, Graft and AnchorOnly must "
                        "reproduce it row for row")
    p.add_argument("--strict-code-hash", dest="strict_code_hash", default="true",
                   choices=("true", "false"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.dry_run and not args.data_dir:
        raise SystemExit("--data-dir is required")
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
