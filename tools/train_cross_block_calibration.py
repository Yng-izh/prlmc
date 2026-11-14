#!/usr/bin/env python3
"""Round 6: train the V11 joint architecture once, against the AnchorOnly baseline.

V14 note: the module trained here is `cross_block_calibration.CrossBlockResidual`, and
its neighbour rule is selected by `--neighbor-mode`. `topk` reproduces V13 exactly;
`complement` (V14) asks every slot for `--k-same` neighbours of its own core/graft kind
plus `--k-cross` from the other side, with equal per-group weight mass. Nothing else in
this file differs between the two arms, so a V13-vs-V14 comparison moves one variable.

V17 note: `--architecture v17` swaps the module for `deep_fusion.DeepFusionResidual`,
which puts the key-anchored / value-open split, the three-band layer gate, the
similarity floor and the graft-attention damping on top of the same residual V13
trains. Every one of its switches defaults to its "off" value and the branch is a
separate `if`, so `--architecture v13` walks the code path it walked before any of
these flags existed. Three groups of switches are independent of the architecture and
are documented where they are used: the window budget (`--quota-mode`, `--min-core`,
`--block-merge-cos`, forwarded to `JointConsolidation`), the objective
(`--kl-topk`, `--layer-norm-weights`) and the counterfactual (`--cf-mode`,
`--cf-random-seed`).

The three mechanisms this project has arrived at are trained together, because
separately each one has an obvious way to look like a failure and together they are
where the capacity actually comes from:

    module 1  reader-utility slot selection   `methods/prlmc/reader_utility.py`
    module 2  (retired) same-block folding -- deleted in V12, measured at -4 questions
    module 3  full-pool residual calibration  `FullPoolResidual` in `anchor_residual.py`

    carriers 9 x 256 -> 1,620 core (folded)   dropped 2,560 -> 684 graft (selected)
    2,304 slots, the same budget the baseline spends on 1,944 verbatim + 360 anchors

What trains
-----------
`FullPoolResidual` only: 4 x rank x hidden plus a 3 -> 32 -> 2 x rank FiLM, with the
graft half initialised from the finished AnchorOnly checkpoint and the core half at
zero. Reader, writer and the probe bank are frozen; the fold assignment and the
graft selection are index operations with no parameters at all.

The objective
-------------
    L = CE(answer) + w_cf * CF + w_protect * KL(calibrated || reference) + w_reg * R

CE        weighted answer cross-entropy; the first answer token (the entity) counts
          `--first-token-weight` times, because "the reader produced the right
          string but the wrong head noun" and "the reader produced nothing" are not
          the same failure and a plain mean treats them as one.
CF        the counterfactual hinge, `max(0, gamma - logP(y|M_true) + logP(y|M_irrel))`,
          where `M_irrel` is the SAME window with the 684 graft slots taken from the
          previous document instead of this one. It is computed only on rows where the
          compressed window is not answering on its own (`--cf-gate reference`, the
          default), because a hinge on a row the window already gets right has nothing
          to teach. The gate deliberately does NOT measure `logP(Full) - logP(window)`:
          that needs a 4,864-slot forward per row, twice the width of the forward the
          training step runs, for one boolean. `--cf-gate full` restores it, `none`
          disables the gate. The number of qualifying steps is recorded either way: a
          CF term that almost never fires is not evidence about the architecture.
          The donor window is scored under `no_grad` -- the negative side enters as a
          measured constant, so the hinge is one-sided pressure on the true window
          instead of a second differentiable branch.
protect   KL to the window the joint run STARTS at -- the V11 window with the initial
          module, i.e. the AnchorOnly calibration on the graft slots and an untouched
          core. KL is exactly zero there, so the term only stops the core half from
          buying graft accuracy with carrier knowledge; it cannot push the run to undo
          the calibration it is built on. Cached per document: the reference depends on
          the document and on frozen weights only, so it never changes.
R         mean over slots of `(||dh|| / rms(h))^2`; keeps the calibration from
          spending the whole bound allowance on a few slots.

Cost, and why the effective batch size is a flag
------------------------------------------------
The plan asks for 384 updates at an effective batch of 8. One document step is three
reader forwards (Full, reference, calibrated) plus the window build, so a batch of 8
multiplies both by eight. The default here is `--grad-accum 1`, which is the
384-update budget every earlier round used and which is enough to answer "does the
joint architecture produce a net gain at all"; `--grad-accum 8` is available and the
dry run prints what it costs.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # noqa: F401  (used directly below, not transitively)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_graft_recovery as gr                          # noqa: E402
import native_eval as nv                                  # noqa: E402
from io_utils import atomic_json, sha                     # noqa: E402
from runtime import load_runtime, reset, tensor_digest    # noqa: E402
from methods.prlmc.anchor_residual import SCHEMA as ANCHOR_SCHEMA   # noqa: E402
from methods.prlmc.core_fusion import apply_delta         # noqa: E402
from methods.prlmc.cross_block_calibration import (       # noqa: E402
    SCHEMA as CALIB_SCHEMA, CrossBlockResidual, conditions)
from methods.prlmc.evict_window import (                  # noqa: E402
    SCHEMA as JOINT_SCHEMA, QUOTA, JointConsolidation)
from methods.prlmc import band_relation as br             # noqa: E402
from methods.prlmc import hierarchical_relation as hr     # noqa: E402
from methods.prlmc import key_safe_projection as ks       # noqa: E402

# V13 keeps the V11 driver shape: the calibration module changed, the window did not.
FULL_POOL_SCHEMA = CALIB_SCHEMA
FullPoolResidual = CrossBlockResidual
from methods.prlmc.probes import _checkpoint_func         # noqa: E402
from methods.prlmc.reader_utility import ReaderScorer     # noqa: E402
from objective import answer_logits                       # noqa: E402
from train_core_fusion import (                           # noqa: E402
    BUDGET_BLOCKS, TARGET_BLOCKS, WINDOW_BLOCKS, chunks_of, load_documents,
    write_blocks)
from train_source_compaction import installed             # noqa: E402

SLOTS_PER_BLOCK = gr.SLOTS_PER_BLOCK
TARGET_SLOTS = TARGET_BLOCKS * SLOTS_PER_BLOCK
# Update 1 can only reach the two output halves, the two gates and the FiLM output:
# `up_core` starts at zero, so nothing upstream of it has a gradient yet, and the FiLM
# input projection sits behind a zero matrix of its own. Update 2 must reach all of it.
STAGE1 = ("up_graft", "up_core", "gate_logit", "type_gate_logit", "cond_up")
STAGE2 = ("down", "up_graft", "up_core", "gate_logit", "type_gate_logit",
          "cond_up", "cond_down")
# V15 only: the modulation matrices must move on update 1 (they are the only route into
# the relation branch) and the five per-layer projections from update 2 on. A zero
# gradient there is a wiring fault, not a result.
V15_STAGE1 = ("mod_scale.weight", "mod_shift.weight")
V15_STAGE2 = tuple(f"cross_proj.{index}.weight" for index in range(hr.KV_LAYERS))
# V16 only. The gate multiplies a ZERO projection at update 0, so gate parameters cannot
# have a gradient on step 1 -- asking for one would be asking for the impossible. The
# projections can (dL/dW = dL/du * gate * ctx, with gate != 0), and they move on step 1;
# from step 2 the gates follow.
V16_STAGE1 = tuple([f"band_proj.{index}.weight" for index in range(br.KV_LAYERS)]
                   + [f"layer_proj.{index}.weight"
                      for index in range(br.NUM_MEMORY_LAYERS - br.SHALLOW_CUTOFF)])
V16_STAGE2 = V16_STAGE1 + ("gate_kappa", "gate_tau", "eta_logit")
# The pre-V17 default of the window-budget floor `JointConsolidation` grew in §3 of the
# V17 spec. Named here so this file can tell "left alone" from "the user asked for
# something", which is what keeps the V13 arm's call site unchanged.
MIN_CORE = 140


# ---------------------------------------------------------------------------
# V17: the deep-fusion module, its key basis, and the switches it reads
# ---------------------------------------------------------------------------
def deep_fusion_module():
    """`methods/prlmc/deep_fusion.py`, imported on demand.

    Lazy on purpose. V13 has to keep training against the code it was measured on, and a
    module-level import would make an unfinished or broken `deep_fusion.py` take every
    arm down with it, including the one the baseline runs on.
    """
    from methods.prlmc import deep_fusion
    return deep_fusion


def band_gates_of(args):
    """`--band-gates 0.0,1.0,1.0` -> three floats, parsed once into `args`.

    Band 0 is forced to zero inside the module, so its value here is only ever recorded;
    keeping the CLI string parsable at both the construction site and the report site
    means a mistyped initialisation shows up in the checkpoint instead of being masked.
    """
    values = args.band_gates
    if isinstance(values, str):
        parts = [part for part in values.replace(" ", "").split(",") if part]
        try:
            parsed = tuple(float(part) for part in parts)
        except ValueError:
            raise SystemExit("--band-gates wants three comma-separated floats "
                             f"(lexical, bridging, readout); got {args.band_gates!r}")
        if len(parsed) != 3:
            raise SystemExit("--band-gates wants exactly three values (lexical, bridging, "
                             f"readout); got {len(parsed)}")
        args.band_gates = parsed
        values = parsed
    return tuple(float(value) for value in values)


def needs_key_basis(args):
    """Whether this V17 run has to build the key map's row space at all.

    `--key-channel project` uses it inside the forward; the three graft-attention numbers
    use it to damp the graft keys at inference time. Building it once here means the
    checkpoint records which reader the basis came from (`basis_digest`) even for runs
    that only need it later, on the evaluation arm.
    """
    return (str(args.key_channel) == "project"
            or float(args.graft_tau) != 1.0
            or float(args.graft_bias) != 0.0
            or float(args.graft_norm_quantile) != 0.0)


def install_basis(deep, module, basis):
    """Inject the basis as `module.key_basis` -- a non-parameter, so never in the state.

    `deep_fusion` may expose the hook as a method or as a free function; both are the
    same operation and neither is allowed to be silently skipped, so the missing case is
    an explicit error rather than a module that raises on its first forward.
    """
    method = getattr(module, "install_key_basis", None)
    if callable(method):
        return method(basis)
    hook = getattr(deep, "install_key_basis", None)
    if callable(hook):
        return hook(module, basis)
    raise ValueError("deep_fusion exposes no install_key_basis: --key-channel project "
                     "cannot be wired")


# The new switches only `DeepFusionResidual` implements. Setting one of them on another
# architecture would train the OLD module and then report "no effect" for a mechanism
# that was never switched on, which is worse than not running at all.
MODULE_BOUND_SWITCHES = ("key_channel", "key_ratio", "value_ratio", "band_mode",
                         "band_gates", "sim_floor", "graft_tau", "graft_bias",
                         "graft_norm_quantile")


def nondefault_switches(args):
    """Every new switch that is not at its default, as `name=value` strings.

    Empty for a V13 run whose flags are all at their defaults, and that is the whole of
    the neutrality claim this file can make in code: nothing below reads a switch that
    is off, so nothing below can compute a different number.
    """
    active = []
    for name, value, default in (
            ("key_channel", args.key_channel, "off"),
            ("key_ratio", args.key_ratio, 0.10),
            ("value_ratio", args.value_ratio, None),
            ("band_mode", args.band_mode, "off"),
            ("band_gates", list(band_gates_of(args)), [0.0, 1.0, 1.0]),
            ("sim_floor", args.sim_floor, 0.0),
            ("graft_tau", args.graft_tau, 1.0),
            ("graft_bias", args.graft_bias, 0.0),
            ("graft_norm_quantile", args.graft_norm_quantile, 0.0),
            ("quota_mode", args.quota_mode, "fixed"),
            ("min_core", args.min_core, MIN_CORE),
            ("block_merge_cos", args.block_merge_cos, 0.0),
            ("layer_norm_weights", args.layer_norm_weights, "off"),
            ("kl_topk", args.kl_topk, 0),
            ("cf_mode", args.cf_mode, "donor"),
            ("cf_random_seed", args.cf_random_seed, 0)):
        if value != default:
            active.append(f"{name}={value}")
    return active


# ---------------------------------------------------------------------------
# module and probe bank
# ---------------------------------------------------------------------------
def load_anchor_state(path, hidden):
    """The finished AnchorOnly checkpoint: its state dict and its config."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != ANCHOR_SCHEMA + "/checkpoint":
        raise SystemExit(f"{path} is not a {ANCHOR_SCHEMA} checkpoint")
    config = dict(payload["config"])
    if int(config.get("hidden_size", -1)) != int(hidden):
        raise SystemExit(f"{path} was built for hidden {config.get('hidden_size')}, "
                         f"the reader is {hidden}")
    return payload["state_dict"], config


def _joint_extra(args):
    """§3's window-budget switches, forwarded to `JointConsolidation`.

    Only the ones that leave their pre-V17 defaults are passed, so a V13 run constructs
    the builder with exactly the argument list it has always used -- textually the same
    call, the same object, and a builder that predates the parameters still loads.
    `--min-core` is only meaningful with `--quota-mode density`; sending it alone would
    look like a request the builder has to guess the intent of.
    """
    extra = {}
    if str(args.quota_mode) != "fixed":
        extra["quota_mode"] = args.quota_mode
    if int(args.min_core) != MIN_CORE:
        extra["min_core"] = int(args.min_core)
    if float(args.block_merge_cos) != 0.0:
        extra["absorb_cos"] = float(args.block_merge_cos)
    return extra


def build_module(args, hidden, anchor_state, anchor_config):
    """The one trainable module: graft half from AnchorOnly, core half at zero."""
    rank = int(args.rank or anchor_config.get("rank"))
    if args.architecture == "v16":
        module = br.BandAlignedRelationResidual(
            hidden_size=hidden, rank=rank, max_ratio=args.max_ratio,
            max_ratio_core=args.max_ratio_core, gate_init=args.gate_init,
            cond_rank=args.cond_rank, k_neighbors=args.k_neighbors,
            shallow_cutoff=args.shallow_cutoff, core_margin=args.core_margin,
            graft_margin=args.graft_margin, gate_kappa=args.gate_kappa,
            eta_init=args.eta_init, core_weight=args.core_weight,
            graft_weight=args.graft_weight)
        module.load_anchor_only(anchor_state, anchor_config)
        return module, rank
    if args.architecture == "v15":
        module = hr.HierarchicalRelationResidual(
            hidden_size=hidden, rank=rank, max_ratio=args.max_ratio,
            gate_init=args.gate_init, cond_rank=args.cond_rank,
            k_neighbors=args.k_neighbors, shallow_cutoff=args.shallow_cutoff,
            tau_init=args.sink_tau, sink_percentile=args.sink_percentile,
            max_ratio_core=args.max_ratio_core, core_weight=args.core_weight,
            graft_weight=args.graft_weight)
        module.load_anchor_only(anchor_state, anchor_config)
        return module, rank
    if args.architecture == "v17":
        deep = deep_fusion_module()
        module = deep.DeepFusionResidual(
            hidden_size=hidden, rank=rank, max_ratio=args.max_ratio,
            max_ratio_core=args.max_ratio_core, gate_init=args.gate_init,
            cond_rank=args.cond_rank, use_cross=(args.cross == "on"),
            k_neighbors=args.k_neighbors, neighbor_mode=args.neighbor_mode,
            k_same=args.k_same, k_cross=args.k_cross, kv_dim=args.kv_dim,
            core_weight=args.core_weight, graft_weight=args.graft_weight,
            span_value=(args.span_value == "on"),
            context_diffusion=(args.context_diffusion == "on"),
            csls=(args.csls == "on"), use_evicted=(args.evicted_context == "on"),
            # V17's own switches. All of them sit at their "off" value by default, so
            # this branch builds the parent's mechanism unless a flag is turned on: the
            # new names are additive, they do not reinterpret an existing one.
            key_channel=args.key_channel, key_ratio=args.key_ratio,
            value_ratio=args.value_ratio, band_mode=args.band_mode,
            band_gates=band_gates_of(args), sim_floor=args.sim_floor,
            # The three attention-damping constants are constructor arguments in
            # `DeepFusionResidual`; leaving them out here would write tau=1.0 into every
            # checkpoint and make `--graft-tau 0.92` a flag that changes nothing.
            graft_tau=args.graft_tau, graft_bias=args.graft_bias,
            graft_norm_quantile=args.graft_norm_quantile,
            # `None` = identity: row i of the window IS reader layer i. The window this
            # module writes into is the reader's full 32-layer stack, so the three bands
            # (0-10 / 11-21 / 22-31) line up with the reader's real layers; passing the
            # five CONTEXT layers here instead made every row look like layer 12+ and
            # made band 0 look empty.
            layer_ids=None)
        module.load_anchor_only(anchor_state, anchor_config)
        return module, rank
    module = FullPoolResidual(hidden_size=hidden, rank=rank,
                              max_ratio=args.max_ratio, gate_init=args.gate_init,
                              cond_rank=args.cond_rank,
                              use_cross=(args.cross == "on"),
                              k_neighbors=args.k_neighbors,
                              neighbor_mode=args.neighbor_mode,
                              k_same=args.k_same, k_cross=args.k_cross,
                              kv_dim=args.kv_dim,
                              max_ratio_core=args.max_ratio_core,
                              core_weight=args.core_weight,
                              graft_weight=args.graft_weight,
                              span_value=(args.span_value == "on"),
                              context_diffusion=(args.context_diffusion == "on"),
                              csls=(args.csls == "on"),
                              use_evicted=(args.evicted_context == "on"))
    module.load_anchor_only(anchor_state, anchor_config)
    return module, rank


def build_probes(args, tokenizer, documents):
    """The query bank the reading score is computed against. No test data in it."""
    if args.probe_text:
        ids = tokenizer(args.probe_text, add_special_tokens=False)["input_ids"]
        return [list(ids)], dict(kind="text", text=args.probe_text, count=1)
    wanted = min(int(args.probe_count), len(documents))
    if wanted <= 0:
        raise SystemExit("the reader-utility score needs at least one probe question")
    bank = [list(document["query_ids"]) for document in documents[:wanted]]
    return bank, dict(kind="training_questions", task=args.probe_task, count=len(bank),
                      documents=[document["id"] for document in documents[:wanted]])


# ---------------------------------------------------------------------------
# one document
# ---------------------------------------------------------------------------
def context_of(module, window, meta, return_report=False, source=None):
    """The module's own relation rule. V13 owns `neighbours`, V15 owns `relations`;
    the loop must not know which one it is talking to. Modules that also read the
    evicted pool return `(in_pool, evicted)` as one payload."""
    if hasattr(module, "relations"):
        return module.relations(window, meta, return_report=return_report)
    if getattr(module, "use_cross", False):
        context, report = module.neighbours(window, meta, return_report=True)
        if getattr(module, "use_evicted", False):
            if source is None:
                raise ValueError("this module reads the evicted pool: the training loop "
                                 "has to pass the source it was evicted from")
            evicted, evicted_report = module.evicted(window, meta, source,
                                                     return_report=True)
            context = (context, evicted)
            report = dict(report or {}, evicted=evicted_report)
            del evicted
        return (context, report) if return_report else context
    return (None, None) if return_report else None


def v16_stats(module, cond, relation, window):
    """Per-band gate distribution, projection norms and the injected RMS (V16 only).

    Cheap and computed once per window: the projections are 1024 -> 64, so the whole
    22-layer cross term is far smaller than one reader forward. These numbers exist so a
    run can be read back -- a gate that never opens must not look like a mechanism.
    """
    if not hasattr(module, "gate") or relation is None:
        return None
    bands, margins = relation
    with torch.no_grad():
        graft = cond[:, 2].float() > 0.5
        gate_core, gate_graft = [], []
        for band in range(bands.shape[1]):
            values = torch.stack([module.gate(graft, margins[:, band].float(), layer).float()
                                  for layer in br.band_members(band)], dim=0)
            gate_core.append(float(values[:, ~graft].mean()))
            gate_graft.append(float(values[:, graft].mean()))
        active = torch.as_tensor([br.band_of_layer(layer) is not None
                                  for layer in range(window.shape[0])])
        cross = module.cross_term(bands, margins, cond,
                                  torch.arange(window.shape[0], device=window.device))
        injected = cross[active].float()
        return dict(gate_core=[round(value, 5) for value in gate_core],
                    gate_graft=[round(value, 5) for value in gate_graft],
                    band_proj_norm=[round(float(projection.weight.detach().float().norm()), 3)
                                    for projection in module.band_proj],
                    layer_proj_norm=round(float(torch.stack([
                        projection.weight.detach().float().norm()
                        for projection in module.layer_proj]).mean()), 3),
                    injection_rms=round(float(injected.square().mean().sqrt()), 6),
                    injection_max=round(float(injected.abs().max()), 5),
                    shallow_leak=round(float(cross[~active].abs().max()), 8)
                    if bool((~active).any()) else 0.0)


def damp_graft_keys(window, meta, module):
    """Apply the graft-attention damping to the window the training step reads.

    This is mechanism four's *application* site. The constants travel in the module
    config, and both the training loop and `deep_fusion.calibrate` read them from there,
    so a checkpoint trained with `--graft-tau 0.92` produces a damped window everywhere
    instead of only on the evaluation arm that someone remembered to switch on.

    Damping runs BEFORE the neighbour context is measured: the context is built from the
    window's keys, so damping afterwards would leave every row measured against keys the
    reader never sees. Returns the window unchanged plus `applied=False` when the
    constants are neutral, so the default path pays one comparison and nothing else.
    """
    tau = float(getattr(module, "graft_tau", 1.0))
    bias = float(getattr(module, "graft_bias", 0.0))
    quantile = float(getattr(module, "graft_norm_quantile", 0.0))
    if tau == 1.0 and bias == 0.0 and quantile <= 0.0:
        return window, dict(applied=False, tau=tau, bias=bias, norm_quantile=quantile,
                            slots_touched=0,
                            note="neutral constants: the window is the one the pool built")
    deep = deep_fusion_module()
    basis = getattr(module, "key_basis", None)
    if basis is None:
        raise ValueError("this module damps graft keys but carries no key basis: the "
                         "row space it projects on has to be installed first")
    table = getattr(module, "layer_ids", None)
    ids = (tuple(range(int(window.shape[0]))) if table is None
           else tuple(int(layer) for layer in table))
    if len(ids) != int(window.shape[0]):
        raise ValueError(f"{len(ids)} model layers for {int(window.shape[0])} window rows")
    damped, report = deep.graft_attention_calibration(
        window, meta, basis, tau=tau, bias=bias, norm_quantile=quantile,
        layer_ids=ids, return_report=True)
    return damped, dict(report, applied=True, site="training window")


def layer_norm_weights_from(per_layer):
    """`w_l = 1 / (D_l + 1e-6)`, normalised to mean 1.

    A layer whose initial residual is small gets a larger weight, i.e. less room to
    grow: the ceiling the regulariser enforces becomes relative to where each layer
    started instead of a constant across layers that differ by two orders of magnitude.
    """
    inverse = [1.0 / (float(value) + 1e-6) for value in per_layer]
    mean = sum(inverse) / max(len(inverse), 1)
    return [value / mean for value in inverse]


def ensure_layer_norm_weights(module, layers):
    """Make sure `module.layer_norm_weights` exists, at the length of THIS window.

    `CrossBlockResidual` registers the buffer sized for the five-layer CONTEXT table, but
    the residual is written into all 32 reader layers, so the two do not agree. Resizing is
    safe only while the buffer is still all ones -- a trained vector of the wrong length is
    a real mismatch and stays an error instead of being silently resized away.
    """
    existing = dict(module.named_buffers()).get("layer_norm_weights")
    if existing is not None:
        if int(existing.numel()) == int(layers):
            return
        if not bool(torch.all(existing == 1)):
            raise ValueError(f"layer_norm_weights has {int(existing.numel())} trained "
                             f"entries, the window has {int(layers)} layers")
        module.register_buffer(
            "layer_norm_weights",
            torch.ones(int(layers), dtype=torch.float32,
                       device=next(module.parameters()).device),
            persistent=False)
        return
    module.register_buffer(
        "layer_norm_weights",
        torch.ones(int(layers), dtype=torch.float32,
                   device=next(module.parameters()).device),
        persistent=True)


def calibrate_with_reg(window, meta, module, layer_chunk=1, source=None,
                       want_per_layer=False):
    """Module 3 with the residual regulariser kept inside the graph.

    Each layer chunk is a `torch.utils.checkpoint` region: the module is 0.52 M
    parameters, so recomputing it on the backward is nearly free, while keeping its
    float32 activations alive is not. Measured on the smoke run, the un-checkpointed
    version peaked at 33.4 GiB and the next update's backward had no room for the
    3.12 GiB memory gradient. The delta is also cast to the window's dtype before it
    meets the slot, so the sum and the RMS restore never promote a [layers, slots,
    hidden] tensor to float32 twice over.

    The chunk is ONE layer by default. The window is the reader's full 32-layer stack, and
    V17 makes three float32 tensors of a chunk live at once (the delta, the key-visible
    half and the complement), so a four-layer chunk asks for 3 x 144 MiB in a single step
    -- which is where the V17 runs hit the ceiling with 63 MiB free. One layer is 36 MiB
    per tensor; it is slower and it never changes what is computed.

    `want_per_layer` adds `info["per_layer_relative_delta"]` (one value per layer) and
    a `module.layer_norm_weights` buffer switches the regulariser from a sum over all
    layers to a weighted per-layer sum. Both are off by default and neither touches the
    arithmetic of the default path: the original expression is still the one that runs
    whenever the weights are absent or still all ones.
    """
    # Mechanism four acts on the window itself, and it acts FIRST: the neighbour context
    # below is measured from these keys. A checkpoint that carries non-neutral constants
    # trains and evaluates on the same damped window; a neutral one gets the pool's
    # window back untouched.
    window, attention_report = damp_graft_keys(window, meta, module)
    cond = conditions(meta, window.shape[1])
    # The module owns its relation rule: V13 (topk) and V14 (complement) differ in
    # nothing else, so this call site must not hard-code k or the mode. The report is
    # kept because the sink mass is the only evidence that V15's gate is doing anything.
    context, relation_report = context_of(module, window, meta, return_report=True,
                                          source=source)
    # Preallocated and filled slice by slice. `torch.cat(pieces)` needs the whole
    # 32-layer output twice alive at once (the list and the concatenation), which at
    # 2,304 slots is 576 MiB of pure transient; writing into a buffer that already
    # has the final shape costs nothing.
    calibrated_all = torch.empty_like(window)
    regulariser = torch.zeros((), device=window.device)
    delta_sq = slot_sq = 0.0
    # A buffer that is still all ones changes nothing, so the weighted branch is only
    # entered once `--layer-norm-weights on` has actually written values in. That is
    # what keeps a V13 run bit-identical rather than 1-ulp-different by reduction order.
    layer_weights = None
    buffer = getattr(module, "layer_norm_weights", None)
    if buffer is not None and int(buffer.numel()) == int(window.shape[0]):
        if not bool(torch.all(buffer == 1)):
            layer_weights = buffer.detach().float()
    collect_layers = bool(want_per_layer) or layer_weights is not None
    delta_sq_layers = (torch.zeros(window.shape[0], dtype=torch.float64)
                       if collect_layers else None)
    slot_sq_layers = (torch.zeros(window.shape[0], dtype=torch.float64)
                      if collect_layers else None)

    def chunk_step(chunk, conditioning, ctx, layer_ids):
        delta = (module(chunk, conditioning, ctx, layer_ids)
                 if getattr(module, "uses_layers", False)
                 else module(chunk, conditioning, ctx))
        calibrated = apply_delta(chunk, delta.to(chunk.dtype))
        rms = chunk.float().square().mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        # Asymmetric, still soft: the carriers pay twice the penalty the graft slots do,
        # which holds their key drift down without freezing it (experiment 4 showed a
        # hard key lock makes the new content useless).
        weight = module.regulariser_weight(conditioning).view(1, -1, 1)
        scaled = (delta.float() / rms).square() * weight
        if layer_weights is None:
            penalty = scaled.mean() * int(chunk.shape[0])
        else:
            # `mean()` over the chunk is a mean over layers too, so multiplying it by the
            # chunk's layer count is a per-layer sum. Written as an explicit per-layer
            # mean so the weights can act on it; with all-ones weights the two forms are
            # the same quantity, which is why the form above is kept for the default.
            penalty = (scaled.mean(dim=(1, 2))
                       * layer_weights.index_select(0, layer_ids)).sum()
        magnitude = delta.detach().float().square().sum()
        if not collect_layers:
            return calibrated, penalty, magnitude
        return (calibrated, penalty, magnitude,
                delta.detach().float().square().sum(dim=(1, 2)))

    # Whether a chunk needs checkpointing is a property of the MODULE, not of the slot
    # tensor: the window comes from a frozen reader, so `chunk.requires_grad` is False even
    # while the residual module is being trained. Testing the tensor disabled the
    # checkpoint on every training step and kept all 32 layers' activations alive -- that
    # is the 36.97 GiB this path was measured at, not a shortage of GPU.
    trainable_module = (torch.is_grad_enabled()
                        and any(parameter.requires_grad for parameter in module.parameters()))
    for start in range(0, window.shape[0], layer_chunk):
        stop = min(start + layer_chunk, window.shape[0])
        chunk = window[start:stop]
        layer_ids = torch.arange(start, stop, device=window.device)
        if trainable_module:
            produced = torch.utils.checkpoint.checkpoint(
                chunk_step, chunk, cond, context, layer_ids,
                use_reentrant=False)
        else:
            produced = chunk_step(chunk, cond, context, layer_ids)
        calibrated, penalty, magnitude = produced[:3]
        calibrated_all[start:stop] = calibrated
        regulariser = regulariser + penalty
        delta_sq += float(magnitude)
        slot_sq += float(chunk.detach().float().square().sum())
        if collect_layers:
            delta_sq_layers[start:stop] = produced[3].detach().double().cpu()
            slot_sq_layers[start:stop] = (chunk.detach().float().square()
                                          .sum(dim=(1, 2)).double().cpu())
    regulariser = regulariser / max(int(window.shape[0]), 1)
    info = dict(slots=int(window.shape[1]),
                core_slots=int((meta["slot_type"] == 0).sum()),
                graft_slots=int((meta["slot_type"] == 1).sum()),
                folded_slots=int((meta["cluster_size"] > 1).sum()),
                relative_delta=(delta_sq / max(slot_sq, 1e-24)) ** 0.5,
                gate=float(torch.sigmoid(module.gate_logit.detach().float())))
    if collect_layers:
        # Per-layer share of the residual -- the same definition as `relative_delta`,
        # applied one layer at a time. `--layer-norm-weights on` reads it once, after
        # update 0, to decide how much room each layer is allowed.
        info["per_layer_relative_delta"] = [
            float(value) for value in
            (delta_sq_layers / slot_sq_layers.clamp_min(1e-24)).sqrt()]
    if relation_report is not None:
        info["relation"] = relation_report
    stats = v16_stats(module, cond, context, window)
    if stats is not None:
        info["v16"] = stats
    if attention_report is not None:
        # Same key the evaluation path uses (`deep_fusion.calibrate` writes `v17`), so a
        # reader of either report finds the damping in one place.
        info["v17"] = dict(attention=attention_report,
                           value_ratio_effective=getattr(module, "value_ratio_effective",
                                                         None))
    return calibrated_all, regulariser, info


@contextmanager
def read_window(model, manager, window, calibrated):
    """Install the calibrated window in the reader's pool with its graph intact."""
    offsets = torch.arange(int(window.shape[1]), device=window.device)
    with installed(model, manager, window.detach(), calibrated, offsets):
        yield


def answer_logp(logits, gold):
    """Sum of the gold answer tokens' log-probabilities."""
    return -F.cross_entropy(logits, gold, reduction="none").sum()


def answer_weight(gold, first_weight):
    """Weights that put `first_weight`x the average emphasis on the first token."""
    weight = torch.ones_like(gold, dtype=torch.float32)
    if float(first_weight) != 1.0:
        weight[0] = float(first_weight)
        weight = weight / weight.mean()
    return weight


def protection_kl(logits, reference_logprob, topk=0):
    """KL(calibrated || reference), optionally restricted to the reference's top-k.

    `topk == 0` is the historical term unchanged -- the same `log_softmax`, the same
    `exp()`ed target, the same `batchmean` -- so a run that leaves the flag alone
    computes the identical float. With `topk > 0` the reference's k most probable
    tokens keep their probability and every other entry is zeroed; the reduction is
    untouched, so this is the same quantity on a smaller support, NOT a renormalised
    one. k is clamped to the vocabulary: a k above it would otherwise ask `topk` for
    more entries than exist.
    """
    target = reference_logprob.exp().to(logits.device)
    log_prob = F.log_softmax(logits.float(), dim=-1)
    if int(topk) <= 0:
        return F.kl_div(log_prob, target, reduction="batchmean")
    k = min(int(topk), int(target.shape[-1]))
    indices = torch.topk(target, k, dim=-1).indices
    mask = torch.zeros_like(target).scatter_(-1, indices, 1.0)
    # `F.kl_div` reduces `target * (log target - input)` through `xlogy`, so a zeroed
    # entry contributes exactly zero instead of a `0 * -inf` nan.
    return F.kl_div(log_prob, target * mask, reduction="batchmean")


def cf_seed(args, document_id, salt=0):
    """A seed that depends on `--cf-random-seed` and the document id, nothing else.

    Derived rather than drawn from a stream: document order, `--val-every` and a
    resumed run must not change which core slot a row's graft slot is replaced with, or
    the negative side stops being comparable across arms.
    """
    digest = hashlib.sha256(f"cf:{int(args.cf_random_seed)}:{document_id}:{int(salt)}"
                            .encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 31 - 1)


def counterfactual_grafts(window, meta, document, args, graft_offsets):
    """The graft slots of the negative window under `--cf-mode random|shuffle`.

    `random` draws one of THIS document's own core slots per graft slot, with
    replacement: the negative keeps the document and loses only the selection claim, so
    the hinge asks "was this content better than the same budget of arbitrary
    carriers". `shuffle` permutes the 684 graft slots among themselves and keeps the
    content, which isolates the slot -> content assignment instead. Both are seeded from
    `--cf-random-seed` + the document id, so the same row always gets the same negative.
    """
    generator = np.random.RandomState(cf_seed(args, document["id"]))
    count = int(graft_offsets.numel())
    if args.cf_mode == "shuffle":
        order = torch.as_tensor(generator.permutation(count), device=window.device)
        return window.index_select(1, graft_offsets).index_select(1, order)
    core_offsets = torch.nonzero(meta["slot_type"] == 0, as_tuple=False).flatten()
    if int(core_offsets.numel()) < 1:
        raise ValueError("--cf-mode random needs at least one core slot to draw from")
    pick = torch.as_tensor(generator.randint(0, int(core_offsets.numel()), size=count),
                           device=window.device)
    return window.index_select(1, core_offsets.index_select(0, pick))


def document_step(model, manager, merge, scorer, joint, module, reference, initial,
                  document, donor_graft, args, train, cache):
    """One document: window, calibration, reference, and the gold answer's CE."""
    qa = dict(prompt_ids=document["query_ids"], answer_ids=document["answer_ids"])
    device = model.memory.device
    key = (document["id"], document.get("context_sha256"))
    reset(manager, initial)
    with torch.no_grad():
        write_blocks(manager, merge, chunks_of(document))
        source, ages, picked = gr.select_carriers(manager, WINDOW_BLOCKS, TARGET_BLOCKS)
        full = None
        if args.cf_gate == "full":
            # The 19-block uncompressed forward. It exists only to measure what the
            # compression COST this row, and at 4,864 slots it is twice the width of
            # the window the training step itself runs -- the single largest
            # transient in the step, carried for one boolean.
            full = cache["full"].get(key)
            if full is None:
                full_logits, full_gold = answer_logits(model, qa, device)
                full = dict(logp=float(answer_logp(full_logits, full_gold)))
                cache["full"][key] = full
                del full_logits, full_gold
        window, meta, telemetry = joint.build(source, picked, manager.block)
        reference_payload = cache["reference"].get(key)
        if reference_payload is None:
            reference_window, _, _ = calibrate_with_reg(window, meta, reference,
                                                        source=source)
            with read_window(model, manager, window, reference_window):
                logits, gold = answer_logits(model, qa, device)
            per_token = -F.cross_entropy(logits, gold, reduction="none")
            reference_payload = dict(logp=float(answer_logp(logits, gold)),
                                     logprob=F.log_softmax(logits.float(), dim=-1),
                                     mean_per_token=float(per_token.mean()))
            # Bounded: the cached tensor is a [answer tokens, 128k] float32 slice, and
            # holding one per document would be hundreds of MiB by the end of a run
            # for no benefit -- the cache only ever hits on a repeated document.
            cache["reference"][key] = reference_payload
            cache["reference_order"].append(key)
            while len(cache["reference_order"]) > 8:
                cache["reference"].pop(cache["reference_order"].pop(0), None)
            del logits, reference_window
    # Two ways to decide which rows the counterfactual term is worth computing on.
    # `full` is the original one and pays for a 4,864-slot forward per row; `reference`
    # gates on how well the compressed window ITSELF answers, which is already measured
    # for the protection term and costs nothing. `none` turns the gate off.
    if args.cf_gate == "full":
        cost = full["logp"] - reference_payload["logp"]
        qualifies = bool(cost > float(args.cf_gate_margin))
    elif args.cf_gate == "reference":
        cost = None
        qualifies = bool(reference_payload["mean_per_token"]
                         < float(args.cf_gate_logp))
    else:
        cost = None
        qualifies = True
    evidence = dict(reference_logp=reference_payload["logp"],
                    mean_per_token=reference_payload["mean_per_token"],
                    full_logp=(full["logp"] if full is not None else None),
                    cost=cost, qualifies=qualifies)

    calibrated, regulariser, info = calibrate_with_reg(window, meta, module,
                                                       source=source)
    with read_window(model, manager, window, calibrated):
        logits, gold = answer_logits(model, qa, device)
    per_token = F.cross_entropy(logits, gold, reduction="none")
    weight = answer_weight(gold, args.first_token_weight).to(per_token.dtype)
    ce = (per_token * weight).sum() / weight.sum()
    protection = protection_kl(logits, reference_payload["logprob"], args.kl_topk)
    loss = (ce + args.protect_weight * protection + args.reg_weight * regulariser)
    relation = info.get("relation") or {}
    record = dict(ce=ce, protection=protection, regulariser=regulariser, loss=loss,
                  relative_delta=info["relative_delta"],
                  folded_slots=info["folded_slots"], module_gate=info["gate"],
                  relation_mass=relation.get("retained_mass_mean"),
                  relation_sink=relation.get("sink_mass_mean"),
                  v16=info.get("v16"), **evidence)
    # `--cf-mode donor` is the historical negative -- the PREVIOUS document's graft
    # slots -- and therefore needs a previous document, so it cannot fire on update 1.
    # `random` and `shuffle` build the negative out of this document's own slots and
    # fire from the first step. The margin is `answer_logp(real) - donor_logp` in all
    # three cases; only the window the second number is read from changes.
    negative_ready = (donor_graft is not None) if args.cf_mode == "donor" else True
    if train and args.cf_weight > 0 and evidence["qualifies"] and negative_ready:
        graft_offsets = torch.nonzero(meta["slot_type"] == 1, as_tuple=False).flatten()
        donor_window = window.clone()
        if args.cf_mode == "donor":
            donor_window[:, graft_offsets] = donor_graft.to(window.dtype)
        else:
            donor_window[:, graft_offsets] = counterfactual_grafts(
                window, meta, document, args, graft_offsets)
        with torch.no_grad():
            donor_calibrated, _, _ = calibrate_with_reg(donor_window, meta, module,
                                                        source=source)
            with read_window(model, manager, donor_window, donor_calibrated):
                donor_logits, _ = answer_logits(model, qa, device)
            donor_logp = float(answer_logp(donor_logits, gold))
            del donor_logits, donor_calibrated, donor_window
        margin = answer_logp(logits, gold) - donor_logp
        counterfactual = F.relu(float(args.cf_gamma) - margin)
        loss = loss + args.cf_weight * counterfactual
        record.update(loss=loss, counterfactual=counterfactual,
                      margin=float(margin.detach()), donor_logp=donor_logp)
    if not train:
        record = {name: (float(value.detach()) if torch.is_tensor(value) else value)
                  for name, value in record.items()}
    carried = dict(id=document["id"])
    if args.cf_weight > 0:
        graft_offsets = torch.nonzero(meta["slot_type"] == 1, as_tuple=False).flatten()
        carried["grafts"] = window.index_select(1, graft_offsets).detach().clone()
    return record, carried, telemetry


# ---------------------------------------------------------------------------
# report pieces
# ---------------------------------------------------------------------------
def critical_manifest():
    return dict(
        joint=sha(ROOT / "methods/prlmc/evict_window.py"),
        calibration=sha(ROOT / "methods/prlmc/cross_block_calibration.py"),
        utility=sha(ROOT / "methods/prlmc/reader_utility.py"),
        residual=sha(ROOT / "methods/prlmc/anchor_residual.py"),
        trainer=sha(__file__))


def module_report(args, rank, hidden, anchor_path, anchor_config):
    if args.architecture == "v17":
        deep = deep_fusion_module()
        # The residual is written into every reader layer, so the band layout is the full
        # 0..31 map -- not the five CONTEXT layers, which are a different thing.
        layout = deep.band_layout(range(deep.BANDS[-1][1] + 1))
        if args.band_mode == "three" and not layout["band0_observable"]:
            print(f"WARNING --band-mode three: none of the window layers "
                  f"{layout['layers']} falls in band 0 ({deep.BANDS[0]}), so only bands "
                  f"{layout['gate_parameters_live']} carry a live gate. The 'shallow "
                  f"layers stay untouched' half of the mechanism is NOT tested by this "
                  f"run, and band_gate[0] is a constant 0.0 that never sees a gradient.",
                  file=sys.stderr, flush=True)
        objective = [name for name, on in
                     (("--kl-topk", int(args.kl_topk) > 0),
                      ("--layer-norm-weights", args.layer_norm_weights == "on"))
                     if on]
        if objective:
            print(f"WARNING {objective} change the TRAINING OBJECTIVE, not just the "
                  f"architecture. If this run differs from V13 in both, its question count "
                  f"cannot be attributed to either; keep them off for the structural "
                  f"ablation and turn them on in a second, separate run.",
                  file=sys.stderr, flush=True)
        return dict(
            schema=deep.SCHEMA, hidden=hidden, rank=rank, architecture="v17",
            max_ratio=args.max_ratio, max_ratio_core=args.max_ratio_core,
            cond_rank=args.cond_rank, k_neighbors=args.k_neighbors,
            conditions=["delta_position", "log_cluster_size", "is_graft"],
            deep_fusion=dict(
                key_channel=args.key_channel, key_ratio=args.key_ratio,
                value_ratio=args.value_ratio, band_mode=args.band_mode,
                band_gates=list(band_gates_of(args)),
                sim_floor=args.sim_floor,
                layer_ids=None,
                layer_table="identity: row i of the 32-row window is reader layer i",
                note="the residual is split into the part the reader's key map can see "
                     "(clipped at key_ratio x the slot RMS) and the part it cannot "
                     "(clipped at value_ratio x the slot RMS, None => the parent's "
                     "max_ratio), so the key side is locked while the value side is not",
                band_note="three bands over the utility layers; band 0 (lexical) is "
                          "masked to zero and keeps its parameter only so the report can "
                          "show what the mask did"),
            graft_attention=dict(
                tau=args.graft_tau, bias=args.graft_bias,
                norm_quantile=args.graft_norm_quantile,
                enabled=(float(args.graft_tau) != 1.0 or float(args.graft_bias) != 0.0
                         or float(args.graft_norm_quantile) > 0.0),
                applied_in_training=True,
                note="applied to the TRAINING window (see `damp_graft_keys`, before the "
                     "neighbour context is measured) and to every evaluation window by "
                     "`deep_fusion.calibrate`; `--v17-no-attn` on the evaluation side is "
                     "the arm that turns it off"),
            layer_norm_weights=(args.layer_norm_weights == "on"),
            initialisation=dict(
                graft_half="copied from the AnchorOnly checkpoint",
                core_half="zero, so every carrier slot is bit-identical at update 0",
                anchor_checkpoint=anchor_path, anchor_sha256=sha(anchor_path),
                anchor_rank=anchor_config.get("rank")),
            applies_to="all 2,304 slots: 1,620 core and 684 graft")
    if args.architecture == "v16":
        return dict(
            schema=br.SCHEMA, hidden=hidden, rank=rank, architecture="v16",
            max_ratio=args.max_ratio, max_ratio_core=args.max_ratio_core,
            shallow_cutoff=args.shallow_cutoff, k_neighbors=args.k_neighbors,
            conditions=["delta_position", "log_cluster_size", "is_graft"],
            relation=dict(
                candidates="one cross-block top-4 chosen from the five-layer CONSENSUS "
                           "similarity (keys centred over slots, L2-normalised)",
                bands="each band softmaxes its own similarity over those four candidates "
                      "and aggregates its own raw values -- the bands are never summed",
                margin="top-4 weighted similarity minus the mean over legal cross-block "
                       "slots (same-block positions excluded from the mean)",
                layer_to_band={str(layer): band
                               for layer, band in br.LAYER_TO_BAND.items()},
                band_layers=[list(pair) for pair in br.BAND_LAYERS]),
            gate=dict(
                definition="z = z_self + omega * eta * u, applied AFTER the projection",
                omega="sigmoid(kappa[type,band] * (margin - tau[type,band]))",
                eta="sigmoid(eta_logit[type,layer])",
                core_margin=args.core_margin, graft_margin=args.graft_margin,
                kappa_init=args.gate_kappa, eta_init=args.eta_init,
                note="no normalisation follows the product; the only bound afterwards is "
                     "the per-type residual ceiling and apply_delta"),
            initialisation=dict(
                self_branch="copied from the AnchorOnly checkpoint; core half at identity",
                cross_projection="zero, so the injection is exactly zero at update 0",
                anchor_checkpoint=anchor_path, anchor_sha256=sha(anchor_path),
                anchor_rank=anchor_config.get("rank")),
            applies_to="all 2,304 slots (1,620 core + 684 graft); cross injection only on "
                       "memory layers 10..31")
    if args.architecture == "v15":
        return dict(
            schema=hr.SCHEMA, hidden=hidden, rank=rank, architecture="v15",
            max_ratio=args.max_ratio, max_ratio_core=args.max_ratio_core,
            shallow_cutoff=args.shallow_cutoff, k_neighbors=args.k_neighbors,
            tau_init=args.sink_tau, sink_percentile=args.sink_percentile,
            conditions=["delta_position", "log_cluster_size", "is_graft"],
            relation=dict(
                per_layer="five independent top-4 relations, one per utility layer",
                sink="per-layer sink logit from the training similarity spread, dropped "
                     "after the softmax so the retained mass can be < 1",
                feature="sum_j w_ij v_j - m_i v_i, the neighbours minus the slot",
                modulation="b * (1 + tanh(W_s c)) + tanh(W_h c) on an RMS-normalised "
                           "sum of five per-layer projections",
                protection=f"memory layers 0..{max(0, args.shallow_cutoff - 1)} keep the "
                           f"self branch only"),
            initialisation=dict(
                graft_half="copied from the AnchorOnly checkpoint",
                core_half="zero, so every carrier slot is bit-identical at update 0",
                modulation="zero, so the relation branch is exactly off at update 0",
                projections="orthogonal and NOT zero: V13's zero init left the cross "
                            "channel without a gradient and needed a 30x step to escape",
                anchor_checkpoint=anchor_path, anchor_sha256=sha(anchor_path),
                anchor_rank=anchor_config.get("rank"),
                anchor_max_ratio=anchor_config.get("max_ratio")),
            applies_to="all 2,304 slots: 1,620 core and 684 graft, memory layers 10..31")
    return dict(
        schema=FULL_POOL_SCHEMA, hidden=hidden, rank=rank, max_ratio=args.max_ratio,
        cond_rank=args.cond_rank,
        trainable_parameters=int(4 * rank * hidden + 2 * rank * args.cond_rank
                                 + 3 * args.cond_rank + 3),
        conditions=["delta_position", "log_cluster_size", "is_graft"],
        neighbour=dict(mode=args.neighbor_mode, k=args.k_neighbors,
                       k_same=(args.k_same if args.neighbor_mode == "complement" else None),
                       k_cross=(args.k_cross if args.neighbor_mode == "complement"
                                else None),
                       use_cross=(args.cross == "on"),
                       note="topk asks for the k most associated cross-block slots, "
                            "kind-blind (V13); complement asks for k_same of the "
                            "query's own core/graft kind and k_cross from the other "
                            "side, with equal per-group weight mass (V14)"),
        initialisation=dict(
            graft_half="copied from the AnchorOnly checkpoint, so the measured "
                       "calibration is the starting point rather than a target",
            core_half="zero, so every carrier slot is bit-identical at update 0",
            film="zero, so the condition cannot move a slot before it is trained",
            anchor_checkpoint=anchor_path, anchor_sha256=sha(anchor_path),
            anchor_rank=anchor_config.get("rank"),
            anchor_max_ratio=anchor_config.get("max_ratio")),
        applies_to="all 2,304 output slots: 1,620 core and 684 graft")


def protection_report(args):
    """The protection term, including whether it is restricted to the reference's head.

    A key is only added when `--kl-topk` is on, so the report of a run that leaves it
    alone is the report those runs have always produced.
    """
    payload = dict(weight=args.protect_weight,
                   reference="the V11 window with the initial module, i.e. the "
                             "AnchorOnly calibration on the graft slots and an "
                             "untouched core",
                   note="KL is zero there, so the term cannot push the run to "
                        "undo its own baseline")
    if int(args.kl_topk) > 0:
        payload["top_k"] = int(args.kl_topk)
        payload["top_k_note"] = ("the KL is restricted to the reference distribution's k "
                                 "most probable tokens; the remaining entries are zeroed "
                                 "and the batchmean reduction is unchanged, so the "
                                 "support shrinks and the term is not renormalised")
    return payload


def objective_report(args):
    return dict(
        loss="CE(answer) + w_cf*CF + w_protect*KL + w_reg*R",
        first_token_weight=args.first_token_weight,
        counterfactual=dict(
            weight=args.cf_weight, gamma=args.cf_gamma,
            gate=args.cf_gate,
            gate_definition=("logP(Full) - logP(window) > margin" if args.cf_gate == "full"
                             else "the compressed window's own mean gold log-prob per "
                                  "token is below the threshold, i.e. it is not "
                                  "answering this row"
                             if args.cf_gate == "reference" else "off, every row"),
            gate_logp=args.cf_gate_logp, gate_margin=args.cf_gate_margin,
            negative_window=(
                "the same window with the graft slots drawn at random from the SAME "
                "document's core slots (seed = --cf-random-seed + document id)"
                if args.cf_mode == "random" else
                "the same window with the graft slots permuted among themselves "
                "(seed = --cf-random-seed + document id)"
                if args.cf_mode == "shuffle" else
                "the same window with the graft slots taken from the previous document"),
            negative_gradient="the donor window is scored under no_grad, so the hinge "
                              "is one-sided pressure on the true window"),
        protection=protection_report(args),
        regulariser=dict(weight=args.reg_weight,
                         definition="mean over slots of (||dh||/rms(h))^2"),
        answers_seen=True)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def train(args):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for flag, value, what in (("--checkpoint", args.checkpoint, "the frozen operator"),
                              ("--anchor", args.anchor, "the AnchorOnly checkpoint")):
        if not value or not Path(value).is_file():
            raise SystemExit(f"this run needs {what}: pass {flag}")
    # V13 and V14 may differ in the neighbour rule and nothing else, so the rule is
    # validated here rather than discovered at the end of a seven-hour run.
    if args.neighbor_mode == "complement":
        if int(args.k_same) + int(args.k_cross) != int(args.k_neighbors):
            raise SystemExit(f"--neighbor-mode complement reads k = k_same + k_cross: "
                             f"{args.k_same} + {args.k_cross} != {args.k_neighbors}")
    elif args.cross == "on" and (int(args.k_same), int(args.k_cross)) != (2, 2):
        raise SystemExit("--k-same/--k-cross only apply to --neighbor-mode complement")
    if args.architecture == "v16":
        if args.cross != "on":
            raise SystemExit("--architecture v16 is the cross-block mechanism: pass "
                             "--cross on")
        if args.neighbor_mode != "topk":
            raise SystemExit("--neighbor-mode only applies to V13; V16 picks its "
                             "candidates from the five-layer consensus")
        if not 0 <= int(args.shallow_cutoff) < br.NUM_MEMORY_LAYERS:
            raise SystemExit(f"--shallow-cutoff must be in [0, {br.NUM_MEMORY_LAYERS})")
    if args.architecture == "v15":
        if args.cross != "on":
            raise SystemExit("--architecture v15 is the cross-block mechanism: pass "
                             "--cross on")
        if args.neighbor_mode != "topk":
            raise SystemExit("--neighbor-mode only applies to V13; V15 builds five "
                             "per-layer top-k relations of its own")
    if args.architecture == "v17":
        # Fail here, before the reader is loaded, rather than eight hours into a run:
        # the module and the band gates have to be resolvable at argument time.
        deep_fusion_module()
        band_gates_of(args)
        if args.neighbor_mode == "complement" and args.cross != "on":
            raise SystemExit("--architecture v17 with --neighbor-mode complement needs "
                             "--cross on")
    if int(args.kl_topk) < 0:
        raise SystemExit("--kl-topk is a token count; 0 disables the restriction")
    if int(args.min_core) < 0:
        raise SystemExit("--min-core cannot be negative")
    if args.value_ratio is not None and float(args.value_ratio) <= 0:
        raise SystemExit("--value-ratio is a fraction of the slot RMS, so it must be > 0")
    switched = nondefault_switches(args)
    if args.architecture != "v17":
        silent = [item.split("=")[0] for item in switched
                  if item.split("=")[0] in MODULE_BOUND_SWITCHES]
        if silent:
            raise SystemExit(f"--{silent} exist only inside --architecture v17: on "
                             f"{args.architecture} they would train the old module and "
                             "then report a null result for a mechanism that was never "
                             "switched on")
    if switched:
        print(json.dumps({"non_default_switches": switched}, ensure_ascii=False),
              flush=True)
    if args.cross == "off" and args.neighbor_mode != "topk":
        raise SystemExit("--cross off has no neighbour rule at all: pass "
                         "--neighbor-mode topk")
    if Path(args.out).exists():
        raise SystemExit(f"{args.out} exists; refuse overwrite")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    documents, counts = load_documents(tokenizer, args.data_dir, args.train_task,
                                       args.max_docs)
    if len(documents) < 4:
        raise SystemExit(f"only {len(documents)} usable {args.train_task} documents")
    generator = np.random.RandomState(args.seed)
    order = generator.permutation(len(documents))
    val_count = min(args.val_docs, len(documents) // 4)
    validation = [documents[index] for index in order[:val_count]]
    training = [documents[index] for index in order[val_count:]]
    if args.smoke:
        training, validation = training[:2], validation[:2]
    probe_pool = load_documents(tokenizer, args.data_dir,
                                args.probe_task or args.train_task, None)[0]
    probes, probe_sources = build_probes(args, tokenizer, probe_pool)

    if args.dry_run:
        print(json.dumps(dict(
            dry_run=True, out=args.out, report=args.report, counts=counts,
            train_documents=len(training), validation_documents=len(validation),
            probes=len(probes),
            quota=dict(core_per_block=args.core_per_block,
                       graft_per_block=args.graft_per_block,
                       core_total=TARGET_BLOCKS * args.core_per_block,
                       graft_total=TARGET_BLOCKS * args.graft_per_block,
                       per_block_total=args.core_per_block + args.graft_per_block,
                       budget=TARGET_SLOTS),
            updates=args.updates, grad_accum=args.grad_accum,
            document_forwards=args.updates * args.grad_accum,
            cost_note="one document step is three reader forwards; --grad-accum 8 is "
                      "the plan's effective batch of 8 and costs 8x the wall clock"),
            indent=2, ensure_ascii=False))
        return None
    if args.core_per_block + args.graft_per_block != SLOTS_PER_BLOCK:
        raise SystemExit("core + graft per block must fill one 256-slot block")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.cuda_device))
    nv.seed_everything(args.seed)
    checkpoint_training = torch.load(args.checkpoint, map_location="cpu").get("training")
    model, tokenizer, manager, merge, initial, cfg, saved = load_runtime(
        args.model_path, checkpoint=args.checkpoint, train=False, architecture=None,
        budget_blocks=BUDGET_BLOCKS, strict_code_hash=args.strict_code_hash,
        training=checkpoint_training)
    merge.operator_mode = "native"
    for dropout in model.modules():
        if isinstance(dropout, nn.Dropout):
            dropout.p = 0.0
    model.train()
    if not args.no_reader_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.model.gradient_checkpointing = True
        model.model._gradient_checkpointing_func = _checkpoint_func()

    layers = tuple(int(x) for x in str(args.utility_layers).split(",") if x.strip())
    scorer = ReaderScorer(model, layers)
    bank = [scorer.probe_bank(ids) for ids in probes]
    joint = JointConsolidation(
        scorer, bank, core_per_block=args.core_per_block,
        graft_per_block=args.graft_per_block, policy="evict",
        mmr_penalty=args.mmr_penalty, novelty_weight=args.novelty_weight,
        **_joint_extra(args))
    anchor_state, anchor_config = load_anchor_state(args.anchor, manager.hidden)
    module, rank = build_module(args, manager.hidden, anchor_state, anchor_config)
    module = module.to(model.memory.device).float()
    # Mirrors the trained module exactly, including `--cross off`: a reference built
    # with different constructor args cannot take the state dict under strict=True.
    reference = type(module)(**module.config())
    reference.load_state_dict(module.state_dict(), strict=True)
    reference = reference.to(model.memory.device).eval().requires_grad_(False)

    # ---- V17: the key map's row space, built ONCE ------------------------------
    # `build_key_bases` folds the LoRA adapter into every layer's key map, runs one SVD
    # per layer and verifies the fold against the real `k_proj`; at 32 layers that is
    # minutes, so it happens here -- not per document, and not twice for the module and
    # its frozen mirror. Both get the basis when the forward needs it: the reference
    # runs the same `key_channel == "project"` path and would raise without one.
    key_basis_info = None
    if args.architecture == "v17" and needs_key_basis(args):
        key_basis, key_report = ks.build_key_bases(model)
        if str(args.key_channel) == "project":
            for target in (module, reference):
                install_basis(deep_fusion_module(), target, key_basis)
        key_basis_info = dict(
            installed=(str(args.key_channel) == "project"),
            basis_digest=ks.basis_digest(key_basis),
            shape=[int(size) for size in key_basis.shape],
            report=key_report,
            note="the row space of the LoRA-folded key map per memory layer; the graft "
                 "attention damping reads the same basis, which is why it is built even "
                 "when --key-channel stays off")
        print(json.dumps({"key_basis": {name: value
                                        for name, value in key_basis_info.items()
                                        if name != "report"}}), flush=True)

    # `cross_proj` starts at exactly zero, so at `--lr` it only ever reaches a scale
    # that is lost inside the residual it is supposed to steer: the first V13 run
    # trained it to norm 1.25 while the residual moved 0.5% against the plain
    # calibration, and not one of the 158 answers changed. It is therefore given its
    # own, larger step so the mechanism gets a fair test instead of a silent zero.
    # The cross/relation channel is zero- or small-started and cannot reach a useful
    # scale at the shared learning rate (V13 needed 30x). V15 gives the five per-layer
    # projections and the two modulation matrices their own, larger step.
    relation_names = ("cross_proj", "evict_proj", "mod_scale", "mod_shift", "cross_norm",
                      "band_proj", "layer_proj", "gate_kappa", "gate_tau", "eta_logit")
    cross_proj = [p for name, p in module.named_parameters()
                  if p.requires_grad and name.startswith(relation_names)]
    shared = [p for name, p in module.named_parameters()
              if p.requires_grad and not name.startswith(relation_names)]
    groups = [dict(params=shared, lr=args.lr)]
    if cross_proj:
        relation_lr = (args.relation_lr
                       if args.architecture in ("v15", "v16") else args.cross_lr)
        groups.append(dict(params=cross_proj, lr=relation_lr))
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    trainable = int(sum(p.numel() for p in module.parameters() if p.requires_grad))
    window_budget = _joint_extra(args)
    geometry = dict(core_per_block=args.core_per_block,
                    graft_per_block=args.graft_per_block,
                    core_total=TARGET_BLOCKS * args.core_per_block,
                    graft_total=TARGET_BLOCKS * args.graft_per_block,
                    budget=TARGET_SLOTS, cross=(args.cross == "on"),
                    joint_schema=JOINT_SCHEMA, trainable_parameters=trainable)
    if args.architecture == "v17":
        geometry["window_budget"] = dict(
            quota_mode=args.quota_mode, min_core=int(args.min_core),
            block_merge_cos=float(args.block_merge_cos),
            forwarded_to_joint_consolidation=sorted(window_budget),
            note="a switch that keeps its pre-V17 default is not forwarded at all, so a "
                 "V13 run builds the window with the call it has always made")
    if key_basis_info is not None:
        geometry["key_basis"] = key_basis_info
    report = dict(
        schema=FULL_POOL_SCHEMA + "/training", status="running",
        data=dict(dir=args.data_dir, task=args.train_task,
                  sha256=sha(Path(args.data_dir) / (args.train_task + ".jsonl")),
                  counts=counts, train_documents=len(training),
                  validation_documents=len(validation),
                  validation_ids=[document["id"] for document in validation],
                  evaluation_task_never_written="hotpotqa"),
        reader=dict(model_path=args.model_path, frozen=True,
                    checkpoint=args.checkpoint, checkpoint_sha256=sha(args.checkpoint),
                    checkpoint_code_hash_verified=(saved or {}).get("code_hash_verified")),
        module=module_report(args, rank, manager.hidden, args.anchor, anchor_config),
        geometry=geometry,
        probe=dict(sources=probe_sources, layers=list(layers), count=len(probes),
                   digest=tensor_digest(torch.tensor(
                       [token for probe in probes for token in probe], dtype=torch.long)),
                   note="a fixed bank encoded before the first test row is read; no "
                        "test question and no reference answer enters the score"),
        objective=objective_report(args),
        optimizer=dict(name="AdamW", lr=args.lr, cross_lr=args.cross_lr,
                       relation_lr=(args.relation_lr
                                    if args.architecture in ("v15", "v16")
                                    else args.cross_lr),
                       weight_decay=args.weight_decay,
                       grad_clip=args.grad_clip, updates=args.updates,
                       grad_accum=args.grad_accum),
        critical_manifest=critical_manifest(), history=[], validation=[],
        peak_allocated_bytes=0)
    # Written only when something was switched on, so a run that left every flag alone
    # produces the report those runs have always produced.
    if switched:
        report["non_default_switches"] = switched
    if args.architecture == "v17":
        # The mechanism lives in someone else's file, so the hash belongs in the
        # checkpoint's manifest; kept out of `critical_manifest()` itself so a V13 run's
        # manifest stays the five entries it has always had.
        report["critical_manifest"]["deep_fusion"] = sha(
            ROOT / "methods/prlmc/deep_fusion.py")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.report, report)

    # ---- identity: the initial module must reproduce its own baseline ----------
    with torch.no_grad():
        reset(manager, initial)
        write_blocks(manager, merge, chunks_of(training[0]))
        source, ages, picked = gr.select_carriers(manager, WINDOW_BLOCKS, TARGET_BLOCKS)
        window, meta, telemetry = joint.build(source, picked, manager.block)
        # `--layer-norm-weights on` needs the buffer before the pass below, so the pass
        # that measures `D_init` runs with a weight of one everywhere -- which is exactly
        # the unweighted regulariser, so the identity check still measures what it always
        # measured.
        if args.layer_norm_weights == "on":
            ensure_layer_norm_weights(module, int(window.shape[0]))
        calibrated, regulariser, info = calibrate_with_reg(
            window, meta, module, source=source,
            want_per_layer=(args.layer_norm_weights == "on"))
        core = meta["slot_type"] == 0
        graft = meta["slot_type"] == 1
        core_identical = bool(torch.equal(calibrated[:, core], window[:, core]))
        anchor = module.anchor_view().to(window.device).float()
        anchor_delta = anchor(window[:, graft])

        # ---- what this module MUST write at update 0 --------------------------------
        # Derived from the mechanisms this run actually switched on, read off the module
        # rather than off `--architecture`. The earlier version keyed the whole rule on the
        # architecture name and then demanded the band mask on a run where `--band-mode` was
        # off: it failed at 3.86e-02 against a 4.0e-01 ceiling, which is the number being
        # fine and the rule being wrong.
        band_on = str(getattr(module, "band_mode", "off")) == "three"
        key_on = str(getattr(module, "key_channel", "off")) == "project"
        if band_on or key_on:
            # `init_delta` applies the same mask and the same clip the forward applies, in
            # the module that implements them, so this expectation cannot drift away from
            # the forward the way a second copy of the rules would.
            anchor_delta = deep_fusion_module().init_delta(module, window[:, graft],
                                                           anchor_delta)
        expected_graft = apply_delta(window[:, graft], anchor_delta)
        # A tolerance rather than bit equality: the graft half is measured here on a gathered
        # copy and inside the training loop on a strided view of the same slots, and the
        # per-slot RMS reduction rounds differently between the two layouts. Measured on the
        # smoke run: 4.4e-3, i.e. fp16 storage noise on the delta.
        deviation = ((calibrated[:, graft].float() - expected_graft.float()).abs()
                     / window[:, graft].float().square().mean(dim=-1, keepdim=True)
                     .sqrt().clamp_min(1e-6))
        graft_deviation = float(deviation.max())

        # The ceiling is storage noise, widened ONLY by a transform that genuinely bounds the
        # residual -- the key/value clip. The band mask must not widen it: `init_delta` zeroes
        # exactly the rows the forward zeroes, so the expectation stays exact and a broken
        # mask would slide through a loose tolerance instead of being caught.
        deviation_bound = 1e-2
        allowances = ["fp16 storage noise"]
        if key_on:
            deviation_bound = float(module.key_ratio + module.value_ratio_effective)
            allowances.append("key_ratio + value_ratio")
        identity_rule = ("the graft half reproduces the delta this module must write at "
                         "update 0, within " + " + ".join(allowances))

        # Masked rows are asserted separately, and only when a mask exists. A nonzero
        # residual on a row whose gate is zero means the mask never reached the forward,
        # which is the failure this check exists for.
        masked_movement = None
        masked_layers = None
        if band_on:
            deep = deep_fusion_module()
            gate = (module.band_gate.detach().float()
                    * module.band_open.to(module.band_gate.device))
            bands = torch.as_tensor([deep.band_of_layer(module._layer_at(position))
                                     for position in range(int(window.shape[0]))],
                                    dtype=torch.long, device=gate.device)
            masked = (gate[bands] == 0)
            masked_layers = int(masked.sum())
            if bool(masked.any()):
                masked_movement = float(
                    (calibrated[:, graft][masked].float()
                     - window[:, graft][masked].float()).abs().max())
        graft_identical = (graft_deviation <= deviation_bound
                           and (masked_movement is None or masked_movement <= 1e-2))
        at_origin = float(regulariser.detach())
    if not core_identical:
        raise ValueError("the initial module moved a core slot: the core half is not "
                         "the identity, so the run does not start on its baseline")
    if not graft_identical:
        # A gate again, for every architecture. The earlier V17 "warning" existed because
        # the expectation was the raw anchor residual, which the band mask contradicts by
        # construction; `deep_fusion.init_delta` now applies the same mask and clip the
        # forward applies, so agreement is once more the thing worth asserting. A run that
        # fails here has a real defect and says so instead of training on.
        raise ValueError(
            f"the initial module does not reproduce the expected graft delta: "
            f"max relative deviation {graft_deviation:.3e} against a ceiling of "
            f"{deviation_bound:.3e}"
            + (f", masked-layer movement {masked_movement:.3e} over {masked_layers} rows"
               if masked_movement is not None else "")
            + f" ({identity_rule})")
    identity = dict(document=training[0]["id"], core_bit_identical=core_identical,
                    graft_matches_anchor_only=graft_identical,
                    identity_rule=identity_rule,
                    band_mask_active=band_on,
                    key_split_active=key_on,
                    masked_layers=(masked_layers),
                    masked_layers_bit_untouched_max=(masked_movement),
                    deviation_ceiling=(deviation_bound),
                    graft_max_relative_deviation=graft_deviation,
                    regulariser_at_origin=at_origin, core_slots=info["core_slots"],
                    graft_slots=info["graft_slots"], folded_slots=info["folded_slots"],
                    merge=telemetry["merge"], probe_passed=True)
    report["identity_check"] = identity

    # ---- V17: per-layer regulariser weights, placed from update 0 and then frozen ----
    # Measured on the pass ABOVE -- the module at update 0, i.e. the AnchorOnly graft
    # calibration and an untouched core -- because the weight is meant to describe where
    # the run starts, not where it has drifted to. Written once and never updated: a
    # weight that keeps moving turns the regulariser into a moving target, and the point
    # of `D_l` is that it is the residual the layer began with.
    if args.layer_norm_weights == "on":
        weights = layer_norm_weights_from(info["per_layer_relative_delta"])
        with torch.no_grad():
            module.layer_norm_weights.copy_(torch.as_tensor(
                weights, dtype=torch.float32,
                device=module.layer_norm_weights.device))
        report["layer_norm_weights"] = dict(
            weights=weights,
            per_layer_relative_delta=list(info["per_layer_relative_delta"]),
            definition="w_l = 1 / (D_init_l + 1e-6), normalised to mean 1, with D_init_l "
                       "the layer's relative delta measured on update 0",
            mean=float(np.mean(weights)), minimum=float(np.min(weights)),
            maximum=float(np.max(weights)),
            note="switched the regulariser from a sum over all layers to a weighted "
                 "per-layer sum; frozen after this point")
        print(json.dumps({"layer_norm_weights": report["layer_norm_weights"]}),
              ensure_ascii=False, flush=True)

    # ---- V15: the sink threshold comes from the data, not from a constant ----
    # A fixed 0.35 would put the gate somewhere nobody checked; the threshold is placed at
    # a per-layer percentile of that layer's own top-k similarity spread, measured on
    # training documents only, and the retained mass is re-measured afterwards so the run
    # records whether the gate is actually closed for weak neighbourhoods.
    if args.architecture == "v15":
        per_document = []
        with torch.no_grad():
            for source_document in training[: max(1, int(args.sink_documents))]:
                reset(manager, initial)
                write_blocks(manager, merge, chunks_of(source_document))
                sink_source, _, sink_picked = gr.select_carriers(manager, WINDOW_BLOCKS,
                                                                 TARGET_BLOCKS)
                sink_window, sink_meta, _ = joint.build(sink_source, sink_picked,
                                                        manager.block)
                _, sink_report = module.relations(sink_window, sink_meta,
                                                  return_report=True)
                key = f"similarity_p{int(args.sink_percentile)}"
                per_document.append([sink_report["per_layer"][f"utility_layer_{layer}"][key]
                                     for layer in range(module.kv_layers)])
                del sink_window, sink_source
        thresholds = [float(np.mean([row[layer] for row in per_document]))
                      for layer in range(module.kv_layers)]
        module.init_sink(thresholds)
        with torch.no_grad():
            _, after = module.relations(window, meta, return_report=True)
        report["sink_init"] = dict(
            percentile=args.sink_percentile, documents=len(per_document),
            tau=[float(x) for x in module.tau.reshape(-1)],
            thresholds=thresholds,
            retained_mass_after=[after["per_layer"][f"utility_layer_{layer}"]
                                 ["retained_mass"] for layer in range(module.kv_layers)],
            sink_mass_after=[after["per_layer"][f"utility_layer_{layer}"]
                             ["sink_mass"] for layer in range(module.kv_layers)],
            note="a retained mass near 1 everywhere would mean the gate is decorative")
        print(json.dumps({"sink_init": report["sink_init"]}), flush=True)
        reset(manager, initial)
    print(json.dumps({"identity_check": identity}, ensure_ascii=False), flush=True)

    cache = dict(full={}, reference={}, reference_order=[])
    donor_graft = None
    started = time.monotonic()
    update = 0
    qualified = 0
    best = None
    while update < args.updates:
        optimizer.zero_grad(set_to_none=True)
        recorded = []
        for accumulated in range(args.grad_accum):
            document = training[(update * args.grad_accum + accumulated) % len(training)]
            measured, carried, telemetry = document_step(
                model, manager, merge, scorer, joint, module, reference, initial,
                document, donor_graft, args, True, cache)
            # The 3.12 GiB memory gradient has to be one contiguous block, and the
            # three no_grad forwards above leave freed-but-reserved blocks in the
            # allocator's cache. Handing them back to the driver first is what turns
            # "262 MiB reserved but unallocated" into usable room.
            torch.cuda.empty_cache()
            (measured["loss"] / args.grad_accum).backward()
            donor_graft = carried["grafts"] if args.cf_weight > 0 else None
            recorded.append({name: (float(value.detach()) if torch.is_tensor(value)
                                    and value.dim() == 0 else value)
                             for name, value in measured.items()})
            if measured["qualifies"]:
                qualified += 1
        update += 1
        gradients = {name: parameter.grad for name, parameter in
                     module.named_parameters() if parameter.grad is not None}
        required = STAGE1 if update == 1 else STAGE2
        if args.architecture == "v15":
            required = (STAGE1 + V15_STAGE1) if update == 1 else (STAGE2 + V15_STAGE2)
        elif args.architecture == "v16":
            # Step 1: the zero projections must start learning. Step 2 on: the gates,
            # which cannot see a gradient while the projections are exactly zero.
            required = (STAGE1 + V16_STAGE1) if update == 1 else (STAGE2 + V16_STAGE2)
        missing = [name for name in required
                   if name not in gradients or not torch.isfinite(gradients[name]).all()
                   or float(gradients[name].abs().max()) == 0.0]
        if missing:
            raise ValueError(f"no gradient on {missing} at update {update}: the call is "
                             f"not attached to the loss")
        norm = float(torch.nn.utils.clip_grad_norm_(
            [p for p in module.parameters() if p.requires_grad], args.grad_clip))
        optimizer.step()
        measured = recorded[-1]
        row = dict(update=update, loss=measured["loss"], ce=measured["ce"],
                   protection=measured["protection"],
                   regulariser=measured["regulariser"],
                   module_gate=measured["module_gate"],
                   delta=measured["relative_delta"], cost=measured["cost"],
                   qualifies=measured["qualifies"], grad_norm=norm,
                   grad_up_core=float(gradients["up_core"].norm()),
                   grad_cond_down=(float(gradients["cond_down"].norm())
                                   if "cond_down" in gradients else None),
                   grad_relation=(float(sum(
                       gradients[name].norm() for name in V15_STAGE2 + V15_STAGE1
                       if name in gradients))
                       if args.architecture == "v15" else (
                           float(sum(gradients[name].norm()
                                     for name in ("band_proj.0.weight",
                                                  "layer_proj.0.weight",
                                                  "gate_kappa", "gate_tau", "eta_logit")
                                     if name in gradients))
                           if args.architecture == "v16" else None)),
                   v16_gate_core=(measured.get("v16") or {}).get("gate_core"),
                   v16_gate_graft=(measured.get("v16") or {}).get("gate_graft"),
                   v16_band_proj=(measured.get("v16") or {}).get("band_proj_norm"),
                   v16_injection_rms=(measured.get("v16") or {}).get("injection_rms"),
                   v16_shallow_leak=(measured.get("v16") or {}).get("shallow_leak"),
                   relation_mass=measured.get("relation_mass"),
                   relation_sink=measured.get("relation_sink"),
                   seconds=time.monotonic() - started)
        if "counterfactual" in measured:
            row["counterfactual"] = measured["counterfactual"]
            row["margin"] = measured["margin"]
        report["history"].append(row)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                          for k, v in row.items()}), flush=True)
        if args.val_every and validation and (update % args.val_every == 0
                                              or update == args.updates):
            scores = []
            for held in validation:
                record, _, held_telemetry = document_step(
                    model, manager, merge, scorer, joint, module, reference, initial,
                    held, None, args, False, cache)
                scores.append((record, held_telemetry))
            summary = dict(
                update=update,
                ce=float(np.mean([record["ce"] for record, _ in scores])),
                reference_logp=float(np.mean([record["reference_logp"]
                                              for record, _ in scores])),
                mean_per_token=float(np.mean([record["mean_per_token"]
                                              for record, _ in scores])),
                merge_evicting_blocks=int(np.mean([
                    telemetry["merge"]["blocks_where_quota_came_from_eviction"]
                    for _, telemetry in scores])))
            if scores and scores[0][0]["full_logp"] is not None:
                summary["full_logp"] = float(np.mean([record["full_logp"]
                                                      for record, _ in scores]))
                summary["cost"] = summary["full_logp"] - summary["reference_logp"]
            report["validation"].append(summary)
            # Keep the module at its best validation CE: 384 updates on a few hundred
            # documents is long enough that the last step is not the best one, and the
            # selected state costs one 10 MiB copy held in host memory.
            if (args.select_best == "val_ce" and "ce" in summary
                    and (best is None or summary["ce"] < best["ce"])):
                best = dict(update=update, ce=float(summary["ce"]),
                            state_dict={name: tensor.detach().cpu().clone()
                                        for name, tensor in module.state_dict().items()})
            report["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            if args.report:
                atomic_json(args.report, report)
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                              for k, v in summary.items()}), flush=True)

    report["cf_effective_steps"] = qualified
    report["cf_effective_fraction"] = round(qualified / max(args.updates, 1), 4)
    report["elapsed_seconds"] = time.monotonic() - started
    report["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    report["status"] = "complete"
    payload_geometry = dict(cross=(args.cross == "on"), neighbor_mode=args.neighbor_mode,
                            architecture=args.architecture,
                            shallow_cutoff=(args.shallow_cutoff
                                            if args.architecture in ("v15", "v16")
                                            else None),
                            layer_to_band=({str(layer): band
                                            for layer, band in br.LAYER_TO_BAND.items()}
                                           if args.architecture == "v16" else None),
                            band_layers=([list(pair) for pair in br.BAND_LAYERS]
                                         if args.architecture == "v16" else None),
                            core_per_block=args.core_per_block,
                            graft_per_block=args.graft_per_block,
                            key_threshold=args.key_threshold,
                            value_threshold=args.value_threshold,
                            max_cluster=args.max_cluster,
                            blend_weight=args.blend_weight,
                            mmr_penalty=args.mmr_penalty,
                            novelty_weight=args.novelty_weight,
                            utility_layers=list(layers), probe_ids=probes,
                            probe_sources=probe_sources)
    # V17 inference knobs that are not constructor arguments, so `module.config()` cannot
    # carry them: the graft-attention numbers and the window budget the arm was built
    # with. Written only for the V17 runs, so a V13 checkpoint keeps its exact geometry.
    if args.architecture == "v17":
        effective_value_ratio = (float(args.max_ratio) if args.value_ratio is None
                                 else float(args.value_ratio))
        payload_geometry["graft_attention"] = dict(
            tau=args.graft_tau, bias=args.graft_bias,
            norm_quantile=args.graft_norm_quantile,
            # `norm_quantile` damps on its own, so leaving it out of `enabled` would
            # record a checkpoint that visibly damps as "not enabled".
            enabled=(float(args.graft_tau) != 1.0 or float(args.graft_bias) != 0.0
                     or float(args.graft_norm_quantile) > 0.0),
            applied_in_training=True,
            site="the training window, before the neighbour context is measured; and "
                 "every evaluation window, inside deep_fusion.calibrate",
            note="`training_objective_unchanged`: the damping is applied to the window "
                 "the loss reads, it does not change the loss")
        payload_geometry["window_budget"] = dict(
            quota_mode=args.quota_mode, min_core=int(args.min_core),
            block_merge_cos=float(args.block_merge_cos),
            forwarded_to_joint_consolidation=sorted(window_budget))
        payload_geometry["objective_switches"] = dict(
            kl_topk=int(args.kl_topk),
            layer_norm_weights=(args.layer_norm_weights == "on"),
            cf_mode=args.cf_mode, cf_random_seed=int(args.cf_random_seed),
            changes_the_objective=[name for name, on in
                                   (("kl_topk", int(args.kl_topk) > 0),
                                    ("layer_norm_weights",
                                     args.layer_norm_weights == "on")) if on])
        payload_geometry["value_ratio_effective"] = effective_value_ratio
        payload_geometry["value_ratio_note"] = (
            "the number the value side actually clips at; `value_ratio=None` means "
            "`max_ratio`, which is the CLI default 0.34 and NOT this constructor's 0.5. "
            "A sweep of explicit ratios has to include this value or it is comparing "
            "against an unlisted arm")
        run_layers = list(range(deep_fusion_module().BANDS[-1][1] + 1))
        payload_geometry["band_layout"] = deep_fusion_module().band_layout(
            range(deep_fusion_module().BANDS[-1][1] + 1))
        buffer_weights = dict(module.named_buffers()).get("layer_norm_weights")
        payload_geometry["layer_norm_weights"] = dict(
            enabled=(args.layer_norm_weights == "on"),
            values=(torch.ones(len(run_layers)).tolist() if buffer_weights is None
                    else [float(value) for value in
                          buffer_weights.detach().float().reshape(-1).tolist()]),
            persisted=True,
            note="w_l = 1/(D_init_l + 1e-6) normalised to mean 1, frozen after update 0. "
                 "The module buffer is non-persistent on purpose (a persistent one would "
                 "break every earlier checkpoint's strict load), so the trained numbers "
                 "are copied into the checkpoint here")
    if key_basis_info is not None:
        payload_geometry["key_basis"] = key_basis_info
    payload = dict(
        schema=(br.SCHEMA if args.architecture == "v16" else
                hr.SCHEMA if args.architecture == "v15" else
                deep_fusion_module().SCHEMA if args.architecture == "v17" else
                FULL_POOL_SCHEMA)
        + "/checkpoint",
        # `module.config()` for EVERY architecture that has one: a hand-written subset
        # silently drops the flags added later (span_value / context_diffusion / csls /
        # use_evicted, and now the whole V17 block), and a checkpoint that cannot
        # describe itself evaluates as a different mechanism.
        config=(module.config() if args.architecture in ("v15", "v16", "v17") else
                dict(hidden_size=manager.hidden, rank=rank, max_ratio=args.max_ratio,
                    gate_init=args.gate_init, cond_size=3, cond_rank=args.cond_rank,
                    # recorded so the checkpoint can be rebuilt: without these the
                    # loader defaults to use_cross=True and a --cross off checkpoint
                    # cannot be loaded at all
                    use_cross=(args.cross == "on"), k_neighbors=args.k_neighbors,
                    neighbor_mode=args.neighbor_mode, k_same=args.k_same,
                    k_cross=args.k_cross,
                    kv_dim=args.kv_dim, max_ratio_core=args.max_ratio_core,
                    core_weight=args.core_weight, graft_weight=args.graft_weight,
                    span_value=(args.span_value == "on"),
                    context_diffusion=(args.context_diffusion == "on"),
                    csls=(args.csls == "on"),
                    use_evicted=(args.evicted_context == "on"))),
        geometry=payload_geometry,
        state_dict={name: tensor.detach().cpu()
                    for name, tensor in module.state_dict().items()},
        training=dict(updates=args.updates, seed=args.seed, lr=args.lr,
                      elapsed_seconds=report["elapsed_seconds"],
                      cf_effective_steps=qualified, report=args.report,
                      manifest=report["critical_manifest"]))
    if args.architecture == "v17":
        # Top-level copy because the module buffer is deliberately non-persistent (a
        # persistent one adds a state_dict key and breaks strict loading of every earlier
        # checkpoint). Without this the trained per-layer weights would exist only in the
        # run's log, and "did the per-layer normalisation do anything" would be
        # unauditable after the fact.
        payload["layer_norm_weights"] = list(
            payload_geometry["layer_norm_weights"]["values"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    report["checkpoint"] = dict(path=args.out, sha256=sha(args.out))
    if best is not None:
        selected = str(Path(args.out).with_name(Path(args.out).stem + "_best.pt"))
        selected_payload = dict(payload)
        selected_payload["state_dict"] = best["state_dict"]
        selected_payload["training"] = dict(payload["training"],
                                            selected_update=best["update"],
                                            selected_validation_ce=best["ce"],
                                            selected_of_updates=args.updates)
        torch.save(selected_payload, selected)
        report["selection"] = dict(metric="val_ce", best_update=best["update"],
                                   best_validation_ce=best["ce"], path=selected,
                                   sha256=sha(selected),
                                   note="same architecture; pick this file when the last "
                                        "update is not the best one")
        print(json.dumps({"selection": report["selection"]}), flush=True)
    if args.report:
        atomic_json(args.report, report)
    print(json.dumps(dict(status="complete", out=args.out, report=args.report,
                          updates=args.updates, cf_effective_steps=qualified,
                          elapsed_seconds=round(report["elapsed_seconds"], 1),
                          peak_allocated_bytes=report["peak_allocated_bytes"]),
                     indent=1, ensure_ascii=False))
    return report


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--train-task", dest="train_task", default="musique")
    p.add_argument("--model-path", default=None)
    p.add_argument("--checkpoint", default=None,
                   help="the frozen V10 operator (runs/natural_merge_v9/full_v10.pt)")
    p.add_argument("--anchor", default=None,
                   help="the finished AnchorOnly checkpoint the graft half starts from")
    p.add_argument("--out", default=None, help="module checkpoint (.pt)")
    p.add_argument("--report", default=None, help="training report (.json)")
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--updates", type=int, default=384)
    p.add_argument("--grad-accum", dest="grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.0)
    p.add_argument("--grad-clip", dest="grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rank", type=int, default=None,
                   help="default: the AnchorOnly checkpoint's rank (64)")
    # Option 3: the graft ceiling is what the CF hinge spends, and 0.50 x rms is 3.3x the
    # carrier ceiling -- that gap is where the "graft slots hijack attention away from
    # rows the carriers already answer" story lives (damage has been pinned at 12-13 in
    # every arm). The historical V13 run used 0.5; the checkpoint records whatever it
    # was built with, so old checkpoints keep loading unchanged.
    p.add_argument("--max-ratio", dest="max_ratio", type=float, default=0.34)
    p.add_argument("--span-value", dest="span_value", default="on",
                   choices=("on", "off"),
                   help="smooth each neighbour's VALUE over its own micro-span before "
                        "aggregation (0.5 centre + 0.25 either side, same block only)")
    p.add_argument("--context-diffusion", dest="context_diffusion", default="on",
                   choices=("on", "off"),
                   help="smooth the finished context over same-block neighbours "
                        "(0.70 centre + 0.15 either side)")
    p.add_argument("--evicted-context", dest="evicted_context", default="off",
                   choices=("on", "off"),
                   help="second cross-block path: for every kept slot, take a Top-4 over "
                        "the 2,560 EVICTED slots (no same-block mask) and add it through "
                        "its own zero-initialised projection")
    p.add_argument("--csls", default="on", choices=("on", "off"),
                   help="CSLS background subtraction before the top-k, so slots that are "
                        "similar to everything lose their seat")
    p.add_argument("--select-best", dest="select_best", default="val_ce",
                   choices=("val_ce", "none"),
                   help="write a second checkpoint holding the module at the best "
                        "validation CE instead of blindly taking the last update")
    p.add_argument("--gate-init", dest="gate_init", type=float, default=0.0)
    p.add_argument("--cond-rank", dest="cond_rank", type=int, default=32)
    p.add_argument("--protect-weight", dest="protect_weight", type=float, default=1.0)
    p.add_argument("--reg-weight", dest="reg_weight", type=float, default=0.05)
    p.add_argument("--cf-weight", dest="cf_weight", type=float, default=0.3)
    p.add_argument("--cf-gamma", dest="cf_gamma", type=float, default=0.5)
    p.add_argument("--cf-gate", dest="cf_gate", default="reference",
                   choices=("reference", "full", "none"),
                   help="which rows the counterfactual term is computed on; `full` "
                        "pays for a 4,864-slot forward per row, `reference` is free")
    p.add_argument("--cf-gate-logp", dest="cf_gate_logp", type=float, default=-0.5)
    p.add_argument("--cf-gate-margin", dest="cf_gate_margin", type=float, default=1.0)
    p.add_argument("--first-token-weight", dest="first_token_weight", type=float,
                   default=2.0)
    p.add_argument("--max-docs", dest="max_docs", type=int, default=640,
                   help="training documents to load; the historical runs used 188+8")
    p.add_argument("--val-docs", dest="val_docs", type=int, default=64,
                   help="held-out documents for validation; the historical runs used 2")
    p.add_argument("--val-every", dest="val_every", type=int, default=48)
    p.add_argument("--cross", default="on", choices=("on", "off"),
                   help="'on' trains the cross-block conditioned calibrator (V13); "
                        "'off' trains the same module without the context term, which is "
                        "V11Evict's calibration")
    p.add_argument("--max-ratio-core", dest="max_ratio_core", type=float, default=0.15,
                   help="residual ceiling for carrier slots (protect them)")
    p.add_argument("--cross-lr", dest="cross_lr", type=float, default=1e-4,
                   help="learning rate for `cross_proj` alone, the zero-initialised "
                        "projection that carries the cross-block context")
    p.add_argument("--k-neighbors", dest="k_neighbors", type=int, default=4,
                   help="sparse cross-block neighbours per slot (k_same + k_cross in "
                        "complement mode)")
    p.add_argument("--neighbor-mode", dest="neighbor_mode", default="topk",
                   choices=("topk", "complement"),
                   help="'topk' (V13) asks for the k most associated cross-block slots, "
                        "kind-blind; 'complement' (V14) asks for k_same slots of the "
                        "query's own core/graft kind plus k_cross from the other side, "
                        "with equal per-group weight mass. Nothing else differs between "
                        "the two arms")
    p.add_argument("--architecture", default="v13", choices=("v13", "v15", "v16", "v17"),
                   help="v13 = one kind-blind top-k context added into the bottleneck; "
                        "v15 = five per-layer sink-gated relations delivered as a FiLM "
                        "modulation; v16 = consensus top-4 candidates, per-band relations "
                        "injected into their own memory layers, margin gate after the "
                        "projection; v17 = V13's residual plus the key-anchored / "
                        "value-open split, the three-band layer gate, the similarity floor "
                        "and the graft-attention damping (deep_fusion.DeepFusionResidual). "
                        "Every V17 switch defaults to off, so this file's numbers for "
                        "--architecture v13 are untouched by the branch")
    # ---- V17's own switches. Off at their defaults: the mechanism has to be turned on
    # one variable at a time, and 67/158 is the arm each one has to beat. ----
    p.add_argument("--key-channel", dest="key_channel", default="off",
                   choices=("off", "project"),
                   help="'project' splits the residual into the part the reader's key map "
                        "can see (clipped at --key-ratio) and the rest (clipped at "
                        "--value-ratio), so the keys are locked while the values move. "
                        "Needs the LoRA-folded key basis, built once per run")
    p.add_argument("--key-ratio", dest="key_ratio", type=float, default=0.10,
                   help="ceiling on the key-visible part of the residual, x the slot RMS")
    p.add_argument("--value-ratio", dest="value_ratio", type=float, default=None,
                   help="ceiling on the part of the residual the key cannot see; "
                        "None => the parent's --max-ratio")
    p.add_argument("--band-mode", dest="band_mode", default="off",
                   choices=("off", "three"),
                   help="'three' scales each utility layer's residual by a per-band "
                        "learnable gate; band 0 (lexical) is masked to zero")
    p.add_argument("--band-gates", dest="band_gates", default="0.0,1.0,1.0",
                   help="initial values of the three band gates, comma separated "
                        "(lexical,bridging,readout); band 0 is forced to zero")
    p.add_argument("--sim-floor", dest="sim_floor", type=float, default=0.0,
                   help="absolute lower bound on neighbour similarity: candidates below "
                        "it are dropped from the top-k (0 = off). A row whose candidates "
                        "are all below the bound falls back to its own best one")
    p.add_argument("--graft-tau", dest="graft_tau", type=float, default=1.0,
                   help="graft-attention damping: multiplier on the graft slots' key "
                        "projection (inference-time arm; recorded in the checkpoint)")
    p.add_argument("--graft-bias", dest="graft_bias", type=float, default=0.0,
                   help="graft-attention damping: first-order attention-bias proxy")
    p.add_argument("--graft-norm-quantile", dest="graft_norm_quantile", type=float,
                   default=0.0,
                   help="graft-attention damping: quantile of the CORE key norms used as "
                        "the target, so outlier graft keys stop stealing attention "
                        "(0 = off)")
    # ---- window budget (§3), forwarded to JointConsolidation ----
    p.add_argument("--quota-mode", dest="quota_mode", default="fixed",
                   choices=("fixed", "density"),
                   help="'density' spends the same 1,620 carrier slots unevenly: more of "
                        "them where the block is redundant, fewer where it is sparse")
    p.add_argument("--min-core", dest="min_core", type=int, default=MIN_CORE,
                   help="floor on a block's carrier quota under --quota-mode density")
    p.add_argument("--block-merge-cos", dest="block_merge_cos", type=float, default=0.0,
                   help="absorb an evicted slot into a core slot of its own block when "
                        "their cosine is above this (0 = off)")
    # ---- objective switches ----
    p.add_argument("--layer-norm-weights", dest="layer_norm_weights", default="off",
                   choices=("off", "on"),
                   help="'on' weights the residual regulariser per layer by "
                        "1/(D_init_l + 1e-6), normalised to mean 1, measured on update 0 "
                        "and frozen afterwards")
    p.add_argument("--kl-topk", dest="kl_topk", type=int, default=0,
                   help="restrict the protection KL to the reference distribution's k "
                        "most probable tokens (0 = the whole vocabulary; k is clamped to "
                        "the vocabulary)")
    p.add_argument("--cf-mode", dest="cf_mode", default="donor",
                   choices=("donor", "random", "shuffle"),
                   help="which window the counterfactual hinge scores as the negative: "
                        "the previous document's graft slots (donor), this document's "
                        "random core slots (random) or its own graft slots permuted "
                        "(shuffle)")
    p.add_argument("--cf-random-seed", dest="cf_random_seed", type=int, default=0,
                   help="seed for --cf-mode random|shuffle; the per-document stream is "
                        "derived from this and the document id, so it is reproducible")
    p.add_argument("--relation-lr", dest="relation_lr", type=float, default=3e-3,
                   help="learning rate for the new cross-block projections and gates "
                        "(V16 default 3e-3; the published V15 run used 5e-4)")
    p.add_argument("--core-margin", dest="core_margin", type=float,
                   default=br.CORE_MARGIN,
                   help="V16 core-slot margin threshold tau in the confidence gate")
    p.add_argument("--graft-margin", dest="graft_margin", type=float,
                   default=br.GRAFT_MARGIN,
                   help="V16 graft-slot margin threshold tau in the confidence gate")
    p.add_argument("--gate-kappa", dest="gate_kappa", type=float,
                   default=br.GATE_KAPPA,
                   help="V16 gate slope (stored through softplus, so always positive)")
    p.add_argument("--eta-init", dest="eta_init", type=float, default=br.ETA_INIT,
                   help="V16 initial layer-acceptance logit")
    p.add_argument("--shallow-cutoff", dest="shallow_cutoff", type=int,
                   default=hr.SHALLOW_CUTOFF,
                   help="memory layers below this index keep the self branch only")
    p.add_argument("--sink-percentile", dest="sink_percentile", type=float,
                   default=hr.SINK_PERCENTILE,
                   help="per-layer percentile of the top-k similarity used to place the "
                        "sink logit")
    p.add_argument("--sink-tau", dest="sink_tau", type=float, default=hr.DEFAULT_TAU,
                   help="softmax temperature on the per-layer similarities")
    p.add_argument("--sink-documents", dest="sink_documents", type=int, default=3,
                   help="training documents whose similarity spread sets the sink")
    p.add_argument("--k-same", dest="k_same", type=int, default=2,
                   help="complement mode: neighbours of the query's own kind")
    p.add_argument("--k-cross", dest="k_cross", type=int, default=2,
                   help="complement mode: neighbours from the other side of the "
                        "core/graft divide")
    p.add_argument("--kv-dim", dest="kv_dim", type=int, default=5120,
                   help="width of the reader key/value representation the context lives "
                        "in: utility_layers x kv_heads x head_dim")
    p.add_argument("--core-weight", dest="core_weight", type=float, default=2.0,
                   help="residual penalty weight on carrier slots")
    p.add_argument("--graft-weight", dest="graft_weight", type=float, default=0.5,
                   help="residual penalty weight on graft slots")
    p.add_argument("--core-per-block", dest="core_per_block", type=int,
                   default=QUOTA["core_per_block"])
    p.add_argument("--graft-per-block", dest="graft_per_block", type=int,
                   default=QUOTA["graft_per_block"])
    p.add_argument("--mmr-penalty", dest="mmr_penalty", type=float, default=0.5)
    p.add_argument("--novelty-weight", dest="novelty_weight", type=float, default=0.5)
    p.add_argument("--key-threshold", dest="key_threshold", type=float, default=0.72)
    p.add_argument("--value-threshold", dest="value_threshold", type=float, default=0.60)
    p.add_argument("--max-cluster", dest="max_cluster", type=int, default=3)
    p.add_argument("--blend-weight", dest="blend_weight", type=float, default=0.25)
    p.add_argument("--utility-layers", dest="utility_layers", default="12,16,20,24,28")
    p.add_argument("--probe-count", dest="probe_count", type=int, default=16)
    p.add_argument("--probe-task", dest="probe_task", default=None)
    p.add_argument("--probe-text", dest="probe_text", default=None)
    p.add_argument("--strict-code-hash", dest="strict_code_hash", default="true",
                   choices=("true", "false"))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-reader-checkpointing", dest="no_reader_checkpointing",
                   action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.data_dir or not args.model_path:
        raise SystemExit("--data-dir and --model-path are required")
    try:
        return train(args)
    except BaseException as error:
        # A crashed run used to leave `status: "running"` on disk forever, so "running"
        # could equally mean "died twenty minutes ago" -- every smoke run in this session
        # was reported that way. The last written report is patched in place instead, with
        # the peak it reached, so the difference is visible without reading the log.
        report_path = getattr(args, "report", None)
        if report_path and Path(report_path).exists():
            try:
                state = json.loads(Path(report_path).read_text())
            except (ValueError, OSError):
                state = {}
            if isinstance(state, dict):
                state["status"] = "failed"
                state["error"] = f"{type(error).__name__}: {error}"
                state["error_traceback_tail"] = traceback.format_exc()[-4000:]
                try:
                    state["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
                except (RuntimeError, AssertionError):
                    pass
                try:
                    atomic_json(report_path, state)
                except OSError:
                    pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
