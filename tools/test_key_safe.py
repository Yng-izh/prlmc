#!/usr/bin/env python3
"""Checks for the key-safe merge, on a synthetic reader, on the CPU.

The unit under test is `methods/prlmc/key_safe_projection.py`, and the thing that
can be wrong while every GPU number still looks fine is the SUBSPACE: a basis that
is orthonormal, a merge that preserves the norm, and a key that does not move --
all of which can hold while the basis describes the wrong matrix, because the
reader's `k_proj` is LoRA-wrapped and the read path selects a different adapter
than the write path does. The last checks in this file exist for exactly that
failure mode: a fake adapter layout that `effective_projection` folds correctly,
and one that it does not, where `build_key_bases` must refuse to build.

What is checked
---------------
  basis_orthonormal          Q^T Q = I on the real columns
  row_space_kills_null       A (I - Q Q^T) x = 0, and A Q Q^T x = A x
  merge_preserves_norm       ||h'|| = ||h|| in float32
  merge_change_is_null       A (h' - h) = 0
  init_identity_bitwise      a zero residual leaves the slots bit-identical
  math_key_error             the fp32 key error is at rounding level
  stored_key_error           the fp16 key error is at the storage floor
  value_moves                v_proj still sees the change, and by a similar amount
  plain_is_not_safe          the plain merge moves the key far more (the reference)
  core_untouched             the 1,944 core slots are byte-identical
  frozen_stage               the frozen stage is the AnchorOnly window
  plain_path_unchanged       key_safe=False still reproduces apply_delta exactly
  no_double_normalisation    the key-safe path is not apply_delta in disguise
  gradient_reaches_fusion    the constraint is differentiable end to end
  per_layer_basis            layer_start selects that layer's basis
  adapter_folded             a LoRA wrapper is folded into the key map
  wrong_adapter_rejected     a wrapper whose fold is wrong stops the build

Usage
-----
    python tools/test_key_safe.py                 # small shapes, ~2 s
    python tools/test_key_safe.py --real-shape    # 4096 / 1024 / 4 layers
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import methods.prlmc.key_safe_projection as ks                      # noqa: E402
from methods.prlmc.anchor_fusion import (                           # noqa: E402
    FrozenAnchorFusion, SparseCandidateFusion, compact_window,
    frozen_stage)
from methods.prlmc.anchor_residual import (                         # noqa: E402
    AnchorResidual, compact_window as compact_anchor)
from methods.prlmc.core_fusion import apply_delta                   # noqa: E402
from methods.prlmc.source_compaction import split_source_anchors    # noqa: E402


# ---------------------------------------------------------------------------
# a reader small enough to run in a second, shaped like the real one
# ---------------------------------------------------------------------------
class RMSNorm(torch.nn.Module):
    """`LlamaRMSNorm`, including the fp32 variance and the cast back."""

    def __init__(self, hidden, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(dtype)


class LoraLinear(torch.nn.Module):
    """The slice of peft's `LoraLayer` this code reads: weight, A, B, scaling."""

    def __init__(self, in_features, out_features, rank=4, alpha=8.0,
                 adapters=("default", "decoder_adapter"), apply_delta=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.lora_A = torch.nn.ModuleDict(
            {name: torch.nn.Linear(in_features, rank, bias=False) for name in adapters})
        self.lora_B = torch.nn.ModuleDict(
            {name: torch.nn.Linear(rank, out_features, bias=False) for name in adapters})
        for name in adapters:
            torch.nn.init.normal_(self.lora_A[name].weight, std=0.02)
            torch.nn.init.normal_(self.lora_B[name].weight, std=0.02)
        self.scaling = {name: alpha / rank for name in adapters}
        self._active_adapter = [adapters[0]]
        self.apply_lora = apply_delta

    def forward(self, x):
        out = torch.nn.functional.linear(x, self.weight)
        if not self.apply_lora:
            return out
        for name in self._active_adapter:
            a = self.lora_A[name].weight
            b = self.lora_B[name].weight
            out = out + self.scaling[name] * torch.nn.functional.linear(
                torch.nn.functional.linear(x, a), b)
        return out


class Attention(torch.nn.Module):
    def __init__(self, hidden, kv, lora):
        super().__init__()
        self.k_proj = lora(hidden, kv) if lora else torch.nn.Linear(hidden, kv, bias=False)
        self.v_proj = lora(hidden, hidden) if lora else torch.nn.Linear(hidden, hidden,
                                                                      bias=False)


class Layer(torch.nn.Module):
    def __init__(self, hidden, kv, lora):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden)
        self.self_attn = Attention(hidden, kv, lora)


class Reader(torch.nn.Module):
    def __init__(self, layers, hidden, kv, lora, add_decoder_lora=True):
        super().__init__()
        self.add_decoder_lora = add_decoder_lora
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [Layer(hidden, kv, lora) for _ in range(layers)])


def build_reader(args, lora=True, apply_delta=True, seed=0):
    torch.manual_seed(seed)
    factory = ((lambda i, o: LoraLinear(i, o, rank=4, alpha=8.0,
                                        apply_delta=apply_delta))
               if lora else None)
    return Reader(args.layers, args.hidden, args.kv, factory)


# ---------------------------------------------------------------------------
# a window with the real layout
# ---------------------------------------------------------------------------
def build_window(hidden, layers):
    """The real 19->9 layout, at the caller's width, so the module and the bases fit."""
    block, core, graft_per_block = 16, 12, 4
    window_blocks, target_blocks = 5, 3
    selected, dropped = [0, 2, 4], [1, 3]
    torch.manual_seed(1)
    window = torch.randn(layers, window_blocks * block, hidden)
    core_positions = [carrier * block + offset
                      for carrier in selected for offset in range(core)]
    offsets, full_positions = [], [None] * (target_blocks * block)
    for slot in range(target_blocks):
        for offset in range(core):
            full_positions[slot * block + offset] = core_positions[slot * core + offset]
        for offset in range(graft_per_block):
            source_block = dropped[(slot + offset) % len(dropped)]
            offsets.append(slot * block + core + offset)
            full_positions[slot * block + core + offset] = (
                source_block * block + 6 + offset)
    telemetry = dict(bridge_slot_offsets=offsets,
                     slot_source_positions=full_positions,
                     selected_blocks=selected)
    parts_ = split_source_anchors(telemetry, block, window_blocks)
    output = torch.randn(layers, target_blocks * block, hidden)
    return dict(window=window, output=output, parts=parts_, layers=layers,
                hidden=hidden)


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------
def run_checks(args):
    results = []

    def check(name, function):
        try:
            detail = function() or {}
            results.append(dict(check=name, passed=True, **detail))
        except AssertionError as error:
            results.append(dict(check=name, passed=False, error=str(error)))
        return results[-1]["passed"]

    reader = build_reader(args)
    bases, report = ks.build_key_bases(reader, verify=True)
    layers = list(reader.model.layers)
    adapter = ks.read_path_adapter(reader)

    def basis_orthonormal():
        ranks = report["ranks"]
        worst = 0.0
        for index, rank in enumerate(ranks):
            q = bases[index][:, :rank]
            gram = q.transpose(0, 1) @ q
            worst = max(worst, float((gram - torch.eye(rank)).abs().max()))
        assert worst < 1e-5, f"Q^T Q is not the identity: {worst:.3e}"
        assert report["verified"], "build_key_bases did not verify its own fold"
        return dict(layers=report["layers"], rank=report["rank"], ranks=ranks,
                    adapter=adapter, lora_folded=report["lora_folded"],
                    gram_error=worst,
                    verification_error=report["verification_error"])

    def row_space_kills_null():
        worst = 0.0
        for index, layer in enumerate(layers):
            a = ks.key_map(layer, adapter)
            q = bases[index][:, :report["ranks"][index]]
            torch.manual_seed(2)
            x = torch.randn(6, a.shape[1])
            projected = (x @ q) @ q.transpose(0, 1)
            null = x - projected
            worst = max(worst, float((a @ null.transpose(0, 1)).norm()
                                     / (a @ x.transpose(0, 1)).norm().clamp_min(1e-12)))
            assert torch.allclose(a @ projected.transpose(0, 1),
                                  a @ x.transpose(0, 1), rtol=1e-4, atol=1e-5), \
                "the row space does not reproduce A x"
        assert worst < 1e-5, f"the null space is not annihilated: {worst:.3e}"
        return dict(relative_key_of_null=worst)

    torch.manual_seed(3)
    base = torch.randn(args.layers, 24, args.hidden)
    delta = 0.15 * torch.randn_like(base)

    def merge_preserves_norm():
        merged, info = ks.key_safe_merge(base, delta, bases)
        before = base.norm(dim=-1)
        after = merged.norm(dim=-1)
        worst = float((after / before - 1.0).abs().max())
        assert worst < 1e-6, f"the norm moved by {worst:.3e}"
        return dict(max_relative_norm_error=worst,
                    row_space_fraction=info["row_space_fraction"])

    def merge_change_is_null():
        merged, _ = ks.key_safe_merge(base, delta, bases)
        worst = 0.0
        for index, layer in enumerate(layers):
            a = ks.key_map(layer, adapter)
            change = merged[index] - base[index]
            worst = max(worst, float((a @ change.transpose(0, 1)).norm()
                                     / (a @ base[index].transpose(0, 1)).norm()
                                     .clamp_min(1e-12)))
        assert worst < 1e-5, f"A (h' - h) is {worst:.3e} of |A h|"
        return dict(relative_key_leak=worst)

    def init_identity_bitwise():
        zero = torch.zeros_like(base)
        merged, info = ks.key_safe_merge(base, zero, bases)
        assert torch.equal(merged, base), "a zero residual changed the slots"
        assert info["null_shift"] == 0.0, "a zero residual moved the null component"
        return dict(bit_identical=True)

    def math_key_error():
        merged, _ = ks.key_safe_merge(base, delta, bases)
        measured = ks.measured_key_error(reader, base, merged)
        assert measured["math"] < 1e-5, \
            f"the fp32 key error is {measured['math']:.3e}"
        return dict(math=measured["math"], stored=measured["stored"],
                    floor=measured["floor"], storage_limited=measured["storage_limited"])

    def stored_key_error():
        merged, _ = ks.key_safe_merge(base, delta, bases)
        measured = ks.measured_key_error(reader, base, merged)
        limit = 3.0 * measured["floor"] + 1e-7
        assert measured["stored"] <= limit, (
            f"the stored key error {measured['stored']:.3e} exceeds the fp32 storage "
            f"floor {measured['floor']:.3e}")
        return dict(stored=measured["stored"], floor=measured["floor"], limit=limit)

    def value_moves():
        merged, _ = ks.key_safe_merge(base, delta, bases)
        measured = ks.measured_value_change(reader, base, merged)
        assert measured["value"] > 1e-3, \
            f"v_proj barely changed ({measured['value']:.3e}): nothing was written"
        return dict(relative_value_change=measured["value"])

    def plain_is_not_safe():
        merged, _ = ks.key_safe_merge(base, delta, bases)
        plain = apply_delta(base, delta)
        safe_error = ks.measured_key_error(reader, base, merged)
        plain_error = ks.measured_key_error(reader, base, plain)
        ratio = plain_error["stored"] / max(safe_error["stored"], 1e-12)
        assert ratio > 10.0, (
            f"the plain merge moves the key only {ratio:.2f}x more than the key-safe "
            f"one: the projection is not doing anything")
        return dict(plain_key_error=plain_error["stored"],
                    key_safe_key_error=safe_error["stored"], ratio=ratio)

    window = build_window(args.hidden, args.layers)
    anchor_module = AnchorResidual(hidden_size=window["hidden"], rank=4)
    with torch.no_grad():
        anchor_module.up.normal_(std=0.05)
    fusion = SparseCandidateFusion(hidden_size=window["hidden"], rank=2, top_k=4)
    with torch.no_grad():
        fusion.fuse_up.normal_(std=0.05)
    small_bases = bases[:window["layers"]].clone()
    safe_module = FrozenAnchorFusion(anchor_module, fusion, key_bases=small_bases,
                                     key_safe=True).eval()
    plain_module = FrozenAnchorFusion(anchor_module, fusion, key_bases=None,
                                      key_safe=False).eval()

    with torch.no_grad():
        anchor_window, _ = compact_anchor(window["output"], window["parts"],
                                          anchor_module)

    def core_untouched():
        with torch.no_grad():
            fused, _ = compact_window(window["output"], window["window"],
                                      window["parts"], safe_module)
        keep = torch.ones(window["output"].shape[1], dtype=torch.bool)
        keep[window["parts"]["anchor_offsets"]] = False
        assert torch.equal(window["output"][:, keep], fused[:, keep]), \
            "the key-safe path changed a core slot"
        return dict(core_slots=int(keep.sum()))

    def frozen_stage_is_anchor_only():
        with torch.no_grad():
            stage = frozen_stage(window["output"], window["parts"], safe_module)
        reference = anchor_window.index_select(1, window["parts"]["anchor_offsets"])
        assert torch.equal(stage, reference), \
            "the frozen stage is not the AnchorOnly window"
        return dict(bit_identical=True)

    def plain_path_unchanged():
        with torch.no_grad():
            fused, info = compact_window(window["output"], window["window"],
                                         window["parts"], plain_module)
            anchors = window["output"].index_select(1, window["parts"]["anchor_offsets"])
            # The plain path is the frozen anchor stage and THEN apply_delta, which is
            # the operator experiment 3 evaluated -- not apply_delta on the raw anchors.
            base = plain_module.anchor_slots(anchors)
            branch = fusion(base,
                            window["window"].index_select(
                                1, window["parts"]["candidate_positions"]),
                            window["parts"]["same_block"])[0]
            expected = window["output"].index_copy(
                1, window["parts"]["anchor_offsets"], apply_delta(base, branch))
        assert torch.equal(fused, expected), \
            "the plain path no longer reproduces apply_delta"
        assert info["fusion_mode"] == "plain", "the plain module reports the wrong mode"
        return dict(mode=info["fusion_mode"])

    def no_double_normalisation():
        with torch.no_grad():
            safe, safe_info = compact_window(window["output"], window["window"],
                                             window["parts"], safe_module)
            plain, _ = compact_window(window["output"], window["window"],
                                      window["parts"], plain_module)
        assert safe_info["fusion_mode"] == "key-safe", "the mode is not recorded"
        assert not torch.equal(safe, plain), \
            "the key-safe path reproduced the plain one: no projection happened"
        assert "row_space_fraction" in safe_info, "the key-safe diagnostics are missing"
        return dict(row_space_fraction=safe_info["row_space_fraction"],
                    null_shift=safe_info["null_shift"],
                    norm_preserved=safe_info["norm_preserved"])

    def gradient_reaches_fusion():
        trainable = FrozenAnchorFusion(
            AnchorResidual(hidden_size=window["hidden"], rank=4),
            SparseCandidateFusion(hidden_size=window["hidden"], rank=2, top_k=4),
            key_bases=small_bases.clone(), key_safe=True)
        with torch.no_grad():
            trainable.fusion.fuse_up.normal_(std=0.05)
        fused, _, stats = trainable(
            window["output"].index_select(1, window["parts"]["anchor_offsets"]),
            window["window"].index_select(1, window["parts"]["candidate_positions"]),
            window["parts"]["same_block"], layer_start=0)
        probe = torch.randn_like(fused)
        (fused * probe).sum().backward()
        grads = {name: (None if parameter.grad is None
                        else float(parameter.grad.norm()))
                 for name, parameter in trainable.named_parameters()}
        missing = [name for name in ("fusion.fuse_up", "fusion.fuse_down",
                                     "fusion.score_q", "fusion.score_k")
                   if not grads.get(name)]
        assert not missing, f"no gradient on {missing}: the constraint detaches"
        assert grads["anchor.up"] is None, "the frozen anchor received a gradient"
        return dict(fusion_fuse_up=grads["fusion.fuse_up"],
                    anchor_up=grads["anchor.up"])

    def per_layer_basis():
        with torch.no_grad():
            first, _ = compact_window(window["output"], window["window"],
                                      window["parts"], safe_module)
            swapped_bases = small_bases.clone()
            swapped_bases[0], swapped_bases[1] = small_bases[1].clone(), small_bases[0].clone()
            swapped = FrozenAnchorFusion(
                anchor_module, fusion, key_bases=swapped_bases,
                key_safe=True).eval()
            second, _ = compact_window(window["output"], window["window"],
                                       window["parts"], swapped)
        assert not torch.equal(first, second), (
            "swapping the two layers' bases changed nothing: the module is not "
            "using the layer's own basis")
        return dict(distinct_per_layer=True)

    def fp16_storage():
        """The regime the real run is in: an fp16 pool.

        Two separate claims. The projection itself is exact on float32 inputs, so
        `math` is at rounding level there. Through an fp16 pool the merged slot is
        rounded on the way in, and the resulting key error lands at the floor that
        rounding sets -- it cannot be smaller, and a constraint that measured smaller
        would mean the measurement is not looking at the real pool.
        """
        half = build_reader(args, seed=7).half()
        half_bases, half_report = ks.build_key_bases(half, verify=False)
        exact_merged, _ = ks.key_safe_merge(base, delta, half_bases)
        exact = ks.measured_key_error(half, base, exact_merged)
        assert exact["math"] < 1e-5, (
            f"the fp32 key error is {exact['math']:.3e}: the projection is wrong")
        b = base.half()
        d = delta.half()
        merged, _ = ks.key_safe_merge(b, d, half_bases)
        measured = ks.measured_key_error(half, b, merged)
        assert measured["storage_limited"], (
            f"at fp16 the stored key error {measured['stored']:.3e} is not at the "
            f"storage floor {measured['floor']:.3e}: the constraint is not saturated "
            f"by the pool")
        return dict(math=exact["math"], stored=measured["stored"],
                    floor=measured["floor"], rank=half_report["rank"],
                    storage_limited=measured["storage_limited"])

    def adapter_folded():
        assert report["lora_folded"], "the synthetic reader's LoRA was not detected"
        delta = ks.effective_projection(layers[0].self_attn.k_proj, adapter) \
            - layers[0].self_attn.k_proj.weight.detach().float()
        assert float(delta.abs().max()) > 0.0, "the folded adapter delta is zero"
        return dict(adapter=adapter, delta_norm=float(delta.norm()))

    def wrong_adapter_rejected():
        broken = build_reader(args, apply_delta=False)
        try:
            ks.build_key_bases(broken, verify=True)
        except ValueError as error:
            return dict(rejected=True, message=str(error)[:120])
        raise AssertionError("build_key_bases accepted a wrapper whose forward ignores "
                             "the adapter: the subspace would annihilate the wrong keys")

    for name, function in (
        ("basis_orthonormal", basis_orthonormal),
        ("row_space_kills_null", row_space_kills_null),
        ("merge_preserves_norm", merge_preserves_norm),
        ("merge_change_is_null", merge_change_is_null),
        ("init_identity_bitwise", init_identity_bitwise),
        ("math_key_error", math_key_error),
        ("stored_key_error", stored_key_error),
        ("value_moves", value_moves),
        ("plain_is_not_safe", plain_is_not_safe),
        ("core_untouched", core_untouched),
        ("frozen_stage", frozen_stage_is_anchor_only),
        ("plain_path_unchanged", plain_path_unchanged),
        ("no_double_normalisation", no_double_normalisation),
        ("gradient_reaches_fusion", gradient_reaches_fusion),
        ("per_layer_basis", per_layer_basis),
        ("fp16_storage", fp16_storage),
        ("adapter_folded", adapter_folded),
        ("wrong_adapter_rejected", wrong_adapter_rejected),
    ):
        check(name, function)

    failed = [item for item in results if not item["passed"]]
    print(json.dumps(dict(
        self_test="failed" if failed else "ok",
        passed=len(results) - len(failed), failed=len(failed),
        shapes=dict(layers=args.layers, hidden=args.hidden, kv=args.kv),
        checks=results), indent=2, ensure_ascii=False, default=str))
    return 1 if failed else 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--kv", type=int, default=64)
    p.add_argument("--real-shape", dest="real_shape", action="store_true",
                   help="use the reader's real shapes: 4096 hidden, 1024 key width")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.real_shape:
        args.hidden, args.kv = 4096, 1024
        args.layers = max(args.layers, 4)
    torch.set_grad_enabled(True)
    return run_checks(args)


if __name__ == "__main__":
    raise SystemExit(main())
