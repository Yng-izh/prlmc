"""Round 4, step 1: write into the slots without moving the reader's keys.

Experiment 3 put a sparse candidate branch on top of the frozen anchor calibration
and lost six questions. The loss was systematic -- on the 47 rows that answer
correctly without compression the branch broke 9 and fixed 1 (p = 0.021) -- and it
happened while the branch's own training CE fell 12%. The reading experiment 3
left is that the branch writes into the same 4,096 dimensions the reader ADDRESSES
with, so new content cannot be added without moving the address of existing facts.

This module separates the two. For layer l the reader computes

    h_norm = h / rms(h) * gamma_l      (input_layernorm; gamma_l is its weight)
    k      = W_K,l h_norm              (k_proj, LoRA delta folded in -- see below)
    v      = W_V,l h_norm              (v_proj)

so the key depends on `A_l h` and on `rms(h)`, where `A_l = W_K,l diag(gamma_l)`.
Requiring BOTH

    A_l (h' - h) = 0        the change lives in the null space of the key map
    ||h'||_2 = ||h||_2      the RMSNorm denominator does not move

gives `k(h') = k(h)` exactly, while `v` may still change: `W_V,l diag(gamma_l)` is a
different matrix, and its row space is not orthogonal to that null space.

Why the obvious projection is not enough
---------------------------------------
Projecting a residual onto `null(W_K)` and then calling `apply_delta` (which
rescales the sum back to the anchor's RMS) breaks both halves at once. The
projection ignores `gamma_l`, and the rescale moves `rms(h')`. The second is the one
that bites: an RMS change is a MULTIPLICATIVE change to the key that no projection
can undo afterwards. So this path does not call `apply_delta`; it preserves the L2
norm exactly inside the merge, which is the guarantee `apply_delta` was providing
and the reason the frozen reader never sees an out-of-range slot.

The LoRA adapter is part of the key map
---------------------------------------
The reader was trained with LoRA on `q_proj`, `v_proj`, `k_proj`, `up_proj`,
`down_proj` and `gate_proj` (r = 16, alpha = 32), with a second adapter named
`decoder_adapter` that the read path selects. `k_proj.weight` is therefore NOT the
key map: the effective map is `W_K + scaling * B_decoder A_decoder`. Reading the
base weight alone gives a null space that does not annihilate the keys the reader
actually computes. `build_key_bases` folds the active adapter in and then VERIFIES
the folded map against the real module on random inputs, so a wrong adapter or a
peft layout this code does not understand stops the run instead of silently
building the wrong subspace.

What this can and cannot promise
--------------------------------
The pool is stored in fp16. Even an exact fp32 merge is rounded on the way in, so
the key moves by the fp16 rounding of `h_norm`, not by zero. `measured_key_error`
reports the fp32 error, the stored error and the storage floor side by side: the
claim this code supports is "the key is preserved to the precision of the pool",
not "the key is bit-identical". A check that only tests `A (h' - h) ~ 0` would miss
that distinction, which is why the measurement runs through the real RMSNorm and
the real `k_proj`.
"""
import hashlib
from contextlib import contextmanager

import torch

SCHEMA = "key_safe_projection/1"


# ---------------------------------------------------------------------------
# the key map of one layer, LoRA included
# ---------------------------------------------------------------------------
def read_path_adapter(reader):
    """The adapter the reader has selected while it READS memory.

    `MemoryLLM.forward` sets `_active_adapter = ['decoder_adapter']` on every
    LoRA-wrapped module unless the forward is a memory injection, which uses
    `['default']`. The 360 graft slots are read, never injected, so the key map is
    the decoder adapter's. When the model carries no decoder adapter there is only
    `default` and that is the one.
    """
    return "decoder_adapter" if bool(getattr(reader, "add_decoder_lora", False)) else "default"


def _adapter_delta(linear, adapter):
    """`scaling * B @ A` for one adapter, or None if this module is not wrapped."""
    lora_a = getattr(linear, "lora_A", None)
    lora_b = getattr(linear, "lora_B", None)
    if lora_a is None or lora_b is None:
        return None
    names = list(lora_a.keys())
    if not names:
        return None
    if adapter not in names:
        if len(names) != 1:
            raise ValueError(f"{type(linear).__name__} carries adapters {names} and "
                             f"none of them is {adapter!r}: the key map cannot be "
                             f"built without knowing which one the read path uses")
        adapter = names[0]
    scaling = getattr(linear, "scaling", None)
    scale = 1.0
    if isinstance(scaling, dict):
        scale = float(scaling.get(adapter, 1.0))
    a = lora_a[adapter].weight.detach().float()
    b = lora_b[adapter].weight.detach().float()
    return scale * torch.matmul(b, a)


def effective_projection(linear, adapter):
    """The matrix the module actually multiplies by, LoRA delta folded in."""
    weight = linear.weight.detach().float()
    delta = _adapter_delta(linear, adapter)
    return weight if delta is None else weight + delta


def key_map(layer, adapter):
    """`A_l = W_K diag(gamma_l)` -- the matrix whose null space a key cannot see."""
    weight = effective_projection(layer.self_attn.k_proj, adapter)
    gamma = layer.input_layernorm.weight.detach().float()
    if weight.shape[-1] != gamma.shape[-1]:
        raise ValueError(f"k_proj reads {weight.shape[-1]} features but "
                         f"input_layernorm scales {gamma.shape[-1]}")
    return weight * gamma.unsqueeze(0)


# ---------------------------------------------------------------------------
# the bases
# ---------------------------------------------------------------------------
def build_key_bases(reader, relative_tolerance=1e-6, verify=True, seed=0):
    """Per-layer orthonormal basis of `row(W_K diag(gamma))`, `[L, hidden, rank]`.

    The row space is the part of a slot the key can see; its orthogonal complement
    is the subspace a residual may move in without moving any key. The basis is
    computed with an SVD rather than an unpivoted QR because the rank has to be
    decided from the singular values: a rank-deficient layer would otherwise be
    truncated at an arbitrary diagonal position.

    Layers are padded to a common rank with ZERO columns, so a rank-deficient layer
    still stacks. Zero columns contribute nothing to `x Q Q^T`, and `ranks` records
    which columns are real.

    `verify` re-derives the key from the folded map and compares it with the real
    `k_proj` on random inputs. That is the check that catches a wrong adapter or an
    unrecognised peft layout -- the failure mode where the subspace looks plausible
    and annihilates the wrong keys.
    """
    layers = list(reader.model.layers)
    adapter = read_path_adapter(reader)
    maps, spectra = [], []
    for layer in layers:
        a = key_map(layer, adapter)
        u, s, vh = torch.linalg.svd(a, full_matrices=False)
        if not torch.isfinite(s).all():
            raise ValueError("the key map has a non-finite singular value")
        floor = float(s[0]) * float(relative_tolerance)
        rank = int((s > floor).sum())
        rank = max(rank, 1)
        maps.append(vh[:rank].transpose(0, 1).contiguous())
        spectra.append(dict(rank=rank, sigma_max=float(s[0]),
                            sigma_min_kept=float(s[rank - 1]),
                            sigma_min=float(s[-1])))
        del u, s, vh
    width = max(basis.shape[1] for basis in maps)
    device = layers[0].self_attn.k_proj.weight.device
    stacked = torch.zeros(len(maps), maps[0].shape[0], width,
                          dtype=torch.float32, device=device)
    for index, basis in enumerate(maps):
        stacked[index, :, :basis.shape[1]] = basis
    ranks = [item["rank"] for item in spectra]

    if verify:
        worst = _verify_fold(layers, int(stacked.shape[1]), adapter, seed)

    report = dict(
        schema=SCHEMA, layers=len(layers), hidden=int(stacked.shape[1]),
        rank=int(stacked.shape[2]), ranks=ranks,
        adapter=adapter, verified=bool(verify),
        verification_error=(worst if verify else None),
        sigma_min_kept=min(item["sigma_min_kept"] for item in spectra),
        sigma_max=max(item["sigma_max"] for item in spectra),
        condition=min(item["sigma_min_kept"] for item in spectra)
        / max(item["sigma_max"] for item in spectra),
        lora_folded=bool(_adapter_delta(layers[0].self_attn.k_proj, adapter) is not None),
    )
    return stacked, report



@torch.no_grad()
def _verify_fold(layers, hidden, adapter, seed):
    """Re-derive the key from the folded map and compare it with the real `k_proj`.

    This is the check that catches a wrong adapter or an unrecognised peft layout --
    the failure mode where the subspace looks plausible and annihilates the wrong
    keys. Returns the worst relative error over the layers.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    worst = 0.0
    for index, layer in enumerate(layers):
        k_proj = layer.self_attn.k_proj
        dtype = k_proj.weight.dtype
        gamma = layer.input_layernorm.weight.detach().to(dtype)
        x = torch.randn(4, hidden, generator=generator, dtype=torch.float32)
        x = x.to(k_proj.weight.device, dtype)
        with adapter_selected(k_proj, adapter):
            expected = k_proj(layer.input_layernorm(x))
        normalised = (x.float()
                      / x.float().square().mean(dim=-1, keepdim=True)
                      .sqrt().clamp_min(1e-6))
        got = torch.matmul(normalised * gamma.float(),
                           key_map(layer, adapter).transpose(0, 1))
        error = float((got.float() - expected.float()).norm()
                      / expected.float().norm().clamp_min(1e-12))
        worst = max(worst, error)
        if error > 1e-2:
            raise ValueError(
                f"layer {index}: the folded key map reproduces k_proj only to "
                f"{error:.3e} relative error. The adapter {adapter!r} or the peft "
                f"layout is not what this code assumes, and a null space built from "
                f"it would annihilate the wrong keys")
    return worst


def basis_digest(bases):
    """SHA256 over the stacked basis, so a checkpoint can name the reader it used."""
    digest = hashlib.sha256()
    x = bases.detach().cpu().contiguous()
    digest.update(str((tuple(x.shape), str(x.dtype))).encode())
    for index in range(x.shape[0]):
        digest.update(x[index].numpy().tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# the merge
# ---------------------------------------------------------------------------
def project_row_space(x, basis):
    """The part of `x` the key map can see. `[L, N, D]` against `[L, D, R]`."""
    q = basis.to(device=x.device, dtype=torch.float32)
    return torch.matmul(torch.matmul(x.float(), q), q.transpose(1, 2))


def key_safe_merge(base, delta, basis):
    """`[L, N, D]` slots plus a residual -> slots with the key map invariant.

    Returns the merged slots and the diagnostics that say what the constraint cost.
    `row_space_fraction` is the share of the residual the projection threw away: at
    hidden 4096 against a 1024-row key map it should sit near 1/4, and a value near
    zero would mean the branch never wrote anything the key could see anyway.

    The direction of the residual inside the null space is free; only its length is
    tied, and it is tied to the null component's original length, not to the
    residual's. That is what keeps `||h'|| = ||h||` while still letting the candidate
    content choose WHERE to move.
    """
    dtype = base.dtype
    b = base.float()
    d = delta.float()
    row_base = project_row_space(b, basis)
    null_base = b - row_base
    null_delta = d - project_row_space(d, basis)
    proposed = null_base + null_delta
    old_norm = null_base.norm(dim=-1, keepdim=True)
    new_norm = proposed.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    new_null = proposed * (old_norm / new_norm)
    merged = b + (new_null - null_base)
    with torch.no_grad():
        info = dict(
            row_space_fraction=float((d - null_delta).norm()
                                     / d.norm().clamp_min(1e-12)),
            null_shift=float((new_null - null_base).norm()
                             / b.norm().clamp_min(1e-12)),
            norm_preserved=float((merged.norm(dim=-1)
                                  / b.norm(dim=-1).clamp_min(1e-12)).mean()),
            row_leak=float((merged - b - (new_null - null_base)).norm()
                           / (new_null - null_base).norm().clamp_min(1e-12)),
        )
    return merged.to(dtype), info


# ---------------------------------------------------------------------------
# measuring it through the real reader
# ---------------------------------------------------------------------------
@contextmanager
def adapter_selected(module, adapter):
    """Run `module` under one named adapter, restoring the previous selection."""
    previous = getattr(module, "_active_adapter", None)
    names = list(getattr(getattr(module, "lora_A", None), "keys", lambda: [])())
    if adapter in names:
        try:
            module._active_adapter = [adapter]
            yield
        finally:
            if previous is None:
                try:
                    del module._active_adapter
                except AttributeError:
                    pass
            else:
                module._active_adapter = previous
    else:
        yield


@torch.no_grad()
def measured_key_error(reader, base, merged, layer_indices=None, adapter=None):
    """What the reader's keys actually do, on the real RMSNorm and the real k_proj.

    Three numbers, because they answer three different questions:

    `math`    the fp32 error. This is what the projection is responsible for; if it
              is not ~1e-7 the null space is wrong.
    `stored`  the error after the pool's dtype rounding, which is what the reader
              will really see. Measured by calling the real `k_proj`.
    `floor`   the error you would get by rounding the merged vector to the pool's
              dtype with no projection at all -- the smallest error the storage
              allows. `stored` at or below this means the constraint is saturated
              by the storage, not by the maths.

    The fp32 path uses `effective_projection`, which `build_key_bases` has already
    checked against the real module; the stored path calls the module itself, so a
    wrong adapter cannot hide behind the fold.
    """
    layers = list(reader.model.layers)
    if layer_indices is None:
        layer_indices = range(base.shape[0])
    adapter = read_path_adapter(reader) if adapter is None else adapter
    totals = dict(math=0.0, stored=0.0, floor=0.0, key_rms=0.0)
    per_layer = []
    for offset, index in enumerate(layer_indices):
        layer = layers[index]
        k_proj = layer.self_attn.k_proj
        dtype = k_proj.weight.dtype
        weight32 = effective_projection(k_proj, adapter)
        b = base[offset:offset + 1]
        m = merged[offset:offset + 1]

        def real(x):
            with adapter_selected(k_proj, adapter):
                return k_proj(layer.input_layernorm(x.to(dtype)))

        def exact(x):
            # Same norm module, float32: `LlamaRMSNorm` keeps its input dtype, so
            # handing it float32 gives the float32 path without a copy of the maths.
            return torch.nn.functional.linear(layer.input_layernorm(x.float()),
                                              weight32)

        k_base = real(b)
        scale = k_base.float().norm().clamp_min(1e-12)
        math_error = float((exact(m) - exact(b)).norm() / scale)
        stored_error = float((real(m) - k_base).float().norm() / scale)
        floor = float((real(m) - exact(m).to(dtype)).float().norm()
                      / real(m).float().norm().clamp_min(1e-12))
        per_layer.append(dict(layer=int(index), math=math_error, stored=stored_error,
                              floor=floor))
        totals["math"] += math_error
        totals["stored"] += stored_error
        totals["floor"] += floor
        totals["key_rms"] += float(k_base.float().pow(2).mean().sqrt())
    count = max(len(per_layer), 1)
    summary = {key: value / count for key, value in totals.items()}
    summary["layers"] = [item["layer"] for item in per_layer]
    summary["per_layer"] = per_layer
    summary["adapter"] = adapter
    summary["storage_limited"] = bool(summary["stored"] <= 3.0 * summary["floor"] + 1e-7)
    return summary


@torch.no_grad()
def measured_value_change(reader, base, merged, layer_indices=None, adapter=None):
    """`||v(h') - v(h)|| / ||v(h)||`: the branch must still be able to WRITE.

    A key-safe merge that also froze the values would be trivially safe and
    useless. This is the other half of the claim: the same slots, the same reader,
    and the values move by an amount comparable to what the plain path produced.
    """
    layers = list(reader.model.layers)
    if layer_indices is None:
        layer_indices = range(base.shape[0])
    adapter = read_path_adapter(reader) if adapter is None else adapter
    totals = dict(value=0.0, value_rms=0.0)
    per_layer = []
    for offset, index in enumerate(layer_indices):
        layer = layers[index]
        v_proj = layer.self_attn.v_proj
        dtype = v_proj.weight.dtype
        b = base[offset:offset + 1]
        m = merged[offset:offset + 1]

        def values(x):
            with adapter_selected(v_proj, adapter):
                return v_proj(layer.input_layernorm(x.to(dtype)))

        v_base = values(b)
        change = float((values(m) - v_base).float().norm()
                       / v_base.float().norm().clamp_min(1e-12))
        per_layer.append(dict(layer=int(index), value=change))
        totals["value"] += change
        totals["value_rms"] += float(v_base.float().pow(2).mean().sqrt())
    count = max(len(per_layer), 1)
    summary = {key: value / count for key, value in totals.items()}
    summary["per_layer"] = per_layer
    summary["adapter"] = adapter
    return summary
