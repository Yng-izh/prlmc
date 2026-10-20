"""v7 step0 gradient probe (CPU only, no GPU, no reader).

Measures the step0 gradient norm of the v7 merge operator at the real
architecture dims (32 layers, hidden 4096, block 256, window 13 -> target 9)
under two stand-in objectives, plus the gain ratio against the retired
forced-unit-RMS expression that the spec figure "1.4e5" would have to come from.
"""
import json, math, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "tools"); sys.path.insert(0, ".")
from natural_merge_runtime import build_merge, state_digest
import methods.prlmc.gated_slot_merge as gsm

torch.manual_seed(0)
torch.set_num_threads(64)
L, H, B, W, T = 32, 4096, 256, 13, 9
SLOTS, TARGET = W * B, T * B


def old_forced_unit_rms(delta, carrier):
    """The pre-v7 expression: delta * rsqrt(mean(delta^2)+eps) * carrier_rms.

    Zero-point gain is 1/sqrt(1e-6) = 1e3, which is the only plausible source of
    a five-digit step0 gradient norm.
    """
    rms = carrier.square().mean(-1, keepdim=True).sqrt()
    return delta * torch.rsqrt(delta.square().mean(-1, keepdim=True) + 1e-6) * rms


def norms(module):
    out = {}
    for name, p in module.named_parameters():
        g = p.grad
        out[name] = None if g is None else float(g.norm())
    total = math.sqrt(sum(v * v for v in out.values() if v))
    out["__total__"] = total
    return out


def ce_loss(out, weight):
    pool = out.view(L, T, B, H)
    logits = pool @ weight
    return F.cross_entropy(logits.reshape(-1, 2), torch.zeros(L * T * B, dtype=torch.long))


def run(arch, bound, seed=0, native=False):
    torch.manual_seed(seed)
    module = build_merge(arch, L, H, B)
    source = torch.randn(L, SLOTS, H) * 1.0
    ref = torch.randn(L, TARGET, H) * .02
    weight = torch.randn(H, 2) / math.sqrt(H)
    carriers = torch.arange(T)
    if bound is not None:
        gsm._soft_bound = bound
    # objective 1: residual MSE against a plausible compressed target
    module.zero_grad(set_to_none=True)
    out = module(source, target=TARGET, block=B, carrier_indices=carriers)
    ((out - ref) ** 2).mean().backward()
    res = norms(module)
    # objective 2: frozen unit-scale readout CE over every merged slot
    module.zero_grad(set_to_none=True)
    out = module(source, target=TARGET, block=B, carrier_indices=carriers)
    ce = ce_loss(out, weight)
    ce.backward()
    cev = norms(module)
    return dict(architecture=arch, bound=("v7 tanh soft bound" if bound is None else "retired forced unit RMS"),
                forward_zero_identity=float((out - source[:, :TARGET]).abs().max()),
                ce_value=float(ce), residual=res, ce=cev,
                state_digest=state_digest(module))


results = {}
results["v7_slot"] = run("grouped_gated_sparse_slot_v7", None)
results["v7_slot_old_rms"] = run("grouped_gated_sparse_slot_v7", old_forced_unit_rms)
results["v7_native"] = run("grouped_gated_sparse_native_v7", None, native=True)
results["v7_native_old_rms"] = run("grouped_gated_sparse_native_v7", old_forced_unit_rms, native=True)
results["dims"] = dict(layers=L, hidden=H, block=B, window_blocks=W, target_blocks=T,
                       window_slots=SLOTS, target_slots=TARGET,
                       merge_parameters=sum(p.numel() for p in build_merge("grouped_gated_sparse_slot_v7", L, H, B).parameters()))
print(json.dumps(results, indent=1))
