"""v7 native_gain step0 gradient, clean (bound restored between runs). CPU only."""
import json, math, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "tools"); sys.path.insert(0, ".")
from natural_merge_runtime import build_merge
import methods.prlmc.gated_slot_merge as gsm

torch.set_num_threads(16)
L, H, B, W, T = 32, 4096, 256, 13, 9
SLOTS, TARGET = W * B, T * B
CURRENT = gsm._soft_bound


def old_forced_unit_rms(delta, carrier):
    rms = carrier.square().mean(-1, keepdim=True).sqrt()
    return delta * torch.rsqrt(delta.square().mean(-1, keepdim=True) + 1e-6) * rms


def norms(module):
    out = {}
    for name, p in module.named_parameters():
        out[name] = None if p.grad is None else float(p.grad.norm())
    out["__total__"] = math.sqrt(sum(v * v for v in out.values() if v))
    return out


def run(arch, old):
    torch.manual_seed(0)
    gsm._soft_bound = old_forced_unit_rms if old else CURRENT
    module = build_merge(arch, L, H, B)
    source = torch.randn(L, SLOTS, H)
    ref = torch.randn(L, TARGET, H) * .02
    weight = torch.randn(H, 2) / math.sqrt(H)
    carriers = torch.arange(T)
    module.zero_grad(set_to_none=True)
    out = module(source, target=TARGET, block=B, carrier_indices=carriers)
    ((out - ref) ** 2).mean().backward()
    res = norms(module)
    module.zero_grad(set_to_none=True)
    out = module(source, target=TARGET, block=B, carrier_indices=carriers)
    ce = F.cross_entropy((out.view(L, T, B, H) @ weight).reshape(-1, 2),
                         torch.zeros(L * T * B, dtype=torch.long))
    ce.backward()
    cev = norms(module)
    return dict(architecture=arch, bound=("retired forced unit RMS" if old else "v7 tanh soft bound"),
                ce_value=float(ce), residual=res, ce=cev)


gsm._soft_bound = CURRENT
result = dict(native_tanh=run("grouped_gated_sparse_native_v7", False),
              native_old=run("grouped_gated_sparse_native_v7", True))
gsm._soft_bound = CURRENT
result["dims"] = dict(layers=L, hidden=H, block=B, window_blocks=W, target_blocks=T)
print(json.dumps(result, indent=1))
