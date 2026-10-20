"""Where the pre-t5 1e3 gain actually sits: patch both `_soft_bound` bindings."""
import json, math, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "tools"); sys.path.insert(0, ".")
from natural_merge_runtime import build_merge
import methods.prlmc.gated_slot_merge as gsm
import methods.prlmc.native_skip_merge as nsm

torch.set_num_threads(16)
L, H, B, W, T = 32, 4096, 256, 13, 9
SLOTS, TARGET = W * B, T * B
CURRENT = gsm._soft_bound


def old_forced_unit_rms(delta, carrier):
    rms = carrier.square().mean(-1, keepdim=True).sqrt()
    return delta * torch.rsqrt(delta.square().mean(-1, keepdim=True) + 1e-6) * rms


def norms(module):
    out = {name: (None if p.grad is None else float(p.grad.norm()))
           for name, p in module.named_parameters()}
    out["__total__"] = math.sqrt(sum(v * v for v in out.values() if v))
    return out


def run(label, patch_inner, patch_outer):
    torch.manual_seed(0)
    gsm._soft_bound = old_forced_unit_rms if patch_inner else CURRENT
    nsm._soft_bound = old_forced_unit_rms if patch_outer else CURRENT
    module = build_merge("grouped_gated_sparse_native_v7", L, H, B)
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
    return dict(label=label, patch_inner=patch_inner, patch_outer=patch_outer,
                ce_value=float(ce), residual=res, ce=norms(module))


result = dict(
    outer_only=run("pre-t5 outer bound only", False, True),
    both=run("pre-t5 inner+outer (full old path)", True, True),
)
gsm._soft_bound = CURRENT; nsm._soft_bound = CURRENT
print(json.dumps(result, indent=1))
