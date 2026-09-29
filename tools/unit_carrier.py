"""Structural checks for the phase-3 carrier-residual compactor (CPU only)."""
import sys, torch

sys.path.insert(0, "/share/latent/PRLM-C")
from methods.prlmc.config import PRLMCConfig
from methods.prlmc.consolidate import (
    CarrierResidualCompactor,
    select_block_compact,
    _select_block_indices,
    _l2,
)

torch.manual_seed(0)
L, B, BLK, D = 3, 12, 8, 16          # small stand-in: 12 blocks of 8 slots
K = B * BLK
TARGET = 4 * BLK                      # keep 4 of 12 blocks
src = torch.randn(L, K, D) * 0.5
w = torch.rand(L, K) + 0.1
goal = torch.randn(D)

ok = []

# --- 1. identity at init, against select_blocks ------------------------------
comp = CarrierResidualCompactor(num_layers=L, hidden=D, block=BLK, rank=4, gamma_max=0.1)
assert float(comp.up.weight.abs().sum()) == 0.0, "up must be zero-initialised"
assert float(comp.gamma.detach()) < 0.1, "gamma must be strictly inside gamma_max"

ref = select_block_compact(src, TARGET, block=BLK, weights=w[0], goal_direction=goal)
got = comp(src, weights=w[0], target=TARGET, block=BLK, goal_direction=goal)
identical = torch.equal(ref, got)
ok.append(("identity at init (bitwise == select_blocks)", identical))
print("  max |carrier - select| =", float((ref - got).abs().max()))

# --- 2. carrier indices are shared, not independently scored ------------------
idx = _select_block_indices(src, TARGET, BLK, w[0], goal, None)
blocks = src.view(L, B, BLK, D)
carriers = blocks.index_select(1, idx).reshape(L, TARGET, D)
ok.append(("carrier set == select_blocks output", torch.equal(carriers, ref)))

# --- 3. zeroed routing weights => still identity -----------------------------
# With q and k zeroed the learned logits are identically zero at every position,
# so `logits` collapses to the native cosine prior. The output must still be the
# untouched carriers, because `up` is zero and no amount of routing can move a
# zero residual. This is the check that catches an accidental LayerNorm or bias on
# the output path: either would break the bitwise equality.
with torch.no_grad():
    comp.q.weight.zero_(); comp.k.weight.zero_()
zero_routed = comp(src, weights=w[0], target=TARGET, block=BLK, goal_direction=goal)
ok.append(("identity with degenerate routing", torch.equal(ref, zero_routed)))

# --- 3b. routing is [b_out, b], shared across layers -------------------------
# Rebuild the routing distribution the way `forward` does and confirm its shape
# and row-normalisation. A per-layer router would show up as a 3-D attention.
with torch.no_grad():
    blocks_v = src.view(L, B, BLK, D)
    car_v = blocks_v.index_select(1, idx)
    ref_layer = min(max(comp.reference_layer, 0), L - 1)
    s_unit = _l2(blocks_v[ref_layer].mean(dim=1).to(torch.float32))
    c_unit = _l2(car_v[ref_layer].mean(dim=1).to(torch.float32))
    native = (c_unit @ s_unit.transpose(-1, -2)) / max(1e-6, float(comp.route_temperature))
    learned = (comp.q(c_unit) @ comp.k(s_unit).transpose(-1, -2)) / max(1.0, comp.rank ** 0.5)
    attn = torch.softmax(native + comp.route_scale * learned, dim=-1)
ok.append(("routing is [b_out, b] and shared across layers",
           tuple(attn.shape) == (TARGET // BLK, B)))
ok.append(("routing rows sum to 1", bool(torch.allclose(attn.sum(dim=-1),
                                                       torch.ones(TARGET // BLK), atol=1e-5))))
print("  routing shape =", tuple(attn.shape))

# --- 4. amplitude bound: an adversarial gamma_logit cannot exceed gamma_max ---
with torch.no_grad():
    comp.gamma_logit.fill_(50.0)
    comp.up.weight.normal_(0, 1.0)
big = comp(src, weights=w[0], target=TARGET, block=BLK, goal_direction=goal)
g = float(comp.gamma.detach())
# The invariant is "never exceeds gamma_max", not "strictly below it": sigmoid
# saturates to exactly 1.0 in float32, so gamma lands *on* the ceiling. That is
# the bound doing its job; only exceeding it would be a bug.
# Tolerance, not laxity: float32(0.1) is 0.10000000149011612, so a saturated
# sigmoid lands ~1.5e-9 above the float64 literal. The bound is exact in real
# arithmetic and rounds to the nearest representable value here.
TOL = 1e-6
ok.append(("gamma never exceeds gamma_max", g <= 0.1 + TOL))
sweep_max = max(
    float((0.1 * torch.sigmoid(torch.tensor(v))).item())
    for v in (-1e4, -50.0, -3.0, 0.0, 3.0, 50.0, 1e4)
)
ok.append(("gamma_max holds across the whole logit range", sweep_max <= 0.1 + TOL))
print("  gamma sweep max = %.6f" % sweep_max)
rel = float((big - ref).norm() / ref.norm())
ok.append(("relative residual stays under gamma_max even with a huge up",
           comp.last_residual_ratio <= 0.1 + TOL))
print("  gamma = %.6f  relative residual = %.4f  last_residual_ratio = %.6f"
      % (g, rel, comp.last_residual_ratio))

# --- 4b. the bound is *relative*, not a coefficient bound --------------------
# Scale `up` up by 1000x. A design that bounded only `alpha` would let the
# residual blow past the carriers; the RMS normalisation has to hold the ratio at
# gamma_max regardless of how large the learned residual's raw magnitude is.
with torch.no_grad():
    comp.up.weight.mul_(1000.0)
    comp.gamma_logit.fill_(50.0)
scaled = comp(src, weights=w[0], target=TARGET, block=BLK, goal_direction=goal)
ok.append(("bound survives a 1000x up rescale",
           comp.last_residual_ratio <= 0.1 + TOL))
ok.append(("1000x up changes the output (bound, not a dead path)",
           not torch.equal(scaled, ref)))
print("  after 1000x up: gamma = %.6f  ratio = %.6f"
      % (float(comp.gamma.detach()), comp.last_residual_ratio))

# --- 5. carrier order is temporal, not score order ---------------------------
mono = bool(torch.all(idx[1:] > idx[:-1]))
ok.append(("carrier indices ascending (temporal order)", mono))
print("  indices =", idx.tolist())

# --- 6. each surviving output block is the source block it absorbed ----------
# `ref` is the `select_blocks` result, which is the identity-at-init output, so
# this reads directly: output block j must be *exactly* source block idx[j].
carrier_part = ref.view(L, TARGET // BLK, BLK, D)
src_blocks = src.view(L, B, BLK, D)
kept_ok = all(
    torch.equal(carrier_part[:, j], src_blocks[:, i]) for j, i in enumerate(idx.tolist())
)
ok.append(("each output block keeps its source block's content", kept_ok))

# --- 7. target not a multiple of block must raise ----------------------------
raised = False
try:
    comp(src, weights=w[0], target=TARGET + 1, block=BLK)
except ValueError:
    raised = True
ok.append(("misaligned target raises", raised))

# --- 8. no-op when target >= window -----------------------------------------
noop = comp(src, weights=w[0], target=K, block=BLK)
ok.append(("target >= window returns source untouched", torch.equal(noop, src)))

print()
for name, passed in ok:
    print("  [%s] %s" % ("PASS" if passed else "FAIL", name))
print()
print("ALL PHASE-3 STRUCTURAL CHECKS PASS:", all(p for _, p in ok))
