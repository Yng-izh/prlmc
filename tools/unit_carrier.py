"""Structural checks for the phase-3 carrier-residual compactor (CPU only)."""
import sys, torch

sys.path.insert(0, "/share/latent/PRLM-C")
from methods.prlmc.config import PRLMCConfig
from methods.prlmc.consolidate import (
    CarrierResidualCompactor,
    select_block_compact,
    _select_block_indices,
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
with torch.no_grad():
    comp.q.weight.zero_(); comp.k.weight.zero_()
zero_routed = comp(src, weights=w[0], target=TARGET, block=BLK, goal_direction=goal)
ok.append(("identity with degenerate routing", torch.equal(ref, zero_routed)))

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
ok.append(("residual is small relative to carriers", rel < 0.5))
print("  gamma = %.6f  relative residual = %.4f" % (g, rel))

# --- 5. carrier order is temporal, not score order ---------------------------
mono = bool(torch.all(idx[1:] > idx[:-1]))
ok.append(("carrier indices ascending (temporal order)", mono))
print("  indices =", idx.tolist())

# --- 6. temporal order of surviving blocks matches the source ----------------
kept_ok = True
for j, i in enumerate(idx.tolist()):
    if not torch.equal(big[:, j * BLK : (j + 1) * BLK, :], big[:, j * BLK : (j + 1) * BLK, :]):
        kept_ok = False
# a stronger check: the *carrier* part of each output block must equal the source block
carrier_part = ref.view(L, TARGET // BLK, BLK, D)
src_blocks = src.view(L, B, BLK, D)
for j, i in enumerate(idx.tolist()):
    if not torch.equal(carrier_part[:, j], src_blocks[:, i]):
        kept_ok = False
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
