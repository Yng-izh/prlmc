#!/usr/bin/env python3
"""CPU-level implementation checks for V16 (no GPU, no reader needed).

    python tools/test_band_relation.py        # exits non-zero on the first failure

Checks, in the order the acceptance list asks for them: consensus candidates and the
margin definition, band routing (L12 -> L10-14 and nothing else), zero cross injection on
memory layers 0-9, the gate actually controlling the injected amplitude, the gradient
order (projections at step 1, gates from step 2), update-0 identity with AnchorOnly,
capacity, and the checkpoint round trip.
"""
import sys, tempfile, os
from pathlib import Path

import torch, types
import torch.utils.checkpoint  # noqa: F401  (not an attribute of torch.utils by default)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# `tools/v9_geometry_50block.py` is not in the repo (it lives only on the training box),
# so the import chain of `core_fusion` needs a stub here. Nothing under test uses it.
class _Geometry:
    """Stands in for the sealed 50-block geometry, which is not in the repo.

    Only class defaults of `core_fusion` read it; `apply_delta`, the only piece under
    test, does not. On the training box the real module exists and no stub is used.
    """
    def __getattr__(self, name):
        return 16


try:
    import tools.v9_geometry_50block  # noqa: F401
except Exception:
    _stub = types.ModuleType("tools.v9_geometry_50block")
    _stub.GEOMETRY_50B = _Geometry()
    sys.modules["tools.v9_geometry_50block"] = _stub

from methods.prlmc import band_relation as br
from methods.prlmc.cross_block_calibration import CrossBlockResidual, calibrate as v13_calibrate

torch.manual_seed(0)
LAYERS, SLOTS, KVL, KVLD, HID, RANK = 32, 40, 5, 8, 16, 4
BLOCKS = [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10

class _Reading: pass
class FakeScorer:
    def __init__(self, key, value): self.key, self.value = key, value
    def reading(self, window, ids, probes):
        r = _Reading(); r.key = self.key.clone(); r.value = self.value.clone(); return r, None

def fake_meta(seed=0, scorer=None):
    torch.manual_seed(seed)
    N = SLOTS
    key = torch.randn(N, KVL * KVLD)
    value = torch.randn(N, KVL * KVLD)
    return dict(source_block=torch.tensor(BLOCKS), slot_type=torch.tensor([0]*28 + [1]*12),
                cluster_size=torch.ones(N), delta_position=torch.zeros(N), scorer=scorer or FakeScorer(key, value))

def fake_window(layers, N=SLOTS, seed=1):
    torch.manual_seed(seed)
    return torch.randn(len(layers), N, HID)

def make_module():
    m = br.BandAlignedRelationResidual(hidden_size=HID, rank=RANK, kv_layers=KVL,
                                       kv_layer_dim=KVLD, k_neighbors=4)
    anchor = dict(down=torch.randn(RANK, HID), gate_logit=torch.zeros(()),
                  up=torch.randn(HID, RANK))
    m.load_anchor_only(anchor, dict(rank=RANK, hidden_size=HID))
    return m

def report(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""))
    assert ok, name

# ---- 1. consensus candidates, per-band relations, margin background ----
meta = fake_meta()
bands, margins, rep = br.band_contexts(torch.zeros(2, SLOTS, HID), meta, return_report=True,
                                       kv_layers=KVL, kv_layer_dim=KVLD)
report("1 shapes", tuple(bands.shape) == (SLOTS, KVL, KVLD) and tuple(margins.shape) == (SLOTS, KVL),
       f"{tuple(bands.shape)} {tuple(margins.shape)}")
# background must average only legal cross-block slots
key = meta["scorer"].key.view(SLOTS, KVL, KVLD)
k = key - key.mean(dim=0, keepdim=True); k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
sim = k[:, 0] @ k[:, 0].t()
legal = torch.tensor(BLOCKS).unsqueeze(1) != torch.tensor(BLOCKS).unsqueeze(0)
expected_bg = (sim * legal).sum(1) / legal.sum(1)
i = 3
# the candidates come from the five-layer consensus, then each band weights its OWN
# similarity over those same four -- the margin is that band's weighted similarity minus
# its own legal background mean.
consensus = torch.stack([(k[:, b] @ k[:, b].t()).masked_fill(~legal, -2.0)
                         for b in range(KVL)], 0).mean(0)
index = consensus[i].topk(4).indices
picked = sim[i][index]
weights = torch.softmax(picked, dim=-1)
expected_margin = (weights * picked).sum() - expected_bg[i]
report("1 margin = top4 - legal background", abs(float(margins[i, 0]) - float(expected_margin)) < 1e-4,
       f"{float(margins[i,0]):.5f} vs {float(expected_margin):.5f}")
report("1 overlap statistic in [0,1]", 0.0 <= rep["candidate_overlap_with_v13"] <= 1.0,
       str(round(rep["candidate_overlap_with_v13"], 3)))

# ---- 2. band routing: L12's relation reaches L10-14 only ----
m = make_module()
torch.manual_seed(5)
for p in m.band_proj: p.weight.data.normal_()
for p in m.layer_proj: p.weight.data.normal_()
m.up_core.data.normal_(); m.up_graft.data.normal_()
cond = torch.zeros(SLOTS, 3); cond[28:, 2] = 1.0
cond[:, 1] = 1.0
layers = list(range(10, 16))                 # 10-14 band 0, 15 band 1
w = fake_window(layers)
rel_a = (bands.clone(), margins.clone())
rel_b = (bands.clone(), margins.clone()); rel_b[0][:, 0, :] += 3.0   # perturb band 0 only
d_a = m(w, cond, rel_a, torch.arange(10, 16))
d_b = m(w, cond, rel_b, torch.arange(10, 16))
changed = (d_a - d_b).abs().flatten(1).max(dim=1).values > 1e-6
report("2 band 0 touches layers 10-14 only", bool(changed[:5].all() and not changed[5]),
       "rows changed " + str(changed.tolist()))
rel_c = (bands.clone(), margins.clone()); rel_c[0][:, 1, :] += 3.0   # perturb band 1 only
d_c = m(w, cond, rel_c, torch.arange(10, 16))
changed_c = (d_a - d_c).abs().flatten(1).max(dim=1).values > 1e-6
report("2 band 1 touches layer 15 only", bool(changed_c[5] and not changed_c[:5].any()),
       "rows changed " + str(changed_c.tolist()))

# ---- 3. shallow layers: cross term exactly zero, self branch alive ----
cross_shallow = m.cross_term(bands, margins, cond, torch.arange(0, 10))
report("3 shallow cross term exactly zero", float(cross_shallow.abs().max()) == 0.0)
d_shallow = m(fake_window(list(range(0, 10))), cond, (bands, margins), torch.arange(0, 10))
report("3 shallow self branch still moves slots", float(d_shallow.abs().max()) > 0.0,
       f"max delta {float(d_shallow.abs().max()):.4f}")

# ---- 4. the gate scales the injection after the projection ----
cross_a = m.cross_term(bands, margins, cond, torch.arange(10, 15))
with torch.no_grad():
    before = torch.sigmoid(m.eta_logit.clone())
    m.eta_logit[0, 10] = -8.0                 # core acceptance ~0 on that layer
cross_b = m.cross_term(bands, margins, cond, torch.arange(10, 15))
core = cond[:, 2] == 0
open_max = float(cross_a[0][core].abs().max())
report("4 gate closes the core injection",
       float(cross_b[0][core].abs().max()) < 1e-2 * open_max and open_max > 1e-4,
       f"open {float(cross_a[0][core].abs().max()):.4f} closed {float(cross_b[0][core].abs().max()):.6f}")
with torch.no_grad():
    m.eta_logit[0, 10] = 1.5
cross_c = m.cross_term(bands, margins, cond, torch.arange(10, 15))
report("4 injection is linear in the gate", torch.allclose(cross_c, cross_a, atol=1e-6))

# ---- 5. gradients: projections at step 1, gates after they move ----
m = make_module()          # fresh: both projections are exactly zero, as at update 0
m.zero_grad()
cond_g = cond.clone(); cond_g.requires_grad_(False)
w = fake_window(list(range(10, 14)))
delta = m(w, cond_g, (bands, margins), torch.arange(10, 14))
delta.float().square().sum().backward()
proj_grad = float(sum(p.weight.grad.abs().sum() for p in m.band_proj if p.weight.grad is not None)
                  + sum(p.weight.grad.abs().sum() for p in m.layer_proj if p.weight.grad is not None))
gate_grad = float(m.gate_kappa.grad.abs().sum() + m.gate_tau.grad.abs().sum() + m.eta_logit.grad.abs().sum())
report("5 projections have a gradient at step 1", proj_grad > 0.0, f"{proj_grad:.4e}")
report("5 gates have none while the projection is zero", gate_grad == 0.0, f"{gate_grad:.4e}")
with torch.no_grad():
    for p in m.band_proj: p.weight.normal_()
    for p in m.layer_proj: p.weight.normal_()
m.zero_grad()
delta = m(w, cond_g, (bands, margins), torch.arange(10, 14))
delta.float().square().sum().backward()
gate_grad = float(m.gate_kappa.grad.abs().sum() + m.gate_tau.grad.abs().sum() + m.eta_logit.grad.abs().sum())
report("5 gates get a gradient once the projection moves", gate_grad > 0.0, f"{gate_grad:.4e}")

# ---- 6. update 0 == AnchorOnly, core half untouched ----
m0 = make_module()
core_off = torch.nonzero(torch.tensor([0]*28 + [1]*12) == 0).flatten().tolist()
graft_off = torch.nonzero(torch.tensor([0]*28 + [1]*12) == 1).flatten().tolist()
w = fake_window(list(range(LAYERS)))
mm = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in meta.items()}
mm["scorer"] = meta["scorer"]
out, info = br.calibrate(w, mm, m0)
from methods.prlmc.core_fusion import apply_delta
anchor = m0.anchor_view()
expected = apply_delta(w[:, graft_off], anchor(w[:, graft_off]).to(w.dtype))
report("6 core half bit-identical at update 0", torch.equal(out[:, core_off], w[:, core_off]))
report("6 graft half reproduces AnchorOnly", torch.allclose(out[:, graft_off], expected, atol=1e-5),
       f"max dev {float((out[:, graft_off] - expected).abs().max()):.2e}")
report("6 no cross injection on L0-9 at update 0", info["shallow_cross_leak"] == 0.0)
report("6 capacity/split preserved", tuple(out.shape) == tuple(w.shape) and
       info["core_slots"] == 28 and info["graft_slots"] == 12,
       f"{tuple(out.shape)} {info['core_slots']}/{info['graft_slots']}")

# ---- 7. checkpoint round trip ----
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "v16.pt")
    torch.save(dict(schema=br.SCHEMA + "/checkpoint", config=m0.config(),
                    state_dict=m0.state_dict(),
                    geometry=dict(core_per_block=180, graft_per_block=76,
                                  layer_to_band={str(k): v for k, v in br.LAYER_TO_BAND.items()})),
               path)
    loaded, cfg, geo = br.load_band(path, HID)
    out2, _ = br.calibrate(w, mm, loaded)
    report("7 checkpoint reload reproduces the output", torch.equal(out, out2))

# ---- 8. V13 unchanged ----
v13 = CrossBlockResidual(hidden_size=HID, rank=RANK, kv_dim=KVL * KVLD)
report("8 V13 module still builds", hasattr(v13, "cross_proj") and v13.neighbor_mode == "topk")

# ---- 9. calibrate report fields the trainer logs ----
info_cal = info
need = ("cross_injection_rms", "shallow_cross_leak", "band_proj_norm", "layer_proj_norm",
        "gate_bands", "core_relative_delta", "graft_relative_delta")
report("9 report fields present", all(k in info_cal for k in need),
       str([k for k in need if k not in info_cal]))
report("9 every band reported", len(info_cal["gate_bands"]) == KVL,
       str(sorted(info_cal["gate_bands"])))
# ---- 10. the GPU training path wraps the module in torch.utils.checkpoint with a
#            TUPLE payload for V16 (bands, margins), which the CPU tests above never do.
#            If that were unsupported the run would die on its first update.
m = make_module()
chunk = fake_window(list(range(10, 13)))
chunk.requires_grad_(True)
torch.manual_seed(3)
for proj in m.band_proj: proj.weight.data.normal_()
for proj in m.layer_proj: proj.weight.data.normal_()
m.up_graft.data.normal_()

def step(slots, conditioning, relation, layer_ids):
    return m(slots, conditioning, relation, layer_ids)

out = torch.utils.checkpoint.checkpoint(step, chunk, cond_g, (bands, margins),
                                       torch.arange(10, 13), use_reentrant=False)
out.float().sum().backward()
grad = float(sum(p.weight.grad.abs().sum() for p in m.band_proj if p.weight.grad is not None))
report("10 checkpoint accepts the tuple payload and keeps the graph", grad > 0.0, f"{grad:.3e}")

print("ALL CHECKS PASSED")
