#!/usr/bin/env python3
"""CPU-level implementation checks for V17 (no GPU, no reader, no checkpoint).

    python tools/test_deep_fusion.py       # exits non-zero on the first failure

Nothing here loads a model, a tokenizer or a checkpoint: every module under test is
built at toy width (hidden 16, rank 4, 20 slots) and fed matrices this file makes up.
The reader is a fake (`FakeScorer`) that reads keys and values straight out of the
slots, so the numbers below are exact and the checks are about the MECHANISM, not about
how well it works.

Checks, in the order the V17 acceptance list asks for them:

 1. `BANDS` / `BAND_NAMES`, `band_of_layer`, `band_layout` -- and the fact that the real
    `UTILITY_LAYERS` put nothing in band 0.
 2. `band_mode="three"`: band 0 is EXACTLY zero, band 1/2 are bit-equal to the parent,
    and band 0 stays zero even if its gate is forced to 1 (it is masked, not trained).
 3. `key_channel="project"`: the key-visible residual is clamped to `key_ratio x rms`,
    the null side is clamped to `value_ratio x rms` and actually reaches it,
    `value_ratio=None` falls back to the parent ceiling, and a missing basis raises.
 4. `graft_attention_calibration`: only `slot_type==1` moves, `tau<1` lowers their key
    norm by exactly tau, the null part is untouched, `norm_quantile` only shrinks, and
    tau=1/bias=0/quantile=0 is a no-op.
 5. `calibrate` (the V17 wrapper) keeps the parent's return shape and report keys.
 6. `cross_block_context(sim_floor=...)`: off is bit-identical to V13, a floor leaves
    rows with >= k candidates alone, drops below-floor seats to zero weight when some
    survive, and falls back to the UNFLOORED scores when a whole row is below.
 7. `DeepFusionResidual.config()` round trip: config -> rebuild -> identical shapes,
    with every V17 key present and the parent keys still carried.
 8. `quota_mode="density"`: the total is exactly `len(carriers) * core_per_block`, every
    block sits in `[min_core, max_core]`, the redundant block is pinned at the floor,
    the build is deterministic, and `meta["pool_block"]` closes position -> block.
 9. `absorb_cos`: off is verbatim, on reports pairs whose mean alpha never exceeds
    `absorb_alpha_max`, only core slots move, and the moved slots keep their RMS.
10. `--cf-mode random`'s seed derivation is reproducible, document-dependent and
    independent of the global RNG (read out of the trainer source, so this runs without
    `transformers`).
11. Defaults reproduce V13: a `DeepFusionResidual` with every V17 knob closed produces
    the parent calibration bit-for-bit.

Where the spec leaves a name or a return type open (the exact `band_layout` payload,
the band-gate parameter's name, the constructor kwargs that carry tau), the check reads
it off the object instead of assuming, and says so in a NOTE line rather than failing.
"""
import ast
import hashlib
import math
import os
import random
import sys
import zlib
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# `tools/v9_geometry_50block.py` lives only on the training box, and `core_fusion`
# imports it at module scope. Nothing under test reads it; the stub only has to exist.
import types  # noqa: E402

try:
    import tools.v9_geometry_50block  # noqa: F401
except Exception:
    class _Geometry:
        """Stands in for the sealed 50-block geometry (class defaults only)."""

        def __getattr__(self, name):
            return 16

    _stub = types.ModuleType("tools.v9_geometry_50block")
    _stub.GEOMETRY_50B = _Geometry()
    sys.modules["tools.v9_geometry_50block"] = _stub

from methods.prlmc import cross_block_calibration as cb   # noqa: E402
from methods.prlmc import deep_fusion as df               # noqa: E402
from methods.prlmc import evict_window as ew              # noqa: E402
from methods.prlmc.key_safe_projection import project_row_space  # noqa: E402
from methods.prlmc.reader_utility import UTILITY_LAYERS   # noqa: E402

TRAINER = ROOT / "tools" / "train_cross_block_calibration.py"

# Toy geometry. Big enough for the mechanisms to have room, small enough to run in a
# second on a laptop: rank-4 bottleneck, 6 of 16 dimensions visible to the key map.
HID, RANK, KVL, BASIS_RANK = 16, 4, 8, 6
SLOTS, CORE_SLOTS = 20, 14
GRAFT = torch.arange(CORE_SLOTS, SLOTS)          # the graft half of the toy window
CORE = torch.arange(0, CORE_SLOTS)
LAYERS5 = (0, 1, 2, 3, 4)                        # window rows, one per utility layer
BAND_LAYERS = (5, 12, 24)                        # one row per band: 0, 1, 2
NOTES = []


def report(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""))
    assert ok, name


def note(text):
    """A measurement this file cannot turn into an assertion, printed not hidden."""
    NOTES.append(text)
    print("NOTE " + text)


# ---------------------------------------------------------------------------
# toy reader and toy metadata
# ---------------------------------------------------------------------------
class _Reading:
    pass


class FakeScorer:
    """`ReaderScorer` stand-in: the key is the slot's own first `kv` dimensions.

    Reading the key out of the slot means the test knows every similarity in advance,
    so it can predict both the neighbour choice and the aggregated context instead of
    just checking that something non-zero came out.

    `maps` is only needed by the graft-attention path, which measures the reader's REAL
    key norm through `k_proj` + `input_layernorm`; `maps[layer]` carries a stand-in
    weight and norm module so that path runs without a reader.
    """

    def __init__(self, kv=KVL, layers=None, hidden=HID, key_out=24, seed=23):
        self.kv = int(kv)
        self.maps = {}
        if layers is not None:
            generator = torch.Generator().manual_seed(int(seed))
            for layer in layers:
                self.maps[int(layer)] = dict(
                    k=torch.randn(key_out, hidden, generator=generator),
                    norm=torch.nn.LayerNorm(hidden))

    def reading(self, slots, positions, probes=None):
        head = slots[0].float()
        reading = _Reading()
        reading.key = _unit(head[:, : self.kv])
        reading.value = head[:, self.kv: 2 * self.kv].clone()
        # Reader utility is a per-slot scalar; `-position` keeps the selection stable
        # and puts every block's "kept" set at its first slots.
        reading.utility = -positions.float()
        return reading, [dict(layer=0, utility_mean=float(reading.utility.mean()))]

    def top_direction(self, layer):
        """The slot direction with the longest `k_proj` image: a key-norm outlier.

        `input_layernorm` is scale invariant, so inflating a slot does NOT make its key
        longer -- only its direction does. This is how the window gets graft slots whose
        reader keys really are outliers.
        """
        weight = self.maps[int(layer)]["k"]
        return torch.linalg.svd(weight, full_matrices=False)[2][0]


def _unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def fake_window(layers, slots=SLOTS, hidden=HID, seed=1):
    torch.manual_seed(seed)
    return torch.randn(len(tuple(layers)), slots, hidden)


def fake_cond(slots=SLOTS, core=CORE_SLOTS):
    cond = torch.zeros(slots, 3)
    cond[core:, 2] = 1.0
    return cond


def fake_context(slots=SLOTS, dim=KVL, seed=2):
    torch.manual_seed(seed)
    return torch.randn(slots, dim) * 0.1


def fake_meta(slot_types=None, blocks=None, slots=SLOTS, scorer=None):
    if slot_types is None:
        slot_types = [0] * CORE_SLOTS + [1] * (slots - CORE_SLOTS)
    if blocks is None:
        blocks = [index * 2 // slots for index in range(slots)]
    return dict(source_position=torch.arange(slots),
                source_block=torch.tensor(blocks, dtype=torch.long),
                slot_type=torch.tensor(slot_types, dtype=torch.long),
                cluster_size=torch.ones(slots, dtype=torch.long),
                delta_position=torch.zeros(slots),
                scorer=scorer if scorer is not None else FakeScorer())


def orthonormal_basis(layers, hidden=HID, rank=BASIS_RANK, seed=7):
    """`[L, D, R]` with orthonormal columns -- the key-visible row space."""
    generator = torch.Generator().manual_seed(int(seed))
    basis = torch.zeros(len(tuple(layers)), hidden, rank)
    for index in range(basis.shape[0]):
        block = torch.randn(hidden, rank, generator=generator)
        basis[index] = torch.linalg.qr(block).Q
    return basis


def make_deep(**kwargs):
    options = dict(hidden_size=HID, rank=RANK, kv_dim=KVL, k_neighbors=4,
                   layer_ids=LAYERS5)
    options.update(kwargs)
    return df.DeepFusionResidual(**options)


def randomise(module, seed=5):
    """Send the residual away from its zero init, so "unchanged" means something."""
    torch.manual_seed(seed)
    with torch.no_grad():
        module.down.normal_()
        module.up_graft.normal_(std=3.0)
        module.up_core.normal_(std=3.0)
        if getattr(module, "use_cross", False):
            module.cross_proj.weight.normal_(std=0.5)
        module.gate_logit.zero_()
    return module


def band_gate_param(module):
    """Whatever the implementation called the per-band gate vector (spec leaves it open)."""
    for name, parameter in module.named_parameters():
        lowered = name.lower()
        if "band" in lowered and "gate" in lowered:
            return name, parameter
    for name in ("band_gate", "band_gates", "band_gate_logit"):
        parameter = getattr(module, name, None)
        if isinstance(parameter, torch.nn.Parameter):
            return name, parameter
    return None, None


def layout_parts(layout):
    """`band_layout()` -> `(bands, zeroed)`; the dict form is accepted too."""
    if isinstance(layout, dict):
        bands = layout.get("bands", layout.get("band_of_layer"))
        zeroed = layout.get("zeroed_layers", layout.get("zero_layers",
                                                       layout.get("forced_zero")))
        if bands is None or zeroed is None:
            raise AssertionError(f"band_layout dict lacks bands/zeroed: {sorted(layout)}")
        return list(bands), list(zeroed)
    if isinstance(layout, (tuple, list)) and len(layout) == 2:
        return list(layout[0]), list(layout[1])
    raise AssertionError(f"cannot read band_layout() -> {type(layout).__name__}")


def rms_of(slots):
    return slots.float().square().mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()


torch.manual_seed(0)

# ---- 1. the band map ------------------------------------------------------
report("1 BANDS is the three-part layer split",
       tuple(tuple(pair) for pair in df.BANDS) == ((0, 10), (11, 21), (22, 31)),
       str(tuple(df.BANDS)))
report("1 BAND_NAMES", tuple(df.BAND_NAMES) == ("lexical", "bridging", "readout"),
       str(tuple(df.BAND_NAMES)))
for layer, expected in ((0, 0), (5, 0), (10, 0), (11, 1), (15, 1), (21, 1),
                        (22, 2), (24, 2), (28, 2), (31, 2)):
    got = int(df.band_of_layer(layer))
    report(f"1 band_of_layer({layer}) == {expected}", got == expected, f"got {got}")

real_bands, real_zeroed = layout_parts(df.band_layout(UTILITY_LAYERS))
report("1 the real utility layers sit in bands 1 and 2, never in band 0",
       [int(band) for band in real_bands] == [1, 1, 1, 2, 2],
       f"bands={[int(band) for band in real_bands]} zeroed={[int(z) for z in real_zeroed]}")
if len(real_zeroed) == 0:
    note("1 no window layer of the real UTILITY_LAYERS is in band 0, so band0-zeroing "
         "is vacuous at the shipped layer list: section 2 uses layer_ids=(5,12,24) to "
         "give band 0 a layer at all")

custom_bands, custom_zeroed = layout_parts(df.band_layout(BAND_LAYERS))
report("1 band_layout maps one window layer per band onto bands 0,1,2",
       [int(band) for band in custom_bands] == [0, 1, 2],
       f"bands={[int(band) for band in custom_bands]}")
report("1 band_layout names the layer it will force to zero",
       set(int(value) for value in custom_zeroed) in ({5}, {0}),
       f"zeroed={sorted(int(value) for value in custom_zeroed)} "
       f"(layer number 5 or window index 0)")

# ---- 2. band_mode="three": band 0 is masked, not trained ------------------
w2 = fake_window(BAND_LAYERS, seed=3)
cond2 = fake_cond()
ctx2 = fake_context()
m_off = randomise(make_deep(band_mode="off", layer_ids=BAND_LAYERS))
m_three = make_deep(band_mode="three", band_gates=(0.0, 1.0, 1.0),
                    layer_ids=BAND_LAYERS)
m_three.load_state_dict(m_off.state_dict(), strict=False)

d_off = m_off(w2, cond2, ctx2)
d_three = m_three(w2, cond2, ctx2)
report("2 band 0's residual is exactly zero",
       float(d_three[0].abs().max()) == 0.0, f"{float(d_three[0].abs().max()):.3e}")
report("2 the parent's band 0 residual is not zero, so the mask is doing work",
       float(d_off[0].abs().max()) > 0.0, f"{float(d_off[0].abs().max()):.4f}")
report("2 band 1 and band 2 match the parent with their gates at 1",
       torch.allclose(d_three[1:], d_off[1:], atol=1e-6),
       f"max dev {float((d_three[1:] - d_off[1:]).abs().max()):.3e} "
       f"(exactly 0.0 if the gate is stored as its value, ~1e-7 if as a logit)")

gate_name, gate_param = band_gate_param(m_three)
if gate_param is not None and int(gate_param.numel()) == 3:
    with torch.no_grad():
        forced = gate_param.detach().clone()
        gate_param[0] = 1.0
    d_forced = m_three(w2, cond2, ctx2)
    with torch.no_grad():
        gate_param.copy_(forced)
    report("2 band 0 stays zero even when its own gate is forced to 1",
           float(d_forced[0].abs().max()) == 0.0, f"gate param {gate_name!r}")
    m_three.zero_grad()
    m_three(w2, cond2, ctx2).float().square().sum().backward()
    grad = gate_param.grad
    report("2 band 0's gate gets no gradient",
           grad is None or float(grad[0].abs()) == 0.0,
           "None" if grad is None else f"{float(grad[0].abs()):.3e}")
    report("2 the other bands' gates do get one (the check is live)",
           grad is not None and float(grad[1:].abs().sum()) > 0.0,
           "None" if grad is None else f"{float(grad[1:].abs().sum()):.3e}")
else:
    note("2 no 3-element band-gate parameter found; the gradient half of the band "
         "check is unverified (the spec does not fix the parameter's name)")

# ---- 3. key_channel="project": lock the key side, free the value side -----
basis = orthonormal_basis(LAYERS5)
w3 = fake_window(LAYERS5, seed=5)
cond3 = fake_cond()
ctx3 = fake_context()
KEY_RATIO, VALUE_RATIO = 0.10, 0.15

m_parent = randomise(make_deep(key_channel="off"))
d_parent = m_parent(w3, cond3, ctx3)
rms3 = rms_of(w3)
p_parent = project_row_space(d_parent, basis)
report("3 the unclamped parent residual does break the key bound",
       float((p_parent.abs() / rms3).max()) > KEY_RATIO,
       f"max |key part| / rms = {float((p_parent.abs() / rms3).max()):.4f}")

m_key = make_deep(key_channel="project", key_ratio=KEY_RATIO, value_ratio=VALUE_RATIO)
m_key.load_state_dict(m_parent.state_dict(), strict=False)
m_key.install_key_basis(basis)
d_key = m_key(w3, cond3, ctx3)
p_key = project_row_space(d_key, basis)
n_key = d_key - p_key
key_seen = float((p_key.abs() / rms3).max())
null_seen = float((n_key.abs() / rms3).max())
report("3 the key-visible residual stays inside key_ratio x rms",
       key_seen <= KEY_RATIO * (1.0 + 1e-5), f"{key_seen:.5f} vs cap {KEY_RATIO}")
report("3 the value/null residual stays inside value_ratio x rms",
       null_seen <= VALUE_RATIO * (1.0 + 1e-5), f"{null_seen:.5f} vs cap {VALUE_RATIO}")
graft_null = float((n_key[:, GRAFT].abs() / rms3[:, GRAFT]).max())
report("3 the null side actually reaches its ceiling on the graft half",
       graft_null >= 0.9 * VALUE_RATIO, f"{graft_null:.5f} vs cap {VALUE_RATIO}")
report("3 both channels still write",
       float(p_key.abs().max()) > 0.0 and float(n_key.abs().max()) > 0.0,
       f"key {float(p_key.abs().max()):.4f} null {float(n_key.abs().max()):.4f}")
report("3 the key basis is not a parameter",
       "key_basis" not in m_key.state_dict()
       and not isinstance(getattr(m_key, "key_basis", None), torch.nn.Parameter))

m_free = make_deep(key_channel="project", key_ratio=KEY_RATIO)   # value_ratio=None
m_free.load_state_dict(m_parent.state_dict(), strict=False)
m_free.install_key_basis(basis)
d_free = m_free(w3, cond3, ctx3)
n_free = d_free - project_row_space(d_free, basis)
free_seen = float((n_free.abs() / rms3).max())
report("3 value_ratio=None falls back to the parent max_ratio",
       VALUE_RATIO + 1e-3 < free_seen <= 0.5 * (1.0 + 1e-5),
       f"{free_seen:.5f} (above the {VALUE_RATIO} this module clipped at before, "
       f"inside the parent 0.5)")

m_missing = make_deep(key_channel="project", key_ratio=KEY_RATIO)
m_missing.load_state_dict(m_parent.state_dict(), strict=False)
raised = False
try:
    m_missing(w3, cond3, ctx3)
except Exception as exc:                                  # noqa: BLE001
    raised = True
    detail = f"{type(exc).__name__}: {exc}"
report("3 a missing key basis raises instead of silently writing free", raised, detail)

# ---- 4. graft_attention_calibration: graft only, tau shrinks the key ------
# The reader key is `W_k . input_layernorm(h)`, and the layernorm is scale invariant, so
# inflating a slot does NOT make its key longer: only its DIRECTION does. The graft slots
# are therefore laid along the long direction of the stand-in `k_proj`, which is what a
# key-norm outlier looks like, and the core slots stay random.
scorer4 = FakeScorer(KVL, layers=LAYERS5)
meta4 = fake_meta(scorer=scorer4)
w4 = fake_window(LAYERS5, seed=11)
with torch.no_grad():
    for position, layer in enumerate(LAYERS5):
        direction = scorer4.top_direction(layer)
        w4[position, GRAFT] = (20.0 * direction.unsqueeze(0)
                               + 0.01 * w4[position, GRAFT])

same0 = df.graft_attention_calibration(w4, meta4, basis, layer_ids=LAYERS5,
                                       return_report=True)
report("4 return_report=True returns (window, report)",
       isinstance(same0, tuple) and len(same0) == 2, str(type(same0).__name__))
unchanged, rep0 = same0
report("4 tau=1, bias=0, norm_quantile=0 is a no-op", torch.equal(unchanged, w4))

damped, rep = df.graft_attention_calibration(w4, meta4, basis, tau=0.5,
                                             layer_ids=LAYERS5, return_report=True)
p_before, n_before = project_row_space(w4, basis), w4 - project_row_space(w4, basis)
p_after = project_row_space(damped, basis)
n_after = damped - p_after
report("4 core slots are bit-identical", torch.equal(damped[:, CORE], w4[:, CORE]))
report("4 the graft key-visible part is scaled by tau",
       torch.allclose(p_after[:, GRAFT], 0.5 * p_before[:, GRAFT], atol=1e-3),
       f"max dev {float((p_after[:, GRAFT] - 0.5 * p_before[:, GRAFT]).abs().max()):.2e}")
report("4 the graft key-visible NORM falls by exactly tau",
       torch.allclose(p_after[:, GRAFT].norm(dim=-1),
                      0.5 * p_before[:, GRAFT].norm(dim=-1), atol=1e-3),
       f"{float(p_before[:, GRAFT].norm()):.3f} -> {float(p_after[:, GRAFT].norm()):.3f}")
report("4 the graft null part is untouched",
       torch.allclose(n_after[:, GRAFT], n_before[:, GRAFT], atol=1e-3),
       f"max dev {float((n_after[:, GRAFT] - n_before[:, GRAFT]).abs().max()):.2e}")
need = ("slots_touched", "key_scale_mean", "key_scale_min", "key_norm_before",
        "key_norm_after", "logit_proxy", "tau", "bias", "norm_quantile", "note")
report("4 the damping report carries every field the trainer logs",
       all(key in rep for key in need), str([key for key in need if key not in rep]))
report("4 slots_touched counts the graft half only",
       int(rep["slots_touched"]) == int(GRAFT.numel()), str(rep["slots_touched"]))
report("4 the reader key norm the report measures falls with tau",
       float(rep["key_norm_after"]) < float(rep["key_norm_before"]),
       f"{float(rep['key_norm_before']):.3f} -> {float(rep['key_norm_after']):.3f}")
report("4 key_scale_mean is tau when nothing else is on",
       abs(float(rep["key_scale_mean"]) - 0.5) < 1e-5, f"{float(rep['key_scale_mean']):.5f}")
report("4 logit_proxy is a finite non-negative first-order proxy",
       math.isfinite(float(rep["logit_proxy"])) and float(rep["logit_proxy"]) >= 0.0,
       f"{float(rep['logit_proxy']):.4g}")

shrunk, rep_q = df.graft_attention_calibration(w4, meta4, basis, norm_quantile=0.5,
                                               layer_ids=LAYERS5, return_report=True)
p_q = project_row_space(shrunk, basis)
report("4 norm_quantile only shrinks (s = min(1, target/||key||))",
       float(rep_q["key_scale_mean"]) < 1.0 and float(rep_q["key_scale_min"]) > 0.0
       and float(p_q[:, GRAFT].norm()) < float(p_before[:, GRAFT].norm()),
       f"scale mean {float(rep_q['key_scale_mean']):.4f} "
       f"min {float(rep_q['key_scale_min']):.4f}")
_biased, rep_b = df.graft_attention_calibration(w4, meta4, basis, bias=1.0,
                                                layer_ids=LAYERS5, return_report=True)
note(f"4 bias=+1: key_scale_mean={float(rep_b['key_scale_mean']):.4f} at "
     f"logit_proxy={float(rep_b['logit_proxy']):.4g} "
     f"(informational: the spec's s_bias is a first-order proxy, not a measurement)")

# ---- 5. the V17 calibrate wrapper keeps the parent's contract -------------
m_wrap = randomise(make_deep())
m_wrap.install_key_basis(basis)
meta5 = fake_meta(scorer=FakeScorer(KVL, layers=LAYERS5))
out5 = df.calibrate(w3, meta5, m_wrap, context=ctx3, return_report=True)
report("5 return_report=True returns (slots, info, context)",
       isinstance(out5, tuple) and len(out5) == 3, str(type(out5).__name__))
calibrated, info, _ = out5
report("5 the calibrated window keeps its shape",
       tuple(calibrated.shape) == tuple(w3.shape), str(tuple(calibrated.shape)))
parent_keys = ("relative_delta", "core_relative_delta", "graft_relative_delta",
               "cross_proj_norm", "core_slots", "graft_slots")
report("5 the parent's report keys survive the wrapper",
       all(key in info for key in parent_keys),
       str([key for key in parent_keys if key not in info]))
report("5 the slot split is unchanged",
       info["core_slots"] == CORE_SLOTS and info["graft_slots"] == int(GRAFT.numel()),
       f"{info['core_slots']}/{info['graft_slots']}")
report("5 the wrapper always records the V17 switches",
       isinstance(info.get("v17"), dict)
       and info["v17"].get("key_channel") == "off"
       and info["v17"].get("band_mode") == "off",
       str(sorted(info.get("v17") or {}))[:140])

import inspect  # noqa: E402  (only used to probe the constructor below)

SIGNATURE = inspect.signature(df.DeepFusionResidual.__init__).parameters
tau_key = next((name for name in ("graft_tau", "tau", "attention_tau")
                if name in SIGNATURE), None)
if tau_key is None:
    note("5 the constructor carries no tau/bias/norm_quantile kwarg, so a module built "
         "from config() cannot re-enter graft_attention_calibration: the eval arm "
         "V17Attn ('tau/bias from the checkpoint') has nothing to read them from")
else:
    m_attn = randomise(make_deep(**{tau_key: 0.5}))
    m_attn.install_key_basis(basis)
    # `context=None`: damping rewrites the very keys a context row is built from, so the
    # wrapper refuses a stale one instead of silently mixing the two windows.
    out5b = df.calibrate(w3, meta5, m_attn, return_report=True, basis=basis)
    damped_window, info_attn, _ = out5b
    attention = (info_attn.get("v17") or {}).get("attention") or {}
    report("5 the damping report lands in info['v17']['attention']",
           isinstance(attention, dict) and abs(float(attention.get("tau", -1)) - 0.5) < 1e-9,
           str(sorted(attention))[:140])
    report("5 damping ran before the calibration and changed the window",
           not torch.equal(damped_window, calibrated),
           f"max dev {float((damped_window - calibrated).abs().max()):.3e}")
    try:
        df.calibrate(w3, meta5, m_attn, context=ctx3, return_report=True, basis=basis)
        refused = False
    except ValueError:
        refused = True
    report("5 a context computed before the damping is refused, not reused", refused)

# ---- 6. sim_floor in cross_block_context ---------------------------------
KV, POOL_HID, BLOCK, NBLOCKS = 8, 16, 8, 4
N6 = BLOCK * NBLOCKS
BLOCKS6 = [index // BLOCK for index in range(N6)]


def similarity_window(seed=17):
    """Keys laid out so the similarity structure -- and therefore the regimes -- are known.

    block 0 sits on one direction; block 1 has two strong matches to it (0.9) and six
    middling ones (0.71); blocks 2 and 3 are orthogonal to everything of block 0. With
    k=4 that gives, for a block-0 row: floor 0.8 -> exactly 2 above (the mixed regime),
    floor 0.95 -> none above (the fallback regime), and never a reshuffled top-4.
    """
    torch.manual_seed(seed)
    window = torch.zeros(1, N6, POOL_HID)
    for index in range(N6):
        direction = torch.zeros(KV)
        block = index // BLOCK
        if block == 0:
            direction[0] = 1.0
        elif block == 1:
            local = index - BLOCK
            if local < 2:
                direction[0], direction[1] = 0.9, math.sqrt(1.0 - 0.81)
            else:
                direction[0], direction[1] = math.sqrt(0.5), math.sqrt(0.5)
        elif block == 2:
            direction[2] = 1.0
        else:
            direction[3] = 1.0
        direction = direction + 0.01 * torch.randn(KV)
        window[0, index, :KV] = direction / direction.norm()
        window[0, index, KV:] = torch.randn(POOL_HID - KV)
    return window


def reference_context(key, value, block, k=4, floor=None):
    """The aggregation the module is supposed to perform, written out longhand."""
    similarity = (key @ key.t()).masked_fill(
        block.unsqueeze(0) == block.unsqueeze(1), -2.0)
    masked = similarity
    if floor is not None and float(floor) > 0.0:
        legal = similarity > -1.5
        below = legal & (similarity < float(floor))
        cleaned = similarity.masked_fill(below, float("-inf"))
        survives = (cleaned > float("-inf")).any(dim=-1, keepdim=True)
        masked = torch.where(survives, cleaned, similarity)
    picked = masked.topk(k, dim=-1).indices
    weights = torch.softmax(masked.gather(1, picked), dim=-1)
    gathered = value.index_select(0, picked.reshape(-1)).reshape(N6, k, -1)
    return (gathered * weights.unsqueeze(-1)).sum(dim=1), masked


window6 = similarity_window()
meta6 = fake_meta(slot_types=[0] * N6, blocks=BLOCKS6, slots=N6, scorer=FakeScorer(KV))
key6 = _unit(window6[0][:, :KV])
value6 = window6[0][:, KV:]
block6 = torch.tensor(BLOCKS6)

ctx_plain = cb.cross_block_context(window6, meta6, k=4)
ref_plain, similarity6 = reference_context(key6, value6, block6)
report("6 the reference reproduces V13's own context (so the checks below mean "
       "something)", torch.allclose(ctx_plain, ref_plain, atol=1e-6),
       f"max dev {float((ctx_plain - ref_plain).abs().max()):.2e}")

ctx_off, rep_off = cb.cross_block_context(window6, meta6, k=4, sim_floor=0.0,
                                          return_report=True)
report("6 sim_floor=0 is bit-identical to the unfloored path",
       torch.equal(ctx_off, ctx_plain))
report("6 a floor of 0 reports no below-floor fraction",
       float(rep_off.get("below_floor_fraction") or 0.0) == 0.0,
       str(rep_off.get("below_floor_fraction")))

ctx_08, rep_08 = cb.cross_block_context(window6, meta6, k=4, sim_floor=0.8,
                                        return_report=True)
above = (similarity6 > -1.5) & (similarity6 >= 0.8)
count_above = above.sum(dim=-1)
ref_08, _ = reference_context(key6, value6, block6, floor=0.8)
report("6 the setup exercises the mixed regime (some rows keep fewer than k seats)",
       int(((count_above > 0) & (count_above < 4)).sum()) > 0,
       f"{int(((count_above > 0) & (count_above < 4)).sum())} rows with 1-3 candidates")
report("6 filtered rows match an aggregation over the above-floor candidates only",
       torch.allclose(ctx_08, ref_08, atol=1e-4),
       f"max dev {float((ctx_08 - ref_08).abs().max()):.2e}")
rich = count_above >= 4
report("6 rows with at least k candidates are untouched by the floor",
       rich.any() and torch.allclose(ctx_08[rich], ctx_plain[rich], atol=1e-5),
       f"{int(rich.sum())} rows unchanged")
mixed = (count_above > 0) & (count_above < 4)
report("6 the floor changes what a starved row actually reads",
       mixed.any() and not torch.allclose(ctx_08[mixed], ctx_plain[mixed], atol=1e-6),
       f"{int(mixed.sum())} rows filtered")
report("6 below_floor_fraction is a fraction in (0, 1)",
       0.0 < float(rep_08["below_floor_fraction"]) < 1.0,
       f"{float(rep_08['below_floor_fraction']):.3f}")

ctx_095, rep_095 = cb.cross_block_context(window6, meta6, k=4, sim_floor=0.95,
                                          return_report=True)
max_legal = similarity6.masked_fill(similarity6 <= -1.5, float("-inf")).max(dim=-1).values
report("6 every row is below a 0.95 floor, so every row falls back",
       float(max_legal.max()) < 0.95, f"best legal similarity {float(max_legal.max()):.4f}")
report("6 an all-below row keeps its unfloored scores (no NaN, nothing selected at "
       "-inf)", torch.isfinite(ctx_095).all() and torch.allclose(ctx_095, ctx_plain,
                                                                 atol=1e-5),
       f"max dev {float((ctx_095 - ctx_plain).abs().max()):.2e}")
report("6 the fallback is counted, not silent",
       abs(float(rep_095["below_floor_fraction"]) - 1.0) < 1e-6
       and int(rep_095.get("floor_fallback_rows", -1)) == N6,
       f"fraction {float(rep_095['below_floor_fraction']):.4f} "
       f"fallback_rows {rep_095.get('floor_fallback_rows')} of {N6}")

# ---- 7. config() round trip ----------------------------------------------
m_cfg = make_deep(key_channel="project", key_ratio=0.10, value_ratio=0.30,
                  band_mode="three", band_gates=(0.0, 1.0, 1.0), sim_floor=0.35,
                  layer_ids=(12, 16, 20, 24, 28))
config = m_cfg.config()
v17_keys = ("key_channel", "key_ratio", "value_ratio", "band_mode", "band_gates",
            "sim_floor", "layer_ids")
report("7 config() names every V17 knob", all(key in config for key in v17_keys),
       str([key for key in v17_keys if key not in config]))
parent_config = cb.CrossBlockResidual(hidden_size=HID, rank=RANK, kv_dim=KVL).config()
report("7 config() still carries the parent's kwargs",
       all(key in config for key in parent_config),
       str([key for key in parent_config if key not in config]))
report("7 config() records the values it was built with",
       config["key_channel"] == "project" and abs(float(config["key_ratio"]) - 0.10) < 1e-9
       and abs(float(config["value_ratio"]) - 0.30) < 1e-9
       and config["band_mode"] == "three"
       and abs(float(config["sim_floor"]) - 0.35) < 1e-9
       and tuple(int(x) for x in config["layer_ids"]) == (12, 16, 20, 24, 28)
       and tuple(float(x) for x in config["band_gates"]) == (0.0, 1.0, 1.0),
       str({key: config[key] for key in v17_keys}))

rebuilt = df.DeepFusionResidual(**config)
report("7 rebuilding from config() reproduces the same parameter shapes",
       all(rebuilt.state_dict()[name].shape == tensor.shape
           for name, tensor in m_cfg.state_dict().items())
       and set(rebuilt.state_dict()) == set(m_cfg.state_dict()),
       f"{len(rebuilt.state_dict())} tensors")
again = rebuilt.config()
report("7 the round trip is a fixed point",
       all(again[key] == config[key] or tuple(again[key]) == tuple(config[key])
           for key in config), str([key for key in config if again.get(key) != config[key]]))

# ---- 8. density quota ----------------------------------------------------
POOL_BLOCK, POOL_BLOCKS = 256, 6
CARRIERS = [0, 1, 2, 3]
POOL_SLOTS = POOL_BLOCK * POOL_BLOCKS          # 1,536 = 6 blocks of 256
CORE_PER_BLOCK, GRAFT_PER_BLOCK = 250, 6
assert CORE_PER_BLOCK + GRAFT_PER_BLOCK == 256, "a block is 256 slots, core + graft"


def build_pool(seed=13):
    """Six 256-slot blocks whose inside-block redundancy is known by construction."""
    generator = torch.Generator().manual_seed(seed)
    pool = torch.randn(1, POOL_SLOTS, HID, generator=generator)
    for block, spread in ((0, 0.01), (1, 0.35), (2, 0.9), (3, 1.4), (4, 1.0), (5, 1.0)):
        base = _unit(torch.randn(KV, generator=generator))
        noise = torch.randn(POOL_BLOCK, KV, generator=generator)
        directions = base.unsqueeze(0) + spread * noise
        pool[0, block * POOL_BLOCK:(block + 1) * POOL_BLOCK, :KV] = _unit(directions)
    return pool


pool = build_pool()
probes = [dict(probe=0)]
joint_fixed = ew.JointConsolidation(FakeScorer(KV), probes,
                                   core_per_block=CORE_PER_BLOCK,
                                   graft_per_block=GRAFT_PER_BLOCK)
joint_dense = ew.JointConsolidation(FakeScorer(KV), probes,
                                    core_per_block=CORE_PER_BLOCK,
                                    graft_per_block=GRAFT_PER_BLOCK,
                                    quota_mode="density", min_core=200, max_core=300,
                                    density_sample=32)
window_f, meta_f, tel_f = joint_fixed.build(pool, CARRIERS, block_size=POOL_BLOCK)
window_d, meta_d, tel_d = joint_dense.build(pool, CARRIERS, block_size=POOL_BLOCK)

report("8 the shipped quota is 9 x 180 = 1,620 core slots",
       ew.QUOTA["core_per_block"] * 9 == 1620, str(ew.QUOTA))
report("8 fixed mode keeps the old telemetry and spends the quota evenly",
       tel_f["quota_mode"] == "fixed"
       and sorted(int(x) for x in tel_f["per_block_quota"]) == [CORE_PER_BLOCK] * 4
       and tel_f["density"] is None,
       str(tel_f["per_block_quota"]))
report("8 density mode records how it spent the same budget",
       tel_d["quota_mode"] == "density" and tel_d["density"] is not None
       and len(tel_d["density"]) == len(CARRIERS),
       f"density={[round(float(x), 3) for x in (tel_d['density'] or [])]}")
quotas = [int(value) for value in tel_d["per_block_quota"]]
report("8 the density quota sums to len(carriers) * core_per_block",
       sum(quotas) == len(CARRIERS) * CORE_PER_BLOCK
       and int(tel_d["quota_total"]) == len(CARRIERS) * CORE_PER_BLOCK,
       f"{sum(quotas)} vs {len(CARRIERS) * CORE_PER_BLOCK}")
report("8 every block's quota sits in [min_core, max_core]",
       all(200 <= value <= 300 for value in quotas), str(quotas))
report("8 the total capacity does not move between quota modes",
       window_d.shape[1] == window_f.shape[1] == len(CARRIERS) * CORE_PER_BLOCK
       + len(CARRIERS) * GRAFT_PER_BLOCK,
       f"{window_f.shape[1]} / {window_d.shape[1]}")
report("8 the redundant block is pinned at the floor and a distinct one is not",
       quotas[0] == 200 and max(quotas[1:]) > quotas[0],
       f"block0 (near-copies) {quotas[0]}, others {quotas[1:]}")
report("8 density mode still copies its sources verbatim",
       torch.equal(window_d, pool.index_select(1, meta_d["source_position"])))
report("8 density mode is deterministic (the quota is a function of the window)",
       [int(value) for value in
        joint_dense.build(pool, CARRIERS, block_size=POOL_BLOCK)[2]["per_block_quota"]]
       == quotas)
report("8 meta['pool_block'] closes position -> block over the whole source pool",
       "pool_block" in meta_d
       and tuple(meta_d["pool_block"].shape) == (POOL_SLOTS,)
       and torch.equal(meta_d["pool_block"].to(torch.long),
                       torch.arange(POOL_SLOTS) // POOL_BLOCK),
       f"{tuple(meta_d['pool_block'].shape) if 'pool_block' in meta_d else 'MISSING'}")

# ---- 9. absorb_cos: near-duplicates are merged, not dropped ---------------
joint_absorb = ew.JointConsolidation(FakeScorer(KV), probes,
                                     core_per_block=CORE_PER_BLOCK,
                                     graft_per_block=GRAFT_PER_BLOCK,
                                     absorb_cos=0.9, absorb_alpha_max=0.5)
window_a, meta_a, tel_a = joint_absorb.build(pool, CARRIERS, block_size=POOL_BLOCK)
source_of = pool.index_select(1, meta_a["source_position"])
changed = (window_a != source_of).any(dim=-1).any(dim=0)
report("9 absorption off changes nothing (the V13 window is the default)",
       int(tel_f["absorbed_pairs"]) == 0
       and torch.equal(window_f, pool.index_select(1, meta_f["source_position"])))
report("9 absorption on reports the pairs it merged",
       int(tel_a["absorbed_pairs"]) > 0, f"{int(tel_a['absorbed_pairs'])} pairs "
       f"per block {tel_a['absorbed_per_block']}")
report("9 alpha never exceeds absorb_alpha_max",
       float(tel_a["absorbed_mean_alpha"]) <= 0.5 + 1e-6
       and float(tel_a["absorbed_mean_alpha"]) > 0.0,
       f"mean alpha {float(tel_a['absorbed_mean_alpha']):.4f} (cap 0.5)")
report("9 every merged pair passed the cosine gate",
       float(tel_a["absorbed_mean_cos"]) > float(tel_a["absorb_cos"]),
       f"mean cos {float(tel_a['absorbed_mean_cos']):.4f} > {float(tel_a['absorb_cos'])}")
report("9 only slots that were merged left their source verbatim",
       int(changed.sum()) <= int(tel_a["absorbed_pairs"]),
       f"{int(changed.sum())} changed slots for {int(tel_a['absorbed_pairs'])} pairs "
       f"(several evicted slots may fold into the same kept slot)")
report("9 absorption only moves core slots",
       int((meta_a["slot_type"][changed] != 0).sum()) == 0,
       f"{int((meta_a['slot_type'][changed] != 0).sum())} graft slots moved")
report("9 a merged slot keeps its RMS",
       torch.allclose(rms_of(window_a)[0][changed], rms_of(source_of)[0][changed],
                      atol=1e-5),
       f"max dev {float((rms_of(window_a)[0][changed] - rms_of(source_of)[0][changed]).abs().max()):.2e}")

# ---- 10. --cf-mode random: the seed derivation is reproducible ------------
def parser_flags(path):
    """`{flag: {choices, default}}` read straight off the source, no import needed."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    flags = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        flag = node.args[0]
        if not (isinstance(flag, ast.Constant) and isinstance(flag.value, str)):
            continue
        keywords = {item.arg: item.value for item in node.keywords if item.arg}
        choices = None
        if isinstance(keywords.get("choices"), (ast.Tuple, ast.List)):
            choices = [element.value for element in keywords["choices"].elts
                       if isinstance(element, ast.Constant)]
        default = keywords["default"].value if isinstance(keywords.get("default"),
                                                          ast.Constant) else None
        flags[flag.value] = dict(choices=choices, default=default)
    return flags


class _CfArgs:
    """The one attribute the trainer's `cf_seed` may read off `args`."""

    __slots__ = ("cf_random_seed",)

    def __init__(self, seed):
        self.cf_random_seed = int(seed)


def cf_seed_callable(path):
    """The trainer's `(base seed, document id) -> seed` helper, lifted out of the source.

    The spec fixes the DERIVATION (`--cf-random-seed` + document id) but not the helper,
    so it is found by shape: a module-level function whose name mentions both ``cf`` and
    ``seed`` and that takes a base seed (or the argparse namespace holding it) plus a
    document id. The source is executed on its own, which keeps this check runnable where
    `transformers` is not installed -- i.e. everywhere except the training box.
    """
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        lowered = node.name.lower()
        if "cf" not in lowered or "seed" not in lowered:
            continue
        arguments = node.args
        if arguments.vararg or arguments.kwarg:
            continue
        positional = len(arguments.args)
        if positional not in (2, 3):
            continue
        source = ast.get_source_segment(text, node)
        if source is None:
            continue
        namespace = dict(np=np_for_seeds(), torch=torch, hashlib=hashlib,
                         random=random, math=math, zlib=zlib, os=os)
        try:
            exec(compile(source, str(path), "exec"), namespace)   # noqa: S102
        except Exception:                                        # noqa: BLE001
            continue
        function = namespace.get(node.name)
        if callable(function):
            # 3 positional parameters means the trainer passes its argparse namespace;
            # 2 means the base seed itself (the shape the spec describes).
            return function, node.name, ("args" if positional == 3 else "base")
    return None, None, None


def np_for_seeds():
    try:
        import numpy as _np
    except Exception:                                            # noqa: BLE001
        return None
    return _np


flags = parser_flags(TRAINER)
cf_choices = (flags.get("--cf-mode") or {}).get("choices")
report("10 the trainer exposes --cf-mode, defaulting to donor",
       "--cf-mode" in flags and flags["--cf-mode"]["default"] == "donor",
       str(flags.get("--cf-mode")))
if cf_choices is None:
    note("10 --cf-mode carries no argparse choices: the three modes are not pinned by "
         "the parser, so a typo reaches the training loop instead of the CLI")
else:
    report("10 --cf-mode offers exactly donor/random/shuffle",
           set(cf_choices) == {"donor", "random", "shuffle"}, str(cf_choices))
report("10 --cf-random-seed defaults to 0",
       "--cf-random-seed" in flags and flags["--cf-random-seed"]["default"] == 0,
       str(flags.get("--cf-random-seed")))
v17_flags = ("--architecture", "--key-channel", "--key-ratio", "--value-ratio",
             "--band-mode", "--band-gates", "--sim-floor", "--graft-tau",
             "--quota-mode", "--min-core", "--block-merge-cos", "--kl-topk")
report("10 the rest of the V17 CLI surface is present",
       all(flag in flags for flag in v17_flags),
       str([flag for flag in v17_flags if flag not in flags]))

seed_function, seed_name = cf_seed_callable(TRAINER)
seed_function, seed_name, seed_form = cf_seed_callable(TRAINER)
if seed_function is None:
    note("10 no module-level `*cf*seed*(base, doc)` helper in the trainer: the "
         "--cf-mode random derivation cannot be exercised here. The spec fixes the "
         "inputs (base seed + document id) but not the helper, so the test looks for "
         "a function of exactly that shape")
else:
    def derive(base, document_id):
        """`--cf-random-seed` x document id -> seed, in whichever shape it was written."""
        if seed_form == "args":
            return int(seed_function(_CfArgs(base), document_id))
        return int(seed_function(base, document_id))

    try:
        first = derive(0, 7)
        failed = None
    except Exception as exc:                                     # noqa: BLE001
        first, failed = None, f"{type(exc).__name__}: {exc}"
    if failed is not None:
        note(f"10 {seed_name}() could not be called in isolation ({failed}): it reads "
             f"more of the trainer's state than the base seed and the document id")
    else:
        report(f"10 {seed_name}() is a pure function of its arguments",
               first == derive(0, 7), f"{first} == {derive(0, 7)}")
        report("10 the derivation depends on the document id",
               derive(0, 7) != derive(0, 8), f"{derive(0, 7)} vs {derive(0, 8)}")
        report("10 the derivation depends on the base seed",
               derive(0, 7) != derive(1, 7), f"{derive(0, 7)} vs {derive(1, 7)}")
        torch.manual_seed(1234)
        _ = torch.randn(4)
        left = derive(3, 11)
        torch.manual_seed(9876)
        _ = torch.randn(4)
        right = derive(3, 11)
        report("10 the derivation does not read the global RNG", left == right,
               f"{left} == {right}")

# ---- 11. every knob closed reproduces V13 -------------------------------
plain = cb.CrossBlockResidual(hidden_size=HID, rank=RANK, kv_dim=KVL, k_neighbors=4)
randomise(plain, seed=9)
deep = df.DeepFusionResidual(hidden_size=HID, rank=RANK, kv_dim=KVL, k_neighbors=4,
                             layer_ids=LAYERS5)
deep.load_state_dict(plain.state_dict(), strict=False)
extra = sorted(set(deep.state_dict()) - set(plain.state_dict()))
report("11 the only extra state a V17 module carries is the band gate",
       all("band" in name for name in extra), str(extra))
meta11 = fake_meta()
out_plain, _ = cb.calibrate(w3, meta11, plain, context=ctx3)
out_deep, info_deep = df.calibrate(w3, meta11, deep, context=ctx3)
report("11 defaults reproduce the V13 calibration bit-for-bit",
       torch.equal(out_plain, out_deep),
       f"max dev {float((out_plain - out_deep).abs().max()):.2e}")
report("11 the V17 wrapper reports the V13 keys on the default path",
       all(key in info_deep for key in ("relative_delta", "core_relative_delta",
                                        "graft_relative_delta")),
       str(sorted(info_deep))[:160])
report("11 the deep-fusion checkpoint schema is the one the evaluator looks for",
       str(df.SCHEMA).startswith("deep_fusion/1") and callable(df.load_deep),
       str(getattr(df, "SCHEMA", "MISSING")))

if NOTES:
    print(f"\n{len(NOTES)} note(s) above are measurements, not results:")
    for text in NOTES:
        print("  - " + text.splitlines()[0])
print("ALL CHECKS PASSED")
