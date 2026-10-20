"""Item 3: checkpoint save/verify contract, all in /tmp (never runs/).

1. Execute the *actual* `checkpoint=dict(...)` expression from
   tools/natural_merge_v7_train.py in a stub namespace, so the save side is
   verified by running the real code, not by reading it.
2. Round-trip it through torch.save/torch.load(weights_only=True) and validate
   it as a v7 checkpoint; load_merge_state must reproduce the digest.
3. Mutual rejection matrix against the real v5 and v6 checkpoints on disk.
"""
import ast, json, sys, types
from pathlib import Path

import torch

sys.path.insert(0, "tools"); sys.path.insert(0, ".")
from natural_merge_runtime import (build_merge, critical_manifest, load_merge_state,
                                   source_manifest, state_digest, validate_merge_checkpoint)
from prepare_natural_merge_v7 import TRAIN_V7

OUT = Path("/tmp/v7dryrun/ckpt")
OUT.mkdir(parents=True, exist_ok=True)
report = {}

# ---- 1. execute the trainer's checkpoint expression -------------------------
source = Path("tools/natural_merge_v7_train.py").read_text()
tree = ast.parse(source)
assign = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
          and any(getattr(t, "id", "") == "checkpoint" for t in n.targets)
          and isinstance(n.value, ast.Call) and getattr(n.value.func, "id", "") == "dict"]
assert len(assign) == 1, "expected exactly one checkpoint=dict(...) in the trainer"
expression = ast.Expression(assign[0].value)
ast.fix_missing_locations(expression)
keywords = {kw.arg for kw in assign[0].value.keywords}
report["checkpoint_dict_keys"] = sorted(keywords)

variant = TRAIN_V7["variants"]["DsupV7"]
module = build_merge(variant["architecture"], 32, 4096, 256)
state = {k: v.detach().cpu() for k, v in module.state_dict().items()}
namespace = dict(
    TRAIN_V7=TRAIN_V7, variant=variant, merge=types.SimpleNamespace(state_dict=lambda: state),
    final=state_digest(module), eligible=True, GEOMETRY=TRAIN_V7["geometry"],
    plan=dict(content_sha256="0" * 64, tokenizer="tokenizer-manifest",
              episodes=[dict(id="train-0", contexts=[
                  dict(article="Article_A", context_sha256="c" * 64, qas=[dict(id="qa0")])])]),
    args=types.SimpleNamespace(variant="DsupV7", plan="/tmp/v7dryrun/plan.json",
                               checkpoint_out=str(OUT / "DsupV7.train.pt")),
    base=dict(protocol=dict(budget_blocks=20, window_blocks=13, target_blocks=9)),
    sha=lambda path: "1" * 64, source_manifest=source_manifest,
    report=dict(model=dict(path="stub", sha256="2" * 64)),
    expected=["c" * 64], covered_qas=["qa0"],
    set=set, sorted=sorted, list=list, dict=dict, len=len, str=str,
)
saved = eval(compile(expression, "<trainer checkpoint>", "eval"), namespace)
report["source_manifest_written"] = "source_manifest" in saved
report["source_manifest_entries"] = len(saved.get("source_manifest", {}))
report["critical_files_in_manifest"] = sorted(
    set(critical_manifest()) - set(saved.get("source_manifest", {}))) == []
report["training_is_TRAIN_V7"] = saved["training"] == TRAIN_V7
report["geometry"] = saved["geometry"]

path = OUT / "DsupV7.train.pt"
torch.save(saved, path)
restored = torch.load(path, map_location="cpu", weights_only=True)
report["roundtrip_validate_v7"] = bool(
    validate_merge_checkpoint(restored, variant["architecture"]))
fresh = build_merge(variant["architecture"], 32, 4096, 256)
load_merge_state(fresh, restored, variant["architecture"])
report["roundtrip_state_digest_equal"] = state_digest(fresh) == state_digest(module)

# ---- 2. mutual rejection matrix -------------------------------------------
def attempt(label, checkpoint, architecture):
    try:
        validate_merge_checkpoint(checkpoint, architecture)
        return label, "accepted"
    except ValueError as error:
        return label, "rejected: " + str(error)[:120]


matrix = []
matrix.append(attempt("v7 ckpt / v7 arch", restored, "grouped_gated_sparse_slot_v7"))
matrix.append(attempt("v7 ckpt / v6 arch", restored, "grouped_gated_sparse_slot_v6"))
matrix.append(attempt("v7 ckpt / v5 arch", restored, "grouped_gated_sparse_slot_v5"))
matrix.append(attempt("v7 ckpt / native_v7 arch", restored, "grouped_gated_sparse_native_v7"))
for name, arch in (("runs/natural_merge_v5_20261018_1050/Dsup.train.pt", "grouped_gated_sparse_slot_v5"),
                   ("runs/natural_merge_v5_20261018_1050/Dsup.train.pt", "grouped_gated_sparse_slot_v7"),
                   ("runs/natural_merge_v6_20261019_0812/DsupV6.train.pt", "grouped_gated_sparse_slot_v6"),
                   ("runs/natural_merge_v6_20261019_0812/DsupV6.train.pt", "grouped_gated_sparse_slot_v7")):
    if not Path(name).exists():
        matrix.append((name + " / " + arch, "missing on disk"))
        continue
    matrix.append(attempt(name + " / " + arch, torch.load(name, map_location="cpu", weights_only=True), arch))
report["matrix"] = matrix

# ---- 3. native variant checkpoint also carries the manifest ----------------
variant = TRAIN_V7["variants"]["DnativeV7"]
module = build_merge(variant["architecture"], 32, 4096, 256)
namespace.update(variant=variant, merge=types.SimpleNamespace(state_dict=lambda: module.state_dict()),
                 final=state_digest(module),
                 args=types.SimpleNamespace(variant="DnativeV7", plan="/tmp/v7dryrun/plan.json",
                                            checkpoint_out=str(OUT / "DnativeV7.train.pt")))
saved_native = eval(compile(expression, "<trainer checkpoint>", "eval"), namespace)
native_path = OUT / "DnativeV7.train.pt"
torch.save(saved_native, native_path)
native_restored = torch.load(native_path, map_location="cpu", weights_only=True)
report["native_roundtrip_validate_v7"] = bool(
    validate_merge_checkpoint(native_restored, variant["architecture"]))
report["native_rejected_by_slot_v7"] = attempt("native ckpt / slot_v7", native_restored,
                                               "grouped_gated_sparse_slot_v7")[1].startswith("rejected")
report["saved_bytes"] = {"DsupV7.train.pt": path.stat().st_size,
                         "DnativeV7.train.pt": native_path.stat().st_size}
print(json.dumps(report, indent=1))
