"""CPU audit using the unchanged production compactor and pool injection."""
import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.nn.utils import parametrize
from methods.prlmc.consolidate import CarrierResidualCompactor
from methods.prlmc.probes import _pool_in_place
from s1_soft_local import make_displacement


def audit(mode, dtype, state, memory, basis):
    def setup():
        model = nn.Module()
        model.memory = nn.Parameter(memory.clone(), requires_grad=False)
        manager = SimpleNamespace(model=model, total=12, used=8)
        arm = CarrierResidualCompactor(2, 8, block=2, rank=4, gamma_max=0.1)
        arm.load_state_dict(state)
        for p in arm.parameters():
            p.requires_grad_(False)
        if mode == "down":
            arm.down.weight.requires_grad_(True)
            trainable = arm.down.weight
        else:
            displacement = make_displacement(arm.down.weight, basis)
            parametrize.register_parametrization(arm.down, "weight", displacement)
            trainable = displacement.value
        return manager, arm, trainable

    results = {}
    tensors = {}
    for path in ("direct", "data_view", "data_clone", "detach_view"):
        manager, arm, trainable = setup()
        before = manager.model.memory.detach().clone()
        source = (manager.model.memory.detach() if path == "detach_view" else manager.model.memory.data)[:, 4:12, :].to("cpu")
        if path == "data_clone":
            source = source.clone()
        version_before = source._version
        candidate = arm(source, torch.ones(8), target=4, block=2)
        output = candidate.detach().clone()
        failure = None
        loss_value = None
        changed = False
        source_diff = 0.0
        version_injected = source._version
        try:
            if path == "direct":
                loss = candidate.float().square().sum()
                loss_value = float(loss.detach())
                loss.backward()
            else:
                with _pool_in_place(manager, "cpu", candidate, 8, 4) as injection:
                    pool = injection.reinject()
                    changed = not torch.equal(manager.model.memory.detach(), before)
                    source_diff = float((source.float() - before[:, 4:12].float()).norm())
                    version_injected = source._version
                    loss = pool[:, 8:12].float().square().sum()
                    loss_value = float(loss.detach())
                    loss.backward()
        except Exception as exc:
            failure = {"type": type(exc).__name__, "message": str(exc)}
        grad = None if trainable.grad is None else trainable.grad.detach().clone()
        tensors[path] = (output, grad)
        results[path] = {
            "loss": loss_value,
            "error": failure,
            "trainable_dof": trainable.numel(),
            "grad_norm": None if grad is None else float(grad.norm()),
            "memory_changed_during_injection": changed,
            "memory_restored_exact": torch.equal(manager.model.memory.detach(), before),
            "source_change_norm_during_injection": source_diff,
            "source_version_before": version_before,
            "source_version_during_injection": version_injected,
        }
    reference_output, reference_grad = tensors["direct"]
    for path, (output, grad) in tensors.items():
        item = results[path]
        item["forward_equal_direct"] = torch.equal(output, reference_output)
        item["forward_max_abs_diff"] = float((output.float() - reference_output.float()).abs().max())
        item["loss_equal_direct"] = item["loss"] == results["direct"]["loss"]
        item["grad_equal_direct"] = None if grad is None else torch.equal(grad, reference_grad)
        item["grad_difference_norm"] = None if grad is None else float((grad - reference_grad).norm())
        item["grad_difference_max_abs"] = None if grad is None else float((grad - reference_grad).abs().max())
        item["grad_difference_relative_norm"] = None if grad is None else float((grad - reference_grad).norm() / reference_grad.norm())
    return {"mode": mode, "memory_dtype": str(dtype), "paths": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    template = CarrierResidualCompactor(2, 8, block=2, rank=4, gamma_max=0.1)
    with torch.no_grad():
        # Represent a warm-up checkpoint; zero up would hide down's gradient.
        template.up.weight.normal_(0, 0.2)
        template.gamma_logit.fill_(-0.5)
    state = copy.deepcopy(template.state_dict())
    memory = torch.randn(2, 12, 8)
    basis = torch.linalg.qr(torch.randn(32, 13)).Q
    runs = [audit(mode, dtype, state, memory.to(dtype), basis)
            for dtype in (torch.float32, torch.float16)
            for mode in ("down", "alpha13")]
    report = {
        "status": "complete", "torch_version": torch.__version__,
        "device": "cpu", "threads": torch.get_num_threads(),
        "production_modules_used": ["CarrierResidualCompactor", "_pool_in_place", "make_displacement"],
        "protocol": {"layers": 2, "hidden": 8, "rank": 4, "block": 2,
                     "memory_slots": 12, "source_slice": [4, 12], "target_slice": [8, 12],
                     "warmup_up_nonzero_norm": float(template.up.weight.detach().norm()),
                     "loss": "candidate.float().square().sum() vs pool[:,8:12].float().square().sum()",
                     "parameters": "all frozen except down.weight or 13D displacement.value",
                     "notes": "Independent identical-state graph for each path; no optimizer/GPU/driver invocation."},
        "runs": runs,
        "verdict": {
            "all_forward_and_loss_equal": all(p["forward_equal_direct"] and p["loss_equal_direct"] for r in runs for p in r["paths"].values()),
            "all_data_view_gradients_differ": all(r["paths"]["data_view"]["grad_difference_norm"] > 0 for r in runs),
            "all_clone_gradients_exact": all(r["paths"]["data_clone"]["grad_equal_direct"] for r in runs),
            "all_memory_restored_exact": all(p["memory_restored_exact"] for r in runs for p in r["paths"].values())},
        "source_lines": {"memory_data": "methods/prlmc/layout.py:309-312",
                         "candidate_local": "tools/s1_local_protected.py:248-252",
                         "training_local": "tools/s1_local_protected.py:322-330",
                         "candidate_soft": "tools/s1_soft_local.py:182-186",
                         "training_soft": "tools/s1_soft_local.py:293-309",
                         "block_view": "methods/prlmc/consolidate.py:583-584",
                         "checkpoint": "methods/prlmc/consolidate.py:639-646",
                         "fp32_cast_inside": "methods/prlmc/consolidate.py:519-521",
                         "injection_write": "methods/prlmc/probes.py:821-826",
                         "injection_restore": "methods/prlmc/probes.py:873-912"}}
    out = Path(args.out)
    if out.exists():
        raise SystemExit("output exists; choose a new path")
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
