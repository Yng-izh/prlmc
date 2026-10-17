"""Audit S1-soft saved tensors on CPU, without importing the training driver."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(report_path, source_path, source_basis, allow_smoke=False):
    import torch
    torch.set_num_threads(4)
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    protocol = report["protocol"]
    assert type(protocol["smoke"]) is bool
    assert not protocol["smoke"] or allow_smoke, "smoke requires --allow-smoke"
    expected_steps = 2 if protocol["smoke"] else 60
    assert protocol["steps"] == expected_steps
    assert protocol["measure_every"] == (1 if protocol["smoke"] else 5)
    assert sha256(source_path) == protocol["checkpoint_sha256"], "source checkpoint hash"
    assert sha256(source_basis) == protocol["basis_sha256"], "source basis hash"
    payload = torch.load(source_path, map_location="cpu", weights_only=True)
    source = payload.get("state_dict", payload)
    formal = torch.load(source_basis, map_location="cpu", weights_only=True)
    basis = torch.load(report_path.with_suffix(".basis.pt"), map_location="cpu", weights_only=True)
    assert basis["original_lost"] == report["sets"]["original_lost"] == formal["lost"]
    assert basis["source_basis_sha256"] == protocol["basis_sha256"]
    formal_ul = formal["ul"]
    assert formal_ul.ndim == 2 and formal_ul.shape[1] == 13
    assert torch.isfinite(formal_ul).all()
    assert basis["ul"].dtype == formal_ul.dtype and torch.equal(basis["ul"], formal_ul), "saved basis provenance"
    ul = formal_ul.float()
    torch.testing.assert_close(formal_ul.double().T @ formal_ul.double(), torch.eye(13, dtype=torch.float64), atol=1e-5, rtol=1e-5)
    for key, value in source.items():
        assert isinstance(value, torch.Tensor) and torch.isfinite(value).all(), ("source", key)
    assert ul.shape[0] == source["down.weight"].numel()
    original_key = "down.parametrizations.weight.original"
    basis_key = "down.parametrizations.weight.0.basis"
    value_key = "down.parametrizations.weight.0.value"
    expected_keys = (set(source) - {"down.weight"}) | {original_key, basis_key, value_key}
    results = []
    for name in ("ul", "soft_p", "soft_k"):
        path = report_path.with_suffix("." + name + ".pt")
        if not path.exists():
            assert report["status"] != "complete", (name, "missing final checkpoint")
            continue
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert set(state) == expected_keys, (name, "source key coverage")
        assert all(isinstance(value, torch.Tensor) and torch.isfinite(value).all() for value in state.values()), name
        final = report["arms"][name]["final"]
        assert final["step"] == expected_steps, "checkpoint is not a final-step artifact"
        for key in set(source) - {"down.weight"}:
            assert state[key].dtype == source[key].dtype and torch.equal(state[key], source[key]), (name, "frozen", key)
        original, stored, value = state[original_key], state[basis_key], state[value_key]
        assert original.dtype == source["down.weight"].dtype and torch.equal(original, source["down.weight"])
        assert stored.dtype == ul.dtype and torch.equal(stored, ul)
        assert value.shape == (13,) and report["arms"][name]["trainable_dof"] == 13
        delta = stored.double() @ value.double()
        norm = float(delta.norm())
        assert norm <= 1 + 1e-5 and protocol["parameter_budget"] == 1
        assert abs(float(value.double().norm()) - final["deviation"]) <= 1e-5
        assert abs(norm - final["deviation"]) <= 1e-5
        leakage = float((delta - formal_ul.double() @ (formal_ul.double().T @ delta)).norm())
        assert leakage <= 1e-5
        results.append({"arm": name, "frozen_parameters_verified": True,
                        "actual_delta_norm": norm, "trainable_dof": value.numel(),
                        "span_leakage": leakage, "checkpoint_sha256": sha256(path)})
    return {"status": report["status"], "source_basis_sha256": sha256(source_basis),
            "saved_checkpoints_verified": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("source_path")
    parser.add_argument("--source-basis", required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.report, args.source_path, args.source_basis, args.allow_smoke), indent=2))


if __name__ == "__main__":
    main()
