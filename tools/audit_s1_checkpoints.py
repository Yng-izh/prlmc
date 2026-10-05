"""Audit saved S1 compactor tensors on CPU without loading the reader."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(report_path, source_path):
    import torch
    torch.set_num_threads(4)
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    source_path = Path(source_path)
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == report["protocol"]["checkpoint_sha256"]
    payload = torch.load(source_path, map_location="cpu", weights_only=True)
    source = payload.get("state_dict", payload)
    basis = torch.load(report_path.with_suffix(".basis.pt"), map_location="cpu", weights_only=True)
    assert basis["lost"] == report["sets"]["lost"]
    assert basis["retained"] == report["sets"]["retained_full_and_select"]
    results = []
    for name in ("unconstrained", "coefficient_oracle", "subspace_ul", "protected_usafe"):
        path = report_path.with_suffix("." + name + ".pt")
        if not path.exists():
            assert report["status"] != "complete", "completed run has missing checkpoint"
            continue
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert all(torch.isfinite(value).all() for value in state.values())
        final = report["arms"][name]["final"]
        assert final["step"] == 60, "checkpoint is not a final-step artifact"
        prefix = "arm." if name == "coefficient_oracle" else ""
        for key in ("q.weight", "k.weight", "gamma_logit", "route_logit"):
            assert torch.equal(state[prefix + key], source[key]), (name, "frozen", key)
        if name == "coefficient_oracle":
            assert torch.equal(state["arm.down.weight"], source["down.weight"])
            torch.testing.assert_close(state["arm.up.weight"], source["up.weight"], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close((state["arm.basis_u"] @ state["arm.basis_r"]).T,
                                       source["up.weight"], atol=1e-6, rtol=1e-5)
            raw = state["oracle.residual"]
            assert torch.equal(raw, state["arm._s1_oracle.residual"])
            raw_norm = float(raw.norm())
            shrink = min(1.0, 0.5 / max(raw_norm, 1e-12))
            effective = raw * shrink
            norm = float(effective.double().norm())
            assert norm <= 0.5 + 1e-6
            assert abs(norm - final["deviation"]) <= 1e-5
            results.append({"arm": name, "frozen_parameters_verified": True,
                            "raw_coefficient_norm": raw_norm, "effective_coefficient_norm": norm})
            continue
        assert torch.equal(state["up.weight"], source["up.weight"])
        original = state["down.parametrizations.weight.original"]
        assert torch.equal(original, source["down.weight"])
        value = state["down.parametrizations.weight.0.value"]
        if name == "unconstrained":
            assert value.shape == original.shape
            delta = value.flatten().double()
        else:
            key = "ul" if name == "subspace_ul" else "safe"
            stored = state["down.parametrizations.weight.0.basis"]
            assert torch.equal(stored, basis[key].float())
            assert value.shape == (stored.shape[1],)
            delta = stored.double() @ value.double()
        norm = float(delta.norm())
        assert norm <= report["protocol"]["parameter_budget"] + 1e-5
        assert abs(float(value.norm()) - final["deviation"]) <= 1e-5
        item = {"arm": name, "frozen_parameters_verified": True,
                "actual_delta_norm": norm, "coordinate_norm": float(value.norm()),
                "trainable_dof": value.numel()}
        assert value.numel() == report["arms"][name]["trainable_dof"]
        if name == "protected_usafe":
            overlap = float((basis["uk"].T @ delta).norm())
            assert overlap <= 1e-5
            item["actual_delta_protection_overlap"] = overlap
        results.append(item)
    return {"status": report["status"], "saved_checkpoints_verified": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("source_checkpoint")
    args = parser.parse_args()
    print(json.dumps(audit(args.report, args.source_checkpoint), indent=2))


if __name__ == "__main__":
    main()
