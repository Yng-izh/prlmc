"""Validate a repeated-compaction manifest without loading a model or GPU.

The manifest describes state IDs and slot counts produced by a host evaluator.
This tool checks that every compaction consumes the immediately previous state and
emits an ancestry record suitable for the frozen-checkpoint report.
"""

import argparse
import json
from pathlib import Path

from methods.prlmc.repeated import evaluate_horizon


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    writes = manifest.get("writes", [])
    if not writes:
        raise SystemExit("manifest.writes must contain at least one real post-write state")

    def compact_fn(state, spec):
        expected = spec.get("input_id")
        if expected is not None and state != expected:
            raise ValueError("compaction input is not the previous real output state")
        return str(spec["output_id"])

    report = evaluate_horizon(
        initial_state=str(manifest.get("initial_id", "initial")),
        initial_slots=int(manifest["initial_slots"]),
        writes=writes,
        compact_fn=compact_fn,
    )
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
