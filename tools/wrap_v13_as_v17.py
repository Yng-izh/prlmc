"""Re-package the V13 checkpoint as a `deep_fusion/1` checkpoint for no-training V17 arms.

Only the graft-attention constants differ from V13. The key channel and the band gate stay
off, so `DeepFusionResidual.forward` returns the parent's delta unchanged, and every weight
is V13's, copied bit for bit. With the damping constants neutralised (`--v17-no-attn`) the
arm must reproduce V13 row for row; that equality is the sanity check of this wrapper.

    python tools/wrap_v13_as_v17.py --v13 runs/cross_block/cross_lr.pt \
        --norm-quantile 0.9 --out runs/cross_block/v13_attn_q90.pt
"""
import argparse
import hashlib
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from methods.prlmc import deep_fusion as df  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v13", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--bias", type=float, default=0.0)
    p.add_argument("--norm-quantile", dest="norm_quantile", type=float, default=0.0)
    args = p.parse_args(argv)
    if args.tau == 1.0 and args.bias == 0.0 and args.norm_quantile == 0.0:
        raise SystemExit("all three damping constants are neutral: the wrapped checkpoint "
                         "would be V13 itself")
    for value in (args.tau, args.bias, args.norm_quantile):
        if not math.isfinite(value):
            raise SystemExit("damping constants must be finite")

    payload = torch.load(args.v13, map_location="cpu", weights_only=True)
    schema = str(payload.get("schema"))
    if not schema.endswith("/checkpoint") or schema.startswith(df.SCHEMA):
        raise SystemExit(f"{args.v13} is not a V13 calibration checkpoint ({schema!r})")
    state = payload["state_dict"]
    config = dict(payload["config"])
    config.setdefault("use_cross", any(str(k).startswith("cross_proj") for k in state))
    config.setdefault("use_evicted", any(str(k).startswith("evict_proj") for k in state))
    config.update(key_channel="off", band_mode="off", layer_ids=None,
                  graft_tau=float(args.tau), graft_bias=float(args.bias),
                  graft_norm_quantile=float(args.norm_quantile))
    module = df.DeepFusionResidual(**config)
    missing, unexpected = module.load_state_dict(state, strict=False)
    if unexpected or set(missing) != {"band_gate"}:
        raise SystemExit(f"V13 state does not map onto DeepFusionResidual: missing "
                         f"{missing}, unexpected {unexpected}")
    for name, tensor in state.items():
        if not torch.equal(module.state_dict()[name], tensor):
            raise SystemExit(f"{name} was not copied bit for bit")

    geometry = dict(payload.get("geometry") or {})
    geometry["graft_attention"] = dict(
        tau=float(args.tau), bias=float(args.bias),
        norm_quantile=float(args.norm_quantile), enabled=True,
        applied_in_training=False, site="eval-only, wrapped V13 weights")
    geometry["wrapped_from"] = dict(path=args.v13, sha256=sha256(args.v13),
                                    schema=schema)
    out = dict(schema=df.SCHEMA + "/checkpoint", config=module.config(),
               state_dict=module.state_dict(), geometry=geometry,
               note="V13 weights, key channel and band off; only graft-key damping "
                    "constants are non-neutral. No training was performed.")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out)
    reloaded, _ = df.load_deep(args.out, int(config["hidden_size"]))
    for name, tensor in state.items():
        if not torch.equal(reloaded.state_dict()[name], tensor):
            raise SystemExit(f"reload changed {name}")
    print(f"wrote {args.out}: tau={args.tau} bias={args.bias} "
          f"norm_quantile={args.norm_quantile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
