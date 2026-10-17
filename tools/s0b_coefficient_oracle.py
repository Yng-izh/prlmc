"""S0b — coefficient-oracle arm: C = I + E over the fixed basis.

Pre-registered question: do the rank^2 degrees of freedom inside the
coefficient space (a temporary transform C on the GELU coefficients) carry
enough capacity to recover lost probes *without* hurting retained ones?

Everything else is frozen: the warm-up compactor checkpoint (E6d interleave)
supplies the basis (U@R via the fixed-qr rewrite), routing, q/k, gamma; only
the (rank, rank) residual E of the oracle is trained. This is the go/no-go
gate before S1/S2: if the in-subspace freedom cannot even hold the
rescue/no-harm contract, functional control has no object to act on.

Memory discipline is inherited from the S0 driver (experiments/S0.md):
gradient checkpointing on the reader, per-probe backward via _pool_in_place
(the torch.cat pool's backward materialises 3.36 GiB and dies), train() mode
with all reader parameters frozen so the checkpointing branch is taken
(modeling_memoryllm.py:2001 requires self.training).
"""

import argparse
import copy
import json
import os
import sys

# GPU pick must happen before torch is imported anywhere.
S0B_NEED_MB = 21000


def _pick_gpu():
    if os.environ.get("S0B_GPU_SET"):
        return
    try:
        with os.popen("nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits") as f:
            rows = [line.strip().split(",") for line in f if line.strip()]
    except OSError:
        return
    for index, free_mb in rows:
        if int(free_mb) >= S0B_NEED_MB:
            os.environ["CUDA_VISIBLE_DEVICES"] = index
            os.environ["S0B_GPU_SET"] = "1"
            print("[s0b] gpu%s, %s MB free" % (index, free_mb), flush=True)
            os.execv(sys.executable, [sys.executable] + sys.argv)
    raise SystemExit("no GPU with >= %d MB free" % S0B_NEED_MB)


_pick_gpu()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import e6a_overfit as E  # noqa: E402
from methods.prlmc.config import PRLMCConfig  # noqa: E402
from methods.prlmc.consolidate import (  # noqa: E402
    CarrierResidualCompactor,
    FixedBasisCarrierResidualCompactor,
)
from methods.prlmc.functional_subspace import CoefficientOracle  # noqa: E402
from methods.prlmc.probes import (  # noqa: E402
    _compacted_mask,
    _pool_in_place,
    active_mask_override,
    differentiable_memory,
)
from s0_functional_sensitivity import (  # noqa: E402
    build_window_and_probes,
    margin_scalar,
)

MODEL = "/share/latent/MemoryLLM-8B"
CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "agent_conf", "prlmc_e6.yaml")


class OracleCompactor(torch.nn.Module):
    """Fixed-basis compactor with C = I + E on the GELU coefficients.

    Wraps a frozen FixedBasisCarrierResidualCompactor; the only trainable
    tensor is the oracle residual E. The forward reproduces the parent's
    `_layer_delta` with one insertion:

        coefficients = gelu(down(context - carrier))
        coefficients = coefficients @ C^T          <-- the S0b arm
        delta = linear(coefficients, basis)

    E is zero-initialised, so step 0 is bit-identical to the checkpoint -- the
    same identity-at-init contract the carrier line was built on.
    """

    def __init__(self, arm: FixedBasisCarrierResidualCompactor, oracle: CoefficientOracle):
        super().__init__()
        self.arm = arm
        self.oracle = oracle

    def forward(self, source, weights=None, target=None, block=None,
                goal_direction=None, select_weights=None):
        """Mirror the parent forward with the coefficient transform inserted.

        The parent's per-layer loop with its own checkpointing is reused by
        calling the parent `_layer_delta`'s pieces: routing, carriers and the
        trust region are identical. Only `_layer_delta` is overridden below.
        """
        return self.arm.forward(source, weights, target, block,
                                goal_direction, select_weights)

    def _layer_delta(self, attn, block_layer, carrier_layer):
        carrier = carrier_layer.to(torch.float32)
        context = torch.einsum("jb,btd->jtd", attn, block_layer.to(torch.float32))
        coefficients = F.gelu(self.arm.down(context - carrier))
        coefficients = self.oracle.apply(coefficients)
        delta = F.linear(coefficients, (self.arm.basis_u @ self.arm.basis_r).transpose(0, 1))
        carrier_rms = carrier.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt()
        delta_var = delta.pow(2).mean(dim=(-2, -1), keepdim=True)
        return delta * torch.rsqrt(delta_var + 1e-6) * carrier_rms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", default="3,17890")
    parser.add_argument("--probes", type=int, default=60)
    parser.add_argument("--max-probe-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--carrier-ckpt", required=True,
                        help="warm-up trained compactor checkpoint (S0 base point)")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--max-deviation", type=float, default=0.5)
    parser.add_argument("--out", default="/tmp/s0b_oracle.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    raw = yaml.safe_load(open(CONFIG))
    config = dict(raw["prlmc"])
    config["device"] = "cuda:0"
    config["dtype"] = "float16"
    config["stats_dir"] = "./outputs/prlmc-s0b"
    cfg = PRLMCConfig.from_dict(config)

    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
    from transformers import AutoTokenizer

    model = MemoryLLM.from_pretrained(MODEL, torch_dtype=torch.float16, attn_implementation="sdpa")
    model = model.to("cuda:0").eval()
    model.initialized.zero_()
    # Checkpointing + frozen reader, identical rationale to the S0 driver.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    from methods.prlmc.layout import ProtectedResidualMemory
    manager = ProtectedResidualMemory(model, tokenizer, cfg)

    row, start = [int(v) for v in args.window.split(",")]
    rows = {row: E.read_fact_lines(row)}
    built = build_window_and_probes(manager, rows, row, start, args)
    if built is None:
        raise SystemExit("window build failed")
    covered, indices, dropped, probes = built

    # Fixed-basis arm from the warm-up checkpoint: QR basis, frozen everything.
    warm = CarrierResidualCompactor(
        num_layers=manager.num_layers, hidden=manager.hidden, block=manager.block,
        rank=int(cfg.carrier_rank), gamma_max=float(cfg.carrier_gamma_max),
    ).to(device="cuda:0", dtype=torch.float32)
    state = torch.load(args.carrier_ckpt, map_location="cuda:0", weights_only=True)
    warm.load_state_dict(state["state_dict"])
    arm = FixedBasisCarrierResidualCompactor.from_warmup(warm)
    arm = arm.to("cuda:0")
    for parameter in arm.parameters():
        parameter.requires_grad_(False)

    oracle = CoefficientOracle(rank=int(cfg.carrier_rank),
                               max_deviation=args.max_deviation).to("cuda:0")
    wrapper = OracleCompactor(arm, oracle)
    # Route the parent's internal _layer_delta calls through the wrapper's
    # override: bind by monkey-patch so the parent's checkpointing loop calls
    # the C-transformed version.
    arm._layer_delta = wrapper._layer_delta

    optimizer = torch.optim.AdamW([oracle.residual], lr=args.lr)

    # Baselines: full pool and the checkpoint's pool (the S0 base point).
    no_mem = E.score_no_memory(manager, probes, args.max_probe_tokens)
    full = E.score_uncompacted(manager, probes, args.max_probe_tokens)
    pool0, mask0 = E.candidate_pool(manager, arm, "cuda:0")
    before = E.score_probes(manager, probes, pool0, args.max_probe_tokens, mask=mask0)
    del pool0, mask0
    torch.cuda.empty_cache()
    gap = E.recovery_report(full, before, before, no_memory=no_mem)
    lost_questions = list(gap["lost_questions"])
    retained_questions = [
        p.question for p in probes
        if p.question not in set(lost_questions)
        and before["correct"].get(p.question, False)
    ]
    print("[s0b] eligible %d, lost %d, retained %d"
          % (gap["n_eligible"], len(lost_questions), len(retained_questions)), flush=True)
    if len(lost_questions) < 1 or len(retained_questions) < 1:
        raise SystemExit("need at least one lost and one retained probe")

    window_used = E.WINDOW_BLOCKS * manager.block
    target_used = E.TARGET_BLOCKS * manager.block
    mask = _compacted_mask(manager, manager.used - window_used + target_used, "cuda:0")

    # The training objective, per pre-registration:
    #   sum of lost probes' margins (rescue) minus a penalty weight times the
    #   number of retained probes that flip to wrong (no-harm contract).
    # The flip penalty is non-differentiable, so the loss is the margin sum
    # and the no-harm contract is enforced by measuring after every step.
    results = {
        "window": {"row": row, "start": start},
        "steps": args.steps, "lr": args.lr,
        "max_deviation": args.max_deviation,
        "n_lost": len(lost_questions),
        "n_retained": len(retained_questions),
        "lost_questions": lost_questions,
        "retained_questions": retained_questions,
        "history": [],
    }

    probe_tensors = {}
    for q in lost_questions + retained_questions:
        probe = next(p for p in probes if p.question == q)
        probe_tensors[q] = probe

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        # Fresh candidate each step: the oracle moves, the graph must follow.
        source = manager._memory[
            :, manager.total - manager.used : manager.total, :
        ].to("cuda:0")
        weights = manager._window_importance(
            manager.total - manager.used, window_used
        )
        candidate = wrapper(
            source, weights, target=target_used, block=manager.block,
            goal_direction=manager.goal_direction,
            select_weights=manager.select_weights,
        )
        del source, weights
        margin_sum = None
        with _pool_in_place(manager, "cuda:0", candidate, window_used, target_used) as injection:
            pool = injection.reinject()
            for q in lost_questions:
                scalar = margin_scalar(manager, probe_tensors[q], pool, mask,
                                       args.max_probe_tokens)
                if scalar is None:
                    continue
                immediate = scalar.backward(retain_graph=True)
                del scalar
        if oracle.residual.grad is None or oracle.residual.grad.abs().sum() == 0:
            print("[s0b] step %d: zero gradient, aborting" % step, flush=True)
            break
        optimizer.step()
        del candidate, pool
        torch.cuda.empty_cache()

        # Post-step measurement: rescue and harm against the frozen base point.
        if step % max(1, args.steps // 12) == 0 or step == args.steps:
            pool_m, mask_m = E.candidate_pool(manager, wrapper, "cuda:0")
            after = E.score_probes(manager, probes, pool_m, args.max_probe_tokens, mask=mask_m)
            del pool_m, mask_m
            torch.cuda.empty_cache()
            rescued = sum(1 for q in lost_questions if after["correct"].get(q, False))
            harmed = sum(1 for q in retained_questions if not after["correct"].get(q, False))
            deviation = float((oracle.matrix() - torch.eye(oracle.rank, device="cuda:0")).norm())
            results["history"].append({
                "step": step, "rescued": rescued, "harmed": harmed,
                "deviation": round(deviation, 4),
            })
            print("[s0b] step %d: rescued %d/%d, harmed %d/%d, |E| %.4f"
                  % (step, rescued, len(lost_questions), harmed, len(retained_questions),
                     deviation), flush=True)

    # Final verdict per pre-registration: rescued >= ceil(0.4 * n_lost) and
    # harmed <= floor(0.1 * n_retained) -> GO; rescued == 0 -> NO-GO.
    final = results["history"][-1] if results["history"] else {"rescued": 0, "harmed": 0}
    import math
    go_rescue = math.ceil(0.4 * len(lost_questions))
    go_harm = math.floor(0.1 * len(retained_questions))
    if final["rescued"] >= go_rescue and final["harmed"] <= go_harm:
        verdict = "GO"
    elif final["rescued"] == 0:
        verdict = "NO-GO"
    else:
        verdict = "INCONCLUSIVE"
    results["verdict"] = verdict
    results["final"] = final
    print("[s0b] verdict %s (rescued %d/%d, harmed %d/%d, thresholds %d/%d)"
          % (verdict, final["rescued"], len(lost_questions), final["harmed"],
             len(retained_questions), go_rescue, go_harm), flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("[s0b] wrote %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
