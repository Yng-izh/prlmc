"""E7 Stage A: measure gradient and functional interference between windows.

This is measurement only.  No new memory module or routing mechanism is introduced.
For the fixed E6c eight-window fixture, the script records:

* one Stage-B probe-loss gradient per window at the shared initial compactor;
* the 8x8 cosine matrix between those gradients;
* for each source window i, the effect of one identical Stage-B update on every
  target window j, including rescued/harmed probe counts and per-question flips.

Every source arm starts from the same initial state.  The gradient arm uses a
capturing optimizer, so its parameters are never updated.  The functional arm
uses one real AdamW step and then scores all windows from that isolated state.

The script has a normal GPU auto-pick guard, matching E6d/E6e.  It is intended
for the host environment; py_compile is sufficient for local static checking.
"""

import argparse
import copy
import hashlib
import json
import os
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = "/share/latent/PRLM-C"
for path in ("/tmp", TOOLS, REPO):
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


if __name__ == "__main__" and not os.environ.get("E7A_GPU_SET") and not os.environ.get("E7A_NO_GPU"):
    import subprocess

    need_mb = int(os.environ.get("E7A_NEED_MB", "26000"))
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    free = []
    for line in output.strip().splitlines():
        index, used = [int(v) for v in line.split(",")]
        free.append((32768 - used, index))
    free.sort(reverse=True)
    if not free or free[0][0] < need_mb:
        raise SystemExit("no GPU with %d MB free" % need_mb)
    print("[e7a] gpu%d, %d MB free" % (free[0][1], free[0][0]), flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(free[0][1])
    os.environ["E7A_GPU_SET"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import torch  # noqa: E402

DEVICE = "cuda:0"


class CaptureAdamW(torch.optim.AdamW):
    """Capture the gradient passed to step without changing any parameter."""

    def __init__(self, named_parameters, *args, **kwargs):
        self._named_parameters = list(named_parameters)
        self.last_gradient = None
        super().__init__([p for _, p in self._named_parameters], *args, **kwargs)

    def step(self, closure=None):
        self.last_gradient = {
            name: (parameter.grad.detach().float().cpu().clone()
                   if parameter.grad is not None else None)
            for name, parameter in self._named_parameters
        }
        return None


def state_md5(state):
    digest = hashlib.md5()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def gradient_vector(named_parameters, captured):
    """Flatten gradients in parameter order, using explicit zeros for None."""
    pieces = []
    for name, parameter in named_parameters:
        value = captured.get(name)
        if value is None:
            value = torch.zeros_like(parameter.detach(), dtype=torch.float32, device="cpu")
        pieces.append(value.reshape(-1))
    if not pieces:
        raise RuntimeError("compactor has no parameters")
    vector = torch.cat(pieces)
    if not torch.isfinite(vector).all():
        raise RuntimeError("non-finite gradient detected")
    if float(vector.norm()) == 0.0:
        raise RuntimeError("zero gradient detected")
    return vector


def cosine_matrix(vectors):
    stacked = torch.stack(vectors).float()
    normalized = stacked / stacked.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (normalized @ normalized.T).tolist()


def key_for(entry):
    return "%d:%d" % (entry["row"], entry["start"])


def correct_map(scored):
    return {str(q): bool(v) for q, v in scored["correct"].items()}


def functional_effect(before, after):
    before_map = correct_map(before)
    after_map = correct_map(after)
    questions = sorted(set(before_map) | set(after_map))
    rescued = [q for q in questions if not before_map.get(q, False) and after_map.get(q, False)]
    harmed = [q for q in questions if before_map.get(q, False) and not after_map.get(q, False)]
    return {
        "before_n_correct": sum(before_map.values()),
        "after_n_correct": sum(after_map.values()),
        "delta_n_correct": len(rescued) - len(harmed),
        "rescued": rescued,
        "harmed": harmed,
        "per_question": {
            q: {"before": before_map.get(q, False), "after": after_map.get(q, False)}
            for q in questions
        },
    }


def make_compactor(manager, cfg, initial_state, B):
    from methods.prlmc.consolidate import CarrierResidualCompactor

    module = CarrierResidualCompactor(
        num_layers=manager.num_layers,
        hidden=manager.hidden,
        block=manager.block,
        rank=int(cfg.carrier_rank),
        gamma_max=float(cfg.carrier_gamma_max),
    ).to(device=DEVICE, dtype=torch.float32)
    if initial_state is not None:
        module.load_state_dict(copy.deepcopy(initial_state))
        return B.before_compactor_for(module, initial_state)
    return module


def score_all(manager, cache, entries, compactor, args, B):
    scored = {}
    for entry in entries:
        key = key_for(entry)
        result = B.evaluate_window(manager, cache[entry["row"]], entry, compactor, args)
        scored[key] = {
            "correct": correct_map(result["after"]),
            "signature": result["after_signature"],
        }
        B.release()
    return scored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gradient-steps", type=int, default=1,
                        help="Stage-B steps for the gradient arm; the capturing "
                             "optimizer never updates parameters, so every step "
                             "recomputes the same initial-point gradient -- keep at 1")
    parser.add_argument("--functional-steps", type=int, default=1,
                        help="Stage-B steps per source window for the functional arm; "
                             "24 is one quarter of E6d's 96-step per-window budget, "
                             "with the same curriculum cadence (4 vs 16 steps per level)")
    parser.add_argument("--probes", type=int, default=20)
    parser.add_argument("--max-probe-tokens", type=int, default=256)
    parser.add_argument("--min-probes", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma-max", type=float, default=0.05)
    parser.add_argument("--out", default="/tmp/e7a_interference_seed0.json")
    args = parser.parse_args()
    if args.gradient_steps < 1 or args.functional_steps < 1:
        raise SystemExit("--gradient-steps and --functional-steps must be positive")

    import random

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        import numpy as np

        np.random.seed(args.seed)
    except ImportError:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    import yaml
    import e6a_overfit as E
    import e6b_mini as B
    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
    from methods.prlmc.config import PRLMCConfig
    from methods.prlmc.layout import ProtectedResidualMemory
    from transformers import AutoTokenizer

    raw = yaml.safe_load(open(E.CONFIG))
    config = dict(raw["prlmc"])
    config["device"] = DEVICE
    config["dtype"] = "float16"
    config["stats_dir"] = "./outputs/prlmc-e7a"
    config["carrier_gamma_max"] = args.gamma_max
    cfg = PRLMCConfig.from_dict(config)

    model = MemoryLLM.from_pretrained(E.MODEL, torch_dtype=torch.float16, attn_implementation="sdpa")
    model = model.to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(E.MODEL)
    manager = ProtectedResidualMemory(model, tokenizer, cfg)

    from e6d_schedule import DEFAULT_WINDOWS

    template = make_compactor(manager, cfg, None, B)
    initial_state = copy.deepcopy(template.state_dict())
    # make_compactor accepts the initial state for a clean clone; the first template
    # is created directly because there is no state before the initial snapshot.
    template = template
    init_md5 = state_md5(initial_state)
    windows = list(DEFAULT_WINDOWS)
    cache = {row: E.read_fact_lines(row) for row in sorted({r for r, _ in windows})}

    entries = []
    for row, start in windows:
        entry = B.measure_window(
            manager, cache[row], row, start, template, initial_state,
            args.probes, args, log_prefix="[e7a] fixture",
        )
        if entry is None:
            raise SystemExit("[e7a] fixture window %d:%d unusable" % (row, start))
        entries.append(entry)
        print("[e7a] fixture %s probes %d eligible %d" % (
            key_for(entry), entry["n_probes"], entry["n_eligible"]), flush=True)
    B.release()

    # The same B-step loss is used for every row. The capturing optimizer makes
    # this pass a pure gradient measurement and leaves the compactor unchanged.
    named = list(template.named_parameters())
    gradients = []
    gradient_norms = {}
    for index, entry in enumerate(entries):
        compactor = make_compactor(manager, cfg, initial_state, B)
        capture = CaptureAdamW(compactor.named_parameters(), lr=args.lr)
        B.train_window(manager, cache[entry["row"]], entry, compactor, capture,
                       "B", args.gradient_steps, args)
        vector = gradient_vector(named, capture.last_gradient or {})
        gradients.append(vector)
        gradient_norms[key_for(entry)] = float(vector.norm())
        print("[e7a] gradient %d/%d %s norm %.6e" % (
            index + 1, len(entries), key_for(entry), gradient_norms[key_for(entry)]), flush=True)
        del compactor, capture
        B.release()

    gradient_cos = cosine_matrix(gradients)
    labels = [key_for(entry) for entry in entries]

    functional = {}
    delta_matrix = []
    rescued_matrix = []
    harmed_matrix = []
    for source_index, source_entry in enumerate(entries):
        source = key_for(source_entry)
        compactor = make_compactor(manager, cfg, initial_state, B)
        optimiser = torch.optim.AdamW(compactor.parameters(), lr=args.lr)
        B.train_window(manager, cache[source_entry["row"]], source_entry,
                       compactor, optimiser, "B", args.functional_steps, args)
        after = score_all(manager, cache, entries, compactor, args, B)
        source_effects = {}
        delta_row, rescued_row, harmed_row = [], [], []
        for target_entry in entries:
            target = key_for(target_entry)
            effect = functional_effect(target_entry["before"], after[target])
            source_effects[target] = effect
            delta_row.append(effect["delta_n_correct"])
            rescued_row.append(len(effect["rescued"]))
            harmed_row.append(len(effect["harmed"]))
        functional[source] = {
            "source_window": source,
            "update_steps": args.functional_steps,
            "effects": source_effects,
        }
        delta_matrix.append(delta_row)
        rescued_matrix.append(rescued_row)
        harmed_matrix.append(harmed_row)
        print("[e7a] functional %d/%d source %s delta %s" % (
            source_index + 1, len(entries), source, delta_row), flush=True)
        del compactor, optimiser, after
        B.release()

    report = {
        "design": {
            "name": "E7 Stage A functional interference graph",
            "question": "can the eight training windows be separated by function?",
            "seed": args.seed,
            "fixture": "E6c/e6d DEFAULT_WINDOWS",
            "windows": labels,
            "gradient_loss": "existing Stage-B train_carrier_compactor objective",
            "gradient_update_steps": args.gradient_steps,
            "functional_update_steps": args.functional_steps,
            "stage_b_curriculum": "existing methods.prlmc.probes._CARRIER_CURRICULUM",
            "window_blocks": E.WINDOW_BLOCKS,
            "lr": args.lr,
            "gamma_max": args.gamma_max,
            "probes": args.probes,
            "max_probe_tokens": args.max_probe_tokens,
            "initial_state_md5": init_md5,
            "structure_changed": False,
            "loss_changed": False,
        },
        "windows": labels,
        "baseline": {
            key_for(entry): {
                "n_probes": entry["n_probes"],
                "n_eligible": entry["n_eligible"],
                "before_n_correct": sum(bool(v) for v in entry["before"]["correct"].values()),
                "before_signature": B.correct_signature(entry["before"]),
            }
            for entry in entries
        },
        "gradient_norms": gradient_norms,
        "gradient_cosine": {
            "labels": labels,
            "matrix": gradient_cos,
        },
        "functional_interference": {
            "delta_n_correct": {"labels": labels, "matrix": delta_matrix},
            "rescued": {"labels": labels, "matrix": rescued_matrix},
            "harmed": {"labels": labels, "matrix": harmed_matrix},
            "by_source": functional,
        },
    }
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print("[e7a] report %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
