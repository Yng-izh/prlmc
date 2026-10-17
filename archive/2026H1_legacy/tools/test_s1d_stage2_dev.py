#!/usr/bin/env python
"""Unit tests for s1d_stage2_dev metric semantics (CPU-only)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from s1d_stage2_dev import exact_match, normalize, stage2_metric, window_sets  # noqa: E402


def make_sets():
    # 10 questions: full-correct universe
    full = {q: True for q in ["q%d" % i for i in range(10)]}
    select = {q: (i < 4) for i, q in enumerate(full)}     # q0-3 retained
    d0 = {q: (i < 2 or i == 6) for i, q in enumerate(full)}  # q0,q1,q6 correct
    return window_sets(full, select, d0), full, select, d0


def test_stage2_metric_new_recovery_excludes_d0_correct():
    sets, full, select, d0 = make_sets()
    # arm gets q2,q3,q4,q5,q6,q7 right; of the d0-wrong rescue set
    # (q4,q5,q7,q8,q9) it recovers q4,q5,q7 -> 3; q6 is d0-right so it
    # never counts toward new recovery
    arm = {q: (i in (2, 3, 4, 5, 6, 7)) for i, q in enumerate(full)}
    m = stage2_metric(arm, sets, d0)
    assert m["new_recovery"] == 3, m["new_recovery"]
    assert m["new_recovery_questions"] == ["q4", "q5", "q7"]
    # S1c-style count would be 4 (q4,q5,q6,q7 all in rescue and correct):
    # q6 inflates it without being new recovery
    assert m["rescue_recovered_s1c_style"] == 4


def test_stage2_metric_d0_harmed_counts_any_d0_correct_loss():
    sets, full, select, d0 = make_sets()
    # arm loses q0 (retained, d0-right) and q6 (rescue, d0-right) while
    # recovering all five d0-wrong rescue questions
    arm = {q: (i not in (0, 6)) for i, q in enumerate(full)}
    m = stage2_metric(arm, sets, d0)
    assert m["d0_harmed"] == 2
    assert m["d0_harmed_questions"] == ["q0", "q6"]
    assert m["new_recovery"] == 5
    assert m["net_vs_d0"] == 5 - 2


def test_stage2_metric_identity_arm_is_all_zero():
    sets, full, select, d0 = make_sets()
    m = stage2_metric(d0, sets, d0)
    assert m["new_recovery"] == 0 and m["d0_harmed"] == 0
    assert m["net_vs_d0"] == 0


def test_normalize_exact_match():
    assert normalize("The  Beatles!") == "beatles"
    assert exact_match("the beatles.", ["The Beatles"]) is True
    assert exact_match("A Queen band", ["queen"]) is False  # extra word
    assert exact_match("An Apple", ["apple"]) is True


def test_window_sets_partitions():
    sets, full, select, d0 = make_sets()
    assert sets["retained"] == ["q0", "q1", "q2", "q3"]
    assert sets["rescue"] == ["q%d" % i for i in range(4, 10)]
    assert sets["protect"] == ["q0", "q1"]
    assert sets["repair"] == ["q2", "q3"]
    assert sets["rescue_d0_correct"] == ["q6"]
    assert sets["rescue_d0_wrong"] == ["q4", "q5", "q7", "q8", "q9"]


def test_reader_evaluation_restores_state():
    import torch
    from s1d_stage2_dev import reader_evaluation
    model = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(2, 2))
    model.memory = torch.nn.Parameter(torch.arange(8.).reshape(2, 2, 2), requires_grad=False)
    model.register_buffer("memory_active_mask", torch.ones(2, dtype=torch.bool))
    model.train()
    model[0].eval()
    modes = [m.training for m in model.modules()]
    rng = torch.get_rng_state().clone()
    memory = model.memory.detach().clone()
    with reader_evaluation(model, verify_memory=True):
        assert not any(m.training for m in model.modules())
        torch.rand(5)
        with torch.no_grad():
            model.memory.add_(1)
    assert [m.training for m in model.modules()] == modes
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(model.memory, memory)
    try:
        with reader_evaluation(model, verify_memory=True):
            torch.rand(5)
            raise ValueError("test exception")
    except ValueError:
        pass
    assert [m.training for m in model.modules()] == modes
    assert torch.equal(torch.get_rng_state(), rng)


def test_oracle_answer_ce_masks_prompt():
    import torch
    import torch.nn.functional as F
    tokens = torch.tensor([[0, 1, 2, 3, 4]])
    stem_len = 3
    logits = torch.randn(1, 5, 7, requires_grad=True)
    labels = tokens.clone()
    labels[:, :stem_len] = -100
    masked = F.cross_entropy(logits[:, :-1].reshape(-1, 7),
                             labels[:, 1:].reshape(-1))
    expected = F.cross_entropy(logits[:, stem_len - 1:-1].reshape(-1, 7),
                               tokens[:, stem_len:].reshape(-1))
    assert torch.equal(masked, expected)
    masked.backward()
    assert torch.count_nonzero(logits.grad[:, :stem_len - 1]) == 0
    assert torch.count_nonzero(logits.grad[:, stem_len - 1:-1]) > 0
    assert torch.count_nonzero(logits.grad[:, -1]) == 0


def test_protection_penalty_is_hinge_over_d0_ce():
    import torch
    baseline = torch.tensor([2.0, 3.0])
    current = torch.tensor([2.5, 2.0], requires_grad=True)
    penalty = torch.relu(current - baseline).mean()
    assert torch.equal(penalty.detach(), torch.tensor(0.25))
    penalty.backward()
    assert torch.equal(current.grad, torch.tensor([0.5, 0.0]))

def test_offloaded_checkpoint_gradient_accumulation():
    import torch
    from torch.utils.checkpoint import checkpoint
    from methods.prlmc.probes import differentiable_memory
    model = torch.nn.Module()
    model.memory = torch.nn.Parameter(torch.zeros(3), requires_grad=False)
    original = model.memory
    basis = torch.tensor([[1., 2.], [3., -1.], [2., 4.]])
    alpha = torch.tensor([0.2, -0.1], requires_grad=True)
    inputs = [torch.tensor([1., 2., 3.]), torch.tensor([3., 1., 2.])]
    expected = sum(((basis @ alpha) * x).square().mean() for x in inputs) / len(inputs)
    reference = torch.autograd.grad(expected, alpha)[0]
    for x in inputs:
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            pool = basis @ alpha
            with differentiable_memory(model, pool):
                loss = checkpoint(lambda v: (model.memory * v).square().mean(),
                                  x, use_reentrant=False) / len(inputs)
                loss.backward()
        assert model.memory is original
    assert torch.allclose(alpha.grad, reference), (alpha.grad, reference)


def test_protected_training_pool_gradients_and_restoration():
    import ast
    import types
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    from methods.prlmc.probes import (
        _pool_in_place, active_mask_override, differentiable_memory)

    class Reader(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.memory = torch.nn.Parameter(
                torch.arange(12.).reshape(1, 6, 2) / 20, requires_grad=False)
            self.register_buffer("memory_active_mask", torch.ones(6, dtype=torch.bool))
            self.projection = torch.arange(20.).reshape(4, 5) / 30

        def logits(self, ids):
            assert not self.memory_active_mask[:4].any()
            assert self.memory_active_mask[-2:].all()
            score = self.memory[:, -2:].reshape(4) @ self.projection
            return score.expand(ids.shape[0], ids.shape[1], 5)

        def forward(self, input_ids, **kwargs):
            # Access the binding during recomputation, not only during forward.
            value = checkpoint(self.logits, input_ids, use_reentrant=False)
            return types.SimpleNamespace(logits=value)

    model = Reader()
    original_parameter = model.memory
    original_memory = model.memory.detach().clone()
    original_mask = model.memory_active_mask
    mask = torch.tensor([False, False, False, False, True, True])
    manager = types.SimpleNamespace(model=model, total=6, used=4)
    alpha = torch.nn.Parameter(torch.tensor([0.2, -0.1]))
    basis = torch.tensor([[1., 2.], [3., -1.], [2., 4.], [-1., 2.]])
    basis_before = basis.clone()
    d0 = torch.tensor([0.1, 0.3, -0.2, 0.4])
    encoded = {q: (torch.tensor([[0, answer]]), 1)
               for q, answer in (("r1", 1), ("r2", 3), ("p1", 2), ("p2", 4))}
    refs = {"p1": 0.0, "p2": 100.0}
    optimizer = torch.optim.SGD([alpha], lr=0.0)
    assert optimizer.param_groups[0]["params"] == [alpha]

    score = (d0 + basis @ alpha) @ model.projection
    ce = {q: F.cross_entropy(score.unsqueeze(0), ids[:, 1])
          for q, (ids, _) in encoded.items()}
    expected = (ce["r1"] + ce["r2"]) / 2
    expected += (torch.relu(ce["p1"] - refs["p1"]) +
                 torch.relu(ce["p2"] - refs["p2"])) / 2
    reference = torch.autograd.grad(expected, alpha)[0]

    # Compile the actual nested driver function, with a tiny reader replacing 8B.
    source = (Path(__file__).parent / "s1d_stage2_dev.py").read_text()
    node = next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.FunctionDef) and n.name == "train_answer_arm")
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and child.value == "cuda:0":
            child.value = "cpu"
    from s1d_stage2_dev import QAParallel
    namespace = dict(torch=torch, F=F, model=model, manager=manager,
                     parallel=QAParallel(),
                     candidate=lambda arm: (d0 + basis @ alpha).reshape(1, 2, 2),
                     _pool_in_place=_pool_in_place,
                     differentiable_memory=differentiable_memory,
                     active_mask_override=active_mask_override,
                     window_used=4, target_used=2,
                     args=types.SimpleNamespace(budget=1.0))
    manager.tokenizer = types.SimpleNamespace(eos_token_id=0)
    exec(compile(ast.Module(body=[node], type_ignores=[]), source, "exec"), namespace)
    result = namespace["train_answer_arm"](
        None, types.SimpleNamespace(value=alpha, clamp=lambda budget: None), optimizer,
        encoded, ["r1", "r2"], mask, protect_questions=["p1", "p2"],
        protect_baseline=refs, protect_weight=1.0, mean_grad=True)
    assert torch.allclose(alpha.grad, reference, atol=1e-6), (alpha.grad, reference)
    assert abs(result["loss"] - expected.item()) < 1e-6
    assert torch.equal(basis, basis_before)
    assert model.memory is original_parameter
    assert torch.equal(model.memory, original_memory)
    assert model.memory_active_mask is original_mask

    try:
        with _pool_in_place(manager, "cpu", torch.ones(1, 2, 2, requires_grad=True), 4, 2) as injection:
            pool = injection.reinject()
            with differentiable_memory(model, pool), active_mask_override(model, mask):
                model(input_ids=encoded["r1"][0]).logits.square().sum().backward()
                raise ValueError("restore after backward failure")
    except ValueError:
        pass
    assert model.memory is original_parameter
    assert torch.equal(model.memory, original_memory)
    assert model.memory_active_mask is original_mask


def _distributed_gradient_worker(rank, world_size, init_file):
    import torch
    import torch.nn.functional as F
    from s1d_stage2_dev import QAParallel
    torch.distributed.init_process_group(
        "gloo", init_method="file://" + init_file, rank=rank, world_size=world_size)
    try:
        parallel = QAParallel(rank, world_size)
        alpha = torch.nn.Parameter(torch.tensor([0.2, -0.1]))
        vectors = [torch.tensor([1., 2.]), torch.tensor([-2., 1.]),
                   torch.tensor([3., -1.])]

        def ce(index):
            z = vectors[index] @ alpha
            return F.cross_entropy(torch.stack([z, -z, z * 0]).unsqueeze(0),
                                   torch.tensor([index]))

        rescue = [0, 1, 2]
        protect = [0, 1]
        refs = {0: 0.0, 1: 100.0}
        expected = sum(ce(i) for i in rescue) / len(rescue)
        expected += sum(torch.relu(ce(i) - refs[i]) for i in protect) / len(protect)
        reference = torch.autograd.grad(expected, alpha)[0]
        totals = [0.0, 0.0]
        for i in parallel.shard(rescue):
            loss = ce(i)
            (loss / len(rescue)).backward()
            totals[0] += loss.item()
        for i in parallel.shard(protect):
            penalty = torch.relu(ce(i) - refs[i])
            (penalty / len(protect)).backward()
            totals[1] += penalty.item()
        sums = parallel.reduce_step([alpha], totals)
        assert torch.allclose(alpha.grad, reference, atol=1e-6)
        assert abs(sums[0] / len(rescue) + sums[1] / len(protect) - expected.item()) < 1e-6
        torch.optim.SGD([alpha], lr=0.03).step()
        parallel.assert_same(alpha.detach().tolist(), "test alpha")
        merged = parallel.merge({i: i * 2 for i in parallel.shard(rescue)}, rescue)
        assert merged == {0: 0, 1: 2, 2: 4}
    finally:
        torch.distributed.destroy_process_group()


def test_distributed_gradient_equivalence():
    import tempfile
    import torch.multiprocessing as mp
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_distributed_gradient_worker, args=(4, directory + "/gloo"),
                 nprocs=4, join=True)


def test_parallel_launcher_report_certification():
    import json
    import tempfile
    from run_s1d_parallel import finalize_report
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "report.json"
        for status, code, expected in (("complete", 0, True),
                                       ("complete", 1, False), ("running", 0, False)):
            output.write_text(json.dumps({"status": status}))
            assert finalize_report(output, code, 8) is expected
            result = json.loads(output.read_text())
            assert result["launcher"]["all_workers_succeeded"] is expected
            assert result["status"] == ("complete" if expected else "failed")


if __name__ == "__main__":
    test_stage2_metric_new_recovery_excludes_d0_correct()
    test_stage2_metric_d0_harmed_counts_any_d0_correct_loss()
    test_stage2_metric_identity_arm_is_all_zero()
    test_normalize_exact_match()
    test_window_sets_partitions()
    test_reader_evaluation_restores_state()
    test_oracle_answer_ce_masks_prompt()
    test_protection_penalty_is_hinge_over_d0_ce()
    test_offloaded_checkpoint_gradient_accumulation()
    test_protected_training_pool_gradients_and_restoration()
    test_distributed_gradient_equivalence()
    test_parallel_launcher_report_certification()
    print("PASS 12/12")
