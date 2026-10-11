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
    print("PASS 9/9")
