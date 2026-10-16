"""Tests of fixed-UL cohorts, finite one-sided penalties and trainable state."""
import copy
import unittest
import torch
from torch.nn.utils import parametrize
from s1_soft_local import (derive_sets, soft_regret, make_displacement,
                           metric_entry, candidate_gate)
from s1_local_protected import disable_reader_dropout


class SoftLocalTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.questions = ["q%d" % i for i in range(17)]
        self.full = dict.fromkeys(self.questions, True)
        self.select = {q: i < 7 or i >= 13 for i, q in enumerate(self.questions)}
        self.d0 = {q: i >= 13 for i, q in enumerate(self.questions)}
        self.original = self.questions[:13]
        self.sets = derive_sets(self.original, self.full, self.select, self.d0)
        self.baselines = {name: {"correct": correct, "ce": dict.fromkeys(self.questions, 1.0)}
                          for name, correct in (("full", self.full), ("select", self.select), ("d0", self.d0))}

    def test_cohorts_do_not_conflate_original_and_selection_loss(self):
        self.assertEqual(self.sets["rescue"], self.questions[7:13])
        self.assertEqual(self.sets["repair"], self.questions[:7])
        self.assertEqual(self.sets["protect"], self.questions[13:])
        self.assertEqual(set(self.sets["retained"]), set(self.sets["repair"] + self.sets["protect"]))
        self.assertTrue(set(self.sets["rescue"]).isdisjoint(self.sets["retained"]))
        self.assertEqual(self.sets["original_lost"], self.original)

    def test_invalid_supervision_and_universe(self):
        with self.assertRaises(ValueError):
            derive_sets(self.original[:-1], self.full, self.select, self.d0)
        invalid = dict(self.d0, q0=True)
        with self.assertRaises(ValueError):
            derive_sets(self.original, self.full, self.select, invalid)
        with self.assertRaises(ValueError):
            derive_sets(self.original, self.full, {}, self.d0)

    def test_single_sided_regret_and_detached_reference(self):
        current = torch.tensor([0.5, 1.0, 1.5], requires_grad=True)
        reference = torch.ones(3, requires_grad=True)
        regret = soft_regret(current, reference)
        torch.testing.assert_close(regret, torch.tensor([0.0, 0.0, 0.25]))
        regret.sum().backward()
        torch.testing.assert_close(current.grad, torch.tensor([0.0, 0.0, 1.0]))
        self.assertIsNone(reference.grad)

    def test_thirteen_coordinates_budget_and_frozen_original(self):
        layer = torch.nn.Linear(7, 5, bias=False)
        original = layer.weight.detach().clone()
        layer.weight.requires_grad_(False)
        basis = torch.linalg.qr(torch.randn(35, 13)).Q
        displacement = make_displacement(layer.weight, basis)
        parametrize.register_parametrization(layer, "weight", displacement)
        self.assertEqual(displacement.value.numel(), 13)
        self.assertTrue(torch.equal(layer.weight, original))
        loss = layer(torch.randn(3, 7)).square().sum()
        loss.backward()
        self.assertGreater(float(displacement.value.grad.norm()), 0.0)
        self.assertIsNone(layer.parametrizations.weight.original.grad)
        with torch.no_grad():
            displacement.value.fill_(1.0)
        displacement.clamp()
        self.assertLessEqual(float(displacement.value.detach().norm()), 1.000001)
        self.assertLessEqual(float((layer.weight.detach() - original).norm()), 1.000001)
        self.assertTrue(torch.equal(layer.parametrizations.weight.original, original))

    def test_zero_lambda_accumulation_matches_original_update(self):
        a = torch.nn.Linear(7, 5, bias=False)
        a.weight.requires_grad_(False)
        b = copy.deepcopy(a)
        basis = torch.linalg.qr(torch.randn(35, 13)).Q
        da, db = (make_displacement(layer.weight, basis) for layer in (a, b))
        for layer, displacement in ((a, da), (b, db)):
            parametrize.register_parametrization(layer, "weight", displacement)
        inputs = [torch.randn(3, 7) for _ in range(13)]
        oa, ob = (torch.optim.AdamW([d.value], lr=.003, weight_decay=.01) for d in (da, db))
        for x in inputs:
            a(x).square().mean().backward()
        for x in inputs:
            b(x).square().mean().backward()
        torch.testing.assert_close(da.value.grad, db.value.grad, rtol=0, atol=0)
        oa.step()
        ob.step()
        self.assertTrue(torch.equal(da.value, db.value))

    def test_deterministic_dropout_with_checkpoint_gradient(self):
        from torch.utils.checkpoint import checkpoint
        layer = torch.nn.Linear(7, 5, bias=False)
        layer.weight.requires_grad_(False)
        basis = torch.linalg.qr(torch.randn(35, 13)).Q
        displacement = make_displacement(layer.weight, basis)
        parametrize.register_parametrization(layer, "weight", displacement)
        model = torch.nn.Sequential(layer, torch.nn.Dropout(.7)).train()
        disable_reader_dropout(model)
        x = torch.randn(3, 7)
        a = checkpoint(model, x, use_reentrant=False)
        b = checkpoint(model, x, use_reentrant=False)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(model.training)
        a.square().sum().backward()
        self.assertGreater(float(displacement.value.grad.norm()), 0)

    def test_separate_reader_graphs_accumulate_regret_gradient(self):
        alpha = torch.nn.Parameter(torch.zeros(13))
        def ce(i):
            return (alpha[i] + 1.2).square()
        for i in range(13):
            ce(i).backward()
        expected = torch.full((13,), 2.4)
        for i in range(4):
            soft_regret(ce(i), 1.0).mul(13 / 4).backward()
            expected[i] += 13 / 4 * 2 * (1.44 - 1.0) * 2.4
        torch.testing.assert_close(alpha.grad, expected)

    def test_pool_injection_accumulation_and_exception_restore(self):
        from types import SimpleNamespace
        from methods.prlmc.probes import _pool_in_place
        model = torch.nn.Module()
        model.memory = torch.nn.Parameter(torch.randn(1, 8, 7), requires_grad=False)
        before = model.memory.detach().clone()
        manager = SimpleNamespace(model=model, total=8, used=6)
        basis = torch.linalg.qr(torch.randn(14, 13)).Q
        alpha = torch.nn.Parameter(torch.zeros(13))
        candidate = (basis @ alpha).reshape(1, 2, 7) + 1.0
        with _pool_in_place(manager, "cpu", candidate, 4, 2) as injection:
            pool = injection.reinject()
            for _ in range(13):
                pool[:, 4:6, :].square().mean().backward(retain_graph=True)
            for _ in range(31):
                soft_regret(pool[:, 4:6, :].square().mean(), .5).mul(13 / 31).backward(retain_graph=True)
        expected = basis.T @ torch.full((14,), 26.0 / 7)
        torch.testing.assert_close(alpha.grad, expected, rtol=1e-5, atol=1e-5)
        self.assertTrue(torch.equal(model.memory, before))
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with _pool_in_place(manager, "cpu", candidate.detach(), 4, 2) as injection:
                injection.reinject()
                raise RuntimeError("fixture")
        self.assertTrue(torch.equal(model.memory, before))

    def test_production_compactor_cloned_source_gradient_equality(self):
        """Regression for the s1-soft training-graph defect.

        Promoted from tools/audit_s1_checkpoint_recompute.py: the production
        `CarrierResidualCompactor` runs through `_pool_in_place` with (a) a
        cloned source (the repaired driver path) and (b) an aliased `.data`
        view of the pool storage (the historical defect). The layout mirrors
        the real run — filler prefix, window, remainder — so the injection
        range [new_start, new_start+target) lands inside the view's tail, the
        exact overlap that corrupted the historical backward. The cloned
        path's parameter gradients must equal the no-injection reference
        exactly; the aliased path must differ, proving the test has teeth.
        See runs/s1_checkpoint_recompute_production_cpu.json.
        """
        from types import SimpleNamespace
        from methods.prlmc.consolidate import CarrierResidualCompactor
        from methods.prlmc.probes import _pool_in_place
        layers, block, d, rank = 2, 2, 11, 2
        b, remainder, filler = 3, 4, 2
        torch.manual_seed(23)
        total = filler + b * block + remainder * block
        used = b * block + remainder * block
        model = torch.nn.Module()
        model.memory = torch.nn.Parameter(torch.randn(1, total, d), requires_grad=False)
        pristine = model.memory.detach().clone()
        manager = SimpleNamespace(model=model, total=total, used=used)
        compactor = CarrierResidualCompactor(num_layers=layers, hidden=d, block=block, rank=rank)
        for p in compactor.parameters():
            p.requires_grad_(True)
        compactor.train()
        begin = total - used
        window_used = b * block
        weights = torch.rand(used)
        goal = torch.randn(1, d)
        select_w = {"importance": 0.5}
        target = 2 * block

        def run(source, inject=True):
            for p in compactor.parameters():
                p.grad = None
            torch.manual_seed(31)
            cand = compactor(source, weights, target=target, block=block,
                             goal_direction=goal, select_weights=select_w)
            loss = cand.square().sum()
            if inject:
                # Backward inside the with-block, exactly like the driver's
                # training step. With `window_used` set to the source width,
                # the injection lands at new_start = total - target, inside
                # the view's tail — the overlap that corrupted the historical
                # backward. An independent source must be immune to it.
                with _pool_in_place(manager, "cpu", cand, window_used, target) as injection:
                    injection.reinject()
                    loss.backward()
            else:
                loss.backward()
            return float(loss.detach()), [p.grad.clone() for p in compactor.parameters()]

        view = model.memory.data[:, begin:begin + used, :]
        direct_loss, direct_grads = run(view.clone(), inject=False)
        clone_loss, clone_grads = run(view.clone())
        alias_loss, alias_grads = run(view)
        self.assertEqual(clone_loss, direct_loss)
        self.assertEqual(alias_loss, direct_loss)
        for got, ref in zip(clone_grads, direct_grads):
            torch.testing.assert_close(got, ref, rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(a, r) for a, r in zip(alias_grads, direct_grads)),
                        "aliased source must corrupt gradients, else this test is blind")
        self.assertTrue(torch.equal(model.memory.detach(), pristine), "pool must be restored")

    def test_net_and_penalty_are_rebuilt_per_question(self):
        correct = dict(self.select)
        correct["q7"] = True
        correct["q13"] = False
        ce = dict.fromkeys(self.questions, 0.5)
        ce["q13"] = 1.5
        entry = metric_entry({"correct": correct, "ce": ce}, self.sets, self.baselines,
                             self.sets["protect"], 1.0, 60)
        self.assertEqual(entry["rescue_recovered"], 1)
        self.assertEqual(entry["repair_recovered"], 7)
        self.assertEqual(entry["protect_harmed"], 1)
        self.assertEqual(entry["retained_harmed"], 1)
        self.assertEqual(entry["net_vs_select"], 0)
        self.assertAlmostEqual(entry["penalty"], 13 * .25 / 4)
        self.assertEqual(entry["original_rescue"], 8)

    def test_strict_improvement_required_not_ce_only(self):
        control = {"original_rescue": 7, "protect_harmed": 0, "rescue_recovered": 14,
                   "repair_recovered": 38, "retained_harmed": 6, "net_vs_select": 8}
        self.assertFalse(all(candidate_gate(control, control).values()))
        candidate = dict(control, rescue_recovered=15, net_vs_select=9)
        self.assertTrue(all(candidate_gate(candidate, control).values()))
        candidate["repair_recovered"] = 37
        self.assertFalse(all(candidate_gate(candidate, control).values()))


if __name__ == "__main__":
    unittest.main()
