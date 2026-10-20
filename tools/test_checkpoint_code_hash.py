#!/usr/bin/env python3
"""CPU contracts for checkpoint source-hash verification (C3/C4).

No CUDA, no reader checkpoint: provenance checks only touch dicts and file hashes.
"""
import inspect
import io
from contextlib import redirect_stderr
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from natural_merge_runtime import (CRITICAL_MERGE_FILES, SOURCE_FILES, critical_manifest,
                                   load_merge_state, load_runtime, source_manifest,
                                   validate_merge_checkpoint)

V5_CHECKPOINT = ROOT/"runs/natural_merge_v5_20261018_1050/Dsup.train.pt"


def executable_saved(manifest="current", **overrides):
    """Minimal executable-shaped checkpoint dict (metadata-only dicts are exempt)."""
    saved = dict(schema="natural_merge_checkpoint/1", state_dict={"up_weight": torch.zeros(2)},
                 effective_gamma_max=.05, evaluation_eligible=True,
                 source_manifest=source_manifest() if manifest == "current" else manifest)
    saved.update(overrides)
    return saved


def metadata_only_saved(**overrides):
    """Same shape as the hand-built fixtures in test_natural_merge_v3/v4/v6."""
    from prepare_natural_merge_v4 import TRAIN_V4
    saved = dict(variant="Dsup", training=TRAIN_V4,
                 module_configuration=TRAIN_V4["variants"]["Dsup"],
                 effective_gamma_max=.05, global_rms_budget=TRAIN_V4["global_rms_budget"],
                 rank=TRAIN_V4["rank"], evaluation_eligible=True)
    saved.update(overrides)
    return saved


class ManifestCoverage(unittest.TestCase):
    def test_critical_files_are_the_six_operator_files(self):
        self.assertEqual(CRITICAL_MERGE_FILES, [
            "methods/prlmc/consolidate.py", "methods/prlmc/layout.py",
            "methods/prlmc/importance.py", "methods/prlmc/slot_merge.py",
            "methods/prlmc/gated_slot_merge.py", "methods/prlmc/native_skip_merge.py"])
        self.assertTrue(set(CRITICAL_MERGE_FILES).issubset(SOURCE_FILES),
                        "critical files must stay inside the recorded manifest")

    def test_critical_manifest_is_sha256_hex(self):
        manifest = critical_manifest()
        self.assertEqual(sorted(manifest), sorted(CRITICAL_MERGE_FILES))
        for name, value in manifest.items():
            self.assertRegex(value, r"^[0-9a-f]{64}$", name)


class StrictVerification(unittest.TestCase):
    def test_matching_manifest_passes(self):
        self.assertTrue(validate_merge_checkpoint(executable_saved(), "block_v1"))

    def test_tampered_file_is_rejected_with_both_prefixes(self):
        manifest = source_manifest()
        victim = "methods/prlmc/gated_slot_merge.py"
        manifest[victim] = "0" * 64
        with self.assertRaises(ValueError) as caught:
            validate_merge_checkpoint(executable_saved(manifest), "block_v1")
        message = str(caught.exception)
        self.assertIn(victim, message)
        self.assertIn("ckpt 00000000", message)
        self.assertIn(f"code {critical_manifest()[victim][:8]}", message)

    def test_missing_critical_entry_in_manifest_is_rejected(self):
        manifest = source_manifest()
        victim = "methods/prlmc/consolidate.py"
        del manifest[victim]
        with self.assertRaises(ValueError) as caught:
            validate_merge_checkpoint(executable_saved(manifest), "block_v1")
        self.assertIn(f"{victim}: ckpt <missing>", str(caught.exception))
        self.assertIn(critical_manifest()[victim][:8], str(caught.exception))

    def test_missing_manifest_is_rejected_by_default(self):
        saved = executable_saved(manifest=None)
        del saved["source_manifest"]
        with self.assertRaises(ValueError) as caught:
            validate_merge_checkpoint(saved, "block_v1")
        self.assertIn("no source_manifest", str(caught.exception))

    def test_non_mapping_manifest_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_merge_checkpoint(executable_saved(manifest=["not", "a", "mapping"]), "block_v1")

    def test_metadata_only_fixture_keeps_structural_validation(self):
        # test_natural_merge_v3/v4/v6 build metadata-only dicts without weights;
        # they must keep validating exactly as before.
        architecture = "grouped_gated_sparse_slot_v5"
        self.assertFalse(validate_merge_checkpoint(metadata_only_saved(), architecture))
        with self.assertRaises(ValueError):
            validate_merge_checkpoint(metadata_only_saved(global_rms_budget=.2), architecture)


class HistoricalCompatibility(unittest.TestCase):
    def _assert_warns(self, saved, architecture):
        buffer = io.StringIO()
        with redirect_stderr(buffer):
            verified = validate_merge_checkpoint(saved, architecture, strict_code_hash=False)
        warning = buffer.getvalue()
        self.assertFalse(verified)
        self.assertIn("WARNING", warning)
        self.assertIn("historical diagnosis only", warning)
        return warning

    def test_missing_manifest_allowed_only_when_not_strict(self):
        saved = executable_saved(manifest=None)
        del saved["source_manifest"]
        self._assert_warns(saved, "block_v1")

    def test_mismatched_manifest_allowed_only_when_not_strict(self):
        manifest = source_manifest()
        manifest["methods/prlmc/native_skip_merge.py"] = "f" * 64
        warning = self._assert_warns(executable_saved(manifest), "block_v1")
        self.assertIn("native_skip_merge.py: ckpt ffffffff", warning)

    def test_matching_manifest_is_verified_under_both_modes(self):
        saved = executable_saved()
        self.assertTrue(validate_merge_checkpoint(saved, "block_v1", strict_code_hash=True))
        buffer = io.StringIO()
        with redirect_stderr(buffer):
            self.assertTrue(validate_merge_checkpoint(saved, "block_v1", strict_code_hash=False))
        self.assertEqual(buffer.getvalue(), "")


class ValidateBeforeLoad(unittest.TestCase):
    class FakeMerge:
        def __init__(self):
            self.loaded = []

        def load_state_dict(self, state_dict, strict=True):
            self.loaded.append(strict)

    def test_bad_manifest_never_reaches_load_state_dict(self):
        merge = self.FakeMerge()
        saved = executable_saved(manifest={name: "0" * 64 for name in CRITICAL_MERGE_FILES})
        with self.assertRaises(ValueError):
            load_merge_state(merge, saved, "block_v1")
        self.assertEqual(merge.loaded, [], "weights must not be copied before provenance checks")

    def test_missing_manifest_never_reaches_load_state_dict(self):
        merge = self.FakeMerge()
        saved = executable_saved(manifest=None)
        del saved["source_manifest"]
        with self.assertRaises(ValueError):
            load_merge_state(merge, saved, "block_v1")
        self.assertEqual(merge.loaded, [])

    def test_verified_checkpoint_loads_strictly(self):
        merge = self.FakeMerge()
        self.assertTrue(load_merge_state(merge, executable_saved(), "block_v1"))
        self.assertEqual(merge.loaded, [True])

    def test_load_runtime_forwards_strict_code_hash(self):
        signature = inspect.signature(load_runtime)
        self.assertIs(signature.parameters["strict_code_hash"].default, True)


class TrainingWriter(unittest.TestCase):
    def test_v4_trainer_persists_source_manifest(self):
        import ast
        tree = ast.parse((ROOT/"tools/natural_merge_v4_train.py").read_text())
        checkpoints = [node for node in ast.walk(tree)
                       if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                       and node.func.id == "dict"
                       and {"schema", "state_dict"} <= {kw.arg for kw in node.keywords}]
        self.assertEqual(len(checkpoints), 1, "expected one saved checkpoint dict")
        self.assertIn("source_manifest", {kw.arg for kw in checkpoints[0].keywords})

    def test_manifest_survives_torch_save_roundtrip(self):
        import tempfile
        saved = executable_saved()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"ckpt.pt"
            torch.save(saved, path)
            restored = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(restored["source_manifest"], saved["source_manifest"])
        self.assertTrue(validate_merge_checkpoint(restored, "block_v1"))


@unittest.skipUnless(V5_CHECKPOINT.is_file(), "historical v5 checkpoint not present")
class HistoricalV5Checkpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = torch.load(V5_CHECKPOINT, map_location="cpu", weights_only=True)

    def test_v5_checkpoint_has_no_source_manifest(self):
        self.assertIn("state_dict", self.saved)
        self.assertNotIn("source_manifest", self.saved)
        self.assertIn("training_sources", self.saved)

    def test_v5_checkpoint_rejected_by_default(self):
        with self.assertRaises(ValueError) as caught:
            validate_merge_checkpoint(self.saved, self.saved["architecture"])
        self.assertIn("no source_manifest", str(caught.exception))

    def test_v5_checkpoint_loads_only_with_explicit_opt_out(self):
        buffer = io.StringIO()
        with redirect_stderr(buffer):
            verified = validate_merge_checkpoint(self.saved, self.saved["architecture"],
                                                 strict_code_hash=False)
        self.assertFalse(verified)
        self.assertIn("historical diagnosis only", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
