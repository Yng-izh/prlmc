#!/usr/bin/env python
"""Unit tests for s1d_selfsup_basis_v2 (pool construction + math, CPU-only)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from s1d_selfsup_basis_v2 import (  # noqa: E402
    eff_rank, energy_coverage, orth, select_lines_v2,
)


class FakeE6a:
    """Minimal stand-in for e6a_overfit with the real TEMPLATES subset."""

    TEMPLATES = [
        (r"^(.+) was created by (.+)$", "Who created %s?", "1"),
        (r"^(.+) is the capital of (.+)$", None, "1"),
    ]

    def first_match(self, text):
        import re
        stripped = text.rstrip(".")
        for slot, (pattern, question, groups) in enumerate(self.TEMPLATES):
            m = re.match(pattern, stripped)
            if m:
                parts = m.groups()
                try:
                    subject = tuple(parts[int(g) - 1] for g in groups.split(","))
                except (IndexError, TypeError):
                    continue
                return slot, subject, parts[-1], question
        return None


def test_select_lines_latest_owner_wins():
    lines = [
        "Alpha was created by Bob.",
        "Alpha was created by Carol.",   # newer serial -> owner
        "Beta is the capital of France.",
        "nothing matching here",
    ]
    # blocks: line0,1 in block 0 (dropped); line2,3 in block 1 (kept)
    line_to_block = {0: 0, 1: 0, 2: 1, 3: 1}
    primary, aux = select_lines_v2(lines, line_to_block, dropped=[0], e6a=FakeE6a())
    # latest owner of (slot0, "Alpha") is line 1; non-template line 3 kept via raw key
    assert primary == [1], primary
    assert aux == [2, 3], aux


def test_select_lines_no_template_lines_counted_once():
    lines = ["free text a", "free text b"]
    line_to_block = {0: 0, 1: 0}
    primary, aux = select_lines_v2(lines, line_to_block, dropped=[0], e6a=FakeE6a())
    assert primary == [0, 1], primary
    assert aux == [], aux


def test_select_lines_dedup_same_subject_across_blocks():
    lines = [
        "Alpha was created by Bob.",    # block 0 dropped
        "Alpha was created by Carol.",  # block 1 kept -> latest owner
    ]
    line_to_block = {0: 0, 1: 1}
    primary, aux = select_lines_v2(lines, line_to_block, dropped=[0], e6a=FakeE6a())
    # line 0 no longer owns its (slot, subject): dedup removes it entirely
    assert primary == [], primary
    assert aux == [1], aux


def test_eff_rank_orthonormal_rows_is_n():
    torch.manual_seed(0)
    rows = torch.linalg.qr(torch.randn(20, 64).double())[0].t()
    assert abs(eff_rank(rows) - 20.0) < 1e-6


def test_eff_rank_identical_rows_is_one():
    v = torch.randn(1, 64).double().repeat(15, 1)
    assert abs(eff_rank(v) - 1.0) < 1e-6


def test_energy_coverage_top13_is_optimal():
    torch.manual_seed(1)
    G = torch.randn(20, 64).double()
    U = orth(G)[:, :13]
    # SVD ordering: first-13 columns capture exactly top-13 singular energy
    lam = torch.linalg.svdvals(G) ** 2
    expect = float(lam[:13].sum() / lam.sum())
    assert abs(energy_coverage(U, G) - expect) < 1e-9


def test_orth_is_svd_ordered():
    torch.manual_seed(3)
    G = torch.randn(30, 64).double()
    U = orth(G)[:, :13]                       # (d, 13) row-space basis
    _, _, vh = torch.linalg.svd(G, full_matrices=False)
    U_opt = vh[:13].t()                       # (d, 13) top-13 right singulars
    M = U.t() @ U_opt
    assert torch.allclose(M @ M.t(), torch.eye(13, dtype=torch.float64), atol=1e-8)


def test_v1_row_dict_key_sorting_is_numeric():
    keys = ["10", "9", "100"]
    assert sorted(keys, key=int) == ["9", "10", "100"]


if __name__ == "__main__":
    test_select_lines_latest_owner_wins()
    test_select_lines_no_template_lines_counted_once()
    test_select_lines_dedup_same_subject_across_blocks()
    test_eff_rank_orthonormal_rows_is_n()
    test_eff_rank_identical_rows_is_one()
    test_energy_coverage_top13_is_optimal()
    test_orth_is_svd_ordered()
    test_v1_row_dict_key_sorting_is_numeric()
    print("PASS 8/8")
