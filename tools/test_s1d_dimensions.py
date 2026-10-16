"""Verify dimension-only ablations reproduce the original nested C subspaces."""
import ast
from pathlib import Path
import torch
from s1d_dimension_basis import VARIANTS, build_basis


def main():
    generator = torch.Generator().manual_seed(11)
    rows = torch.randn(40, 96, generator=generator, dtype=torch.float64)
    u, s, vh = torch.linalg.svd(rows.double(), full_matrices=False)
    for variant, (dimension, projected) in VARIANTS.items():
        basis, audit = build_basis(rows, dimension)
        assert not projected and audit['recovery_energy_retained'] == 1
        torch.testing.assert_close(basis, vh[:dimension].T.float().contiguous(), atol=0, rtol=0)
        assert basis.shape == (96, dimension)
        assert audit['protection_rows'] == 0 and audit['effective_rank'] == 40
        try:
            build_basis(rows[:dimension-1], dimension)
        except ValueError:
            pass
        else:
            raise AssertionError('rank-deficient basis accepted')
        print('PASS ' + variant, flush=True)
    p = Path(__file__).parent
    original = ast.parse((p/'s1d_projected_train.py').read_text())
    dimension_driver = ast.parse((p/'s1d_dimension_train.py').read_text())
    for name in ('train_arm', 'fact_gradient_rows', 'evaluate', 'candidate',
                 'fresh_window', 'select_fact_cohorts', 'stage2_metric', 'validate_c_reference'):
        get = lambda tree: next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(get(original), include_attributes=False) == ast.dump(get(dimension_driver), include_attributes=False), name
    print('PASS unchanged C training and evaluation', flush=True)


if __name__ == '__main__':
    main()
