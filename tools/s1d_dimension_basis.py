"""CPU-only construction of a protected, low-dimensional C update space.

Hard projection is local to D0 and the selected write-time facts. It does not
guarantee preservation of nonlinear reader predictions after finite updates.
"""
import hashlib
import math
import os
from pathlib import Path
import tempfile

import torch

RELATIVE_RANK_TOL = 1e-7
ABSOLUTE_RANK_TOL = 1e-10
NUMERICAL_TOL = 5e-6
VARIANTS = {f"C{dimension}": (dimension, False) for dimension in (13, 20, 24, 28, 32)}
PROTECTION_MAX = 40


def projection_protocol(variant):
    dimension, projected = VARIANTS[variant]
    return dict(version=1, variant=variant, dimension=dimension,
                projected=projected, projection="hard orthogonal projection in full down.weight space before SVD",
                protection_selection="up to 40 encoded kept latest-owner facts with finite 0<=D0 answer CE<=2; sorted key",
                protection_max=PROTECTION_MAX,
                protection_ce_ceiling=2.0, protection_weight_scan=False,
                relative_rank_tol=RELATIVE_RANK_TOL, absolute_rank_tol=ABSOLUTE_RANK_TOL,
                numerical_tol=NUMERICAL_TOL, qa_used_for_basis=False,
                objective="unchanged C summed tail-4 CE gradients; no protection penalty",
                checkpoint_selection="final step60 only", seed=0)


def select_protection(records, encoded_keys, d0_ce, max_count=PROTECTION_MAX):
    keys = [r["key"] for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate kept fact key")
    encoded_keys = set(encoded_keys)
    selected = sorted(k for k in keys if k in encoded_keys and
                      math.isfinite(d0_ce[k]) and 0 <= d0_ce[k] <= 2.0)
    return selected[:max_count]


def tensor_sha256(tensor):
    t = tensor.detach().cpu().contiguous()
    h = hashlib.sha256(str((str(t.dtype), tuple(t.shape))).encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()


def _rows(rows, label):
    if not isinstance(rows, torch.Tensor) or rows.ndim != 2 or min(rows.shape) == 0:
        raise ValueError(label + " must be a nonempty row matrix")
    if not torch.isfinite(rows).all():
        raise ValueError("nonfinite " + label)
    return rows.detach().to(device="cpu", dtype=torch.float64)


def _svd(rows, scale=None):
    _, singular, vh = torch.linalg.svd(rows, full_matrices=False)
    # For projected rows, use the original damaged spectrum as the reference.
    # A nearly annihilated matrix must not acquire artificial relative rank.
    reference = float(singular.max()) if scale is None else scale
    keep = singular > max(ABSOLUTE_RANK_TOL, reference * RELATIVE_RANK_TOL)
    return singular, vh[keep].T.contiguous()


def build_basis(damaged_rows, dimension, protection_rows=None):
    if dimension not in (13, 20, 24, 28, 32):
        raise ValueError("only predeclared dimensions 13, 20, 24, 28 and 32 are supported")
    damaged = _rows(damaged_rows, "damaged gradients")
    raw_singular, raw_basis = _svd(damaged)
    original_energy = float(damaged.square().sum())
    if original_energy == 0:
        raise ValueError("zero recovery signal")
    protected = None
    protected_rank = zero_rows = 0
    projected = damaged
    if protection_rows is not None:
        protected = _rows(protection_rows, "protection gradients")
        if protected.shape[1] != damaged.shape[1]:
            raise ValueError("gradient ambient dimensions differ")
        norms = protected.norm(dim=1)
        zero_rows = int((norms == 0).sum())
        protected = protected[norms > 0] / norms[norms > 0, None]
        if not len(protected):
            raise ValueError("all protection gradients are zero")
        _, protect_basis = _svd(protected)
        protected_rank = protect_basis.shape[1]
        projected = damaged - (damaged @ protect_basis) @ protect_basis.T
    singular, basis = _svd(projected, scale=float(raw_singular.max()))
    rank = basis.shape[1]
    stats = dict(dimension=dimension, ambient_dimension=damaged.shape[1],
                 damaged_rows=len(damaged), raw_rank=raw_basis.shape[1],
                 protection_rows=0 if protection_rows is None else len(protection_rows),
                 protection_rank=protected_rank, zero_protection_rows=zero_rows,
                 effective_rank=rank, raw_singular_values=raw_singular.tolist(),
                 projected_singular_values=singular.tolist(),
                 original_gradient_energy=original_energy,
                 projected_gradient_energy=float(projected.square().sum()),
                 recovery_energy_retained=float(projected.square().sum()) / original_energy)
    if rank < dimension:
        raise ValueError("effective rank %d < requested %d; no fallback; audit=%s" % (rank, dimension, stats))
    basis = basis[:, :dimension].float().contiguous()
    gram = basis.double().T @ basis.double()
    stats["orthonormal_error"] = float((gram - torch.eye(dimension)).abs().max())
    stats["protection_max_abs_directional_ce"] = (0.0 if protected is None else
        float((protected @ basis.double()).abs().max()))
    stats["retained_energy_in_selected_basis"] = float((projected @ basis.double()).square().sum()) / original_energy
    stats["basis_sha256"] = tensor_sha256(basis)
    if max(stats["orthonormal_error"], stats["protection_max_abs_directional_ce"]) > NUMERICAL_TOL:
        raise ValueError("float32 basis fails orthogonality/protection audit: %s" % stats)
    return basis, stats


def validate_protection_cache(saved, metadata, keys, ambient_dimension):
    if saved.get("schema") != "s1d_protection_rows/1" or saved.get("metadata") != metadata:
        raise ValueError("protection cache provenance mismatch")
    if saved.get("keys") != keys:
        raise ValueError("protection cache fact order mismatch")
    rows = _rows(saved["rows"], "cached protection gradients")
    if rows.shape != (len(keys), ambient_dimension):
        raise ValueError("protection cache row shape mismatch")
    if saved.get("rows_sha256") != tensor_sha256(saved["rows"]):
        raise ValueError("protection cache tensor digest mismatch")
    return rows


def save_new_cache(saved, path):
    """Publish a fully written cache atomically; never replace another writer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
        try:
            torch.save(saved, handle)
            handle.flush()
            os.fsync(handle.fileno())
            os.link(temp, path)
        finally:
            temp.unlink(missing_ok=True)
