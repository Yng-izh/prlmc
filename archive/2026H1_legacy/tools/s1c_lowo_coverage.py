#!/usr/bin/env python
"""S1c LOWO 梯度覆盖率审计（层 1 失败根因裁决）。

对 s1c_global_rows.pt 的 240 行 rescue Jacobian 做 leave-one-window-out：
用 7 个训练窗口的行建 SVD 基 U (d x r)，测未见窗口行的梯度能量覆盖率
    rho(U, g) = || U^T g ||_2^2 / || g ||_2^2     (行级)
    rho(U, G) = || U^T G^T ||_F^2 / || G ||_F^2   (窗口级聚合)
比较 r = 13 / 32 / 64，并报告窗口内自建基（train-on-test 上界）作参照。
基全部行中心化后 SVD，g 同样用训练均值中心化（消平移偏移的贡献）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone

import torch

RANKS = (13, 32, 64)
SCRIPT_SHA = None  # 运行时填充


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path: str):
    d = torch.load(path, map_location="cpu", weights_only=False)
    rbw = d["rows_by_window"]
    windows = {}
    for wk, rows in rbw.items():
        # 排序保证跨运行键序稳定
        windows[wk] = torch.stack([rows[q] for q in sorted(rows)])
    return windows, d.get("checkpoint_sha256", "")


def svd_basis(G: torch.Tensor, rank: int):
    """G: (n, d) 行中心化后取前 rank 个右奇异向量，返回 (d, rank)。"""
    Gc = G - G.mean(dim=0, keepdim=True)
    # 经济 SVD：n≈210 x d=131072，V 从 (d, min(n,d)) 取
    _, _, Vt = torch.linalg.svd(Gc, full_matrices=False)
    return Vt[:rank].t().contiguous()


def coverage(U: torch.Tensor, g: torch.Tensor) -> float:
    """单行覆盖率 ||U^T g||^2 / ||g||^2。"""
    proj = U.t() @ g
    return (proj @ proj).item() / (g @ g).item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="runs/s1c_global_rows.pt")
    ap.add_argument("--out", default="runs/s1c_lowo_coverage.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    SCRIPT_SHA = sha256_file(os.path.abspath(__file__))
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    windows, ckpt_sha = load_rows(args.rows)
    win_keys = sorted(windows)
    assert len(win_keys) == 8, f"expect 8 windows, got {len(win_keys)}"

    report = {
        "schema_version": "s1c_lowo_coverage/1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rows_sha256": sha256_file(args.rows),
        "checkpoint_sha256": ckpt_sha,
        "driver_sha256": SCRIPT_SHA,
        "ranks": list(RANKS),
        "protocol": {
            "method": "leave-one-window-out SVD coverage",
            "centering": "train-mean centering for basis; test rows centered with train mean",
            "row_level": "rho_i = ||U^T g_i||^2 / ||g_i||^2",
            "aggregate": "energy-weighted: sum_i ||U^T g_i||^2 / sum_i ||g_i||^2",
            "basis_ranks": list(RANKS),
            "note_selfbasis": "self basis = SVD of the held-out window's own rows (oracle upper bound)",
        },
        "windows": [],
    }

    # 预取全部张量到设备（8 窗口共 240x131072 fp32 ≈ 126 MB，显存足够）
    G_dev = {wk: windows[wk].to(dev) for wk in win_keys}
    n_total = sum(G_dev[wk].shape[0] for wk in win_keys)
    assert n_total == 240, f"expect 240 rows, got {n_total}"

    grand_rows = {r: [] for r in RANKS}
    grand_self = {r: [] for r in RANKS}

    for held in win_keys:
        train_keys = [k for k in win_keys if k != held]
        G_tr = torch.cat([G_dev[k] for k in train_keys], dim=0)
        mu = G_tr.mean(dim=0, keepdim=True)
        G_te = G_dev[held]  # (n_h, d)
        n_h = G_te.shape[0]

        entry = {
            "held_out_window": held,
            "n_train_rows": int(G_tr.shape[0]),
            "n_test_rows": int(n_h),
            "ranks": {},
        }

        for r in RANKS:
            U = svd_basis(G_tr, r)  # (d, r)
            Gt = G_te - mu  # 测试行用训练均值中心化
            # 覆盖率：分子 sum ||U^T g||^2，分母 sum ||g||^2
            P = (U.t() @ Gt.t())  # (r, n_h)
            num = (P * P).sum().item()
            den = (Gt * Gt).sum().item()
            agg = num / den
            # 行级分布：P 为 (r, n_h)，逐列求能量
            row_num = (P * P).sum(dim=0)          # (n_h,)
            row_den = (Gt * Gt).sum(dim=1)         # (n_h,)
            rho_rows = row_num / row_den
            # 自建基（该窗口自己的行）作为上界
            Uself = svd_basis(G_te, r)
            Ps = Uself.t() @ Gt.t()
            snum = (Ps * Ps).sum().item()
            sagg = snum / den
            entry["ranks"][str(r)] = {
                "lowo_energy_coverage": agg,
                "lowo_row_mean": rho_rows.mean().item(),
                "lowo_row_median": rho_rows.median().item(),
                "lowo_row_min": rho_rows.min().item(),
                "lowo_row_q25": torch.quantile(rho_rows, 0.25).item(),
                "lowo_row_q75": torch.quantile(rho_rows, 0.75).item(),
                "self_energy_coverage": sagg,
            }
            grand_rows[r].append((num, den))
            grand_self[r].append((snum, den))

        report["windows"].append(entry)
        print(f"[{held}] " + " ".join(
            f"r{r}:lowo={entry['ranks'][str(r)]['lowo_energy_coverage']:.4f},self={entry['ranks'][str(r)]['self_energy_coverage']:.4f}"
            for r in RANKS), flush=True)

    # 全窗口汇总（能量加权）
    report["summary"] = {}
    for r in RANKS:
        num = sum(x[0] for x in grand_rows[r])
        den = sum(x[1] for x in grand_rows[r])
        snum = sum(x[0] for x in grand_self[r])
        report["summary"][str(r)] = {
            "pooled_lowo_energy_coverage": num / den,
            "pooled_self_energy_coverage": snum / den,
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
