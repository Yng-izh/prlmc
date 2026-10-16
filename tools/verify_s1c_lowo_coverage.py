#!/usr/bin/env python
"""服务器端 verifier：独立重算 LOWO 覆盖率关键数字，校验 s1c_lowo_coverage.json。"""
import hashlib
import json
import sys

import torch

ROWS = "/share/latent/PRLM-C/runs/s1c_global_rows.pt"
REP = "/share/latent/PRLM-C/runs/s1c_lowo_coverage.json"
RANKS = (13, 32, 64)

fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name} {detail}")
    if not ok:
        fails.append(name)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


rep = json.load(open(REP))

# 1. 哈希链
check("rows_sha256", rep["rows_sha256"] == sha256_file(ROWS))

# 2. 独立重算：不用驱动脚本的函数，直接 QR 投影实现
d = torch.load(ROWS, map_location="cpu", weights_only=False)
rbw = d["rows_by_window"]
wins = {wk: torch.stack([rbw[wk][q] for q in sorted(rbw[wk])]) for wk in rbw}
keys = sorted(wins)
check("n_windows", len(keys) == 8, f"got {len(keys)}")
check("n_rows", sum(wins[k].shape[0] for k in keys) == 240)

pooled_num = {r: 0.0 for r in RANKS}
pooled_den = 0.0
self_num = {r: 0.0 for r in RANKS}

for held in keys:
    G_tr = torch.cat([wins[k] for k in keys if k != held])
    mu = G_tr.mean(0, keepdim=True)
    Gt = wins[held] - mu
    den = (Gt * Gt).sum().item()
    pooled_den += den
    # 训练基：中心化 + SVD
    Gc = G_tr - G_tr.mean(0, keepdim=True)
    _, _, Vt = torch.linalg.svd(Gc, full_matrices=False)
    for r in RANKS:
        U = Vt[:r].t()
        # 投影能量：trace(U^T Gt^T Gt U) = ||Gt U||_F^2（与 ||U^T g||^2 等价）
        num = (Gt @ U).pow(2).sum().item()
        pooled_num[r] += num
        # 报告值核对
        entry = next(w for w in rep["windows"] if w["held_out_window"] == held)
        reported = entry["ranks"][str(r)]["lowo_energy_coverage"]
        recomputed = num / den
        # 跨设备 float32 SVD 舍入差容限
        ok = abs(reported - recomputed) < 1e-3
        check(f"lowo r{r} {held}", ok, f"rep={reported:.6f} rec={recomputed:.6f}")
        # 自建基
        Gs = wins[held] - wins[held].mean(0, keepdim=True)
        _, _, Vts = torch.linalg.svd(Gs, full_matrices=False)
        Us = Vts[:r].t()
        snum = (Gt @ Us).pow(2).sum().item()
        self_num[r] += snum
        rs = entry["ranks"][str(r)]["self_energy_coverage"]
        ok2 = abs(rs - snum / den) < 1e-3
        check(f"self r{r} {held}", ok2, f"rep={rs:.6f} rec={snum/den:.6f}")

# 3. 汇总一致性
for r in RANKS:
    rec = pooled_num[r] / pooled_den
    rep_v = rep["summary"][str(r)]["pooled_lowo_energy_coverage"]
    check(f"pooled lowo r{r}", abs(rep_v - rec) < 1e-9, f"rep={rep_v:.6f} rec={rec:.6f}")
    recs = self_num[r] / pooled_den
    reps = rep["summary"][str(r)]["pooled_self_energy_coverage"]
    check(f"pooled self r{r}", abs(reps - recs) < 1e-9, f"rep={reps:.6f} rec={recs:.6f}")

# 4. 判定数字：秩 13→64 提升幅度
d13 = rep["summary"]["13"]["pooled_lowo_energy_coverage"]
d64 = rep["summary"]["64"]["pooled_lowo_energy_coverage"]
s13 = rep["summary"]["13"]["pooled_self_energy_coverage"]
check("delta_r13_r64", d64 - d13 < 0.15, f"gain={d64-d13:.4f} (<0.15 → 非秩不足)")
check("self_vs_lowo_gap", s13 - d13 > 0.5, f"gap={s13-d13:.4f} (>0.5 → 窗口子空间错开)")

print(f"\n{len(fails)} FAIL" if fails else "\nALL PASS")
sys.exit(1 if fails else 0)
