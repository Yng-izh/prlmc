# PRLM-C v5 同轴复核（clean 16k）预检版

**状态：尚未执行。本文件只记录 dry-run 预检结果与命令矩阵，没有启动任何 GPU 评测/训练，没有产生新评测数据。**

日期：2026-10-20 | 预检产物：`/tmp/v5_clean_dryrun/`（`plans-fa6a22b/`、`plans-82d2548/` 各 14 份 dry-run 计划，`stats/rehearsal.json` 统计演练）
预检基线：`fa6a22b`（t2+t3）与 `82d2548`（t4）两个 worktree，**14 行矩阵在两边各跑一遍，均 14/14 通过**

---

## 0. 结论摘要

1. **矩阵 14 行 × 2 基线 = 28 次 dry-run 全部 `rc=0`**：参数解析、identity 构造、几何校验正常，且 `runs/longbench_prlmc_v5_16k_clean/` **始终未被创建**（dry-run 不写文件、不加载模型）。
2. **基线选择会改变"同轴"的含义，已裁决选 A**（§2.2）：16k 的 10 行在两个基线上**数值等价**；10-block 的 4 行**不等价**——`82d2548` 起 C1 分代配额在评测链路真实生效，而 b10 每样本压缩 2–4 次，从第 2 轮起选块就变了。
3. **16k 行数值等价有硬依据**：16k/budget 20 下**每个样本最多压缩 1 次**（实测分布见 §4.1），首次压缩时窗口内所有块 `age=0`；`_select_block_indices` 的 `generational = ages is not None and (ages > 0).any()` 对全 0 直接走原路径，且 `0.75 ** 0 == 1.0` 在 IEEE 下精确。因此 hotpotqa/narrativeqa 的 clean 结果应**逐位复现**历史 prediction（§5 自检门）。
4. **C5 决定"不能在 v7 HEAD 上跑 v5 ckpt"**：tanh 软有界会作废 v5 ckpt 的数值语义（`experiments/v7-spec-contract.md` C5.5）。`82d2548` 尚未落 C5（`gated_slot_merge.py` 哈希未变），所以 **16k 行在 A/B 上仍等价**（b10 行因 C1 已不等价，见 §0.2）；**t5（`7320135`）已落地，在 ≥7320135 的代码上跑 v5 ckpt 语义即作废**，16k 行也分叉（§2.3）。
5. **当前 GPU 条件不满足**：峰值显存 23.15–24.84 GB，需单卡 ≥26 GB 空闲；2026-10-20 12:44 实测 GPU 1 仅 23.4 GB 空闲（一个 9.1 GB 进程常驻）、GPU 2–5 被占满（31.7 GB / 100%）。**现在开跑会 OOM 或抢卡**。
6. **t10 的 13→9 仍不可执行**（不在本矩阵，但必须记录）：`82d2548` 已让 `MergeStream` 接受显式 `window_blocks/target_blocks`（`stream_kwargs(13,9)` 不再报错），但 `StreamingMemory.write` 的触发条件仍是旧的 `used + block > budget`（`public_retention_eval.py:193`），C8 的 `used >= min(budget, window+1)` 尚未落地（t6 范围）。此刻跑 13/9 会在 `used=20` 才触发、窗口只取 13 块——**不报错但语义错**，比报错更危险。t10 矩阵中该行一律标 **BLOCKED-BY-T6**，不得静默按旧触发跑。
7. 统计口径演练用 6 组历史 JSON 完成，与已发布的 `runs/longbench_prlmc_v5_16k/t2_final_stats.json`、`experiments/v6-consolidate-16k-budget.md` **逐位一致**（§4.2）。

---

## 1. 预检环境与代码指纹

```bash
git -C /share/latent/PRLM-C worktree add /share/latent/PRLM-C-evalv5 fa6a22b   # v5 算子语义
git -C /share/latent/PRLM-C worktree add /share/latent/PRLM-C-evalv7 82d2548   # t4 基线（C1/C2 已落）
# worktree list:
#   /share/latent/PRLM-C        82d2548 [master]
#   /share/latent/PRLM-C-evalv5 fa6a22b (detached)
#   /share/latent/PRLM-C-evalv7 82d2548 (detached)
```

| 项 | 值 |
|---|---|
| checkpoint | `/share/latent/PRLM-C/runs/natural_merge_v5_20261018_1050/Dsup.train.pt` |
| ckpt sha256 | `13700eca4c9b772b9adcb8e72f4f4de9e39a37a98b68755df12c80b6680ab470` |
| 架构 / 变体 | `grouped_gated_sparse_slot_v5` / `Dsup` |
| select / reference layer | 16 / 16 |
| 数据 | `/share/latent/benchmarks/LongBench/data`（hotpotqa sha256 `a0005ab2a1bc2ac3…b92ef6f`） |
| 解释器 | `/share/latent/mllm_env/bin/python` |

`critical_source_manifest`（算子代码指纹，前 16 位）：

| 文件 | `fa6a22b`（v5 语义） | `82d2548`（t4） | 是否变 |
|---|---|---|---|
| `methods/prlmc/consolidate.py` | `467ea2d268380d2e` | `c16e5837cfd9aa23` | **变**（C1 配额） |
| `methods/prlmc/layout.py` | `92138693a236f890` | `35e358c44e25b417` | **变**（C2 年龄） |
| `methods/prlmc/gated_slot_merge.py` | `c95e9d72bbbf7f94` | `c95e9d72bbbf7f94` | 未变（C5 未落） |
| `methods/prlmc/native_skip_merge.py` | `59d21f3dc9affd83` | `59d21f3dc9affd83` | 未变（C5 未落） |
| `methods/prlmc/importance.py` | `94c5dc82d62841dc` | `94c5dc82d62841dc` | 未变 |
| `methods/prlmc/slot_merge.py` | `73a2b8c4c40dfa7a` | `73a2b8c4c40dfa7a` | 未变 |

> 这两个 manifest 是本次预检唯一变化的 identity 字段（14 行全部如此）。它同时是"这次跑的是哪套算子"的凭证：执行前记录 manifest，与表中目标行逐项相等才算同一算子。

---

## 2. 命令矩阵（14 行）

任务集 = v5 既有队列的 5 任务：`hotpotqa, 2wikimqa, musique, qasper, narrativeqa`（`multifieldqa_en` 不在 v5 队列，无 v5 数据）。
臂 = `SharedMerge`（v5 ckpt，treatment）vs `Selection`（同预算无合并，baseline）。
输入 = 16k / tail 2048 / seed 0 / greedy / n=200。

| # | 行名 | 臂 | budget | window→target | `--out`（相对 `/share/latent/PRLM-C`） | GPU |
|---|---|---|---|---|---|---|
| 1 | 16k__Selection_hotpotqa | Selection | 20 | 19→9 | `runs/longbench_prlmc_v5_16k_clean/16k__Selection_hotpotqa.json` | 2 |
| 2 | 16k__SharedMerge_hotpotqa | SharedMerge | 20 | 19→9 | `…/16k__SharedMerge_hotpotqa.json` | 1 |
| 3 | 16k__Selection_2wikimqa | Selection | 20 | 19→9 | `…/16k__Selection_2wikimqa.json` | 3 |
| 4 | 16k__SharedMerge_2wikimqa | SharedMerge | 20 | 19→9 | `…/16k__SharedMerge_2wikimqa.json` | 4 |
| 5 | 16k__Selection_musique | Selection | 20 | 19→9 | `…/16k__Selection_musique.json` | 3 |
| 6 | 16k__SharedMerge_musique | SharedMerge | 20 | 19→9 | `…/16k__SharedMerge_musique.json` | 4 |
| 7 | 16k__Selection_qasper | Selection | 20 | 19→9 | `…/16k__Selection_qasper.json` | 5 |
| 8 | 16k__SharedMerge_qasper | SharedMerge | 20 | 19→9 | `…/16k__SharedMerge_qasper.json` | 5 |
| 9 | 16k__Selection_narrativeqa | Selection | 20 | 19→9 | `…/16k__Selection_narrativeqa.json` | 1 |
| 10 | 16k__SharedMerge_narrativeqa | SharedMerge | 20 | 19→9 | `…/16k__SharedMerge_narrativeqa.json` | 2 |
| 11 | b10__Selection_hotpotqa | Selection | 10 | 11→5 | `…/b10__Selection_hotpotqa.json` | 1 |
| 12 | b10__SharedMerge_hotpotqa | SharedMerge | 10 | 11→5 | `…/b10__SharedMerge_hotpotqa.json` | 2 |
| 13 | b10__Selection_narrativeqa | Selection | 10 | 11→5 | `…/b10__Selection_narrativeqa.json` | 3 |
| 14 | b10__SharedMerge_narrativeqa | SharedMerge | 10 | 11→5 | `…/b10__SharedMerge_narrativeqa.json` | 4 |

- 14 条 `--out` 互不相同（dry-run 实测 `distinct outs: 14`），全部落在**新目录**，与 `runs/longbench_prlmc_v5_16k/` 无交集。
- 所有 `SharedMerge` 行必须带 `--checkpoint <ckpt> --strict-code-hash false`，**并且**按 §2.3 打语义状态标记；该组合让 `evaluation_eligible=false`，JSON 顶层写 `code_hash_verified=false` + "historical diagnosis only" warning。
- 10-block 高压用**预注册表**几何 `budget 10 → (11,5)`（`public_retention_eval.py:116`），两个基线都可执行；C8 的 13→9 属 t10，不在本矩阵（§0.6）；t10 中该行标 **BLOCKED-BY-T6**。
- 选 `hotpotqa + narrativeqa` 做高压：唯一有 v5 16k 历史数据的两条（可做同轴自检），且历史融合子集 Δ 为正。若要最大化触发覆盖率，替代为 `musique`（95.5%）。

### 2.1 可直接执行的脚本（Phase 1 = 行 1–10，Phase 2 = 行 11–14）

```bash
set -u
WT=/share/latent/PRLM-C-evalv5            # 执行基线，见 §2.2；改基线只改这一行
REPO=/share/latent/PRLM-C                 # --out 写进主工作树，便于 t11 提交
PY=/share/latent/mllm_env/bin/python
DATA=/share/latent/benchmarks/LongBench/data
CKPT=$REPO/runs/natural_merge_v5_20261018_1050/Dsup.train.pt
OUT=$REPO/runs/longbench_prlmc_v5_16k_clean
mkdir -p "$OUT"
cd "$WT"                                  # 关键：cwd 决定 import 的算子代码

run() { # gpu arm task budget extra
  local gpu=$1 arm=$2 task=$3 budget=$4; shift 4
  local tag=16k; [ "$budget" = 10 ] && tag=b10
  echo "[$(date +%H:%M:%S)] start gpu$gpu $tag $arm $task"
  $PY -u tools/longbench_prlmc_eval.py \
    --data-dir "$DATA" --out "$OUT/${tag}__${arm}_${task}.json" \
    --total-tokens 16384 --tail-tokens 2048 --cuda-device "$gpu" \
    --arm "$arm" --budget-blocks "$budget" --tasks "$task" "$@" \
    > "$OUT/logs_${tag}__${arm}_${task}.log" 2>&1
  echo "[$(date +%H:%M:%S)] done  gpu$gpu $tag $arm $task rc=$?"
}

# --- Phase 1: 16k / budget 20（先跑，过 §5 自检门再进 Phase 2） ---
( run 1 Selection narrativeqa 20; run 1 SharedMerge hotpotqa 20 --checkpoint "$CKPT" --strict-code-hash false ) &
( run 2 SharedMerge narrativeqa 20 --checkpoint "$CKPT" --strict-code-hash false; run 2 Selection hotpotqa 20 ) &
( run 3 Selection musique 20; run 3 Selection 2wikimqa 20 ) &
( run 4 SharedMerge musique 20 --checkpoint "$CKPT" --strict-code-hash false; \
  run 4 SharedMerge 2wikimqa 20 --checkpoint "$CKPT" --strict-code-hash false ) &
( run 5 Selection qasper 20; run 5 SharedMerge qasper 20 --checkpoint "$CKPT" --strict-code-hash false ) &
wait

# --- Phase 2: 10-block 高压（仅在 §5 自检通过后） ---
( run 1 Selection hotpotqa 10 ) &
( run 2 SharedMerge hotpotqa 10 --checkpoint "$CKPT" --strict-code-hash false ) &
( run 3 Selection narrativeqa 10 ) &
( run 4 SharedMerge narrativeqa 10 --checkpoint "$CKPT" --strict-code-hash false ) &
wait
```

不要在外面设 `CUDA_VISIBLE_DEVICES`——`--cuda-device` 只在未设置时 `setdefault`，外部设置会覆盖物理卡选择（v4 曾因此把权重压到繁忙卡 0 上 OOM）。

### 2.2 执行基线：已裁决 A（影响 b10 四行的语义）

| 方案 | 执行基线 | 16k 十行 | b10 四行 | 适用 |
|---|---|---|---|---|
| **A（推荐做 v5 基线）** | `fa6a22b`（v5 算子语义） | v5 原函数，可逐位复现历史 | v5 原函数 | t10 的"v5 同轴 Δ"必须用这个，否则比较的是两件事 |
| **B（按 t4 基线）** | `82d2548` | 与 A **数值等价**（≤1 次压缩，首轮不变式） | **不等价**：C1 配额从第 2 轮起生效 | 若只想验证新代码路径可跑；b10 须改标 "v5 ckpt + v7 C1 配额" |
| C（现状 HEAD） | `>= 7320135`（t5 已落） | **不再等价**（C5 tanh 作废 v5 ckpt 语义） | 不再等价 | 不是 v5 基线，是"v5 权重 × v7 算子"消融 |

一句话：**A 才能回答"v5 原来是多少"；B 能回答"新代码路径是否可跑"；C 只能回答"v5 权重被 v7 算子改成什么样"。** 三者不可混用，报告里必须写明用了哪个。

> **裁决（captain，2026-10-20）**：选 **A（`fa6a22b`）** 作为 v5 同轴参照；B（`82d2548`）仅用于验证新代码路径，其 b10 四行改标 "v5 ckpt + v7 C1 配额"；C 本阶段不用。

### 2.3 v5 ckpt 行的强制标注（t5 落地后新增）

t5（`7320135`）已把 `_group_delta` 换成 `carrier_rms * tanh(...)`，**v3/v4/v5 ckpt 的数值语义自此作废**（v6 保持只读可复现，`experiments/v7-spec-contract.md` C5.5）。因此本矩阵的 7 条 `SharedMerge` 行必须**同时**带两个标记：`--strict-code-hash false`（t2 的 manifest 默认拒绝决定了这是唯一入口）+ 语义状态标注。语义状态随基线不同，不能一概而论：

| 基线 | 16k 五行（每任务 1 条） | b10 两行 | 标注文本 |
|---|---|---|---|
| **A** `fa6a22b`（C5 未落） | **v5 原语义完好**，可逐位复现历史（§5 自检门） | v5 原语义完好 | `--strict-code-hash false`；"v5 原语义（C5 前）；evaluation_eligible=false，仅历史参照" |
| **B** `82d2548`（C5 未落、C1 已落） | 同 A（≤1 次压缩，首轮不变式） | **选块已变**（C1 配额从第 2 轮起） | 16k 同 A；b10 另加 "v5 ckpt + v7 C1 配额" |
| **C** `>= 7320135`（C5 已落） | **语义已作废** | **语义已作废** | `--strict-code-hash false`；"语义已作废、仅历史参照" |

**硬规则**：

1. 任何 v5 ckpt 行，`--strict-code-hash false` 与"仅历史参照"两个标记必须同时出现；在 C 基线上再加"语义已作废"。
2. 报告里不得把 v5 ckpt 的分数写成"当前算子的能力"；它只回答"v5 当时是多少"（且仅在 A/B 上成立）。
3. A 基线上 v5 语义**没有**作废（C5 未落），所以"逐位复现"自检门与"语义已作废"不能同时成立——这正是必须按基线分层标注的原因。

---

## 3. dry-run 预检结果

### 3.1 正向：28/28 通过（14 行 × 2 基线）

| 行组 | budget | window→target | select/ref | ckpt sha[:12] | verified | eligible |
|---|---|---|---|---|---|---|
| 16k__Selection_*（5 条） | 20 | 19→9 | 16/16 | – | `null` | `true` |
| 16k__SharedMerge_*（5 条） | 20 | 19→9 | 16/16 | `13700eca4c9b` | `false` | `false` |
| b10__Selection_*（2 条） | 10 | 11→5 | 16/16 | – | `null` | `true` |
| b10__SharedMerge_*（2 条） | 10 | 11→5 | 16/16 | `13700eca4c9b` | `false` | `false` |

- 两个基线的计划里 `total_tokens=16384`、`tail_tokens=2048`、`write_prefix_tokens=14336`、`seed=0`、`do_sample=false`、`geometry` 完全一致；**唯一差异是 `critical_source_manifest` 与 `source_manifest`**（§1 表）。
- `runs/longbench_prlmc_v5_16k_clean/` 在 28 次 dry-run 后仍不存在；用绝对 `--out` 复测同样不创建父目录。
- `Selection` 行 `code_hash_verified=null`、`evaluation_eligible=true`：无 ckpt，无需校验，属预期。

### 3.2 负例（9 条，两个基线的行为差异已标注）

| # | 输入 | `fa6a22b` | `82d2548` |
|---|---|---|---|
| N1 | `--window-blocks 13 --target-blocks 9`（budget 20） | dry-run 通过 | dry-run 通过（identity 记 13/9） |
| N2 | `--window-blocks 5 --target-blocks 9` | `ValueError: need 0 < target_blocks < window_blocks` | 同左 |
| N3 | `--window-blocks 60` | `ValueError: window_blocks 60 exceeds the physical 50-block pool` | 同左 |
| N4 | `--budget-blocks 10 --target-blocks 12` | `ValueError: … got window=11 target=12` | 同左 |
| N5 | `--budget-blocks 10 --window-blocks 20 --target-blocks 12` | `ValueError: target_blocks 12 exceeds budget_blocks 10` | 同左 |
| N6 | `--total-tokens 32768` | `ValueError: … validated for (8192, 16384) only` | 同左 |
| N7 | v5 ckpt + 默认 `--strict-code-hash true` | 不硬失败：`warnings=[…no source_manifest…]` + `evaluation_eligible=false` + stderr `WARNING: dry-run would reject this run` | 同左 |
| N8 | `stream_kwargs()` 显式 13/9 | `RuntimeError: … need a MergeStream that accepts explicit window_blocks/target_blocks` | **通过**：`({'budget_blocks': 20, 'window_blocks': 13, 'target_blocks': 9}, (13, 9))` |
| N9 | `--arm Selection --checkpoint …` | `ValueError: … drop --checkpoint` | 同左 |

### 3.3 两条必须记录的缺口

**(a) `--dry-run` 不覆盖运行时握手。** `dry_run_plan()` 只走 `effective_geometry()`，不调用 `stream_kwargs()`。在 `fa6a22b` 上 N1 通过但 N8 失败——dry-run 给出假绿灯；真跑时会在 `load_runtime()`（已加载 8B 权重、约 24 GB）之后才抛错。在 `82d2548` 上两者都通过，所以这个缺口**现在不致命但仍未修**：t10 若再加几何参数，同样会被 dry-run 放过。已裁决（captain）：该断言改由 t7 以**不加载 8B 权重**的 CPU 几何握手测试落地（tiny config 构造 manager/stream，断言实际 window/target 与触发阈值真到达 `StreamingMemory`），t10 沿用同一断言。

**(b) 任务书写的几何不变量 `0<target<window<=budget` 对 budget 10 不成立。** 预注册表是 `10 → (11,5)`：window 11 > budget 10，属设计上的瞬时超窗（`public_retention_eval.py:128-137` 注释），稳态仍 ≤10 块。`validate_geometry()` 实际执行的是 `0<target<window`、`target<=budget`、`window<=50`（物理池），**没有** `window<=budget`。矩阵 4 条 b10 行因此报 `window<=budget VIOLATED`，属预期而非缺陷；验收文案应按实际口径写。

---

## 4. 预期触发率与统计口径

### 4.1 触发分布（历史实测，供执行后对账）

16k/budget 20 下每样本压缩次数（`compression_count`）：

| 任务 | 0 次 | 1 次 | ≥2 次 | 触发率 |
|---|---|---|---|---|
| hotpotqa | 76 | 124 | 0 | 62.0% |
| narrativeqa | 49 | 151 | 0 | 75.5% |
| musique | 9 | 191 | 0 | 95.5% |
| qasper | 196 | 4 | 0 | 2.0% |
| 2wikimqa | 175 | 25 | 0 | 12.5% |

**这是"16k 两基线等价"的直接原因**：最多 1 次压缩 ⇒ 压缩时窗口内 `age` 全 0 ⇒ C1 走原路径。

高压侧（hotpotqa 实测）：b12 = `{0:29, 1:27, 2:45, 3:99}`、b8 = `{0:22, 1:15, 2:29, 3:35, 4:99}`。b10 的几何与 b12 同为 (11,5)、预算更低一档，触发点比 b12 早一块，故 **≥2 次压缩的样本预计占 72–82%**（b12 72.0%、b8 81.5% 夹逼）——这正是 b10 在 `82d2548` 上语义改变的量级。

未触发样本两臂 prediction 逐字相同，故全集 Δ = 融合子集 Δ ×（n_fused/n）。

### 4.2 统计口径与离线演练

口径（与 v4/v6 一致，可直接复用 `tools/fused_subset_report.py`）：

- 主指标 `official_qa_f1_truncated`；配对键 `(task, id, sample_index)`；两臂 n 必须相等。
- 三套子集：`full`（n=200）、`active_merge_subset`（`compression_count>0`）、`no_merge_subset`（`==0`）。
- 配对胜负：Δ>0 记 W、Δ<0 记 L、Δ=0 记 T；bootstrap 95% CI，**2000 resamples，seed 0**（`fused_subset_report.bootstrap_ci`）。
- 融合子集由 **treatment（SharedMerge）侧** 的 `compression_count>0` 定义；同时必须报告两臂触发条数是否一致，不一致即标红。

用 6 组历史 JSON 做的离线演练（未产生新数据；`/tmp/v5_clean_dryrun/stats/rehearsal.json`）：

| 演练对 | full（treat/base） | Δfull [CI95] | W/L/T | 融合 n | fused（treat/base） | Δfused [CI95] | W/L/T |
|---|---|---|---|---|---|---|---|
| v5-hotpotqa-b20 | 36.70 / 35.01 | +1.70 [-1.45,+4.83] | 19/19/162 | 124 | 38.49 / 35.76 | +2.74 [-2.70,+8.05] | 19/19/86 |
| v5-narrativeqa-b20 | 22.60 / 21.85 | +0.75 [-1.46,+2.99] | 30/26/144 | 151 | 22.63 / 21.63 | +0.99 [-1.72,+3.99] | 30/26/95 |
| v6-musique-b20 | 21.48 / 21.03 | +0.45 [-3.10,+4.09] | 25/24/151 | 191 | 21.43 / 20.96 | +0.47 [-3.32,+4.13] | 25/24/142 |
| v6-narrativeqa-b20 | 21.43 / 21.85 | -0.43 [-2.39,+1.55] | 18/25/157 | 151 | 21.07 / 21.63 | -0.57 [-3.15,+2.11] | 18/25/108 |
| v6-hotpotqa-b12 | 31.79 / 31.42 | +0.37 [-4.13,+4.94] | 31/26/143 | 171 | 30.43 / 30.00 | +0.43 [-5.00,+5.76] | 31/26/114 |
| v6-hotpotqa-b8 | 31.23 / 32.43 | -1.20 [-4.99,+2.75] | 27/25/148 | 178 | 29.55 / 30.90 | -1.35 [-5.57,+2.84] | 27/25/126 |

**口径可用性证明**：前两行与 `runs/longbench_prlmc_v5_16k/t2_final_stats.json` 逐项相等（`diff_full` 1.696/0.749、`fused_delta` 2.736/0.992、W/L/T 19/19/86 与 30/26/95、CI [-2.72,+8.05] 与 [-1.75,+3.99]）；b12/b8 行与 `experiments/v6-consolidate-16k-budget.md` 相等（+0.43 / -1.35、31/26/114 / 27/25/126）。工具输出与逐样本重算的融合 Δ 最大差 5.6e-17。

**口径限制（t10 需注意）**：

1. `fused_subset_report.py` **不校验两份 JSON 的 identity**（arm/checkpoint/geometry 都可能不同），只校验 schema 与 `status=complete`。clean 协议下应先用 t3 落盘的 `identity` 块做显式比对再统计。
2. 该工具只算融合子集；`full` 的 W/L/T 与 CI 需用同一 `bootstrap_ci()` 在全集上重算（本次演练即如此，一行代码，无需新工具）。

### 4.3 分代统计口径（t4 提醒，t10 直接沿用）

**配额是上限不是保底**：`max_old_ratio=0.45` 只限制老块**最多** 4/9 席，不保证老块一定拿到席位；tiny fixture 里 recency 主导时老块三轮仍停在 age=1。因此**不得**用"平均 age"代替分布，也不得假设 age 随轮次单调增长。

必须报告的分布（每任务 × 每臂）：

- **age 直方图**：`0 / 1 / 2 / 3+` 四档的块占比（不是均值）；
- **老块席位占比**：`selected_block_ages >= 1` 的选中块数 / `target_blocks`（9 或 5）；
- **触发轮数**：`compression_count` 的 `0/1/2/3+` 分布。

数据来源（`82d2548` 已内置，无需另算）：

- `samples[*].compression_events[*].selected_block_ages`——每次压缩选中的块的年龄，长度应等于 `target_blocks`（9 或 5）；长度不符即数据版本不对；
- `ProtectedResidualMemory` 的 compact 报告里另有 `block_age_max` 与 `block_age_histogram`（layout 路径，评测走的是 `StreamingMemory.write` + `runtime.install`，以 `compression_events` 为准）。

**注意**：`fa6a22b` 的 events **没有** `selected_block_ages` 字段（t4 才加），所以若执行基线选 A（`fa6a22b`），本项统计**取不到**，只能用 `compression_count` 分布代替，并在报告里写明缺失原因。

### 4.4 判读门（供 t10 复用）

v7 成立需同时满足：`Δfused(v7) > Δfused(v5 同轴)` 且 `CI95 下界 > 0`。注意本次 v5 复核的 ckpt 无法通过源码哈希校验（`evaluation_eligible=false`），因此 v7 侧（可验证）与 v5 侧（仅诊断）存在**证据等级不对称**，最终报告必须写明。

---

## 5. 正式执行前置条件（checklist）

1. **基线裁决**：先定 §2.2 的 A/B/C。定了之后 `WT=` 变量、manifest 期望值、b10 行标签三者一起定。
2. **GPU**：单卡 ≥26 GB 空闲（实测峰值 23.15–24.84 GB），一卡一进程（两张 24 GB 进程塞不进 32 GB 卡）。2026-10-20 12:44 实测：GPU 0 空闲 14.4 GB、GPU 1 空闲 23.4 GB（一个 9.1 GB 进程常驻）、GPU 2–9 全部 31.7 GB/100%。→ **当前 1–5 号卡无一满足，必须等卡或协商释放**。
3. **代码冻结**：`cd $WT`，跑完前不得切换 commit、不得 merge 后续 v7 改动（C5 一落地 v5 语义即失效）。开始前记录 `critical_source_manifest`，与 §1 目标列逐项相等。
4. **ckpt**：路径与 sha256 与 §1 一致；`--strict-code-hash false` 是**必须**的（v5 ckpt 无 `source_manifest`，默认 true 会拒绝）。
5. **输出目录**：`runs/longbench_prlmc_v5_16k_clean/` 必须不存在（或仅含本矩阵文件）；严禁写入 `runs/longbench_prlmc_v5_16k/`。
6. **自检硬门（Phase 1 结束后、Phase 2 之前）**：clean 的 `16k__{SharedMerge,Selection}_{hotpotqa,narrativeqa}` 与 `runs/longbench_prlmc_v5_16k/` 对应历史文件**逐样本 prediction 必须完全一致**（同 ckpt、同数据、greedy+seed 0；A/B 两种基线都成立，依据 §0.3）。不一致 → 停跑定位，不得用新数字。
7. **`--resume` 只能续同一行**：identity 含 `critical_source_manifest`，代码一变续跑即被拒（设计意图，不要绕过）。
8. **不跑 8k**：8k 前缀 ≤6144 tokens < 20 块，`compression_count` 恒 0，两臂 prediction 逐字相同，无信息量。

### 5.1 耗时估算（按历史单样本均值，n=200/行）

| 任务 | 单样本 | 单行耗时 | 16k 两臂 | b10 两臂 |
|---|---|---|---|---|
| hotpotqa | 29.5–31.6 s | ~32 min | 64 min | ~168 min（按 b12/b20=0.84 折算） |
| 2wikimqa | 11 s | ~37 min | 74 min | 未安排 |
| musique | 39.2–40.9 s | ~133 min | 266 min | 未安排 |
| qasper | 20 s | ~67 min | 134 min | 未安排 |
| narrativeqa | 44.6–46.2 s | ~153 min | 306 min | ~260 min（折算 0.84） |
| **合计** | | | **844 min ≈ 14.1 GPU·h** | **428 min ≈ 7.1 GPU·h** |

5 卡排队：**墙钟约 5.3 h**（Phase 1 关键路径 185 min + Phase 2 130 min），GPU 总消耗约 21.2 GPU·h。
Phase 1：GPU1 `Sel narra + SM hotpot`（185）；GPU2 `SM narra + Sel hotpot`（185）；GPU3 `Sel musique + Sel 2wiki`（170）；GPU4 `SM musique + SM 2wiki`（170）；GPU5 `Sel qasper + SM qasper`（134）。
Phase 2：GPU1–4 各 1 条（hotpotqa 2 条各 84 min，narrativeqa 2 条各 130 min）。

---

## 6. 尚未执行 / 待裁决

- **尚未执行任何 GPU 评测或训练**；本文件所有数字要么是 dry-run 输出，要么是历史 JSON 的离线重算。
- 已裁决 1：执行基线 = **A（`fa6a22b`）**；B（`82d2548`）仅验证新代码路径，其 b10 行改标 "v5 ckpt + v7 C1 配额"。
- 已裁决 2：`--dry-run` 的握手缺口由 **t7** 的 CPU 几何握手断言兜住（不加载 8B 权重），t10 沿用同一断言；t3 文件不再改。
- 阻塞项：GPU 空闲 ≥26 GB（§5 第 2 条）。
- 依赖项：t10 的 13→9 需 t5（tanh/稀疏门控）**且** t6 的 `StreamingMemory.write` 触发规则（C8：13/9→`used>=14`、19/9→`used>=20`）；该行标 **BLOCKED-BY-T6**，不得静默按旧触发跑。本矩阵不含 13→9，因此不被这两项阻塞。

---

## 7. t6 落地后的状态更新（2026-10-20 13:25，append-only，不改上文结论）

t6（`7c46c9a`，218 项测试全绿）已把 C8 触发真正落到 `StreamingMemory.write`：

```python
# tools/public_retention_eval.py:228-236
if self.arm not in ("Full", "NativeDefault") \
        and m.used >= self.window \
        and m.used >= compaction_trigger(self.budget_blocks, self.window_blocks) * m.block:
    self.compact(index)
```

`compaction_trigger = min(budget, window+1)`，`write` 另加合取项 `used >= window`。**§0.6 与 §3.3(a) 中"13→9 不可执行 / 静默按旧触发跑"的结论自此失效**；§6 的 t6 依赖项已满足。

实测阈值（按 t6 代码口径）：`8/9→9`、`10/11→11`、`12/11→12`、`20/19→20`、**`20/13→14`**。10/11 之所以是 11 而非 10，是因为 `used >= window` 把下限抬到 11；四档与旧规则逐格等价。

### 7.1 对本矩阵的影响：矩阵不变

- v7 标准几何（t10 口径）：budget 20 + **window 13 + target 9**（显式 `--window-blocks 13 --target-blocks 9`，触发点 14）；10-block 高压组为 budget 10 + **window 9 + target 4**（显式覆盖，**不是**表里的 11/5）。
- **A 层跑不了显式几何**：`fa6a22b` 的 `stream_kwargs` 对显式几何直接抛 `RuntimeError`；且 `StreamingMemory.__init__`（`public_retention_eval.py:142-148`）先接收调用方参数、紧接着用 `budget_geometry(budget_blocks)` 覆盖。所以 A 层只能跑 v5 自己的表内几何：**20/19/9 与 10/11/5**。
- 因此本矩阵 14 行保持原样：16k 行 19/9、b10 行 11/5。若 t10 的高压组改成 (9,4)，**A 层无法生产对应的 v5 基线**——该组只能做 v7 vs Selection 的同预算对照，或改用 B 层（须标 "v5 ckpt + v7 C1 配额"）。

### 7.2 需要 t10 决定的一件事

v7 (20/13/9) 与 v5 (20/19/9) 的差异**同时包含算子（C1/C5/C9）与几何（19→9 改为 13→9）两个因素**，直接相减无法归因。若要分离，建议在 C 层（≥`7c46c9a`）加一行 **v7 ckpt @ 19/9**（显式 `--window-blocks 19 --target-blocks 9`，触发点 20，与表等价、零额外语义风险）：

- `v7@19/9 − v5@19/9` = 纯算子效应；
- `v7@13/9 − v7@19/9` = 纯几何效应；
- 两者之和 = `v7@13/9 − v5@19/9` = t10 门控要的 Δfused 差。

该行成本约 1 条 16k hotpotqa（~32 min），只跑最有代表性的任务即可。

---

## 8. t10 几何口径裁决与 C 层预检（2026-10-20 13:35，append-only）

### 8.1 裁决：高压组不做 v5 同轴 Δ

- **A 层保留表内 (11,5)** 作"历史协议"参照，§2 矩阵 14 行不动。
- **高压组主体改为 v7(9,4) vs Selection(9,4)**，只报 v7 相对同预算 Selection 的净收益，**不参与 v5 同轴 Δ**。理由：高压组要回答的是"低预算下融合模块能否净恢复"，v7 vs Selection 直接回答；掺一个语义被 C1 污染的 v5 基线反而更脏。
- 若要 B 层的 v5(9,4) 读数可以加，但必须标 **"v5 ckpt + v7 C1 配额（每样本 2–4 次压缩，语义非原始 v5）"**，且**不得进主判定门**。

### 8.2 裁决：C 层加 v7@19/9 一行做归因拆解

`v7@19/9 − v5@19/9` = 纯算子效应（C1/C5/C9）；`v7@13/9 − v7@19/9` = 纯几何效应（19→9 改 13→9）；两者之和 = `v7@13/9 − v5@19/9` = t10 门控要的总 Δ。成本约 1 条 16k hotpotqa（~32 min）。

### 8.3 C 层（`7c46c9a`）dry-run 预检：5/5 通过

worktree `/share/latent/PRLM-C-evalv7c @ 7c46c9a`；**未加载模型、未写结果 JSON**（`runs/longbench_prlmc_v7_clean/` 未创建）。

| 行 | 臂 | budget / window / target | 触发点 | rc |
|---|---|---|---|---|
| v7geo13_Selection | Selection | 20 / 13 / 9 | 14 | 0 |
| v7geo13_SharedMerge | SharedMerge（v5 ckpt 占位） | 20 / 13 / 9 | 14 | 0 |
| v7geo19_Selection | Selection | 20 / 19 / 9 | 20 | 0 |
| v7hp94_Selection | Selection | 10 / 9 / 4 | 10 | 0 |
| v7hp94_SharedMerge | SharedMerge（v5 ckpt 占位） | 10 / 9 / 4 | 10 | 0 |

`stream_kwargs()` 三组显式几何全部通过（`fa6a22b` 上会抛 `RuntimeError`，§3.3(a) 的假绿灯缺口在 C 层已消失）。`compaction_trigger` 网格实测：

```
budget= 8 window= 9 -> term= 8  effective= 9
budget=10 window=11 -> term=10  effective=11
budget=12 window=11 -> term=12  effective=12
budget=20 window=19 -> term=20  effective=20
budget=20 window=13 -> term=14  effective=14
budget=10 window= 9 -> term=10  effective=10
```

identity 关键字段（`v7geo13_SharedMerge`）：`arm=SharedMerge`、`budget_blocks=20`、`window_blocks=13`、`target_blocks=9`、`total_tokens=16384`、`tail_tokens=2048`、`select_layer=16`、`reference_layer=16`、`code_hash_verified=false`、`evaluation_eligible=false`；`critical_source_manifest` 中 `gated_slot_merge.py=403dfe6c…`、`native_skip_merge.py=493f6832…`（C5 已落，v5 语义作废的代码凭证），`consolidate.py=c16e5837…`、`layout.py=35e358c4…`（C1/C2）。stderr 两条 WARNING 照旧（ckpt 无 source_manifest + 未校验哈希），属预期。

**注意**：此处 checkpoint 用 v5 ckpt 占位（v7 ckpt 尚未训练），所以 identity 里的 ckpt sha 是 v5 的；本次验证的是**几何与触发口径**，不是 v7 权重。真正开跑时须替换成 v7 ckpt 并重跑 dry-run 核对 sha。

### 8.4 顺带发现的一处文案缺陷（建议 t7 修）

`tools/longbench_prlmc_eval.py:346` 把 `protocol.compaction` 硬编码成 `"{window}->{target} block merge before the write that would exceed the budget"`。对表内四档仍准确，但对 **20/13/9 已不准确**：C8 下实际在 `used>=14` 就压，远早于"预算将被突破"。该字符串会写进每一份 v7 结果 JSON 的 protocol 块，建议改为按 `compaction_trigger()` 生成，例如 `"13->9 block merge at used>=14 (min(budget, window+1), and used>=window)"`。
