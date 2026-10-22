# PRLM-C v7 评测 dry-run：命令矩阵、统计口径与判定门

**状态：尚未执行任何 GPU 评测。** 本文只有 dry-run 计划、历史数据的离线重算与 CPU 契约测试结果；没有加载 8B 权重、没有写任何结果 JSON、没有占用 GPU。

日期：2026-10-20 | 预检产物：`/tmp/t10_dryrun/`（`plans-c/` 16 份 C 栏计划、`stats/rehearsal.json` 统计演练）
C 栏预检 worktree：`/share/latent/PRLM-C-evalv7c @ 22cf7a9`

---

## 0. 结论摘要

1. **三层矩阵共 34 行，dry-run 全部通过**：A 栏 14 行（t8，`fa6a22b`）、B 栏 14 行（t8，`82d2548`）、C 栏 16 行（本次，`22cf7a9`），合计 44 次 dry-run，`rc=0`，`runs/longbench_prlmc_v7_eval/` 与 `runs/longbench_prlmc_v5_16k_clean/` 均未被创建。
2. **v7 几何已可用**：`20/13/9` 触发点 **14**、`10/9/4` 触发点 **10**、归因行 `20/19/9` 触发点 **20**，与 t6 的 `compaction_trigger` 逐格一致；`protocol.compaction` 文案已由 t7 改为动态生成（`13->9 block merge at used>=14`），不再是旧文案。
3. **v7 ckpt 尚未训练**，C 栏 dry-run 用 v5 ckpt 顶位，因此 identity 里 `architecture=grouped_gated_sparse_slot_v5`、`code_hash_verified=false`、`evaluation_eligible=false`。**正式开跑前必须用真 v7 ckpt 重跑 dry-run**，逐字段核对 §5.1 的清单；不一致不得开跑。
4. **判定门改成"差值的 bootstrap"**：不是比较两个点估计，而是对每个样本算 `d_v7 − d_v5` 再取 95% CI 下界 > 0（§5.2）。点估计相减会把两个各自的噪声当成一个信号。
5. **统计口径演练复现历史**：6 组历史对与 `t2_final_stats.json`、`experiments/v6-consolidate-16k-budget.md` 逐位一致；age 通路用合成 events 验证（历史数据无 `selected_block_ages`，回退逻辑已跑通）。
6. **耗时**：A 栏 21.2 GPU·h、C 栏 22.3 GPU·h，合计 **43.5 GPU·h**；5 卡排期墙钟约 **9 h**（不含 v7 训练）。每卡需 ≥26 GB 空闲，训练需 27–30 GB，**训练与评测不能同卡**。

---

## 1. 三层基线与代码指纹

| 层 | commit | 用途 | v5 ckpt 语义 | 触发规则 |
|---|---|---|---|---|
| **A** | `fa6a22b`（t2+t3） | 回答"v5 原来是多少" | **完好**，可逐位复现历史 | 旧式 `used+block>budget`（与表内四档等价） |
| **B** | `82d2548`（t4） | 仅验证新代码路径 | 16k 行等价；b10 行被 C1 改变 | 同上（C8 未落） |
| **C** | `22cf7a9`（t1/t5/t6/t7/t9 全落） | v7 正式评测 | **已作废**（C5 tanh） | `used>=min(budget,window+1)` 且 `used>=window` |

C 栏 `critical_source_manifest`（前 8 位）：`consolidate.py=c16e5837`、`layout.py=35e358c4`、`gated_slot_merge.py=403dfe6c`、`native_skip_merge.py=493f6832`、`importance.py=94c5dc82`、`slot_merge.py=73a2b8c4`。
A 栏：`consolidate.py=467ea2d2`、`layout.py=92138693`、`gated_slot_merge.py=c95e9d72`、`native_skip_merge.py=59d21f3d`，其余同。

触发点实测（C 栏 `compaction_trigger`）：

```
budget= 8 window= 9 -> 9      budget=20 window=19 -> 20
budget=10 window=11 -> 11     budget=20 window=13 -> 14
budget=12 window=11 -> 12     budget=10 window= 9 -> 10
```

---

## 2. 命令矩阵

### 2.1 A 栏（14 行）：v5 同轴参照

`--out` 落 `runs/longbench_prlmc_v5_16k_clean/`，**绝不写** `runs/longbench_prlmc_v5_16k/`。

| # | 行名 | 臂 | budget/window/target | 触发点 | GPU |
|---|---|---|---|---|---|
| A1 | 16k__Selection_hotpotqa | Selection | 20/19/9 | 20 | 2 |
| A2 | 16k__SharedMerge_hotpotqa | SharedMerge | 20/19/9 | 20 | 1 |
| A3 | 16k__Selection_2wikimqa | Selection | 20/19/9 | 20 | 3 |
| A4 | 16k__SharedMerge_2wikimqa | SharedMerge | 20/19/9 | 20 | 4 |
| A5 | 16k__Selection_musique | Selection | 20/19/9 | 20 | 3 |
| A6 | 16k__SharedMerge_musique | SharedMerge | 20/19/9 | 20 | 4 |
| A7 | 16k__Selection_qasper | Selection | 20/19/9 | 20 | 5 |
| A8 | 16k__SharedMerge_qasper | SharedMerge | 20/19/9 | 20 | 5 |
| A9 | 16k__Selection_narrativeqa | Selection | 20/19/9 | 20 | 1 |
| A10 | 16k__SharedMerge_narrativeqa | SharedMerge | 20/19/9 | 20 | 2 |
| A11 | b10__Selection_hotpotqa | Selection | 10/11/5 | 11 | 1 |
| A12 | b10__SharedMerge_hotpotqa | SharedMerge | 10/11/5 | 11 | 2 |
| A13 | b10__Selection_narrativeqa | Selection | 10/11/5 | 11 | 3 |
| A14 | b10__SharedMerge_narrativeqa | SharedMerge | 10/11/5 | 11 | 4 |

标注：SharedMerge 行 `--checkpoint <v5 ckpt> --strict-code-hash false`，语义标记 = "v5 原语义（C5 前）；evaluation_eligible=false，仅历史参照"。
**b10 高压组在 A 栏只是历史协议参照**，不进 v7 门控（§5）。

### 2.2 B 栏（4 行可执行）：仅验证新代码路径

只保留 b10 四行；16k 十行与 A 栏**数值等价**（每样本 ≤1 次压缩，首轮 age 全 0），不重复跑。

| # | 行名 | 臂 | budget/window/target | 触发点 | 标注 |
|---|---|---|---|---|---|
| B1 | b10__Selection_hotpotqa | Selection | 10/11/5 | 11 | — |
| B2 | b10__SharedMerge_hotpotqa | SharedMerge | 10/11/5 | 11 | **"v5 ckpt + v7 C1 配额"** |
| B3 | b10__Selection_narrativeqa | Selection | 10/11/5 | 11 | — |
| B4 | b10__SharedMerge_narrativeqa | SharedMerge | 10/11/5 | 11 | **"v5 ckpt + v7 C1 配额"** |

B 栏的 `SharedMerge` 成绩**不得**写成 v5 原始成绩（C1 配额从第 2 轮起生效，每样本压缩 2–4 次）。
**建议**：B 栏只做 dry-run（已完成），不占 GPU——它的存在意义是证明新代码路径可跑，这一点 44 次 dry-run 已经证明。

### 2.3 C 栏（16 行）：v7 正式矩阵

`--out` 落 `runs/longbench_prlmc_v7_eval/`。

**C1 标准几何 20/13/9（触发 14）**：

| # | 行名 | 臂 | GPU |
|---|---|---|---|
| C1 | std__Selection_hotpotqa | Selection | 1 |
| C2 | std__SharedMerge_hotpotqa | SharedMerge | 1 |
| C3 | std__Selection_2wikimqa | Selection | 2 |
| C4 | std__SharedMerge_2wikimqa | SharedMerge | 2 |
| C5 | std__Selection_musique | Selection | 3 |
| C6 | std__SharedMerge_musique | SharedMerge | 3 |
| C7 | std__Selection_qasper | Selection | 4 |
| C8 | std__SharedMerge_qasper | SharedMerge | 4 |
| C9 | std__Selection_narrativeqa | Selection | 5 |
| C10 | std__SharedMerge_narrativeqa | SharedMerge | 5 |

**C2 高压组 10/9/4（触发 10）——只做 v7 vs Selection，不参与 v5 同轴 Δ**：

| # | 行名 | 臂 | GPU |
|---|---|---|---|
| C11 | hp94__Selection_hotpotqa | Selection | 1 |
| C12 | hp94__SharedMerge_hotpotqa | SharedMerge | 1 |
| C13 | hp94__Selection_narrativeqa | Selection | 2 |
| C14 | hp94__SharedMerge_narrativeqa | SharedMerge | 2 |

**C3 归因行 20/19/9（触发 20）——两条都要，因为 Selection 也压缩，几何一变选块就变**：

| # | 行名 | 臂 | GPU |
|---|---|---|---|
| C15 | attr199__Selection_hotpotqa | Selection | 3 |
| C16 | attr199__SharedMerge_hotpotqa | SharedMerge | 3 |

C 栏所有行：`--arm <arm> --budget-blocks <b> --window-blocks <w> --target-blocks <t>`，SharedMerge 另加 `--checkpoint <v7 ckpt>`（**默认 strict-code-hash=true，不要加 false**——v7 ckpt 自带 `source_manifest`，加了 false 反而丢掉可发表资格）。

### 2.4 可执行脚本

```bash
set -u
PY=/share/latent/mllm_env/bin/python
DATA=/share/latent/benchmarks/LongBench/data
REPO=/share/latent/PRLM-C
OUT=$REPO/runs/longbench_prlmc_v7_eval
CKPT_V7=$REPO/runs/natural_merge_v7_<date>/DsupV7.train.pt     # 训练产出后替换
mkdir -p "$OUT"

run() { # gpu arm task budget window target tag
  local gpu=$1 arm=$2 task=$3 b=$4 w=$5 t=$6 tag=$7
  local extra=""
  [ "$arm" = "SharedMerge" ] && extra="--checkpoint $CKPT_V7"
  echo "[$(date +%H:%M:%S)] $tag $arm $task gpu$gpu b$b w$w t$t"
  # shellcheck disable=SC2086
  $PY -u tools/longbench_prlmc_eval.py --data-dir "$DATA" \
    --out "$OUT/${tag}__${arm}_${task}.json" --total-tokens 16384 --tail-tokens 2048 \
    --cuda-device "$gpu" --arm "$arm" --budget-blocks "$b" \
    --window-blocks "$w" --target-blocks "$t" --tasks "$task" $extra \
    > "$OUT/logs_${tag}__${arm}_${task}.log" 2>&1
  echo "[$(date +%H:%M:%S)] done $tag $arm $task rc=$?"
}

cd /share/latent/PRLM-C-evalv7c    # C 栏必须在这个 worktree 跑

# --- Phase C1: 标准几何 20/13/9 ---
( run 1 Selection hotpotqa    20 13 9 std; run 1 SharedMerge hotpotqa    20 13 9 std ) &
( run 2 Selection 2wikimqa    20 13 9 std; run 2 SharedMerge 2wikimqa    20 13 9 std ) &
( run 3 Selection musique     20 13 9 std; run 3 SharedMerge musique     20 13 9 std ) &
( run 4 Selection qasper      20 13 9 std; run 4 SharedMerge qasper      20 13 9 std ) &
( run 5 Selection narrativeqa 20 13 9 std; run 5 SharedMerge narrativeqa 20 13 9 std ) &
wait

# --- Phase C2: 高压 10/9/4 ---
( run 1 Selection hotpotqa    10 9 4 hp94; run 1 SharedMerge hotpotqa    10 9 4 hp94 ) &
( run 2 Selection narrativeqa 10 9 4 hp94; run 2 SharedMerge narrativeqa 10 9 4 hp94 ) &
wait

# --- Phase C3: 归因 20/19/9 ---
( run 3 Selection hotpotqa 20 19 9 attr199; run 3 SharedMerge hotpotqa 20 19 9 attr199 ) &
wait
```

A 栏脚本见 `experiments/v7-v5-clean-protocol.md` §2.1（`WT=/share/latent/PRLM-C-evalv5`），只需把 `CKPT` 指向 v5 ckpt。

---

## 3. dry-run 预检结果

| 层 | worktree | 行数 | rc=0 | 目标目录创建？ |
|---|---|---|---|---|
| A | `PRLM-C-evalv5 @ fa6a22b` | 14 | 14 | 否 |
| B | `PRLM-C-evalv7 @ 82d2548` | 14（其中 4 行可执行） | 14 | 否 |
| C | `PRLM-C-evalv7c @ 22cf7a9` | 16 | 16 | 否 |

C 栏 identity 关键字段（`std__SharedMerge_hotpotqa`，**v5 ckpt 顶位**）：

```
geometry : budget 20 / window 13 / target 9 / total_blocks 50
identity : arm=SharedMerge  architecture=grouped_gated_sparse_slot_v5  variant=Dsup
           total_tokens=16384  tail_tokens=2048  select_layer=16  reference_layer=16
           code_hash_verified=false   evaluation_eligible=false
protocol : compaction = "13->9 block merge at used>=14 (trigger=min(budget,window+1))"
```

几何握手（t7 的 `tools/test_geometry_handshake.py`，C 栏）：**6/6 OK**——13 块不压、第 14 块首次压缩；默认 19/9 仍在 20 块；8/10/12/20 四档首次压缩 9/11/12/20 逐格一致；dry-run 在打印前调用 `stream_kwargs`，不再假绿灯。

`protocol.compaction` 三条实测文案：`13->9 … at used>=14`、`9->4 … at used>=10`、`19->9 … at used>=20`。

---

## 4. 统计口径

### 4.1 指标与子集

| 项 | 定义 |
|---|---|
| 主指标 | `official_qa_f1_truncated`（与 v4/v5/v6 表一致） |
| 配对键 | `(task, id, sample_index)`；两臂 n 必须相等（各 200） |
| 全集 | `full`，n=200 |
| 融合子集 | `active_merge_subset` = treatment 侧 `compression_count>0` |
| 未触发子集 | `no_merge_subset` = `compression_count==0`（两臂 prediction 应逐字相同） |
| 配对胜负 | Δ>0 记 W、Δ<0 记 L、Δ=0 记 T |
| CI | `bootstrap_ci(deltas, resamples=2000, seed=0)`，95% |

### 4.2 age 分布与老块席位（t4 口径）

**配额是上限不是保底**，所以不报均值、只报分布：

- **age 直方图**：`selected_block_ages` 的 `0 / 1 / 2 / 3+` 四档块数与占比；
- **老块席位占比**：`selected_block_ages >= 1` 的块数 / `target_blocks`（9 或 4）；
- **触发轮数分布**：`compression_count` 的 `0/1/2/3+` 占比。

数据来源 `samples[*].compression_events[*].selected_block_ages`（t4 起写入，长度必须等于 `target_blocks`，不等即数据版本不对）。
**A/B 栏数据没有该字段**（早于 t4），必须退回 `compression_count` 分布并注明缺失——不得用均值伪造。

### 4.3 离线演练（无新数据）

6 组历史对（`/tmp/t10_dryrun/stats/rehearsal.json`）：

| 演练对 | full（treat/base） | Δfull [CI95] | W/L/T | 融合 n | fused Δ [CI95] | W/L/T | 触发（treat/base） |
|---|---|---|---|---|---|---|---|
| v5-hotpotqa-b20 | 36.70 / 35.01 | +1.70 [-1.45,+4.83] | 19/19/162 | 124 | +2.74 [-2.70,+8.05] | 19/19/86 | 124/124 |
| v5-narrativeqa-b20 | 22.60 / 21.85 | +0.75 [-1.46,+2.99] | 30/26/144 | 151 | +0.99 [-1.72,+3.99] | 30/26/95 | 151/151 |
| v6-musique-b20 | 21.48 / 21.03 | +0.45 [-3.10,+4.09] | 25/24/151 | 191 | +0.47 [-3.32,+4.13] | 25/24/142 | 191/191 |
| v6-narrativeqa-b20 | 21.43 / 21.85 | -0.43 [-2.39,+1.55] | 18/25/157 | 151 | -0.57 [-3.15,+2.11] | 18/25/108 | 151/151 |
| v6-hotpotqa-b12 | 31.79 / 31.42 | +0.37 [-4.13,+4.94] | 31/26/143 | 171 | +0.43 [-5.00,+5.76] | 31/26/114 | 171/171 |
| v6-hotpotqa-b8 | 31.23 / 32.43 | -1.20 [-4.99,+2.75] | 27/25/148 | 178 | -1.35 [-5.57,+2.84] | 27/25/126 | 178/178 |

与 `t2_final_stats.json`（前两行）和 `experiments/v6-consolidate-16k-budget.md`（b12/b8 行）逐位一致，口径可用。
两臂触发条数全相等（124/124、151/151…）——若 v7 出现不等，说明两臂写入了不同内容，先查 identity 再统计。

age 通路演练：历史数据 `with_ages=0` → 回退逻辑正确报"字段缺失 + compression_count 分布"；合成 events（`[0,0,1,2]` + `[3,0,0,0]`）得 `seats=8, old_seats=3, old_seat_share=0.375, histogram 0:5(0.625) 1:1 2:1 3:1`，数学正确。

---

## 5. 判定门

### 5.1 开跑前置（硬）

1. **v7 ckpt 真实存在**，且 dry-run 的 identity 同时满足：
   `architecture=grouped_gated_sparse_slot_v7`、`variant=DsupV7`、`code_hash_verified=true`、`evaluation_eligible=true`、`critical_source_manifest` 六项与 §1 的 C 栏值相等、`window_blocks=13`/`target_blocks=9`、`protocol.compaction` 含 `used>=14`。
   **任何一项不符 → 停，不得开跑。**（当前顶位 ckpt 的 `architecture` 是 v5、`eligible=false`，正是这一条要挡的情况。）
2. C 栏全部跑在 `22cf7a9`（或更新的、manifest 已核对的 commit），A 栏跑在 `fa6a22b`，两者不得混用 worktree。
3. A 栏 Phase 1 完成并通过**自检门**（clean 16k hotpotqa/narrativeqa 与历史 prediction 逐字一致）后，才继续花 GPU。

### 5.2 门控定义

记 `d_x` 为某行的逐样本得分，`Δ` 为其均值，配对键 `(task,id,sample_index)`。

- **G1 主判定（16k 标准几何）**：对每个样本算 `e_i = (v7@13/9 − Selection@13/9)_i − (v5@19/9 − Selection@19/9)_i`，取 2000 次 bootstrap 的 95% CI。**通过条件：`mean(e) > 0` 且 CI 下界 > 0。**
  注意是**差值再做 CI**，不是 `Δ_v7 > Δ_v5` 的点估计比较——后者会把两个各自 ±4 的噪声当成一个信号。
- **G2 归因**：`Δop = Δ(v7@19/9) − Δ(v5@19/9)`（纯算子：C1+C5+C9）、`Δgeo = Δ(v7@13/9) − Δ(v7@19/9)`（纯几何：19→9 改 13→9）、`Δtotal = Δop + Δgeo`。三者各报点估计与 CI，并注明 `Δtotal` 必须等于 `Δ(v7@13/9) − Δ(v5@19/9)`（恒等式自检）。
- **G3 高压组**：`Δfused(v7@10/9/4) vs Selection@10/9/4 > 0` 且 CI 下界 > 0。**单独报告，不并入 G1**；A 栏 b10（11/5）只作历史协议参照，不参与。
- **G4 完整性**：v7 行 `evaluation_eligible=true`；v5 行 `evaluation_eligible=false` 且报告标"仅历史参照"；两臂 `compression_count>0` 的条数一致；`selected_block_ages` 长度 == `target_blocks`。
- **G5 几何自检**：每份 v7 JSON 的 `protocol.compaction` 触发点与 §1 表格一致（14/10/20），`geometry` 三字段与命令行一致。

### 5.3 报告必须写明的三条不对称

1. v5 侧 `evaluation_eligible=false`（无 manifest），v7 侧可验证——证据等级不同，不能并列成"同等可信"。
2. A 栏 b10 与 C 栏 hp94 是**不同几何**（11/5 vs 9/4），不可直接相减。
3. v6 若作对照行，其 ckpt 同样无 `source_manifest`（只有 v4/v7 trainer 写），只能 `strict_code_hash=False` 且 `evaluation_eligible=false`（t9 实测）。

---

## 6. 耗时与排期

| 栏 | 行数 | GPU·h | 说明 |
|---|---|---|---|
| A | 14 | 21.2 | 16k 844 min + b10 428 min（t8 实测均值） |
| B | 4（建议不跑） | 0 | dry-run 已证明代码路径 |
| C1 std | 10 | 14.1 | 与 v5 同任务同耗时，v7 算子开销可忽略 |
| C2 hp94 | 4 | 7.1 | hotpotqa 84 min×2 + narrativeqa 130 min×2（±10%，触发点 10 比 11 早） |
| C3 attr | 2 | 1.1 | hotpotqa 32 min×2 |
| **合计** | **30** | **43.5** | |

5 卡排期（每卡 ≥26 GB 空闲）：

- **Phase 0**：v7 训练（27–30 GB，单卡独占，4.5–5.5 h）——**与评测不能同卡**。
- **Phase 1**：A 栏 16k 10 行（844 min / 5 卡 ≈ 185 min 墙钟）+ 自检门。
- **Phase 2**：A 栏 b10 4 行 + C1 std 10 行（1272 min / 5 卡 ≈ 254 min）。
- **Phase 3**：C2 hp94 4 行 + C3 attr 2 行（492 min / 5 卡 ≈ 100 min）。
- **墙钟合计 ≈ 9 h**（不含训练）。

---

## 7. 尚未执行

- 未加载 8B 权重，未写任何结果 JSON，未占用 GPU。
- v7 ckpt 未训练；C 栏 identity 是 v5 ckpt 顶位，**不能作为开跑依据**（§5.1）。
- 统计口径的 age 部分只做了合成验证，真实分布要等 v7 数据。
- 待办：v7 训练完成后按 §5.1 重跑 C 栏 dry-run 并核对六项 manifest。
