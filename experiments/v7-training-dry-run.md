# v7 训练 dry-run 记录（t9）

**结论：训练尚未启动。** 本记录只覆盖计划生成与校验、CPU 上的 step-0 梯度实测、checkpoint 契约验证。
全程 `CUDA_VISIBLE_DEVICES=""`，未启动任何 GPU kernel，未写 `runs/natural_merge_v7_*`。
机器可读版本：`runs/dryrun_v7_20261020.json`。

## 1. 计划

| 项 | 值 |
|---|---|
| 文件 | `data/natural_merge_v7/train.json`（11,800,079 B） |
| schema | `natural_merge_v7_training_plan/1` |
| `content_sha256` | `9a0102ea…28fc7c5`（= `digest(plan − content_sha256)`，已核） |
| 文件 sha256 | `fb74e756…b7828d` |
| 源 v4 计划 | `data/natural_merge_v4/train_v5.json`，文件 sha `cdf9de76…4cdc489` 与 `original_v4_file_sha256` 一致 |
| 变体 | `DsupV7`、`DnativeV7` |
| 几何 | budget 20 / window 13 / target 9 |
| recovery_gate | `ce_full_max 2.5, gap_min .25, gap_cap 2.0, scale 3.0, background .25` |
| 规模 | 8 episode × 39 context = 312 context / 216 article；64 个 QA 事件（32 single + 32 recurrent）；1402 unique QA 占 1600 slot |
| 递归事件位置 | `before_write ∈ {14, 18}`（每个恰 14 个常驻块 = 13 window + 1 tail） |
| 单窗起点 | `{0, 1, 20}` |
| 计划更新数 | 64 |

校验全部通过：`validate_v7_plan` OK；内嵌 `original_v4_plan` 通过 `validate_v4_plan`；`qa_events == make_schedule_v7(episodes)`；`split_audit` 与源 v4 计划逐字段相同；训练集 312 context / 216 article 与 `observed_*`、`development_articles`、`formal_articles`、`excluded_previous_distractor_articles` 全部不相交；SQuAD 两个原始文件与 `published_distractors_sha256` 均实测匹配；tokenizer 与 `/dev/shm/s1d_reader_fp16` manifest 一致（`eos_token_id=128001`）。

**已知隐患（须 t8/t10 注意）**：`plan["base_plan"]["protocol"]` 是继承的 v4 审计记录，仍写着 `window_blocks 19 / target_blocks 9`；v7 生效几何是 `plan["training"]["geometry"] = 20/13/9`。评测 CLI 不读该字段（`longbench_prlmc_eval.py:202-208` 从 CLI 取几何），但**必须显式传 `--window-blocks 13 --target-blocks 9`**，否则以 19/9 评测，与训练窗口不一致（C8 明令禁止）。

另：`data/natural_merge_v4/train.json` 是 v5 之前的旧计划（架构 `*_v4`），已通不过当前 `validate_v4_plan`；v4 源必须是 `train_v5.json`。

## 2. step-0 梯度实测（CPU，真实维度）

实例化 `build_merge(<arch>, 32, 4096, 256)`（32 层 / hidden 4096 / block 256 / rank 32 / 8 组，**2,392,096** 个融合参数），`source` 形状 `[32, 13·256, 4096]`，`target=9·256`，`carrier_indices=0..8`，fp32、seed 0，从零初始化状态做一次前向+反向。两个替身目标：残差 MSE（对 σ=0.02 的目标），以及"冻结单位尺度线性读出 CE"（对全部 2304 个融合槽）。

| 配置 | 目标 | `up_weight` | `native_gain` | 总范数 |
|---|---|---|---|---|
| v7 slot（tanh 软界） | 残差 MSE | 2.56e-6 | — | 2.56e-6 |
| v7 slot（tanh 软界） | 读出 CE | **8.94e-4** | — | 8.94e-4 |
| v7 native（tanh 软界） | 残差 MSE | 4.34e-6 | 3.39e-3 | 3.39e-3 |
| **v7 native（tanh 软界）** | **读出 CE** | **1.94e-3** | **6.31e-4** | 2.04e-3 |
| v7 native，仅外层退回旧式强制单位 RMS | 读出 CE | 1.94 | 0.631 | 2.04 |
| v7 native，内外两层都退回旧式 | 读出 CE | 1941.7 | 0.631 | 1941.7 |
| v7 slot，退回旧式 | 读出 CE | 0.894 | — | 0.894 |

`down_weight`、`slot_gate.*`、`rel_gate.*`、`q.weight`、`k.weight` 在 step 0 的梯度**精确为 0**：`up_weight=0` 使 `delta=0`，`alpha·delta` 与 `up^T·…` 两条路径都被乘零。这与真实训练首个 update 的日志一致（v6 run 里这些项也是 0.0）。

真实 8B reader 桥接的对照（现成 run，未新跑 GPU）：

| run | 代码 | step-0 `up_weight` |
|---|---|---|
| `runs/natural_merge_v5_20261018_1050` | t5 之前（强制单位 RMS） | **1.9658** |
| `runs/natural_merge_v6_20261019_0812` | v6 单边软界 | **4.3365e-3** |

替身与真实链的量级吻合（替身单界 1.94 vs 真实 v5 1.966，差 1.3%），所以替身的绝对量级可信。

## 3. 对 C7 "step-0 native_gain 梯度 1.4e5" 的裁决：**否证（refuted）**

- **机制成立、量级不成立。** 旧式 `rsqrt(mean(combined²)+1e-6)` 在零点确有 `1e3` 增益，实测 ×1000.0–1000.1：`native_gain` 6.31e-4 → 0.6307（只退回外层界），`up_weight` 1.94e-3 → 1.94。两层界叠加则是 ×1e6：`up_weight` → 1941.7。
- **但没有任何一条路径接近 1.4e5。** v7 上 `native_gain` = 6.31e-4、`up_weight` = 1.94e-3；即便按 C7 假定的旧代码，`native_gain` 也只有 0.631，与 1.4e5 差 5.4 个数量级。真实 reader 链上唯一可查的 step-0 数字是 `up_weight` 1.97（v5）与 4.34e-3（v6）。
- **五位数量级只可能来自"旧 native 模块的双重重归一化"**（1e6 增益，作用在 `up_weight` 上），而不是 `native_gain`（1e3 增益）。`native_skip_merge.py` 里 `adapted` 已经被界一次、`combined` 再被界一次，这正是实测 1941.7 的来源；换成真实 19 块窗口、14 个 QA 的事件后放大到 ~1e4 是可能的，1e5 也在同一量级带内。
- v7 在每个点都是 tanh（零点增益 1，两层相乘仍为 1），所以该数值**不可能在 v7 复现**。
- 需要的措辞修改（t11 落笔，t9 不动他人文件）：C7 从"1.4e5 待验证"改为"t9 否证：native_gain step-0 6.31e-4、up_weight 1.94e-3（CPU 替身）；reader 桥接 v5 1.97 / v6 4.34e-3；1e3 零点增益在 t5 之前真实存在，已由 tanh 软界移除"。
- 若要在 GPU 上把绝对值钉死：`--smoke-updates 1` 跑一次，读 `updates[0].gradient_norms_before_clip` 与 `grad_norm_before_clip`。

## 4. checkpoint 契约

保存端**执行了 trainer 源码里真实的 `checkpoint=dict(...)` 表达式**（AST 取出后在桩命名空间 eval），得到的 dict 键含 `source_manifest`（42 条，覆盖全部 6 个 `critical_manifest()` 文件）、`training == TRAIN_V7`、`geometry = 20/13/9`、`recovery_gate`、`training_sources`、`train_content_sha256`、`final_state_sha256`。

`/tmp/v7dryrun/ckpt/` 下 `torch.save` → `torch.load(weights_only=True)` → `validate_merge_checkpoint` 全绿，`load_merge_state` 后 `state_digest` 与保存前相等；DsupV7 9,584,059 B、DnativeV7 9,584,352 B。

互斥矩阵（全部实测）：

| checkpoint | 架构 | 结果 |
|---|---|---|
| v7 slot | `*_slot_v7` | accepted |
| v7 slot | `*_slot_v6` / `*_slot_v5` / `*_native_v7` | rejected（architecture/training/budget mismatch） |
| v7 native | `*_native_v7` | accepted |
| v7 native | `*_slot_v7` | rejected |
| 真实 v5 ckpt | `*_slot_v5` 与 `*_slot_v7` | rejected（no `source_manifest`） |
| 真实 v6 ckpt | `*_slot_v6` 与 `*_slot_v7` | rejected（no `source_manifest`） |

**发现**：只有 v4 与 v7 trainer 写 `source_manifest`；真实 v6 checkpoint 没有，因此在严格校验下**连它自己的架构也加载不了**。v6 基线只能用 `strict_code_hash=False` 加载，而那样 `evaluation_eligible=False`。若最终报告要放 v6 对照行，需按此措辞说明。

## 5. 显存与耗时估计

- 依据：实测 v6 run（19/9，真实 8B reader）`peak_reserved` **30.0 GB**、`peak_allocated` **28.7 GB**，64 update 用 **2.42 h**（136 s/update）；同卡上另有一次 v6 run OOM 崩掉（`runs/natural_merge_v6_20261018_1718_crashed_oom`）。
- v7 把压缩窗口从 19 块降到 13 块（4864 → 3328 槽，窗口相关激活 −32%），但 reader 权重 16.1 GB（fp16）+ 12800 槽池 3.36 GB 是主体，故峰值基本不动：**估计 27–30 GB**。32 GB 卡仍然紧张。
- 耗时：单变体 **2.2–2.6 h**，两变体串行 **4.5–5.5 h**。

## 6. 正式训练命令（未执行）

```bash
# 单次冒烟，钉死 step-0 绝对值
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=tools:. /share/latent/mllm_env/bin/python tools/natural_merge_v7_train.py \
  --plan data/natural_merge_v7/train.json --variant DnativeV7 --cuda-device 0 --smoke-updates 1 \
  --out /tmp/v7smoke/DnativeV7.train.json --checkpoint-out /tmp/v7smoke/DnativeV7.train.pt

# 正式训练
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=tools:. /share/latent/mllm_env/bin/python tools/natural_merge_v7_train.py \
  --plan data/natural_merge_v7/train.json --variant DsupV7 --cuda-device 0 \
  --out runs/natural_merge_v7_<ts>/DsupV7.train.json --checkpoint-out runs/natural_merge_v7_<ts>/DsupV7.train.pt
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=tools:. /share/latent/mllm_env/bin/python tools/natural_merge_v7_train.py \
  --plan data/natural_merge_v7/train.json --variant DnativeV7 --cuda-device 0 \
  --out runs/natural_merge_v7_<ts>/DnativeV7.train.json --checkpoint-out runs/natural_merge_v7_<ts>/DnativeV7.train.pt
```

评测侧固定 `--budget-blocks 20 --window-blocks 13 --target-blocks 9`。

## 7. 复现命令

脚本已随本记录提交在 `experiments/v7-dryrun/`（输出写到 `/tmp`）：

```bash
cd /share/latent/PRLM-C
CUDA_VISIBLE_DEVICES="" PYTHONPATH=tools:. /share/latent/mllm_env/bin/python tools/prepare_natural_merge_v7.py \
  --v4-train-plan data/natural_merge_v4/train_v5.json --out data/natural_merge_v7/train.json
CUDA_VISIBLE_DEVICES="" PYTHONPATH=tools:. /share/latent/mllm_env/bin/python experiments/v7-dryrun/verify_plan.py
CUDA_VISIBLE_DEVICES="" PYTHONPATH=tools:. /share/latent/mllm_env/bin/python experiments/v7-dryrun/verify_ckpt.py
CUDA_VISIBLE_DEVICES="" PYTHONPATH=tools:. /share/latent/mllm_env/bin/python experiments/v7-dryrun/grad_probe.py   # slot 与 native，含旧界对照
CUDA_VISIBLE_DEVICES="" PYTHONPATH=tools:. /share/latent/mllm_env/bin/python experiments/v7-dryrun/grad_probe2.py  # native 净测（界已复原）
CUDA_VISIBLE_DEVICES="" PYTHONPATH=tools:. /share/latent/mllm_env/bin/python experiments/v7-dryrun/grad_probe3.py  # 双层界归因
```

`grad_probe.py` 的首次运行有一处已知缺陷（`_soft_bound` 未复原，导致第三段沿用旧界），native 的净值以 `grad_probe2.py` 为准；归因以 `grad_probe3.py` 为准。每个脚本在真实维度上单次约 5–8 分钟（CPU）。

## 8. 回归

`CUDA_VISIBLE_DEVICES="" PYTHONPATH=tools:. /share/latent/mllm_env/bin/python -m unittest discover -s tools -p 'test_*.py'` → **235 tests OK**（t9 未改任何被测试代码）。
