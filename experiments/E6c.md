## E6c — 训练档容量压力测试：50% 预算下三臂对比（训练侧）

### 目的

E6b-mini 留下的口径混淆（窗口数与每窗口步数纠缠）与 E6 的未训练状态，
在本轮同时消除：每窗口 Stage B 步数从 24 提到 96（4 倍），结构、loss、
selector、trigger、gate 逐字不动，随后 checkpoint 进入真实 benchmark 的三臂对比。
单一步骤把"更紧预算"与"训练过"两个变量同时压上——`k_max 6144`（当前 12288 的 50%）
是唯一的压力轴。

### 配置

| 项 | 值 |
|---|---|
| 训练窗口 | 8（parquet row 3） |
| held-out 窗口 | 2（parquet row 2） |
| 每窗口 probe 数 | 训练 20，held-out 60 |
| Stage A | 8 窗口 × 4 步 = 32 |
| Stage B | 8 窗口 × 96 步 = 768 |
| 合计 | 800 步，`forward_calls 1608` |
| `window_blocks` / `target_blocks` | 19 / 9 |
| `lr` / `gamma_max` | 1e-3 / 0.05 |
| `seed` | 未传入；`torch.manual_seed` 未执行（缺陷 17 仍未修） |
| 优化器 | 单个 `AdamW` 跨全部窗口与两阶段 |
| retention 测量 | 开启（每窗口步后即时测一次，结束时再测一次） |
| 候选筛选 | 沿用 E6b 的 11 候选 / 10 硬窗口，未重筛 |

行分离与 context 分离与前轮一致：训练侧 context `8678ca61`，
held-out 侧 `0875f57e`，`row_separated True`、`context_separated True`。

### 结果

`integrity: ok`（权重改变且至少一个窗口改变了答案）；`identity_at_init: True`。

| 项 | 8×24（E6b-mini） | 8×96（本轮） |
|---|---|---|
| 训练恢复 | 7/30 = 0.233 | **5/30 = 0.167** |
| held-out 恢复 | 2/16 = 0.125 | **3/16 = 0.188** |
| `final_state.gamma` | 0.0030234 | 0.0054774 |
| `final_state.residual_ratio` | 0.0029830 | 0.0054663 |
| `final_state.route_scale` | 0.0238682 | 0.0378233 |
| `final_state.routing_entropy` | 2.9417140（0.99907×ln19） | **2.4068639（0.81743×ln19）** |
| `forward_calls` | 456 | 1608 |
| retention `immediate` | 8/30 = 0.267 | **15/30 = 0.500** |
| retention `final` | 7/30 = 0.233 | **5/30 = 0.167** |
| retention `net_change` | −1 | **−10** |
| retention reading | `retained` | **`overwrites`** |
| verdict | `partial` | `partial` |

held-out 两个窗口逐条：

| row | start | before | after | recovery | recovered question |
|---|---|---|---|---|---|
| 2 | 3450 | 22/60 | 15/60 | 1/7 | In what city did Gamal Abdel Nasser die? |
| 2 | 3830 | 24/60 | 13/60 | 2/9 | In what country was opéra comique created?; What sport is Atlético Choloma associated with? |

`trained_as_before_matches_after: True`（两个 held-out 窗口都成立），
评分路径与前轮逐字节同口径。

训练侧逐窗口（8 窗口）：

| row | start | lost | immediate | final | recovered question |
|---|---|---|---|---|---|
| 3 | 17890 | 3 | 2/3 | 2/3 | Charlie Manuel（运动）; Victor Turner（受雇于谁） |
| 3 | 17130 | 2 | 0/2 | 0/2 | — |
| 3 | 16750 | 6 | 2/6 | 0/6 | — |
| 3 | 14470 | 5 | 1/5 | 0/5 | — |
| 3 | 17510 | 4 | 3/4 | 0/4 | — |
| 3 | 14850 | 3 | 2/3 | 0/3 | — |
| 3 | 16370 | 5 | 4/5 | 2/5 | Jean-Jacques Bertrand（宗教）; Belfast Celtic F.C.（运动） |
| 3 | 15230 | 2 | 1/2 | 1/2 | Tove Jansson（工作城市） |

### 判读

**步数翻 4 倍没有买到泛化。** held-out 从 2/16 到 3/16（n=16，噪声内），
训练恢复反而 7/30 → 5/30。

**机制侧动了一项。** `routing_entropy` 从 0.99907×ln19 降到 0.81743×ln19，
routing 分布被显著推开；但 `residual_ratio` 只到 0.0054663（hidden 的 0.55%），
仍远低于 5%。routing 非均匀化的假设成立，残差幅度没有跟上。

**E6b-mini 的"每窗口预算是约束"推论被证伪。** 8×24 时 reading 为 `retained`
（8 个即时恢复存活 7 个，net −1），当时的判读指向"每窗口步数太少"。
8×96 把步数翻 4 倍后 reading 翻转为 `overwrites`：immediate 15/30、final 5/30、
net −10，4 个窗口（16750/14470/17510/14850）在**自己**的步数内恢复过、
结束时归零。约束不在每窗口预算，而在**训练 schedule 本身**——
每个窗口内部恢复得更多，随后被后续窗口覆盖得同样多。

一个判读纪律：8×24 的 `retained` 分支与 8×96 的 `overwrites` 分支各自给出的
"下一步该花在哪"互为否定（一个指向加步数、一个指向步数已足够多），
说明该字段只能事后解释数据，**不能用来选择下一步**。本轮不改 verdict 逻辑，
`overwrites`/`retained` 分支的 reason 字符串原样保留在产物里，供下次判读时参考。

`verdict: partial` 沿用旧 0.5 阈值标签；本轮设计明确不设 0.5 门槛，
checkpoint 直接进入下一节的三臂 benchmark，该标签不作为失败读。

### 复现方式

```text
E6A_NEED_MB=21000 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/share/latent/mllm_env/bin/python /tmp/e6b_mini.py \
  --steps-a-per-window 4 --steps-b-per-window 96 \
  --train-windows 8 --eval-windows 2 \
  --probes 20 --eval-probes 60 --max-probe-tokens 256 \
  --lr 1e-3 --gamma-max 0.05 --min-probes 4 --min-lost 2 \
  --max-screen 66 --screen-budget-min 100 \
  --candidates /tmp/e6a_screen_data.json \
  --out /tmp/e6c_train_report.json --ckpt /tmp/e6c_ckpt.pt
```

驱动 `tools/run_e6c_train.sh`（md5 `d6175dc4c5c0b8438e098c72cfc48099`）、
训练脚本 `tools/e6b_mini.py`（md5 `1e78e4decdaf8cb027fba21addbdbe3c`）、
`methods/prlmc/layout.py`（md5 `a1e23be610b88617194448db5376ebfe`，含 P0 修复，
本章节下一步的三臂 bench 使用同一份）。完整报告 `runs/e6c_train_report.json`、
完整日志 `runs/e6c_train.log`。运行于 v100-2 gpu9，2026-09-30
16:44 UTC 启动、12:00 UTC 写盘，`EXIT=0`。

### 附：产物

| 文件 | 大小 | md5 |
|---|---|---|
| `runs/e6c_train_report.json` | 43528 B | `1b7cee77dd04dcec57b786edfb29ff3f` |
| `runs/e6c_train.log` | 116717 B | `d13d52c14711cfaab94bfe2959b29b01` |
| `/tmp/e6c_ckpt.pt`（v100-2，未入库） | 2099858 B | `8e79037efd7cd8902d5c847d02b5cebb` |

checkpoint 含 6 个键：`gamma_logit`、`route_logit`、`q.weight (32,4096)`、
`k.weight (32,4096)`、`down.weight (32,4096)`、`up.weight (4096,32)`。

### 附：三臂 benchmark（未运行）

训练完成后的下一步是三臂对比。三份配置已入库
（`configs/agent_conf/prlmc_e6c-{carrier,evict,select}.yaml`），驱动
`tools/run_e6c_bench.sh`、对比脚本 `tools/e6c_compare.py` 均已入库。
判定标准（预注册）：`Acc_carrier > Acc_selection` **且**
`Acc_carrier > Acc_eviction` 稳定成立；持平则对剩余融合能力做一次清晰证伪。
运行前应 grep 排空后的 `prlmc_stats.jsonl` 查 `compactor_error`，
并核对 `compactor_errors` 计数器（P0 修复新增），确认 carrier 臂未静默退化成驱逐。

---

[← 索引：EXPERIMENTS.md](../EXPERIMENTS.md)
