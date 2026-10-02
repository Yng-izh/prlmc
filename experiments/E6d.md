## E6d — 训练顺序三臂消融：连续窗口训练是否是遗忘的原因（种子固定）

### 目的

E6c 把每窗口步数提到 96 后，retention reading 翻转为 `overwrites`
（immediate 15/30 → final 5/30）。但那只是一条训练轨迹的事后快照，
"后续窗口覆盖先前恢复"是推断，不是测量。本轮固定种子（0），
把**训练顺序**单独切出来：同样的 8 个窗口、同样的 800 步、同样的每窗口 96 步，
只改调度，并第一次给每道题记录全程 10 点对错 trace（学会时间/遗忘时间）。

三臂调度：

| 臂 | 调度 | 说明 |
|---|---|---|
| forward | 1→8 窗口连续，每窗 96 步 | 与 E6c 完全同构（种子除外） |
| reverse | 8→1 窗口连续，每窗 96 步 | 顺序反向，容量压力相同 |
| interleave | 每窗口 4 步然后换下一个，8 窗口轮流，循环 24 轮 | 每窗 96 步不变，但任意 4 步间隔 28 步 |

interleave 臂能跑，依赖本轮新加的教师前向缓存
（`methods/prlmc/probes.py` 的 `_TEACHER_CACHE`）：teacher 前向是
(pool, probe batches) 的纯函数，按 (row, start, context_hash, probe 数)
缓存激活，200 次 teacher 前向降为 8 次，OOM 死亡消失。

### 配置

| 项 | 值 |
|---|---|
| 窗口 | E6c 同一 8 窗（row 3，start 17890/17130/16750/14470/17510/14850/16370/15230） |
| probe | 每窗口 20 题，10 个 node（cum 0/32/128/224/320/416/512/608/704/800） |
| Stage A / B | 8×4=32 步 / 8×96=768 步 |
| seed | 0（`torch.manual_seed(0)`、`torch.use_deterministic_algorithms(True)`） |
| lr / gamma_max | 1e-3 / 0.05 |
| 结构 / loss | 与 E6c 逐字相同（`structure_changed: false`、`loss_changed: false`） |

可复现性验证：三臂 baseline 全部 26/160、lost 30、after-Stage-A 26/160 + 1/30
恢复，`initial_state_md5` 三臂同值 `0f70659546efc9b3513b403f3ded8a86`。

### 结果

| 项 | forward | reverse | interleave |
|---|---|---|---|
| 最终 / 160 | 37 | 33 | **56** |
| net vs baseline | +11 | +7 | **+30** |
| 学会题数（learned） | 17 | 15 | **31** |
| 遗忘题数（forgotten） | 1 | 2 | 1 |
| 学会事件 / 遗忘事件 | 61 / 50 | 49 / 42 | **36 / 6** |
| 每 trace 平均翻转 | 0.69 | 0.57 | **0.26** |
| held（学会且未忘） | 12 | 9 | 31 |
| 丢失（baseline 对→末态错） | 1 | 2 | 1 |

node 轨迹（n_correct@cum）：

| 臂 | 0 | 32 | 128 | 224 | 320 | 416 | 512 | 608 | 704 | 800 |
|---|---|---|---|---|---|---|---|---|---|---|
| forward | 26 | 26 | 33 | 32 | 31 | 31 | 36 | 30 | 43 | 37 |
| reverse | 26 | 26 | 32 | 36 | 35 | 36 | 32 | 32 | 32 | 33 |
| interleave | 26 | 26 | 35 | 36 | 37 | 38 | 38 | 42 | 51 | 56 |

interleave 全程单调上升；forward 在 608（−6）与 800（−6）两次大回落，
reverse 在 512（−4）与 704（−4）回落。遗忘事件按窗口集中在中后段
（forward：608 段 11 次、800 段 12 次；interleave 全程仅 6 次）。

单题示例（`3:17890` "What sport is Charlie Manuel associated with?"，
10 点 trace）：

- forward：错→对→对→对→错→对→错→对→对→对（7 次翻转）
- interleave：错→对→对→对→对→对→对→对→对→对（1 次翻转，32 步处学会后不再忘）

### 判读

**遗忘的主因是连续窗口训练，且可被纯调度手段消除。** interleave 同样
800 步、同样每窗 96 步、同样只训 compactor，净提升 +30 vs forward +11、
reverse +7；遗忘事件从 50 降到 6。这把 E6c 的 `overwrites` 推断
从相关性升级为因果：不是"训得太多"，而是"连续训一个窗口太久"。

**顺序方向本身不重要。** forward 与 reverse 差异（37 vs 33，+11 vs +7）
远小于两者与 interleave 的差距；reverse 同样有中段回落。容量/覆盖
冲突与顺序方向无关，与窗口连续长度有关。

**代价在拟合深度。** 每窗口末 loss：interleave 各窗 3.3–4.4，
forward 连续训练末窗低至 0.8–2.2。interleave 用拟合深度换全局保留，
在保留指标上赢得彻底（56/160 持有、只丢 3 道持有中题、0.26 翻转/题）。

**学会/遗忘在时间轴上可分离。** 单题 trace 显示典型模式：连续臂中
"学→忘→学"反复拉锯（61 学 vs 50 忘），interleave 中学会一旦发生
（如 cum 32/128/416/800 处）即持有到终点。

### 复现方式

三臂分别运行（forward 即 E6c 配置 + seed 0）：

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
/share/latent/mllm_env/bin/python /tmp/e6b_mini.py \
  --seed 0 --schedule forward|reverse|interleave \
  --steps-a-per-window 4 --steps-b-per-window 96 \
  --train-windows 8 --eval-windows 0 \
  --probes 20 --eval-probes 0 --max-probe-tokens 256 \
  --lr 1e-3 --gamma-max 0.05 --min-probes 4 --min-lost 2 \
  --candidates /tmp/e6a_screen_data.json \
  --out /tmp/e6d_<arm>.json
```

运行于 v100-2：forward gpu4（OOM 死亡 2 次后带 alloc conf 重启完成）、
reverse gpu3、interleave gpu6（教师缓存版本）。脚本
`tools/e6b_mini.py`（md5 `ae05653cfb205ee36d7e708e09f850af`）、
探针支持 `methods/prlmc/probes.py`（md5 `6bd8ca1932b0bb9338b75b6d39f81a27`）。

### 附：产物

| 文件 | 大小 | md5 |
|---|---|---|
| `runs/e6d_forward.json` | 229417 B | `993f76c6b1f92186241c5d78c63c6991` |
| `runs/e6d_reverse.json` | 228995 B | `5e52451d63e1ec1cb383811c63b64fbb` |
| `runs/e6d_interleave.json` | 274855 B | `5a8efd15c7db57e98220a21522bbb952` |

`arms[arm].question_records` 结构：window → question →
{trace: 10 bools, learned_at: [cum], forgotten_at: [cum],
right_at_baseline, right_at_end}。
