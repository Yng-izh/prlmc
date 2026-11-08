# V13：跨块条件化的潜空间校准（代码口径）

本文只写**对当前代码有约束力**的内容。数字与结论见 [03-results.md](03-results.md)；
本轮实验尚未运行，跑完回填。

## 一句话

保留 V11Evict 已经跑通的窗口结构（1,620 core + 684 graft，固定 2,304 预算，冻结 Reader/Writer），
只改**校准阶段**：校准器在修改某个槽位时，除了看它自己，还能看到**其他块里与它最相关的 4 个槽位**
（Reader key 余弦、同块屏蔽、稀疏 top-4），把这些邻居的**内容**作为条件注入有界低秩残差。
槽位本身从不被平均、不被替换——只是被同一个有界低秩映射推动。

## 为什么只改校准

实验 7 把"选槽 + 配额 + 搬运 + 全池校准"这个包拆开了：同配额同搬运下，把训练出来的全池残差
去掉，61 题掉到 44 题（−17）；而实验 8 又证明纯选择路线（Reader 效用 / 记忆自生显著度 + 桥接
排序）无论怎么调都停在 41–47，天花板就是不压缩的 Full 的 47。**可动的收益只剩校准**，而且它
此前的输入只有"槽位自己 + 三个标量"，无法表达"这两个槽位属于同一个事实"。

## 三个部件

### 窗口 `methods/prlmc/evict_window.py`（V11Evict 结构，无参数）

```
载体块 9 × 256  --Reader 效用前 180，逐字-->  1,620 core
被删块 10 × 256 --效用 × novelty + MMR---->    684 graft
                                              2,304 槽位
```

`JointConsolidation` 沿用 V11 的类名与构造签名（驱动不用改形状），只接受 `policy="evict"`：
折叠已实测 −4 题，代码已删。装配器用 `torch.equal` 校验每个装进去的槽位都是源槽位逐字拷贝。

### 跨块稀疏关联 `cross_block_calibration.cross_block_context`

```
s_ij = <key_i, key_j>            key 来自 Reader 自己的多层 key（单位行）
同块邻居屏蔽                     机制要求跨块，这一步保证它真的是跨块
top-k（k = 4）softmax            稀疏：只有 4 个邻居有权重
c_i = Σ_j w_ij · value_j        邻居的**内容**，不是把邻居平均进槽位
```

key/value 在**窗口本身**上算（`scorer.reading(window, ..., probes=None)`），所以 donor 臂换了
内容，条件跟着换——不会出现"窗口是别人的、条件还是自己的"。

### 校准器 `cross_block_calibration.CrossBlockResidual`

V11 的 `FullPoolResidual` 加一项：`cross_proj`（零初始化）把 `c_i` 投进瓶颈。
`up_graft / up_core` 两套输出、`type_gate`、`tanh` 有界、RMS 逐层恢复，全部与 V11 相同。
零初始化意味着 **step 0 精确等于 V11 的校准器**（核心半边是恒等），这是构造保证，不是经验结论。

有区别的**软**约束，两侧都不锁：

| | core（载体） | graft（搬入） |
| --- | --- | --- |
| 残差上限 | `0.15 × rms` | `0.50 × rms` |
| 训练残差正则权重 | `2.0` | `0.5` |

实验 4 证明硬锁 key 会让新内容失效（50/158），实验 5 证明内容主要走 key 通道，所以这里是
"载体槽只许动 15%"而不是"载体槽不许动"。

## 训练

`tools/train_cross_block_calibration.py --cross on|off`，两个变体同数据、同步数、同 seed：

- `--cross on` → V13；
- `--cross off` → 同一个模块去掉条件项，即 **V11Evict 的校准器**（实验 7 的权重已随 V11 代码
  下线删除，所以这一臂在本轮框架内重训；这样比较时只有一个变量不同，比跨轮比较更干净）。

目标与 V11 相同：真实答案 CE + 真实 graft / 无关 graft 反事实 hinge + 保护 KL + 残差正则
（正则按上面的 2.0 / 0.5 加权）。Graft 半边继续从 AnchorOnly checkpoint 初始化。

## 八个臂

| 臂 | 内容 |
| --- | --- |
| `Full` | 19 块不压缩，损伤参照 |
| `Selection` / `Graft` / `AnchorOnly` | 三条复现臂 + 56/158 基线 |
| `V11Evict` | 同窗口 + `--cross off` 校准（本轮的"要超过的那一臂"） |
| `V11EvictDonor` | 上式把 684 个 graft 换成上一题的文档 |
| `V13` | 同窗口 + `--cross on` 校准 |
| `V13Donor` | 同上换成上一题的 graft |

四条老臂必须逐题复现参考报告，否则整轮作废。

## 判定（三条必须同时成立）

1. `V13 − V11Evict ≥ +5`：跨块交互带来额外收益，而不是残差校准本身有效（后者两臂都有）；
2. `V13 − V13Donor ≥ +5`：收益依赖**真实搬入的内容**；
3. `V13` 在 Full 可答的 47 题上的损伤 ≤ 基线：收益与损伤的关系改善，而不是搬运。

## 硬校验

- 每条臂恰好 2,304 槽、1,620 / 684、全局时间序、每槽逐字等于源槽；
- 两半都必须被校准推动，且 RMS 恢复成立；
- `--v11` 必须**不是**跨块条件版，`--v11-evict` 必须**是**（否则臂跑错，直接报错）；
- donor 窗口必须与真实窗口不同；
- 每行的跨块邻居块分布被记录：如果上下文只来自同块，那是 bug 不是结果。

## 独立数据

HotpotQA 158 题已被本项目反复用于挑机制。论文口径要求另一次自然多跳问答的独立验证：
`--task/--data-dir` 换成另一个数据集重跑同一驱动，`protocol` 里记录 `task`、`data_dir` 与
sha256，两次报告可以直接对照。本轮代码不预设数据集，只保证驱动与协议与 HotpotQA 完全一致。

## 代码位置与入口

| 文件 | 内容 |
| --- | --- |
| `methods/prlmc/evict_window.py` | V11Evict 窗口（1,620 + 684），`JointConsolidation` |
| `methods/prlmc/cross_block_calibration.py` | 跨块稀疏关联、`CrossBlockResidual`、`calibrate` |
| `tools/train_cross_block_calibration.py` | 训练（`--cross on|off` 两个变体） |
| `tools/eval_cross_block_calibration.py` | 八臂评测、三个分组、损伤、跨块诊断 |

```
# 两个变体各训一次（384 更新）
python tools/train_cross_block_calibration.py --cross off --updates 384 \
  --data-dir ... --train-task musique --model-path /dev/shm/s1d_reader_fp16 \
  --checkpoint runs/natural_merge_v9/full_v10.pt \
  --anchor runs/anchor_residual/anchor_only_rank64.pt \
  --out runs/cross_block/plain.pt --report runs/cross_block/plain.json

python tools/train_cross_block_calibration.py --cross on --updates 384 \
  ... --out runs/cross_block/cross.pt --report runs/cross_block/cross.json

# 评测（八臂同进程）
python tools/eval_cross_block_calibration.py \
  --data-dir ... --task hotpotqa --supporting-facts-glob '...' \
  --model-path /dev/shm/s1d_reader_fp16 \
  --checkpoint runs/natural_merge_v9/full_v10.pt \
  --anchor runs/anchor_residual/anchor_only_rank64.pt \
  --v11 runs/cross_block/plain.pt --v11-evict runs/cross_block/cross.pt \
  --verify-against reports/anchor_fusion/eval_anchor_fusion_hotpotqa.json \
  --cuda-device 0 --out reports/cross_block/eval_v13_hotpotqa.json
```
