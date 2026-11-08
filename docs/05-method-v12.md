# V12：桥接路由 + 动态容量分配（代码口径）

本文只写**对当前代码有约束力**的内容。数字与结论见 [03-results.md](03-results.md)；
本轮实验尚未运行，跑完回填。

## 一句话

在固定的 2,304 个潜在记忆槽位下，载体块不再是"每块固定让出 76 个"，而是**每块先保 140 个**，
剩下的 1,044 个由整个窗口竞争：未入选的载体槽和全部被删块槽位一起排序，按**冻结 Reader 自己的
可读性分数 + 跨块桥接分数**取前 1,044。一个被删块能拿多少容量，是排序的结果，而不是配置参数。

```
载体块 9 × 256 = 2,304  --每块保底 140-->  1,260 protected
released 9 × 116 = 1,044  ┐
被删块 10 × 256 = 2,560   ┴--全局竞争-->  1,044 contested
                                          2,304 槽位（不变）
```

槽位处理全程是**原生搬运**：装进窗口的每一个槽位都是写进池子里那个向量的逐位拷贝，代码里用
`torch.equal` 校验，不是容差。本轮不改向量、不折叠、不校准、不训练任何参数。

## 与 V11 的差别

| | V11（已下线） | V12（当前代码） |
| --- | --- | --- |
| 载体块的让位 | 固定 180 core + 76 graft，**每块让多少是参数** | 每块保底 140，**让多少是排序结果** |
| 被删块的容量 | 684 个，在被删块内部按效用排序 | 1,044 个，**与 released 载体槽一起竞争** |
| 排序信号 | 只在本块/被删块内部 | 全局：Reader 效用 + 跨块桥接分数 |
| 搬运方式 | 折叠（改值）+ 淘汰 | 只淘汰，逐字搬运 |
| 参数 | 0.52 M 可训练残差（384 更新） | **0**（本轮不训练任何参数） |
| 臂数 | 8 | 6 |

V11 的折叠（模块 2）与全池校准（模块 3）已从代码里删除：实测折叠比淘汰差 4 题（57 vs 61）
且过不了 donor 检验，而校准改的值和折叠改的值是同一批值，两者重复。

## 三个部件

### 模块 1 `reader_utility.ReaderScorer.reading`（沿用）

换掉 `disc_leverage` 的那个分数，V11 起就在用，本轮不改：

```
U_i = Σ_{l∈{12,16,20,24,28}} Σ_h α_i^{(l,h)} · ‖v_i^{(l,h)}‖₂
k_i = W_K·RMSNorm(h_i)    v_i = W_V·RMSNorm(h_i)    q_t = W_Q·RMSNorm(h_t)
α = softmax_slots(q·kᵀ/√128)
```

探针是**训练侧**的固定问题（`--probe-task musique` 的前 `--probe-count 16` 条 query），
在第一条测试题被读之前编码一次、158 题复用；没有任何测试问题或参考答案进入这个分数，
报告里记 `probe.sha256`。几何口径是**源窗口位置**（槽位在 19 块窗口里的原位），不是压缩后的位置。

### 模块 2 `reader_utility.ReaderScorer.affinity`（新）

同一个投影、同一个 RoPE，把"记忆槽当作 query"：

```
B_i = mean_{s∈seeds} max_{h∈heads} softmax_{candidates}( q_s·k_iᵀ / √128 )
```

`seeds` 是每个载体块效用最高的 `--seeds-per-block 8` 个槽位（9 × 8 = 72 个），它们按构造本来
就在 protected 集合里——问的是"我已经保留的东西指向哪些候选"。softmax 在**候选池内部**做，
逐个 seed、逐层，池化用"该 seed 最尖的那个 head"，再对 seed 取平均。head 上平均在 3,600 量级的
池子上是一堆 1/N 的噪声，一开始就弃用了。

**这是一个代理，不是测量。** 部署时的读路径里 query 只来自问题，记忆槽从来不当 query；
`B` 只回答"这个保留的槽会去看谁"，不回答"Reader 会怎么用它"。它值不值这 1,044 个名额里的权重，
正是这一轮要跑的实验要证明的事。报告里同时记录 `bridge_utility_correlation`：如果桥接分数与
效用几乎共线，这套机制只是在给同一个排序加噪声。

### 模块 3 `bridge_routing.BridgeRouter`（新）

三个决定：

1. **保底**：每个载体块效用前 140 逐字保留，进 protected；
2. **竞争**：released 的 9 × 116 = 1,044 个载体槽 + 全部 2,560 个被删块槽，共 3,604 个候选，
   一起竞争 1,044 个名额。哪块进多少、载体槽反抢回多少，都是排序结果；
3. **排序**：`(1-w)·效用 + w·桥接`，再乘去重折扣 `(1-nw+nw·novelty)`，最后 MMR 去冗余。
   `novelty = 1 - max cos(候选 key, protected key)`，`w = 0.5`、`nw = 0.5` 是默认值。

`FixedQuotaRouter` 是同一个装配器上的对照组（臂 B）：每块效用前 180 逐字、684 个被删块槽，
即 V11 淘汰版的选择逻辑去掉校准。两条路由共用 `read`（一次投影、一次读数）与 `_assemble`
（时间序 + 逐字校验），所以两者之差只可能来自分配规则与桥接项。

## 预算算术

| 项 | 数 |
| --- | --- |
| 窗口 | 19 块 × 256 = 4,864 |
| 载体块 | 9（`gr.select_carriers`，一次选择，所有臂共用） |
| 预算 | 2,304 = 9 × 256 |
| protected | 9 × 140 = 1,260 |
| contested | 2,304 − 1,260 = 1,044 |
| 候选池 | 1,044 released + 2,560 被删块 = 3,604 |

1,044 不是凑出来的：它正好等于载体块让出的数量，所以预算仍然严格是 2,304，**每进来一个
被删块槽位就必然挤掉一个载体槽**——容量中性，和前面每一轮的算术一致。

## 六个臂

| 臂 | 内容 |
| --- | --- |
| `Full` | 19 块不压缩，参考上限 |
| `Selection` | 9 个载体块逐字，容量基线 |
| `Graft` | V10 原生嫁接 1,944 + 360 |
| `AnchorOnly` | 冻结 `AnchorResidual`，臂 A，56/158 基线 |
| `V11NativeGraft` | V11 固定配额（180 + 76）原生搬运，无校准，臂 B |
| `Bridge` | 140 保底 + 1,044 全局竞争 + 桥接分数，臂 C |

前四条臂必须逐题复现参考报告（`--verify-against`），否则整轮作废。

## 判定规则

三条，报告里机械判定而不是靠眼看：

1. `Bridge − V11NativeGraft ≥ +5` **且** 损伤不超过基线的 13 题 → 路由本身站得住，下一步加微跨度；
2. 救回题数 > 改错题数，但净收益不到 +5 → 先改 140 这个保底值，不叠加新模块；
3. 与臂 B 持平或更差 → 停，回到"桥接分数与 Reader 的实际利用是否相关"这个问题上，不急着加模块。

同时看三个分组（被删块证据 n=30、保留块证据 n=44、Full 可答的保护题）与 `p_gold` 方向。
增删题数必须一起报：救 8 题改错 10 题不算收益。

## 本轮明确不做

- 微跨度/成组搬运（相邻槽绑定），邻域摘要，Fourier-FiLM 相位校准；
- 任何训练、任何 checkpoint、任何探针实验；
- donor 对照臂（按指示只跑三个对照组）。所以本轮的正结果只能归因到
  "保底 + 全局竞争 + 桥接项"这一整包，不能归因到内容；报告的 `known_limits` 里写明了这一点。

## 代码位置与入口

| 文件 | 内容 |
| --- | --- |
| `methods/prlmc/reader_utility.py` | `ReaderScorer.reading`（效用）、`.affinity`（桥接）、`mmr_select` |
| `methods/prlmc/bridge_routing.py` | `FixedQuotaRouter`（臂 B）、`BridgeRouter`（臂 C）、共用装配器 |
| `tools/eval_bridge_routing.py` | 六臂同进程评测，逐题配对，分配诊断，判定规则 |
| `methods/prlmc/anchor_residual.py` | 只剩 `AnchorResidual`（臂 A 用），V11 的全池残差已删 |

```
python tools/eval_bridge_routing.py --dry-run \
    --data-dir /data/.../LongBench/extracted/data --model-path /dev/shm/s1d_reader_fp16

python tools/eval_bridge_routing.py \
    --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
    --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
    --model-path /dev/shm/s1d_reader_fp16 \
    --checkpoint runs/natural_merge_v9/full_v10.pt \
    --anchor runs/anchor_residual/anchor_only_rank64.pt \
    --verify-against reports/anchor_fusion/eval_anchor_fusion_hotpotqa.json \
    --cuda-device 0 --out reports/bridge_routing/eval_v12_hotpotqa.json
```

没有 GPU 也能检查的东西：`--dry-run` 只读 tokenizer 和题目，打印会跑多少题、六个臂、
桥接与保底参数。
