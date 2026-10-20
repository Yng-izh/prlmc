# PRLM-C v7 规格契约（规格为准）

> 状态：**实现契约（未改代码）**。本文件是 t2–t11 的唯一对齐依据。
> 仓库：`v100-2:/share/latent/PRLM-C`，`master @ d5e51e0`，工作树对已跟踪文件干净（仅未跟踪脚本）。
> 阅读范围：`methods/prlmc/{consolidate,layout,gated_slot_merge,native_skip_merge,slot_merge,merge}.py`、
> `tools/{natural_merge_runtime,public_retention_eval,longbench_prlmc_eval,natural_merge_v4_objective,
> prepare_natural_merge_v2,v3,v4,v6}.py`、`tools/test_natural_merge*.py`、`tools/test_budget_and_fused_report.py`。

## 0. 总原则（强制）

1. **规格是最新目标。** 规格与仓库冲突时以规格为准；仓库现状只用于回答"改哪里、怎么改"。
2. **不得用"仓库原本如此"削减规格。** 若某条规格在现有结构里无法逐字落地，必须给出等价实现并证明等价性（第 3 节裁决记录）。
3. **每条规格必须有可执行验收断言**（第 2 节每条的"验收"栏），且必须能在 CPU 上跑（`CUDA_VISIBLE_DEVICES=""`）。
4. **未验证的量化断言不得写成既成事实**（见 C7）。
5. 本阶段全部为 dry-run：不启动 GPU 训练/评测，不写 `runs/`。

术语：
- **块（block）**：原生 256 槽记忆块，`model.num_tokens`。
- **窗口/目标/预算**：`window_blocks` / `target_blocks` / `budget_blocks`（原生块数）。
- **载体（carrier）**：选择器保留、被写入输出前 `target_blocks` 位的块。
- **age**：该块经历过的合并轮数。`age=0` = 新世代（本轮新写入或从未被压缩过）；`age>=1` = 老世代（幸存过至少一轮压缩）。
- **老块霸位率**：`#{载体 age>=2} / target_blocks`（t10 的代理指标，需从 events 读取）。

---

## 1. 契约条目索引

| 编号 | 主题 | 现状锚点 | 实现任务 | 状态 |
|---|---|---|---|---|
| C1 | 分代配额块选择（`block_ages`/`age_decay=0.75`/`max_old_ratio=0.45`） | `consolidate.py:204`、`:171` | t4 | 待实现 |
| C2 | `_block_ages` 真实写入点与滑动方向 | `layout.py:690/755/766/810/829/1181`、`public_retention_eval.py:186`、`natural_merge_runtime.py:153` | t4 | 待实现 |
| C3 | runtime 签名/字段变更（`install`、`MergeStream`、`load_runtime`、`validate_merge_checkpoint`） | `natural_merge_runtime.py:153/88/286/272` | t2, t4, t6 | 待实现 |
| C4 | 无 `source_manifest` 的历史 ckpt 默认拒绝 | `natural_merge_runtime.py:15/37/272` | t2 | 待实现 |
| C5 | 废除强制单位 RMS，改 tanh 软有界（作废旧 ckpt） | `gated_slot_merge.py:173-181`、`native_skip_merge.py:64-93` | t5 | 待实现 |
| C6 | V6 子类处置与 v7 registry 命名/隔离 | `gated_slot_merge.py:301/485/511`、`natural_merge_runtime.py:229-240` | t5(类) + t6(注册) | 待实现 |
| C7 | "step-0 native_gain 梯度 1.4e5" 待验证 | `consolidate.py:394`、`native_skip_merge.py:93` | t9 实测 | 未验证 |
| C8 | 滑动微融合 13→9 几何与触发规则 | `public_retention_eval.py:116-120/146-148/190-196`、`longbench_prlmc_eval.py:268-273/391` | t6（+t3 CLI） | 待实现 |
| C9 | 稀疏高信噪比门控（`active_match_weight`） | `gated_slot_merge.py:268-279`、`:455-461` | t5 | 待实现 |
| C10 | 靶向恢复权重（objective） | `natural_merge_v4_objective.py:53-54`、`prepare_natural_merge_v3.py:26` | t6 | 待实现 |
| C11 | CPU 契约测试清单与文件 | `tools/test_*.py`（基线 112 项，1 项预存失败） | t2/t4/t5/t6/t7 | 见 §2.C11 |

---

## 2. 逐条契约

### C1 分代配额块选择

**现状（file:line）**

- `methods/prlmc/consolidate.py:204-211`：`_select_block_indices(source, target, block, weights, goal_direction, select_weights)`，无 age 参数。
- `:230-233`：`b = k//block`，`b_out = target//block`，`b_out >= b` 直接返回全块。
- `:244`：块摘要取 `x[min(_SELECT_LAYER, layers-1)]` 的均值；`:115` `_SELECT_LAYER = 16`。
- `:246-250` importance（权重均值或离散度）、`:252` recency（位置）、`:254-260` goal（与 `goal_direction` 余弦）、`:262-266` 加权求和。
- `:268` `_greedy_diverse(score, unit_summary, b_out, cfg["diversity"])`。
- `:171-201` `_greedy_diverse`：每轮 `base = _unit(score)`，`combined = base + diversity_weight*nearest`，`~available` 掩为 `-inf`，`argmax`；返回 `sorted(chosen)`（时间序）。
- 默认权重 `:122-127`：importance .4 / recency .3 / goal .2 / diversity .1。

**规格要求**

- 选择器必须吃 `block_ages`，参数名与默认值：`age_decay=0.75`、`max_old_ratio=0.45`。
- 分代配额：老世代最多 `floor(target_blocks*0.45)` 席（9 目标 → 4 席），新世代保底 `target_blocks - floor(...)` 席（→ 5 席）。
- 若新块不足，按剩余额度回填，**输出块数不得少于 `target_blocks`**。
- 首轮 `ages` 全 0 时，选择结果必须与现行实现**逐位一致**（首轮不变量）。

**落地方式**

签名扩展（全部为带默认值的关键字参数，17 个现网调用点无需改动）：

```python
def _select_block_indices(
    source, target, block, weights, goal_direction, select_weights,
    block_ages: Optional[torch.Tensor] = None,   # [b] long, 0 = 新世代
    age_decay: float = 0.75,
    max_old_ratio: float = 0.45,
) -> torch.Tensor:
```

1. `block_ages is None or not bool(block_ages.any())` → **直接走 `:235-269` 原路径**（早退分支，不是"乘 1.0 的近似"）。这是首轮不变量的第一道保险。
2. 分代分支：
   - `ages = block_ages.to(torch.long).view(b)`；`old = ages > 0`。
   - `score = score * torch.pow(age_decay, ages.to(torch.float32))`。`0.75**0 == 1.0` 在 IEEE 下是精确乘法，因此即便走分代分支，全 0 ages 的分数也与原路径逐位相同（第二道保险）。
   - `old_cap = int(math.floor(b_out * max_old_ratio))`（9 → 4）；`new_floor = b_out - old_cap`（→ 5）。
3. `_greedy_diverse` 增加两个可选参数：`group: Optional[Tensor] = None`、`group_cap: Optional[int] = None`。**配额在贪心循环内部生效**，而不是先选后修正：
   - 每轮在算出 `combined` 后，若 `group is not None` 且已选老块数 `>= group_cap`，则 `combined = combined.masked_fill(group & available, -inf)`；
   - 若掩码后 `combined` 全为 `-inf`（新块已耗尽），**回填**：撤掉本轮的配额掩码，在剩余 `available` 上照常 `argmax`；
   - diversity 项只在剩余候选集上参与 `argmax`，因此不改变贪心次序：配额不触发时（全 0 ages、或老块数未超限）循环体与现状逐字相同。
4. 返回值仍为 `sorted(...)` 升序，调用方 `index_select` 保持时间序（`:221-222` 的既有约束不变）。

**验收断言**

- `tools/test_block_ages.py::test_first_round_bit_identical`：固定 `torch.manual_seed(3)`、`source=randn(4,38,8)`、`block=2`、`target=18`，`block_ages=None` 与 `block_ages=zeros(19)` 的返回值 `torch.equal` 且等于附录 A 的黄金向量。
- `test_old_quota_cap`：构造 ages（5 新 + 14 老），断言 `#{selected old} <= floor(9*0.45) == 4`、`len(selected) == 9`、`#{selected new} >= 5`。
- `test_backfill_when_new_scarce`：只给 2 个新块，断言 `len(selected) == 9` 且老块回填 7 席（不缩短输出）。
- `test_age_decay_monotone`：同一分数下 `age` 越大被选中的概率单调不增（构造两个仅 age 不同的同分块，断言选 age=0 者）。

---

### C2 `_block_ages` 的真实写入点与滑动方向

**现状（file:line）——两条互不相交的写入路径**

A. **托管写入（`ProtectedResidualMemory`）**
- `layout.py:511-654` `write()`：novelty → 分配/寻址 → gate → `:610` `_maybe_compact` → `:612` `_place`。
- `:690-764` `_place()`：`:709-720` `full_rewrite/anchor_rewrite` 直接覆写尾部并 `used=need`；`:722-726` 寻址写 → `_merge_into_block`；`:737-753` 腾空间循环（`compact` 失败则 `_evict_oldest(need)`）；`:755-757` `_shift_left(need)` 后把新块写到物理尾部，`:758` `used += need`。
- `:810-827` `_shift_left(need)`：把 distinct 后缀整体左移 `need` 槽（前→后分块拷贝）。
- `:766-808` `_merge_into_block(target, delta)`：合并后把该块搬到尾部（`:789-799` 前向闭合间隙），`used` 不变，`:803-807` 若原属 consolidated 区则 `n_consolidated` 回退。
- `:829-842` `_evict_oldest(n)`：`used -= n`，不搬数据（幸存者本来就在尾部）。
- `:1181-1182` `compact_with_report`：`used = new_used`、`n_consolidated = target`；`:1176-1180` 把压缩结果写到 `new_start`，尾部 remainder 右移。
- `:329-357` `reset()`：`used=0`、`n_consolidated=0`。

B. **评测/训练写入（StreamingMemory + install）——这才是 longbench/natural-merge 实际走的路径**
- `tools/longbench_prlmc_eval.py:167` 调 `stream.write(record, write_index)`，**不经过** `manager.write`/`_place`。
- `tools/public_retention_eval.py:186-222` `StreamingMemory.write`：`:203-215` 手工把 `[old_start, total)` 左移一个块后写入尾部（`old_start - m.block : m.total - m.block`），`:216` `m.used += m.block`。
- `tools/natural_merge_runtime.py:153-170` `install(manager, candidate, window)`：把候选写回窗口位置、尾部整体右移、`m.used = replacement.shape[1]`、`:169-170` 校验 active mask 计数。
- `tools/public_retention_eval.py:155-184` `StreamingMemory.compact`：`SharedE6d`/`Selection`/`Recent` 三臂各自算 `selected`（`:167`/`:170`/`:173`），就地拼接写回（`:180-185`），**不调用 `install`**。

**规格要求**

- `ProtectedResidualMemory` 维护 `_block_ages`（长度 `total//block`，`long`），在真实写入路径同步滑动，新块 `age=0`、旧块左移。
- 提供 `window_block_ages(start_slot, window_slots)` 供选择器使用。
- `install(manager, candidate, window, selected_blocks=None)`：载体 age+1、尾部未压缩块保持、超窗清零；`reset()` 清零。
- `_block_ages` 不得进入 state hash 契约（除非裁决需要）。

**落地方式**

1. **物理对齐**：`_block_ages[j]` = 物理第 `j` 块的 age，有效区 `[nb - used//block, nb)`，`nb = total//block`。总宽与 `used` 都是 `block` 的整数倍，因此 `start_slot//block` 直接就是物理块号，`window_block_ages` 是一次切片。
2. **滑动方向裁决（回答任务问题）**：规格的 `torch.roll(-1)` **等于**实际追加方向。证明：追加一块后后缀物理区间从 `[nb-n, nb)` 变为 `[nb-n-1, nb)`；`roll(-1)` 得 `new[j] = old[j+1]`，对 `j ∈ [nb-n-1, nb-2]` 正是"旧块左移一格"；`new[nb-1] = old[0]`（回卷垃圾）随即被 `= 0` 覆盖，即"新块 age=0"。边界与有效区起点同时对齐，无需额外修补。`_place`（`:755-757`）与 `StreamingMemory.write`（`:210-215`）都是同一方向。
3. **一般化助手**（放在 `ProtectedResidualMemory` 上，两条写入路径共用同一真源）：

```python
def note_append(self, appended=1, dropped=0):      # 物理块数
    self._block_ages = torch.roll(self._block_ages, appended - dropped)
    self._block_ages[-appended:] = 0               # 新块 age=0

def note_merge(self, selected_blocks, window_blocks, tail_blocks):
    # selected_blocks: 窗口内被选载体的块号（相对窗口起点，升序）
    # 输出块 j 的 age = 被选载体 age + 1；尾部未压缩块保持；其余清零
```

- `_place` 追加路径：`note_append(appended=1, dropped=evicted)`（`evicted` 为 `:742` 的计数；`_evict_oldest`/`_shift_left` **不得**各自再动 ages，否则重复计数）。
- `full_rewrite/anchor_rewrite`（`:709-720`）：`_block_ages.zero_()`，仅 1 块。
- `_merge_into_block(target, delta)`：块 `target` 搬到尾部、其后各块左移一格 →
  `_block_ages = cat(ages[:t], ages[t+1:], ages[t:t+1])`。**age 保持不变**（同一内容原位刷新，不是新内容；裁决 D-04）。
- `_evict_oldest`：若被独立调用（当前仅 `:741`，与 `_place` 的 append 成对），由 `note_append(dropped=...)` 统一结算。
- `compact_with_report`（`:1096-1204`）：输出块 age = 对应载体 age + 1；remainder 保持。需要拿到"实际被选中的块号"——见 C1 落地第 5 点（把 `index` 从 `_run_compactor` 透出，见下）。
- `install`（`:153-170`）：`first_phys = (total - used_before)//block`，`ages_out[j] = ages[first_phys + selected_blocks[j]] + 1`，尾部 `tail_blocks` 保持，其余置 0；`selected_blocks=None` 时**不动 ages**（保持既有调用与既有测试不变，裁决 D-12）。
- `StreamingMemory.compact`（`:155-184`）：同样在就地写回后调用 `m.note_merge(selected, window_blocks, tail_blocks)`；`MergeStream` 的各分支把 `selected` 传给 `install`。
- `reset()`（`layout.py:329-357`、`runtime.py:330-343`）：`_block_ages.zero_()`。
- `snapshot()`/`restore()`（`layout.py:359-386` / `:388-412`）必须加入 `_block_ages` 的 clone/copy：`compact()` 失败时靠这对函数回滚，漏了 ages 会让"回滚后的池"与"回滚后的年龄"不一致（R-07）。
- `stats()`（`:1393-1420`）与 compaction report 增加 `block_age_histogram`（诊断用，不入 hash）。
- 事件字段：`install`/`compact` 的 events 增加 `selected_block_ages`（t10 需要 `age>=2` 占比；缺该字段就无法离线补算）。

**验收断言**

- `test_roll_mapping`：`ages=[0,1,2,0]`（`nb=4`），`note_append(1,0)` 后有效区尾部为 `[1,2,0]`，且"原 index1 的块现在在 index0"。
- `test_evict_then_append`：`note_append(1,1)` 后 ages 与"丢最老 + 追加新块"逐位一致（不是回卷）。
- `test_merge_permutation`：`_merge_into_block(1)` 后原块 1 的 age 落在尾部，其后各块左移一格。
- `test_install_ages`：给定 `selected=[1,3,5]`，输出三块 age 各 +1，尾部块不变，其余为 0。
- `test_reset_clears_ages`。

---

### C3 runtime 签名/字段变更

**现状（file:line）**

- `natural_merge_runtime.py:153` `def install(manager, candidate, window)`：无 `selected_blocks`。
- `:88-96` `MergeStream.__init__(..., budget_blocks=20)`：只透传 `budget_blocks`；`window/target` 由 `StreamingMemory.__init__` 用 `budget_geometry` 覆盖。
- `:98-150` `MergeStream.compact`：`:105` 复算 `selected`（不带 ages）；`:121`/`:132`/`:143` 三处 `install(m, output, self.window)` 均不传 `selected`；`:148` 兜底走 `StreamingMemory.compact`（该路径自己 `install`，不返回 `selected`）。
- `:286-327` `load_runtime(model_path, checkpoint=None, train=False, architecture=None, budget_blocks=20)`；`:309` `k_max=budget_blocks*model.num_tokens`（**已符合规格，不改**）；`:320-324` 先 `merge.load_state_dict(...)` 后 `validate_merge_checkpoint(...)`（顺序错误）。
- `:272-283` `validate_merge_checkpoint(saved, architecture)`：只校验 `effective_gamma_max/evaluation_eligible/variant/training/module_configuration/global_rms_budget/rank`，**无源码哈希**。
- `:15-34` `SOURCE_FILES`（40 项）、`:37-38` `source_manifest()`：已存在，但只在训练端写成 `training_sources`，校验端从不读。

**规格要求**

- `install` 支持 `selected_blocks`，用于年龄更新。
- `MergeStream` 支持 13→9 的 window/target，且预算合法性显式校验。
- `load_runtime` 透传 `strict_code_hash`；校验必须在加载权重**之前**。
- `validate_merge_checkpoint` 默认强校验源码哈希，缺 manifest 拒绝。

**落地方式（逐处签名/字段）**

| 位置 | 变更 |
|---|---|
| `runtime.py:153` | `def install(manager, candidate, window, selected_blocks=None)`；尾部新增 ages 结算（C2） |
| `runtime.py:88-96` | `MergeStream.__init__(..., budget_blocks=20, window_blocks=None, target_blocks=None)`；`None` → `budget_geometry` 旧行为 |
| `runtime.py:105` | `_select_block_indices(..., block_ages=m.window_block_ages(start, self.window))` |
| `runtime.py:121/132/143` | `install(m, output, self.window, selected_blocks=selected)`；`Consolidate` 臂传 `kept_blocks`（`:133`） |
| `runtime.py:148` | 兜底路径改为 `super().compact()` 后读 `self.last_selected_blocks`（由 `StreamingMemory.compact` 记录）再结算 ages |
| `runtime.py:272` | `validate_merge_checkpoint(saved, architecture, strict_code_hash=True)`：新增 `CRITICAL_MERGE_FILES` 逐文件 SHA256；缺 `source_manifest` 且 strict → 拒绝；失配报 `path: ckpt {8hex} != code {8hex}` |
| `runtime.py:286` | `load_runtime(..., strict_code_hash=True)` 透传 |
| `runtime.py:320-324` | **先** `validate_merge_checkpoint`，**后** `load_state_dict` |
| `runtime.py:307-308` | `budget_blocks` 白名单保留 8/10/12/20；新增几何合法性：`0 < target_blocks < window_blocks`、`target_blocks <= budget_blocks`、`window_blocks <= manager.total//block`（13/9/20 合法） |
| `runtime.py:229-240` | 注册 `grouped_gated_sparse_slot_v7` / `grouped_gated_sparse_native_v7`（C6） |
| `runtime.py:251-254` | `architecture.endswith(("_v5","_v6","_v7"))` 才注入 `native_prior_weight/route_mix_temperature/balance_strength` |
| `runtime.py:258-269` | `_v7` → `TRAIN_V7`（`prepare_natural_merge_v7.py`） |
| `runtime.py:346-351` `costs()` | 增加 `block_age_max`、`old_generation_slots`（可选，仅诊断） |
| `public_retention_eval.py:167/170` | `selected` 计算带 `block_ages`，并写 `self.last_selected_blocks` |
| `public_retention_eval.py:183-185` | events 增加 `selected_block_ages` |

`CRITICAL_MERGE_FILES` 建议最小集（6 个算子文件，全部在 `SOURCE_FILES` 内）：
`methods/prlmc/consolidate.py`、`methods/prlmc/layout.py`、`methods/prlmc/importance.py`、
`methods/prlmc/slot_merge.py`、`methods/prlmc/gated_slot_merge.py`、`methods/prlmc/native_skip_merge.py`。

**验收断言**

- `install` 旧调用（3 参）行为逐位不变：`tools/test_natural_merge.py` 的 E1 序列与 `:202` 的 `before_write == [20,30,40,50,60]` 不变。
- `load_runtime(strict_code_hash=False)` 可加载 v5 ckpt 且 stderr 有 WARNING；`strict_code_hash=True` 抛 `ValueError` 且消息含 8 位哈希（t2 的 `test_checkpoint_code_hash.py`）。
- 校验先于加载：给一个"manifest 正确但 state_dict 缺键"的临时 ckpt，断言抛的是 manifest/结构错误而不是 `load_state_dict` 的 KeyError（可用 mock 计数断言 `load_state_dict` 未被调用）。

---

### C4 无 `source_manifest` 的历史 ckpt 默认拒绝

**现状（实测，非推断）**

`runs/natural_merge_v5_20261018_1050/Dsup.train.pt` 顶层键：
`architecture, effective_gamma_max, evaluation_eligible, final_state_sha256, global_rms_budget, model,
module_configuration, objective, protocol, rank, schema, state_dict, tokenizer, train_articles,
train_content_sha256, train_context_hashes, train_plan_sha256, training, training_sources, variant`
→ **无 `source_manifest`**（有 `training_sources`，但从未被校验）。

且 `training_sources` 与 HEAD 代码已 **4/40 失配**：

```
tools/natural_merge_runtime.py        a348a036 -> 099d7973
tools/public_retention_eval.py        354465d7 -> f517546c
methods/prlmc/gated_slot_merge.py     c0a92c5c -> c95e9d72
tools/natural_merge_v4_objective.py   be68ed0a -> 0754b171
```

**规格要求**：默认拒绝无 manifest 的 ckpt；历史 ckpt 只能在 `strict_code_hash=False` 下加载，且必须打 WARNING。

**落地方式（强制契约）**

1. 新字段名 `source_manifest`（`runtime.py:37` 的返回值原样写入 ckpt），**不是** `training_sources`；`training_sources` 保留（封存计划审计用），不参与哈希校验。
2. `validate_merge_checkpoint(..., strict_code_hash=True)`：
   - `"source_manifest" not in saved` → `raise ValueError("checkpoint has no source_manifest; ...")`；
   - 逐文件比对 `CRITICAL_MERGE_FILES`；失配报 `f"{path}: ckpt {saved[:8]} != code {code[:8]}"`；
   - `saved["source_manifest"]` 缺某关键文件同样视为失配。
3. `strict_code_hash=False`：**必须** `print(..., file=sys.stderr)` 且文本含 `WARNING` 与 `historical diagnosis only`；允许加载但不得写 `evaluation_eligible=True` 的产物路径（契约：调用方在 `strict_code_hash=False` 时把 `evaluation_eligible` 视为 `False`，t8 的命令矩阵据此标注）。
4. CLI：`--strict-code-hash {true,false}` 默认 `true`（t3 负责 eval 端，t8 负责矩阵标注）。
5. **后果明确**：v5 及更早的 ckpt 在默认配置下**不可加载**；这是规格要求，不接受"为了能跑 v5 而放宽默认值"。

**验收断言**（t2 的 `tools/test_checkpoint_code_hash.py`）

- manifest 一致 → 通过；
- 篡改任一 `CRITICAL_MERGE_FILES` 文件的哈希 → 拒绝且消息含两个 8 位前缀；
- 删掉 `source_manifest` 键 → 默认拒绝；
- `strict_code_hash=False` → 放行且 `capsys`/`redirect_stderr` 捕到 WARNING（含 `historical diagnosis only`）；
- 真实 v5 ckpt：`strict_code_hash=True` 拒绝、`False` 放行（用 `torch.load` 真实文件，只读）。

---

### C5 废除强制单位 RMS：tanh 软有界（作废旧 ckpt）

**现状（file:line）**

- `gated_slot_merge.py:173-181` `_group_delta`（v3/v4/v5 共享基类）：
  `delta * torch.rsqrt(delta.square().mean(-1, keepdim=True) + _DELTA_VAR_EPS) * carrier_rms`
  → 把**任意**幅度（含零初始化的 0）归一化到载体 RMS，零处增益 `1/sqrt(1e-6)=1e3`。
- `native_skip_merge.py:64-93` `NativeSkipSlotMerge._group_delta`（v5 native）：`:93` 同样的 `rsqrt(mean+eps)*carrier_rms`。
- `consolidate.py:394` `_DELTA_VAR_EPS = 1e-6`；`:552-553` 载体重残差同样的 `rsqrt` 形式。
- v6 已改为**单边 clamp**（`gated_slot_merge.py:351-354`、`:546`），但规格要求的是 **tanh**，且要求从共享基类废除强制单位 RMS。

**规格要求**：`carrier_rms * tanh(raw_delta / carrier_rms)`；`up_weight=0` 时精确为 0；不放大背景噪声；全局 5% RMS 预算（`_cap`）不得放宽。

**落地方式**

1. 模块级助手（`gated_slot_merge.py`，供基类、v5 native、v7 共用）：

```python
def _soft_bound(delta, carrier):
    rms = carrier.square().mean(-1, keepdim=True).sqrt()
    return rms * torch.tanh(delta / rms.clamp_min(_DELTA_VAR_EPS))
```

   - `|out| <= rms` 逐元素成立（tanh 值域）；
   - `delta=0` → `0.0`（精确，非近似）；
   - 零点导数 `d out/d delta = 1`（不再有 `1e3` 放大）；
   - `clamp_min` 而非 `+eps`：正常幅度下分母就是 `rms`，不引入虚假缩放。
2. `gated_slot_merge.py:180-181` → `return _soft_bound(delta, carrier_layer.float().flatten(0, 1))`。
3. `native_skip_merge.py:90-93` → `combined = adapted + torch.tanh(gain) * native`；`return _soft_bound(combined, carrier)`。
   （`torch.tanh(gain)` 把 native 支路增益限在 `(-1,1)`；`native_gain=0` 时 `tanh(0)=0`，step 0 仍精确等于 Selection。）
4. **v6 子类不动**：`GraftedSlotMemoryMergeV6._group_delta`（`:336-354`）与 `NativeSkipSlotMergeV6._group_delta`（`:524-547`）保持单边 clamp，旧 v6 ckpt 可只读复现（C6）。
5. **接受后果**：v3/v5 的 Dsup/Dnative ckpt 数值语义作废（`_group_delta` 变了），v7 必须重训。旧 ckpt 在 `strict_code_hash=True` 下会被 C4 直接拒绝，因此不会"静默算出不同数字"。

**受影响既有测试（已逐条核对，结论：无需修改断言，但必须全量复跑）**

| 测试 | 断言 | tanh 后 |
|---|---|---|
| `test_natural_merge_v3.py:38-52` | `torch.equal(out, source[:,20:])`（零初始化恒等） | 通过（`up=0 → delta=0 → +alpha*0`，浮点加 0 精确） |
| `test_natural_merge_v3.py:54-72` | `ratio <= .050001` 且 `ratio > 0`；q/k 梯度 > 0 | 通过（`_cap` 未动；tanh 零点导数为 1） |
| `test_natural_merge_v3.py:87-102` | 分块一致性 `allclose(1e-6)`、`state_digest` 相等 | 通过（逐元素变换，无跨槽耦合） |
| `test_natural_merge_v4.py:25-53` | 恒等、梯度 > 0、`ratio <= .050001`、掩码退化 | 通过 |
| `test_natural_merge_v4.py:96-113` | `train_event` 有限、`stored_residual_rms_ratio <= .050001` | 通过 |
| `test_natural_merge_v6.py:30-58/100-127` | v6 单边 clamp 语义 | 通过（v6 未改） |

全仓 `tools/test_*.py` 中**没有任何断言**编码"强制单位 RMS"（`grep -n "carrier_rms" tools/test_*.py` 无命中；`_group_delta` 仅 v6 测试直接调用），因此不存在"需要更新的旧断言"；风险在于数值漂移，用全量 unittest 兜底（基线 112 项，见附录 B）。

**验收断言**（t5 的 `tools/test_soft_bound_gating.py`）

- `up_weight=0` → `out` 与 `select_block_compact` 逐位相等；
- `|out - carrier| <= carrier_rms * (1+1e-6)`，对 `up_weight` 放大 1e3 倍仍成立；
- 微小 delta（`1e-6 * rms`）→ `out - carrier ≈ delta`（相对误差 < 1e-3），证明无 `1e3` 放大；
- 梯度有限：`up_weight` 在零初始化处 `grad` 有限且非 NaN（回归 `consolidate.py:536-551` 记录的 NaN 事故）；
- `_cap` 不变：`last_residual_ratio <= 0.05 + 1e-6`。

---

### C6 V6 子类处置与 v7 registry

**现状（file:line）**

- `gated_slot_merge.py:301` `GraftedSlotMemoryMergeV6`（arch `grouped_gated_sparse_slot_v6`）：分层组门 + 相似度硬掩码 + 单边 clamp。
- `:485` `SupervisedDSlotMergeV6`：**arch 名与父类相同**（沿用 v5 的命名习惯）。
- `:511` `NativeSkipSlotMergeV6`（arch `grouped_gated_sparse_native_v6`）：`native_gain` 分组标量。
- `natural_merge_runtime.py:229-240` registry；`:232-236` 注释说明 v4 已退役（名字不进 registry，旧 ckpt 会"响亮地"报错）。

**规格要求**：v7 取代 V6 成为主线；registry 命名明确；旧 V6 保留**只读加载**能力，且不得与 v7 混用。

**落地方式**

1. 新类（t5，`gated_slot_merge.py` / `native_skip_merge.py`）：
   - `GraftedSlotMemoryMergeV7(GraftedSlotMemoryMergeV6)`，`architecture = "grouped_gated_sparse_slot_v7"`：
     继承 v6 的分层组门 + 相似度硬掩码；**覆写** `_group_delta` 为 `_soft_bound`（C5）；`forward` 中 `alpha` 乘稀疏活跃掩码（C9）。
   - `SupervisedDSlotMergeV7(GraftedSlotMemoryMergeV7)`，同名 arch（沿用 v6 习惯）。
   - `NativeSkipSlotMergeV7(SupervisedDSlotMergeV7)`，`architecture = "grouped_gated_sparse_native_v7"`：`combined = adapted + tanh(native_gain)*native`，再 `_soft_bound`。
   - **不得**改动 v6 三个类的任何数值路径（只读复现）。
2. registry（t6，`runtime.py:229-240`）新增两行；**v6 两行保留**（只读加载）。
3. `training_for`（`:258-269`）新增 `_v7` → `TRAIN_V7`；`build_merge` 的 v5/v6 追加 kwargs 分支扩为 `("_v5","_v6","_v7")`。
4. 隔离：`validate_merge_checkpoint` 的 `module_configuration["architecture"] == architecture` 与 `training == TRAIN_Vx` 已能拒绝跨代；C4 的 manifest 哈希再加一层。契约要求：v6 ckpt 用 v7 架构名加载必须失败，反之亦然（t6 测试）。
5. 变体名：`DsupV7` / `DnativeV7`；计划 schema `natural_merge_v7_training_plan/1`；`TRAIN_V7` 由 `prepare_natural_merge_v7.py` 产出，且**不得**修改 `TRAIN_V4/TRAIN_V6`（既有测试对它们做深度相等断言：`test_natural_merge_v6.py:142-145`）。

**验收断言**

- `build_merge("grouped_gated_sparse_slot_v7", ...)` 返回 `SupervisedDSlotMergeV7`；`gamma_max == .20`；`num_groups == 8`。
- `build_merge("grouped_gated_sparse_slot_v6", ...)` 仍返回 v6 类（只读加载路径未断）。
- v6 ckpt ↔ v7 架构互斥：两个方向都 `assertRaises(ValueError)`。
- `TRAIN_V4` 与 `TRAIN_V6` 在改动后与改动前 `==`（用 `copy.deepcopy` 前后比对）。

---

### C7 "step-0 native_gain 梯度 1.4e5"：待验证，不得写成事实

**现状**：该数字未在任何代码/文档中被测过。机制上可解释但**未证实**：
- `native_skip_merge.py:93` 的 `rsqrt(mean(combined²)+1e-6)` 在 `combined=0` 处给出 `1/sqrt(1e-6)=1e3` 的增益；
- 若 native 支路自身量级为 `1e2`，`d loss/d native_gain ≈ 1e3 × 1e2 = 1e5` 量级 —— 与 1.4e5 **相容**，但这是量级推断，不是测量。

**契约规定**

1. 本文件、报告、commit message 中一律写作"**待 t9 dry-run 实测确认或否证**"。
2. t9 必须在 CPU 上实例化 v7 模块（或其缩小替身），前向+反向一次，记录 `native_gain.grad` 与 `up_weight.grad` 的范数，并**按实测写结论**（确认/否证/量级不同）。
3. 若实测否证，t9 与 t11 需同时修正本文件 C7 与最终报告的措辞。
4. tanh 改造（C5）后预期零点增益从 `1e3` 降为 `1`，因此**v7 上不应再出现 1e5 量级**；若仍出现，说明另有放大源，t9 必须定位到 file:line。

**验收断言**：`runs/dryrun_v7_<ts>.json`（t9）含 `step0_grad_norms: {native_gain, up_weight}` 与 `claim_1.4e5: confirmed|refuted|different`。

---

### C8 滑动微融合 13→9

**现状（file:line）**

- `public_retention_eval.py:116-120` `budget_geometry`：硬编码 `{8:(9,4), 10:(11,5), 12:(11,5), 20:(19,9)}`。
- `:122-153` `StreamingMemory.__init__`：`:146-148` **用 `budget_geometry` 覆盖调用方传入的 `window_blocks/target_blocks`**，因此 CLI 传 13/9 无效。
- `:149-151` 的合法性检查是**死代码**：`if not 0 < self.target < self.window and self.target <= self.budget and ...` —— 运算符优先级使第一个合取项恒为 `False`（合法几何时），整条永不触发。
- `:190-196` 触发条件：`m.used + m.block > self.budget` 且 `m.used >= self.window`。
- `longbench_prlmc_eval.py:268-273` identity 的 `window_blocks/target_blocks` 取自 `budget_geometry`；`:391-396` 只有 `--budget-blocks`，没有 `--window-blocks/--target-blocks`（t3 负责放开）。

**规格要求**：`--window-blocks 13 --target-blocks 9` 全链路可用；满 14 块触发，前 13 块压成 9；与 `budget_blocks` 的合法性约束（`0 < target < window <= budget` 的实际口径）明确。

**落地方式**

1. **解耦几何**：`StreamingMemory.__init__(manager, compactor, arm, budget_blocks=20, window_blocks=None, target_blocks=None)`：
   `None` → `budget_geometry(budget_blocks)`（旧行为，19/9 不变）；显式传入 → 使用显式值。
   校验（真正生效的写法）：`0 < target_blocks < window_blocks`、`target_blocks <= budget_blocks`、`window_blocks <= manager.total//manager.block`，否则 `ValueError`。
2. **触发规则统一**：`used >= window` 且 `used >= min(budget_blocks, window_blocks+1)`。
   等价性证明（对四个既有几何逐格相同，因此既有结果语义不变）：

   | budget | window | target | 旧触发（`used+1>budget` ∧ `used>=window`） | 新触发 `min(budget,window+1)` | 结果 |
   |---|---|---|---|---|---|
   | 8 | 9 | 4 | used≥8 ∧ used≥9 → 9 | min(8,10)=8，used≥9 → 9 | 同 |
   | 10 | 11 | 5 | used≥10 ∧ used≥11 → 11 | min(10,12)=10，used≥11 → 11 | 同 |
   | 12 | 11 | 5 | used≥12 ∧ used≥11 → 12 | min(12,12)=12，used≥11 → 12 | 同 |
   | 20 | 19 | 9 | used≥20 ∧ used≥19 → 20 | min(20,20)=20，used≥19 → 20 | 同 |
   | 20 | **13** | **9** | used≥20 ∧ used≥13 → 20（错） | min(20,14)=**14** | v7 新几何 |

   13→9 稳态：`used=14` 时压缩窗口 13 块 → 9，尾部保留最新 1 块，`used=10`；此后每 4 次写入再触发一次。
3. CLI（t3）：`--window-blocks`、`--target-blocks` 默认 `None`（回落到 `budget_geometry`），显式给出时覆盖并写入 identity。
4. 训练侧几何（t6）：`natural_merge_v4_objective.py:130` 硬编码 `window, target = 19*m.block, 9*m.block`，`:135-137` 的选择、`:157` 的 `_compacted_mask`、`:158` 的 `_pool_in_place` 全部依赖它；13→9 训练必须把这四处与计划里的 `single_window_starts`/`recurrent_event_before_writes` 一起参数化，**不得**让训练窗口与评测窗口不一致。
5. `install`/`MergeStream` 的既有约束 `0 < target < window <= m.used`（`runtime.py:160`）对 13/9 成立（14 满窗时 `window=13 <= used=14`）。

**验收断言**

- `test_geometry_table`：对 8/10/12/20 逐格断言触发块号与旧实现相同（用现有 `test_natural_merge.py:195-207` 的 E1 序列作回归：`before_write == [20,30,40,50,60]`）。
- `test_micro_fusion_13_9`：`budget_blocks=20, window_blocks=13, target_blocks=9`，写入 14 块后 events 出现一次压缩，`source_slots=13*block`、`target_slots=9*block`、`remaining_slots=10*block`。
- `test_geometry_validation`：`target>=window`、`window>budget`、`window>total` 三种都抛 `ValueError`（替换 `:149-151` 的死代码）。

---

### C9 稀疏高信噪比门控

**现状（file:line）**

- v5 `forward`：`:268-278` 计算三特征并逐块 z-score：`gate_in`（hidden 去均值/除标准差）、`redundancy`、`sharpness = probabilities[:,0]-probabilities[:,-1]`、`affinity = top1`；`:275` `rel = stack((affinity, sharpness, redundancy), -1)`，`:276` 逐块 z-score；`:277-279` `gate_logits = slot_gate(gate_in) + rel_gate(rel)`，`alpha = gamma_max * sigmoid(gate_logits)`。
- v6 `forward`：`:450-461` 同样的三特征与 z-score，`:459-461` 分组门 + `alpha = alpha * sim_reject`。
- **不存在** `_slot_alpha` 方法（任务描述里的名字是规格侧命名，仓库里 alpha 是内联的）。

**规格要求**

- `active_match_weight = sigmoid(4*(top1_affinity_z + 0.15))`；
- `alpha = clamp(base_alpha * active_match_weight, 0, gamma_max)`；
- 目标：约 Top-25% 活跃槽位，局部写入可达 0.10–0.20；背景槽位 `alpha ≈ 0`；
- 必须用仓库现有几何特征，不得臆造。

**落地方式**

1. 特征锚点：`affinity_z == rel[..., 0]`（v5 `:275-276`、v6 `:457-458`，`rel` 已 `detach()`）。契约固定这个下标含义，禁止改 `stack` 顺序。
2. 抽出模块级助手（v5/v7 共用；v6 不调用，保持冻结）：

```python
def _active_match_weight(affinity_z):
    return torch.sigmoid(4.0 * (affinity_z + 0.15))
```

3. v5 与 v7 的 `forward`：`alpha = torch.clamp(alpha * _active_match_weight(rel[..., 0]).unsqueeze(-1), 0.0, self.gamma_max)`（v7 的 alpha 形状 `[slots, groups]`，掩码 `[slots]` → `unsqueeze(-1)`）。
4. 掩码必须 `detach()`（沿用 `:275` 的既有约定）：离散可靠性决策，不允许门参数把掩码学回去。
5. telemetry（`last_fusion`）新增：`active_slot_fraction`（`weight>0.5` 的槽位占比）、`alpha_active_mean`（`weight>0.5` 上的 alpha 均值）、`sparse_gate_form`。t9/t10 用它们核对"约 Top-25% / 0.10–0.20"的目标（**目标值不是单元测试断言**，是训练期观测量）。
6. step-0 等价性不受影响：`up_weight=0 → delta=0`，与 alpha 取值无关。

**验收断言**

- `test_background_slots_silent`：`affinity_z=-3` → `weight ≈ 1.1e-5`，`alpha < 1e-4*gamma_max`；
- `test_high_snr_slots_pass`：`affinity_z=+2` → `weight > 0.999`；
- `test_alpha_clamped`：任意输入下 `0 <= alpha <= gamma_max`；
- `test_mask_detached`：`weight.grad_fn is None`；
- `test_active_fraction_telemetry`：随机特征下 `0 <= active_slot_fraction <= 1`。

---

### C10 靶向恢复权重

**现状（file:line）**

- `natural_merge_v4_objective.py:53-54`：`raw = 1 + min(2, max(0, selection_ce - full_ce)) if recover else 0.`（上限 3，仅按 cohort 门控）。
- `:56-58` 按 context 归一（`average` 与 `counts`），`:162` 调 `balanced_coefficients`。
- 公式字符串被封存在计划里：`prepare_natural_merge_v3.py:26`
  `recovery_weight="1 + clamp(CE_Selection - CE_Full, 0, 2); mean one within noncarrier cohort"`。
- teacher KL 资格阈值独立：`prepare_natural_merge_v2.py:25` `teacher_max_gold_ce=2.0`。

**规格要求**：`ce_full <= 2.5` 且 `gap = ce_selection - ce_full > 0.25` 时 `raw_weight_q = 1 + 3*min(gap, 2)`（最高 7x）；否则 `0.25` 背景锚定；teacher KL/CE 复合项不变。

**落地方式**

1. 公式（v7 计划专属）：
   ```
   gap = ce_selection - ce_full
   raw = 1 + 3*min(gap, 2)      if recover and ce_full <= 2.5 and gap > 0.25
   raw = 0.25                    otherwise
   ```
   边界：`gap=0.25` → `1.75`；`gap=2` → `7.0`；`gap>2` → `7.0`；`ce_full=2.5` 含（`<=`）。
2. **必须由计划版本驱动，不能改成全局常量**：`TRAIN_V3:26` 的 `recovery_weight` 字符串是封存计划的一部分，`validate_v3_plan/validate_v4_plan/validate_v6_plan` 都对 `training` 做深度相等断言。因此：
   - `TRAIN_V7` 里写新字符串（`"1 + 3*clamp(gap,0,2) gated by ce_full<=2.5 and gap>0.25; else 0.25 background anchor"`）；
   - `balanced_coefficients` 增加 `plan` 参数（默认 `TRAIN_V4`，保持既有行为逐位不变），v7 传 `TRAIN_V7`。
3. 与 teacher 资格的关系：`teacher_max_gold_ce=2.0` **不变**；2.5 只用于恢复权重门控，因此存在 `2.0 < ce_full <= 2.5` 的样本"有权重、无 KL teacher"——这是规格要的效果，不是 bug，需在计划字符串里写明。
4. `protect` cohort 现在得到 `0.25` 而非 `0`：`objective_terms`（`:102`）的 `recovery = coefficients["recovery"]*ce` 因此对 protect 样本也非零；`protection` 项不变。契约要求 t6 在计划里显式记录该行为变化。

**验收断言**（`tools/test_natural_merge_v7.py`）

- `gap=0.25, ce_full=2.0` → `raw==1.75`（严格大于阈值的边界：`gap=0.25` 不满足 `>0.25` → 应为 `0.25`；**契约取严格大于**，测试按 `gap=0.2501 → 1.75`、`gap=0.25 → 0.25` 两条断言）；
- `gap=3, ce_full=1.0` → `7.0`；
- `gap=1.0, ce_full=2.51` → `0.25`；
- `ce_full=2.5, gap=1.0` → `4.0`（含边界）；
- 归一化后 `sum(weight)/nr` 口径不变（`test_natural_merge_v4.py:64-69` 的 context 平衡断言仍成立）。

---

### C11 CPU 契约测试清单与文件

**基线（实测，2026-10-20，HEAD d5e51e0）**

```
CUDA_VISIBLE_DEVICES="" /share/latent/mllm_env/bin/python -m unittest discover -s tools -p "test_*.py"
Ran 112 tests in 18.498s
FAILED (failures=1)
FAIL: test_budget_and_fused_report.FusedSubsetReportContracts.test_bootstrap_ci_bounds_and_degenerate
  AssertionError: 0.09999999999999996 != 0.1   (tools/test_budget_and_fused_report.py:113)
```

→ **基线不是全绿**：112 项中 1 项预存失败（浮点相等断言），属 t3 的写入范围（`test_budget_and_fused_report.py`）。t7 报告"全绿"前必须先修掉它，且**不得**把它算作 v7 引入的回归。

**契约测试矩阵**

| # | 契约 | 文件（归属） | 断言要点 |
|---|---|---|---|
| 1 | step-0 等价 Selection | `tools/test_soft_bound_gating.py`（t5） | v7 模块输出与 `select_block_compact` 同源逐位相等（`torch.equal`），fp32 与 fp16 各一次 |
| 2 | 源屏蔽退化 Selection | `tools/test_soft_bound_gating.py`（t5） | `source_allow_mask` 只标载体 → 输出逐位等于载体；**全 False 且载体非空仍是 `ValueError`**（`gated_slot_merge.py:205-206` 的既有校验不得放宽） |
| 3 | 状态哈希确定性 | `tools/test_natural_merge_v7.py`（t6）+ 既有 `test_natural_merge_v3.py:101` | 同权重两实例 `state_digest` 相等；`"block_ages" not in module.state_dict()`；`_block_ages` 不入 `state_digest` |
| 4 | 首轮选块逐位一致 | `tools/test_block_ages.py`（t4） | `block_ages=None` / `zeros` / 分代分支（全 0）三者 `torch.equal`，且等于附录 A 黄金向量 |
| 5 | tanh 有界性 | `tools/test_soft_bound_gating.py`（t5） | `|out-carrier| <= carrier_rms`；`up=0 → 精确 0`；微小 delta 不放大；零点梯度有限 |
| 6 | 分代配额上限 | `tools/test_block_ages.py`（t4） | 老 ≤ 4、新 ≥ 5、总数 = 9、新块不足时回填到 9 |
| 7 | 年龄滑动/合并/淘汰 | `tools/test_block_ages.py`（t4） | roll 映射、evict+append 净效果、merge 置换、install 载体 +1、reset 清零 |
| 8 | 13→9 几何与旧几何回归 | `tools/test_natural_merge_v7.py`（t6） | 14 块触发、`used=10`；19/9 的 `before_write` 序列不变 |
| 9 | manifest 强校验 | `tools/test_checkpoint_code_hash.py`（t2） | 一致通过 / 篡改拒绝 / 缺失拒绝 / 非严格放行 + WARNING |
| 10 | objective 权重公式 | `tools/test_natural_merge_v7.py`（t6） | 四个边界（C10 验收） |
| 11 | v6/v7 互斥 | `tools/test_natural_merge_v7.py`（t6） | 两方向 `ValueError`；v6 registry 仍在 |
| 12 | 计划完整性 | `tools/test_natural_merge_v7.py`（t6） | `validate_v7_plan` 拒绝坏 schema/篡改 `content_sha256`；`TRAIN_V4/TRAIN_V6` 深度相等不变 |

**执行口径**：`CUDA_VISIBLE_DEVICES=""`，禁止 GPU；t7 记录 `Ran N tests` 与 `OK`；t7 的 dry-run 产物写 `/tmp` 或 `experiments/dryrun/`，不写 `runs/`。

---

## 3. 裁决记录（规格优先）

| ID | 议题 | 裁决 | 依据 / 影响 |
|---|---|---|---|
| D-01 | 仓库现状 vs 规格 | 规格为准；仓库现状只用于定位改法 | 任务总原则 |
| D-02 | 首轮逐位一致的实现方式 | **双重保险**：`block_ages` 为 None/全 0 时早退到原路径；分代分支用 `score * age_decay**age`（`0.75**0` 精确为 1.0） | 早退保证未来改动不会悄悄改变首轮；乘 1.0 保证分代分支自身也逐位一致 |
| D-03 | `age_decay` 的语义 | `age_decay` 是**分数衰减因子**，`age_decay**age` 乘到综合分数上；`age` 是"经历过的合并轮数"（整数） | 规格只给了 `age_decay=0.75` 与"分代配额"；衰减乘子是最小且可测的解释。若用户规格另有所指，**只改这一处**（`_select_block_indices` 内一行） |
| D-04 | 合并/寻址写后载体的 age | **+1**（幸存一轮），**不是** 0；`_merge_into_block` 的同块刷新**保持**原 age | 合并输出是老内容的融合，若置 0 会让旧内容"洗白成新世代"，直接击穿 `max_old_ratio` 的意义；同块刷新没有产生新块 |
| D-05 | 配额与贪心的交互 | 在 `_greedy_diverse` 循环内以 `group & available` 掩码实现老块上限；掩码后无候选则**当轮回填** | 先选后修正会破坏 diversity 的 max-min 语义；循环内掩码不改变"配额不触发"时的贪心次序 |
| D-06 | `_block_ages` 与 state hash | **不进** `state_dict`/`state_digest`；只进 `stats()`/events 诊断字段 | 保持既有"状态哈希确定性"契约；t4 的测试据此断言 `not in state_dict` |
| D-07 | 13→9 触发规则 | `used >= window` 且 `used >= min(budget, window+1)` | 对 8/10/12/20 四个既有几何**逐格等价**（§C8 表），故不改变既有结果语义；13/9 得到 14 |
| D-08 | `StreamingMemory` 覆盖 window/target | 停止覆盖：显式参数优先，`None` 才回落 `budget_geometry`；并修掉 `:149-151` 的死校验 | 否则 CLI 的 `--window-blocks/--target-blocks` 永远无效 |
| D-09 | tanh 改在哪一层 | 改**共享基类** `gated_slot_merge.py:173` 与 `native_skip_merge.py:64`；v6 子类保持不动 | 规格明确要求废除基类的强制单位 RMS；v6 覆写使其天然隔离，旧 v6 ckpt 仍可只读复现 |
| D-10 | v7 与 v6 的关系 | v7 = 新类 + 新 arch 名（`..._v7`），v6 类与 registry 条目保留（只读） | 互斥可被 `module_configuration`/`training`/manifest 三层拒绝 |
| D-11 | 历史 ckpt 的默认处置 | 默认拒绝；`strict_code_hash=False` 显式放行 + stderr WARNING（含 `historical diagnosis only`） | v5 实测无 `source_manifest`，且 `training_sources` 已 4/40 失配 |
| D-12 | `install` 的向后兼容 | `selected_blocks=None` 时**不动** ages | 保持既有 3 参调用与既有测试逐位不变 |
| D-13 | "1.4e5" | 只作**待验证假设**，由 t9 实测裁决 | 任务要求；且 C5 的 tanh 改造预期把零点增益从 1e3 降为 1 |
| D-14 | 稀疏门控的掩码梯度 | 掩码 `detach()` | 与既有几何特征（`:275`）一致；防止门参数把可靠性掩码学回去 |
| D-15 | 恢复权重公式的作用域 | 由 `TRAIN_V7` 驱动，默认参数保持 `TRAIN_V4` 行为 | 封存计划的深度相等断言不允许原地改公式 |

---

## 4. 下游写入范围矩阵（避免互相踩踏）

| 文件 | t2 | t3 | t4 | t5 | t6 | 冲突处置 |
|---|---|---|---|---|---|---|
| `tools/natural_merge_runtime.py` | ✔ | | ✔ | | ✔ | 串行：t2 → t4 → t6；t6 不得回退 t2/t4 改动 |
| `methods/prlmc/layout.py` | | | ✔ | | | |
| `methods/prlmc/consolidate.py` | | | ✔ | | | 选择器与 `_soft_bound` 助手同文件：t4 先落，t5 只改 `gated_slot_merge/native_skip_merge` |
| `methods/prlmc/gated_slot_merge.py` | | | | ✔ | | v7 类在此定义（t5），registry 在 runtime（t6） |
| `methods/prlmc/native_skip_merge.py` | | | | ✔ | | |
| `tools/public_retention_eval.py` | | | | | ✔ | 不在 t4 范围，但 C2/C8 要求它更新 ages/几何 → **t6 承担** |
| `tools/longbench_prlmc_eval.py` | | ✔ | | | | |
| `tools/natural_merge_v4_objective.py` | | | | | ✔ | |
| `tools/prepare_natural_merge_v7.py` / `natural_merge_v7_train.py` | | | | | ✔ | 新建 |
| `tools/test_budget_and_fused_report.py` | | ✔ | | | | 含预存失败断言（:113），t3 顺手修 |

依赖顺序（按契约）：`t2 → t4 → {t5, t6} → t7 → {t8, t9, t10} → t11`。
t5 与 t6 可并行，但 t6 注册 v7 类名依赖 t5 的类存在，故 t6 的 registry 测试应在 t5 落地后复跑。

---

## 5. 未决项与风险

| ID | 风险 | 影响 | 处置 |
|---|---|---|---|
| R-01 | `age_decay` 的确切数学形式只有规格侧知道（D-03） | 若规格是"age 衰减阈值"而非"分数乘子"，C1 的验收会偏 | 契约把改动面收敛到 `_select_block_indices` 一行；captain 可向用户确认后单点替换 |
| R-02 | `public_retention_eval.py` 不在 t4 的写入范围，但 C2 要求它维护 ages | 若不处理，评测路径的 ages 永远为 0，分代配额在评测中失效 | 已把该文件划给 t6；t4 必须提供 `note_append/note_merge` API 供其调用 |
| R-03 | `StreamingMemory.compact` 自建写回路径，不调 `install` | 年龄更新会漏一条路径 | 契约要求两条路径都调 `manager.note_merge`（C2 落地第 5 点） |
| R-04 | 基线 1 项预存失败 | t7 的"全绿"门槛无法直接达成 | t3 修 `test_budget_and_fused_report.py:113`（改 `assertAlmostEqual`） |
| R-05 | 13→9 训练几何与评测几何不一致 | 训练-评测错配，v7 结果不可解释 | C8 第 4 点强制四处参数化 + 计划常量同步 |
| R-06 | v5 同轴复核（t8）依赖 `strict_code_hash=false` | 默认拒绝后，v5 复核必须显式非严格 | t8 的命令矩阵逐条标注（已在 t8 描述中） |
| R-07 | `_block_ages` 与 `snapshot/restore` | `compact()` 失败回滚时 ages 可能已改 | 契约：`snapshot()`/`restore()`（`layout.py:359-386/388-412`）必须包含 `_block_ages` 的 clone/copy，否则回滚后的 ages 与池不一致 |
| R-08 | fp16 下的 tanh 有界性 | `rms*tanh(x/rms)` 在 fp16 可能因舍入略微越界 | 测试容差取 `1+1e-6`；`_cap` 仍是最终兜底（`<=0.05`） |

---

## 附录 A：首轮选择黄金向量（HEAD d5e51e0 实测）

固定 `torch.manual_seed(3)`、`source = torch.randn(4, 38, 8)`、`block=2`、`target=18`（19 块选 9 块）：

| 配置 | `_select_block_indices` 返回 | 复算稳定 |
|---|---|---|
| `weights=None, goal=None, select_weights=None` | `[9, 10, 11, 12, 13, 14, 15, 17, 18]` | ✓ |
| `weights=randn(38).abs(), goal=None` | `[4, 9, 10, 12, 13, 15, 16, 17, 18]` | ✓ |
| `weights=randn(38).abs(), goal=randn(8)` | `[4, 7, 9, 10, 12, 13, 15, 16, 17]` | ✓ |
| `select_weights={"importance":.1,"recency":.9,"goal":0,"diversity":0}` | `[10, 11, 12, 13, 14, 15, 16, 17, 18]` | ✓ |

t4 的 `test_first_round_bit_identical` 必须逐行断言这四条（`block_ages=None` 与 `zeros(19)` 两种入参各跑一遍）。

## 附录 B：证据与命令

1. **仓库状态**：`git log --oneline -3` → `d5e51e0 / 872c09e / 424b024`；`git status --porcelain -uno` 为空。
2. **v5 ckpt 键**：见 C4；`torch.load(..., weights_only=False)`，40 项 `training_sources`，4/40 与 HEAD 失配。
3. **基线测试**：见 C11；`Ran 112 tests`、1 failure（`test_budget_and_fused_report.py:113`）。
4. **选择器黄金向量**：见附录 A；同一输入两次调用 `torch.equal == True`（无 RNG）。
5. **tanh 前的零点增益**：`_DELTA_VAR_EPS = 1e-6`（`consolidate.py:394`）→ `1/sqrt(1e-6)=1e3`；这是 C7 待验证数字的机制来源，**不是**测量值。

## 附录 C：逐条 file:line 索引

- 选择器：`consolidate.py:115,122-127,171-201,204-211,230-269,310-312,587-589`
- 载体重残差：`consolidate.py:394,397,501-505,512-553,587-589,657`
- 写入/布局：`layout.py:105,226,241-260,329-357,359-408,511-654,690-764,766-808,810-827,829-842,1096-1204,1206-1236,1358-1389,1393-1420`
- 门控与 v5/v6/v7：`gated_slot_merge.py:93,96-133,135-167,173-181,183-298,301-354,356-375,377-482,485,511-547`
- native 支路：`native_skip_merge.py:19,45,59,64-93`
- runtime：`natural_merge_runtime.py:15-34,37-38,88-96,98-150,153-170,223-255,258-269,272-283,286-327,330-343`
- 流式写入与几何：`public_retention_eval.py:116-120,122-153,155-184,186-222`
- 评测 CLI：`longbench_prlmc_eval.py:167,240-300,391-396`
- objective：`natural_merge_v4_objective.py:53-58,100-114,130-137,162-163`
- 计划常量：`prepare_natural_merge_v2.py:18-25`、`prepare_natural_merge_v3.py:14-27`、`prepare_natural_merge_v4.py:16-41`、`prepare_natural_merge_v6.py:19-49`
