# C 组风险感知保护：固定方案 v1

本协议对应实验已完成；最终结果、判门与归档见 [本轮总报告](S1d-round-20261013-results.md)。
2026-10-12，运行前固定协议。检验在原 C 梯度基和尾部训练目标上，仅加入写入事实保护是否提高净恢复。原 C 弱窗口 R=6/H=5/net=+1；强窗口 R=10/H=1/net=+9。

后续用户已授权 [四卡两线并行](S1d-two-lines-v1.md)，取代本文原串行调度与强窗口严格门槛：保留同一训练机制，强弱并行，强窗口最新推进标准net≥7并单独报告H。本文保留初始预注册历史，不应再按旧启动器重复派发强窗口。

## 机制与固定项

保护候选来自写入事实 latest-owner 去重后的 kept 块。复用 C 缓存的完整 D0 事实答案 CE；只保留有限、非负且 ≤2.0 的 CE，按 CE 降序、key 升序选前20条，不足20用全部，无候选则失败。阈值2.0沿用既有事实可学习性阈值，20沿用既有保护条目规模。选择不依赖 Full CE、评测 QA 答案、QA 正确性或训练后的得分。

较高 CE 表示模型对写入事实的置信度较低，是风险代理，不能由 CE 直接认定事实答对或离错误边界的真实距离。CE>2 的事实不纳入本固定版本。保护监督用写入事实自身的答案字段，不可称为完全无答案信息。

令原 C 尾部训练行集合为 T，保护集为 P：

`L = Σ_{t∈T} CE_tail4(t;α) + λ/|P| Σ_{p∈P} ReLU(CE_answer(p;α) − CE_answer(p;D0))`

λ固定1.0，无额外margin、风险连续权重或扫描。D0 CE参考值固定，保护集合训练期间不变。保留原 C 的求和尾部梯度尺度；保护项为均值，二者不额外归一化。分别反向累加梯度后，只执行一次 AdamW step和一次范数投影。报告分别记录尾部CE求和/均值、保护hinge与总目标；损失是更新前值，QA指标是更新后值。

完全复用原 C 受损事实梯度缓存和 SVD-13（Full CE≤2.0、D0−Full≥0.10、最多40梯度、无rank回填）。reader、carrier基础参数和basis冻结；只训练13个alpha，零初始化；原C尾部最多40行及4-token遮罩不变。AdamW lr=.003、weight_decay=.01，seed=0，残差L2预算1.0。60步，每10步测量；QA256，TF-only。新增≤20条保留事实前向/反向，因此计算量增加，但原训练行数和损失尺度不变。

## 阶段与预设裁决

1. CPU回归：风险排序、无效CE、稳定去重；实际driver在checkpoint重计算下的组合梯度、保护inactive等价于C、单次AdamW/预算投影、memory恢复及reader冻结。
2. 弱窗口 `(3,17510)` 单步smoke：检查原C逐题基线/cohort、基础权重和basis哈希、事实完整集合、选集及梯度/预算；保存alpha与basis并与原C二进制逐元素核对。smoke无资格触发强窗口。
3. 弱窗口正式60步。只用最终step60判定：`net≥5 且 H≤4`。净恢复门槛来自用户要求；H≤4将“减少现有5道损伤”操作化。未过即终止，不运行强窗口、不扫λ、不叠加模块。
4. 弱过门才以完全相同机制运行强窗口 `(3,16370)` 60步；保住原C优势定义为 `net≥9 且 H≤1`。不改变λ、数量、阈值或训练参数。
5. 两个开发门槛通过后，才有资格另行制定未见context与官方生成协议；本启动器停在开发验证，不自动选context或开展官方评测。

R只计预先定义 `rescue_d0_wrong` 中新恢复；H统计全部D0-correct损伤，net=R−H。中途测量仅用于轨迹解释，不选best checkpoint。这是两个已见开发窗口上的单seed机制试验，门槛通过也不能称为统计显著或泛化成功。计算较高的保留事实CE未必覆盖真正受损的评测题，失败将终止这个保护版本。

## 代码与运行

- `tools/s1d_experiment1.py --exp1-arm C --risk-protection`：原C驱动上的独立开关；不开启时保留原ABCD行为。
- `tools/s1d_risk_protection.py`：固定选择与协议，无QA输入接口。
- `tools/test_s1d_risk_protection.py`：实际训练函数的CPU梯度验证。
- `tools/verify_s1d_risk_protection.py`：冻结C对照、事实来源/完整性、全部轨迹逐题指标、预算与最终门槛检查。
- `tools/run_s1d_risk_protection.py`：smoke→弱60→条件强60，记录命令、PID、真实退出码、脚本哈希、结果与证书；显存不足等待，不终止其他任务。

```bash
ssh v100-2
cd /share/latent/PRLM-C
PYTHONPATH=. /share/latent/mllm_env/bin/python tools/test_s1d_risk_protection.py
PYTHONPATH=. /share/latent/mllm_env/bin/python tools/run_s1d_risk_protection.py \
  --gpu 0 --prefix s1d_c_risk_v1 --cache-dir /dev/shm/s1d_reader_fp16
```

日志/报告前缀 `runs/s1d_c_risk_v1`，阶段名smoke/weak/strong；launcher状态用JSON原子写入。启动前检测本机验证过的FP16缓存、所选卡≥30000MiB空闲；排除v100-2 GPU5/7。不并发运行强弱窗口，避免在弱窗口失败时消耗强窗口实验。输入仍需原C carrier、selfsup缓存、v1事实缓存、v3原C JSON和alpha二进制；复用缓存不复用训练alpha。已存在状态或输出会拒绝覆盖；不自动重启失败实验。

历史C报告没有保存尾部训练token指纹，验证器可比其行数与基础输入；本次额外保存行号和token哈希。CPU测试验证旧尾部梯度等价性。结果出来前不填入效果结论。
