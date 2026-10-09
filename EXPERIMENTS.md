# PRLM-C 实验日志

实现 *残差记忆设计_方案整理* 的完整实验记录。每条记录给出配置、命令、原始数字与结论。

基线为未修改的 MemoryLLM-8B on MemoryAgentBench，参数见 `_mllm_patch/BASELINE.md`。
本文件中的实验主要使用同一环境（`/share/latent/mllm_env`，V100-32GB）；各实验评分、探针和reader模式以正文协议为准。历史训练态LoRA dropout与本轮确定性评分、teacher-forced token-exact与官方生成EM之间不作直接数值比较。

本文件只做索引。每条实验的正文在 `experiments/` 目录下，一实验一文件；本文件不随实验追加内容。

## 当前方法主线

**S1c 跨窗口裁决 FAIL，13D 功能子空间主线停线。**单窗口内（S1c-wash）corrected 13D 臂 net+9、random 崩溃 −40，方向承载功能成立；但 8 训练窗口 240 行 rescue 梯度 SVD 得到的 U_global（energy 0.6425）在 2 个 held-out 窗口四臂裁决中：global 25/21/+4、random 25/32/−7、unconstrained 29/18/+11、local 28/20/+8——rescue_gap=0 且 net 低于 unconstrained−2，预注册门 **FAIL**。单窗口正结论依赖窗口自身监督，不携带跨窗口结构；层 2（冻结 U_global + 残差预测器）的前提被否定，取消立项。S1-local hard-nullspace NO-GO、S1-soft-local NO-IMPROVEMENT 维持原判；历史 S0/S0b 训练态 LoRA dropout 随机性使其旧分数不作为本轮确定性可比对照。

E0–E7、SIC 及读出诊断归入问题发现与设计依据，不作为新方法主表。此前条件方案中的主表与预算0.25/0.5/0.75实验因local未过门均未执行。没有held-out建基或答案监督；原 SIC-P1 测试集保持封存。

## 目录（experiments/）

| 编号 | 运行 | 目的 | 状态 | 正文 |
|---|---|---|---|---|
| E0 | Baseline Tier 1 | 冻结参照分 | 已完成，5 数据集 | [E0.md](experiments/E0.md) |
| E1 | `prlmc-smoke` | 端到端链路验证 | 已完成 | [E1.md](experiments/E1.md) |
| E2 | `prlmc-obs` | 阈值标定（`compact_mode: none`） | 已完成，500 次写入 | [E2.md](experiments/E2.md) |
| E3 | `prlmc-r1` | occupancy-only 结构消融 | 已完成，**accuracy 全 0** | [E3.md](experiments/E3.md) |
| E3b | 纯代码分析 | 归零的机制根因（不运行） | 已完成 | [E3b.md](experiments/E3b.md) |
| E4 | 2×2 矩阵 | 分离 gate 幅值与 compactor | 已完成，**compactor 是充分原因** | [E4.md](experiments/E4.md) |
| E5a | `recent_blocks` | 只保留原生 block 时 reader 是否正常 | 已完成，**EM 26.00** | [E5a.md](experiments/E5a.md) |
| E5b | `select_blocks` | 打分选择能否胜过纯 recency | 已完成，**与 E5a 逐条相同** | [E5b.md](experiments/E5b.md) |
| E5c | E5b + active mask | 真正 K→K' 的无训练 compaction baseline | 已完成，**EM 23.00**（与 E5a 不再相同） | [E5c.md](experiments/E5c.md) |
| E5d | E5c，但 filler 不受限 | 排除 E5c 的两个开关混淆 | 已完成，**EM 24.00**，与 E5c 逐条 97/100 相同 | [E5d.md](experiments/E5d.md) |
| E6 | `carrier_residual` | learned fusion 是否超过 selection | 已完成，**EM 24.00**；无 ckpt，初始化即恒等，故仅验证接线 | [E6.md](experiments/E6.md) |
| E6a | 离线单窗口 overfit | 训练链能否跑通（不读 MemoryAgentBench 指标） | 已完成，单窗口 220 步恢复 2/3 | [E6a.md](experiments/E6a.md) |
| E6b-mini | 离线多窗口共享 checkpoint | 单窗口恢复能否泛化到未见窗口 | 已完成（**2026-09-29 重跑，前一轮作废**），训练 7/30 = 0.233、held-out 2/16 = 0.125，判为 `partial` | [E6b-mini.md](experiments/E6b-mini.md) |
| E6c | 训练档容量压力测试 | 50% 预算下训练过的 compactor 三臂对比 | **训练侧已完成**（2026-09-30，8×96 步），训练 5/30 = 0.167、held-out 3/16 = 0.188，retention reading 翻转为 `overwrites`；三臂 bench 未运行 | [E6c.md](experiments/E6c.md) |
| E6-gate | 压缩门（g1→g2q） | lost-probe 恢复是否 6/6，判定项目继续/停止 | **FAIL**（2026-10-01，最佳 3/6 @ rank 256）；SVD 信息可读，瓶颈在解码侧容量/优化，非信息销毁 | [E6-gate-g2.md](experiments/E6-gate-g2.md) |
| E6d | 三臂顺序消融（种子 0） | 连续窗口训练是否是遗忘原因 | 已完成（2026-10-02），interleave **56/160（+30）** vs forward 37（+11）vs reverse 33（+7），遗忘事件 6 vs 50 vs 42 | [E6d.md](experiments/E6d.md) |
| E6e | 凶手探针 | 遗忘发生时哪组参数在动 | 已完成（2026-10-02），预注册三选项全不成立，判为漂移压过信号的临界翻转，非定向覆盖 | [E6e.md](experiments/E6e.md) |
| E7 | 内容路由稀疏 residual experts | 主动知识分工（阶段 A 干扰图 → B oracle 上界 → C 内容路由） | 设计定稿（2026-10-02，正文 [E7_PLAN.md](E7_PLAN.md)），未开始运行 | [E7_PLAN.md](E7_PLAN.md) |
| E7a | 阶段 A 干扰图（seed 0） | 8 窗梯度可分性与单步功能干扰 | 已完成（2026-10-02），梯度余弦 0.05–0.32 无冲突无重复，单步 harmed=0、rescue 集中在 2 道边界题，corr(cos,rescue)≈0；一阶近似下无定向干扰，与 E6e 漂移结论一致 | [E7a.md](experiments/E7a.md) |
| E7a-24 | 阶段 A 深度扩展 + interleave 3-seed | 24 步功能矩阵；E6d 净增跨 seed 稳定性 | 已完成（2026-10-03，干扰矩阵 2/3 seed：seed 1 被 OOM 挤死待重跑），interleave 复现 +24~+30/forget_ev≤12；24 步 harmed=5 且两 seed 逐条相同、效应不对称（17130 整行零、16370 共振带）、rescue:harmed=31:5 | [E7a-24.md](experiments/E7a-24.md) |
| E7a-sim | 阶段 A 收尾判读 | 内容相似度能否预测功能干扰（router 机制前提） | 已完成（2026-10-03），内容带宽 0.9976–0.9991（同 context 8 窗在 reader 表征中为同一点），与梯度/功能信号零相关，ARI<0 → **停止门槛 1 触发：不做语义 router**，B 阶段 oracle 改测纯窗口隔离 | [E7a-sim.md](experiments/E7a-sim.md) |
| S0 | 残差子空间方案诊断 1：功能敏感子空间扰动验证 | lost probes 的 margin Jacobian 子空间是否真实承载可操作功能信号（soft filter/hard projection 的目标是否真实） | 已完成（2026-10-03），以 E6d interleave ckpt 为基点，sketch rank 13 满秩；敏感方向救回 6.625/13 lost、低敏感 0.25、随机 0，separation 0.490 单调上升 → **PASS**；附带确认初始 compactor（up=0）是数学死区（rank 0），S0 必须用 warm-up ckpt | [S0.md](experiments/S0.md) |
| S0-分区 | 敏感子空间内部功能结构 | 子空间是否按内容主题分区 | 已完成（2026-10-03），非主题聚类：**公共轴 6 探针**（被 7–8/8 方向救，两两余弦 0.63）+ 专属尾迹 5 + 不可达 2（方向存在但增益不够）；Gram top1 占 58.8%、top2 72.6%；S0b 救回的 7 个恰为公共轴+1 尾迹 | [S0.md](experiments/S0.md) |
| S0b | 系数 oracle 最小训练（C=I+E） | rank×rank 系数自由度能否救 lost 而不伤 retained | 已完成（2026-10-03），**INCONCLUSIVE**：rescued 7/13 达标（峰值 step20 达 9），harmed 3/5 超标（阈值 0 过严，n_retained=5 统计脆弱）；‖E‖ 全程钉 0.500 → 容量钳制饱和；判读为「有信号但全局系数变换不够」→ 允许进 S1 但 S1 须自带 no-harm 验证 | [S0b.md](experiments/S0b.md) |
| Native-QA | 数据集原生问答成对对照 | 完整写入、未压缩记忆 vs 同状态手动压缩记忆 vs 原生 full-rewrite；真实生成、官方 EM/SUB 重算 | 已完成（2026-10-03）：官方口径 row0 三臂 5/6/7 EM、row4 30/31/13 EM；压缩 vs 未压缩净差 ±1/100 为噪声级，full-rewrite 仅 row4 大幅掉分；根因=ckpt 来自 row3 窗口、监督与评测任务错位 | [Native-QA.md](experiments/Native-QA.md) |
| Native-QA-warmup | 任务匹配功能监督训练 | row0 原生 24 题 gold+KD 训 96 步，零重叠 row4 100 题评测 | 已完成（2026-10-03）：loss 3.95→3.01 有效，但 EM 31/100 与未训练 selection ckpt 逐题相同（γ 仅用预算 5.2%）；**判定读出/格式瓶颈 >> 压缩损失，停止压缩算子侧变体** | [Native-QA.md](experiments/Native-QA.md) |
| Readout-probe | 读出修复三臂并行（A 指令对抗 / B few-shot 锚定 / C reader 注意力增益） | 同一 row4 无损池上 10 组对照，官方两层打分 + McNemar | 已完成（2026-10-03）：**B2 few-shot+stop 唯一显著，EM 30→42（+15/−3，p=0.0075）**，增益全为先验压制题；无 stop 的 B 官方 EM 崩到 1（续写 Question/Answer 对劫持 parse 层）；C attn×8 +11/−6 不显著且增益与 B2 仅重叠 7/49；组合 BC 39<42 无增益；row0 验证无增益无损（5/97→5/97 排除泄漏），跨行未证实 | [Native-QA.md](experiments/Native-QA.md) |
| S1-SDC | 感度定向整固：冻结 reader，功能梯度显著性选择 + 一阶正交合并 + 回滚门 | 预注册G0–G5；未见题验证与协议审计优先 | G0/G1历史结果保留；G2版本机制FAIL；**G2c独立200题未过门**：128 eligible select25/sic29，p=0.424；singleton净增益为零，独占信息机制未证实。暂停由小样本直接推进G3方法有效性主张 | [S1-SDC.md](experiments/S1-SDC.md) |
| G0 | S1-SDC 第一门：版本盲区测量 | 同主体新旧事实块跨层余弦是否近 1（否则全案止损） | 已完成（2026-10-04），rows 0–3 四窗 120 版本对：**mean 0.9751、min 0.9626、对照 0.9734 与之重叠**——比预注册更强：不仅新旧近重复，且与随机对照同平，版本信息在相似度度量下不存在；文本平（E7a-sim）→ 池平且版本盲（G0）→ 行为掉分（E6 lost）三层闭环 | [G0.md](experiments/G0.md) |
| G1 | S1-SDC 第二门：显著性选择 vs 原生相似度选择 | 同窗口同探针五臂对照（select/sic/oldest/random），判据 lost 降 ≥30% + ≥3 contexts + McNemar<0.05 | 已完成（2026-10-05），4 行 76 eligible：**select 19 / sic 9 / oldest 18 / random 16**，ratio 0.474，McNemar χ²=5.06 p=0.0245；sic 四行无一更差；kept 重叠仅 4–5/9，sic 独保窗口头部老块（梯度信号在老块），select 独取中段（recency 所致）——与 G0 版本盲互证 | [G1.md](experiments/G1.md) |
| G2 | S1-SDC 第三门：版本分层，废弃块被丢率 sic vs select ≥2x | 判据字面 FAIL（0.688<2.0），机制假设按预注册证伪 | 已完成（2026-10-06），四窗口 1520 行：select/oldest 废弃丢率 0.775、现行 0.250；sic 0.533/0.542 对称不做版本区分；探针级 select 丢的 19 题里 18 题支撑在 singleton 独占行——G1 赢法更像独占信息保护而非版本正确性 | [G2.md](experiments/G2.md) |
| G2b | S1-SDC 多样探针复验：selection-independent + current/singleton 平衡 + discovery/held-out 分离 | held-out 方向复现，目标降 ≥30%；统计显著需扩大样本 | 已完成（2026-10-07），held-out 33 eligible：select 7 / sic 4，降 42.9%，ratio 0.571；但 McNemar b=4/c=1，p=0.371，方向复现、统计不充分；原始 G2 版本机制 FAIL 保留，下一步扩大 held-out 后再入 G3 | [G2b.md](experiments/G2b.md) |


| G2c | 固定200道selection-independent held-out题，全文latest gold、全局去重、discovery隔离、eval评分 | eligible lost降≥30%且双侧精确McNemar p<0.05 | **NOT_PASSED**：128 eligible，select lost25 / sic29（增16%），b=5/c=9，p=0.424；200题正确率53.5%/53.0%；singleton两臂lost18/18，撤回已证实独占保护机制的强归因；22种关系，四窗口评分重复一致 | [G2c.md](experiments/G2c.md) |

| SIC-O1 | SIC评分/覆盖选择优化：原窗口、五候选、旧200题开发与82道全新确认题 | 先开发选型后冻结独立确认，不依赖窗口头尾规则 | 开发完成：select107/200、coverage112/200；128 eligible丢25→24，仅4%，p=1.0；Taylor丢29。82确认题继续封闭，推进O2；G3暂停 | [SIC-O1.md](experiments/SIC-O1.md) |

| SIC-O2 | 训练探针资格过滤 + discovery真实压缩CE校准 | 排除与记忆保留目标无关的梯度，避免完整池一阶代理直接决定离散压缩 | Finite CE swap四窗开发完成：row1/2/3 CE改善，row0不变；82题确认总正确55→56，但64 eligible丢题10→11，净-1、p=1.0，未通过确认门槛。swap不替换O2方案，G3继续暂停 | [SIC-O2.md](experiments/SIC-O2.md) |

| SIC-P1 | 论文级受控实验：实体隔离24 episode、冻结协议、真实生成EM | 脱离旧4上下文/旧82题，主指标贪心生成EM、support oracle逼近门、episode级推断；16测试episode仍封存 | 数据冻结+smoke完成：ep0 SIC稳定性候选22/64低于native 23/64；候选未达标。按新要求暂停开发，仅保留历史诊断 | [SIC-P1.md](experiments/SIC-P1.md) |
| S1-local | Protected Functional Subspace Residual 五臂机制验证 | 同 S0 hard window，扩大 full∧select retained；U_safe 更新是否保留 rescue 并压低 harm | 正式五臂60步完成；**hard-nullspace构造NO-GO**：protected rescue1/13、harm37/75；UL为7/13、6/75、净+8。本地正证据待独立验证，历史判门不变 | [S1-local.md](experiments/S1-local.md) |
| S1-soft-local | 固定13维UL的单边有限CE保护 | 原监督与UL不变，P=31保护与K=75消融，分开rescue/repair/protect | 别名缺陷修复（`00c4880`）后重跑smoke与正式三臂60步；控制臂UL 7/6/+9（net≥8单边裁决，逐题差1题为诊断），soft_p +8、soft_k +9均无严格超越；verdict=**NO-IMPROVEMENT**，λ=1单边罚项不优于无约束UL；双端指标/权重audit通过 | [S1-soft-local.md](experiments/S1-soft-local.md) |
| S1c-basis-wash | 修正图上重建U_L的四臂对照：corrected/old/random/unconstrained | 同S1-soft协议仅换basis；预注册门含random差距；裁决别名是否污染basis本身 | formal完成verdict=**PASS**：corrected 15/0/+9、old 14/0/+8、random-13D 4/44/**−40**、unconstrained复现10/17/−7；corrected与old主角度≤0.005rad——basis未被别名污染（别名只污染梯度），13D**方向**承载功能（random崩溃排除"任意低维有效"），S1正结论在干净图上成立；阶段2（几何审计）放行 | [S1c-basis-wash.md](experiments/S1c-basis-wash.md) |
| S1c-geometry | R/P/T 111题梯度行对13D子空间的投影几何（仅分析） | 干净梯度行→z_i投影、cohort均值余弦、投影能量占比；预注册cos(g_R,ḡ_P)>0.3→不解耦 | 111行全算完：13D投影cos(R,P)=**0.9498**、R–T 0.925、P–T 0.864（全空间0.76–0.83）；**共向判定成立→不做shared/private/conflict分解（零代码）**；rescue监督顺带压低protect损伤是一阶原理而非预算副作用；停止条件②触发 | [S1c-geometry.md](experiments/S1c-geometry.md) |
| S1c-global-oracle | 8训练窗口rescue行240行SVD前13的U_global，2个held-out窗口四臂裁决跨窗口迁移（oracle α诊断） | global_13d vs random_13d vs unconstrained vs local_13d；预注册门：net≥3∧rescue_gap≥3∧net_gap≥3∧≥unc | formal完成verdict=**FAIL**：汇总global 25/21/**+4**、random 25/32/−7、unconstrained 29/18/**+11**、local 28/20/+8；rescue_gap=0且net低于unc−2——U_global方向与随机13D无区分度，单窗口正结论依赖窗口自身监督；**停止条件③触发，S1c主线（13D功能子空间跨窗口）停线**，层2（frozen residual）取消；verifier 51项全PASS | [S1c-global-oracle.md](experiments/S1c-global-oracle.md) |
| S1c-lowo-coverage | 240行梯度缓存leave-one-window-out建基，rank 13/32/64未见窗口能量覆盖率，裁决FAIL根因（秩不足/窗口差异/分布偏移三选一） | ρ(U,G)=‖UᵀGᵀ‖²F/‖G‖²F，训练均值中心化，self基为上界；verifier fp64独立重算 | 完成：LOWO ρ pooled r13=**0.324**、r32=0.374、r64=0.414，self 0.91–0.98——r13→64仅+0.09且差距不缩，**排除秩不足**；8窗口自建r13基pairwise mean align=**0.256**（近正交，随机期望≈0.01）、mean cov=0.299，**窗口差异成立**；中心化后无独立偏移分量，排除分布偏移。**根因=窗口间功能子空间近正交，任何预建全局基只能覆盖未见窗口约1/3梯度能量，停线维持且获机制解释**；verifier 29+2判定项PASS，30数值项差≤1.1e-5（fp32舍入） | [S1c-lowo-coverage.md](experiments/S1c-lowo-coverage.md) |
| S1d-selfsup-basis | 无答案窗口自监督建基 stage 1：事实行尾部遮罩补全CE的Jacobian行SVD出每窗口13D基，测对gold rescue子空间覆盖（v1模板过滤→v2扩池：移除模板过滤+latest-owner去重+40有效行/窗口，双卡分片） | 监督只用压缩前可观测事实行，held-out QA答案不触碰；gate=mean gRho≥0.20∧≥1.5×LOWO r13；SVD序基（v1 QR序缺陷已修正）；有效秩审计+rank-覆盖曲线 | v2完成：mean gRho=**0.5116**、ratio vs LOWO=**1.6226**（v1擦线1.5003），8/8窗口高于LOWO，aux回填=0（dropped primary池深199+行/窗口）；**gate PASS**但用户稳健性双标（ratio≥1.65∧有效秩≥13）未全达：有效秩均值7.41（40行时σ1/σ13=4.6–9.0，top-13占行池能量90–95%），系dropped块事实行梯度天然聚集非采样不足；v1/v2重叠203行最坏rel diff 1.24e-2（fp16非确定性）；扩池收益分解≈+0.024扩池+0.005排序修正。**stage 2（D0+U_window·α无答案优化）资格成立，待用户裁决** | [S1d-selfsup-basis.md](experiments/S1d-selfsup-basis.md) |

## 证据边界（论文推进审计）

历史82题已在O2开封，后续swap/local-oracle复用只属探索验证。support oracle最优值仅是指定文本支撑块最小遗漏，不是reader错误率下界。teacher-forced答案token完全匹配不等于官方完整生成EM；同一上下文多题不增加独立上下文数。以上边界优先于旧条目的强归因表述；G3继续暂停。

## 固定参考文件

| 内容 | 正文 |
|---|---|
| 全部超参数清单 | [experiments/params.md](experiments/params.md) |
| 已修复的缺陷 | [experiments/fixes.md](experiments/fixes.md) |
| 未解问题 | [experiments/open-questions.md](experiments/open-questions.md) |
| 复现方式 | [experiments/reproduce.md](experiments/reproduce.md) |

线性读法：E1 发现触发器不触发 → E2 标定阈值 → E3 修触发器后首次测 accuracy → **发现 accuracy 归零，根因未分离** → E4 定位到 compactor → E5a/E5b 用原生 block 选择把 accuracy 恢复到 26.00 → E5c 打开 active mask，free region 真正移出 reader 视野，EM 23.00 → E5d 放开 filler 限制以排除开关混淆，与 E5c 逐条 97/100 相同 → E6 把 compactor 换成 `carrier_residual`，因未训练而与 E5c 逐条 98/100 相同 → E6a 用离线单窗口 overfit 验证训练链；首轮打分跨度错误使全部指标作废，修复后筛出 `lost = 3` 的硬窗口并训练到恢复 2/3 → E6b-mini 把单窗口扩到 8 训练 + 2 held-out；**首轮因度量缺陷把 held-out 分子结构性固定为 0，`window_overfit` 与 0/16 全部作废；修复后重跑得训练 7/30、held-out 2/16，两侧无统计差异（Fisher p = 0.4628）** → E6c 把每窗口 Stage B 步数提到 96，训练 5/30、held-out 3/16，retention reading 从 `retained` 翻转为 `overwrites`——**schedule 才是约束，不是每窗口预算** → E6-gate 在固定硬窗口上跑压缩门（压缩后恢复率 6/6 才算过）：**FAIL，最佳 3/6**；恢复数随解码侧容量（0→64→256 rank）单调上升 0→2→3，证伪「SVD 销毁信息」，但 6/6 门未过，项目进入解释性中间地带 → E6d 固定种子把训练顺序单独切出来：interleave（每窗 4 步轮转）**56/160 净增 +30** vs forward +11 vs reverse +7，遗忘事件 6 vs 50 vs 42——连续窗口训练是遗忘主因，可被纯调度消除；3-seed 复验（E7a-24）确认 +24~+30、forget_ev≤12 稳定 → E6e 凶手探针：预注册的三个定向覆盖假设全部不成立，遗忘是参数漂移压过信号的临界翻转 → E7 设计定稿（阶段 A 干扰图 → B oracle 上界 → C 内容路由，含停止门槛）→ E7a 干扰图 1 步版：梯度余弦 0.05–0.32 无冲突无重复，单步 harmed=0——一阶近似下无定向干扰 → E7a-24 深度扩展：24 步 harmed=5 且跨 seed 逐条相同、效应高度不对称（17130 整行惰性、16370 共振带），干扰结构真实但以 rescue 为主（31:5），1 步版的"无干扰"是深度不足的伪象 → E7a-sim 内容相似度判读：**同 context 8 窗在 reader 表征中为同一点（带宽 0.9976–0.9991），与梯度/功能信号零相关，ARI<0——停止门槛 1 触发，不做语义 router**；阶段 B 的 oracle 臂改为测纯窗口隔离，C 阶段唯一存活的 router 候选是功能梯度路由 → S0 残差子空间诊断 1：以 E6d interleave ckpt 为基点做扰动验证，敏感方向救回 6.625/13 lost（separation 0.490 单调上升）而低敏感/随机方向几乎救不回，**功能敏感子空间真实且强可操作，PASS**；附带确认初始 compactor 是数学死区 → S1-local hard-nullspace 五臂 **NO-GO**（protected 1/13 vs UL 7/13）→ S1-soft-local 单边罚项 **NO-IMPROVEMENT**（+8/+9 不超 +9 控制）→ S1c-wash 干净图四臂 **PASS**（corrected +9 vs random −40，方向承载功能）→ S1c-geometry **共向不解耦**（cos 0.9498）→ S1c-global-oracle 跨窗口裁决 **FAIL**（U_global与随机13D rescue_gap=0，net+4<unconstrained+11）→ S1c-lowo-coverage LOWO建基审计 **窗口差异主导**（r13→64 LOWO ρ仅0.32→0.41，8窗口子空间pairwise对齐0.256近正交）——**13D 功能子空间主线在跨窗口设定下停线，S1c 系列闭环，且FAIL根因获得梯度几何层面的机制解释**→ S1d-selfsup-basis 无答案自监督建基 stage 1：事实行尾部遮罩补全CE行SVD出每窗口13D基，v1模板过滤gate擦线（ratio 1.5003、有效秩3.9–9.6<13）触发用户扩池裁决，v2移除模板过滤+latest-owner去重+40有效行/窗口（双卡分片）后 **gate PASS**（mean gRho 0.5116、ratio 1.6226、8/8窗口超LOWO、aux回填0），有效秩均值7.41系dropped块梯度天然聚集——**stage 2（D0+U_window·α 无答案优化）资格成立**。

## 交叉引用约定

`methods/prlmc/consolidate.py`、`COMPACTOR_PLAN.md` 等文件里的 `EXPERIMENTS.md E4`、
「缺陷 14–16」一类引用按编号指向上表对应文件；缺陷编号见 [experiments/fixes.md](experiments/fixes.md)。
