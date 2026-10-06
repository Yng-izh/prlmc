# S1-soft-local：固定13维UL的单边有限CE保护

## 预注册状态

本协议在训练前冻结，承接S1-local的UL本地正证据，但不更改历史hard-nullspace五臂的NO-GO。用户选择先固定原13维与原13题监督、隔离保护机制；真实selection-loss监督与重新建基留待独立协议。本轮不启动泛化或论文预算实验，不追加lambda变体。

同一S0窗口row=3/start=17890，19→9 blocks，E6d interleave D0，原窗口写入顺序与seed=0。写入后关闭384个reader dropout子模块，父级train保留供梯度检查点使用。296道相同探针，teacher-forced answer-token exact match，不是官方生成EM。

正式basis SHA256：`3af25e44b94464432a0221e6dabe8685ef2bd91c522addcbe279387425bb97b2`。复用其中13维UL，不重算、不重排、不投影。D0与历史驱动hash须匹配历史JSON；expanded select/D0逐题正确性、原60题full正确性、question/gold来源及重复D0评分须复现历史，错误则INVALID-BASELINE。

## Cohort与训练监督

完整序列化full/select/D0的逐题正确性与答案token平均CE：

- R（真正压缩损失）：full正确且select错误；再拆D0已正确与D0错误。
- P（已有能力保护）：full、select、D0均正确，预期31题。
- T（checkpoint退化repair）：full、select正确且D0错误，预期44题。
- K=P∪T：full且select正确，预期75题。

R/P/T互斥。历史13题lost是在D0丢失的原目标，保留为唯一监督集合L_original，逐题报告与R/P/T交集。T不另加训练监督。原13题rescue≥7继续报告，但不视作selection-loss恢复率。

## 三臂与冻结目标

| 训练臂 | 训练空间 | 原目标 | 单边罚项 |
|---|---|---|---|
| UL | 原13维UL | 原13题CE求和 | 无，lambda=0 |
| UL-soft-P（候选） | 同一UL | 同上 | P的有限CE恶化，lambda=1 |
| UL-soft-K（cohort消融） | 同一UL | 同上 | K的有限CE恶化，lambda=1 |

全部`D=D0+reshape(UL alpha)`，alpha零初始化。AdamW lr=0.003、weight_decay=0.01、60steps、seed=0，每5步评分，参数位移预算1.0；仅down位移可训练，其他参数冻结。控制臂保留原逐题CE累积顺序，lambda=0不执行保护前向。

`loss = sum_l CE_l(D) + 13 * lambda * mean_k relu(CE_k(D)-CE_k(D0))^2`

D0 CE在相同确定性reader/压缩mask上预先计算并detach。改善不罚，恶化才罚。逐题有限CE反向累积，不保留75个reader前向图；复用原pool注入。它不是每步gradient cone投影，不保证token-exact不变。固定lambda=1，不扫描、不按结果选参数。乘13使保护项和原监督的题均值尺度对应，同时保持原CE求和路径。

native select与D0为无训练参照，历史hard臂只作历史对照。UL-soft-K不得看结果后替换候选。

## 验证与判门

先2step、每步测量的SMOKE-ONLY，使用完整296题和三臂，不计入方法结论。单测覆盖cohort、单边梯度、reference detach、13DOF与预算、checkpoint/dropout梯度、lambda0路径及指标；独立核验器重建全部集合、CE罚项、net、列表/计数、history与终点；CPU真实权重审计所有冻结参数与保存基底。

正式UL第60步须逐题复现历史UL，并满足original rescue7/13、K harm6/75、net+8。不一致标INVALID-CONTROL，停止后续臂，不推断方法有效性。

候选UL-soft-P在第60步同时满足：original rescue≥7；P新增harm=0；R恢复≥UL；T repair≥UL；K harm≤UL；net≥+8；且R恢复、T repair或K harm至少一项严格改善，才LOCAL-PASS。其余为NO-IMPROVEMENT，不选中间峰值。运行错误单独标failed，不写作方法NO-GO。

净收益=`R恢复-K harm`，不等于全题正确数差。分别记录original rescue、R恢复及D0既有R能力丢失、T repair、P新增harm、K harm、全部D0新增harm、296题正确数。完整full正确性让net可独立逐题重建，修复历史核验缺口。

## 机制解释与文献边界

历史lost与K重叠7题，梯度同时出现在JL/JK中，已足以解释13→6的精确秩下降。保存基的6个非零投影奇异值0.540–0.862，另7个约1e-14；因此秩下降不独立证明公共功能轴被删除，也不称剩余6维为边缘tail。`span(P_K_perp UL)`一般不等于`span(UL)∩K_perp`，历史safe方向事实上离开原UL span。

[GEM](https://arxiv.org/abs/1706.08840)及其[官方代码](https://github.com/facebookresearch/GradientEpisodicMemory)、[A-GEM官方实现](https://github.com/facebookresearch/agem)已覆盖单边允许有益迁移的思想。本轮只验证固定reader residual、固定13维空间的逐题有限CE惩罚，不声称单边约束本身新颖，不增加外部baseline依赖。

## 运行与结果

尚未完成正式训练。预注册实现与解释修订已在结果前以`ae254bf`推送；本地和原远端均20项新增测试通过。原历史五臂JSON继续通过原核验器、SHA256未变。

完整基线快照独立核验通过：R=36（D0正确4、错误32），P=31，T=44，原13题与R/T交集为6/7。此处是基线分解，不是三臂最终方法结果。

```bash
/share/latent/mllm_env/bin/python -B -m unittest discover -s tools -p 'test_*s1_soft*.py' -v
/share/latent/mllm_env/bin/python -u tools/s1_soft_local.py --carrier-ckpt runs/e6d_interleave_ckpt.pt --basis runs/s1_local_protected.basis.pt --historical-result runs/s1_local_protected.json --smoke --cuda-device 9 --out runs/s1_soft_smoke.json
/share/latent/mllm_env/bin/python -u tools/s1_soft_local.py --carrier-ckpt runs/e6d_interleave_ckpt.pt --basis runs/s1_local_protected.basis.pt --historical-result runs/s1_local_protected.json --cuda-device 9 --out runs/s1_soft_local.json
python3 tools/verify_s1_soft_result.py runs/s1_soft_local.json --historical runs/s1_local_protected.json --require-complete
/share/latent/mllm_env/bin/python tools/audit_s1_soft_checkpoints.py runs/s1_soft_local.json runs/e6d_interleave_ckpt.pt --source-basis runs/s1_local_protected.basis.pt
```

三臂短验证与正式训练结果在完成独立核验后追加，以上命令列为复现入口，不表示尚未执行的步骤已经完成。本预注册段保留不改。

### 已完成的短验证

三臂完整296题、2steps、每步测量，进程exit0，verdict=SMOKE-ONLY。初始化均逐题正确性和CE等于D0；全部基线检查通过。三臂末步original rescue0/13、R恢复4/36、T repair1/44、P新增harm0/31、K harm43/75、net−39；短步数仅验证链路，不用于方法判门。

候选soft-P第2步罚项约8e-6；soft-K约5.95e-4。真实alpha/位移已不同于UL，单边罚项梯度实际参与更新。三臂13DOF，全部源compactor参数冻结，预算通过；实际delta范数UL0.01964989、soft-P0.01965276、soft-K0.01972804，正式UL span泄漏约4.9e-10。

原远端Torch2.5.1与本地Torch2.8独立指标/权重审计均通过，四个二进制工件已收回、hash逐项匹配远端；权重不纳入普通Git提交。结果见[smoke JSON](../runs/s1_soft_smoke.json)与[日志](../runs/s1_soft_smoke.log)，[远端指标audit](../runs/s1_soft_smoke_metric_audit.json)、[权重audit](../runs/s1_soft_smoke_checkpoint_audit.json)、[本地指标audit](../runs/s1_soft_smoke_metric_audit_local.json)、[本地权重audit](../runs/s1_soft_smoke_checkpoint_audit_local.json)、[工件清单](../runs/s1_soft_smoke_artifact_manifest.json)。smoke audit的正式control门不适用，记null；gate仅为诊断重算，不影响SMOKE-ONLY状态。

### 正式运行状态

短验证双audit通过后，已在原GPU9启动固定60步的正式运行。训练驱动与预注册版本相同。先完成UL并逐题核对历史第60步结果，控制通过才继续两个soft臂；不选中间峰值，不改lambda或门槛。正式终点与核验结果尚待完成。

运行中第5/10步快照独立指标核验通过，baseline与P/T/R集合匹配；与历史UL的rescue/harm计数相同，但逐题分别2/1题预测不同，alpha范数相差约3.1e-6/1.0e-5，net分别−34/−22而历史−35/−23。见[控制轨迹诊断](../runs/s1_soft_control_trajectory_audit.json)。这不是已通过复现：继续按第60步逐题一致与7/6/+8的原门核对，不放宽门槛。当前尚未确认偏差原因，不将其直接归因为数值非确定性。

### 正式运行中止：训练图无效

第10步时发现并独立复现了训练图缺陷，正式运行随即停止，未完成第60步，也未运行soft-P/soft-K。中间文件保留为[无效运行记录](../runs/s1_soft_local_invalid_training_graph.json)，原始状态快照为[运行JSON](../runs/s1_soft_local_snapshot.json)。该记录不参与任何方法判门。

缺陷链为：`layout.py`的`_memory`返回`model.memory.data`；`s1_soft_local.py`先从其切片构建candidate；`CarrierResidualCompactor`在`use_reentrant=False` checkpoint中保存该storage-backed view；随后`_pool_in_place`把candidate写回重叠的memory区间；backward重算读取了已改变的source。CPU直接调用生产`CarrierResidualCompactor`、`_pool_in_place`与`make_displacement`的审计显示，forward和loss逐位相同，但`.data`路径梯度与未注入路径不同；clone source路径梯度逐位一致，memory退出逐位恢复。fp32/13D alpha梯度差范数为0.2081410736，fp16/13D为0.2125255316；不是CUDA数值波动。

证据见[生产CPU复现JSON](../runs/s1_checkpoint_recompute_production_cpu.json)与[独立审计脚本](../tools/audit_s1_checkpoint_recompute.py)。对应代码位置是[源切片](../methods/prlmc/layout.py#L309)、[compactor view](../methods/prlmc/consolidate.py#L583)、[checkpoint](../methods/prlmc/consolidate.py#L639)、[原地注入](../methods/prlmc/probes.py#L821)、[soft训练反向](../tools/s1_soft_local.py#L293)。因此UL第5/10步虽可完成指标计算，逐题与历史轨迹已有差异，不能判为控制复现；不把其中间数字解释成方法效果。

修复方向已确定为：在candidate进入checkpoint前使用独立source副本，使保存的checkpoint输入不再与可注入memory共享storage；修复后需要重新冻结driver hash、重新运行三臂，并重新审视“UL必须逐题复现历史S1-local”的控制门。当前不提交未经用户确认的修复运行，也不启动泛化或论文预算。

## 修复后的协议修订（结果前冻结）

用户已授权先修复再继续。修订内容如下，其余预注册条款不变：

- **驱动修复**：`tools/s1_soft_local.py`的`candidate()`改为对`manager._memory`窗口切片`.clone()`，切断checkpoint保存输入与注入区间的storage别名。`s1_local_protected.py`保持逐字节不变，历史driver hash链继续成立。
- **梯度回归测试**：`tools/test_s1_soft_local.py`新增`test_production_compactor_cloned_source_gradient_equality`，用生产`CarrierResidualCompactor`+`_pool_in_place`在CPU上对比clone/别名/无注入三路径：clone路径参数梯度与无注入参照逐位相等，别名路径必须不相等（证明测试有判别力）。
- **控制门修订**：修复后UL不再要求逐题复现历史UL终点——那条轨迹本身就是别名梯度缺陷的产物，其逐题终点是bug产物而非控制目标。修订后的控制为运行内UL第60步的计数不变量：original rescue 7、K harm 6、net+8；历史逐题匹配降级为诊断字段（`control_checks.per_question_match`与核验器`control_detail.historical_per_question_match`），不参与判门。不满足计数不变量仍标INVALID-CONTROL。
- **net 边界的单边化（运行后裁决）**：修复梯度使UL控制臂终步落在net=+9（比历史buggy终点多救回1题，逐题差恰为"Who is Ethel Skakel Kennedy married to?"；original 7与K harm 6逐位成立，rescued清单其余逐题一致）。等式net==8会把更强的控制判为失败，与控制的目的（恢复original、伤害受限）相悖。经用户裁决，net不变量单边化为net≥8：超出幅度与逐题差异记录在`control_detail`，等式语义废弃。该裁决在看到soft臂终点之后作出，但UL终态在soft臂运行前已冻结，两臂终点不因该裁决改变。
- **审计链**：驱动新增`--gradient-audit`必选参数，指向`runs/s1_checkpoint_recompute_production_cpu.json`；其verdict全部为真才允许启动，hash记入protocol（`gradient_repair_audit_sha256`）。核验器同步要求该字段为64位十六进制。

## 修复后正式运行结果（60步，GPU9）

修复提交`00c4880`（driver hash `17db9cfe…`）与远端逐字节同步后，先跑smoke 2/1（`runs/s1_soft_smoke2.json`，verdict=SMOKE-ONLY，双端指标/权重audit通过），随后启动正式三臂60步。命令行在预注册基础上新增`--gradient-audit runs/s1_checkpoint_recompute_production_cpu.json`（sha256 `63059678…`，verdict全真）。

三臂终步（step 60，296题全量）：

| 臂 | original | R | T | K_harm | P_harm | net | penalty |
|---|---|---|---|---|---|---|---|
| ul（控制） | 7 | 15 | 38 | 6 | 0 | +9 | 0 |
| soft_p | 7 | 14 | 38 | 6 | 0 | +8 | 0.001407 |
| soft_k | 7 | 15 | 38 | 6 | 0 | +9 | 0.007022 |

UL轨迹：net从step5的−34单调爬升，step40达+8，step45起+9并保持到终步；soft_p在step25-40停在+7、step45起+8；soft_k在step30-35停在+7、step40起+8、step45起+9。soft_k的penalty从0.067（step15）单调降到0.007（step60），即惩罚随训练被最小化，但未换来任何超过控制的恢复。

**verdict=NO-IMPROVEMENT**。gate七项中`rescue_not_below_control`与`strict_improvement`失败：soft_p的R=14低于控制15，soft_k与控制打平（R15/T38/K6逐项相同）。λ=1的单边CE保护罚项在两种形式（P臂保护rescue_d0_correct、K臂保护retained）下都未能产生任何严格优于无约束UL的指标。S1-soft-local的结论为：在该预算（13 DOF、60步、AdamW lr.003）下，单边软惩罚不优于无约束基线。

控制判定（修订语义）：original_7 ✓、harm_6 ✓、net≥8 ✓（+9，超出记录为诊断）。历史逐题匹配为False（1题差异，诊断字段）。

权重audit（远端+本地独立各一次）三臂全部通过：13 DOF、源compactor参数冻结逐位一致、span泄漏~1.1e-8、delta范数UL 0.4420/soft_p 0.4420/soft_k 0.4405。指标audit：`verify_s1_soft_result.py --require-complete --historical`通过，报告sha256 `e707b139…`。

工件：[结果JSON](../runs/s1_soft_local.json)、[训练日志](../runs/s1_soft_local.log)、四组.pt（basis/ul/soft_p/soft_k，hash见下）。smoke2工件：`runs/s1_soft_smoke2.*`（json sha256 `4b25c47b…`）。正式driver hash：`17db9cfe76bb841af30eb0999e9c5a2cc4c06f9ac337083eeb53fb0de84ff984`。核验器/测试同步修订（net单边化）后全部测试通过（`test_s1_soft_local.py` 11项、`test_verify_s1_soft_result.py` 10项）。

无效运行快照（别名缺陷期的step5/10）保留为`runs/s1_soft_local_invalid_training_graph.json`与`runs/s1_soft_local_snapshot.json`，仅作缺陷证据，不参与判门。
