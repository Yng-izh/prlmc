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
