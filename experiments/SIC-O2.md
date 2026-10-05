# SIC-O2：训练探针资格过滤与真实压缩损失校准

## 状态

开发完成；选出eligible_normalized，128 eligible丢25→23（8%），200题正确107→115。82道确认题的唯一冻结方案验证运行中，不用确认结果重选方法。G3继续暂停。
本轮不按事实位于上下文开头/结尾或支撑块位置判断重要性。

## 实现级假设

O1解决跨题梯度幅值不平衡和独立top-k缺少覆盖约束，但其梯度目标仍对所有训练题计算CE。最终评价却只把完整池可答、无记忆不可答的题视为记忆依赖题。原来可凭先验答对的题、完整池也答错的题，可能消耗评分权重。

O1首窗诊断：38道训练题的梯度能量权重最大单题占20.06%，有效等权题数约8.73；这是幅值不平衡的实测现象，不证明失败原因唯一，也不证明过滤必然有效。

另一个目标错位是：梯度在完整19块池上估计，真实操作为19→9离散删除。O2在训练探针上实际构造压缩池，测CE来校准梯度生成的有限候选，而不声称一阶代理等同真实删除代价。

## 固定输入与隔离

复用 `runs/sic_o1_plan.json`：97 training discovery，200 development（原G2c旧held-out），82 fresh confirmation。四个原窗口start30不变。O1开发结果是可复用的开发数据，不是O2的新测试。

- O2代码只对discovery取full/no-memory正确性，并使用缓存energy/taylor。
- 训练合格探针定义：full正确且no-memory错误；至少4道才生成过滤评分臂。不足时记录状态，省略该臂，不将未运行结果填成零。
- 三个新臂：合格探针归一化top-k、合格探针覆盖选择、合格探针Taylor覆盖选择。
- `joint`：对原生选择、五个原SIC候选及可用的新候选，在同一组训练合格探针上测真实压缩CE；选择平均CE最低集合，平局优先原生选择。若合格探针不足，明确记为原生回退。
- 校准不使用development/confirmation问题或答案；development只用于在方法间选择一次。训练CE最低不构成未来未知题无害保证。
- 冻结确认题不参与任何方法构造或调参。最终选择后保存choice及每窗保留集合，再执行一次确认评分。

## 成本与可重复性

本轮复用O1逐题梯度矩阵，不新增reader反传。训练资格过滤需每题full/no-memory两次无梯度forward；真实候选校准按不同保留集合去重，需要候选数×合格训练题数的forward。必须报告实际数量，不凭总训练题数误报每窗口成本。

O2开发重建原窗口，核对原生选择与O1缓存一致。所有开发题的full/no-memory/select还在新进程逐题重测，必须与源报告完全相同，否则中止以免混合不同池上的新旧对照。旧候选的开发结果可复用O1记录，新候选保留集合若与旧候选相同则复用对应逐题结果，否则评分两次核对correct。上述检查仍不等同完整池逐字复现，报告继续注明限制。训练候选CE只评分一次，新开发候选评分两次。

## 决策

方法选择仍按开发eligible丢题、全部题错误、平均CE、名字排序；只允许四窗口均可用的臂进入整体选型。

确认初步改善门槛沿用O1：相对原生选择eligible丢题至少少10%，全部题正确数净增；强门槛另报30%与精确双侧McNemar p<0.05。门槛用整数百分比比较，避免浮点边界误判。

若开发没有改善，不用82确认题继续挑方案，继续在开发协议内诊断或调整；不靠换开头/尾部、换有利子集制造改善。

## 实现与验证

`tools/sic_refine.py`仅对训练探针进行筛选/校准，开发完整报告按哈希验证后复用。4组CPU回归测试通过，覆盖输入完整性、缓存参考逐题变化、CE异常、单/双次评分及异常清理；加上SIC评分/统计/确认入口24项单测，合计28组。审计后修正了开发报告中夸大的`cross_process_kept_equal`，明确仅native重算通过，其他源arm使用缓存；修改仅涉及元数据，不涉及分数、块集或胜者。确认入口另增加来源哈希绑定、完整开发统计与排序复核，已在当前真实输入上独立通过。

```bash
python -u tools/sic_refine.py --plan runs/sic_o1_plan.json --source runs/sic_o1_dev.json --out runs/sic_o2_dev.json --choice runs/sic_o2_choice.json
```

选型后独立确认命令（未运行）：

```bash
python -u tools/sic_optimize.py --stage confirm --plan runs/sic_o1_plan.json --out runs/sic_o2_confirm.json --choice runs/sic_o2_choice.json --development-report runs/sic_o2_dev.json --frozen-kept
```

确认模式只使用开发阶段已经冻结的每窗保留集合，不按确认题重新挑块；会重新核对原生/raw集合并复测评分，但没有重新执行完整训练资格/CE校准算法，因此不声称完整算法跨进程复现。

## 结果

开发报告 `runs/sic_o2_dev.json`，冻结选择 `runs/sic_o2_choice.json`；远端进程退出码0。

| 方法 | 200题正确 | 128 eligible丢题 | rescued/harmed vs select | eligible减少 | eligible精确p |
|---|---:|---:|---:|---:|---:|
| select | 107 | 25 | 0/0 | 0% | 1.000 |
| O1 coverage | 112 | 24 | 6/5 | 4% | 1.000 |
| eligible_normalized | 115 | 23 | 10/8 | 8% | 0.815 |
| eligible_coverage | 115 | 24 | 10/9 | 4% | 1.000 |
| eligible_taylor_coverage | 107 | 27 | 6/8 | -8% | 0.791 |
| joint（训练真实CE选型） | 110 | 26 | 8/9 | -4% | 1.000 |

按冻结规则选择eligible_normalized。全部题正确净增8，b=17/c=9，精确p=0.169；仍只是已知开发题上的非显著改善，不当成独立验证证据。

训练合格探针为19/17/20/12，合计68/97。eligible_normalized逐窗eligible丢题9→10、9→9、4→2、3→2，净收益为-1/0/+2/+1，开发收益并非四窗一致。joint训练选型为eligible_normalized、eligible_coverage、raw、raw；其中row2选raw的开发正确数28，而资格过滤臂为33，说明训练CE最低在本轮没有确保开发题最好。该观察不能强归因为单一过拟合机制，但足以否定“真实训练CE校准必然提高未见题”的强说法。

确认结果 `runs/sic_o2_confirm.json` 已完成，使用开发阶段冻结的eligible_normalized逐窗块集；没有按确认题替换方法。

| 方法 | 82题正确 | 64 eligible丢题 | rescued/harmed vs select | 丢题减少 | 精确p |
|---|---:|---:|---:|---:|---:|
| select | 55 | 10 | 0/0 | 0% | 1.000 |
| 冻结 eligible_normalized | 56 | 11 | 3/4 | -10% | 1.000 |
| raw（预设对照） | 58 | 8 | 3/1 | 20% | 0.625 |

冻结方法在确认题上比select多答对1题，但记忆依赖题多丢1题，初步改善门槛未通过；raw的确认优势是预设对照，不能回头替换已冻结方法。确认题只有四个共享上下文的全新问题，不能称作新上下文泛化。确认阶段运行时的CUDA、SDPA和CuBLAS仍产生非确定性警告；报告标记跨进程完整复现未测试。

## Native support oracle

新增 `tools/sic_support_oracle.py`，在开发题上枚举全部 `C(19,9)=92378` 个原生组合，使用已知最终事实所在 block 最大化 eligible 问题的支撑 block 保留数。它不运行 reader CE，而是一个答案支撑上界，只用于诊断原生选择空间，确认题未读取。

结果（eligible lost 下界 / 当前 select lost / 冻结O2 lost）：

| row | oracle下界 | select | O2 eligible_normalized |
|---|---:|---:|---:|
| 0 | 5 | 11 | 10 |
| 1 | 10 | 14 | 14 |
| 2 | 10 | 20 | 20 |
| 3 | 5 | 12 | 12 |

oracle与select每窗重合5–6个block，与O2每窗重合5–6个block。这个差距证明原生19选9组合仍有空间，但不能证明换成oracle集合后reader CE一定达到该下界，因为reader存在冗余、读出失败和格式错误。

曾启动真实reader CE组合oracle的100组合标定，但单组合要对50道题完整运行8B reader，成本过高，进程在模型加载阶段终止，未产生数值结果；没有把它计入实验结果。有限逼近脚本 `tools/sic_swap.py` 已实现：从冻结O2集合出发，用discovery eligible题的梯度增益从90个one-for-one交换中预筛8个，再用真实discovery CE排序；support oracle只在结果后计算结构gap，`oracle_aware=false`。row0真实运行在base评分阶段因远端GPU并发显存占用触发OOM，未生成swap数值报告；脚本已增加`--cuda-device`，待空闲GPU后重跑。该失败是资源状态，不计入算法结果。

