# SIC-O1：评分与覆盖选择优化

## 状态与目标

G3 暂停。本轮只优化 SIC 选择，不运行合并/回滚门，也不改变 reader 权重。
目标是先取得独立确认题上的初步改善，不宣称通用可插拔已经成立。

状态：开发对照完成，82确认题仍未评分；进入SIC-O2开发优化。最终计划见 `runs/sic_o1_plan.json`。

## 评分前冻结协议

- 原四上下文 rows0–3，原窗口 `start=30`，19块压成9块，原样保留被选块及时间顺序。
- 数据集 rows4–7 与 rows0–3 的事实内容哈希一一相同；八行不等于八个独立上下文。
- 旧 G2c 的200道题本轮明确转为开发题，允许用于方法选择，不再作为本轮独立测试。
- 全新确认题82道：rows0/1/2各25题，row3有7题。确认题不与历史272道训练/测试题或本轮训练题共享事实key/问题字符串。
- 训练探针97道：38/23/23/13。保留历史72道 discovery，增加25道；先分配确认题，再补训练题。
- 全文按 `(relation, subject)` 取最终事实，答案支撑需在原窗口内；答案1–5词。关系分桶轮转随机抽样，不按开头/结尾、支撑块位置或基线丢块筛题。
- 所有训练、开发、确认题全局key及问题字符串独立，训练和确认的数量只根据未评分数据可用性确定。
- 教师强制作答token完全正确评分，非B2实际生成/官方EM。完整池正确且无记忆错误定义eligible。

准备阶段曾检查长上下文尾部窗口、八上下文拆分方案，但均未完成计划或执行效果评分：尾部位置规则按用户要求撤销，八行拆分被哈希去重否定。最终方案只使用原窗口，不根据窗口边界估计信息价值。

## 对照

原生 `select` 是比较基线。五个SIC候选共享每题的同一次梯度：

1. `raw`：梯度平方按块求和，再跨探针累加，原 top-9。
2. `normalized`：每题块梯度能量L1归一化，再跨题累加top-9，消除题与题之间的总梯度幅值权重。
3. `coverage`：对逐题归一化能量贪心最大化 `sum_p sqrt(sum_{j in kept} score[p,j])`，让同一道题的重复覆盖边际收益递减。
4. `taylor`：每题逐块计算 `abs(sum(gradient * memory))`，L1归一化后top-9；这是删除方向一阶幅值代理，不是真实删除损伤保证。
5. `taylor_coverage`：Taylor逐题分数的覆盖贪心。

这些是工程修正对照，并非新颖性主张。Taylor梯度乘激活的剪枝谱系已有论文及公开实现：
[Molchanov et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.pdf)、[NVlabs/Taylor_pruning](https://github.com/NVlabs/Taylor_pruning)。
它们并非本实验可直接替换的MemoryLLM潜在池压缩基线，本轮不依赖下载这些代码。

## 决策与限制

开发选择规则：先最少eligible丢题，再最少全部题错误，再最低四上下文平均CE，最后名字确定性打破平局。选择后保存带计划SHA256的choice文件；确认阶段只评分select、raw、冻结候选。

初步改善标准：确认题eligible丢题至少减少10%，全部题正确数净增。强门槛另报：eligible丢题至少减少30%，双侧精确McNemar p<0.05。初步改善不等于强门槛、不等于跨上下文泛化。

82题仍聚集在四个历史上下文，按题McNemar只描述题级配对差异；必须同时报告每个上下文的rescued/harmed与方向。

reader写入/评分使用eval；反传使用train以启用非reentrant checkpoint，禁用dropout。每种保留集合评分两次并核对逐题相同。设置Python/Torch/CUDA种子及确定性算法警告模式；同池评分重复不等于跨进程复现。

反传仍需要已知QA答案，仍需每个探针一次reader反传。新增选择函数并未解决生产环境通用探针来源与成本问题。

## 实现与运行

- `methods/prlmc/sic.py`：新增逐题归一化、覆盖贪心和分块Taylor代理，旧函数行为保持。
- `tools/test_sic_selection.py`：23项纯张量/统计单测，归一化、覆盖、确定性、边界、符号抵消、旧函数回归、精确配对与整数百分比门槛均通过。原20项在远端环境也通过。
- `tools/sic_optimize.py`：冻结计划、训练梯度矩阵复用、开发选型、确认评分与配对统计。能量归约逐块进行以避免整窗fp32副本；确认阶段还要求与冻结开发报告逐行核对select/raw/选定候选的保留集合。

远端 `/share/latent/PRLM-C`，GPU9，Python `/share/latent/mllm_env/bin/python`：

```bash
python tools/sic_optimize.py --stage prepare --plan runs/sic_o1_plan.json --out runs/sic_o1_dev.json --choice runs/sic_o1_choice.json
python -u tools/sic_optimize.py --stage develop --plan runs/sic_o1_plan.json --out runs/sic_o1_dev.json --choice runs/sic_o1_choice.json
python -u tools/sic_optimize.py --stage confirm --plan runs/sic_o1_plan.json --out runs/sic_o1_confirm.json --choice runs/sic_o1_choice.json --development-report runs/sic_o1_dev.json
```

## 结果

完整报告 `runs/sic_o1_dev.json`，选型 `runs/sic_o1_choice.json`，远端进程退出码0。

| 方法 | 200题正确 | 128 eligible丢题 | rescued/harmed vs select | 丢题减少 | eligible精确p |
|---|---:|---:|---:|---:|---:|
| select | 107 | 25 | 0/0 | 0% | 1.000 |
| raw（扩展探针） | 108 | 25 | 6/6 | 0% | 1.000 |
| normalized | 111 | 24 | 5/4 | 4% | 1.000 |
| coverage | 112 | 24 | 6/5 | 4% | 1.000 |
| taylor | 104 | 29 | 2/6 | -16% | 0.289 |
| taylor_coverage | 104 | 29 | 2/6 | -16% | 0.289 |

按冻结规则选出coverage。全部题对select为13 rescued/8 harmed，p=0.383；记忆依赖题只有净增1，不足以认为方法已经取得可靠改善。Taylor幅值代理本轮更差，不继续围绕该臂单独调参寻找有利子集。

每窗全部题correct（select/coverage）：23/24、26/26、29/31、29/31。eligible丢题分别为9/9、9/9、4/4、3/2，前三窗净收益为零，唯一净少丢1题来自row3；不能把全部题+5解释成四窗稳定保护记忆。

第一窗38道训练探针：最大单题能量权重20.06%，有效等权题数8.73；这是需要修正权重的实测现象，但normalized只少丢1道，并不证明幅值不平衡是唯一瓶颈。

决策：82确认题继续封闭；先在开发协议内运行SIC-O2训练资格过滤与真实压缩CE校准。O1旧200题已用于选型，后续不再把它们称作独立held-out证据。
