# S1d 本轮最终结果：保护、自由度与真实生成

整理日期：2026-10-13。所有训练按最终 step 60 比较，不选择中途最高分。
原始 JSON、逐题预测、日志、smoke 与校验证书统一归档在
[reports/s1d_round_20261013](../reports/s1d_round_20261013/)。
原 C 对照见 [实验1归档](../reports/s1d_exp1_fact_v3/)。

## 方法实验：固定协议

原 C 使用窗口写入时的受损事实梯度构造局部基，仅训练 alpha。
冻结 reader、basis 和其余 carrier 参数；沿用 40 行尾部 4-token CE
求和梯度、AdamW lr=0.003 / weight_decay=0.01、seed=0、L2 残差预算1、
60步、每10步测量、QA上限256、TF-only。保护事实来自写入时保留事实，
不使用评测 QA 答案或正确性选择保护对象。

- RiskProtect：原 C13 basis 不变，D0 CE 有限且在 [0,2] 的 kept
  latest-owner 事实按 CE 降序选择最多20条；固定 lambda=1 的平均
  ReLU(current answer CE − D0 answer CE)，无权重扫描。
- C20/C24/C28/C32：同一受损梯度矩阵的前 k 个奇异向量，只改变自由度。
- P13/P32：先在完整 down.weight 空间投影掉最多40条保留事实的梯度方向，
  再构造13/32维基；保护候选按稳定 key 选择，D0 CE∈[0,2]，不叠加 hinge。
  原梯度和投影后有效秩均为40；投影后保留原受损梯度能量约45.59%。
  一阶正交保护不保证有限 AdamW 更新后的逐题预测不变。

R 只统计预定义 rescue_d0_wrong 中的新恢复；H 统计全部 D0-correct
问题的损失，net=R−H。因此 net 不一定等于全题正确数之差。
弱窗口推进门槛 net≥5 且 H≤4；强窗口最新门槛 net≥7。

## 方法实验最终结果

弱窗口 `(3,17510)`：

| 方法 | R | H | net | 推进门槛 |
| --- | ---: | ---: | ---: | --- |
| 原 C13 | 6 | 5 | +1 | 未通过 |
| M1：C13+RiskProtect | 6 | 5 | +1 | 未通过 |
| C20 | 5 | 3 | +2 | 未通过 |
| C24 | 5 | 3 | +2 | 未通过 |
| C28 | 5 | 3 | +2 | 未通过 |
| C32 | 5 | 3 | +2 | 未通过 |
| P13 | 6 | 3 | +3 | 未通过 |
| P32 | 5 | 4 | +1 | 未通过 |

强窗口 `(3,16370)`：原 C 与 M2（C13+RiskProtect）均为
R=10、H=1、net=+9。M2 保住现有优势，但没有改善；M1 未通过，因此
RiskProtect 整体不满足下一阶段推进条件。C20/C24/C28/C32/P13/P32
只跑弱窗口，没有新增强窗口结果。

三组维度对照与三组投影对照均已完成并有 valid=true 的最终证书。
C20 首次被 GPU 占用预检拦截，随后自动等待并成功完成；失败记录保留，
正式结果使用 `s1d_C_dimensions_v1_C20retry.C20.json`。
M1 由监控器接管，未直接取得原训练进程退出码；完整报告和冻结 basis
证书作为其完成依据，不声称其退出码为0。

## 现有 C 的完整生成结果

A1/A2 加载原 C13 已发布的最终 step60 checkpoint，不重新训练。
固定 B2 few-shot、换行 stop、greedy、max_new_tokens=32、attention_scale=1。
压缩组三臂使用相同2816 active slots；Full保留完整记忆。
Full是未压缩记忆参考，不是严格 oracle 或数学性能上限。

| 方法 | 强 `(3,16370)`，265题 | 弱 `(3,17510)`，276题 |
| --- | ---: | ---: |
| Full | 171，64.53% | 174，63.04% |
| Selection | 113，42.64% | 121，43.84% |
| D0 | 98，36.98% | 100，36.23% |
| C | 122，46.04% | 124，44.93% |
| C−D0 | +24，+9.06个百分点 | +24，+8.70个百分点 |
| C−Selection | +9，+3.40个百分点 | +3，+1.09个百分点 |

| 生成配对指标 | 强窗口 | 弱窗口 |
| --- | ---: | ---: |
| 全题 D0错→C对 | 28 | 27 |
| 全题 D0对→C错 | 4 | 3 |
| 全题净增加正确数 | +24 | +24 |
| Full对/Selection错/D0错中的恢复 R | 9 | 6 |
| R−全部D0损伤H | +5 | +3 |
| C相对Selection新增/损伤 | 30/21 | 18/15 |
| TF恢复题数 | 10 | 6 |
| 这些TF恢复题在C生成中答对 | 7 | 4 |
| 这些TF恢复题在生成中相对D0新恢复 | 6 | 1 |

两组全部预测均经原官方评分源码独立重算，证书 valid=true。
使用官方计分规则，但问题来自 S1d 开发窗口事实模板，**不是原生官方
问题集的最终 benchmark 成绩**。整体生成提升不能全部归因于 TF
诊断的恢复题，也不能用旧 C 生成结果证明新保护变体的生成效果。

## 本轮裁决与限制

RiskProtect 未改善原 C 的最终 R/H；扩维20–32D没有持续收益，不能据此
认为13D是当前主要瓶颈。P13保留6道恢复并将损伤5→3，是本轮最高净恢复，
但仍只有+3，未到+5。没有候选取得扩大方法验证的资格，不继续扫描保护
权重、prompt或挑选中途checkpoint。原 C 的开发窗口生成有收益，但相对
Selection的提升较小，且仍远低于Full。

当前证据来自两个已见开发窗口和单seed；没有未见自然context或独立测试
的泛化结论。24个受控episode只完成准备，8开发/16测试划分不变，16测试
保持封存，未运行模型评分。历史 S1-local hard-nullspace NO-GO 与
S1c跨窗口FAIL结论仍有效，不因本轮局部结果被撤销。

## 代码与产物索引

- 风险保护：`s1d_risk_protection.py`、`s1d_c_risk_train.py`、
  `verify_s1d_risk_protection.py`；M1原入口为` s1d_experiment1.py`。
- 冻结生成：`s1d_c_generate.py`、`s1d_c_runtime.py`、
  `s1d_c_generation.py`、`audit_s1d_generation.py`。
- 投影对照：`s1d_projection.py`、`s1d_projected_train.py`、
  `run_s1d_projection_weak.py`、`verify_s1d_projection.py`。
- 维度对照：独立` s1d_dimension_basis.py`、`s1d_dimension_train.py`、
  `run_s1d_dimensions.py`、`verify_s1d_dimensions.py`，没有改动活跃投影脚本。
- 控制集准备：`prepare_s1d_controlled.py`、`s1d_controlled_contract.py`，
  此轮未调用真实test开封。

原服务器路径为`/share/latent/PRLM-C/runs/`，Python为
`/share/latent/mllm_env/bin/python`，reader cache为`/dev/shm/s1d_reader_fp16`。
JSON保留原运行时绝对路径与哈希，不为归档改写原始报告。
总索引与CSV见归档`summary.json`、`training_results.csv`、`generation_results.csv`；
`SHA256SUMS.json`记录归档文件校验和。
