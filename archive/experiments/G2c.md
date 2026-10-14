# G2c — 固定200道独立留出题复验

状态：**完成，NOT_PASSED**。归档时间：2026-10-04（本机日期）；旧实验文档日期原样保留。抽样种子 20261008，四个不同 context 各50题，共200题。

脚本：`tools/g2c_heldout200.py`；评分前清单：`runs/g2c_plan.json`；逐题结果：`runs/g2c_report.json`。

## 冻结设计

- 采样不读取 select/sic 的保留块、正确率或 eligible 标签，不按成绩补题。
- 每题对应一个全局不同的关系/主体键；四context之间也去重，discovery 与 held-out 的键全局不重合。
- 答案采用完整 context 的最后一条事实；其支撑行必须在待压缩窗口内。
- discovery 数目分别为24、19、16、13；held-out 固定每行50。
- 优先覆盖 current 和 singleton；版本事实稀缺时以不同 singleton 事实补足，不宣称等量平衡。
- reader 冻结，19块压缩至9块；显著性只使用 discovery。
- 写池和评分均使用 eval；反传临时开启 train 和梯度检查点，显式关闭 dropout。
- 每个压缩臂全题评分重复两遍，要求逐题完全一致。
- 使用现有 teacher-forced 答案token全匹配评分，不冒称B2生成式官方EM。

## 判定

同时报告200题总体正确率及 eligible 子集（full 对、no-memory 错）的 lost。
主要验证条件：eligible lost 相对下降至少30%，双侧精确 McNemar p<0.05。
额外报告 current/singleton 分层和逐context结果，不事后更换主指标。
题目共享四个context，题级McNemar为条件性证据，不等于200个独立context的泛化。

原始G2 FAIL、G2b的小样本结果保留；本次修复采样和运行态，因此结果不是旧实验的简单加题。

## 结果

200道全局不同问法，覆盖22种关系：54道 current、146道 singleton。
current 定义为全文内同关系/主体键重复出现，包含同答案重复，并不全是冲突题。
72道 discovery 的键与全部 held-out 的键完全不重合。

| 指标 | select | sic |
|---|---:|---:|
| 200题正确数 | 107 | 106 |
| 200题正确率 | 53.5% | 53.0% |
| 128道 eligible 的 lost | 25 | 29 |
| eligible lost率 | 19.53% | 22.66% |

未压缩窗口+anchor 正确128/200；零内容池正确0/200。
主指标 lost 相对变化为增加16%，不是下降30%；b=5（select错sic对）、c=9（select对sic错），双侧精确 McNemar p=0.423950。
200题总体 b=10/c=11，精确 p=1.0。

逐context eligible / select lost / sic lost：

- row0：31 / 9 / 13。
- row1：32 / 9 / 9。
- row2：33 / 4 / 5。
- row3：32 / 3 / 2。

| eligible分层 | 题数 | select lost | sic lost | 精确p |
|---|---:|---:|---:|---:|
| current | 21 | 7 | 11 | 0.125 |
| singleton | 107 | 18 | 18 | 1.0 |

## 核验与边界

- 每个压缩臂在同一池上完整评分两次，四窗口逐题布尔正确性完全相同。这不等于写池、梯度和选块的跨运行复现。
- 272道输入（discovery+held-out）最长172 token；超过256的数量为0，未发生答案尾部截断。
- 四context重新读取后的全文SHA256与冻结清单一致。
- 200是原始留出题数，128才是主要lost检验样本；不按eligible补题。
- 使用模板单跳事实题和teacher-forced答案匹配，未验证B2真实生成、跨数据集、多跳题或sicm合并。
- 题目共享四个context，题级检验是条件性诊断，不提供广泛泛化保证。

## 结论修正

本次未复现G1/G2b优势，且点估计反向；p不显著，亦不证明sic必然更差。
**撤回“独占信息保护机制已证实”的结论**：本次singleton净效果为零；之前12个救回题是singleton，仅是描述性证据，且旧报告实际有13个救回题（12 singleton、1 current），不是全部12题。
保留G1的历史协议数字，但它用同批题指导选块又评测，且以select丢弃块为条件；旧运行中同kept的select/oldest成绩不同，重复性亦有疑点。因此G1的PASS不升级为未见题泛化证据。

当前研究结论：功能梯度的存在不等于少量discovery探针能指导保留未见题所需信息。暂停据G1直接推进方法有效性的强主张；下一项若启动，应独立检验discovery覆盖率与held-out收益的关系，而不是用合并臂掩盖选择准则尚未验证的问题。

