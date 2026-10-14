# SQuAD author512：37题中间结果与压缩失败分析

这是正在运行的100题公开SQuAD知识保持实验的中间诊断，**不是最终成绩，也未通过最终全量校验**。观察时间为2026-10-13 20:25（UTC+8）。固定队列中的37个完整样本ID为0–35及39，涉及26个原始context；它们不是随机抽出的独立37个场景。后续题目继续运行，本文不据此挑选checkpoint、调整提示或更新模块。

协议见 [Public-retention-frozen-SharedE6d.md](Public-retention-frozen-SharedE6d.md)。固定公开Processed数据版本、既有E6d checkpoint、100题；每题先写相关段落，再写30份干扰文本；作者512-token截断预处理，每次写入后生成10 token。主评分为作者**大小写敏感答案包含率**，不是SQuAD官方normalized EM。Selection/Recent/SharedE6d共享20块有效预算；Full和NativeDefault使用完整50块物理容量，作为容量不同的参考。

## 中间观察

五个arm均完成的同一37题，末步写入31时：

| 方法 | 答对 | 答案包含率 |
| --- | ---: | ---: |
| Full | 21/37 | 56.76% |
| Selection | 2/37 | 5.41% |
| Recent | 2/37 | 5.41% |
| SharedE6d | 0/37 | 0% |
| NativeDefault | 7/37 | 18.92% |

SharedE6d相对Selection新增答对0题、损伤2题（sample6、25），净增−2。低样本量、相邻同context题目和共同干扰流限制推断，不能提前写成正式泛化结论。

同一37题的阶段变化：

| 方法 | 写入1（相关段落） | 写入20（压缩前） | 写入21（第一次压缩后再写入） | 写入30 | 写入31 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 20 | 20 | 20 | 20 | 21 |
| Selection | 20 | 20 | 2 | 1 | 2 |
| Recent | 20 | 20 | 2 | 1 | 2 |
| SharedE6d | 20 | 20 | 1 | 1 | 0 |
| NativeDefault | 21 | 11 | 6 | 8 | 7 |

注意写入21的测量在压缩及下一次写入后进行，不是单独执行压缩前后、完全无其他操作的因果消融。但Full的相同步骤仍保持20题，而三种受管理arm在该边界明显下降，支持压缩路径是主要定位方向。

## 已确认的实现与行为事实

**这37题的Selection与SharedE6d第一次压缩全部保留窗口相对块10–18，与Recent完全一致。** 包含相关段落的块0没有一次作为原生carrier保留下来。三组都执行两次19→9压缩。SharedE6d与Selection第一次压缩前没有残差干预，选中的carrier确实相同。

这解释了Selection与Recent均低分：此任务询问最早写入段落，而目前选块偏好保留后续干扰块。这里只确认**这批输入上**Selection退化为Recent，不能推断它在所有输入上等价于Recent。近期项与几何重要性究竟如何贡献，需要额外记录各项分数，现有记录不能单独归因于某个权重。

原始块不再被直接保留不等于所有相关知识立即完全消失：native写入可能传递信息，Shared残差也能从整个压缩窗口融合信息。因此这里确认的是**相关原生载体被淘汰**，其余信息残留量需进一步分析。

Full在压缩边界仍答对20题，表明至少这部分知识当前reader能够读取。不能把三种压缩arm的低分全部归因于reader无能力；也不能因为Full仅约57%就认定reader没有影响。例sample7输出“Fighting horsemen”而gold为“fighting horsemen”，按作者大小写敏感规则判错，必须保留这一协议限制，不能事后更换主评分。

## 结构与训练的解释

目前较合理的判断是：**选块、残差信息搬运能力和训练—评测分布偏移共同作用**，而不是只把问题解释为训练不足或整个MemoryLLM架构失败。

1. **载体选择与当前知识保持需求不匹配。** SharedE6d在Selection选出的native carrier上加残差，不能训练选块本身。此次要把早期相关段落的信息迁移到近期干扰记忆上；此前要求读取最新版本的受控冲突任务，对近期块的偏好可能有利，不能据此推断其保留早期知识的能力。
2. **当前残差更接近受限修正，尚未证明可以承担通用重新编码。** `CarrierResidualCompactor`由参考层块均值产生块级路由，各层按相同槽位位置加权融合source，然后通过 `up(GELU(down(context-carrier)))` 生成rank32残差。这个结构有机会保留被丢弃信息，但没有显式证明能重新组织自然文本事实并让冻结reader读出。
3. **实际修改幅度有限。** CPU读取固定checkpoint得到 `gamma=0.0063441284`、`route_scale=0.0551528037`。残差逐输出块按carrier RMS规范化，再乘gamma，因此其RMS修改幅度上限约为carrier的0.634%。gamma_max0.05不是此次实际系数。幅度小是可能瓶颈，**不能仅凭此值证明信息容量不足**，更不能推出只要放大gamma就会有效；大更新同样可能破坏reader可读性。
4. **训练偏移包含内容、状态和时间过程。** E6d训练主要来自同一个结构化事实context的8个局部窗口，多窗口interleave优化；此轮为自然段落、相关文本先写、连续干扰和重复压缩。旧训练窗口还使用anchor及不同初始化流程。本轮没有答案保护anchor，使用checkpoint初始状态。这里不仅换了question或文本模板，还换了记忆状态和生命周期。

上述第1项有直接选块日志支持；第2–4项是由代码结构/训练协议支持的机制假设，尚未有控制实验划分各自贡献。整个MemoryLLM底座的普遍架构失败、增加rank必然解决问题、单独放大gamma必然解决问题，都不受当前证据支持。

## 最有价值的后续诊断（未启动）

现有冻结100题实验保持协议继续完成。之后若分析第一次压缩，应记录块0分配到各输出carrier的路由权重、被丢弃块与残差的关联、输出残差范数，以及单独压缩前后readout。需要区分“目标信息没有进入残差”与“进入了表示但reader读不出来”。任何query/gold只能用于诊断读取，不能用于正式选块或测试时训练。

不能通过保留已知答案块、在这批测试题上扫描gamma/rank或改提示词，将本轮负面结果改写为正式方法收益。若开展这种实验，只能单独标为oracle/机制诊断。当前不恢复人工数据调优或逐episode alpha路线。

相关背景：MemoryLLM及后续M+讨论了固定记忆池更新下的知识保持和遗忘；M+引入长期存储与检索，说明该问题不限于本轮模块，但其不同容量与reader训练不能当作当前同预算问题的直接解决方案。官方来源：[M+，ICML2025](https://proceedings.mlr.press/v267/wang25au.html)、[作者实现](https://github.com/wangyu-ustc/MemoryLLM)。本文判断主要依赖本轮日志和实现，不借用其他方法的保护保证。

## 归档证据

- [固定37题诊断快照](../reports/public_retention_squad_interim_20261013/analysis_snapshot.json)：保存原始sample ID、question/gold、五个arm的关键阶段生成token/文本/评分、选块位置，以及归档时worker身份与文件哈希。
- [数据准备与版本记录](../reports/public_retention_setup_20261013/preparation.json)。
- 运行中完整分片仍在v100-2：`runs/public_retention_squad_author512.parallel4_v1.shard0.json` 至 `.shard3.json`。其内容持续增长；快照记录的worker文件哈希仅对应归档时文件，不能假装为最终结果文件哈希。

最终100题成绩和独立重评分证书出来后，应另外归档正式报告；保留此中间判断供审计，不将其替代最终结果。
