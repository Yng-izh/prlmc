# 原生策略低分与 MemoryLLM 论文协议核查

2026-10-13，对完成的 8 个 controlled development episode 进行只读检查，未改动结果、未打开封存测试、未运行新的 GPU 实验。

## 已完成结果

| 方法 | 正确/512 | 官方规则生成 EM |
|---|---:|---:|
| NativeDefault（冷启动，12800 槽） | 112 | 21.875% |
| NativeBudget（冷启动，2816 槽） | 124 | 24.21875% |
| Selection | 199 | 38.8671875% |
| SharedE6d | 232 | 45.3125% |

## 论文与代码依据

- MemoryLLM: Towards Self-Updatable Large Language Models, ICML 2024：https://arxiv.org/html/2402.04624v2 。§3.1.2/3.1.3 描述每次用最后 K 个记忆 token 做 self-update、随机删除 K 个旧 token、追加新 token；指数式保留量不是答案正确率保证。§4.5 使用 SQuAD/NaturalQA，自然文本知识注入后加入无关段落，提示为 Question/Answer，生成 10 token，以答案是否被输出包含计分。§3.3 的主模型为 Llama2-7B，7680 槽。
- 官方仓库：https://github.com/wangyu-ustc/MemoryLLM 。README 发布了后续 Llama3-8B / 12800 槽模型以及预训练版 Question/Answer 读出示例。
- 官方 QA 脚本：https://github.com/wangyu-ustc/MemoryLLM/blob/main/test_qa_memory.py 。现有公开脚本直接加载模型，未在每个样本前调用 initialized.zero_；支持备份/恢复记忆，部分设置跨样本连续更新。因此不是我们这次逐场景冷启动的严格复现，且 README 中此脚本示例仍是 7B。
- 官方模型代码：https://github.com/wangyu-ustc/MemoryLLM/blob/main/modeling_memoryllm.py 。本地下载后用去除位置属性的 AST 比较，inject_memory、drop_memory、update_memory_with_delta_memory、cat_memory_and_hiddens 四个函数与 v100-2 的文件全部相同；原生算法未被 PRLM-C 改写。
- M+: Extending MemoryLLM with Scalable Long-Term Memory, ICML 2025：https://arxiv.org/html/2502.00592v1 。指出随机淘汰会造成部分信息损失，并增加长文训练及检索机制；该文的 Llama3.1-8B 分阶段模型不可直接等同当前 Llama3-8B checkpoint。

## 已确认的协议差别

1. 本轮是 128 个实体、每个 2–3 个随机赋值版本，要求按编号回答最新版本；这不是论文自然段落保持测试。答案往往与真实世界知识相悖，读出需同时遵循覆盖规则。
2. 本轮统一用 B2 FEWSHOT / newline / 32 新 token / 官方两层 EM；论文保持实验是短 Question/Answer 提示、10 新 token、答案包含判定。相同提示便于本轮配对比较，但不能将本轮分数当成论文成绩。
3. 原生冷启动首次注入会把 256 槽复制铺满全池。第一段 30 行包含的实体与全部 evaluation 问题实体隔离。随后 19 次原生写入，每层随机丢弃 1/B 的槽；默认 B=50 时首块来源槽的预期保留占比为 (49/50)^19≈68.12%，等预算 B=11 时约16.35%。这是槽来源/直接副本占比的推算，并非事实信息占比或解释正确率的因果证明。后续 delta 也会条件于上次记忆，不可据此声称前段完全无用。
4. NativeBudget 改变原模型训练时的池长，由12800缩至2816；算法不变，但读出长度分布发生变化。它是原生策略的等预算配置，不应描述成未改配置的默认模型。
5. Selection/E6d 使用 append-only 写入、固定 anchor 和 active mask；Native 使用随机槽级淘汰、无 anchor。因此 E6d 相比 Native 的巨大差距包含写入/布局/掩码/选择策略及残差的联合差异，不能全部归因于共享模块。

## 输出证据

| 方法 | EM正确数 | substring正确数 | 错答中精确命中旧版本 |
|---|---:|---:|---:|
| Full | 186 | 187 | 68 |
| Selection | 199 | 199 | 34 |
| SharedE6d | 232 | 233 | 52 |
| NativeBudget | 124 | 124 | 19 |
| NativeDefault | 112 | 115 | 80 |

NativeDefault 的400道错答中80道精确命中同一事实的旧版本。其它错误未全面标注原因。简单放宽为现有官方 substring score 仅增加3题，当前差距不是单由 EM 格式严苛造成。

例：SønderjyskE Fodbold 的写入版本依次为 chess/boxing/gridiron football；NativeDefault 输出 boxing，SharedE6d 输出 gridiron football。Eric Decker 的版本为 goalkeeper/starting pitcher/flanker；NativeDefault 输出 wide receiver（不属于三个写入版本），SharedE6d 输出 flanker。后一输出与真实知识一致，但这里只举例，未将全部非版本错误认定为世界知识回退。

最终版本所在块的分布：early0–5 18题，middle6–12 147题，late13–18 347题。评测明显偏向后部，不是均匀保持测试；更小的池可能清除更多旧版本与初始副本，不能据这次结果声称缩小记忆总能提升。

## 结论与最小后续核验

本轮数据没有暴露可确认的原生写入/淘汰实现错误，低分存在合理机制解释；但当前只证明受控冲突任务、冷启动、固定 B2 下的行为。不得写成官方原生模型一般只有22% EM，或与原论文矛盾。

最有归因价值的已验证模块收益仍是 SharedE6d−Selection=+6.4453125pp。原生对照属于整个管线比较。

建议先用一个预先固定开发 episode，对默认12800槽比较：当前冷启动与每个场景恢复同一导出 checkpoint 记忆后再写入，两者保持 B2 不变。这能检查初始化敏感性。若核验论文可复现性，再运行官方 SQuAD/NaturalQA 的一个小批次及作者 Question/Answer/答案包含协议。后者应作为单独的论文 sanity check，不得按最佳提示替换当前受控对照分数，也不应打开封存测试。
