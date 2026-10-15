# 自然记忆融合 v3：C/D 固定对照

2026-10-14。代码阶段；尚未运行 GPU smoke、训练或生成，不包含新实验成绩。既有 A/B/C 结果和原始 source snapshot 保持不变。

## 要回答的问题

同一训练监督、总体残差预算及课程下，分组低秩变换、稀疏语义路由与逐槽位门控的组合，是否优于已有 C 的稠密槽位融合？这是一次整体结构对照，不将结果分别归因于三个组件。

| 项目 | C：已有槽位架构、新训练 | D：新结构、相同训练 |
| --- | --- | --- |
| checkpoint architecture | `budgeted_dense_slot_v3` | `grouped_gated_sparse_slot_v3` |
| 载体选择 | 原 content-only Selection，不改打分 | 相同 |
| 路由 | 第 0 层，所有丢弃槽位稠密注意力 | 第 16 层，Top-8 丢弃槽位注意力 |
| value down/up | 全层共享，rank32 | 8 组，每 4 层共享，rank32 |
| 非线性 | `GELU(context - carrier_values)` | 相同，避免额外更换激活与分支 |
| RMS 对齐 | 每个载体块 | 每个槽位 |
| 门控 | 单标量，上限 .05，初始 .02 | 可学习逐槽位门控，上限 .20，初始 .02 |
| 总体残差上限 | 实际存储变化 RMS / 载体 RMS ≤ .05 | 相同 |

两组都采用 `output = carrier + gate * normalized_delta`，up 零初始化，step 0 精确等价于 Selection。D 的门控由载体第 16 层表征产生，各层共享同一个槽位门控和 Top-8 路由，以保持槽位对应关系。没有 QA 输入模块，没有新 reader、冗余度选槽或选块器改造。D 的局部 .20 上限允许集中写入，不能扩大整体 .05 扰动预算。

总体上限覆盖全部 32 层及全部输出槽位。先用同一可微缩放限制残差，为最终 dtype 转换留出舍入余量，再检查写回 FP16 后的实际比率。零载体也不得产生非零写入。记录 `budget_scale`、`stored_residual_rms_ratio`、`stored_changed_fraction` 和 D 门控范围；参数及存储开销按实报告，不声称 C/D 参数量相同。

## 数据、目标与训练

沿用 v2 固定的 312 个原始 SQuAD train context、1402 道原始 QA、8 个 39-context 写入流，保持文章隔离、原文和原始 QA。准备器从原始答案文本重新 tokenize，不拼接字面量 `</s>`，不按固定 token ID 截断。原 v2 计划保存在 v3 计划中，旧计划、开发集 scoring target、生成长度和评分规则不修改。

每次更新沿用一 context 一道原始 QA 的确定性轮换。来源于当前保留原生载体或未压缩 tail 的 QA 属于保护组；其余已写入 context 的 QA 属于恢复组。

循环压缩中保留原生来源索引随块移动，不把融合进载体的历史丢弃 context 当作原生载体。因此第二次压缩时，之前丢弃的事实仍进入恢复组。源索引使用原始 context ID，不使用压缩后的位置替代。这个划分只在训练数据上使用，测试生成不选题、不创建 optimizer。

训练损失：

\[
L = \operatorname{mean}_{q\in Q_{recover}} w_q CE_{Shared}(q)
 + 0.5\operatorname{mean}_{q\in Q_{recover}, CE_{Full}\le2} KL(Full\Vert Shared)
 + \operatorname{mean}_{q\in Q_{protect}}\operatorname{ReLU}(CE_{Shared}-CE_{Selection}).
\]

`w=1+clamp(CE_Selection−CE_Full,0,2)`，只在恢复组内归一化为均值 1。Full 是同原始写入前缀的独立未压缩教师；Selection 使用当前 student 历史，只把当前窗口换为同一批原生载体。保护 λ=1、margin=0、KL 温度=1；无合格教师题时 KL 为 0。保护组不参与主 CE 或 KL。

共同训练设置：AdamW，矩阵 lr=.001、门控 lr=.01，weight decay=.01，gradient clip=1，seed0。reader 冻结，仅更新模块。64 次更新：前 32 次单次 19→9 合并，后 32 次循环写入合并。沿用既有 source CPU 暂存与 reader 梯度 VJP 桥。smoke 不续训、不作候选；正式训练全新初始化，只使用第 64 次更新的 checkpoint。

## 入口与执行

| 文件 | 功能 |
| --- | --- |
| `methods/prlmc/gated_slot_merge.py` | C 全局预算封装、D 分组稀疏门控 |
| `tools/prepare_natural_merge_v3.py` | v3 固定计划、原始答案重新编码 |
| `tools/natural_merge_v3_objective.py` | 两组共同损失、优化器和原生来源追踪 |
| `tools/natural_merge_v3_train.py` | fresh 64-update 训练 |
| `tools/natural_merge_v3_training_fit.py` | 固定首个训练流前 19 context 全 QA 生成诊断 |
| `tools/run_natural_merge_v3.py` | 单卡/双卡队列、日志、PID 与状态 |
| `tools/summarize_natural_merge_v3.py` | C/D 配对结果、压缩损失恢复与开发门槛 |
| `tools/test_natural_merge_v3.py` | CPU 模块、梯度、来源、计划及汇总契约 |

准备计划（需要匹配 tokenizer 的模型缓存）：

```bash
cd /share/latent/PRLM-C
/share/latent/mllm_env/bin/python tools/prepare_natural_merge_v3.py \
  --v2-train-plan data/natural_merge_v2/train.json \
  --model-path /dev/shm/s1d_reader_fp16 \
  --out data/natural_merge_v3/train.json
```

按实际空闲 GPU 替换 `--devices`，两卡并行各跑一组；一张卡时依次排队：

```bash
/share/latent/mllm_env/bin/python tools/run_natural_merge_v3.py \
  --train-plan data/natural_merge_v3/train.json \
  --dev-plan data/natural_merge_v1/E1-development.json \
  --model-path /dev/shm/s1d_reader_fp16 \
  --devices 1,2 --variants C,D \
  --out-dir runs/natural_merge_v3_cd
```

输出目录必须不存在，每卡启动前需要至少 28000 MiB 空闲。队列持有 GPU 文件锁并检查空闲显存；不会关闭无关进程。每组执行 smoke → 全新训练 → 训练内生成诊断 → 开发 E1，保存日志、checkpoint、逐题结果及 `comparison.json`。评测继续复用 `natural_merge_eval.py`、`verify_natural_merge.py`；runtime 严格区分新旧 architecture 并检查模块配置。保留旧 A/B/C state_dict 加载能力。

## 裁决与边界

主要比较 Shared−Selection、D−C、非载体完整融合−屏蔽融合，以及 Full-correct/Selection-wrong 的恢复数量。源屏蔽在路由之前移除丢弃块，固定同一载体。C/D 屏蔽后都精确返回 Selection。

沿用既有开发推进门槛：Full 最低可读性通过，Shared−Selection 净差严格为正，非载体完整融合−屏蔽融合净差严格为正。训练内结果不替代开发门槛；开发集已参与方法开发，不作为独立泛化证据。队列不自动开启正式 E1/E2/E3，不扫描超参，不选择中间 checkpoint；正式阶段仍需 freeze，E2 context-bootstrap CI 下界严格大于零后才能进入 E3。

本次只完成代码。CPU 契约测试不能确认 V100 上的峰值显存、速度或方法收益，这些由首次 GPU smoke 和完整生成报告给出。
