# 自然记忆融合 v2：A/B/C 运行方案

执行已完成：A/B/C 各完成 64 次更新、训练内诊断和开发 E1。三组均未通过开发推进门槛，未启动正式 E1/E2/E3。完整结果见 [本轮裁决](../reports/natural_merge_v2_20261014/results.md)；下文保留预定方案与运行命令。

目的：区分训练监督与槽位融合结构的作用，检验冻结共享模块是否恢复被选择压缩丢失的知识。原始自然数据、MemoryLLM reader、生成与评分协议沿用 v1；训练计划另存，不覆盖旧文件。

| 组别 | 融合结构 | 目标 |
| --- | --- | --- |
| A | 现有块级 SharedMemoryMerge | 平均原始答案 CE |
| B | 与 A 相同 | 压缩损失加权 CE + Full 答案位置 KL |
| C | 新 SlotMemoryMerge | 与 B 相同 |

A 是新课程下重新训练的对照，不能将历史 v1 checkpoint 当成 A。三组均从 seed=0 的零残差初始化开始，不使用 smoke checkpoint 续训。B−A 对应监督变化，C−B 对应结构变化。C 仅从待丢弃块提取信息，屏蔽非载体后精确退化为 Selection；它与旧结构相比同时改变槽位匹配和融合源限制，C−B 衡量这项整体结构改动，不能单独归因于注意力粒度。

## 固定训练协议

- 同一批 312 个原始 SQuAD train context、8 条流，保留既有文章级 train/dev 隔离。恢复这些 context 的全部原始 QA，每轮按排序后的 QA ID 轮换一个。原文、问题、答案均不重写。
- 64 次更新、4 轮；每轮每条流 2 次更新。AdamW，lr=1e-4、wd=0.01、clip=1；rank=32、gamma_max=0.05，19→9 块压缩。
- 前 32 次：单次合并。第 0 轮使用每条流的位置 `[0:19]` 与 `[20:39]`；第 1 轮使用 `[1:20]` 与 `[20:39]`，每个窗口独立 reset。这个错位确保全部 39 个 context 都参与过压缩监督。
- 后 32 次：完整 39-context 流，在写入位置 20、30 之前更新。早期压缩状态由当前模块产生并截断梯度；不替换成 Full 记忆。
- 每个事件每个已写入 context 监督一道原始 QA；三组 QA、顺序、更新次数完全相同。
- 只保存最后计划更新的 checkpoint，不按中间生成结果选 checkpoint。smoke 输出永久不具备正式评测资格。

Full teacher 使用独立 reset 后的未压缩写入历史，在与 student 相同原始写入前缀上读取。连续阶段需要能容纳至少 39 个原生写入块的物理池；超容量会报错，不把已压缩历史称为 Full。Selection reference 来自当前 student 的同一压缩前状态，只对本次窗口做选择，因此连续阶段是局部压缩对照，不是独立完整 Selection 系统。

B/C 的训练 QA 权重：

```text
raw_weight_q = 1 + clamp(CE_Selection_q - CE_Full_q, 0, 2)
weight_q = raw_weight_q / mean(raw_weight)
loss = mean(weight_q * CE_Shared_q) + 0.5 * mean_eligible(KL(Full || Shared))
```

KL 温度固定 1，仅在 Full gold-answer CE ≤ 2 的训练 QA 上启用；若没有合格 QA，KL 为零。该阈值是固定的训练可靠性代理，不等于生成正确性保证。CE/KL 都按答案 token 平均，包含与旧协议一致的答案终止 token。没有开发 QA 加权、生成正确性选样或测试时优化。A 也计算 Full/Selection reference 以保留同样诊断与 reader 计算，但不使用加权和 KL。

## 模块和诊断

C 使用载体槽位作为 query，允许融合的非载体槽位作为 key/value；在参考层算路由、跨层共享，在低秩 value 空间聚合，分 query chunk 计算分数。残差写回原载体，零初始化与相对 RMS 上限保留。载体选择仍由原 content-only selector 决定，问题不进入模块 forward。

每次更新保存逐题 Full/Selection/Shared CE、QA 权重、teacher eligibility/KL、TF token accuracy、各参数梯度范数、融合 trace，以及实际 FP16 写回残差 RMS 比和改变的元素比例。逐题 CE 是 optimizer 更新前测量，安装后的残差是更新后测量；两者不混称同一 checkpoint 的读出。训练 TF 诊断不能代替真实生成恢复。

冻结开发 E1 使用原有四组 Full/Selection/SharedMerge/CarrierOnlyFusion、全部 93 道原始 QA、10-token greedy generation 与作者答案包含率，另报 normalized EM/F1。保留较短生成上限是为了与原实验可比，不能将这项协议称为已经优化的生成协议。

## 部署文件

在本地仓库根目录执行以下命令只同步代码，不启动 GPU：

```bash
tar -cf - \
  methods/prlmc/slot_merge.py \
  tools/prepare_natural_merge_v2.py \
  tools/natural_merge_v2_train.py \
  tools/natural_merge_v2_training_fit.py \
  tools/run_natural_merge_v2.py \
  tools/summarize_natural_merge_v2.py \
  tools/test_natural_merge_v2.py \
  tools/natural_merge_runtime.py \
  tools/natural_merge_eval.py \
  experiments/Natural-merge-v2-ABC.md | \
  ssh v100-1 'tar -xf - -C /share/latent/PRLM-C'
```

source_manifest 新增 v2 文件并记录全部哈希。旧归档保留原有 source manifest；新代码不能配用旧 freeze 证书。旧 block checkpoint 未写 architecture 时仍按原结构严格加载。

## 准备数据与运行

服务器执行一次数据准备。这会恢复同一训练 context 的全部原始 QA，不读取开发/正式评测答案来制定训练计划：

```bash
cd /share/latent/PRLM-C
/share/latent/mllm_env/bin/python tools/prepare_natural_merge_v2.py \
  --base-train-plan data/natural_merge_v1/train.json \
  --dataset-dir data/public_retention_original_c609067 \
  --model-path /dev/shm/s1d_reader_fp16 \
  --out data/natural_merge_v2/train.json
```

选实际空闲的物理卡号。以下以 GPU 1、2 为例；运行器会拒绝空闲显存不足 28000 MiB 的卡，且不会停止其他任务：

```bash
/share/latent/mllm_env/bin/python tools/run_natural_merge_v2.py \
  --train-plan data/natural_merge_v2/train.json \
  --dev-plan data/natural_merge_v1/E1-development.json \
  --model-path /dev/shm/s1d_reader_fp16 \
  --devices 1,2 \
  --variants B,C,A \
  --out-dir runs/natural_merge_v2_abc
```

运行器前台运行；需要脱离终端时，可用 tmux 执行相同命令。它先并行 B/C，各组独立执行 1-update smoke → 全新 64-update 训练 → 固定训练窗口生成诊断 → 冻结开发 E1；A 随空闲卡排队。三张卡时传入三个卡号即可三组并发，一张卡时自动顺序执行。输出目录必须不存在，避免覆盖或重复启动；失败后先检查日志，再用新输出目录与 `--variants` 指定需要重跑的组，不自动从失败 checkpoint 恢复。终止运行器时只终止它自己启动的子任务。

也可以分别手动启动单组，先 smoke：

```bash
/share/latent/mllm_env/bin/python tools/natural_merge_v2_train.py \
  --plan data/natural_merge_v2/train.json --variant B --cuda-device 1 \
  --out runs/v2_B.smoke.json --checkpoint-out runs/v2_B.smoke.pt --smoke-updates 1
```

smoke 成功后正式训练使用新输出，去掉 `--smoke-updates`；完成后冻结生成：

```bash
/share/latent/mllm_env/bin/python tools/natural_merge_v2_train.py \
  --plan data/natural_merge_v2/train.json --variant B --cuda-device 1 \
  --out runs/v2_B.train.json --checkpoint-out runs/v2_B.train.pt

/share/latent/mllm_env/bin/python tools/natural_merge_eval.py \
  --plan data/natural_merge_v1/E1-development.json \
  --checkpoint runs/v2_B.train.pt --cuda-device 1 \
  --out runs/v2_B.e1_development.json
```

## 输出和裁决

队列目录保留 `launcher.json`（PID/GPU/命令/退出码）、A/B/C 的 smoke/train/fit/eval 日志与 JSON、训练 checkpoint，以及 `comparison.json`。fit 固定使用第一条训练流的前 19 个 context、全部原始 QA，报告明确标为训练内拟合诊断，不能用于正式验收或泛化结论。

汇总独立重算评分并检查三组模型、输入记忆、载体、Full/Selection 预测完全一致；报告 B−A、C−B 的逐题新增/损伤与 context bootstrap、Full-correct/Selection-wrong 集合和恢复数、非载体完整融合−屏蔽融合收益。

开发推进门槛保持：Full 最低可读性达标、Shared−Selection > 0、非载体完整融合−屏蔽融合 > 0。不能因为 CE 下降或偶然多答对一道就宣称跨块恢复。该开发集已经被观察，正结果仍只是方法开发证据。

运行器在开发对比后停止，不自动开启正式 E1/E2/E3，不改门槛、不扫描超参、不根据 NaturalQA 选择方案。通过后才按原冻结工具制作新的证书。若多个候选通过，需要先固定选择规则再进行独立正式评测。

## 验证状态

CPU 测试覆盖原 checkpoint 行为、槽位融合的源屏蔽/置换行为、梯度桥、reader 冻结、KL 与权重、训练数据恢复和隔离、课程覆盖、实际 FP16 写回诊断、开发汇总。

```bash
/share/latent/mllm_env/bin/python -m unittest discover -s tools -p 'test_natural_merge*.py' -v
/share/latent/mllm_env/bin/python -m unittest discover -s tools -p 'test_public_retention.py' -v
```

首次真实 V100 smoke 在 frozen-reader backward 分配 3.12 GiB 物理池梯度时 OOM，尚未进行参数更新。修复将原始 source window 在 reader backward 期间暂存 CPU，释放 source/carriers GPU 副本；随后恢复原值计算 compactor VJP。该修复不改变数据、目标或压缩预算。失败日志保留在 `runs/natural_merge_v2_abc_20261014_022647/`。运行器中的 smoke 是正式启动前的必要检查；实际峰值显存、耗时与方法效果以重跑日志为准。单次 smoke 覆盖第一次合并；连续阶段另有 CPU teacher/写入/梯度契约测试，真实连续阶段仍需以训练日志确认。

重跑 `runs/natural_merge_v2_abc_20261014_023355_r2/` 中 A/B/C smoke 全部通过，三组正式 64-update 训练已在 GPU 1/2/3 同时启动；启动成功不代表最终方法有效性已经通过。

## 最终执行结果（2026-10-14）

队列和全部 12 个阶段均完成，退出码均为 0。开发 E1 的 Full / Selection / A / B / C 为 35 / 26 / 27 / 27 / 27（93 题；作者答案包含率），A/B/C 正误集合相同；新增同一道题且损伤为零。Full-correct / Selection-wrong 的 9 道题均未恢复，44 道非载体来源题上的完整融合−屏蔽融合净收益均为零，因此没有候选通过门槛。

固定训练窗口的 88 道原始 QA 上，Full / Selection / A / B / C 为 47 / 28 / 29 / 29 / 29；该结果只属于训练内诊断。新监督和槽位结构在本次开发生成指标上未改善主要结果，不据此推断所有自然任务或其他训练预算均无效。没有扫描超参、选择中间 checkpoint、放宽门槛或运行正式测试。

完整原始报告、日志、最终 checkpoint、运行时源码快照与 SHA256 清单见 [reports/natural_merge_v2_20261014](../reports/natural_merge_v2_20261014/)。
