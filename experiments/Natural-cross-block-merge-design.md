# 自然多块记忆合并：代码与固定协议

本轮只实现、review 和 CPU 验证，不启动 GPU 训练或新生成评测。正在运行的 SQuAD author512 四卡实验继续原协议，其结果只作为已观察的开发诊断。

## 方法及边界

主方法只有冻结 reader 下的共享 `CarrierResidualCompactor`。新增 `SharedMemoryMerge` 保持全部 checkpoint 参数名、rank=32、gamma_max=0.05、原路由和残差归一化。没有增加新的训练参数，也没有改变载体选择目标。默认完整融合与旧 E6d 在相同输入、参数下逐元素一致。

E1 的 `CarrierOnlyFusion` 固定与完整融合完全相同的 9 个载体，再从源均值、learned key、attention softmax 和 values 中移除其余 10 个源块。屏蔽块数值的任意变化不能影响屏蔽版输出，屏蔽块的值梯度为零。这是一项因果输入消融，不把 attention 权重当作可读知识转移的证据。

默认策略是内容相关的原选择器，目标和运行时均不接收评测 QA。连续实验中先执行 OURS 的内容相关调度，再让 Selection 重放同一逐次载体索引；这样第一轮残差造成后续排序变化时，仍能隔离融合与选块差异。后续源表示本身当然已经不同，这是重复合并要检验的状态变化。

## 文件与职责

| 文件 | 职责 |
| --- | --- |
| `methods/prlmc/merge.py` | 与旧 checkpoint 兼容的完整融合、固定载体和源块屏蔽 |
| `tools/prepare_natural_merge.py` | 固定原始版本、文章隔离、原文/QA、分词与 suite 哈希 |
| `tools/natural_merge_runtime.py` | 共享 writer、安装布局、评分、NativeBudget 容量恢复 |
| `tools/natural_merge_train.py` | 一次共享自然训练；仅优化模块，连续压缩 |
| `tools/natural_merge_eval.py` | E1/E2/E3 完整逐题生成及压缩前后独立测量 |
| `tools/freeze_natural_merge.py` | 依据完整开发 E1 固定 checkpoint、模型、代码和全部正式计划 |
| `tools/run_natural_merge_parallel.py` | 按完整 context/写入流分片，多 GPU 评测、完整汇总与审计 |
| `tools/verify_natural_merge.py` | 独立重算逐题分数、载体与预算、配对收益、聚类置信区间 |
| `tools/test_natural_merge.py` | 源屏蔽、数据隔离、连续布局、真实梯度桥等 CPU contract 测试 |

## 数据及隔离

原始 SQuAD 来自固定 `YuWangX/KnowledgeRetention` revision `c609067a4b7464dc4bf949cbb31996e47dd9be5a` 的 train-v2.0/dev-v2.0，必须有逐文件 SHA256 的 `public_dataset_manifest.json`。复用已有准备器的下载能力，不以人造数据补齐缺失文件。

准备器强制提供已观察 100 题 SQuAD 的原计划，以及匹配其已固定作者 release 的 distractor 文件。通过原始 context 对齐原始 dev 文章，排除该文章的全部段落；无法全部对齐则停止，而不是假定隔离成立。已观察干扰段落所在训练文章也排除。其余 train 文章用于共享训练；dev 文章按固定 SHA256 排序，交替分配开发/正式，训练/dev/formal 按文章和 context 隔离。

E1 和训练的原始段落需连同 `Context: ` 完整落在 16–512 tokens；筛选只基于长度和原始有答案标注，不调用 reader、不根据 Full 正确性筛题。E1 保留入选 19 个 context 的全部原始有答案 QA；训练按原 QA ID 排序固定每 context 一题。保留原始 answer spans 和全部原始 aliases。

正式 E2 默认 32 个独立 dev contexts，保留全部原 QA。这是未采用作者 answerability 筛选的原始子集，用作者 author512 写入/生成/主评分协议；**不声称与历史作者固定 100 题是同一样本复现**。author512 仍会截断超过 512 tokens 的自然文本，完整保留原文及截断数，不能称为 lossless。

SQuAD 30/60 流使用同一组固定的自然干扰段落（60 的前 30 与 30 一致）。干扰文章与训练共享模块的文章隔离。NaturalQA 复用固定公开 author512/30 计划，保存原始公开字段和输入，匹配同一 tokenizer；公开干扰段落若源于 SQuAD，也核对其训练文章隔离。NQ 不参与模型选择或调参。

## 固定训练

- 冻结 MemoryLLM 参数，只训练一个共享模块；rank32、gamma_max0.05、20 块有效预算、19→9 不变。
- 原始 SQuAD train：8 条流，每条 39 个不同 context，4 轮流间交替，共 64 个压缩更新。没有扫描 rank、gamma、学习率或 checkpoint。
- 单个 AdamW（lr=1e-4，weight_decay=0.01）贯穿全部流；梯度范数上限 1。
- 每次压缩监督此前已写入的各来源原 QA，context 均衡的答案 CE 均值；gold 为主，第一版不使用 teacher（权重 0），不进行 Full-correct 过滤。
- 每流第 21/31 次写入前更新压缩器。第二次的源包含第一次已压缩状态。历史状态在事件边界 detach，明确采用 truncated/online gradient，而非全序列 BPTT。
- reader 每 QA 使用单独图，候选梯度累计后恢复源池，再重建一次压缩器图反传，复用现有 V100 节省显存的梯度桥。reader 参数不进 optimizer。
- 只保存最后预定更新的 checkpoint。`--smoke-updates` 的 checkpoint 明确 `evaluation_eligible=false`，不能混入正式结果。
- 初始化为新模块的原生载体恒等残差、固定 seed0；旧结构化 E6d 仅用于 E1 开发诊断，不作为新自然训练的初始化。

这是预先固定的一次训练配置，不是已证明最优的训练预算。可选 uncompacted teacher 的新配置需要另立协议，不能在本轮结果后悄悄加权。

## 三项实验与推进约束

**E1**：同一份预写入 19 块记忆，Full / Selection / SharedMerge / CarrierOnlyFusion。保留全部 QA 正确率、Full-correct 保留率诊断、成功来源 context 数、非载体 QA 上完整融合−屏蔽融合的配对新增与损伤。只将旧 E6d 放入 development E1。开发 E1 预先要求 Full 全 QA 包含率至少 10% 且至少读出 2 个来源 context（仅作最低工作负载可读性检查，不是效果目标）、正的 Shared−Selection、正的非载体完整−屏蔽净收益后，冻结自然 checkpoint 才能开启正式计划；这是推进门槛，不是统计显著性的保证。

**E2**：独立 SQuAD 30 干扰，共 31 次写入。六组 Full / Selection / Recent / SharedMerge / NativeBudget / NativeDefault。主指标为 OURS−Selection 作者答案包含率，按 context 聚类 bootstrap 95% CI；辅助 normalized EM/F1 分开报告。作者主指标保留 `skip_special_tokens=false` 解码；SQuAD 辅助指标另用同一生成 token 的 `skip_special_tokens=true` 文本，避免 EOS 控制 token 被当成回答文字，两份文本都存储并独立核对。所有 QA 的 predictions/tokens、原文、答案和来源标识完整保存。

**E3**：同一冻结 checkpoint 的 SQuAD 60（61 次写入）和 NaturalQA 30（31 次写入）。需提供完整同一冻结方案的正式 E2 报告，且 E2 的 context CI 下界大于 0 才放行。正式/迁移入口无 optimizer、basis 或 alpha 训练。不是根据 NQ 成绩再选 checkpoint。

每个 managed 20 块 arm 在写入 21、31、41、51、61 前，按需触发 19→9。单次 hook 的 before/after 之间只有压缩，没有下一次注入；另存完整逐次写入曲线。Full 超过物理容量时标 `capacity_exceeded`，末步分数为 unavailable，保留此前读出轨迹，绝不截短成有效最终 Full。

NativeBudget 使用 checkpoint 初始化状态的末尾 20 原生块，同时调整 native 容量及位置配置，调用原生写入/随机 slot 淘汰。与保持 50 块物理池、active mask 控制 20 块有效容量的主比较存在布局、初始化容量及 rotary offsets 差异，明确仅为系统级基线。NativeDefault 保持原物理容量。两个 arm 都不新增压缩模块；不启动之前暂停的冷启动对照。

## 准备与后续命令

下面只给可运行入口，本轮不执行 GPU 命令。先确保固定原始文件已下载，现有 NaturalQA 作者计划已准备：

```bash
cd /share/latent/PRLM-C
/share/latent/mllm_env/bin/python tools/prepare_natural_merge.py \
  --dataset-dir data/public_retention_original_c609067 \
  --observed-squad-plan <已经观察的100题原计划.json> \
  --published-distractors data/public_retention_processed_03a8db0/squad/unrelated_contexts_from_train.json \
  --naturalqa-plan <已准备的NaturalQA-author512-30计划.json> \
  --out-dir data/natural_merge_v1
```

先跑旧 E6d 诊断（E1 只有一个 episode，不适合拆到四卡）：

```bash
/share/latent/mllm_env/bin/python tools/natural_merge_eval.py \
  --plan data/natural_merge_v1/E1-development.json \
  --checkpoint runs/e6d_interleave_ckpt.pt --legacy-diagnostic \
  --cuda-device <空闲卡> --out runs/natural_merge_old_e1.json
```

自然共享训练、开发 E1、冻结（按顺序）：

```bash
/share/latent/mllm_env/bin/python tools/natural_merge_train.py \
  --plan data/natural_merge_v1/train.json --cuda-device <空闲卡> \
  --out runs/natural_merge_train.json --checkpoint-out runs/natural_merge.pt
/share/latent/mllm_env/bin/python tools/natural_merge_eval.py \
  --plan data/natural_merge_v1/E1-development.json --checkpoint runs/natural_merge.pt \
  --cuda-device <空闲卡> --out runs/natural_merge_e1_development.json
/share/latent/mllm_env/bin/python tools/freeze_natural_merge.py \
  --suite data/natural_merge_v1/suite.json --checkpoint runs/natural_merge.pt \
  --development-plan data/natural_merge_v1/E1-development.json \
  --development-result runs/natural_merge_e1_development.json \
  --out runs/natural_merge_freeze.json
```

正式 E1 使用同一冻结入口加载 `E1-formal.json`；随后 E2 按 context 分到四张确实空闲的 GPU：

```bash
/share/latent/mllm_env/bin/python tools/run_natural_merge_parallel.py \
  --plan data/natural_merge_v1/E2-squad30.json --checkpoint runs/natural_merge.pt \
  --freeze runs/natural_merge_freeze.json --suite data/natural_merge_v1/suite.json \
  --devices <四张空闲卡号> --out runs/natural_merge_e2.json
```

E3 使用该入口，更换计划为 `E3-squad60.json` 或 `E3-naturalqa30.json`，追加：

```bash
--e2-result runs/natural_merge_e2.json --e2-plan data/natural_merge_v1/E2-squad30.json
```

并行 launcher 只接收每流/每原始 context 的分片，完整汇总后再次验证全部 QA、计划/模型/code/checkpoint 同一性和载体重放。卡空闲预检默认需要每卡至少 28000 MiB；真实 GPU 峰值仍需后续 smoke 才能确认。

## 成本及论文表述

报告有效 slots、物理 slots/字节、额外模块参数/状态字节、逐次压缩 wall time、总时间与 CUDA peak allocated/reserved。物理池仍维持模型配置大小，不能宣称节省物理模型显存。不同 arm 的 NativeBudget 物理容量另行标注。

Full-correct retention 是机制诊断；正式主分数始终包括全部原始 QA。主 CI 的单位是 context，且固定的共同 distractor 流带来相关性，因此该区间条件于当前流，不等于跨流或跨任务泛化 CI。NaturalQA 的冻结一次迁移结果才能提供跨数据集证据。

人工 24 episode、C13、扩维/投影/保护权重历史只用于附录诊断，不接入本轮 train 或正式评测。

## 本轮实现验证与尚未执行项目

2026-10-13 本地 19 项新 CPU tests 和 15 项现有 public-retention tests 通过；新 tests 在 v100-2 的实际 Python/PyTorch 环境也执行，GPU 可见性置空。包括完整准备流程的测试 fixture、源屏蔽的输出与梯度隔离、旧算子逐元素兼容、64-update 协议的 train/QA 隔离、梯度桥、重复压缩、NativeBudget 异常恢复、逐题 token 重解码、分片覆盖与 E3 推进门槛。fixture 只验证代码，不属于研究数据。

代码实现时尚未补齐原始数据。后续数据准备进度见下节；当前 checkpoint 的真实 GPU 前向、一次自然训练更新和显存峰值仍未验证，不能把 CPU contracts 写成正式实验已完成。

## 原始数据已补齐：2026-10-13

固定原始 SQuAD revision 的 train/dev/官方索引已下载到服务器 `data/public_retention_original_c609067/`，并保存 `public_dataset_manifest.json`。服务器直连不可用、镜像返回 403，因此由本机 Python 下载固定 Hugging Face 原始文件，再经 SSH 传输。train 与索引逐字节大小和 LFS SHA256 均匹配上游；dev 大小和 Git blob SHA1 匹配上游，同时另存本地 SHA256。传输后服务器重新校验全部原始文件。

NaturalQA 复用服务器已经固定的作者 processed release `03a8db0d2d9ffd2caecae8f5f7af09294ece6e96`；该轮迁移只需要公开 release 原 context/question/answer 与 30 段干扰输入，不额外下载本轮不使用的完整 NQ train/dev JSONL。

`data/natural_merge_v1/suite.json` 及六份计划已经生成，未启动新训练或生成评测：

| 计划 | 原始 context | 原始 QA | 每流写入 |
| --- | ---: | ---: | ---: |
| train | 312 | 312 | 39，8 条流 |
| E1-development | 19 | 93 | 19 |
| E1-formal | 19 | 94 | 19 |
| E2-squad30 | 32 | 153 | 31 |
| E3-squad60 | 同 E2 的 32 | 同 E2 的 153 | 61 |
| E3-naturalqa30 | 99 | 100 | 31 |

已观察的 SQuAD 100 题映射至 `Normans` 和 `Computational_complexity_theory` 两个文章，均从新划分排除。训练、开发与正式数据按文章隔离；E1-formal 与 E2-context 互不重复，允许属于相同的正式文章池。SQuAD30 与 SQuAD60 明确复用相同正式 contexts 和前 31 次写入。NaturalQA 原 100 题中两个问题共享一个 context，所以合并成 99 条写入流，保留全部 100 题。

新增 `tools/stage_natural_merge_data.py` 支持显式 endpoint、固定版本、重试和上游 Git/LFS 校验；`tools/verify_natural_merge_data.py` 不加载 reader，独立核对全部 SQuAD 原文/答案跨度/QA 标识、逐次重新分词、公开 NQ 字段与写入、计划哈希、文章/context 隔离、旧文章排除及30/60输入一致性。实数据审计已通过，证书为 `reports/natural_merge_implementation_20261013/data_certificate.json`；suite 索引副本为同目录 `data_suite.json`。

训练、E1、E2/E3 入选相关 SQuAD 段落均未发生 512-token 截断；NaturalQA 100 题中 12 次相关段落写入发生作者 author512 截断，保持官方既定输入协议，不称为 lossless。以上均为输入准备与完整性核验，不是实验结果。

## Execution outcome (2026-10-14)

The earlier implementation-only statements describe historical status. A real GPU smoke, all 64 scheduled shared training updates, and legacy/new checkpoint E1 development evaluations have now completed on v100-1. Full / Selection / SharedMerge / CarrierOnlyFusion scored 35 / 26 / 27 / 27 correct out of 93 original questions. The noncarrier full-minus-masked net was zero, so the predeclared progression gate failed. Formal E1, E2 and E3 were not launched; automatic follow-up is paused.

See `reports/natural_merge_initial_20261013/results.md` for the final decision and this directory for question-level predictions, training traces, logs, the final shared checkpoint, plan identities/data certificate, and the failed freeze command. Public raw datasets are downloaded separately using the pinned manifest in `data/public_retention_original_c609067/public_dataset_manifest.json`. Implementation certificates retain their historical scope; actual GPU evidence is recorded in the completed run reports.
