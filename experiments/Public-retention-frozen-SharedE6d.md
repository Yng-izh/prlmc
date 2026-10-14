# 冻结共享残差压缩器：公开原始知识保持任务

## 主张与研究范围

主方法为训练后完全冻结的共享 `CarrierResidualCompactor`。核心问题是在公开原始任务上，同预算 SharedE6d 是否优于 Selection，以及冻结模块是否能迁移到未参与训练的任务。此前人工24 episode及45.31%结果只作为附录机制诊断，不进入论文主要结果表。C13逐场景训练、C20–32、P13、RiskProtect、新架构和新共享模型训练暂停。

2026-10-13 用户取消当前自建数据实验。v100-2等待调度器72776和seed0/1 smoke79020/79021已停止，相关fleet/queue标记 `cancelled_by_user`；原始文件保留。此代码变更不启动替代GPU实验。

## 已实现入口

- `tools/prepare_public_retention.py`：按HuggingFace提交下载公开文件，逐文件SHA256；准备不可覆盖的输入计划。
- `tools/public_retention_eval.py`：冻结既有E6d，连续写入、预算触发压缩、逐次生成。
- `tools/verify_public_retention.py`：从保存的token重新decode/评分，核对输入、预算、完整配对和摘要。
- `tools/test_public_retention.py`：CPU模型合同测试。测试夹具不进入实验数据。
- `references/memoryllm_official/`：固定作者源码与来源哈希，供审计而非直接运行历史加载器。

## 数据版本与适配边界

作者仓库固定在 `5acaf52d2e60087f6bc38a06063d7558ab9a2c2c`，来源 `wangyu-ustc/MemoryLLM`。

| 数据版本 | 固定提交 | 选择规则 |
| --- | --- | --- |
| `YuWangX/KnowledgeRetention` | `c609067a4b7464dc4bf949cbb31996e47dd9be5a` | 作者.npy固定indices前100，不排序；原始dev问答、train干扰段落 |
| `YuWangX/KnowledgeRetentionProcessed` | `03a8db0d2d9ffd2caecae8f5f7af09294ece6e96` | 作者8B已筛选的 `val_data.json` 前100，公开干扰段落文件前30 |

Processed已经过作者的Llama3-8B可回答性筛选，不能描述为完整未筛选SQuAD/NQ测试集。原始版本与Processed分开报告，不混合分数。仅根据固定发布顺序选择，不用本轮答案或正确性选样。保留原context、question、answer及来源文件哈希；不生成事实、不改配答案。

原始NQ保留作者 `yes_no_answer == 'None'` 的字面过滤条件；合法token span和HTML过滤遵循作者实现。作者训练加载器在缺少document_tokens时以token位置切字符，存在问题：`author512`遇到此路径明确失败；`lossless`明确使用document_text token-offset适配并报告，不声称逐字节复现这个错误。加载完整所需索引范围，修正作者按 `indices[num]` 提前停止可能遗漏非排序indices的问题。

## 输入与读出协议

两种模式不得合并结果：

1. `author512`：相关段落加作者 `Context: ` 前缀并截前512 tokens；干扰段落按作者decode/re-encode后截512。报告每次截断数量。这是作者输入预处理对齐模式，不能宣称完整原文写入。
2. `lossless`（默认）：保存作者抽取的完整相关段落、完整发布干扰文本，以不超过512-token的字符片段写入，拼接必须等于源文本。完整指抽取段落，并非NQ整张维基页面。拼接干扰原文只使用作者的原始顺序与空格分隔。

原问题只按作者装饰：SQuAD `Questions: {question} Answer:`，NQ `Questions: {question}? Answer:`。每次写入后greedy生成10 token，不用B2、不加换行stop。评分为作者case-sensitive答案包含率，答案按作者tokenizer encode/decode并移除 `</s>`。这不是SQuAD官方normalized EM；报告字段明确为answer containment。完整保存每步generated token IDs、文本、评分及原始gold。问题或答案需截断时直接拒绝，不能悄悄改题。

每个样本、每个arm都恢复同一checkpoint初始memory和initialized；这是显式独立样本变体，不是作者无backup模式的跨样本连续记忆。查询保存并恢复memory、mask、initialized、模式及RNG，避免中间生成改变后续写入。短文本不丢弃，<16-token写入另作标记：作者提示此长度未匹配训练分布，结果解释需考虑该读写限制。

## 压缩条件与公平比较

固定已训练checkpoint SHA256：

`a9fe2ea3f7a4e2c21590cf2219591edb11fc65297ce36ddeed96c236d8020c5d`

rank32、gamma_max0.05；reader与共享参数均 `requires_grad=False`，无optimizer/backward/局部basis/alpha更新。共享参数前后状态哈希必须相同。记录MemoryLLM缓存manifest和实际建模代码哈希。checkpoint不符合固定哈希时拒绝，避免混入smoke或新模型。

Selection、Recent和SharedE6d固定20个native块的有效预算，V100模型block256时为5120槽。达到20块后、下一次写入前，将最旧19块压到9块，保留最新1块，再追加新块；以后按同样规则重复。没有只读答案anchor，没有未来question/gold参与选块。Selection和Shared调用同一个carrier scorer和同样写入重要性配置；Recent只保留压缩窗口最新9块。零初始化Residual必须复现Selection。连续压缩后各自记忆不同，后续native delta和选块允许随各自状态变化，这是系统累积效应。

物理pool维度保持checkpoint原样，通过active mask管理有效槽；5120不是总GPU内存或完整存储bytes。另报共享模块参数bytes、物理memory槽、峰值GPU分配和时间，不能宣传物理pool缩小。

Full不淘汰、不压缩，最多容纳模型物理pool；容量不足明确 `capacity_exceeded`，保留此前轨迹，末步Full不参与完整Full指标，不将部分流称为全量。Selection等主配对不因Full失败剔除样本。

NativeDefault使用原生 `update_memory=True`、原生随机逐槽淘汰、完整物理容量；其他输入、初始checkpoint和生成协议对齐。其容量与受管理的5120槽不同，必须单列为原生管线参考，不能将差值归因于残差模块。

## 结果与有效性前提

主指标：所有固定样本末步SharedE6d−Selection的配对答案包含率；同时保存逐步保持曲线、Recent及NativeDefault参考。主比较不得因失败样本或无压缩样本而自行挑选题目。

每次写入记录token数量/哈希、active slots、压缩次数；每次压缩记录source/target槽、保留块位置和时间。没有实际压缩标记 `not_applicable_no_compression`。相关文本阶段Full或NativeDefault从未答对时，列入reader不可回答样本；全部不可回答标记 `reader_reference_failed`，禁止恢复结论。少量可回答不能证明整体reader有效，报告完整覆盖情况。

Shared−Selection的所有gained/harmed作为行为收益。严格knowledge rescue只统计末步Full正确、Selection错误、Shared正确；Full并非分数上界。保留Full容量失败、reader不可回答样本，不将其低分直接归因于压缩。

95% bootstrap按原context聚类抽样，同一context所有QA一起抽，不把100QA当100独立场景。干扰流在所有样本共用，区间只是此固定流下的条件估计，不代表跨流/跨任务泛化。

## 运行命令（尚未启动GPU）

```bash
cd /share/latent/PRLM-C
HF_ENDPOINT=https://hf-mirror.com /share/latent/mllm_env/bin/python \
  tools/prepare_public_retention.py download \
  --repo YuWangX/KnowledgeRetentionProcessed \
  --revision 03a8db0d2d9ffd2caecae8f5f7af09294ece6e96 \
  --dataset-dir data/public_retention_processed_03a8db0

/share/latent/mllm_env/bin/python tools/prepare_public_retention.py prepare \
  --dataset-dir data/public_retention_processed_03a8db0 \
  --task squad --model-path /dev/shm/s1d_reader_fp16 \
  --mode lossless --num-samples 100 --unrelated-contexts 30 \
  --out runs/public_retention_squad_lossless.plan.json

/share/latent/mllm_env/bin/python tools/public_retention_eval.py \
  --plan runs/public_retention_squad_lossless.plan.json \
  --checkpoint runs/e6d_interleave_ckpt.pt \
  --model-path /dev/shm/s1d_reader_fp16 --cuda-device 0 \
  --out runs/public_retention_squad_lossless.results.json

/share/latent/mllm_env/bin/python tools/verify_public_retention.py \
  --plan runs/public_retention_squad_lossless.plan.json \
  --report runs/public_retention_squad_lossless.results.json \
  --model-path /dev/shm/s1d_reader_fp16 \
  --out runs/public_retention_squad_lossless.certificate.json
```

NaturalQA改 `--task naturalqa`，使用独立输出文件；author512用独立计划和结果。原始大文件版使用另一个dataset-dir和对应repo/revision，不混用indices。

## 验证与后续范围

本轮14项专用CPU测试通过，覆盖数据、同预算重复压缩、非零残差、生成状态隔离和保存结果篡改检测。本地及v100-2均执行专用测试。先前全仓159项测试中157项通过，2项历史native测试因本机缺服务器modeling文件而失败；未冒称全仓全部通过。真实8B GPU smoke及正式生成还未执行。

服务器已下载Processed固定版本并完成4个100样本计划：`runs/public_retention_{squad,naturalqa}_{author512,lossless}.plan.json`。author512各31次写入，超过20块预算；lossless SQuAD各61次、NaturalQA61–76次。lossless超过Full的50块容量，预计末步Full不可用，将如实报告。两种任务均有完整末步Selection/Recent/SharedE6d配对。准备过程为CPU数据处理，没有模型GPU推理。

实际准备发现NaturalQA有3个相关文本短于16 tokens，lossless干扰流有3个短片段。保留这些原始写入并在逐步记录中标记，不删样本；正式结果需报告读写限制和reader参考覆盖，不声称这些短片段已经通过8B有效性检验。

MemoryAgentBench EventQA/FactConsolidation标准任务扩展放在此门槛之后，其官方主评分应为substring_exact_match。旧 `s1d_shared_natural_eval.py` 是一次压缩、B2/EventQA recall诊断入口，不能当成新版标准协议主结果；本轮没有假装已完成该扩展，也不启动任何新训练或人工数据评测。

## SQuAD四卡分片执行

用户随后授权启动SQuAD两种输入模式，再明确优先 `author512`、暂停 `lossless`。原GPU0/2进程29594/30662已停止并保留旧输出；author512旧结果标记 `superseded_by_sample_sharding`，lossless标记 `stopped_by_user`。没有恢复lossless队列。

新增 `tools/run_public_retention_parallel.py` 和worker参数 `--shard-index/--shard-count`。维持同一个完整plan的原始SHA256，按原始sample_index模4分配到GPU0/2/3/4，每卡25题。每题全部五个arm都在同一卡评测，每个sample/arm恢复checkpoint与固定随机种子，分片不改变写入、评分、预算或方法。旧部分输出仅保留，新的100题结果由四个分片重新运行形成。

启动前逐卡预检used<=256MiB、free>=30000MiB。运行prefix为 `runs/public_retention_squad_author512.parallel4_v1`，launcher日志 `.launcher.log`，进度/PID登记 `.fleet.json`，worker结果 `.shard0.json` 至 `.shard3.json`，worker日志同名 `.log`。完成后逐分片重新decode/评分、检查原sample集合及协议/模型/共享参数哈希一致性；只有完整100题通过才输出总结果 `.json` 和 `.certificate.json`。任何worker失败保留原始文件、记录failed，不用部分题目伪造完整结果。

新增分片测试验证原始100题完整且各25题，以及漏题、重复shard、身份不一致的拒绝逻辑。本机与v100-2的15项CPU合同测试通过。

```bash
/share/latent/mllm_env/bin/python tools/run_public_retention_parallel.py \
  --plan runs/public_retention_squad_author512.plan.json \
  --devices 0,2,3,4 \
  --prefix runs/public_retention_squad_author512.parallel4_v1
```
