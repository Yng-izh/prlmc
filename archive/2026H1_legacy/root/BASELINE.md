# MemoryLLM-8B on MemoryAgentBench — 已固化 baseline

冻结时间：2026-09-24
状态：链路已验证（Tier 0 端到端跑通），Tier 1 尚未运行

本文件记录跑分所需的全部参数。改变了其中任何一项，结果就与已有记录不可比。

## 一、模型与权重

| 项 | 值 |
|---|---|
| 权重路径 | `/share/latent/MemoryLLM-8B` |
| 来源 | `YuWangX/memoryllm-8b`，8 个 safetensors，共 39,134,461,953 字节 |
| 原始精度 | float32（39.13 GB，超出单卡 V100 32 GB） |
| 使用精度 | **float16 ≈19.6 GB** |
| 加载 | `MemoryLLM.from_pretrained(path, torch_dtype=float16, attn_implementation="sdpa")` |
| 分片 | **禁用 device_map="auto"** |
| 架构 | `MemoryLLM(LlamaForCausalLM)`，32 层，hidden 4096，vocab 128256 |
| 记忆池 | 32 层 × 12800 token × 4096 = 1.6777 B 参数，约 6.71 GB(fp32) |
| 记忆架构 | `num_blocks 50` × `num_tokens 256` = 12800 token/层 |
| LoRA | r=16, alpha=32, dropout=0.1, target q/k/v/gate/up/down，附加 `decoder_adapter` |

`config.json` sha256 前 16 位 `ffb703537dac7bd7`；`generation_config.json` `11951cc646c18d8e`；`model.safetensors.index.json` `a59754562bc49d55`。

## 二、环境

| 包 | 版本 |
|---|---|
| Python | 3.10.12 |
| torch | 2.5.1+cu124 |
| transformers | **4.48.2** |
| peft | 0.10.0 |
| tokenizers | 0.21.4 |
| huggingface_hub | 0.36.2 |
| accelerate | 1.15.0 |
| numpy | 1.26.4 |
| GPU | Tesla V100-PCIE-32GB, sm_70 (7,0), CUDA 12.4 |

解释器：`/share/latent/mllm_env/bin/python`。**bench 全部模块在此环境下直接 import，无需跨环境桥接。**

未安装且有意排除：`flash_attn`、`minference`、`faiss-gpu`（用 `faiss-cpu` 替代）、`frontend`、`hipporag`。

## 三、运行参数（固定）

| 参数 | 值 | 理由 |
|---|---|---|
| `attn_implementation` | `"sdpa"` | V100 是 sm_70；flash-attn 2.x 要求 sm_80+，且容器无 `nvcc` |
| `torch_dtype` | `float16` | fp32 装不下；bf16 在 sm_70 无硬件加速 |
| `chunk_size` | **400** | 模型训练上限 512，bench 还会前置约 60 token 模板，400 留余量 |
| `inject_max_tokens` | 512 | 与模型 `max_length` 一致 |
| `max_new_tokens` | 32 | 默认；ICL 类任务如需更长可调，但改了就不可比 |
| `do_sample` | `False` | 贪心解码。模型自带 `generation_config` 是 `do_sample=true, temperature=0.6`，**被 adapter 覆盖** |
| `reset_memory` | 每个 context 前执行 | checkpoint 自带 `initialized=1` |
| `MIN_INJECT_TOKENS` | 16 | 模型卡硬下限；低于此值会**扰动**而非更新池 |
| `device` | 单卡 `cuda:0` | 固定 6/7/9 中一张空闲卡 |

Agent 配置：`configs/agent_conf/MemoryLLM/memoryllm_8b.yaml`，sha256 `a2100000dfd89cfc...`

**`chunk_size` 不能用 bench 默认的 4096。** 4096 会被 `inject` 截断到 512，每个 chunk 丢失 87% 内容。

## 四、代码指纹

| 文件 | sha256 前 16 位 |
|---|---|
| `methods/memoryllm.py` | `aa9def532fd800b0` |
| `agent.py` | `0289a09a2a5a950f` |
| `utils/templates.py` | `691384d9f2fdfffe` |
| `utils/eval_data_utils.py` | `05f37b56e60fad7a` |
| `methods/memoryllm_modeling/modeling_memoryllm.py` | `48b509c7f1476239` |
| `methods/memoryllm_modeling/configuration_memoryllm.py` | `abd00d3985f96fa2` |

改动三处宿主文件（补丁幂等，`apply_patch.py`）：`agent.py` 加 `memoryllm` 初始化与分发分支；`utils/templates.py` 加 `'memoryllm': 'rag_agent'`；`utils/eval_data_utils.py` 加本地 parquet 回退。

## 五、Tier 0 实测（唯一已跑过的配置）

`factconsolidation_sh_6k`，chunk 400，5 条 query，GPU 6。

| 指标 | 值 |
|---|---|
| context 数 | 1（共 100 条 query，实跑 5） |
| chunk 数 | 16 |
| 模型加载 | 234.6 秒（**一次性，进程内共享**） |
| 记忆构建 | 6.67 秒（首次 query 记的值），后续 4 条记 0.0 |
| 单条 query | 3.48–3.98 秒，均值 3.59 |
| 整 context 墙钟 | 258.88 秒 |
| exact_match | 20.0 |
| f1 | 26.67 |
| substring_exact_match | 40.0 |
| rougeL_f1 | 31.67 |
| input_len / output_len | 161.8 / 32.0 |

**这 5 条 query 的分数不构成评测结论**，只证明链路连通。

### 时间语义（易误读）

`memory_construction_time` 是 `time.time() - agent_start_time`，只覆盖从 agent 构造到该次 query 之间的时段，**且只在第一条 query 上非零**。16 个 chunk 的注入合计 6.67 秒，即 **0.42 秒/chunk**。

我此前按 14.2 秒/chunk 做的耗时外推**作废**——那是把模型加载时间错误摊到了 chunk 上。

## 六、Tier 1 数据量（已核实的 chunk 数）

以下 chunk 数由 `ConversationCreator.get_chunks()` 在 chunk_size=400 下实测，非估算。

| 域 | 子集 | contexts | chunk/ctx | 总 chunk | query 可用 |
|---|---|---|---|---|---|
| Conflict_Resolution | factconsolidation_sh_6k | 1 | 16 | 16 | 100 |
| Conflict_Resolution | factconsolidation_mh_6k | 1 | 16 | 16 | 100 |
| Conflict_Resolution | factconsolidation_sh_32k | 1 | 83 | 83 | 100 |
| Conflict_Resolution | factconsolidation_mh_32k | 1 | 83 | 83 | 100 |
| Conflict_Resolution | factconsolidation_sh_64k | 1 | 169 | 169 | 100 |
| Conflict_Resolution | factconsolidation_sh_262k | 1 | 688 | 688 | 100 |
| Test_Time_Learning | icl_banking77 | 1 | 272 | 272 | 100 |
| Test_Time_Learning | icl_clinic150 | 1 | 39 | 39 | 100 |
| Test_Time_Learning | icl_nlu | 1 | 333 | 333 | 100 |
| Test_Time_Learning | icl_trec_coarse | 1 | 270 | 270 | 100 |
| Test_Time_Learning | icl_trec_fine | 1 | 263 | 263 | 100 |
| Long_Range | detective_qa | 10 | 257–306 | 3271 | 6/ctx，共 71 |
| Accurate_Retrieval | eventqa_65536 | 5 | 170–173 | 859 | 100/ctx，共 500 |
| Accurate_Retrieval | eventqa_131072 | 5 | 339–346 | 1716 | 100/ctx，共 500 |

按 0.42 秒/chunk 注入 + 3.59 秒/query：

| 子集（samples=1, cap 20） | 注入 | query | 合计 |
|---|---|---|---|
| factconsolidation_sh_6k / mh_6k | 7 s | 72 s | **1.3 min** |
| factconsolidation_sh/mh_32k | 35 s | 72 s | **1.8 min** |
| factconsolidation_sh_64k | 71 s | 72 s | **2.4 min** |
| factconsolidation_sh_262k | 289 s | 72 s | **6.0 min** |
| icl_banking77 / nlu / trec_* | 110–140 s | 72 s | **3.0–3.5 min** |
| icl_clinic150 | 16 s | 72 s | **1.5 min** |
| detective_qa | 108 s | 22 s | **2.2 min** |
| eventqa_65536 | 73 s | 72 s | **2.4 min** |
| eventqa_131072 | 144 s | 72 s | **3.6 min** |

**整个 Tier 1 + Tier 2 在 0.42 秒/chunk 下总计不到 45 分钟。** 唯一固定成本是每进程 234.6 秒的模型加载——所以应按"每个进程多跑几个任务"来组织，而不是一任务一进程。

## 七、已知问题（读数前必看）

**1. `drop_memory` 未设种子，结果不可复现。**
`modeling_memoryllm.py:1578` 用 `torch.randperm` 随机淘汰。池容量 12800 token = 50 个 block，context 超过约 6k token 后池饱和，此后留下的是随机幸存者。bench 的 `seed: 42` 只用于数据采样（`eval_data_utils.py:288` 的 `random.seed`），**不控制 torch**。同一配置两次运行结果不同。

**2. 大 context 子集测的不是长程记忆。**
池满即随机淘汰，所以 32k/64k/262k 与 eventqa_131072 测的是"随机保留 + 生成"，不是完整上下文回忆。这是 MemoryLLM 设计使然，但报告结论时必须说明。

**3. `--max_test_queries_ablation` 是全局累计上限，不是每 context 上限。**
`main.py:60` 的 `has_reached_query_limit` 比较跨 context 累计的 `query_index`。设为 20 时，eventqa(5 context) 和 detective_qa(10 context) 只会用到第 1 个 context。要覆盖多 context 只能设 0 并靠 `max_test_samples` 控制。

**4. agent save 目录跨运行持久。**
路径 `./agents/{agent_name}_{sub_dataset}/exp_{i}`。`initialize_and_memorize_agent` 在目录存在时走 `load_agent()`，而 MemoryLLM 的 `load_agent` 只是打印"not implemented"并返回——**记忆池静默为空**。`run_sampling.sh` 会先删该目录。手动跑时必须自己清。

**5. `pad_token_id == eos_token_id == 128001`。**
transformers 会警告 attention mask 无法推断。当前贪心解码下未观察到异常，但批量推理时需注意。

**6. 查询模板复用 RAG 措辞。**
`utils/templates.py` 中 `'memoryllm': 'rag_agent'`。MemoryLLM 无检索器，措辞里的"检索到的文档"与机制不完全对应。这是有意的对齐选择，但与其他 baseline 的 prompt 不完全等同。

**7. 未安装 `pytorch_model.bin`。**
权重只有 safetensors。不影响的加载路径。

## 八、启动命令

单任务：

```bash
cd /share/latent/MemoryAgentBench
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=6 TOKENIZERS_PARALLELISM=false \
  /share/latent/mllm_env/bin/python main.py \
  --agent_config configs/agent_conf/MemoryLLM/memoryllm_8b.yaml \
  --dataset_config <data_conf.yaml> \
  --chunk_size_ablation 400 \
  --max_test_queries_ablation 20 \
  --force
```

`--force` 必须带，否则会加载旧结果。

`HF_HUB_OFFLINE=1` 必须带：`hub/datasets--ai-hyz--MemoryAgentBench/` 下只有 `refs/main`，没有 `snapshots/`。数据实际由 `_load_from_local_parquet` 从 `data/*.parquet` 读取。

## 九、待你定夺

`drop_memory` 的随机淘汰使 baseline 不可复现。若要与其它方法公平对比，建议在 adapter 初始化中加 `torch.manual_seed`——但这会把淘汰模式固定成一条特定轨迹，也是一种偏差。当前**未添加**，保持与官方代码一致。
