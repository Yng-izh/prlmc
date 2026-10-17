# Shared E6d：同预算基线、独立共享训练与冻结后验证

本轮以共享 `CarrierResidualCompactor` 为候选，不继续逐 episode C13。代码已实现；本文不代表新的 GPU 实验已经运行，也不代表封存测试已开启。

## 1. 固定的开发基线

八个开发 ID：`0,2,4,8,14,15,17,18`，每个64题。每个 episode 只写入一次，五个 arm 从同一源记忆构造，并逐题恢复生成可能修改的记忆。

| Arm | 行为 | 有效槽位 |
|---|---|---:|
| Full | anchor + 全部19个写入块，未压缩参考 | 5376 |
| Selection | E6d相同的原生carrier选择 | 2816 |
| SharedE6d | 同一冻结checkpoint，rank32，gamma_max=.05 | 2816 |
| Recent9 | 最近9块，调用已有recent_block_compact | 2816 |
| Random9 | seed0抽取9块，各层共享索引，按原时间顺序放回 | 2816 |

压缩组预算包含512个anchor槽；`19→9`指工作块。Random9只做一次固定抽样，不选择好看的种子。固定B2、换行stop、32个新token、官方fact-consolidation评分。冻结参数状态哈希、源记忆哈希在生成前后核对。每个arm记录压缩时间和总时间、逐题预测、恢复和损伤集合；checkpoint与模块状态大小另报，不将活跃槽预算等同全部存储成本。

```bash
cd /share/latent/PRLM-C
/share/latent/mllm_env/bin/python tools/run_s1d_shared_compactor_eval.py \
  --devices 5,9 --checkpoint runs/e6d_interleave_ckpt.pt \
  --arms Full,Selection,SharedE6d,Recent9,Random9 --random-seed 0 \
  --gamma-max .05 --prefix runs/s1d_shared_baselines_old_v2
```

卡号为示例；launcher要求指定卡至少30000MiB空闲且使用量不超过256MiB，不抢占已有任务。1–8卡队列调度，每卡一个worker。正式五组重评保证同一代码版本下比较；没有将旧归档的分数直接拼入新实验。

## 2. 新训练数据与监督

`prepare_s1d_shared_training.py`从原始Conflict_Resolution构造默认8个新训练episode，排除已保留24个episode全部实体、E6d历史8个训练窗口实体，以及原生成器排除的历史实体。检查封存数据的结构与实体集合不生成测试预测、不创建开封ledger。若剩余实体不足，报告`insufficient_data`，不自动缩减数据或复用开发/测试实体。

保持410行（30anchor+380写入）、19个工作块、版本冲突与relation均衡规则。每个新训练episode预先取discovery的前20个relation均衡事实，固定为最新写入事实的训练QA，保存为`training_probes`；训练文件没有development/test评测分区。**这是有训练QA答案监督的离线共享训练**，不是无需答案监督的方法。监督选样不读取开发QA正确性，也不按支持块位置/Full是否正确过滤。这与历史E6d以丢弃块事实筛选probe的数据选样有区别，应在比较中准确说明。

```bash
/share/latent/mllm_env/bin/python tools/prepare_s1d_shared_training.py \
  --dataset /share/latent/MemoryAgentBench/data/Conflict_Resolution-00000-of-00001.parquet \
  --templates tools/e6a_overfit.py \
  --reserved-plan runs/sic_paper_controlled_plan.json \
  --tokenizer /dev/shm/s1d_reader_fp16 --episodes 8 --seed 20261014 \
  --out runs/s1d_shared_training_plan_v2.json
```

写入tokenizer的文件哈希与512-token无截断审计写入计划。训练加载时重新核对计划哈希、实体隔离和tokenizer；训练prompt+答案超过256token即失败。

## 3. 一次共享训练优化、两个固定种子

沿用现有rank32结构、gamma_max=.05、19→9预算与`probes._stage_weights`：

| 阶段 | Gold | KD | Reader hidden | Carrier | Stats |
|---|---:|---:|---:|---:|---:|
| A |0|1|.1|.1|.02|
| B |1|.5|.1|.05|.02|

默认从原有架构的零up初始化训练新共享模块，不加载旧E6d继续拟合；旧E6d保留为对照。种子0、1控制初始化，数据、episode顺序与每个episode的预算相同。每个episode A4步；B每次4步、轮转24轮，每个episode共96步；默认总800步。一个种子在全部场景和两个阶段只创建一个AdamW（lr=.001，weight_decay=.01），优化器状态不重置。reader冻结，只有compactor训练。teacher缓存key包含训练计划、源池与probe哈希，重建源池必须逐位一致。OOM显式失败，不自动换预算或重启部分步骤。

```bash
# GPU smoke须单独前缀，仅运行首个事件1步；不输出可用于正式评测的checkpoint。
/share/latent/mllm_env/bin/python tools/run_s1d_shared_compactor_train.py \
  --devices 5,9 --training-plan runs/s1d_shared_training_plan_v2.json \
  --prefix runs/s1d_shared_training_smoke_v2 --smoke

# 正式训练：两卡并行两个种子；单卡时依次运行。
/share/latent/mllm_env/bin/python tools/run_s1d_shared_compactor_train.py \
  --devices 5,9 --training-plan runs/s1d_shared_training_plan_v2.json \
  --prefix runs/s1d_shared_training_v2
```

只保存最终scheduled checkpoint；不根据开发分数挑中间步。两个种子都完整保留并汇报。新checkpoint分别用阶段1的五组入口评测，`--checkpoint`换为`.seed0.pt`/`.seed1.pt`并使用两个不同前缀。种子0作为预设主要新候选、种子1检验重复性；不将两者中较高的单个分数包装为稳定效果。若新训练没有优于旧E6d，保留旧checkpoint，不继续加模块。

## 4. 封存测试门槛与一轮评测

`freeze_s1d_shared_method.py`要求旧E6d、新seed0、新seed1均有完整8-episode五组评测结果，并重新评分所有原始预测和校验provenance。生成包含具体chosen checkpoint、源代码、模型、数据、有效gamma和协议的冻结文件。这个操作本身不开封测试。

```bash
/share/latent/mllm_env/bin/python tools/freeze_s1d_shared_method.py \
  --checkpoint runs/e6d_interleave_ckpt.pt \
  --development-prefixes runs/s1d_shared_baselines_old_v2,runs/s1d_shared_baselines_seed0_v2,runs/s1d_shared_baselines_seed1_v2 \
  --decision '根据完整开发对照选择旧E6d；此处填写实际裁决依据' \
  --out runs/s1d_shared_method_freeze_v2.json

# 仅在方法已经选定、正式决定开封后执行：完整16个episode，不能挑子集。
/share/latent/mllm_env/bin/python tools/run_s1d_shared_compactor_eval.py \
  --devices 5,9 --stage test --freeze runs/s1d_shared_method_freeze_v2.json \
  --checkpoint runs/e6d_interleave_ckpt.pt --gamma-max .05 \
  --arms Full,Selection,SharedE6d,Recent9,Random9 --random-seed 0 \
  --prefix runs/s1d_shared_sealed_v2
```

launcher通过已有`claim_sealed_test`原子创建单次开封ledger；worker核对相同冻结清单，拒绝直接绕过launcher启动。已经开封后拒绝再次挑选方法。汇总不静默丢掉失败episode；存在失败就标`incomplete`并不计算完整集CI。置信区间按episode bootstrap，固定10000次、seed20261012。

## 5. 自然EventQA冻结入口

`s1d_shared_natural_eval.py`支持直接读取原始parquet完整context与全部QA；`run_s1d_shared_natural_eval.py`可将预先指定、context互不重复的row在1–8卡排队。全部使用同一checkpoint，不训练，不选择checkpoint。五组从同一完整写入池构造，每个压缩组仍2816槽；整个工作记忆一次压缩为9块。输入工作块数量可能不同于训练的19块，报告会明确记录，属于额外形状迁移，不声称19→9完全相同。

阶段4创建freeze时必须同时传入`--natural-dataset`、`--natural-prompt-file`、`--natural-rows`，锁定自然评测资产。自然launcher核对与freeze完全相同的dataset/prompt/完整row清单、checkpoint、模型与源码，拒绝评测后再改prompt或挑row。

自然query模板必须预先存成文件，入口要求显式`--prompt-file`并记录内容与哈希；它不能直接继承冲突任务要求“最新版本”的B2模板。长度固定32、新行stop，不做prompt扫描。**EventQA主指标为官方`eventqa_recall`：全部required answer elements出现在预测中才算成功**；评分直接使用归档官方源码的recall表达式，并保存源码哈希。为兼容现有paired helper，JSON内`em`字段在该入口表示二值EventQA recall，报告明确标记，不把其称为fact-consolidation EM。

```bash
/share/latent/mllm_env/bin/python tools/run_s1d_shared_natural_eval.py \
  --dataset /path/to/eventqa.parquet --rows 0,1 \
  --checkpoint runs/e6d_interleave_ckpt.pt \
  --prompt-file /path/to/frozen_eventqa_prompt.txt \
  --freeze runs/s1d_shared_method_freeze_v2.json \
  --devices 5,9 --prefix runs/s1d_shared_eventqa_v2
```

上面的dataset、row、prompt为待锁定实际资产，不是已验证存在的实验输入。完整context按512token无损切分；超出48个工作块时标`capacity_exceeded`并返回非零，少于10工作块也明确不适用，**不裁剪context、不把小窗口当完整自然任务**。如实际EventQA长context均超出容量，必须另行制定共享模块的连续压缩协议，然后冻结后运行；本入口不自动更改写入或压缩生命周期。因此尚不能声称已完成整个MemoryAgentBench EventQA接线/复现。

## 6. 验证与输出边界

```bash
PYTHONPATH=tools /share/latent/mllm_env/bin/python -m unittest \
  test_s1d_shared_training test_s1d_shared_compactor_eval test_sic_paper_data \
  test_s1d_controlled_contract -v
```

CPU验证覆盖真实candidate池复制与mask、共享优化器更新、reader冻结/源池恢复、训练与保留实体隔离、预算不足失败、单次开封ledger、完整context容量与EventQA多答案评分。CPU测试不替代V100真实训练/生成smoke。本次代码工作不启动GPU训练/生成、不生成16个封存episode预测，也不开展已暂停的原生冷启动对照。

2026-10-13验证记录：本地和v100-2的上述三个unittest测试文件均37项通过；`test_s1d_controlled_contract.py`直接执行的单次开封与bootstrap两项检查通过。脚本已同步到`/share/latent/PRLM-C/tools/`；远端存在历史顶层工具副本，新增入口显式优先加载维护中的`tools/`模块。归档官方EventQA评分源已同步到原有报告路径。真实模型GPU smoke、训练数据容量/分词审计与自然任务资产锁定尚未执行。

现有数据支持同模板跨实体受控迁移；未完成自然完整context对照和连续压缩前，不扩大到跨自然任务或长时间记忆保持结论。NativeDefault/NativeBudget仍是管线级参考，不能将其差值归因于E6d模块。

## 7. 2026-10-13 实际启动记录

**本节为历史启动记录，当前已取消。** 用户随后明确停止自建数据实验；调度器72776与两项smoke79020/79021已停止，fleet/queue标记cancelled_by_user。下述自动后续不再执行。新的主协议见 `Public-retention-frozen-SharedE6d.md`。

用户要求在空闲GPU启动后，旧E6d五组开发评测已在v100-2物理卡`0,2,3,4`运行，前缀`runs/s1d_shared_baselines_old_v2`。首波episode0/2/4/8，次波14/15/17/18；实时状态与worker PID在`.fleet.json`，launcher与逐episode日志使用相同前缀。

独立训练数据`runs/s1d_shared_training_plan_v2.json`已完成容量、哈希、实体隔离与tokenizer审计。排除5661个实体后剩2175个候选，8个训练episode需要1264个实体，容量充足。这项准备没有评分封存测试。

初次与再次训练launcher因显存预检拒绝启动，未创建训练worker。物理卡5/7/9当时均被其他任务占用，不应称为当前空闲卡。随后启动`wait_s1d_shared_training.py`等待调度器，PID72776，状态`runs/s1d_shared_training_v2.queue.json`、日志`.queue.log`。该队列每30秒检查空卡（free>=30000MiB、used<=256MiB），避开仍在跑基线fleet的卡。找到卡后为seed0/1分别执行1步GPU smoke→正式800步训练→冻结五组8个开发episode评测；两卡空闲时并行两个种子。无可用卡时等待，不杀掉其他进程。smoke/训练失败明确记录，不自动改预算。

启动核验时四个基线worker存活，Full、Selection、SharedE6d已完成首波的逐题生成，正在继续Recent9/Random9；两个共享训练种子仍等待可用卡。封存测试与自然EventQA未启动。
