# S1d 方法线与生成应用线并行协议

2026-10-12。用户新安排取代原“弱窗口通过再串行强窗口”调度，训练算法完全不变。两条独立目标：方法线只增加原C的风险保护；应用线只加载既有C最终checkpoint生成，不依据应用得分调整方法、prompt或attention。

## 四任务与资源

| 任务 | 实验 | 初始资源 | 输出 |
|---|---|---|---|
| M1 | C+RiskProtect弱 `(3,17510)`，60步 | v100-2 GPU0，保留原PID76584 | `runs/s1d_c_risk_v1.weak.json` |
| M2 | C+RiskProtect强 `(3,16370)`，60步 | v100-1 GPU5 | `runs/s1d_two_lines_v2.M2.json` |
| A1 | 原C强窗口，Full/Selection/D0/C生成 | v100-1 GPU6 | `runs/s1d_two_lines_v2.A1.json` |
| A2 | 原C弱窗口，Full/Selection/D0/C生成 | v100-1 GPU8 | `runs/s1d_two_lines_v2.A2.json` |

任务编号不指定物理卡。新任务独立smoke后自动进入正式阶段；显存不足排队，不改变内容，不中断其他用户任务。v100-2 GPU5/7排除。v100-1缓存1063个tensor重新核对原FP16证书，两台manifest SHA256一致；日志保留证据。

M1原smoke已通过并在训练，不重训。原串行派发器PID76074被终止，仅保留训练子进程，并用监控器接管；因此M1完成依赖完整报告、二进制basis和一致性证书，监控器不会伪造原进程退出码（字段null）。新M2/A1/A2的真实退出码由新父启动器直接wait取得。原launcher标记`superseded_parallel`，其旧强窗口条件不再用于派发。

## 方法线

沿用 [C风险保护v1](S1d-C-risk-protection-v1.md) 全部训练设置：原C受损事实梯度SVD13、原40行尾4-token求和梯度，冻结reader/basis/其余参数，仅优化13alpha。保护候选为写入kept latest-owner事实，有限非负D0答案CE≤2.0，按CE降序/key稳定排序前20；固定lambda1、mean ReLU(current CE−D0 CE)，无扫描。AdamW .003/.01，L2预算1，seed0，60步/测量10，QA256，TF-only。

推进标准仅看step60。弱窗口net≥5并减少原5道损伤（H≤4）；强窗口按用户最新目标net≥7，H单独报告与原C的1道比较，不设旧net≥9/H≤1为当前强窗口门槛。新证书同时保留旧严格标准与最新门槛，历史训练报告内旧协议不回改；这只是预先变更裁决标准，不改变loss、lambda或梯度。

原C弱R6/H5/net1，强R10/H1/net9。R只计预定义rescue_d0_wrong中新恢复；H计全部D0-correct损失。弱不通过就停止扩展此保护机制；已并行启动的强窗口完成用于检查其伤害，不能据此重新调lambda/叠加模块。门槛通过不等于统计显著或泛化成立。

## 生成应用线

独立入口 `tools/s1d_c_generate.py` 加载发布的 `s1d_exp1_fact_v3.C.w{16370,17510}.alpha.pt`；复用C窗口写入、carrier恢复、SVD13、池注入、active mask和TF测量。生成前对Full/Selection/D0逐题TF、C最终逐题TF、checkpoint window/carrier/训练driver/basis/alpha严格校验。只加载与测量，不运行optimizer step或backward、不挑checkpoint。smoke每臂前2题，只验证工程；正式每臂全题，265/276道。

固定复用 `readout_probe.FEWSHOT`（B2）和 `official_score`：greedy、max_new_tokens32、stop_strings换行、attention gain1。问题不拼接gold；Few-shot中三个既有演示保持不变，不扫描。Full是相同写入窗口的未压缩记忆，压缩三组使用相同active槽预算；不声称Full与压缩组具有相同容量。

每题后检查并恢复候选pool，包括`.data`写入造成的版本计数漏检；各臂后验证原pool和mask恢复，避免问题间生成写回污染配对。保存raw prediction、官方EM、substring EM、question/aliases、模板/模块/checkpoint/model证书哈希、上下文行/写入范围/池指纹。

报告Full正确Selection错误的generation lost；其D0-wrong子集的C恢复为R，全部D0-correct的C损伤为H，target net=R−H。同时单独报告全题C−D0和C−Selection的gained/harmed、delta EM；全题EM差不等于target net。记录TF恢复题在C生成中是否答对、是否相对D0新恢复，不把teacher-forced EM当生成成绩。

**范围限制：** 原S1d问题由窗口事实模板构造，不是原生官方多跳QA；使用官方计分规则的真实生成构成开发窗口应用可行性证据，不能写成完整原生MemoryAgentBench官方成绩。三个既有few-shot例子与语料共享背景，不能声称零开发关联。报告不会把原C生成收益直接归因于尚未生成评测的新RiskProtect。

## 接线与复现

为不修改正在运行且哈希锁定的M1代码，新增 `s1d_c_runtime.py` 是当前C驱动的冻结快照。其尾部/保护反向、建基、候选池、写入函数经AST一致性测试与原驱动完全一致，仅增加load-only generation分支。`s1d_c_risk_train.py` 只允许训练入口；`s1d_c_generate.py` 强制checkpoint和reference，进入生成分支后直接return，不触达训练步骤。

`s1d_c_generation.load_verified_c/evaluate_four_arms` 是可复用评测API，本身不限制row或窗口，接受合格参考报告、相同重建pool和冻结basis/alpha。当前CLI复现的是两个已有开发窗口；controlled episode写入/basis/训练适配需先在开发episode完成，不能拿这两个窗口的basis直接跨episode使用。

```bash
cd /share/latent/PRLM-C
PYTHONPATH=. /share/latent/mllm_env/bin/python tools/test_s1d_c_generation.py
# v100-1，分别smoke->formal，三个任务独立并发
PYTHONPATH=. /share/latent/mllm_env/bin/python tools/run_s1d_two_lines.py \
  --devices 5,6,8 --host-label v100-1 --prefix s1d_two_lines_v2
# v100-2，监控已接管的M1；首次迁移时--adopt-m1已执行，不能重复
PYTHONPATH=. /share/latent/mllm_env/bin/python tools/monitor_s1d_two_lines.py \
  --prefix s1d_two_lines_v2
```

启动器记录每个任务的smoke/formal报告、PID、命令、退出码、哈希和证书。代码冻结后拒绝覆盖旧报告或中途变更脚本。监控器等待四任务结束，自动生成 `runs/s1d_two_lines_v2.summary.json` 供一次联合裁决，不自动开启封存测试。独立`watch_s1d_generation_audit.py`在生成报告完成后直接从原MemoryAgentBench源码提取官方EM/parse函数重算每题，记录官方source哈希及独立证书；18个规则边界案例已验证当前readout scorer等价。

实际有效批次为 `s1d_two_lines_v2`：首次三个smoke在生成前发现active-slot预算检查漏算固定anchor，终止仅属于本次的smoke进程，修正后以新prefix重跑。v1日志和状态保留并标记superseded；M1正式训练未重启。压缩预算应是 `k_anchor + used − 19*block + 9*block`。最新启动命令用v2 prefix，不再执行`--adopt-m1`，避免重复终止旧派发器；最终汇总路径为 `runs/s1d_two_lines_v2.summary.json`。

## 独立验证准备

`prepare_s1d_controlled.py` 仅做数据结构/实体隔离/原计划和episode哈希/封存ledger检查，不加载模型、不生成预测，不创建test-opened ledger。计划为24个controlled episode（8开发16测试），保持原划分。结构验证会核对答案与最新事实的一致性，这不是模型评测或测试选型；不向输出打印测试题/答案。

后续先在8开发episode适配C的write-time建基与训练，冻结候选、训练/生成协议、代码与权重/data哈希；16测试一次性全量打开，禁止选子集、重选或回调。basis rank不足13明确记失败，不偷偷换basis；不能因某episode失败删除它。

封存测试主统计为episode级配对EM变化：C−D0和C−Selection，10000次episode bootstrap、seed20261012、95%CI；报告失败数，不把同pool的问题当独立样本。数据共享源语料/模板，因此只支持受控跨实体泛化；Conflict_Resolution实际四个自然context限制自然context泛化结论。未见episode的执行仍待本轮联合裁决与冻结，不在此次监控器自动派发。

`s1d_controlled_contract.py`实现冻结manifest匹配、原test-opened ledger检查、排他S1d single-open ledger和episode级bootstrap；CPU检查验证manifest不符时不开封、二次开封失败、置信区间可重复、失败episode不能静默删除。该接口本轮只以合成数据测试，没有调用真实测试开封。
