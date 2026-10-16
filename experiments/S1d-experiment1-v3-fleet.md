# PRLM-C 实验 1：实现与并行运行

## 代码和环境

服务器为 `v100-1` 和 `v100-2`，共享工作目录 `/share/latent/PRLM-C`。
Python：`/share/latent/mllm_env/bin/python`。
Reader：`/dev/shm/s1d_reader_fp16`，需存在 `fp16_cache_manifest.json`。
原正式脚本 `s1d_stage2_dev.py` 保持原样。

新增文件：

- `tools/s1d_experiment1.py`：A/B/C/D 驱动。
- `tools/run_s1d_experiment1_parallel.py`：八作业并行、一步 smoke、严格验证、60 步正式运行。
- `tools/run_s1d_experiment1_fleet.py`：两台服务器共享队列，按可用显存自动分配任务；本批次使用此入口。
- `tools/verify_s1d_experiment1.py`：独立严格校验入口，可复验 smoke 或正式报告。
- `tools/status_s1d_exp1.py`：当前 fleet 状态汇总，可指定批次前缀。
- `tools/test_s1d_experiment1.py`：17 项回归与梯度、状态恢复、事实解析、真实 tokenizer 和缓存保护集验证。

## 四组定义

| arm | basis | 训练损失 |
|---|---|---|
| A | 原窗口 selfsup SVD 13D | 原尾 4 token CE |
| B | 原窗口 selfsup SVD 13D | 新事实答案 CE + 保留事实 hinge |
| C | 受损事实答案梯度 SVD 13D | 原尾 4 token CE |
| D | 受损事实答案梯度 SVD 13D | 新事实答案 CE + 保留事实 hinge |

## 冻结协议

- 开发窗口 `(3,16370)` 和 `(3,17510)`；不触碰正式窗口。
- reader 冻结，basis 冻结，只有 13 个 alpha 可训练。
- AdamW，lr=0.003，weight_decay=0.01，60 步，每 10 步测量。
- 残差 L2 范数上限 1.0，seed=0。
- QA token 上限 256，TF-only；QA 只用于评估。
- 训练事实总数上限 40。A/C 复用原 dropped-first、最新版本规则和尾部 CE；B/D 最多 20 条恢复事实，剩余预算分配给保留事实。
- 原尾部损失保留原实现的求和梯度，日志显示平均 CE。新损失为恢复事实平均答案 CE + lambda=1.0 的保留事实平均 hinge。两者的梯度尺度区别属于训练目标变化，报告中须披露。

## 新事实目标与建基

1. 直接解析写入时原始事实，复用 `first_match` 和最新 owner 去重。
2. 用原句的主体与关系作为 prefix，仅监督最终答案字段；真实 tokenizer offset 验证答案边界。
3. 从 dropped 块中挑选 Full 答案 CE <= 2.0，且 D0−Full 答案 CE >= 0.10 的事实。
4. 按 CE 恶化降序、key 稳定排序，最多 40 条用于建基。
5. C/D 在 D0 处求事实答案 CE 对 down.weight 的梯度，再取 SVD 前 13 维。rank 不足 13 时明确失败，不混入旧梯度或随机方向。
6. B/D 用上述集合前最多 20 条作为恢复训练；从 kept 块选 D0 CE <= 2.0 的最新事实作为保留约束。
7. 保留损失为 `mean ReLU(current_fact_answer_CE − D0_fact_answer_CE)`，权重固定 1.0。

这些阈值是本轮事先固定的工程配置；没有根据评测 QA 调参或选样本。

旧损失路径同步修正 checkpoint 重算的候选记忆绑定，保持候选 pool 和 active mask 到 backward 完成。每个事实重建图并 offload saved tensors，避免保留整个 40 条事实图。新旧路径均有小型 reader 的梯度等价测试。

## 本次 GPU 分配

2026-10-12 12:34（新加坡时间）确认的单步 smoke 分配：

| 窗口 | A | B | C | D |
|---|---|---|---|---|
| `(3,16370)` | v100-1 GPU9 | v100-1 GPU5 | v100-1 GPU3 | v100-2 GPU9 |
| `(3,17510)` | 排队 | v100-1 GPU8 | v100-1 GPU4 | v100-1 GPU2 |

7 个作业已完成模型加载，正在计算，尚无正式结果。v100-2 的 GPU5 和 GPU7 不纳入调度；v100-1 按实际空闲显存调度。每张卡最多一个本批次 reader，要求启动时空闲显存至少 30000 MiB。现有其他任务保留。第 8 组以及正式任务自动使用满足门槛的空闲卡，后续分配以 fleet 状态文件为准。

## 启动与验证

```bash
cd /share/latent/PRLM-C
PYTHONPATH=. /share/latent/mllm_env/bin/python tools/test_s1d_experiment1.py
ssh v100-2
cd /share/latent/PRLM-C
/share/latent/mllm_env/bin/python tools/status_s1d_exp1.py --logs
```

本批次已启动，无需再次启动。两个 nohup 调度器 PID 分别为 v100-2 的 `65495`、v100-1 的 `2906241`。共享队列首先运行 8 个单步 smoke。8 个全部成功并通过校验后，自动开启 8 个 60 步正式实验。只有完成状态、非零 alpha、冻结 basis、固定 cohort、完全一致的 Full/select/D0 基线，以及 A/B basis、C/D basis 和 B/C/D 事实集合一致时，才会出具验证证书。任何不一致会阻止正式实验启动。

实际任务使用 nohup 运行，SSH 断开后仍继续。

## 日志和产物

- 调度器日志：`runs/s1d_exp1_fact_v3.{v100-1,v100-2}.fleet.log`。
- 持久状态和所有子进程 PID：`runs/s1d_exp1_fact_v3.fleet.json`。
- smoke：`runs/s1d_exp1_fact_v3.smoke.{A,B,C,D}.w{16370,17510}.{json,log}`。
- 正式：`runs/s1d_exp1_fact_v3.{A,B,C,D}.w{16370,17510}.{json,log}`。
- smoke 校验证书：`runs/s1d_exp1_fact_v3.smoke.verification.json`。
- 正式校验证书：`runs/s1d_exp1_fact_v3.verification.json`。
- alpha 和显式 basis：对应 `.alpha.pt`。
- C/D 的原始事实梯度及筛选 CE：对应 `.fact_rows.pt`。
- 回归日志：`runs/s1d_exp1_fact_v3.cpu_tests.log`（PASS 17/17）。
- v100-1 模型缓存校验证书：`runs/s1d_v1001_reader_cache_verified.json`，16 个文件全部 SHA256 匹配，共 19,576,750,707 字节。

每个测量点报告恢复 R、损伤 H、净收益 R−H、全题 TF 正确数、alpha 范数、loss、运行时间、峰值 GPU 显存；保存代码和 checkpoint 哈希与显式 basis 存储字节数。

## 结果判读

首先检验同窗口四组的 cohort、基线和预算一致。工程推进门槛：弱窗口净收益至少 +5；强窗口至少 +7；全题正确数不下降。该门槛不是统计显著性标准。正式结果完成前，不宣称新方法有效；也不自动启动实验 2 或官方基准。

## 当前批次

批次前缀 `s1d_exp1_fact_v3`。修复了共享事实缓存读取时误把保护集限于 dropped 事实的问题：现在使用完整 Full/D0 CE 映射，恢复事实仅来自 dropped 块，保护事实仅来自 kept 块。B/C/D 共用一致的事实选择；C/D 共用原始梯度缓存形成相同 basis。保护集来源、key/record 对齐、checkpoint 和梯度维度均严格校验。

本地 `outputs/` 保存了本批次脚本副本。服务器目录当前没有 `.git`，本批次修改未声称已提交或推送。

可用以下命令查看状态：

```bash
ssh v100-2 '/share/latent/mllm_env/bin/python /share/latent/PRLM-C/tools/status_s1d_exp1.py --prefix s1d_exp1_fact_v3 --logs'
```
