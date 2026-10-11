# S1d stage 2 protected oracle

- **状态**：代码与协议已冻结；GPU 运行在原实验节点执行，结果以独立校验器确认后更新。
- **用途**：开发阶段诊断对照，不参与论文方法结果，不作为 held-out 或泛化证据。
- **窗口与基**：固定 `(3,17510)`；沿用 `runs/s1d_selfsup_rows_v2.pt` 的 per-window selfsup SVD-13 basis；不使用 QA 重建 basis。
- **可训练参数**：仅 13 个 `alpha`；reader、carrier、basis 冻结；`D_new = D0 + reshape(U_selfsup @ alpha)`。
- **恢复监督**：`rescue_d0_wrong` 的真实 QA 第一答案别名 answer-token CE，prompt token 全部 mask。
- **保护监督**：所有 D0 原本答对的题。先用 D0 计算固定参考 CE `ce_d0(q)`，每步加入
  `mean_q ReLU(ce_current(q) - ce_d0(q))`。
- **固定权重**：`lambda_protect = 1.0`，只做一次诊断，不扫描权重。
- **优化**：AdamW，lr=0.003，weight decay=0.01，60 步，每 10 步测量，位移预算 1.0，seed=0。
- **判读**：仍以 TF `new_recovery / d0_harmed / net_vs_d0` 为主；GEN 只记录，不作为决定性指标。与原 oracle 的唯一实验变量是保护 CE 惩罚。

## 运行状态

当前工作站没有历史 `/share/latent/mllm_env` 且没有 CUDA；驱动已同步到原 `v100-2` 节点的 `/share/latent/PRLM-C`，使用 GPU1 执行。报告生成后运行：

```bash
python3 tools/test_s1d_stage2_dev.py
python3 tools/verify_s1d_stage2_oracle.py
python3 tools/verify_s1d_stage2_oracle_protected.py
```

目标报告：`runs/s1d_stage2_oracle_protected.w17510.json`。

## 显存修复与重跑

32GB V100 上原训练路径在第一次 backward 达到约 31.6GiB 后 OOM；未发现该卡被其他 GPU 进程抢占。保留图的 QA 路径改为逐题重建候选池与梯度累加，并使用 `torch.autograd.graph.save_on_cpu(pin_memory=True)` 卸载保存张量。候选 memory 与 active mask 上下文覆盖完整 backward；保护权重、cohort 均值损失、优化器与 13D basis 保持原定义。

本次验证恢复 `qa_max_tokens=256`，cohort 为 rescue=43、rescue_d0_wrong=38、全部 D0-correct protection=79（初始日志的 protect=60 仅指 retained 保护子集）。128-token 的失败尝试不作为可比较结果。单步 smoke 已完成全部 117 道训练题的 backward，报告 status=complete，loss=3.7742887547，alpha norm=0.0108166523，TF recovery/harm/net=0/0/0；full/select/D0 基线与原 oracle 逐项一致，basis 未变。正式实验执行 60 步，GEN 使用 `--skip-generation` 跳过，最终仅判读 TF 指标；单步结果不是最终实验结论。

上述 smoke 证明显存卸载可完成完整训练步。后续独立代码复查发现 QA 内外层重复使用非可重入的 memory 上下文；正式运行在 step 2 早期停止并归档，去除内层重复绑定后从 step 1 重跑。新增直接调用真实 `train_answer_arm` 的小模型回归，验证多题固定权重均值梯度、hinge 开关、候选池梯度路由、checkpoint 重算读 memory、basis 保持与异常恢复，测试 `PASS 10/10`。早期中断报告不作为最终实验结果。

## 多卡执行

按用户要求切换为 8 卡题目分片并行，物理设备 `0,1,2,3,4,6,8,9`，保留 GPU5/7 上已有服务。每个 rank 复制同一个冻结 reader 与初始记忆，启动时核对完整 memory 和 basis 的 SHA256。TF 基线、D0 参考 CE、训练题与评估题按 round-robin 分片；评估 gather 后检查题集无遗漏与重复。

每个 rank 的 rescue CE 与保护 hinge 均使用全局 cohort 数目归一化，单步将 13 个 alpha 的局部梯度以 Gloo `SUM` 汇总，然后各 rank 做相同 AdamW 更新并严格核对 alpha 一致。只有 rank0 写最终报告。此配置是一个实验的分布式加速，而不是 8 次独立训练。CPU 四进程不均匀/空分片回归、真实训练函数与启动器状态认证全部通过 `PASS 12/12`。GPU 多卡单步验证已成功并放行正式 60 步。

```bash
GLOO_SOCKET_IFNAME=lo python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 tools/s1d_stage2_dev.py --window 3,17510 --arms oracle_13d_protected --cuda-devices 0,1,2,3,4,6,8,9 --skip-generation --qa-max-tokens 256 --out runs/s1d_stage2_oracle_protected.w17510.json
```

生产启动使用 nohup，标准输出和错误直接重定向到远端日志文件；不使用 SSH stdout/tee 管道承载进程生命周期。`tools/run_s1d_parallel.py` 负责等待缓存、执行单步 smoke、严格核对 full/select/D0 基线与 cohort 后自动启动 60 步，最终以 torchrun 的退出码认证全部 worker 成功；非零退出会把残留 running/complete 报告标成 failed，rank0 在 complete 前还有全 rank barrier。

原始 37GB FP32 safetensors 位于 GPFS，8 路并发 mmap 加载首个 shard 达 250s，因而通过 `tools/stage_s1d_reader_cache.py` 一次性生成 `/dev/shm/s1d_reader_fp16` 的共享 FP16 缓存。每个浮点张量按驱动原本的 torch.float16 加载精度转换，并对保存后张量逐项重算 SHA256，保留原 tokenizer/config/index。缓存仅改变加载位置与存储精度，reader 实际 FP16 权重不变；多卡 smoke 必须复现原基线/cohort 才放行。完整测试含启动器状态认证，`PASS 12/12`。

后台流程 PID=45274；共享缓存已经生成并逐张量校验，约19GB。8卡 smoke 全部 worker 成功退出（torchrun exit=0，launcher all_workers_succeeded=true），报告 `runs/s1d_stage2_oracle_protected.w17510.parallel_smoke.json` 已复制本地。与单卡 smoke 的 full/select/D0 基线、cohort、逐题 TF 图、loss=3.7742887547141626、norm=0.010816652327775955 全部一致，alpha 最大差异2.3283064365386963e-10为FP32梯度求和顺序舍入，basis 未变，step1 recovery/harm/net=0/0/0。

启动器自动进入正式60步，torchrun PID=46728。日志分别为 `runs/s1d_stage2_oracle_protected.parallel8_launcher.log`、`runs/s1d_stage2_oracle_protected.parallel8.parallel_smoke.log`、`runs/s1d_stage2_oracle_protected.parallel8.log`。正式结果尚未完成，最终报告需通过独立 verifier 与全worker退出码认证。

独立 verifier 进一步核对原始 checkpoint/selfsup_rows/gold_rows 哈希、原 oracle basis SHA256、共享 FP16 缓存 manifest、AdamW/weight decay、D0 参考 CE 的有限性、逐题 accuracy 与固定lambda=1的损失分解。新增8类篡改检测（状态、worker、accuracy、basis、参考CE、loss、输入artifact、缓存），完整测试 `PASS 13/13`。测试用合成 history 仅在内存构造以验证检测器，不保存为实验报告；训练驱动未改动，正式报告 driver SHA 保持不变。

## 中途测量（非最终结果）

step10 快照 `runs/s1d_stage2_oracle_protected.parallel8.step10_snapshot.json` 保持 status=running。已独立重算逐题 recovery/harm/net，并核对原始基线/cohort、basis 未变、alpha norm 与预算、sup_loss=rescue_loss+protect_penalty。

| step | rescue CE | protect hinge | total loss | alpha norm | new recovery | D0 harmed | net |
|---|---|---|---|---|---|---|---|
| 10 | 3.6297352855 | 0.0022770714 | 3.6320123569 | 0.0986444950 | 0 | 1 | -1 |

缓存证明 `runs/s1d_reader_fp16_manifest.json` 已复制本地，SHA256 与多卡 smoke 报告完全一致，涵盖8个 shard 的1063个张量。正式60步完成与全worker认证之前，不据中途指标作最终裁决。

## 解释边界

保护项若降低 harm，说明此前损伤确实可由目标函数中的 no-harm 约束部分缓解；只有在 recovery 仍保持且 net 改善时，才能支持“目标错配是主要瓶颈”的解释。若保护项仅牺牲 recovery 而没有正 net，说明固定 13D 基内的恢复/保持权衡仍然是主要限制。无论结果如何，这仍是单窗口、固定基、有限步诊断，不构成方法主结果或数学上界。
