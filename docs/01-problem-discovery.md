# 01 — 问题发现线（E0–E7、SIC、Native-QA/Readout）

旧 MemoryAgentBench 线的问题定位记录。**原索引文件 EXPERIMENTS.md 与 BASELINE 冻结记录已归档至 archive/2026H1_legacy/root/EXPERIMENTS.md 与 archive/2026H1_legacy/root/BASELINE.md**。基线为未修改的 MemoryLLM-8B on MemoryAgentBench；本线实验主要使用 /share/latent/mllm_env（V100-32GB）。

> **正文缺失声明**：EXPERIMENTS.md 目录表引用的逐实验正文（experiments/E0.md、E1–E7、E6-gate、Native-QA、Readout-probe、SIC-O1/O2/P1、params.md、fixes.md、open-questions.md、reproduce.md 等）在仓库中不存在（2026-10-17 重组时确认），不重建。结论数字逐字摘自归档索引；原始证据见 runs/、reports/、outputs/。

## 线性读法（摘自归档索引，数字不改写）

E1 发现触发器不触发 → E2 标定阈值 → E3 修触发器后首次测 accuracy → **accuracy 全 0，根因未分离** → E4 定位到 compactor（**compactor 是充分原因**）→ E5a/E5b 原生 block 选择恢复 accuracy 至 **EM 26.00**（E5b 与 E5a 逐条相同，打分无增益）→ E5c 打开 active mask（EM **23.00**）→ E5d 放开 filler 限制排除开关混淆（EM **24.00**，与 E5c 逐条 97/100 相同）→ E6 carrier_residual（EM **24.00**，未训练仅验证接线）→ E6a 离线单窗口 overfit（220 步恢复 2/3）→ E6b-mini 8 训练 + 2 held-out（首轮度量缺陷作废后重跑：训练 7/30、held-out 2/16，Fisher p=0.4628）→ E6c 每窗 96 步（训练 5/30、held-out 3/16，retention reading 翻转为 overwrites——**schedule 才是约束，不是每窗口预算**）→ E6-gate 压缩门 **FAIL**（最佳 3/6 @ rank 256；恢复数随解码侧容量 0→64→256 单调上升 0→2→3，证伪「SVD 销毁信息」）→ E6d 顺序消融（interleave **56/160 净增 +30** vs forward +11 vs reverse +7；遗忘事件 6 vs 50 vs 42）→ E6e 凶手探针（预注册三假设全不成立；漂移压过信号的临界翻转）→ E7 设计定稿 → E7a 干扰图（梯度余弦 0.05–0.32，单步 harmed=0）→ E7a-24 深度扩展（harmed=5 跨 seed 逐条相同；rescue:harmed=31:5）→ E7a-sim（同 context 8 窗在 reader 表征带宽 0.9976–0.9991，与梯度/功能信号零相关，ARI<0——**停止门槛 1 触发，不做语义 router**）。

## SIC 线与读出诊断（结论）

- G0 版本盲区（mean 0.9751、min 0.9626，与随机对照 0.9734 重叠）→ G1 显著性选择（select 19 / sic 9 / oldest 18 / random 16，McNemar p=0.0245）→ G2 版本分层判字面 FAIL（0.688<2.0）→ G2b held-out 方向复现但 p=0.371 → **G2c 独立 200 题 NOT_PASSED**（128 eligible，select lost25 / sic29，p=0.424；singleton 净增益归因撤回）→ SIC-O1/O2 开发完成、确认未过（82 题 55→56，净 -1，p=1.0）→ SIC-P1 数据冻结 + smoke（ep0 候选 22/64 低于 native 23/64），开发暂停，**原 SIC-P1 测试集保持封存**。
- Native-QA：官方口径 row0 三臂 5/6/7 EM、row4 30/31/13 EM；压缩 vs 未压缩净差 ±1/100 为噪声级；warmup（EM 31/100 与未训练逐题相同，γ 仅用预算 5.2%）→ **读出/格式瓶颈 >> 压缩损失，停止压缩算子侧变体**。
- Readout-probe：**B2 few-shot+stop 唯一显著，EM 30→42（+15/−3，p=0.0075）**；无 stop 的 B 官方 EM 崩到 1；C attn×8 +11/−6 不显著；组合 BC 39<42。

## 证据边界（摘自归档索引）

历史 82 题已在 O2 开封。support oracle 最优值仅是指定文本支撑块最小遗漏，不是 reader 错误率下界。teacher-forced 答案 token 完全匹配不等于官方完整生成 EM；同一上下文多题不增加独立上下文数。历史训练态 LoRA dropout 随机性使其旧分数不作为本轮确定性可比对照。以上边界优先于旧条目的强归因表述。

## 证据位置

- outputs/prlmc-{smoke,obs,r1,e5a,e5b,e5c,e5d,e6,e6-gate}/、logs/、runs/。
- 完整索引原文：archive/2026H1_legacy/root/EXPERIMENTS.md。
