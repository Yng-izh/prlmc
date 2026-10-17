# 05 — Public Retention（冻结 SharedE6d 公开原始知识保持）

协议正文（活跃，原位）：experiments/Public-retention-frozen-SharedE6d.md。

## 研究问题

在公开原始任务上，同有效预算（20 native 块 = 5120 槽）下冻结的共享 CarrierResidualCompactor（SharedE6d）是否优于 Selection，以及冻结模块能否迁移到未参与训练的任务。主方法为训练后完全冻结的共享压缩器；人工 24 episode 与 45.31% 结果只作附录机制诊断，不进论文主表。2026-10-13 用户取消当前自建数据实验；C13 逐场景训练、C20–32、P13、RiskProtect、新架构与新共享模型训练暂停。

## 固定要点（详见协议正文）

- 数据：YuWangX/KnowledgeRetention @ c609067a（原始）与 KnowledgeRetentionProcessed @ 03a8db0（作者 8B 可回答性筛选）分别报告，不混合分数；仅按固定发布顺序选样。
- 两种输入模式不得合并：author512（作者预处理对齐，报告截断数量）与 lossless（默认，拼接等于源抽取段落）。
- 生成：每写一步 greedy 10 token，不用 B2、不加换行 stop；评分为作者 case-sensitive answer containment（明确不是 SQuAD official EM）。短文本不丢弃，<16-token 写入另作标记。
- 冻结 checkpoint SHA256：a9fe2ea3f7a4e2c21590cf2219591edb11fc65297ce36ddeed96c236d8020c5d（rank32、gamma_max 0.05）；共享参数前后状态哈希必须相同。
- 压缩规则与 v4 一致：达 20 块后将最旧 19 块压至 9 块、保留最新 1 块；Selection/Shared 用同一 carrier scorer 与写入重要性配置；零初始化 Residual 必须复现 Selection。
- 主指标：所有固定样本末步 SharedE6d−Selection 配对答案包含率；严格 knowledge rescue 只统计末步 Full 正确、Selection 错误、Shared 正确；95% bootstrap 按原 context 聚类抽样；Full 容量失败样本保留，不剔除。

## 状态与证据

GPU 运行尚未启动（协议含完整 prepare/eval/verify 命令）。计划文件（runs/public_retention_*_plan.json）与工具入口 tools/{prepare_public_retention,public_retention_eval,run_public_retention_parallel,verify_public_retention,test_public_retention}.py 已就绪。
