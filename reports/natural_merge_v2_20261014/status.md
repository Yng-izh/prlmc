# 自然跨块融合 A/B/C 启动记录

用户授权在 v100-1 同时运行全部三组，并允许关闭自己的占卡进程。

已准备 `data/natural_merge_v2/train.json`：312 个原始 SQuAD train context、1402 道原始 QA，训练计划文件 SHA256 为 `624b5fea38db5b185dfa9ee7b0aa97e9eb1637b9bd3599255f629c0f233f0144`。使用原 E1-development 计划，未启动正式 E1/E2/E3。

仅停止 GPU 3 的 `/share/qwen35-vllm/venv/bin/python - 3 24000`，PID 3185806，SIGTERM。其余占卡任务保留。

首次 smoke 三组均在 reader backward 分配 3.12 GiB 物理记忆池梯度时 OOM，0 次参数更新；失败记录已归档。修复将原始 source window 在 frozen-reader backward 期间暂存 CPU，释放无需参与 reader graph 的 source/carriers GPU 副本；随后恢复原始 source 进行 compactor VJP。三组使用同一修复，研究协议未改变。修改后本地 21 项 v2 CPU 测试通过。

重跑目录：`v100-1:/share/latent/PRLM-C/runs/natural_merge_v2_abc_20261014_023355_r2/`。

三组 smoke 各完成 1 次更新，退出码全部为 0；正式训练进程已经同时启动：

| 组 | GPU | 正式训练 PID | 目标 |
| --- | --- | --- | --- |
| A | 1 | 3204520 | 原块级融合 + 原 CE |
| B | 2 | 3204523 | 原块级融合 + 压缩损失加权 + Full KL |
| C | 3 | 3204526 | 槽位融合 + 与 B 相同的目标 |

监督进程 PID 3203794：每组自动执行 smoke → 全新 64-update 训练 → 固定训练窗口生成诊断 → E1 开发冻结生成，最后生成 comparison.json。不选择中间 checkpoint，不扫描超参，不自动启动正式评测。

本记录确认启动与 smoke 成功，不代表训练或最终评测已经完成。完整命令、PID、失败和重跑记录见本目录 JSON 与远程队列日志。
