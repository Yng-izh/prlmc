# 历史实验与清理记录

当前入口见[仓库 README](../README.md)和[自然融合 A/B/C 协议](../experiments/Natural-merge-v2-ABC.md)。早期实验文档已集中到 `archive/experiments/`，保留原始文本及当时的结论。

## 保留的关键历史

- [E0 原生公开基线](experiments/E0.md)：历史配置下的原生结果，不能直接当作当前同条件对照。
- [E6d 共享训练](experiments/E6d.md)：早期共享模块与 interleave 路线。
- [原始 Native-QA](experiments/Native-QA.md)：原始 context 和问题的读出诊断。
- [C13、保护与扩维汇总](experiments/S1d-round-20261013-results.md)：方法开发窗口的完整正负结果。
- [受控开发 episode 汇总](experiments/S1d-controlled-development-20261013-results.md)：人造任务的同类迁移证据，已退出论文主结果。
- [公开记忆保持方案](experiments/Public-retention-frozen-SharedE6d.md)与[SQuAD author512 中间分析](experiments/SQuAD-author512-interim-analysis-20261013.md)：转向原始自然任务的依据。

历史文档中的运行路径、相对链接和状态叙述描述其创建时的仓库布局；复现旧实验请使用对应历史 Git 版本，而不是将已删除旧脚本与当前冻结协议混合。原始报告及校验证书仍在 [reports](../reports/README.md)。

## 2026-10-14 清理

移除不属于当前自然融合依赖链的旧扫描、监视、启动和审计脚本、过时配置，以及仓库根目录 `runs/` 中的旧临时运行产物。保留当前 A/B/C、自然评测、FP16 缓存工具、兼容依赖、完整历史报告和新实验文件；正在运行的服务器没有改动。

清理前提交为 `08539990f7af2742b1abdfd76b43ab012ca5007f`。旧跟踪文件可按路径查看或导出：

```bash
git show 08539990f7af2742b1abdfd76b43ab012ca5007f:tools/run_s1d_parallel.py
git archive 08539990f7af2742b1abdfd76b43ab012ca5007f runs/ > historical-runs.tar
```

逐文件删除与归档映射见 [cleanup-20261014.json](cleanup-20261014.json)。清理前的完整本地工作树（含尚未提交的 A/B/C 文件）另有仓库外备份。整理只减少当前 checkout 的冗余文件，不改写 Git 历史，也不会减少已有历史对象的下载体积。
