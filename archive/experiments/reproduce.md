## 复现方式

所有运行均在 `/share/latent/PRLM-C`（MemoryAgentBench 的副本），
`apply_patch.py` 已把 `prlmc` 分支装入宿主 `agent.py` 与 `utils/templates.py`。
宿主无 `git`，每次本地改动都必须重新上传。

```bash
# E1
GPU=6 QUERIES=2 bash run_smoke.sh
# E2
GPU=6 QUERIES=8 bash run_obs.sh
# E3
GPU=6 QUERIES=0 bash run_r1.sh
# E4（三格串行；D 即 E3，不需重跑）
CELL=a GPU=6 QUERIES=0 bash run_2x2.sh
CELL=b GPU=6 QUERIES=0 bash run_2x2.sh
CELL=c GPU=6 QUERIES=0 bash run_2x2.sh
# 或一次串起：bash chain_2x2.sh
# E5a / E5b / E5c
RUN=e5a GPU=6 QUERIES=0 bash run_e5.sh
RUN=e5b GPU=6 QUERIES=0 bash run_e5.sh
RUN=e5c GPU=6 QUERIES=0 bash run_e5.sh
RUN=e5d GPU=6 QUERIES=0 bash run_e5.sh
RUN=e6  GPU=6 QUERIES=0 bash run_e5.sh
```

集群紧张时用重试器（自动挑最空的卡并重试）：

```bash
RUN=e6 QUERIES=0 NEED_MB=21000 bash retry_run.sh
```

**不要在 `retry_run.sh` 运行期间覆盖该文件**：bash 按字节偏移增量读取脚本，
中途覆盖会让下次读取落到半行。E6 第一次投递就是这样在收尾阶段报出
`genuine: command not found` 的。

E5c 需要先给 modeling fork 打补丁（该文件不在本地 git 里，`tools/patch_modeling.py`
是幂等且可回滚的）：

```bash
# 状态检查（只读）
/share/latent/mllm_env/bin/python tools/patch_modeling.py --root /share/latent/PRLM-C --check
# 应用
/share/latent/mllm_env/bin/python tools/patch_modeling.py --root /share/latent/PRLM-C
# 回滚
/share/latent/mllm_env/bin/python tools/patch_modeling.py --root /share/latent/PRLM-C --revert
```

未打补丁时 `active_memory_mask: true` **静默失效**（模型没有该 buffer，
`_sync_active_mask` 直接返回），池子仍按原样读取。

运行结果落在 `outputs/<run>/`，内部信号落在同目录的 `prlmc_stats.jsonl`，
每次查询追加一行，`writes_since_last_query` 含该查询之前的所有写入记录。
压缩事件以 `{"event": "compact", ...}` 独立行写入同一列表。

`tools/calibrate_obs.py` 读取 `prlmc_stats.jsonl` 并打印上述信号分布。

E6a / E6b-mini / E6c 的离线训练在 v100-2 上直接驱动 `/tmp` 副本
（宿主无 `git`，每次本地编辑必须重新上传并清 `__pycache__`）：

```bash
# E6a：单窗口 overfit（无统一启动脚本，直接 python /tmp/e6a_overfit.py，详见 E6a 章节）
# E6b-mini：8 训练 + 2 held-out，24 步/窗口
bash tools/run_e6b_mini.sh
# E6c：同一脚本，96 步/窗口，产物 /tmp/e6c_ckpt.pt + /tmp/e6c_train_report.json
bash tools/run_e6c_train.sh
# E6c 三臂 bench（依赖 /tmp/e6c_ckpt.pt 存在；三个臂各投一次 retry_run.sh）
bash tools/run_e6c_bench.sh
# 三臂对比（paired QA；--show-questions 列出每题获救/受损）
cd /share/latent/PRLM-C && /share/latent/mllm_env/bin/python tools/e6c_compare.py --show-questions
```

`/tmp` 为分主机目录，`/tmp/e6a_screen_data.json` 在两台机器上内容不同——
训练臂必须投在筛选数据所在的那台（E6b/E6c 均为 v100-2）。
运行前应确认宿主上 `methods/prlmc/layout.py` 的 md5 与本地一致。

---

[← 索引：EXPERIMENTS.md](../EXPERIMENTS.md)
