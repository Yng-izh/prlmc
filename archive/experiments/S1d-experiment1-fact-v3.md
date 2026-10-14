# S1d 实验1：受损事实建基与事实恢复目标的交叉对照

2026-10-12，批次 `s1d_exp1_fact_v3`。两个开发窗口、四组、每组60步已全部完成。服务器最终一致性证书 `valid=true`。弱窗口最佳净收益仅 +1，低于事先记录的 +5 推进门槛；本轮不进入实验2或官方基准。

## 最终结果

R 只统计预先指定的 `rescue_d0_wrong` 集合中新恢复的题；H 统计全部 D0-correct 题的新损伤，net=R−H。全题 TF 正确数还包含指定恢复集合之外的改善，因此其变化不等于 net。

| 窗口 | 组别 | 建基 | 训练目标 | R | H | net | 全题TF正确 | alpha范数 |
|---|---|---|---|---:|---:|---:|---:|---:|
| 强 `(3,16370)` | A | 原selfsup 13D | 原尾4-token CE | 8 | 1 | +7 | 105/265 | 0.373887 |
| 强 `(3,16370)` | B | 原selfsup 13D | 新事实答案CE+保护hinge | 12 | 5 | +7 | 112/265 | 0.424762 |
| 强 `(3,16370)` | C | 受损事实梯度13D | 原尾4-token CE | 10 | 1 | +9 | 112/265 | 0.327028 |
| 强 `(3,16370)` | D | 受损事实梯度13D | 新事实答案CE+保护hinge | 9 | 4 | +5 | 109/265 | 0.508247 |
| 弱 `(3,17510)` | A | 原selfsup 13D | 原尾4-token CE | 5 | 5 | 0 | 96/276 | 0.270495 |
| 弱 `(3,17510)` | B | 原selfsup 13D | 新事实答案CE+保护hinge | 6 | 9 | −3 | 105/276 | 0.363108 |
| 弱 `(3,17510)` | C | 受损事实梯度13D | 原尾4-token CE | 6 | 5 | +1 | 101/276 | 0.332637 |
| 弱 `(3,17510)` | D | 受损事实梯度13D | 新事实答案CE+保护hinge | 5 | 6 | −1 | 105/276 | 0.492366 |

强窗口基线 Full/select/D0 全题正确数为115/88/69；弱窗口为132/92/79。同窗口四组逐题基线、cohort完全一致。

## 判读

1. 只改建基的 C 是本轮最好的候选：相较 A，强窗口 net +7→+9，弱窗口0→+1，H分别保持1和5。两个窗口只有一次固定seed运行，不能据此宣称统计显著或跨context泛化。
2. 新训练目标 B 增加恢复，但损伤抵消了收益：强窗口较 A 多恢复4道且多损伤4道，net持平；弱窗口多恢复1道却多损伤4道，net降到−3。
3. 完整 D 在两个窗口都不超过 A，未观察到组合增益。相同新basis下，D相比C损伤更多、指定恢复更少。
4. B/D 全题TF正确数高于A并不代表压缩损失定向恢复更好。核心门槛仍以指定恢复集合的R−H判断；弱窗口没有任何组达到+5。
5. 本轮保留 C 的建基改善信号和新目标的负结果，不自动提高rank、追加loss扫描或进入实验2。官方生成EM、未参与选择的context和多seed验证均未运行。

## 固定协议

- 窗口 `(3,16370)`、`(3,17510)`，seed=0；冻结reader和basis，只训练13个alpha。
- AdamW lr=0.003、weight_decay=0.01；60步，每10步测量；残差L2范数上限1.0。
- QA token上限256，TF-only，跳过generation。评测QA仅用于测量，不用于建基、优化或事实选择。
- 使用写入时原始事实、最新owner去重。事实答案补全利用写入内容中的答案字段，不能称为完全无答案信息。
- dropped事实中选Full CE≤2.0、D0−Full CE≥0.10，按恶化值降序/key稳定排序，最多40条原始梯度建基；rank不足13明确失败。
- B/D用前最多20条恢复事实，再从kept块选择D0 CE≤2.0的事实，总训练条目≤40。保护项为保留事实平均 `ReLU(CE_current−CE_D0)`，lambda=1.0。
- A/C维持旧尾4-token损失的求和梯度；B/D为平均恢复CE+平均保护hinge。二者梯度尺度不同，所以B/D比较的是整个新目标配置，不能把差异全部归因于答案字段或保护项。
- B/C/D共用完整Full/D0事实CE缓存和一致事实选择，C/D共用相同原始梯度；缓存来自v1 C smoke，仅复用写入事实的CE与梯度，不复用旧alpha或QA训练。

## 运行与验证

共享目录 `/share/latent/PRLM-C`，Python `/share/latent/mllm_env/bin/python`；两台服务器均使用经校验的本机 `/dev/shm/s1d_reader_fp16`。调度先完成8个单步smoke，再开放8个正式任务；最多8张卡并发，不中断已有其他任务。

正式分配：v100-1 GPU2/3为D强/弱，4/5为C强/弱，8/9为B强/弱；v100-2 GPU9/0为A强/弱。v100-2 GPU5/7未纳入本批调度。最后启动的A弱窗口已完成最终60步，本文所有数值都是最终值。

回归测试17/17通过；smoke与正式证书检查完成状态、预算、冻结basis、相同driver/cohort/基线、A/B相同basis、C/D相同basis、B/C/D相同事实集合及保护事实来自kept块。归档后独立重算8报告共48个测量点的逐题R/H/net、全题TF正确数及accuracy，全部一致。

v100-1调度器曾因共享临时文件竞争退出，训练子进程继续运行；临时路径改用进程ID，重启调度器接管原PID，没有重新训练。v100-1恢复监控通过进程消失与完整报告判定结束，fleet中记载的exit=0并非恢复后直接wait取得的原子进程退出码；证书证明报告完整及一致性，不应把这一字段当作独立的原退出码证据。v100-2两个正式任务的退出码由原父调度器直接回收。

## 仓库产物

完整文本产物位于 [reports/s1d_exp1_fact_v3](../reports/s1d_exp1_fact_v3)：

- `summary.csv`、`summary.json`：最终结果表。
- `s1d_exp1_fact_v3.{A,B,C,D}.w{16370,17510}.{json,log}`：8个正式报告与完整日志，包含每10步的逐题结果。
- `s1d_exp1_fact_v3.smoke.*.{json,log}`：8个smoke报告与日志。
- `s1d_exp1_fact_v3.verification.json`、`.smoke.verification.json`：最终与smoke证书。
- `s1d_exp1_fact_v3.fleet.json`、两台`.fleet.log`、`.cpu_tests.log`、`.commands.json`：状态、调度、测试及命令记录。
- `SHA256SUMS.json`：归档文本文件SHA256；`s1d_exp1_fact_v3.binary_artifacts.json`：服务器保存的alpha/basis/原始梯度二进制大小与哈希；`s1d_v1001_reader_cache_verified.json`：v100-1缓存证书。

二进制alpha/basis/原始梯度及模型权重继续保存在服务器，仓库本次提交结果与日志、二进制校验清单和复现脚本。输入还需服务器的 `runs/e6d_interleave_ckpt.pt`、`runs/s1d_selfsup_rows_v2.pt` 及清单中两个v1事实梯度缓存。原始报告与证书保留服务器绝对路径，不重写原证据。要在服务器复验：

```bash
cd /share/latent/PRLM-C
PYTHONPATH=. /share/latent/mllm_env/bin/python tools/test_s1d_experiment1.py
/share/latent/mllm_env/bin/python tools/verify_s1d_experiment1.py --prefix s1d_exp1_fact_v3 --steps 60
```

本次脚本随提交保存在 `tools/`。完成状态文件不用于重新启动训练；复跑应使用新的prefix，先确认输入缓存及显存可用。
