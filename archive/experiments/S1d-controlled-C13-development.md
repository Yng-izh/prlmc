# 冻结 C13 的独立受控开发 episode 生成评测

本轮仅验证原 C13 离开 row 3 两个方法开发窗口后的生成收益，不再增加保护机制、rank 扫描或提示词扫描。

## 数据与主比较

- 数据：`runs/sic_paper_controlled_plan.json`，实体隔离的受控版本事实流。
- 仅运行 development episode `0, 2, 4, 8, 14, 15, 17, 18`。
- 16 个 test episode 保持封存；入口没有 test stage，不生成 test ledger 或预测。
- 每个 episode：30 行 anchor + 380 行写入，19 个 256-slot native blocks，压缩至 9 个块。
- 相同 64 道 evaluation QA 上比较 Full / Selection / D0 / C。
- 主指标：每个 episode 的配对 `EM(C) − EM(Selection)`，以 episode 为重采样单位报告均值、95% bootstrap 区间及正负 episode 数。
- 次指标：`C − D0`、Full、压缩 lost、rescue、harmed、训练与建基时间和峰值显存。不将 512 道题视为 512 个独立场景。

冻结指算法与超参数；不同 episode 的写入事实对应不同的局部 basis/alpha，不能复用 row 3 的窗口专属 checkpoint。

## 固定方法

`tools/s1d_controlled_c13.py` 的 base、candidate、fact target、事实 CE、fact basis、tail pool 和 tail training kernel 从 `tools/s1d_experiment1.py` 原 C 分支逐函数复制，CPU AST 回归要求保持一致。

1. 按原受控协议写入带序号的事实行；训练目标使用去序号后的原始事实文本，与原 C 一致。
2. 根据 native Selection 确定 dropped blocks。latest-owner 仅依据已写入文本计算。
3. 计算写入事实答案 span 的 Full/D0 CE；保留 Full CE ≤ 2、D0−Full CE ≥ 0.1 的 dropped latest facts，按差值排序最多取 40 行。
4. 对这些事实的 D0 `down.weight` 梯度做 SVD，固定前 13 维；有效秩不足 13 明确记失败，无替代空间或阈值调整。
5. 原 C 的最多 40 行 tail-4-token CE 求和，alpha 零初始化，只训练 13 个系数。AdamW，lr 0.003，weight decay 0.01，norm budget 1，60 steps，seed 0。无保护惩罚。
6. 先保存 step60 basis/alpha 与来源哈希，然后打开 evaluation QA。discovery/calibration QA 不参与选择或训练。
7. 复用 `s1d_c_generation.evaluate_four_arms`：固定 B2 FEWSHOT、greedy、newline stop、max_new_tokens 32、attention scale 1.0、官方 EM。

写入事实的答案字段是允许的监督，不能称为完全无答案监督。本数据由共享语料和模板构造，只支持受控跨实体泛化判断，不能代替独立自然 context 泛化。

## 运行与审计

入口先做 episode 0 的 1-step / 2-question smoke；完整的写入、事实发现与 13D 建基不缩减。smoke 只检查执行路径，不据结果改变方法。

```bash
cd /share/latent/PRLM-C
PYTHONPATH=tools:. /share/latent/mllm_env/bin/python -m unittest tools/test_s1d_controlled_c13.py -v
/share/latent/mllm_env/bin/python tools/test_s1d_c_generation.py
/share/latent/mllm_env/bin/python tools/test_s1d_controlled_contract.py

/share/latent/mllm_env/bin/python -u tools/s1d_controlled_c13.py \
  --episode 0 --cuda-device 0 --smoke \
  --out runs/s1d_controlled_c13_v1.smoke.ep0.json

/share/latent/mllm_env/bin/python -u tools/run_s1d_controlled_dev.py \
  --devices 0,1,2,3,4,6,8,9 \
  --smoke-report runs/s1d_controlled_c13_v1.smoke.ep0.json \
  --prefix runs/s1d_controlled_c13_v1
```

启动器要求 smoke 完成且源码哈希一致，实时检查 8 张卡有至少 30GB 可用显存且占用不超过 256MiB。不会终止其他任务。每个 episode 单独保存训练轨迹、checkpoint、basis rows、逐题生成、配对集合及成本。完成报告拒绝覆盖；若生成中断，`--resume` 只加载来源一致的已冻结 checkpoint 再做评测。

汇总时独立重算逐题官方分数和配对统计，核对每组来源一致。失败的 episode 保留并报告；未全部完成时不偷偷丢弃失败场景计算整体置信区间。两个既有 row 3 窗口不并入本轮独立结果。

## 决策边界

本轮结束后先查看 C−Selection 的 episode 级收益及不确定性，不按结果改 rank、λ、训练 checkpoint、提示词或 attention 参数。若稳定正收益，冻结方法与模型/代码/数据哈希后，再按明确授权进行一次封存测试。本启动器不会自动打开测试集。
