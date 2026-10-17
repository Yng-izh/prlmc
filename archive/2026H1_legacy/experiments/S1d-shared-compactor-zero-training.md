# 共享残差模块的零训练跨 episode 评测

本脚本对应方法定义的第一步：加载已有 E6d interleave 的共享 `CarrierResidualCompactor` checkpoint，在新的 controlled development episode 上只写入记忆、执行固定 19→9 压缩并生成回答。

## 评测边界

- 使用同一份共享 E6d 参数处理全部 8 个 development episode：`0, 2, 4, 8, 14, 15, 17, 18`。
- 每个 episode 从空记忆开始，但不构建事实梯度 basis、不训练 alpha、不创建 optimizer、不调用 backward。
- 16 个 test episode 没有命令行入口，脚本不会创建 test ledger，也不会生成测试预测。
- 每个 episode 比较 `Full`、`Selection`、`SharedE6d`，使用同一组 64 evaluation QA 和固定 B2 FEWSHOT / newline stop / max_new_tokens=32。
- 主指标是 episode 级 `EM(SharedE6d) - EM(Selection)`，汇总以 episode 为独立重采样单位；题目不是独立场景。

## 完整性约束

- checkpoint 必须是固定的 `runs/e6d_interleave_ckpt.pt`，默认 `gamma_max=0.05` 与 E6d 训练记录一致。
- 每个 episode 必须写入 30 行 anchor 和 19 个 20 行块，得到 19 个 native 256-slot blocks；任何截断、eviction 或自动 compaction 都直接失败。
- Selection 直接使用原生 `_select_block_indices`；SharedE6d 使用同样的 source、importance、goal 和 select weights 经过冻结 compactor。脚本验证零残差 carrier 与 Selection pool 完全一致。
- 每组生成前后检查 memory 和 active mask，生成不能污染下一个 arm 或下一个 episode。
- 单 episode report 保存来源哈希、逐题 prediction、EM、压缩 active slots、显存和耗时。启动器拒绝覆盖已有 artifacts，并在汇总时重新检查 paired summary。

## 运行方式

```bash
cd /share/latent/PRLM-C
PYTHONPATH=tools:. /share/latent/mllm_env/bin/python -m unittest tools/test_s1d_shared_compactor_eval.py -v

/share/latent/mllm_env/bin/python -u tools/run_s1d_shared_compactor_eval.py \
  --devices 5,9 \
  --model-path /dev/shm/s1d_reader_fp16 \
  --checkpoint runs/e6d_interleave_ckpt.pt \
  --prefix runs/s1d_shared_e6d_dev_v1
```

启动器支持 1–8 张卡，每张卡同一时间仅运行一个 episode，完成后自动领取后续 episode。启动时要求每张卡至少 30000MiB 可用显存且占用不超过 256MiB。当前 C13 逐 episode 适配实验与这个零训练评测是两条不同证据，不能混合汇总。

## 解释口径

若 SharedE6d 在未见 development episode 上稳定超过 Selection，这支持共享内容条件残差算子具有迁移信号；若不超过，只能说明当前 E6d checkpoint 尚未证明迁移，不应退回到测试 episode 的 alpha 适配。该评测不能代替后续用独立训练 episode 重新训练 B2，也不能代替封存测试或跨数据集验证。
