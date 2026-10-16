# 原生 MemoryLLM 写入／淘汰策略：受控开发生成评测

在相同的 8 个 controlled development episode（0,2,4,8,14,15,17,18）上补跑原生策略。16 个封存测试 episode 不评测。

与既有 `s1d_shared_e6d_dev_gpu59_v1` 保持相同模型缓存、410 行场景、30+19×20 行原始写入文本、64 道 evaluation QA、B2 FEWSHOT、换行停止、32 新 token 和官方 scorer。该实验隔离原生写入／淘汰行为，不宣称执行了完整官方 benchmark；输入与读出固定为既有受控协议。

两项预先固定的容量设置：

| 设置 | 容量 | 说明 |
|---|---:|---|
| NativeDefault | 12,800 槽（50×256） | 原始导出模型的默认容量，显著大于 E6d 的有效预算 |
| NativeBudget | 2,816 槽（11×256） | 只缩小容量，与 Selection/E6d 有效槽数一致，写入／淘汰代码不变 |

每个场景冷启动，首个 chunk 按原生函数复制铺满记忆池；其后每次注入由原生 `inject_memory(update_memory=True)` 执行，各层独立随机淘汰 256 槽、保留剩余槽的顺序、末尾追加新 delta。不建立 anchor，不调用 PRLM-C manager/compactor，不训练参数。每个容量设置从同样的 seed=0 开始，结果依赖随机淘汰种子；这轮不扫种子。

运行前通过 AST 比较确认 `inject_memory`、`drop_memory`、`update_memory_with_delta_memory`、`cat_memory_and_hiddens` 与服务器原始 `modeling_memoryllm.py.bak` 一致。关闭 memory active mask，attention scale=1，走未管理的原始读出分支。

生成使用既有 `_generate`，为每组绑定临时记忆池并在每道题后恢复；保存原始预测、评分、写入槽数、原始策略校验、模型／数据／代码哈希、显存和耗时。汇总逐题复算所有新旧 arm 的评分，再比较 SharedE6d 相对两个原生设置及 Selection 的配对收益、损伤与 episode 级 bootstrap 区间。

```bash
cd /share/latent/PRLM-C
PYTHONPATH=tools:. /share/latent/mllm_env/bin/python -m unittest tools/test_s1d_native_policy_eval.py -v
/share/latent/mllm_env/bin/python -u tools/run_s1d_native_policy_eval.py \
  --devices 5,9 --prefix runs/s1d_native_policy_dev_gpu59_v1
```

两张 GPU 按队列各运行一个场景，同一 worker 顺序运行两个容量设置；失败时保留失败产物并停止分发后续 episode。既有 E6d 结果只读，不重新生成或选择 checkpoint。
