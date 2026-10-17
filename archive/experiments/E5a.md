## E5a — `recent_blocks`：原生 block 纯 recency 选择

### 动机

E4 已经把范围收窄到一个开关。同一 `gate=ones` 下：

| cell | compactor | EM |
|---|---|---|
| A | `none` | 27.00 |
| C | `svd_tokens` | **0.00** |

两格唯一差异是 compactor。所以 `svd_tokens` 是充分原因，且 `svd_tokens` 被移出主线。

E5a 换上目前最保守、但仍然真实做 K → K' 的算子：把窗口切成完整 256-slot 原生
block，保留最近的 B_out 个，逐 slot 原样复制。不合成任何新向量，reader 见到的
是它本来就训练过要读的 memory。因此 E5a 隔离的是"compact / 写回生命周期本身
能不能活"，而不是"算子好不好"。

### 实现

`methods/prlmc/consolidate.py`：

```python
def recent_block_compact(source, target, block=256):
    layers, k, d = source.shape
    b = k // block
    b_out = target // block
    x = source.view(layers, b, block, d)
    kept = x[:, -b_out:, :, :]
    return kept.reshape(layers, b_out * block, d).contiguous()
```

`window` 与 `target` 必须同时是 `block` 的整数倍。`compact_with_report` 在最终
对 `used - protect` 做 clamp 之后可能把 `window` 打回非整数倍（当最新 block
窄于 `block` 时），因此新增一条显式守卫，返回 `reason_skipped:
block_misaligned:...`，而不是让算子在 try 里抛异常、把原因丢进
`compactor_error`。

配置：`configs/agent_conf/prlmc_e5a.yaml`，相对 E4 的 cell A 只改三行
（`output_dir`、`stats_dir`、`compact_mode: none → recent_blocks`）。

### 结果

`factconsolidation_sh_32k`，n=100，`gate_mode: ones`，`trigger: occupancy`，
`chunk_size 400`，82 次写入。

| 指标 | 基线 | A（none） | C（svd_tokens） | **E5a（recent_blocks）** |
|---|---|---|---|---|
| exact_match | 22.00 | 27.00 | 0.00 | **26.00** |
| f1 | 25.990 | 32.019 | 0.000 | **31.82** |
| substring_exact_match | 24.00 | — | 0.00 | **28.00** |
| rougeL_f1 | 26.355 | 32.101 | 0.000 | **31.90** |

内部信号：82 次写入，5 次压缩，`evicted_blocks 0`，`peak_used 9984`，
`used 8192`，`n_consolidated 2304`，`occupancy 0.667`，
`memory_construction_time 27.6 s`，`query_time_len 1.47 s`，
总时长 815.7 s。

### 结论

**生命周期是活的。** 压缩真实发生（5 次）、真实释放容量（`used` 9984 → 8192）、
且 reader 仍然正常作答（EM 26.00）。

**算子决定了 0 与 26 的差别。** C 与 E5a 的唯一差异是输出向量来自原生 memory
还是来自 SVD 子空间。EM 0.00 → 26.00，与 E3b 的机制判断一致：冻结的 reader
无法解释一种它没见过的 latent 编码。

**代价相对 `none` 是 1 个 EM（27.00 → 26.00）。** n=100 下差值标准误约 6.1
个百分点，这个差**不显著**，不能读成"压缩造成了 1 点损失"。

---

[← 索引：EXPERIMENTS.md](../EXPERIMENTS.md)
