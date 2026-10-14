# S1d stage 2 oracle — 固定自监督 13D 基可达性诊断

- **状态**：已完成；结论为 **未显示正净收益**。
- **用途**：仅开发阶段 oracle 诊断；不参与论文方法结果，不作为 held-out 或泛化证据。
- **窗口**：`(3,17510)`，原 stage 2 自监督臂最差开发窗口；选择明确为事后诊断。
- **基**：沿用 `runs/s1d_selfsup_rows_v2.pt` 中该窗口缓存，按原驱动相同排序与 SVD 取前 13 列；不使用 QA 梯度重新建基或改变基。
- **参数**：`D_new = D0 + reshape(U_selfsup @ alpha)`；仅 13 个 alpha 可训练；reader、carrier 原参数和基均冻结；alpha 从零开始。
- **监督**：38 道 `rescue_d0_wrong = full ∧ ¬selection ∧ ¬D0` 的真实 QA 第一答案别名。仅答案 token CE，prompt 不进入损失；每步累积各题答案 token 平均 CE 的梯度。D0 正确题不进入训练，仍参与损伤评分；未额外加入保护项。
- **优化**：60 步，AdamW，lr=0.003，weight decay=0.01，位移预算 1.0；每 10 步 TF，末步 GEN；seed=0。保持原 stage 2 优化器、学习率、预算和步数。
- **对照**：同窗口原 selfsup-13d 最终恢复 5、损伤 5、净收益 0；random-13d 为 0/1/-1，unconstrained 为 4/7/-3。
- **指标**：完整同一 QA 题集上，恢复仅统计 `rescue_d0_wrong`，损伤统计全部 D0 原本正确而训练后错误的题；net = recovery - harm，并非全题正确数差。

## 结果

| 步数 | 答案 CE | 位移范数 | 新恢复 | D0 损伤 | 净收益 |
|---:|---:|---:|---:|---:|---:|
| 10 | 3.6359 | 0.100 | 0 | 1 | -1 |
| 20 | 3.3526 | 0.180 | 3 | 4 | -1 |
| 30 | 2.9612 | 0.247 | 6 | 6 | 0 |
| 40 | 2.7937 | 0.310 | 6 | 8 | -2 |
| 50 | 2.7435 | 0.352 | 5 | 8 | -3 |
| 60 | 2.7026 | 0.380 | 5 | 8 | **-3** |

原 `(3,17510)` selfsup-13d 最终为 `5/5/0`。因此答案 CE 确实能短暂提高恢复数，step 30 达到 `6`，但没有形成正净收益；继续优化答案 CE 增加了 D0 损伤，最终净收益降至 `-3`。相对原 selfsup 臂，最终恢复数相同、损伤增加 3。

**诊断**：在这个窗口、固定当前 selfsup 13D 基、60 步 AdamW 和位移预算下，oracle 没有显示“净收益明显提高”。这削弱了“主要只是无答案训练信号导致失败”的解释，并把下一步优先级推向基构建/目标设计审查。但这不是当前基缺少功能方向的严格证明：step 30 的 `6/6/0` 说明基内存在部分可达恢复方向；同时该 oracle 是有限步、答案 token CE 目标，不是数学上界，也没有直接优化 net 或 no-harm 目标。

## 解释边界

60 步 oracle 是有限优化下的可达性诊断，并非严格的数学上界，也不保证找到 CE 最优点或净收益最优点。最终第 60 步为主对比；中间测量点只作为轨迹诊断，不事后替换最终结果。

若在相同固定基内明显提高 net，说明当前基仍承载可用恢复方向，优先检查无答案训练信号与真实 QA 目标的对齐。若没有明显提高，下一步可以优先检查建基方式，但仍需保留有限优化、CE 与净收益的目标差异及恢复—保护权衡等解释；负结果不构成方向缺失证明。

GEN 沿用原贪心生成、32 token 整段 EM；已知续写使原 full/D0/trained 整段 EM 无判别力。该指标保留记录，但不用于此次诊断的决定性结论，不临时改变答案抽取协议。

## 复现与审计

```bash
/share/latent/mllm_env/bin/python tools/s1d_stage2_dev.py \
  --window 3,17510 --arms oracle_13d --cuda-device 1 \
  --out runs/s1d_stage2_oracle.w17510.json
python3 tools/test_s1d_stage2_dev.py
python3 tools/verify_s1d_stage2_oracle.py
```

运行于 `v100-2` GPU1。驱动 SHA256：`8053c26cc34e11ecb4b9a0b845334c23e755ba2e63dad2cca40ee71c54a6f190`。

审计包括：与旧报告的 checkpoint/selfsup rows/gold rows 哈希、TF full/select/D0 逐题基线及 cohorts 一致；严格参数/功能 identity-at-init；D0 generation-sandwich；逐模块模式/RNG/memory 恢复；每次测量检查基缓冲区不变、13 个 alpha 与范数预算；逐题重算 TF/GEN 指标。真实 QA 答案仅进入 oracle 臂梯度。

结果产物为 `runs/s1d_stage2_oracle.w17510.json`；独立校验器为 `tools/verify_s1d_stage2_oracle.py`。
