## E7a — 阶段 A：功能干扰图（8×8 梯度余弦 + 单步功能矩阵，seed 0）

### 目的

E7 计划的阶段 A，问题：**8 个训练窗口的知识能否从梯度/功能上分开？**
若梯度方向高度重叠，专家路由只是在随机隔离梯度；若语义相似度能预测
功能干扰，内容路由有机制基础。纯测量，不改结构、不改 loss
（`structure_changed: false`、`loss_changed: false`）。

### 设计

- 复用 E6c/E6d 的 8 个固定窗口（row 3，`DEFAULT_WINDOWS`）、每窗 20 题
  probe、19→9 carrier compactor、既有 Stage-B loss（gold+0.5KD+0.1hidden
  +0.05carrier+0.02stats）、lr 1e-3、gamma_max 0.05。
- **梯度臂**：每个窗口在 bitwise 相同的初始 compactor（`initial_state_md5
  f1a709a642244bd17b7691ba5eb0ece5`，注意与 E6d 的 `0f7065…` 不同——本
  脚本未加载 seed-0 的 E6d 检查点，而是用 seed 0 重新初始化的 compactor）
  上计算一个 Stage-B 步的完整梯度，用捕获优化器（`CaptureAdamW.step`
  只拍照不更新）获得，参数不动。8 个梯度两两余弦得 8×8 矩阵。
- **功能臂**：每个 source 窗口从同一初始 compactor 独立出发，执行 1 个
  真实 AdamW Stage-B 步（teacher 缓存命中），然后用更新后的 compactor
  评分全部 8 个窗口，与 baseline（初始 compactor 的 before 评分）对比，
  记录每 target 的 rescued/harmed 题目与逐题翻转。

### 配置

| 项 | 值 |
|---|---|
| 窗口 | E6c 8 窗（3:17890…3:15230） |
| 更新量 | 1 个 Stage-B 步（与 E6d 每窗总量的 1/96） |
| seed | 0 |
| lr / gamma_max | 1e-3 / 0.05 |
| 脚本 | `tools/e7a_interference.py`（md5 `dd223cb68f9e4bf0e606e8d989592783`） |
| 运行 | v100-2 gpu0，PID 68924，2026-10-02 06:55 UTC 启动、07:54 UTC 写盘（59 min） |
| 输出 | `/tmp/e7a_interference_seed0.json` → `runs/e7a_interference_seed0.json` |

### 结果

**梯度余弦矩阵（offdiag）**：mean 0.195，range 0.046–0.319，无负值、
无 >0.5 的对。最高对 3:17130–3:14470（0.319），最低对 3:16750–3:17130
（0.046）。8 个窗口的梯度方向**低重叠但全部正相关**——没有强冲突
（无负余弦），也没有近似重复（无 >0.5）。

**功能干扰矩阵（1 步更新）**：

- harmed：全矩阵 **0**。单步更新不伤害任何窗口的任何题。
- rescued：共 10 次。对角（自救）2 次：3:17890→3:17890、3:16750→
  3:16750。offdiag 8 次，全部指向两个 target：
  - **3:17890 的 "Charlie Manuel sport" 被 8/8 个 source 全部救回**
    （含自己），包括与 3:17890 梯度余弦最低的窗口（3:16750，cos 0.114）。
  - 3:16750 的 "Western International League sport" 被 2/8 source 救回
    （3:17890、3:16750 自身）。
- 其余 6 个窗口的 offdiag rescue 全为 0。

**梯度相似度不预测功能效果**：corr(cos, n_rescued) = −0.116（n=56，
无意义水平）。Charlie Manuel 题被包括最不相似窗口在内的全部 source
救回，说明这个 rescue 不是"相似知识互相帮助"，而是**任何沿 Stage-B
loss 方向的小步都能救它**——该题处于初始 compactor 的决策边界上，
一步就跨过去。

### 判读

**当前证据支持"8 窗不可分但无冲突"**：梯度低重叠（0.05–0.32）说明
各窗口的优化方向不同（分知识的前提存在），但单步功能矩阵几乎全零、
唯一的系统性 rescue 是决策边界题的普遍翻转，与 E6e 的"漂移压过信号"
一致——一阶近似下窗口间没有可测的功能干扰，遗忘是 96 步连续训练的
累积漂移，不是单步冲突。

**对 E7 的影响**：

1. 单步功能矩阵太粗糙，`harmed=0` 不等于"无干扰"，只说明单步太弱。
   E6e 已证明干扰在 24+ 步尺度显现。若用此矩阵设计路由，会得出"随便
   分都行"的错误结论。
2. 梯度余弦的"低重叠"是内容路由的一阶有利证据（方向可分），但
   corr(cos, rescue)≈0 表明一阶梯度方向不预测功能效果，二阶/多步
   效应才是 E6e 观察到的干扰来源。
3. 下一步（等用户指令）：多步版本（如 24/96 步）的功能矩阵、内容
   相似度（fact-line embedding）与干扰矩阵的聚类对比、3 seed 复验。

### 附：产物

| 文件 | 大小 | md5 |
|---|---|---|
| `runs/e7a_interference_seed0.json` | 205165 B | `44dbd06e26c3bf73c5f7014aa63e2bc2` |

报告结构：design（种子/初始 md5/loss/步数）、baseline（每窗 probe/eligible
/before_correct）、gradient_norms、gradient_cosine（8×8）、
functional_interference（delta/rescued/harmed 矩阵 + by_source 逐题）。
