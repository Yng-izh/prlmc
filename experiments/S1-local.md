# S1-local：Protected Functional Subspace Residual

## 最终结论：NO-GO（本轮hard-nullspace构造）

同一 S0 hard window（row=3、start=17890）、E6d interleave 检查点上的五臂正式比较完成，运行退出码 0。所有训练臂固定 60 步、每 5 步测量，最终第 60 步判门，没有选择中间峰值。

| 臂 | Rescue / 13 | 总 harm / 75 | D0 新增 harm：保护集 / 31 | D0 新增 harm：全部题 | 正确 / 296 | Full-eligible select-relative 净收益 |
|---|---:|---:|---:|---:|---:|---:|
| Native select | 7 | 0 | 0 | 22 | 78 | 0 |
| Unconstrained residual | 10 | 17 | 1 | 11 | 84 | −7 |
| S0b coefficient oracle | 7 | 15 | 1 | 9 | 93 | −5 |
| U_L alpha | 7 | 6 | 0 | 10 | 107 | +8 |
| U_safe alpha | 1 | 37 | 0 | 1 | 65 | −32 |

D0 仅作初始状态诊断，不是第六个方法臂：rescue=0/13、总 harm=44/75、正确54/296。native select 的全题集 D0 新增 harm=22 是替换 D0 为 selection 后的题级差异，不是训练损失。

保护臂失败条件明确：rescue 1<7；总 harm 37 不低于 oracle 15、不满足 half-oracle 上限7，也不低于自由更新17与UL6；净收益−32<0。只有 retained≥10 条件通过。按冻结协议停止该方法，不运行 S1-generalization 或 S1-paper，不增加变体。

保护臂在保护集合内新增 harm=0，并不代表部署无损：它仍保留了 D0 的大量既有错误，总 harm=37/75；仅救回1/13。UL 控制臂本地表现更好，但本实验不验证其泛化，也不把它替换为新主线。

## 冻结协议与确定性修正

固定 19→9 blocks、原窗口写入顺序与 seed=0。保留原 60 道 dropped-block 探针及原 S0 13 个 lost 主目标。保护探针扩展到同一已写入窗口的有效事实，遵守 latest-fact、窗口归属、去重、答案长度和官方 QUERY_TEMPLATE；共有296题。保护集合严格为 full正确且native select正确，共75题，不引入 fullK。

实际 reader attention_dropout=0，但两个 PEFT adapter 的 LoRA dropout=0.1。冻结权重不会关闭 dropout。结果前修正保留原窗口写入随机路径，在写入完成后将384个 dropout 子模块置为 eval，父级仍 train 以启用梯度检查点。评分、Jacobian 与四臂训练使用相同模式，后续没有递归 train 重置。原60题在扩展评分中重复测试一致性，通过才进入训练。

确定性 lost 总数14，旧13题全部仍丢失，仅额外新增 `Who founded Tiantai?`。正式训练与主 rescue 分母固定为原13题；额外题仅作诊断，不用于满足 rescue≥7。若旧13题任何一题不再 lost，驱动保存审计并停止，而非替换目标。历史随机训练态 S0/S0b 分数不是本轮确定性结果的可比对照；所有指定臂均重跑。

原13题中有7题也属于 select-retained：lost 指 D0 丢失，retained 指 selection 保住。没有删保护题来消除重叠。native select 自身达到7/13，rescue达标本身不是超过selection的证据。

本轮在相同D0上重算逐题 gold-answer CE Jacobian，旧S0 rows仅用于目标来源审计，不用于建基。

`U_L = orth(J_L^T)`；`U_K = orth(J_K^T)`；`U_safe = orth[(I-U_K U_K^T)U_L]`。CPU float64 薄SVD，不构造131072×131072矩阵。正式 ranks=13/75/6，基的保护正交误差5.48e-15。这是一阶CE保护条件，不是有限更新的token-exact保证。

指定五臂为 native select、自由down位移、S0b C=I+E、D0+reshape(U_L alpha)、D0+reshape(U_safe alpha)。四训练臂共享lost CE、AdamW lr=0.003、60steps、seed=0，其他参数冻结。三个参数位移臂Frobenius上限1.0；oracle系数有效位移上限0.5，两种空间预算不宣称等价。

冻结门槛：retained≥10；protected最终rescue≥7；总harm严格低于oracle且≤floor(oracle harm/2)；harm不高于自由更新与UL；full-eligible select-relative净收益≥0。任一未过即停止，不启动后续阶段。计数门不冒充统计显著性。

正确率是 teacher-forced answer-token exact match，不是官方生成EM。总harm相对select；新增harm同时报告保护集合内与全部296题。净收益定义为全部full正确、select错误题的恢复数减去75道select-retained的总harm，不等于全题单正确数差。

## 核验与证据

原始结果：`runs/s1_local_protected.json`，status=complete，verdict=NO-GO；完整日志：`runs/s1_local_protected.log`。结果JSON SHA256：`7debe0e4b14d56d166f9223d088919bfeebf469d0a43adadfb33854042218af5`。

`tools/verify_s1_local_result.py --require-complete` 独立核验原13题、历史仍丢失条件、逐题rescue/harm列表、全题集新增破坏、accuracy/harm rate、四臂各12次测量且终点60步，并重新计算判门。计数篡改与未完成结果检测通过。限制：扩展全池逐题正确性未序列化，full-eligibility不能由该核验器重建，净收益门仅核验记录值；rescue与harm可逐题重算，且二者已独立足以判定NO-GO。

`tools/audit_s1_checkpoints.py` 在CPU从真实权重核验源检查点SHA256、冻结q/k/gamma/route/up、原始down、实际位移预算及保存子空间。全部四臂通过：自由更新实际范数0.99999997；oracle原始E范数1.02284646，实际有效C-I范数0.49999998；UL实际范数0.44197263；protected实际范数0.39388690，实际增量对U_K投影范数1.08e-9。说明保护臂确实按声明的六维空间运行，失败不是未启用投影或冻结权重失效。

实际reader配置：`runs/s1_reader_config.json`。正式basis、逐题Jacobian与四臂最终权重在远端 `/share/latent/PRLM-C/runs/s1_local_protected.*.pt`，五个工件的本地副本已完整收回，SHA256与远端逐项一致。大型二进制不纳入普通Git提交，大小、双端路径与哈希见 `runs/s1_local_artifact_manifest.json`。独立核验记录见 `runs/s1_local_metric_audit.json` 与 `runs/s1_local_checkpoint_audit.json`。

源检查点SHA256：`a9fe2ea3f7a4e2c21590cf2219591edb11fc65297ce36ddeed96c236d8020c5d`；历史S0 rows SHA256：`b35f1eb5835936f8832d3991fcd4d46e1d7330ab73970f42dac5d1347a110e1b`。正式驱动SHA256记录在结果JSON中。

## 复现与执行历史

```bash
/share/latent/mllm_env/bin/python -u tools/s1_local_protected.py \
  --carrier-ckpt runs/e6d_interleave_ckpt.pt \
  --s0-rows runs/s0_jacobian_rows.pt \
  --cuda-device 9 --steps 60 --measure-every 5 \
  --out runs/s1_local_protected.json
python3 tools/verify_s1_local_result.py runs/s1_local_protected.json --require-complete
/share/latent/mllm_env/bin/python tools/audit_s1_checkpoints.py \
  runs/s1_local_protected.json runs/e6d_interleave_ckpt.pt
```

此前首轮短验证因oracle外置参数未被内部checkpoint扫描而主动停止；修正为同一oracle注册到内部arm，不改变前向或门槛。第二轮因历史lost失配在基线处退出；gold差异=0、输入哈希一致。第三轮在确认LoRA dropout后主动停止。第四轮关闭dropout的短验证完成五臂，标记SMOKE-ONLY；压力检查使用14题、2步，与正式13题/60步分开，未用于正式判门。正式驱动九项回归测试通过。

本实验仅为同一窗口的监督局部机制验证，没有held-out泛化结果，也没有论文主表。泛化与论文实验的条件未满足，均未启动；SIC、router、reader adapter、SVD保持历史诊断，不新增分支。

## 后续研究判读修订

原NO-GO的对象是本轮hard-nullspace保护构造，不是全部functional-subspace路线。UL以13个自由度得到rescue7/13、harm6/75、净+8，优于指定自由更新和系数oracle的本地折中，构成继续受控验证的正证据；单窗口监督、不同优化坐标与最终位移范数仍限制其归因，不宣称普遍有效或已证实泛化。

K=75题是在select正确状态定义的，但JK取自D0；其中31题D0正确、44题D0错误。原13题中7题也是K，实际属于checkpoint退化repair，不能称为selection丢失。硬约束对正确与错误基点的CE变化同样限制，不允许一阶改善；这与“保留已有正确能力”的目标不同。它不是把44题永久冻结：有限更新仍修复7题，CE局部约束也不等价于token预测不变。

保存基CPU复核：投影奇异值6个非零（0.540、0.653、0.714、0.777、0.826、0.862），另7个约1e-14。lost与K的7题重叠已足以解释全部精确秩下降，因此13→6本身不独立证明公共轴被删，也不称剩余6维为边缘tail。投影后的span一般不是原UL与K正交补的交集；实际safe基在原UL外的Frobenius残差1.656。S0公共轴比例是历史训练态证据，不能直接当作本轮确定性谱的解释。

经用户明确批准，另立S1-soft-local协议：固定正式13维UL与原13题监督，按真实R（full正确/select错误）、P（full/select/D0正确）、T（full/select正确/D0错误）分开评价，仅对有限CE恶化作单边惩罚。它不撤回原判门、不改原工件，也不自动启动泛化或论文预算；预注册与新结果见[S1-soft-local](S1-soft-local.md)。
