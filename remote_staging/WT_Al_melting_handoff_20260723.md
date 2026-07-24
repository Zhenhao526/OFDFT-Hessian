# 108 原子 WT-MPN 自由能熔点计算暂停交接

更新时间：2026-07-23 18:34 CST  
计算节点：`node01`  
状态：**按用户要求暂停，尚未得到最终熔点，不得标记完成**

## 1. 任务目标

本任务使用 WT 路径计算 Al 固相和液相的绝对自由能，并以 900 K 的已验证自由能差为锚点，通过 Gibbs-Helmholtz 积分外推 `Delta G_liq-solid(T)`。求根得到熔点后，还必须用 1728 原子直接固液共存计算验证界面迁移方向，才能形成最终结论。

当前阶段不是早期的 MPN 直接共存扫描，而是 108 原子 WT 自由能/Gibbs-Helmholtz 主线。两条路线的数据都保留在 `runs/al/free_energy_wt`，但恢复时应继续 WT 自由能主线。

## 2. 登录与主要路径

登录：

```bash
ssh root@101.200.216.125
ssh -p 2200 shenwei01@localhost
```

代码仓库：

```text
/scratch/xzh/OFDFT-Hessian
```

WT 自由能全部运行数据：

```text
/scratch/xzh/OFDFT-Hessian/runs/al/free_energy_wt
```

主流水线：

```text
/scratch/xzh/OFDFT-Hessian/runs/al/free_energy_wt/melting_pipeline_pairv2_recovery_T1100_v2_node01
```

焓差采样与 Gibbs-Helmholtz 分析：

```text
/scratch/xzh/OFDFT-Hessian/runs/al/free_energy_wt/fusion_enthalpy_sampling/T0900_T0975_T1050_adaptive_T1100_pairv2_steps3000_v2_node01
```

最终 1728 原子共存验证目录：

```text
/scratch/xzh/OFDFT-Hessian/runs/al/free_energy_wt/direct_coexistence_validation/pairv2_recovery_T1100_v2_node01
```

## 3. 暂停操作

暂停时间：2026-07-23 18:34 CST。

已向当前 1100 K 固相和液相的两个 `prterun` 主进程发送 `SIGTERM`，两条 MPI 任务均正常退出。随后清除了本项目的三个 tmux 会话：

```text
wt_enthalpy_recovery
wt_coexist_validate
wt_melt_pipeline
```

停机后复查未发现以下项目进程残留：

```text
watch_wt_enthalpy_recovery_node01.sh
watch_wt_coexistence_after_recovery_node01.sh
T1100_steps3000_round2
abacus_wt_ti_cpu
```

未终止其他用户或其他项目的任务。

## 4. 已正式验证的结果

### 4.1 900 K 自由能锚点

文件：

```text
runs/al/free_energy_wt/melting_free_energy_T0900_pairv2_v2.json
```

状态：

```text
anchor_temperature_free_energy_verified
```

主要数值：

```text
Delta G_liq-solid(900 K) = +9.3391195 meV/atom
保守不确定度             = 4.2903515 meV/atom
初步熔点线性估计         = 1003.84 K
```

`Delta G_liq-solid > 0` 表示 900 K 下固相自由能较低。`1003.84 K` 只使用了初步、近似常数的熔化焓，不能作为最终熔点。

### 4.2 975 K 第二轮焓差采样

目录：

```text
critical_extensions/T0975_steps3000_round2
```

结果文件：

```text
confirmation_summary.json
```

正式状态：

```text
all_confirmations_passed
```

固相：

```text
max_step                    = 3000
后半段平均温度              = 961.032 K
后半段平均压力              = -0.963 kbar
最终最近邻                  = 2.407 A
相态                        = solid_verified
```

液相：

```text
max_step                    = 3000
后半段平均温度              = 968.176 K
后半段平均压力              = +0.749 kbar
最终最近邻                  = 2.411 A
相态                        = liquid_verified
```

### 4.3 1050 K 第二轮焓差采样

目录：

```text
critical_extensions/T1050_steps3000_round2
```

正式状态：

```text
all_confirmations_passed
```

固相：

```text
max_step                    = 3000
后半段平均温度              = 1046.987 K
后半段平均压力              = -1.272 kbar
最终最近邻                  = 2.380 A
相态                        = solid_verified
```

液相：

```text
max_step                    = 3000
后半段平均温度              = 1040.532 K
后半段平均压力              = -1.116 kbar
最终最近邻                  = 2.353 A
相态                        = liquid_verified
```

## 5. 1100 K 暂停点

目录：

```text
critical_extensions/T1100_steps3000_round2
```

来源为第一轮 1100 K 轨迹的 `step 2995`，目标是固相和液相各延长 3000 步。暂停时尚未达到目标步数。

最后完整、固液同步的坐标帧：

```text
solid MD_dump  = step 1350
liquid MD_dump = step 1350
```

热力学日志最后记录：

```text
solid max_step  = 1352
liquid max_step = 1356
```

恢复或另建续跑时应使用最后完整的 `step 1350`，不要使用仅有热力学记录而没有完整坐标的 1352/1356。

暂停后只读分析文件：

```text
solid/phase_analysis_at_pause.json
liquid/phase_analysis_at_pause.json
```

固相当前指标：

```text
平均温度                    = 1105.97 +/- 87.33 K
平均压力                    = -1.648 kbar
最终最近邻                  = 2.377 A
全轨迹最小最近邻            = 2.232 A
最终非仿射 MSD              = 0.348 A^2
后半段 MSD 斜率             = 5.93e-7 A^2/step
最终 CSP 中位数             = 3.872 A^2
最近 20 帧 CSP 中位数均值   = 3.734 A^2
最近 20 帧有序比例均值       = 0.250
```

液相当前指标：

```text
平均温度                    = 1090.24 +/- 87.69 K
平均压力                    = +0.266 kbar
最终最近邻                  = 2.308 A
全轨迹最小最近邻            = 2.129 A
最终非仿射 MSD              = 4.108 A^2
后半段 MSD 斜率             = 0.003503 A^2/step
最终 CSP 中位数             = 15.377 A^2
最近 20 帧 CSP 中位数均值   = 15.333 A^2
最近 20 帧有序比例均值       = 0
```

物理判断：当前没有原子重叠或结构塌缩，固相仍保持低扩散和明显有序，液相保持扩散和无序，两相区分正常。

注意：通用 `analyze_phase_run.py` 将本次固相标为 `solid_not_verified`，唯一失败项是 `initial_ordered_fraction_gt_0_9`。这是因为该分析器把续跑轨迹误按“从完美晶格新制备固相”的初态标准检查，而本支路本来就是从有限温度第一轮轨迹续跑；当前结构、最近邻、MSD、后段斜率和最近 CSP 门控均通过。恢复时仍需使用适用于 continuation 的正式确认脚本重新门控，不应简单把此状态解释成固相已熔化。

## 6. 第一轮统计门控及继续延长的原因

文件：

```text
discard_convergence_summary_extension_round1.json
```

900 K 第一轮已经通过：

```text
Delta H(d50)                = 82.676 meV/atom
最大 block SE               = 2.810 meV/atom
discard spread              = 1.156 meV/atom
```

975 K 需要第二轮：

```text
原因                        = d25 half drift 失败，discard sensitivity 过大
Delta H(d25/d50/d75)        = 87.905 / 90.176 / 92.486 meV/atom
discard spread              = 4.581 meV/atom
```

1050 K 需要第二轮：

```text
原因                        = 最大 block SE 4.173 > 3.0 meV/atom
Delta H(d25/d50/d75)        = 81.388 / 81.190 / 79.611 meV/atom
```

1100 K 需要第二轮：

```text
原因                        = 最大 block SE 3.498 > 3.0 meV/atom
Delta H(d25/d50/d75)        = 82.265 / 81.847 / 81.869 meV/atom
```

因此 975、1050、1100 K 第二轮的目的主要是降低统计误差和检验丢弃比例敏感性，不是重新寻找相态。

## 7. 已知失败支路和禁止事项

1050 K 第二轮最初错误地使用了第一轮 `step 2995` 固相帧。该帧处在边缘失序状态：

```text
CSP 中位数                 = 6.068 A^2
有序比例                   = 0.0926
```

该支路已停止并归档：

```text
critical_extensions/T1050_steps3000_round2_badsource2995_20260723_143647
```

后来改用健康的 `step 2950`：

```text
固相 CSP 中位数            = 4.267 A^2
固相有序比例               = 0.25
固相最近邻                 = 2.393 A
```

正确的 1050 K 第二轮已经完成并通过。禁止恢复坏源支路。

另外：

- 不要恢复旧的 MPN 直接拼接、旧 NPT 压力外推或已判失败的支路。
- 不要在 Gibbs-Helmholtz 根尚未通过统计门控前启动最终 1728 原子验证。
- 不要把 900 K 的初步线性熔点 `1003.84 K` 当作最终结果。
- 不要直接从 1100 K 日志中的 1352/1356 构造续跑，必须使用完整 `step 1350`。

## 8. 推荐恢复步骤

1. 登录 node01，确认没有本项目残留进程：

```bash
tmux ls
ps -u shenwei01 -o pid,ppid,stat,etime,pcpu,args | \
  grep -E 'wt_enthalpy|wt_coexist|T1100_steps3000_round2|abacus_wt_ti_cpu'
```

2. 先复核 1100 K 的两个 `phase_analysis_at_pause.json`，并以 `step 1350` 为固液共同源建立新的 continuation 目录。保留速度，不要重新随机化速度。

3. 将 1100 K 有效总采样延长到原计划的统计长度。续跑后必须生成正式 `confirmation_summary.json`，要求：

```text
固液都达到计划步数
最近邻 > 2 A
固相 continuation 门控通过
液相扩散/CSP 门控通过
后半段平均温度在目标容差内
后半段平均压力在零压容差内
```

4. 重新生成第二轮 975/1050/1100 K 的 d25、d50、d75 焓差报告和 discard convergence summary。至少要求：

```text
所有报告 verified
最大 block SE <= 3.0 meV/atom
discard spread <= 2.0 meV/atom
half drift 门控通过
```

5. 运行 Gibbs-Helmholtz 积分，生成：

```text
gibbs_helmholtz_melting.json
```

检查 `Delta G_liq-solid(T)` 是否只有一个物理解、不同 discard 选择的根是否稳定，并给出统计和系统不确定度。

6. Gibbs-Helmholtz 结果通过后，再启动等待中的 1728 原子直接共存验证：

```text
runs/al/free_energy_wt/direct_coexistence_validation/pairv2_recovery_T1100_v2_node01
```

最终必须用固/液区域 CSP、MSD、z 向有序度和界面迁移方向验证自由能熔点。只有自由能根和直接共存验证一致后，任务才可标记完成。

## 9. 关键脚本

```text
scripts/watch_wt_enthalpy_recovery_node01.sh
scripts/watch_wt_coexistence_after_recovery_node01.sh
scripts/launch_prepared_zero_pressure_pair_node01.sh
scripts/analyze_phase_run.py
scripts/analyze_wt_fusion_enthalpy_series.py
scripts/summarize_wt_enthalpy_convergence.py
```

代码仓库在暂停时有未提交修改。Git HEAD：

```text
fcfcddd32b2a820983bd7323750d55a672fb7779
```

完整 `git status --short` 已保存在备份的：

```text
pre_stop/git_status_short.txt
```

恢复时必须以备份中的实际工作树为准，不要只检出 Git HEAD，否则会丢失本次自由能流程新增的脚本和修改。

## 10. 备份

备份根目录：

```text
/scratch/xzh/OFDFT-Hessian_backups/wt_melting_pause_20260723_1833
```

内容：

```text
repository/runs/al/free_energy_wt/   全部 WT 原始轨迹、输入、日志和分析结果
repository/scripts/                  脚本快照
repository/mpn_melting/              Python 包快照
repository/config/                   配置快照
repository/tests/                    测试快照
repository/assets/                   所需资源快照
repository/net.pt                    模型文件
pre_stop/                            停机前后进程、tmux、Git 和步数快照
SHA256SUMS.txt                       备份文件 SHA-256 清单
rsync_verify_data.txt                运行数据 rsync 差异校验
```

备份统计：

```text
运行数据文件                 = 5321
运行数据字节数               = 4,606,987,816
备份普通文件总数             = 5629
du 显示大小                  = 4.3 GB
rsync 二次差异项             = 0
SHA-256 条目                 = 5629
```

校验示例：

```bash
cd /scratch/xzh/OFDFT-Hessian_backups/wt_melting_pause_20260723_1833
sha256sum -c SHA256SUMS.txt
```

## 11. 当前结论

已经得到可靠的 900 K 自由能锚点，并完成 975 K、1050 K 的第二轮固液焓差轨迹。1100 K 第二轮在 1350 个完整步处安全暂停，两相仍保持，但尚未达到 3000 步统计目标。

因此目前**没有经完整统计门控和直接共存验证的最终熔点**。下一位执行者应先完成或等效替代 1100 K 的统计延长，再进行 Gibbs-Helmholtz 求根和 1728 原子直接共存验证。
