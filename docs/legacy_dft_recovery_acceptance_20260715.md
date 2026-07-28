# structures25 历史 Graphformer 恢复验收报告

机器报告：`/scratch/xzh/dft/restore/final_acceptance_report_20260715.json`

最终状态：**passed**。本状态覆盖恢复、无泄漏侧车划分、双密度点科学基线、代表性密度优化以及 force/Hessian 接入准备；不表示 physical total-OFDFT force/Hessian 已实现。

## 验收门

| 检查 | 结果 |
| --- | --- |
| density_optimization_representatives | PASS |
| environment_frozen | PASS |
| execution_wrappers_frozen | PASS |
| failed_attempts_retained | PASS |
| fixed_density_force_hessian_funnel | PASS |
| forbidden_path_hard_failure | PASS |
| full_content_audit | PASS |
| group_safe_splits | PASS |
| local_derivative_components_audited | PASS |
| local_derivative_unit_tests | PASS |
| models_frozen_before_test | PASS |
| multi_mode_model_acceptance | PASS |
| one_shot_test_runs | PASS |
| original_data_and_checkpoints_unchanged | PASS |
| test_caches | PASS |
| test_dual_density_coverage | PASS |
| total_derivative_plan_audited | PASS |
| validation_caches | PASS |
| validation_dual_density_coverage | PASS |
| validation_freeze_test_order | PASS |
| validation_runs | PASS |

## 数据完整性与划分

| 物理来源 | archives | SCF configs | bytes | 异常 |
| --- | --- | --- | --- | --- |
| QM9_perturbed_fock | 133,884 | 4,693,856 | 198,779,641,309 | 0 |
| QMUGS | 850 | 9,439 | 2,461,730,725 | 0 |
| QMUGS_perturbed_fock | 23,721 | 742,952 | 46,635,464,852 | 0 |

| 虚拟数据集 | train | val | test | 跨划分身份重叠 | split SHA256 |
| --- | --- | --- | --- | --- | --- |
| QM9_perturbed_fock | 107,077 | 13,402 | 13,405 | 0 | `58d012cee4f3f1817971081f6267a96591812397584636ce4d8a0c4e85eb6b99` |
| QMUGSBin0QM9_perturbed_fock | 128,266 | 15,934 | 14,255 | 0 | `c3e7fdbe9ddadc40d916a941a0db68900d2d17a6c00ba7d7f3ee7c8c7d48c0a8` |
| QMUGSBin0_perturbed_fock | 21,189 | 2,532 | 849 | 0 | `3f54320dc475c4e609e3e66a27e936547ac425366f3f952e2379c394cbd17ab2` |

原始 split 未修改。原始 QM9、QMUGS Bin0、组合数据集分别有 32、50、66 个 canonical-SMILES 跨划分；组合原始 split 还改变了 QM9 的历史 validation/test 身份，因此科学基线只使用按 canonical-SMILES/parent 连通组生成的侧车 split。精确几何、距离不变量、整条密度轨迹和单个构型哈希的跨/内划分重复均为 0。

## 恢复验收与 golden regression

| 模型 | 样本 | checkpoint SHA256 | GPU32 E/e max | GPU32 grad max | golden SHA256 |
| --- | --- | --- | --- | --- | --- |
| QM9_perturbed_fock | 21 | `9759da26660c619de9c3bbf4c2dc164343ee90e08e22b3fcdcc9682dacb9bd09` | 2.2008239e-06 | 3.5132267e-06 | `431aa8975adb484bd6f50bb92e63a1667f2789713f0bbadf64aa9ec4d984acb3` |
| QMUGSBin0_perturbed_fock | 21 | `dde9e2e940ebbfcf4c74681b3264c1add71bf3539634e1b81bacffd5bd08be32` | 2.6866504e-06 | 4.8780531e-06 | `dcfbffc23a8c291c7eef5308066e1aaa41f74c039dcf52fd5cebf4cb36672efb` |

每个模型覆盖 7 个 train/val/test、原子数/元素/系数维度代表分子和 3 个密度角色，共 21 个样本；CPU float64 重复运行、GPU float32/float64、batch size>1、电子数、系数顺序和基组元数据检查均通过。无历史逐样本预测可用，因此通过所有门后冻结 CPU float64 输出为 golden。

## 固定 validation 与一次性 Test 科学基线

每个物理标签评估 archived SCF step 6 与最终密度。能量误差按分子聚合；density-gradient 与 difference 同时提供按系数 pooled MAE/RMSE。最终密度的 difference 标签恒为零，因此必须与非平凡的 SCF-6 组分开阅读。

### validation

| 模型/数据域 | rows | E MAE | E RMSE | grad MAE | grad RMSE | diff MAE | diff RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qm9_model__combined_domain | 31,868 | 0.004174325 | 0.0098867489 | 0.016346593 | 0.03043865 | 0.0038566585 | 0.0091994559 |
| qm9_model__qm9_domain | 26,804 | 0.0012336216 | 0.0023265839 | 0.015611341 | 0.029783347 | 0.0038608392 | 0.0092317581 |
| qmugs_model__combined_domain | 31,868 | 0.0036696644 | 0.0047038232 | 0.014875609 | 0.02836948 | 0.0038566585 | 0.0091994559 |
| qmugs_model__qmugs_domain | 5,064 | 0.0029950556 | 0.0037686059 | 0.014130145 | 0.027727039 | 0.0038417255 | 0.0090831383 |

| 模型/数据域 | 密度角色 | rows | E MAE | grad MAE | diff MAE |
| --- | --- | --- | --- | --- | --- |
| qm9_model__combined_domain | ground_state_final | 15,934 | 0.0029846042 | 0.0030233014 | 0 |
| qm9_model__combined_domain | trained_perturbed_step | 15,934 | 0.0053640459 | 0.029669884 | 0.0077133169 |
| qm9_model__qm9_domain | ground_state_final | 13,402 | 0.00065106681 | 0.0021443134 | 0 |
| qm9_model__qm9_domain | trained_perturbed_step | 13,402 | 0.0018161765 | 0.029078368 | 0.0077216783 |
| qmugs_model__combined_domain | ground_state_final | 15,934 | 0.0042700719 | 0.0024882888 | 0 |
| qmugs_model__combined_domain | trained_perturbed_step | 15,934 | 0.003069257 | 0.027262928 | 0.0077133169 |
| qmugs_model__qmugs_domain | ground_state_final | 2,532 | 0.0026957435 | 0.0018988785 | 0 |
| qmugs_model__qmugs_domain | trained_perturbed_step | 2,532 | 0.0032943677 | 0.026361411 | 0.007683451 |

| 模型/数据域 | 物理来源 | rows | E MAE | grad MAE | diff MAE |
| --- | --- | --- | --- | --- | --- |
| qm9_model__combined_domain | QM9_perturbed_fock | 26,804 | 0.0012336228 | 0.015611341 | 0.0038608392 |
| qm9_model__combined_domain | QMUGS_perturbed_fock | 5,064 | 0.019739606 | 0.01897283 | 0.0038417255 |
| qm9_model__qm9_domain | QM9_perturbed_fock | 26,804 | 0.0012336216 | 0.015611341 | 0.0038608392 |
| qmugs_model__combined_domain | QM9_perturbed_fock | 26,804 | 0.0037971242 | 0.015084312 | 0.0038608392 |
| qmugs_model__combined_domain | QMUGS_perturbed_fock | 5,064 | 0.0029950134 | 0.014130144 | 0.0038417255 |
| qmugs_model__qmugs_domain | QMUGS_perturbed_fock | 5,064 | 0.0029950556 | 0.014130145 | 0.0038417255 |

| 模型/数据域 | 误差 | q50 | q90 | q99 | max |
| --- | --- | --- | --- | --- | --- |
| qm9_model__combined_domain | absolute_energy_error | 0.00094604492 | 0.014038086 | 0.040791626 | 0.12182617 |
| qm9_model__combined_domain | gradient_mae_per_coefficient | 0.024312761 | 0.031002683 | 0.034440125 | 0.069066674 |
| qm9_model__combined_domain | difference_mae_per_coefficient | 0.0027666059 | 0.0082986169 | 0.0094469597 | 0.013343275 |
| qm9_model__qm9_domain | absolute_energy_error | 0.00073242188 | 0.0028686523 | 0.0067138672 | 0.095581055 |
| qm9_model__qm9_domain | gradient_mae_per_coefficient | 0.024312761 | 0.03032438 | 0.033309757 | 0.069066674 |
| qm9_model__qm9_domain | difference_mae_per_coefficient | 0.0027666059 | 0.0083236524 | 0.009473627 | 0.013343275 |
| qmugs_model__combined_domain | absolute_energy_error | 0.0029296875 | 0.0076293945 | 0.013275146 | 0.055664062 |
| qmugs_model__combined_domain | gradient_mae_per_coefficient | 0.021897683 | 0.028626742 | 0.031306583 | 0.079943433 |
| qmugs_model__combined_domain | difference_mae_per_coefficient | 0.0027666059 | 0.0082986169 | 0.0094469597 | 0.013343275 |
| qmugs_model__qmugs_domain | absolute_energy_error | 0.0025634766 | 0.0054321289 | 0.010681152 | 0.055664062 |
| qmugs_model__qmugs_domain | gradient_mae_per_coefficient | 0.015594223 | 0.027578967 | 0.030361569 | 0.04037315 |
| qmugs_model__qmugs_domain | difference_mae_per_coefficient | 0.0029268116 | 0.00819189 | 0.0092719026 | 0.011802158 |

### test

| 模型/数据域 | rows | E MAE | E RMSE | grad MAE | grad RMSE | diff MAE | diff RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qm9_model__combined_domain | 28,510 | 0.027428092 | 0.13321207 | 0.014062396 | 0.026650899 | 0.0027568804 | 0.0077760384 |
| qm9_model__qm9_domain | 26,810 | 0.0012118202 | 0.0020402433 | 0.015604468 | 0.029771563 | 0.0038646785 | 0.0092394811 |
| qmugs_model__combined_domain | 28,510 | 0.0050742843 | 0.0098163917 | 0.012384175 | 0.024917676 | 0.0027568804 | 0.0077760384 |
| qmugs_model__qmugs_domain | 1,698 | 0.024506478 | 0.034981019 | 0.0051520403 | 0.0099077065 | 7.5245794e-05 | 0.00043553289 |

| 模型/数据域 | 密度角色 | rows | E MAE | grad MAE | diff MAE |
| --- | --- | --- | --- | --- | --- |
| qm9_model__combined_domain | ground_state_final | 14,255 | 0.026854584 | 0.0045185835 | 0 |
| qm9_model__combined_domain | trained_perturbed_step | 14,255 | 0.028001599 | 0.023606209 | 0.0055137609 |
| qm9_model__qm9_domain | ground_state_final | 13,405 | 0.00063832791 | 0.002134686 | 0 |
| qm9_model__qm9_domain | trained_perturbed_step | 13,405 | 0.0017853125 | 0.029074251 | 0.0077293569 |
| qmugs_model__combined_domain | ground_state_final | 14,255 | 0.0057505543 | 0.0034149403 | 0 |
| qmugs_model__combined_domain | trained_perturbed_step | 14,255 | 0.0043980144 | 0.021353409 | 0.0055137609 |
| qmugs_model__qmugs_domain | ground_state_final | 849 | 0.024469418 | 0.0050201205 | 0 |
| qmugs_model__qmugs_domain | trained_perturbed_step | 849 | 0.024543538 | 0.0052839601 | 0.00015049159 |

| 模型/数据域 | 物理来源 | rows | E MAE | grad MAE | diff MAE |
| --- | --- | --- | --- | --- | --- |
| qm9_model__combined_domain | QM9_perturbed_fock | 26,810 | 0.0012118031 | 0.015604468 | 0.0038646785 |
| qm9_model__combined_domain | QMUGS | 1,700 | 0.44087438 | 0.010329604 | 7.5306746e-05 |
| qm9_model__qm9_domain | QM9_perturbed_fock | 26,810 | 0.0012118202 | 0.015604468 | 0.0038646785 |
| qmugs_model__combined_domain | QM9_perturbed_fock | 26,810 | 0.003841397 | 0.015371435 | 0.0038646785 |
| qmugs_model__combined_domain | QMUGS | 1,700 | 0.024517643 | 0.0051531103 | 7.5306746e-05 |
| qmugs_model__qmugs_domain | QMUGS | 1,698 | 0.024506478 | 0.0051520403 | 7.5245794e-05 |

| 模型/数据域 | 误差 | q50 | q90 | q99 | max |
| --- | --- | --- | --- | --- | --- |
| qm9_model__combined_domain | absolute_energy_error | 0.00079345703 | 0.0040893555 | 0.80872803 | 1.1850586 |
| qm9_model__combined_domain | gradient_mae_per_coefficient | 0.0097757098 | 0.030240867 | 0.033052366 | 0.045675531 |
| qm9_model__combined_domain | difference_mae_per_coefficient | 9.1593738e-06 | 0.0082940421 | 0.0095105841 | 0.013224934 |
| qm9_model__qm9_domain | absolute_energy_error | 0.00073242188 | 0.0028686523 | 0.0065307617 | 0.067138672 |
| qm9_model__qm9_domain | gradient_mae_per_coefficient | 0.020223718 | 0.030310938 | 0.033136001 | 0.045675505 |
| qm9_model__qm9_domain | difference_mae_per_coefficient | 0.0027576785 | 0.008333159 | 0.009535961 | 0.013224934 |
| qmugs_model__combined_domain | absolute_energy_error | 0.0032348633 | 0.0092773438 | 0.046364746 | 0.12646484 |
| qmugs_model__combined_domain | gradient_mae_per_coefficient | 0.0053020609 | 0.029208117 | 0.032167334 | 0.051328693 |
| qmugs_model__combined_domain | difference_mae_per_coefficient | 9.1593738e-06 | 0.0082940421 | 0.0095105841 | 0.013224934 |
| qmugs_model__qmugs_domain | absolute_energy_error | 0.015625 | 0.063232422 | 0.1074292 | 0.12646484 |
| qmugs_model__qmugs_domain | gradient_mae_per_coefficient | 0.0048356298 | 0.00629193 | 0.0079087458 | 0.010634218 |
| qmugs_model__qmugs_domain | difference_mae_per_coefficient | 9.1593738e-06 | 0.00023538874 | 0.00039594658 | 0.00057062507 |

完整机器报告还包含按物理来源、原子数和元素组成的分组，误差分位数，以及每项指标 top-100 异常样本；逐样本结果保存在各 run 的 `rows/shard_*.jsonl.gz`，其 SHA256 在报告中冻结。所有模型在 validation 完成并写入 freeze artifact 后才执行一次 Test 前向。

## 模型驱动密度优化（validation 代表集）

| 模型域 | 任务 | 严格收敛 | 成功率 | cycles median/max | 最终残差 max | 电子数误差 max | 耗时 max(s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| QM9_perturbed_fock | 3 | 3 | 1 | 337/510 | 9.8727381e-05 | 1.1368684e-13 | 26.138658 |
| QMUGSBin0_perturbed_fock | 3 | 2 | 0.66666667 | 414/10128 | 0.061616859 | 3.6948222e-13 | 876.51414 |

QMUGS `0987810.zarr.zip` 是明确的科学非收敛：stage 1 后残差约 0.00957，stage 2 达到 10000-cycle 上限并以 0.0616 结束；日志和逐样本曲线保留。报告严格区分标签密度前向能量误差、优化后密度能量误差和系数/密度误差。

## force/Hessian 接入边界

| 层级 | 关键结果 |
| --- | --- |
| 固定密度 validation smoke | full Hessian 1/1 finite；HVP 4/4 finite；h=0.001 Bohr |
| weight=1 validation Tier2 | E/force/relaxed-proxy H MAE=0.023098984/0.0024957682/0.01337224；strict 858/858 |
| frozen Test100 fixed | autograd-vs-FD MAE=2.1133813e-06；fixed-H PBE MAE=0.0093163355；HVP/PBE-force-secant MAE=0.019276056 |
| frozen Test100 relaxed proxy | 100/100 Hessians；strict 10638/10638；MAE=0.010290859 |

恢复的历史 Graphformer 只预测能量、密度梯度和 density difference，不是 force/Hessian 模型。固定密度结果只微分 learned scalar energy，缺少密度响应、classical integral/nuclear 与 Pulay 坐标导数；密度松弛结果是 incomplete-derived-force proxy，也不是 physical total-OFDFT Hessian。单位为 Hartree、Bohr、Hartree/Bohr、Hartree/Bohr²。

## 保守 total-OFDFT 导数实施门

0. **freeze interfaces and reporting boundaries** — Tensor scalar energy retains grad_fn and passes first/second coordinate derivative smoke tests in float64.
   - Keep legacy float/Energies APIs for logs only and add an explicit tensor-returning scalar-energy API.
   - Make .item(), float(), NumPy, detach and CPU transfers fail dynamic graph tests when reached before the reporting boundary.
   - Version and hash all integral providers, basis metadata, checkpoint and validation geometry lists.
1. **fixed-coefficient total scalar energy** — Every component and the sum agree with central finite differences over a validation-only small-molecule set and several step sizes.
   - Assemble learned kinetic+XC, Hartree 0.5*c^T*J*c, electron-nuclear c^T*v_ext and nuclear-nuclear repulsion as tensors.
   - Keep electron normalization q(R)^T c=N in tensor form and validate coefficient/basis ordering.
2. **moving-integral and Pulay response** — Component VJP/JVP adjoint consistency plus full fixed-density force agreement with total scalar-energy central differences.
   - Provide coordinate JVP/VJP for Coulomb J(R), nuclear attraction v_ext(R), overlap/metric S(R), dual integrals q(R), transformations and E_nn(R).
   - Use analytic derivatives or audited custom autograd; if finite-difference-backed, expose truncation/error controls and adjoint tests.
   - Include moving auxiliary basis/local frames and all Pulay terms rather than treating cached tensors as constants.
3. **constrained relaxed total force** — Relaxed force matches fully reoptimized E*(R+h)-E*(R-h) and closed-loop work is step-convergent toward zero on validation geometries.
   - Solve stationarity of L(c,mu,R)=E(c,R)+mu*(q(R)^T*c-N) in the electron-number tangent space.
   - Use the envelope derivative -partial_R L at a converged density and include q(R), normalization and Pulay response.
   - Separate solver failure, nonstationary residual and derivative failure in reports.
4. **implicit density response and Hessian-vector product** — KKT residual, tangent electron constraint, HVP finite-difference agreement and symmetry all pass preregistered validation tolerances.
   - Build the KKT/tangent operator [L_cc q; q^T 0] and solve its directional response with dense reference then MINRES/PCG-compatible blocks.
   - Combine direct coordinate curvature, mixed c-R blocks, q(R) derivatives and implicit dc/dR without unrolling optimizer branches.
   - Check HVP against finite differences of the conservative total force and verify Hessian symmetry.
5. **physical interpretation gate** — Until all previous gates pass, label outputs as fixed-density or incomplete-derived-force proxies, never physical total-OFDFT Hessians.
   - Freeze implementation and validation tolerances before one test evaluation.
   - Only after translation/rotation response, conservation, units and mass weighting pass may physical vibrational quantities be reported.

本地候选组件测试：25 passed，errors=0，failures=0。远端恢复快照尚未部署这些接口；在 clean code-release audit 前保持 `planned_not_deployed_on_recovery_server`。在中心差分、闭合回路、KKT/HVP、平移/转动及单位门全部通过前，不给出物理振动结论。

## 失败任务与计算成本

| 任务 | 状态 | 原因 | 替代/结论 |
| --- | --- | --- | --- |
| content_merge_noncovering_index_job575 | cancelled | SQLite merge used non-covering random reads and did not finish at full scale. | Rebuilt a covering index; replacement merge job 664 passed. |
| acceptance_batch_attempt1_jobs642_643 | failed | Variable-size square preprocessing tensors were included in PyG batching. | Excluded consumed overlap matrices from model-input batching. |
| acceptance_batch_attempt2_jobs648_649 | failed | The single-sample diagnostic tried to convert a three-graph batch error to a scalar. | Separated per-sample diagnostics from batched consistency checks. |
| acceptance_absolute_energy_gate_job662 | failed_gate | A fixed absolute tolerance was inappropriate for extensive total energy. | Predeclared relative and per-electron float32 gates; checkpoint outputs were not changed. |
| baseline_calibration_variable_matrix_batch | failed | Calibration loader attempted to concatenate variable-size overlap matrices. | Excluded preprocessing-only matrices in the production evaluator. |
| raw_validation_transform_job667 | cancelled | Four model/domain runs redundantly rebuilt expensive natural-representation transforms and produced no first batch. | Created read-only-source transformed caches and verified raw-to-cache equivalence. |
| remote_total_derivative_unit_job663 | failed_preflight | Recovery-server code snapshot does not contain the new total-derivative test modules. | Ran the candidate implementation locally (25 passed) and explicitly left remote deployment pending code-release audit. |
| test_cache_gpu_transform_smoke_job760 | cancelled | The largest 216-atom sample remained CPU/PySCF integral-build bound before reaching the GPU eigensolver. | Retained the CPU-sharded cache path; no scientific metric was produced. |
| final_only_test_jobs704_705 | cancelled_before_execution | The full-split final-density-only difference target is identically zero and was not adequate scientific coverage. | Cancelled before any forward pass; replaced by frozen SCF-6 plus final-density Test jobs. |
| concurrent_scf6_cache_job761 | cancelled_before_sample_output | Concurrent raw transform arrays contended on the shared filesystem: about 20 minutes elapsed, about 7 CPU seconds accrued, and zero samples were written. | Preserved sacct/task logs and resubmitted the cache/baseline/Test funnel serially after the final-density cache. |
| superseded_final_report_job795 | cancelled_before_execution | Slurm had frozen an earlier final-report wrapper before immutability and completion-audit stages were added. | Cancelled at zero elapsed time and resubmitted the current wrapper as job 796 with the same afterok:794 scientific dependency. |
| test_qm9_combined_batch64_oom_job793_tasks16_23 | failed_infrastructure_no_summary | All eight qm9_model__combined_domain shards exceeded one 80-GB A100 at batch_size=64 on the large QMUGS Test graphs; no shard summary or accepted row artifact was produced. | Archived the OOM logs and reran only the eight missing shards with the same frozen checkpoint, split, SCF selectors, float32 dtype, cache and metrics at batch_size=1. Computational batching is not a model or metric change. |

上述任务均标记 `used_as_scientific_result=false`，保留日志 SHA256。baseline 报告中的 `cost` 给出各 shard 设备秒总和、批次数与并行 wall-time 估计；密度优化与 Test100 proxy 另存逐任务 cycles/耗时。

## 可复现命令与路径策略

所有入口只接受 `DFT_DATA=/scratch/xzh/dft/data`、`DFT_MODELS=/scratch/xzh/dft/models` 或等价显式参数。`hparams_resolved.yaml` 中 `/export/scratch/ialgroup` 仅作为 provenance 记录；若成为有效路径，程序立即报错。代表性 baseline shard 命令如下，其余 shard 只改变 shard index：

`validation/qm9_model__combined_domain`

```bash
/scratch/xzh/envs/structures25/bin/python /scratch/xzh/dft/restore/legacy_scientific_baseline_shard.py --code-root /scratch/xzh/code/structures25 --data-root /scratch/xzh/dft/data --models-root /scratch/xzh/dft/models --hparams '/scratch/xzh/dft/models/train/runs/088__from_checkpoint_009__str25\qm9_tf/hparams.yaml' --dataset-name QMUGSBin0QM9_perturbed_fock --split-file /scratch/xzh/dft/restore/group_safe_splits_20260715/QMUGSBin0QM9_perturbed_fock/split.pkl --partition val --shard-index 0 --num-shards 8 --device cuda:0 --dtype float32 --batch-size 64 --num-workers 4 --scf-iterations=6,-1 --preprocessed-cache-root /scratch/xzh/dft/restore/baseline_cache_validation_20260715/samples --output-jsonl-gz /scratch/xzh/dft/restore/scientific_baseline_validation_full_20260715/qm9_model__combined_domain/rows/shard_0.jsonl.gz --output-summary-json /scratch/xzh/dft/restore/scientific_baseline_validation_full_20260715/qm9_model__combined_domain/summaries/shard_0.json --progress-interval 100
```

`validation/qm9_model__qm9_domain`

```bash
/scratch/xzh/envs/structures25/bin/python /scratch/xzh/dft/restore/legacy_scientific_baseline_shard.py --code-root /scratch/xzh/code/structures25 --data-root /scratch/xzh/dft/data --models-root /scratch/xzh/dft/models --hparams '/scratch/xzh/dft/models/train/runs/088__from_checkpoint_009__str25\qm9_tf/hparams.yaml' --dataset-name QM9_perturbed_fock --split-file /scratch/xzh/dft/restore/group_safe_splits_20260715/QM9_perturbed_fock/split.pkl --partition val --shard-index 0 --num-shards 8 --device cuda:0 --dtype float32 --batch-size 64 --num-workers 4 --scf-iterations=6,-1 --preprocessed-cache-root /scratch/xzh/dft/restore/baseline_cache_validation_20260715/samples --output-jsonl-gz /scratch/xzh/dft/restore/scientific_baseline_validation_full_20260715/qm9_model__qm9_domain/rows/shard_0.jsonl.gz --output-summary-json /scratch/xzh/dft/restore/scientific_baseline_validation_full_20260715/qm9_model__qm9_domain/summaries/shard_0.json --progress-interval 100
```

`validation/qmugs_model__combined_domain`

```bash
/scratch/xzh/envs/structures25/bin/python /scratch/xzh/dft/restore/legacy_scientific_baseline_shard.py --code-root /scratch/xzh/code/structures25 --data-root /scratch/xzh/dft/data --models-root /scratch/xzh/dft/models --hparams '/scratch/xzh/dft/models/train/runs/214__num_workers-32__qmugs_bin0_perturbed_fock__str25\qmugs_hard_cutoff_hierarc_tf__lr-1e-5__max_epochs-30__from_weight_checkpoint_110/hparams.yaml' --dataset-name QMUGSBin0QM9_perturbed_fock --split-file /scratch/xzh/dft/restore/group_safe_splits_20260715/QMUGSBin0QM9_perturbed_fock/split.pkl --partition val --shard-index 0 --num-shards 8 --device cuda:0 --dtype float32 --batch-size 64 --num-workers 4 --scf-iterations=6,-1 --preprocessed-cache-root /scratch/xzh/dft/restore/baseline_cache_validation_20260715/samples --output-jsonl-gz /scratch/xzh/dft/restore/scientific_baseline_validation_full_20260715/qmugs_model__combined_domain/rows/shard_0.jsonl.gz --output-summary-json /scratch/xzh/dft/restore/scientific_baseline_validation_full_20260715/qmugs_model__combined_domain/summaries/shard_0.json --progress-interval 100
```

`validation/qmugs_model__qmugs_domain`

```bash
/scratch/xzh/envs/structures25/bin/python /scratch/xzh/dft/restore/legacy_scientific_baseline_shard.py --code-root /scratch/xzh/code/structures25 --data-root /scratch/xzh/dft/data --models-root /scratch/xzh/dft/models --hparams '/scratch/xzh/dft/models/train/runs/214__num_workers-32__qmugs_bin0_perturbed_fock__str25\qmugs_hard_cutoff_hierarc_tf__lr-1e-5__max_epochs-30__from_weight_checkpoint_110/hparams.yaml' --dataset-name QMUGSBin0_perturbed_fock --split-file /scratch/xzh/dft/restore/group_safe_splits_20260715/QMUGSBin0_perturbed_fock/split.pkl --partition val --shard-index 0 --num-shards 8 --device cuda:0 --dtype float32 --batch-size 64 --num-workers 4 --scf-iterations=6,-1 --preprocessed-cache-root /scratch/xzh/dft/restore/baseline_cache_validation_20260715/samples --output-jsonl-gz /scratch/xzh/dft/restore/scientific_baseline_validation_full_20260715/qmugs_model__qmugs_domain/rows/shard_0.jsonl.gz --output-summary-json /scratch/xzh/dft/restore/scientific_baseline_validation_full_20260715/qmugs_model__qmugs_domain/summaries/shard_0.json --progress-interval 100
```

`test/qm9_model__combined_domain`

```bash
/scratch/xzh/envs/structures25/bin/python /scratch/xzh/dft/restore/legacy_scientific_baseline_shard.py --code-root /scratch/xzh/code/structures25 --data-root /scratch/xzh/dft/data --models-root /scratch/xzh/dft/models --hparams '/scratch/xzh/dft/models/train/runs/088__from_checkpoint_009__str25\qm9_tf/hparams.yaml' --dataset-name QMUGSBin0QM9_perturbed_fock --split-file /scratch/xzh/dft/restore/group_safe_splits_20260715/QMUGSBin0QM9_perturbed_fock/split.pkl --partition test --shard-index 0 --num-shards 8 --device cuda:0 --dtype float32 --batch-size 1 --num-workers 4 --scf-iterations=6,-1 --preprocessed-cache-root /scratch/xzh/dft/restore/baseline_cache_test_20260715/samples --output-jsonl-gz /scratch/xzh/dft/restore/scientific_baseline_test_full_20260715/qm9_model__combined_domain/rows/shard_0.jsonl.gz --output-summary-json /scratch/xzh/dft/restore/scientific_baseline_test_full_20260715/qm9_model__combined_domain/summaries/shard_0.json --progress-interval 500
```

`test/qm9_model__qm9_domain`

```bash
/scratch/xzh/envs/structures25/bin/python /scratch/xzh/dft/restore/legacy_scientific_baseline_shard.py --code-root /scratch/xzh/code/structures25 --data-root /scratch/xzh/dft/data --models-root /scratch/xzh/dft/models --hparams '/scratch/xzh/dft/models/train/runs/088__from_checkpoint_009__str25\qm9_tf/hparams.yaml' --dataset-name QM9_perturbed_fock --split-file /scratch/xzh/dft/restore/group_safe_splits_20260715/QM9_perturbed_fock/split.pkl --partition test --shard-index 0 --num-shards 8 --device cuda:0 --dtype float32 --batch-size 64 --num-workers 4 --scf-iterations=6,-1 --preprocessed-cache-root /scratch/xzh/dft/restore/baseline_cache_test_20260715/samples --output-jsonl-gz /scratch/xzh/dft/restore/scientific_baseline_test_full_20260715/qm9_model__qm9_domain/rows/shard_0.jsonl.gz --output-summary-json /scratch/xzh/dft/restore/scientific_baseline_test_full_20260715/qm9_model__qm9_domain/summaries/shard_0.json --progress-interval 100
```

`test/qmugs_model__combined_domain`

```bash
/scratch/xzh/envs/structures25/bin/python /scratch/xzh/dft/restore/legacy_scientific_baseline_shard.py --code-root /scratch/xzh/code/structures25 --data-root /scratch/xzh/dft/data --models-root /scratch/xzh/dft/models --hparams '/scratch/xzh/dft/models/train/runs/214__num_workers-32__qmugs_bin0_perturbed_fock__str25\qmugs_hard_cutoff_hierarc_tf__lr-1e-5__max_epochs-30__from_weight_checkpoint_110/hparams.yaml' --dataset-name QMUGSBin0QM9_perturbed_fock --split-file /scratch/xzh/dft/restore/group_safe_splits_20260715/QMUGSBin0QM9_perturbed_fock/split.pkl --partition test --shard-index 0 --num-shards 8 --device cuda:0 --dtype float32 --batch-size 64 --num-workers 4 --scf-iterations=6,-1 --preprocessed-cache-root /scratch/xzh/dft/restore/baseline_cache_test_20260715/samples --output-jsonl-gz /scratch/xzh/dft/restore/scientific_baseline_test_full_20260715/qmugs_model__combined_domain/rows/shard_0.jsonl.gz --output-summary-json /scratch/xzh/dft/restore/scientific_baseline_test_full_20260715/qmugs_model__combined_domain/summaries/shard_0.json --progress-interval 100
```

`test/qmugs_model__qmugs_domain`

```bash
/scratch/xzh/envs/structures25/bin/python /scratch/xzh/dft/restore/legacy_scientific_baseline_shard.py --code-root /scratch/xzh/code/structures25 --data-root /scratch/xzh/dft/data --models-root /scratch/xzh/dft/models --hparams '/scratch/xzh/dft/models/train/runs/214__num_workers-32__qmugs_bin0_perturbed_fock__str25\qmugs_hard_cutoff_hierarc_tf__lr-1e-5__max_epochs-30__from_weight_checkpoint_110/hparams.yaml' --dataset-name QMUGSBin0_perturbed_fock --split-file /scratch/xzh/dft/restore/group_safe_splits_20260715/QMUGSBin0_perturbed_fock/split.pkl --partition test --shard-index 0 --num-shards 8 --device cuda:0 --dtype float32 --batch-size 64 --num-workers 4 --scf-iterations=6,-1 --preprocessed-cache-root /scratch/xzh/dft/restore/baseline_cache_test_20260715/samples --output-jsonl-gz /scratch/xzh/dft/restore/scientific_baseline_test_full_20260715/qmugs_model__qmugs_domain/rows/shard_0.jsonl.gz --output-summary-json /scratch/xzh/dft/restore/scientific_baseline_test_full_20260715/qmugs_model__qmugs_domain/summaries/shard_0.json --progress-interval 100
```

## 结论与下一步

1. 恢复、数据审计、侧车无泄漏划分和历史模型验收已冻结；不需要因工程恢复而重训。
2. 先处理 QMUGS 优化 hard case，并仅在 validation 上冻结任何 solver 策略；Test 不用于调参。
3. 将本地 tensor-energy/integral/Pulay/KKT 候选通过 clean release 部署远端，按 stage 0–5 逐门验证。
4. 保持 fixed-density 与 incomplete-derived-force proxy 的命名边界；conservative total force 通过完全松弛总能量中心差分和闭合回路后，才进入 implicit Hessian 与物理振动。
5. 只有上述 validation 门冻结后才决定是否启动全量重训；当前没有科学依据立即重训历史 Graphformer。
