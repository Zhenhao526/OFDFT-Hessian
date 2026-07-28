# QM9PBEForcePilot P1-410 Early Pilot Report

日期：2026-07-03 Asia/Singapore

## 1. Scope

本轮只使用现有 `QM9PBEForcePilot` 410 molecule 数据快照，目标是启动 P1-410 early pilot 训练和小规模 force-level evaluation。

明确约束：

- 不重新生成 P1 labels。
- 不启动 1000 molecule datagen。
- 不删除 `kohn_sham`、`labels`、cached labels 或已有 checkpoints。
- EGF force 来自 scalar energy 对坐标的导数：`F_pred = -dE_pred/dR`。
- 不加入独立 force head 作为主方案。
- 当前 410 molecule 结果只作为 early pilot，不作为最终 Hessian 结论。

## 2. Frozen Data Snapshot

数据根目录：

```text
DFT_DATA=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1
DFT_MODELS=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models
TMPDIR=/mnt/afs/home/xiazhenhao/dft/_tmp
```

当前快照计数：

| Artifact | Count |
|---|---:|
| KS `.chk` | 1640 |
| label `.zarr.zip` | 1640 |
| cached transformed label | 1640 |
| dataset statistics | 1 |

force label 检查沿用 410 smoke 的完整扫描结果：

```text
_runtime/qm9_p1/QM9PBEForcePilot/qm9_p1_410_force_check_smoke_train.json
```

关键结论：1640 个 force labels 均无 NaN/Inf，`force_norm_max=0.19067808023212568`。

split：

```text
_runtime/qm9_p1/QM9PBEForcePilot/split.pkl
_runtime/qm9_p1/QM9PBEForcePilot/reports/qm9_p1_410_split_check.json
```

split 摘要：

| Split | Molecules | Label files |
|---|---:|---:|
| train | 328 | 1312 |
| val | 41 | 164 |
| test | 41 | 164 |

无 molecule leakage。

statistics：

```text
_runtime/qm9_p1/QM9PBEForcePilot/dataset_statistics/dataset_statistics_labels_local_frames_global_symmetric_natrep_e_kin_plus_xc.zarr
```

生成记录：

```text
_runtime/qm9_p1_models/statistics/runs/006_qm9_p1_410_stats_cached_batch_size-16__num_workers-0__qm9_pbe_force_pilot__enforce_tags-false__print_config-false__overwrite-true__n_batches-16
```

## 3. Minimal Execution Plan

训练设备：优先使用 `CUDA_VISIBLE_DEVICES=1`，当前是空闲 A100 80GB。

本轮采用受控 `max_steps`，目的不是收敛，而是获得比 2-batch smoke 更真实的 early pilot loss 轨迹和 checkpoint。

| Run | Experiment | Force loss weight | Batch size | Max steps | Val interval |
|---|---|---:|---:|---:|---:|
| EG baseline | `str25/qm9_pbe_force_pilot_eg` | 0 | 32 | 300 | 100 steps |
| EGF lambda 0.01 | `str25/qm9_pbe_force_pilot_egf` | 0.01 | 4 | 300 | 100 steps |
| EGF lambda 0.1 | `str25/qm9_pbe_force_pilot_egf` | 0.1 | 4 | 300 | 100 steps |
| EGF lambda 1.0 | `str25/qm9_pbe_force_pilot_egf` | 1.0 | 4 | 300 | 100 steps |

每个 run 记录：

- `train_loss/total`
- `val_loss/total`
- `val_loss/energy_loss`
- `val_loss/gradient_loss`
- `val_loss/force_loss`，如果适用
- checkpoint path
- run dir
- wall time、CPU、max RSS

## 4. Training Commands And Outputs

共同环境变量：

```bash
TMPDIR=/mnt/afs/home/xiazhenhao/dft/_tmp
DFT_DATA=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1
DFT_MODELS=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models
CUDA_VISIBLE_DEVICES=1
```

EG baseline：

```bash
.venv/bin/python -m mldft.ml.train \
  experiment=str25/qm9_pbe_force_pilot_eg \
  name=qm9_p1_410_eg_early_pilot_s300 \
  hydra.run.dir=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_early_pilot_s300 \
  hydra.callbacks.git_logging.clean=false \
  extras.enforce_tags=false \
  extras.print_config=false \
  trainer.max_epochs=20 \
  +trainer.max_steps=300 \
  +trainer.val_check_interval=100 \
  trainer.log_every_n_steps=25 \
  data.datamodule.batch_size=32 \
  data.datamodule.num_workers=0
```

EGF lambda scan 使用同一模板，仅 `name`、`hydra.run.dir`、time file 和 `model.loss_function.force_loss.weight` 不同：

```bash
.venv/bin/python -m mldft.ml.train \
  experiment=str25/qm9_pbe_force_pilot_egf \
  name=<run_name> \
  hydra.run.dir=<run_dir> \
  hydra.callbacks.git_logging.clean=false \
  extras.enforce_tags=false \
  extras.print_config=false \
  trainer.max_epochs=20 \
  +trainer.max_steps=300 \
  +trainer.val_check_interval=100 \
  trainer.log_every_n_steps=25 \
  data.datamodule.batch_size=4 \
  data.datamodule.num_workers=0 \
  model.loss_function.force_loss.weight=<0.01|0.1|1.0>
```

训练输出：

| Run | Run dir | Checkpoint | Exit |
|---|---|---|---:|
| EG baseline | `_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_early_pilot_s300` | `checkpoints/epoch_000.ckpt`, `last.ckpt` | 0 |
| EGF lambda 0.01 | `_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam001_early_pilot_s300` | `checkpoints/epoch_000.ckpt`, `last.ckpt` | 0 |
| EGF lambda 0.1 | `_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam01_early_pilot_s300` | `checkpoints/epoch_000.ckpt`, `last.ckpt` | 0 |
| EGF lambda 1.0 | `_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_early_pilot_s300` | `checkpoints/epoch_000.ckpt`, `last.ckpt` | 0 |

checkpoint 大小：EG checkpoint 约 224570088 bytes；EGF checkpoint 约 224570920 bytes。

## 5. Training Results

TensorBoard scalar 和资源汇总：

```text
_runtime/qm9_p1_models/eval/qm9_p1_410_early_pilot/training_scalar_resource_summary.json
```

loss 初步趋势：

| Run | `train_loss/total` first -> last | `val_loss/total` first -> last | `val_loss/energy_loss` last | `val_loss/gradient_loss` last | `val_loss/force_loss` last |
|---|---:|---:|---:|---:|---:|
| EG baseline | 0.40531 -> 0.07074 | 0.86614 -> 0.86247 | 8.22548 | 0.05526 | N/A |
| EGF lambda 0.01 | 0.35873 -> 0.05803 | 0.91342 -> 0.86343 | 8.17086 | 0.05728 | 0.14480 |
| EGF lambda 0.1 | 0.25255 -> 0.08817 | 0.89614 -> 0.87902 | 8.33072 | 0.05772 | 0.00669 |
| EGF lambda 1.0 | 0.67145 -> 0.15629 | 0.91038 -> 0.88582 | 8.33753 | 0.05792 | 0.00670 |

解读：

- 四个 run 的 train loss 都从初始记录下降到 step 299，说明 300-step early pilot 有实际训练信号，不是 2-batch smoke。
- validation total loss 在 300 step 内变化较小，这是 early pilot 预期内现象，不代表收敛。
- lambda 0.01 的 validation force loss 末值升高到 0.14480；lambda 0.1 和 1.0 的 validation force loss 维持在约 0.0067。

资源占用：

| Run | Wall time | CPU | Max RSS |
|---|---:|---:|---:|
| EG baseline | 3:00.98 | 329% | 2318880 KB |
| EGF lambda 0.01 | 2:45.14 | 334% | 2292452 KB |
| EGF lambda 0.1 | 2:39.80 | 276% | 2332348 KB |
| EGF lambda 1.0 | 2:34.69 | 376% | 2292280 KB |

未在最终训练日志中发现 `ERROR`、`Traceback`、`RuntimeError`、CUDA OOM 等关键词。

已知 warning：

- PySCF B3LYP definition warning，非本轮 PBE 训练阻断项。
- PyVista future/deprecation warning，来自 mesh/image logging。
- Lightning `num_workers=0` 性能提示，本轮为降低变量显式设置。
- 个别 train metrics ground-state 子集可能出现 `nan`，主 loss 和 force eval 均正常 finite。

## 6. Force-Level Test Evaluation

评估脚本：

```text
scripts/qm9_force_eval.py
```

评估定义：

- 对 EG 和 EGF checkpoint 均临时开启 `force_supervision`。
- `F_pred = -dE_pred/dR`，不使用 force head。
- 在 test split 上读取 `metadata/pbe_derivatives/forces` 作为 PBE force label。
- 汇总 force component MAE/RMSE 和 atom vector MAE。

评估结果：

| Run | Checkpoint | Force component MAE | Force component RMSE | Force vector MAE | Failures |
|---|---|---:|---:|---:|---:|
| EG baseline | `qm9_p1_410_eg_early_pilot_s300/checkpoints/epoch_000.ckpt` | 0.114749 | 2.124800 | 0.266596 | 0 |
| EGF lambda 0.01 | `qm9_p1_410_egf_lam001_early_pilot_s300/checkpoints/epoch_000.ckpt` | 0.104810 | 0.327565 | 0.227792 | 0 |
| EGF lambda 0.1 | `qm9_p1_410_egf_lam01_early_pilot_s300/checkpoints/epoch_000.ckpt` | 0.006936 | 0.011253 | 0.014392 | 0 |
| EGF lambda 1.0 | `qm9_p1_410_egf_lam1_early_pilot_s300/checkpoints/epoch_000.ckpt` | 0.004896 | 0.009269 | 0.010417 | 0 |

评估输出：

```text
_runtime/qm9_p1_models/eval/qm9_p1_410_early_pilot/eg_force_eval.json
_runtime/qm9_p1_models/eval/qm9_p1_410_early_pilot/egf_lam001_force_eval.json
_runtime/qm9_p1_models/eval/qm9_p1_410_early_pilot/egf_lam01_force_eval.json
_runtime/qm9_p1_models/eval/qm9_p1_410_early_pilot/egf_lam1_force_eval.json
```

每个 eval 覆盖 test split 的 1821 个 SCF samples、21949 个 atoms，均无失败。

force eval 初步结论：

- EG derived force 明显不稳定，存在大 outlier；worst sample 集中在 `0000109.0000002`、`0000323.0000002`。
- EGF lambda 0.01 相比 EG 去除了极端 outlier，但整体 force MAE 仍偏大。
- EGF lambda 0.1 和 1.0 在 test force 上明显优于 EG；本轮 300-step early pilot 中 lambda 1.0 的 test force MAE 最低。
- 这只是 410 molecule / 300-step early pilot，不构成最终泛化结论。

## 7. Hessian Mini-Eval Preparation

新增流程脚本：

```text
scripts/qm9_hessian_probe.py
```

当前 label 事实：

- `configs/datagen/preset/qm9_pbe_force_pilot.yaml` 中 `compute_hessian: false`。
- 抽样 label 的 `metadata/pbe_derivatives` 只有 `forces`、`nuclear_gradient`、`dft_level`、`forces_enabled`、`hessian_enabled`。
- 当前 410 molecule label 没有 `metadata/pbe_derivatives/hessian_matrix`，因此不能做 PBE Hessian MAE 对比。

直接 autograd 二阶导 probe：

```text
_runtime/qm9_p1_models/eval/qm9_p1_410_early_pilot/hessian_probe_lam1_5mol.json
```

结果：5 个 held-out molecule 流程可执行，但 Hessian entries 出现 NaN；直接二阶 autograd 路线当前不稳定。

有限差分 fallback probe：

```text
_runtime/qm9_p1_models/eval/qm9_p1_410_early_pilot/hessian_probe_lam1_5mol_fd.json
```

设置：EGF lambda 1.0 checkpoint，5 个 test molecule，`scf=1`，中心差分位移 `1e-3`，仍使用 `F_pred=-dE_pred/dR`。

| Molecule | Natoms | Hessian shape | Finite | Symmetry max abs error | Reference Hessian |
|---|---:|---:|---|---:|---|
| `0000010` | 6 | 18 x 18 | yes | 2.27e-05 | unavailable |
| `0000019` | 9 | 27 x 27 | yes | 1.06e-04 | unavailable |
| `0000049` | 12 | 36 x 36 | yes | 7.22e-05 | unavailable |
| `0000052` | 9 | 27 x 27 | yes | 4.66e-05 | unavailable |
| `0000062` | 8 | 24 x 24 | yes | 1.29e-04 | unavailable |

有限差分 probe 资源：wall time 0:37.09，CPU 65%，max RSS 1332104 KB，exit 0。

Hessian mini-eval 结论：

- 目前可以验证模型侧 Hessian 生成流程的形状、finite 和近似对称性。
- 由于当前 P1-410 没有 PBE Hessian reference，不能报告 Hessian MAE，也不能声明 Hessian 物理结论。
- 直接二阶 autograd 路线需要单独排查 NaN；短期 Hessian mini-eval 可使用有限差分 derived-force 路线作为流程验证。

### 7.1 Force Improvement To Hessian Probe

问题：force-level MAE 改善是否会带来 Hessian 改善？

严格答案：当前不能直接从全量数据下结论，因为 P1-410 labels 没有 PBE Hessian reference。为做小例子验证，本轮补了两类测试。

#### Directional Hessian/Secant Test

脚本：

```text
scripts/qm9_hessian_directional_eval.py
```

输出：

```text
_runtime/qm9_p1_models/eval/qm9_p1_410_early_pilot/hessian_directional_5mol_eg_vs_egf.json
_runtime/qm9_p1_models/eval/qm9_p1_410_early_pilot/hessian_directional_5mol_eg_vs_egf.csv
```

定义：利用每个 molecule 已有的 4 个几何扰动，在 `scf=1` 上比较 `H*dR ~= -dF` 的方向响应。这里不是 full Hessian MAE，而是 Hessian-vector/secant proxy。数值越小越好。

样本：test split 中 `0000010`、`0000019`、`0000049`、`0000052`、`0000062`，共 15 个 perturbation pairs。

| Run | Selected force component MAE | Mean `H*dR` action MAE | Mean slope MAE | Mean directional curvature abs error |
|---|---:|---:|---:|---:|
| EG | 0.047521 | 0.048325 | 3.143598 | 2.399606 |
| EGF lambda 0.1 | 0.006913 | 0.005507 | 0.310649 | 0.291591 |
| EGF lambda 1.0 | 0.005589 | 0.004854 | 0.274642 | 0.239141 |

Directional probe 初步结论：

- 在这 5 个 held-out molecule 的 secant proxy 上，force MAE 变好同时带来了明显更好的 Hessian-vector 响应。
- EG 在 `0000062` 的一个扰动方向上出现大 outlier，导致平均 Hessian proxy 明显变差；EGF lambda 0.1/1.0 消除了这个 outlier。
- lambda 1.0 在这组例子里略优于 lambda 0.1。

#### One-Molecule Full PBE Hessian Reference

脚本：

```text
scripts/qm9_hessian_reference_eval.py
```

输出：

```text
_runtime/qm9_p1_models/eval/qm9_p1_410_early_pilot/hessian_reference_0000010_eg_vs_egf.json
_runtime/qm9_p1_models/eval/qm9_p1_410_early_pilot/pbe_hessian_0000010_0000000.npz
```

设置：`0000010.0000000`，6 atoms，`scf=1`。PBE analytic Hessian 从现有 `.chk` 恢复 SCF 后临时计算，不写回 label。单个 PBE Hessian 用时约 162 s。

| Run | Full Hessian MAE vs PBE | RMSE | Relative Fro error | Model symmetry max abs error |
|---|---:|---:|---:|---:|
| EG | 0.030058 | 0.135346 | 0.795382 | 6.57e-05 |
| EGF lambda 0.1 | 0.043611 | 0.134590 | 0.790935 | 5.12e-05 |
| EGF lambda 1.0 | 0.036853 | 0.160801 | 0.944971 | 2.21e-05 |

Full Hessian 单例结论：

- 对 `0000010.0000000` 这个单分子 full matrix，EGF 没有稳定优于 EG：lambda 0.1 的 RMSE/relative Fro 与 EG 接近，MAE 更差；lambda 1.0 的对称性更好，但 RMSE/relative Fro 更差。
- 这说明“force MAE 改善”不必然等价于“full Hessian matrix 立刻改善”，尤其当前只训练 300 steps，且 Hessian 是二阶量。
- 更可靠的说法是：EGF 已改善 force field 的局部一阶响应方向 proxy，但 full Hessian 是否改善需要更多 PBE Hessian reference 和更长训练验证。

## 8. Next Recommendations

- 在 410 early pilot 上，EGF lambda 1.0 的 test force MAE 最好；若继续 formal pilot，可优先复跑 lambda 0.3/1.0/3.0 或延长 lambda 1.0 的训练步数。
- EG baseline 可保留为对照，但当前 force-level 表现明显弱于 EGF。
- Hessian 方向下一步必须先决定是否允许生成小规模 PBE Hessian reference；否则只能做模型内部 Hessian sanity check，不能做 Hessian MAE。
- 当前结果仍是 410 molecule early pilot，不应外推为最终 P1/P2 结论。
