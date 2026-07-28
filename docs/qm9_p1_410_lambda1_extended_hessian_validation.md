# QM9PBEForcePilot P1-410 Lambda1 Extended Hessian Validation

日期：2026-07-03 Asia/Singapore

## 1. Scope

本报告只覆盖现有 `QM9PBEForcePilot` P1-410 early pilot 数据快照，以及 `EG` vs `EGF lambda=1.0` 从 s1000 延长到 s3000 后的小规模 force/Hessian 验证。

约束保持不变：

- 不生成 1000 molecule labels。
- 不删除现有 `.chk`、labels、cached labels、checkpoints 或 Hessian references。
- 不添加独立 force head。
- force 仍由 scalar energy 派生：`F_pred = -dE_pred/dR`。
- 结论只限 P1-410 early pilot，不外推到最终 P1/P2。

## 2. Data Snapshot

数据环境：

```text
DFT_DATA=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1
DFT_MODELS=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models
TMPDIR=/mnt/afs/home/xiazhenhao/dft/_tmp
```

沿用已验证快照：

| Artifact | Count / Result |
|---|---:|
| molecules | 410 |
| KS `.chk` | 1640 |
| raw label `.zarr.zip` | 1640 |
| cached transformed label `.zarr.zip` | 1640 |
| force label finite check | 1640/1640 pass |
| split train / val / test molecules | 328 / 41 / 41 |
| split train / val / test labels | 1312 / 164 / 164 |

本轮没有做任何新的 labelgen，也没有写回原始 labels。

## 3. s3000 Training Runs

| Run | Loss | Resume checkpoint | Run dir | s3000 checkpoint | Max steps | Batch |
|---|---|---|---|---|---:|---:|
| EG_s3000 | energy + density-gradient | `_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_warmstart_s1000/checkpoints/epoch_001.ckpt` | `_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200` | `checkpoints/epoch_006.ckpt` | 3000 | 32 |
| EGF_lam1_s3000 | energy + density-gradient + force | `_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_warmstart_s1000/checkpoints/epoch_000.ckpt` | `_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000` | `checkpoints/epoch_000.ckpt` | 3000 | 4 |

EG command：

```bash
TMPDIR=/mnt/afs/home/xiazhenhao/dft/_tmp \
DFT_DATA=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1 \
DFT_MODELS=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models \
CUDA_VISIBLE_DEVICES=1 \
/usr/bin/time -v -o _runtime/qm9_p1_models/train/qm9_p1_410_eg_resume_s3000_vci200_time.txt \
.venv/bin/python -m mldft.ml.train \
  experiment=str25/qm9_pbe_force_pilot_eg \
  name=qm9_p1_410_eg_resume_s3000_vci200 \
  hydra.run.dir=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200 \
  hydra.callbacks.git_logging.clean=false \
  extras.enforce_tags=false \
  extras.print_config=false \
  ckpt_path=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_warmstart_s1000/checkpoints/epoch_001.ckpt \
  trainer.max_epochs=20 \
  +trainer.max_steps=3000 \
  +trainer.val_check_interval=200 \
  trainer.log_every_n_steps=100 \
  data.datamodule.batch_size=32 \
  data.datamodule.num_workers=0 \
  callbacks.rich_progress_bar=null
```

EGF lambda=1.0 command：

```bash
TMPDIR=/mnt/afs/home/xiazhenhao/dft/_tmp \
DFT_DATA=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1 \
DFT_MODELS=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models \
CUDA_VISIBLE_DEVICES=1 \
/usr/bin/time -v -o _runtime/qm9_p1_models/train/qm9_p1_410_egf_lam1_resume_s3000_time.txt \
.venv/bin/python -m mldft.ml.train \
  experiment=str25/qm9_pbe_force_pilot_egf \
  name=qm9_p1_410_egf_lam1_resume_s3000 \
  hydra.run.dir=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000 \
  hydra.callbacks.git_logging.clean=false \
  extras.enforce_tags=false \
  extras.print_config=false \
  ckpt_path=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_warmstart_s1000/checkpoints/epoch_000.ckpt \
  trainer.max_epochs=20 \
  +trainer.max_steps=3000 \
  +trainer.val_check_interval=500 \
  trainer.log_every_n_steps=100 \
  data.datamodule.batch_size=4 \
  data.datamodule.num_workers=0 \
  model.loss_function.force_loss.weight=1.0 \
  callbacks.rich_progress_bar=null
```

训练 loss 取自 TensorBoard event 文件末次 scalar：

| Run | train total | train energy | train gradient | train force | val total | val energy | val gradient | val force |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EG_s3000 | 0.027667 | 0.048564 | 0.025346 | N/A | 0.814533 | 7.959032 | 0.031874 | N/A |
| EGF_lam1_s3000 | 0.043545 | 0.123318 | 0.033777 | 0.004192 | 0.844071 | 8.087317 | 0.040892 | 0.003551 |

资源：

| Run | Wall time | Max RSS | CPU |
|---|---:|---:|---:|
| EG_s3000 train | 13:56.59 | 2461244 KB | 207% |
| EGF_lam1_s3000 train | 5:51.23 | 2368652 KB | 351% |

注意事项：

- 第一次 EG resume 曾因 Hydra struct override 写成 `trainer.max_steps=3000` 失败；正确写法为 `+trainer.max_steps=3000`。
- 第二次 EG resume 曾因 `val_check_interval=500` 大于 EG 每 epoch train batch 数 `450` 失败；最终用 `+trainer.val_check_interval=200`。
- 这些失败没有生成有效 checkpoint，也没有影响数据或最终 run。

## 4. Force Test Evaluation

评估脚本：

```text
scripts/qm9_force_eval.py
```

评估对象：

- test split 展开后 `1821` SCF samples / `21949` atoms。
- failures：0。
- 参考力来自 label 中 PBE force。
- 预测力来自 scalar energy 的坐标梯度：`F_pred = -dE_pred/dR`。

输出：

```text
_runtime/qm9_p1_models/eval/qm9_p1_410_lambda1_extended_hessian_validation/eg_s3000_force_eval.json
_runtime/qm9_p1_models/eval/qm9_p1_410_lambda1_extended_hessian_validation/egf_lam1_s3000_force_eval.json
```

s1000 vs s3000：

| Run | Force component MAE | Force component RMSE | Force vector MAE | Max component abs error |
|---|---:|---:|---:|---:|
| EG_s1000 | 0.084362 | 0.335655 | 0.180492 | 8.938153 |
| EG_s3000 | 0.064490 | 0.097535 | 0.135681 | 1.031316 |
| EGF_lam1_s1000 | 0.003014 | 0.005467 | 0.006292 | 0.075387 |
| EGF_lam1_s3000 | 0.003082 | 0.005146 | 0.006439 | 0.047706 |

force 结论：

- EG 从 s1000 到 s3000 有改善，但仍显著差于 EGF。
- EGF lambda=1.0 从 s1000 到 s3000 的 component MAE 基本持平，RMSE 和 max error 有改善。
- s3000 上，EGF lambda=1.0 相比 EG 的 force component MAE 约降低 `95.22%`，约 `20.9x`。

## 5. 10 Molecule Hessian Displacement Sensitivity

使用已有 10 molecule PBE Hessian reference：

```text
_runtime/qm9_p1_models/eval/qm9_p1_410_hessian_ref_benchmark/pbe_hessian_reference_manifest.json
```

对 s3000 checkpoint 做 finite difference of derived force，测试 displacement：

- `5e-4`
- `1e-3`
- `2e-3`

输出：

```text
_runtime/qm9_p1_models/eval/qm9_p1_410_lambda1_extended_hessian_validation/full_hessian_s3000_10mol_disp5e-4.json
_runtime/qm9_p1_models/eval/qm9_p1_410_lambda1_extended_hessian_validation/full_hessian_s3000_10mol_disp1e-3.json
_runtime/qm9_p1_models/eval/qm9_p1_410_lambda1_extended_hessian_validation/full_hessian_s3000_10mol_disp2e-3.json
```

结果：

| Displacement | Run | N success | N failed | Hessian MAE | Hessian RMSE | Relative Fro | Symmetry max abs |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5e-4 | EG_s3000 | 10 | 0 | 0.418629 | 1.976236 | 12.800830 | 4.9996e-03 |
| 5e-4 | EGF_lam1_s3000 | 10 | 0 | 0.019493 | 0.054171 | 0.392697 | 3.0710e-04 |
| 1e-3 | EG_s3000 | 10 | 0 | 0.418413 | 1.975012 | 12.793331 | 3.1910e-03 |
| 1e-3 | EGF_lam1_s3000 | 10 | 0 | 0.019495 | 0.054172 | 0.392700 | 9.7047e-05 |
| 2e-3 | EG_s3000 | 10 | 0 | 0.417714 | 1.970926 | 12.767760 | 9.8904e-03 |
| 2e-3 | EGF_lam1_s3000 | 10 | 0 | 0.019493 | 0.054166 | 0.392665 | 8.3332e-05 |

sensitivity 结论：

- 三档 displacement 下，EGF lambda=1.0 的 Hessian MAE/RMSE/relative Fro 都稳定优于 EG。
- EGF 的数值对 displacement 不敏感；MAE 维持在约 `0.01949`，RMSE 维持在约 `0.05417`。
- 因优势稳定，继续扩展到 20 molecule reference set。

## 6. 20 Molecule PBE Hessian Reference Set

从 test split 中按小分子优先扩展到 20 个 `sample_id=0` molecule。前 10 个复用已有 `.npz` cache，后 10 个新算；manifest 单独保存，没有覆盖原 10 molecule manifest。

命令：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/usr/bin/time -v -o _runtime/qm9_p1_models/eval/qm9_p1_410_lambda1_extended_hessian_validation/pbe_hessian_reference_set_20_time.txt \
.venv/bin/python scripts/qm9_pbe_hessian_reference_set.py \
  --dataset-dir _runtime/qm9_p1/QM9PBEForcePilot \
  --output-dir _runtime/qm9_p1_models/eval/qm9_p1_410_hessian_ref_benchmark/pbe_hessians \
  --manifest-json _runtime/qm9_p1_models/eval/qm9_p1_410_lambda1_extended_hessian_validation/pbe_hessian_reference_manifest_20.json \
  --manifest-csv _runtime/qm9_p1_models/eval/qm9_p1_410_lambda1_extended_hessian_validation/pbe_hessian_reference_manifest_20.csv \
  --max-molecules 20 \
  --sample-id 0 \
  --workers 2
```

输出：

```text
_runtime/qm9_p1_models/eval/qm9_p1_410_lambda1_extended_hessian_validation/pbe_hessian_reference_manifest_20.json
_runtime/qm9_p1_models/eval/qm9_p1_410_lambda1_extended_hessian_validation/pbe_hessian_reference_manifest_20.csv
_runtime/qm9_p1_models/eval/qm9_p1_410_hessian_ref_benchmark/pbe_hessians/
```

manifest 汇总：

| Metric | Value |
|---|---:|
| requested | 20 |
| success | 20 |
| failed | 0 |
| finite | 20 |
| cached | 10 |
| newly computed | 10 |
| wall time | 1:50:49 |
| max RSS | 2428300 KB |

reference molecules：

| Molecule | Natoms | Status | Elapsed s | Finite |
|---|---:|---|---:|---|
| 0000010 | 6 | cached | 0.008 | yes |
| 0000323 | 6 | cached | 0.008 | yes |
| 0000109 | 7 | cached | 0.002 | yes |
| 0000171 | 7 | cached | 0.002 | yes |
| 0000062 | 8 | cached | 0.002 | yes |
| 0000116 | 8 | cached | 0.002 | yes |
| 0000144 | 8 | cached | 0.002 | yes |
| 0000019 | 9 | cached | 0.002 | yes |
| 0000052 | 9 | cached | 0.001 | yes |
| 0000350 | 9 | cached | 0.002 | yes |
| 0000192 | 11 | new | 1294.108 | yes |
| 0000328 | 11 | new | 1267.825 | yes |
| 0000341 | 11 | new | 1233.177 | yes |
| 0000348 | 11 | new | 1164.127 | yes |
| 0000355 | 11 | new | 1242.817 | yes |
| 0000049 | 12 | new | 1197.917 | yes |
| 0000064 | 12 | new | 1226.620 | yes |
| 0000310 | 12 | new | 1561.607 | yes |
| 0000340 | 12 | new | 1513.801 | yes |
| 0000366 | 12 | new | 1381.389 | yes |

失败样本：无。

## 7. 20 Molecule Full Hessian Evaluation

评估脚本：

```text
scripts/qm9_hessian_eval_reference_set.py
```

命令：

```bash
DFT_DATA=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1 \
DFT_MODELS=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models \
CUDA_VISIBLE_DEVICES=1 \
/usr/bin/time -v -o _runtime/qm9_p1_models/eval/qm9_p1_410_lambda1_extended_hessian_validation/full_hessian_s3000_20mol_disp1e-3_time.txt \
.venv/bin/python scripts/qm9_hessian_eval_reference_set.py \
  --manifest-json _runtime/qm9_p1_models/eval/qm9_p1_410_lambda1_extended_hessian_validation/pbe_hessian_reference_manifest_20.json \
  --run EG_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200=_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200/checkpoints/epoch_006.ckpt \
  --run EGF_lam1_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000/checkpoints/epoch_000.ckpt \
  --output-json _runtime/qm9_p1_models/eval/qm9_p1_410_lambda1_extended_hessian_validation/full_hessian_s3000_20mol_disp1e-3.json \
  --output-csv _runtime/qm9_p1_models/eval/qm9_p1_410_lambda1_extended_hessian_validation/full_hessian_s3000_20mol_disp1e-3.csv \
  --scf-iteration 1 \
  --displacement 1e-3 \
  --num-workers 0 \
  --device cuda:0
```

输出：

```text
_runtime/qm9_p1_models/eval/qm9_p1_410_lambda1_extended_hessian_validation/full_hessian_s3000_20mol_disp1e-3.json
_runtime/qm9_p1_models/eval/qm9_p1_410_lambda1_extended_hessian_validation/full_hessian_s3000_20mol_disp1e-3.csv
```

资源：

| Job | Wall time | Max RSS | CPU |
|---|---:|---:|---:|
| full Hessian eval 20mol | 1:29.64 | 1427684 KB | 85% |

结果：

| Run | N success | N failed | Hessian MAE | Hessian RMSE | Relative Fro | Symmetry max abs |
|---|---:|---:|---:|---:|---:|---:|
| EG_s3000 | 20 | 0 | 0.354476 | 1.725975 | 14.042393 | 4.4333e-03 |
| EGF_lam1_s3000 | 20 | 0 | 0.016708 | 0.048423 | 0.425208 | 9.8166e-05 |

Hessian 20mol 结论：

- EGF lambda=1.0 在 20/20 reference molecules 上均成功评估。
- EGF lambda=1.0 相比 EG 的 mean Hessian MAE 约降低 `95.29%`，约 `21.2x`。
- mean Hessian RMSE 约降低 `97.19%`，约 `35.6x`。
- mean relative Frobenius error 约降低 `96.97%`，约 `33.0x`。
- model Hessian symmetry error 也从 EG 的 `4.43e-3` 降到 EGF 的 `9.82e-5`。

## 8. Energy-Force-Hessian Tradeoff

核心 tradeoff：

| Metric | EG_s3000 | EGF_lam1_s3000 | Change |
|---|---:|---:|---:|
| val energy loss | 7.959032 | 8.087317 | +0.128286 / +1.61% |
| val gradient loss | 0.031874 | 0.040892 | +0.009018 |
| val total loss | 0.814533 | 0.844071 | +0.029538 |
| val force loss | N/A | 0.003551 | enabled |
| test force component MAE | 0.064490 | 0.003082 | -95.22% |
| 20mol Hessian MAE | 0.354476 | 0.016708 | -95.29% |
| 20mol Hessian RMSE | 1.725975 | 0.048423 | -97.19% |
| 20mol relative Fro | 14.042393 | 0.425208 | -96.97% |

解释：

- 加入 force loss 后，当前 s3000 的 validation energy loss 相比 EG 有小幅退化，绝对增加 `0.128286`，相对约 `1.61%`。
- 这个 energy 退化在本轮 early pilot 中是可接受的，因为 force 和 Hessian 指标改善幅度约为一个数量级以上。
- 但 energy loss 与 gradient loss 并未优于 EG；如果后续目标同时强调能量精度，需要继续调 `lambda`、训练长度、学习率或多目标权重，而不能只看 Hessian。

## 9. Conclusion

在 P1-410 early pilot 的当前数据和模型设置下，`EGF lambda=1.0` 的 s3000 扩展验证通过：

- dataloader / forward / backward / checkpoint 在 s3000 训练中正常。
- s3000 force eval 无失败，EGF lambda=1.0 明显优于 EG。
- 10 molecule displacement sensitivity 在 `5e-4`、`1e-3`、`2e-3` 下稳定，EGF 优势不依赖单一 displacement。
- 20 molecule PBE Hessian reference 全部 success/finite。
- 20 molecule full Hessian eval 中，EGF lambda=1.0 在 MAE、RMSE、relative Fro 和 symmetry 上均明显优于 EG。

本轮可以下的有限结论：

> 在 P1-410 early pilot、20 个小分子 Hessian reference、finite-difference displacement `1e-3` 的条件下，加入由 scalar energy 派生 force 的 `lambda=1.0` force loss 呈现稳定 Hessian 改善趋势；energy loss 相比 EG 有轻微退化，但在当前验证目标下可接受。

明确限制：

- 20 molecule 仍是小 benchmark，不是最终结论。
- reference 分子偏小，最大 12 atoms，不能代表完整 QM9 分布。
- full Hessian 模型侧仍使用 finite difference of derived force，尚未做频率/mass-weighted Hessian 指标。
- 结论只属于 P1-410 early pilot，不外推到最终 P1/P2 或 1000 molecule 设置。
