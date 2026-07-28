# QM9PBEForcePilot P1-410 Hessian Reference Mini-Benchmark

日期：2026-07-03 Asia/Singapore

## 1. Scope

本报告只覆盖现有 `QM9PBEForcePilot` P1-410 early pilot 数据快照和本轮 10 molecule Hessian reference mini-benchmark。

保持约束：

- 不继续生成 1000 molecule labels。
- 不删除现有 `.chk`、labels、cached labels 或 checkpoints。
- 不添加独立 force head。
- EGF force 仍由 scalar energy 导出：`F_pred = -dE_pred/dR`。
- 当前结论仍只属于 P1-410 early pilot，不是最终 Hessian 结论。

## 2. Data Snapshot

数据环境：

```text
DFT_DATA=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1
DFT_MODELS=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models
TMPDIR=/mnt/afs/home/xiazhenhao/dft/_tmp
```

当前快照：

| Artifact | Count / Result |
|---|---:|
| KS `.chk` | 1640 |
| raw label `.zarr.zip` | 1640 |
| cached transformed label `.zarr.zip` | 1640 |
| force label finite check | 1640/1640 pass |
| molecules | 410 |
| samples per molecule | 4 |
| max force norm | 0.19067808023212568 |

force label 检查输出：

```text
_runtime/qm9_p1_models/eval/qm9_p1_410_hessian_ref_benchmark/label_force_finite_check.json
```

split：

| Split | Molecules | Label files |
|---|---:|---:|
| train | 328 | 1312 |
| val | 41 | 164 |
| test | 41 | 164 |

## 3. Training Runs

本轮从 s300 early pilot checkpoint 做 warm-start，只加载模型权重，不恢复 trainer/optimizer 状态。原因是直接用 `ckpt_path` 恢复 EG 时触发 Lightning `RichProgressBar` assertion；该失败 run 没有产生 checkpoint：

```text
_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s1000
```

实际使用的训练 run：

| Run | Loss | Warm-start checkpoint | Run dir | Best checkpoint | Max steps | Batch |
|---|---|---|---|---|---:|---:|
| EG_s1000 | energy + density-gradient | `qm9_p1_410_eg_early_pilot_s300/checkpoints/epoch_000.ckpt` | `_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_warmstart_s1000` | `checkpoints/epoch_001.ckpt` | 1000 | 32 |
| EGF_lam1_s1000 | energy + density-gradient + force | `qm9_p1_410_egf_lam1_early_pilot_s300/checkpoints/epoch_000.ckpt` | `_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_warmstart_s1000` | `checkpoints/epoch_000.ckpt` | 1000 | 4 |
| EGF_lam03_s1000 | energy + density-gradient + 0.3 force | `qm9_p1_410_egf_lam1_early_pilot_s300/checkpoints/epoch_000.ckpt` | `_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam03_warmstart_s1000` | `checkpoints/epoch_000.ckpt` | 1000 | 4 |
| EGF_lam3_s1000 | energy + density-gradient + 3.0 force | `qm9_p1_410_egf_lam1_early_pilot_s300/checkpoints/epoch_000.ckpt` | `_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam3_warmstart_s1000` | `checkpoints/epoch_000.ckpt` | 1000 | 4 |

训练命令模板：

```bash
TMPDIR=/mnt/afs/home/xiazhenhao/dft/_tmp \
DFT_DATA=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1 \
DFT_MODELS=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models \
CUDA_VISIBLE_DEVICES=1 \
.venv/bin/python -m mldft.ml.train \
  experiment=<str25/qm9_pbe_force_pilot_eg|str25/qm9_pbe_force_pilot_egf> \
  name=<run_name> \
  hydra.run.dir=<run_dir> \
  hydra.callbacks.git_logging.clean=false \
  extras.enforce_tags=false \
  extras.print_config=false \
  weight_ckpt_path=<s300_checkpoint> \
  trainer.max_epochs=20 \
  +trainer.max_steps=1000 \
  +trainer.val_check_interval=200 \
  trainer.log_every_n_steps=50 \
  data.datamodule.batch_size=<32|4> \
  data.datamodule.num_workers=0 \
  model.loss_function.force_loss.weight=<0.3|1.0|3.0>
```

训练 scalar 和资源汇总：

```text
_runtime/qm9_p1_models/eval/qm9_p1_410_hessian_ref_benchmark/training_scalar_resource_summary_s1000.json
```

loss 摘要：

| Run | train total first -> last | val total first -> last | final val energy | final val gradient | final val force | Wall time | Max RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| EG_s1000 | 0.067276 -> 0.041218 | 0.841304 -> 0.828277 | 8.006227 | 0.042004 | N/A | 9:59.35 | 2351988 KB |
| EGF_lam1_s1000 | 0.099925 -> 0.048783 | 0.882804 -> 0.852728 | 8.124901 | 0.047064 | 0.003510 | 7:18.64 | 2380284 KB |
| EGF_lam03_s1000 | 0.167042 -> 0.056161 | 0.879864 -> 0.847120 | 8.083419 | 0.048133 | 0.004040 | 7:17.28 | 2351944 KB |
| EGF_lam3_s1000 | 0.181595 -> 0.059093 | 0.879087 -> 0.861502 | 8.118966 | 0.048770 | 0.003847 | 7:11.04 | 2338624 KB |

所有主 loss 标量均 finite。`EGF_lam03_s1000` 终端里出现过部分 train grouped metric 的 `nan`，但主 train/val loss、force eval、Hessian eval 均 finite。

## 4. PBE Hessian Reference Set

从 test split 中按小分子优先选择 10 个 molecule。PBE analytic Hessian 使用现有 `.chk` 恢复 SCF 信息后计算，结果单独保存为 `.npz`，没有写回原始 label。

输出：

```text
_runtime/qm9_p1_models/eval/qm9_p1_410_hessian_ref_benchmark/pbe_hessian_reference_manifest.json
_runtime/qm9_p1_models/eval/qm9_p1_410_hessian_ref_benchmark/pbe_hessian_reference_manifest.csv
_runtime/qm9_p1_models/eval/qm9_p1_410_hessian_ref_benchmark/pbe_hessians/
```

命令：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv/bin/python scripts/qm9_pbe_hessian_reference_set.py \
  --dataset-dir _runtime/qm9_p1/QM9PBEForcePilot \
  --output-dir _runtime/qm9_p1_models/eval/qm9_p1_410_hessian_ref_benchmark/pbe_hessians \
  --manifest-json _runtime/qm9_p1_models/eval/qm9_p1_410_hessian_ref_benchmark/pbe_hessian_reference_manifest.json \
  --manifest-csv _runtime/qm9_p1_models/eval/qm9_p1_410_hessian_ref_benchmark/pbe_hessian_reference_manifest.csv \
  --molecules 0000010,0000323,0000109,0000171,0000062,0000116,0000144,0000019,0000052,0000350 \
  --max-molecules 10 \
  --sample-id 0 \
  --workers 2
```

manifest 摘要：

| Molecule | Natoms | Shape | Elapsed s | Cached | Success | Ref symmetry max abs |
|---|---:|---|---:|---|---|---:|
| 0000010 | 6 | 18 x 18 | 0.003 | yes | yes | 6.20e-14 |
| 0000323 | 6 | 18 x 18 | 526.237 | no | yes | 5.42e-14 |
| 0000109 | 7 | 21 x 21 | 544.056 | no | yes | 3.82e-14 |
| 0000171 | 7 | 21 x 21 | 564.176 | no | yes | 1.08e-13 |
| 0000062 | 8 | 24 x 24 | 668.030 | no | yes | 1.82e-14 |
| 0000116 | 8 | 24 x 24 | 607.329 | no | yes | 4.47e-15 |
| 0000144 | 8 | 24 x 24 | 725.128 | no | yes | 2.22e-14 |
| 0000019 | 9 | 27 x 27 | 612.220 | no | yes | 2.34e-13 |
| 0000052 | 9 | 27 x 27 | 926.908 | no | yes | 3.11e-14 |
| 0000350 | 9 | 27 x 27 | 1150.748 | no | yes | 7.90e-14 |

reference job 资源：wall time `53:54.87`，max RSS `2176892 KB`，exit 0。

失败样本：无。

## 5. Force Test Evaluation

评估脚本：

```text
scripts/qm9_force_eval.py
```

定义：

- test dataloader 展开到 1821 SCF samples / 21949 atoms。
- 参考力来自 `metadata/pbe_derivatives/forces`。
- 预测力由 scalar energy 求导得到：`F_pred = -dE_pred/dR`。

输出：

```text
_runtime/qm9_p1_models/eval/qm9_p1_410_hessian_ref_benchmark/*_force_eval.json
```

结果：

| Run | Force component MAE | Force component RMSE | Force vector MAE | Max component abs error | Failures |
|---|---:|---:|---:|---:|---:|
| EG_s1000 | 0.084362 | 0.335655 | 0.180492 | 8.938153 | 0 |
| EGF_lam03_s1000 | 0.003703 | 0.006656 | 0.007805 | 0.074606 | 0 |
| EGF_lam1_s1000 | 0.003014 | 0.005467 | 0.006292 | 0.075387 | 0 |
| EGF_lam3_s1000 | 0.003043 | 0.005386 | 0.006383 | 0.051012 | 0 |

force-level 结论：

- 1000-step 后 EGF 明显优于 EG。
- `lambda=1.0` 的 force component MAE 最低，`lambda=3.0` 非常接近但未超过。

## 6. Full Hessian Evaluation

评估脚本：

```text
scripts/qm9_hessian_eval_reference_set.py
```

定义：

- 对每个 model checkpoint 和每个 reference molecule，在 `scf_iteration=1` 的 batch 上计算模型 Hessian。
- 模型 Hessian 使用 derived force 的 finite difference，displacement `1e-3`。
- 比较 PBE analytic Hessian reference：
  - Hessian MAE
  - Hessian RMSE
  - relative Frobenius error
  - model Hessian symmetry max abs error

输出：

```text
_runtime/qm9_p1_models/eval/qm9_p1_410_hessian_ref_benchmark/full_hessian_eval_s1000.json
_runtime/qm9_p1_models/eval/qm9_p1_410_hessian_ref_benchmark/full_hessian_eval_s1000.csv
```

结果：

| Run | N success | N failed | Mean Hessian MAE | Mean Hessian RMSE | Mean relative Fro | Mean model symmetry max abs |
|---|---:|---:|---:|---:|---:|---:|
| EG_s1000 | 10 | 0 | 0.513751 | 3.166156 | 19.907544 | 2.392e-02 |
| EGF_lam03_s1000 | 10 | 0 | 0.025269 | 0.071137 | 0.510189 | 9.745e-05 |
| EGF_lam1_s1000 | 10 | 0 | 0.022534 | 0.066419 | 0.489187 | 1.122e-04 |
| EGF_lam3_s1000 | 10 | 0 | 0.023642 | 0.069581 | 0.504850 | 1.103e-04 |

full Hessian 结论：

- 在这 10 个小分子 reference 上，EGF 三组都显著优于 EG。
- `lambda=1.0` 在 mean Hessian MAE、RMSE 和 relative Frobenius error 上都是本轮最佳。
- `lambda=3.0` 接近 `lambda=1.0`，但没有稳定超过；`lambda=0.3` 略弱。
- 模型 Hessian symmetry error 也从 EG 的约 `2.4e-2` 降到 EGF 的约 `1e-4`。

full Hessian eval 资源：wall time `1:58.15`，max RSS `1484348 KB`，exit 0。

## 7. Directional / Secant Proxy

评估脚本：

```text
scripts/qm9_hessian_directional_eval.py
```

定义：

- 使用同一批 10 个 molecule。
- 每个 molecule 用 sample 0 作为 base，和 sample 1/2/3 形成 perturbation pair。
- 总计 30 pairs。
- 报告 `H*dR ~= -dF` 的 proxy 误差：
  - `mean_h_action_mae`
  - `mean_h_slope_mae`
  - `mean_curvature_abs_error`

输出：

```text
_runtime/qm9_p1_models/eval/qm9_p1_410_hessian_ref_benchmark/hessian_directional_s1000_10mol.json
_runtime/qm9_p1_models/eval/qm9_p1_410_hessian_ref_benchmark/hessian_directional_s1000_10mol.csv
```

结果：

| Run | Selected force component MAE | Pairs | Mean H action MAE | Mean H slope MAE | Mean curvature abs error |
|---|---:|---:|---:|---:|---:|
| EG_s1000 | 0.135735 | 30 | 0.102031 | 6.244686 | 3.690568 |
| EGF_lam03_s1000 | 0.004939 | 30 | 0.004225 | 0.246014 | 0.234741 |
| EGF_lam1_s1000 | 0.003903 | 30 | 0.003646 | 0.213061 | 0.197048 |
| EGF_lam3_s1000 | 0.003941 | 30 | 0.003881 | 0.226715 | 0.225312 |

directional proxy 结论：

- proxy 与 full Hessian 方向一致：EGF 显著优于 EG。
- `lambda=1.0` 仍是本轮最佳。
- selected force MAE 的排序也与 Hessian proxy 基本一致。

directional eval 资源：wall time `1:48.51`，max RSS `1463364 KB`，exit 0。

## 8. Failures And Notes

失败/异常列表：

| Item | Status | Note |
|---|---|---|
| Direct EG `ckpt_path` resume | failed | Lightning `RichProgressBar` assertion，未产生 checkpoint；后续改用 `weight_ckpt_path` warm-start |
| PBE Hessian reference | 10/10 success | 无失败样本 |
| Force eval | 4/4 success | 首次 EG eval 缺少 `DFT_DATA` 环境变量，补齐环境变量后成功 |
| Full Hessian eval | 4 runs x 10 refs success | 无失败样本 |
| Directional proxy | 4 runs x 30 pairs success | 无缺失 molecule |

mass-weighted Hessian / frequency 指标本轮暂不报告。原因是当前模型 Hessian 与 PBE Hessian 的单位、坐标单位、质量加权变换和频率换算常数尚未在本 repo 中形成受控脚本；为了避免给出不可靠频率结论，本轮只报告 Cartesian Hessian 对比和 directional/secant proxy。

## 9. Conclusion

在 P1-410 early pilot 的 10 molecule Hessian reference mini-benchmark 上，EGF 呈现明确 Hessian 改善趋势：

- force test：EGF `lambda=1.0` component MAE `0.003014`，EG 为 `0.084362`。
- full Hessian：EGF `lambda=1.0` mean Hessian MAE `0.022534`，EG 为 `0.513751`。
- full Hessian relative Frobenius：EGF `lambda=1.0` 为 `0.489187`，EG 为 `19.907544`。
- directional proxy：EGF `lambda=1.0` mean H action MAE `0.003646`，EG 为 `0.102031`。

本轮最有希望的权重仍是 `lambda=1.0`。`lambda=3.0` force RMSE 和 max force error 接近或略好，但 full Hessian 和 directional proxy 没有超过 `lambda=1.0`；`lambda=0.3` 稍弱。

限制：

- reference 只有 10 个小分子，仍不是最终结论。
- 训练只有 warm-start 1000 steps，不代表收敛模型。
- `lambda=0.3` 和 `lambda=3.0` 从 `lambda=1.0` 的 s300 checkpoint warm-start，不是完全独立 random-init sweep。
- full Hessian 采用 finite difference of derived force，displacement 固定为 `1e-3`；尚未做 displacement sensitivity。
- 未报告 mass-weighted Hessian/frequency，单位处理需单独确认。

下一步建议：

1. 将 `lambda=1.0` 作为 P1-410 Hessian pilot 主线，延长到 3000 steps，并保持 EG 对照同步或至少保留 1000-step 对照。
2. 对同一 10 molecule reference 做 displacement sensitivity，例如 `5e-4`、`1e-3`、`2e-3`。
3. 若 3000-step 后趋势仍稳定，再扩到 20 molecule Hessian reference；仍不需要启动 1000 molecule label generation。
4. 单独实现并验证 mass-weighted Hessian/frequency 单位管线后，再加入频率指标。
