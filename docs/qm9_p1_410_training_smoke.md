# QM9PBEForcePilot 410 Molecule Training Smoke

日期：2026-07-02 21:51 Asia/Singapore

## 1. Scope

本次只基于 `QM9PBEForcePilot` 已完成的 410 个 molecule / 1640 个 labels 做短训练 smoke。

明确未执行：

- 未删除现有 `.chk`。
- 未删除现有 labels。
- 未启动更大批量 datagen 或 labelgen。

## 2. Data Check

产物数量：

| 产物 | 数量 |
|---|---:|
| KS `.chk` | 1640 |
| label `.zarr.zip` | 1640 |
| cached transformed label | 1640 |

force/NaN 检查：

```json
{
  "checked_files": 1640,
  "expected_molecules": 410,
  "expected_samples": 4,
  "failures": [],
  "force_labels": 1640,
  "force_norm_max": 0.19067808023212568,
  "molecules": 410
}
```

检查输出：

```text
_runtime/qm9_p1/QM9PBEForcePilot/qm9_p1_410_force_check_smoke_train.json
```

## 3. Split Check

当前 `split.pkl` 覆盖全部 1640 labels，按 molecule 分组，无泄漏：

```json
{
  "summary": {
    "train": {"labels": 1312, "molecules": 328, "sizes_field": 15740},
    "val": {"labels": 164, "molecules": 41, "sizes_field": 1902},
    "test": {"labels": 164, "molecules": 41, "sizes_field": 1985}
  },
  "total_labels_in_split": 1640,
  "total_labels_on_disk": 1640,
  "total_molecules": 410,
  "leak_count": 0,
  "bad_sample_count_molecules": 0,
  "missing_from_disk": 0,
  "extra_on_disk": 0
}
```

说明：`sizes_field` 是 SCF iteration 数，不是 label 文件数。

## 4. EG Baseline Smoke

配置：`experiment=str25/qm9_pbe_force_pilot_eg`

监督：energy + density-gradient，不加载 force label，不使用 force loss。

命令：

```bash
TMPDIR=/mnt/afs/home/xiazhenhao/dft/_tmp \
DFT_DATA=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1 \
DFT_MODELS=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models \
CUDA_VISIBLE_DEVICES=1 \
/usr/bin/time -v -o _runtime/qm9_p1_models/train/qm9_p1_410_eg_smoke_time.txt \
.venv/bin/python -m mldft.ml.train \
  experiment=str25/qm9_pbe_force_pilot_eg \
  name=qm9_p1_410_eg_smoke \
  hydra.run.dir=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_smoke \
  hydra.callbacks.git_logging.clean=false \
  extras.enforce_tags=false \
  extras.print_config=false \
  trainer.max_epochs=1 \
  +trainer.limit_train_batches=2 \
  +trainer.limit_val_batches=1 \
  trainer.log_every_n_steps=1 \
  data.datamodule.batch_size=2 \
  data.datamodule.num_workers=0
```

输出目录：

```text
_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_smoke
```

checkpoint：

```text
_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_smoke/checkpoints/epoch_000.ckpt
_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_smoke/checkpoints/last.ckpt
```

loss 记录：

| metric | first | last |
|---|---:|---:|
| `train_loss/total` | 0.326776 | 1.810050 |
| `train_loss/energy_loss` | 2.864090 | 17.672356 |
| `train_loss/gradient_loss` | 0.044852 | 0.047572 |
| `train_loss/coefficient_loss` | 0.000000 | 0.001014 |
| `val_loss/total` | 0.490601 | 0.490601 |
| `val_loss/energy_loss` | 4.176861 | 4.176861 |
| `val_loss/gradient_loss` | 0.081017 | 0.081017 |
| `val_loss/coefficient_loss` | 0.014418 | 0.014418 |

说明：只跑 2 个 train batches，train loss 的 first/last 来自不同 batch，不代表收敛趋势。validation 在 fit 后和显式 validate 各记录一次，数值一致。

资源：

| 项目 | 数值 |
|---|---:|
| wall time | 49.01 s |
| CPU | 1493% |
| max RSS | 2366912 KB |
| GPU | `CUDA_VISIBLE_DEVICES=1`, A100 80GB |
| exit status | 0 |

## 5. EGF Force Smoke

配置：`experiment=str25/qm9_pbe_force_pilot_egf`

监督：energy + density-gradient + force loss。force 由 scalar energy 对坐标求导得到，不使用独立 force head。

命令：

```bash
TMPDIR=/mnt/afs/home/xiazhenhao/dft/_tmp \
DFT_DATA=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1 \
DFT_MODELS=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models \
CUDA_VISIBLE_DEVICES=1 \
/usr/bin/time -v -o _runtime/qm9_p1_models/train/qm9_p1_410_egf_smoke_time.txt \
.venv/bin/python -m mldft.ml.train \
  experiment=str25/qm9_pbe_force_pilot_egf \
  name=qm9_p1_410_egf_smoke \
  hydra.run.dir=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_smoke \
  hydra.callbacks.git_logging.clean=false \
  extras.enforce_tags=false \
  extras.print_config=false \
  trainer.max_epochs=1 \
  +trainer.limit_train_batches=2 \
  +trainer.limit_val_batches=1 \
  trainer.log_every_n_steps=1 \
  data.datamodule.batch_size=1 \
  data.datamodule.num_workers=0
```

输出目录：

```text
_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_smoke
```

checkpoint：

```text
_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_smoke/checkpoints/epoch_000.ckpt
_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_smoke/checkpoints/last.ckpt
```

loss 记录：

| metric | first | last |
|---|---:|---:|
| `train_loss/total` | 0.096322 | 3.105516 |
| `train_loss/energy_loss` | 0.540741 | 30.628555 |
| `train_loss/gradient_loss` | 0.052180 | 0.051145 |
| `train_loss/force_loss` | 0.005042 | 0.017449 |
| `train_loss/coefficient_loss` | 0.000237 | 0.006437 |
| `val_loss/total` | 0.741767 | 0.741767 |
| `val_loss/energy_loss` | 6.779915 | 6.779915 |
| `val_loss/gradient_loss` | 0.079516 | 0.079516 |
| `val_loss/force_loss` | 0.001624 | 0.001624 |
| `val_loss/coefficient_loss` | 0.013694 | 0.013694 |

说明：`force_loss` finite，验证了 dataloader force label、`pred_forces = -dE/dR`、force loss 和 backward 可以串通。只跑 2 个 train batches，first/last 不代表收敛趋势。

资源：

| 项目 | 数值 |
|---|---:|
| wall time | 42.42 s |
| CPU | 1175% |
| max RSS | 2354324 KB |
| GPU | `CUDA_VISIBLE_DEVICES=1`, A100 80GB |
| exit status | 0 |

## 6. Warnings And Errors

预启动配置错误：

- 第一次 EG 启动使用了 `trainer.limit_train_batches=2` 和 `trainer.limit_val_batches=1`。
- Hydra 返回 `Could not override 'trainer.limit_train_batches'. To append to your config use +trainer.limit_train_batches=2`。
- 该次未进入 dataloader 或训练；随后改用 `+trainer.limit_train_batches=2` 和 `+trainer.limit_val_batches=1` 后训练正常完成。

最终 EG/EGF smoke run 未发现：

- `ERROR`
- `Traceback`
- `RuntimeError`
- CUDA OOM

出现但不影响本轮 smoke 验收的 warning：

- PyVista future/deprecation warning，来自 mesh/image logging。
- `torchmetrics` 遇到 `nan` 后移除，来自单个 validation batch 中 ground-state 子集为空。
- Lightning 提示 `num_workers=0` 可能是瓶颈，这是本轮 smoke 为降低复杂度的显式设置。

validation 主 loss 均为 finite；`val_metrics_ground_state/*` 的 `nan/inf` 与早先 110 molecule smoke 一致，原因是 1 个 validation batch 未包含 ground-state 样本。

## 7. Conclusion

410 molecule 节点的正式训练 smoke 通过：

- dataloader 可读取 cached transformed labels。
- EG baseline forward/backward 正常。
- EGF force-supervision forward/backward 正常。
- loss logging 正常。
- TensorBoard event files 正常生成。
- `epoch_000.ckpt` 和 `last.ckpt` 均已保存。
