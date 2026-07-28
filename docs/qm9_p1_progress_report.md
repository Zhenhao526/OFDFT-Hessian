# QM9 OFDFT Force/Hessian P1 Progress Report

日期：2026-07-01

最近更新：2026-07-02 21:51 Asia/Singapore

## 1. 当前状态

本报告记录 `qm9_force_hessian_p1_p2_taskbook.md` 中 P1 的当前执行进展。

已完成：

- P1-0 pilot datagen preset。
- molecule-level grouped split 支持。
- P0 真实 labels 的 grouped split。
- 正式 ML dataloader 的可选 force label 读取。
- 正式 MLDFT Lightning module 的可选 `-dE/dR` force prediction。
- `ForceLoss` 组件。
- P1 EG/EGF 训练配置入口。
- P0 labels 上的正式 MLDFT force forward/backward smoke。

未完成：

- P1-2 生成 1000 molecule pilot labels。当前已完成 410 molecule / 1640 labels。
- P1 完整 EG/EGF training。
- P1 Hessian mini-eval。

## 2. 新增配置

### 2.1 Datagen

新增：

```text
configs/datagen/preset/qm9_pbe_force_pilot.yaml
```

关键设置：

```yaml
dataset.name: QM9PBEForcePilot
dataset.filename: qm9_pbe_force_pilot
n_molecules: 1000
dataset.include_reference: true
dataset.num_perturbations: 3
kohn_sham.xc: PBE
kohn_sham.basis: 6-31G(2df,p)
kohn_sham.compute_forces: true
kohn_sham.compute_hessian: false
```

Hydra compose 验证通过：

```yaml
dataset_name: QM9PBEForcePilot
filename: qm9_pbe_force_pilot
n_molecules: 1000
num_perturbations: 3
include_reference: true
compute_forces: true
compute_hessian: false
```

### 2.2 ML training

新增：

```text
configs/ml/data/qm9_pbe_force_pilot.yaml
configs/ml/model/loss_function/l1_force.yaml
configs/ml/experiment/str25/qm9_pbe_force_pilot_eg.yaml
configs/ml/experiment/str25/qm9_pbe_force_pilot_egf.yaml
```

EG 配置：

```yaml
data.dataset_name: QM9PBEForcePilot
model.force_supervision: false
data.datamodule.dataset_kwargs.load_force_label: false
```

EGF 配置：

```yaml
data.dataset_name: QM9PBEForcePilot
model.force_supervision: true
data.datamodule.dataset_kwargs.load_force_label: true
model.loss_function: l1_force
```

Hydra compose 验证通过：

```yaml
str25/qm9_pbe_force_pilot_eg:
  dataset_name: QM9PBEForcePilot
  batch_size: 32
  load_force_label: false
  force_supervision: false
  loss_keys:
    - energy_loss
    - gradient_loss
    - coefficient_loss

str25/qm9_pbe_force_pilot_egf:
  dataset_name: QM9PBEForcePilot
  batch_size: 16
  load_force_label: true
  force_supervision: true
  loss_keys:
    - energy_loss
    - gradient_loss
    - force_loss
    - coefficient_loss
```

## 3. Molecule-level Split

`mldft/utils/create_dataset_splits.py` 已新增：

```text
--group-by-molecule
```

用途：

- 对 `0000001.0000000.zarr.zip` 和 `0000001.0000001.zarr.zip` 这类多 sample label，按 molecule id 分组划分。
- 确保同一 molecule 的 reference 与 perturbation 不会跨 train/val/test 泄漏。

P0 split 已生成：

```text
_runtime/qm9_p0/QM9PBEForceSmoke/split.yaml
_runtime/qm9_p0/QM9PBEForceSmoke/split.pkl
```

P0 split 检查：

```python
{
  "summary": {
    "train": {"labels": 160, "molecules": 80},
    "val": {"labels": 20, "molecules": 10},
    "test": {"labels": 20, "molecules": 10}
  },
  "sizes": {"train": 1788, "val": 222, "test": 224},
  "leaks": []
}
```

说明：`sizes` 是 SCF iteration 数，不是 label 文件数。

## 4. Force Dataloader 与 Loss

### 4.1 Force label 读取

`OFData.from_file` 新增可选参数：

```python
load_force_label: bool = False
force_key: str = "metadata/pbe_derivatives/forces"
```

开启后会读取：

```text
metadata/pbe_derivatives/forces
```

并写入：

```python
sample.force_label
```

默认关闭，不影响旧训练配置。

### 4.2 Force prediction

`MLDFTLitModule` 新增：

```python
force_supervision: bool = False
```

开启后内部通过 scalar energy 计算：

```math
F_{\mathrm{pred}} = -\frac{\partial E_{\mathrm{pred}}}{\partial R}
```

默认 `forward(batch)` 仍返回历史三元组：

```python
pred_energy, pred_gradients, pred_diff
```

训练步骤内部使用新增的：

```python
forward_predictions(batch)
```

返回：

```python
pred_energy, pred_gradients, pred_diff, pred_forces
```

### 4.3 Force loss

新增：

```python
mldft.ml.models.components.loss_function.ForceLoss
```

`ForceLoss` 比较：

```python
pred_forces
batch.force_label
```

输出 per-atom force loss，再做 reduction。

## 5. P0 正式 MLDFT Force Smoke

新增脚本：

```text
scripts/qm9_force_mldft_smoke.py
```

该脚本使用：

- 真实 P0 `.zarr.zip` labels；
- `OFDataset`；
- `OFLoader`；
- `OFData.from_file(load_force_label=True)`；
- `MLDFTLitModule(force_supervision=True)`；
- `ForceLoss`；
- PyTorch optimizer backward。

运行命令：

```bash
TMPDIR=/mnt/afs/home/xiazhenhao/dft/_tmp \
.venv/bin/python scripts/qm9_force_mldft_smoke.py \
  --max-paths 4 \
  --batch-size 2 \
  --steps 2
```

结果：

```json
{
  "batch_size": 2,
  "dataset": "QM9PBEForceSmoke",
  "force_loss_final": 0.003935095180607754,
  "force_loss_initial": 0.006647372307650179,
  "force_norm_max": 0.018095369281656042,
  "loss_final": 5.396103467035204,
  "loss_initial": 3.8583152181433826,
  "paths": 4,
  "samples": 4,
  "steps": 2
}
```

结论：

- 正式 ML 数据读取路径可以读取 PBE force label。
- 正式 MLDFT module 可以从 scalar energy 对坐标求导得到 force。
- force loss 可以参与 backward。
- 该 smoke 使用 ToyNet，不等价于完整 Graphformer 训练；但已经覆盖 P1-1 的核心 autograd 和 loss 接口风险。

## 6. 验证记录

通过的验证：

```bash
.venv/bin/python -m py_compile \
  mldft/utils/create_dataset_splits.py \
  mldft/ml/data/components/of_data.py \
  mldft/ml/models/components/loss_function.py \
  mldft/ml/models/mldft_module.py \
  scripts/qm9_force_mldft_smoke.py
```

通过的测试：

```bash
.venv/bin/python -m pytest -q \
  tests/ml/test_loading.py \
  tests/ml/test_loss_functions.py \
  tests/ml/test_mldft_module.py \
  tests/utils/test_create_dataset_splits.py
```

结果：

```text
28 passed
```

额外通过：

```bash
.venv/bin/python -m pytest -q \
  tests/utils/test_create_dataset_splits.py \
  tests/datagen/test_qm9_geometry_perturbation.py
```

结果：

```text
4 passed
```

## 7. P1-2 小切片结果

已完成 10 molecule 小切片，用于验证 P1 pilot 配置在真实 datagen 上可运行。

运行设置：

```text
DFT_DATA=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1
TMPDIR=/mnt/afs/home/xiazhenhao/dft/_tmp
dataset.raw_data_dir=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p0/QM9/raw
dataset.name=QM9PBEForcePilot
n_molecules=10
num_perturbations=3
include_reference=true
```

产物：

```text
_runtime/qm9_p1/QM9PBEForcePilot/kohn_sham
_runtime/qm9_p1/QM9PBEForcePilot/labels
```

数量：

| 产物 | 数量 |
|---|---:|
| molecule | 10 |
| samples per molecule | 4 |
| `.chk` | 40 |
| `.zarr.zip` labels | 40 |
| force labels | 40 |

force/NaN 检查：

```json
{
  "checked_files": 40,
  "expected_molecules": 10,
  "expected_samples": 4,
  "failures": [],
  "force_labels": 40,
  "force_norm_max": 0.08147435064961737,
  "molecules": 10
}
```

小切片 grouped split 已生成：

```text
_runtime/qm9_p1/QM9PBEForcePilot/split.yaml
_runtime/qm9_p1/QM9PBEForcePilot/split.pkl
```

split 检查：

```python
{
  "summary": {
    "train": {"labels": 32, "molecules": 8},
    "val": {"labels": 4, "molecules": 1},
    "test": {"labels": 4, "molecules": 1}
  },
  "sizes": {"train": 292, "val": 36, "test": 35},
  "leaks": []
}
```

说明：`sizes` 是 SCF iteration 数，不是 label 文件数。

P1 小切片正式 MLDFT force smoke：

```bash
TMPDIR=/mnt/afs/home/xiazhenhao/dft/_tmp \
.venv/bin/python scripts/qm9_force_mldft_smoke.py \
  --data-root /mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1 \
  --dataset-name QM9PBEForcePilot \
  --max-paths 4 \
  --batch-size 2 \
  --steps 2
```

结果：

```json
{
  "batch_size": 2,
  "dataset": "QM9PBEForcePilot",
  "force_loss_final": 0.008396857592859946,
  "force_loss_initial": 0.00703271431239543,
  "force_norm_max": 0.022613921906568676,
  "loss_final": 3.94614122798426,
  "loss_initial": 3.950711766911343,
  "paths": 4,
  "samples": 4,
  "steps": 2
}
```

结论：

- P1 pilot datagen 配置可以真实生成 reference + 3 perturbation labels。
- P1 labels 的 PBE force label 可读且数值有限。
- P1 molecule-level grouped split 无泄漏。
- 正式 ML dataloader + `-dE/dR` force path 可以在 P1 labels 上 forward/backward。

## 8. P1-2 110 Molecule 批次结果

已在 10 molecule 小切片基础上追加 100 molecule 批次，当前 P1 pilot 共完成 110 molecule。

运行设置：

```text
DFT_DATA=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1
TMPDIR=/mnt/afs/home/xiazhenhao/dft/_tmp
dataset.raw_data_dir=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p0/QM9/raw
preset=qm9_pbe_force_pilot
start_idx=10
n_molecules=100
num_processes=4
num_threads_per_process=1
max_memory_per_process=5000
```

耗时：

| 步骤 | 任务量 | 耗时 |
|---|---:|---:|
| KS `.chk` | 100 molecule / 400 samples | 2:06:36 |
| label `.zarr.zip` | 100 molecule / 400 samples | 1:41:30 |

累计产物：

| 产物 | 数量 |
|---|---:|
| molecule | 110 |
| samples per molecule | 4 |
| KS `.chk` | 440 |
| label `.zarr.zip` | 440 |
| force labels | 440 |

force/NaN 检查：

```json
{
  "checked_files": 440,
  "expected_molecules": 110,
  "expected_samples": 4,
  "failures": [],
  "force_labels": 440,
  "force_norm_max": 0.1589626321100853,
  "molecules": 110
}
```

summary 文件：

```text
_runtime/qm9_p1/QM9PBEForcePilot/qm9_p1_110_force_check.json
```

grouped split 已重建：

```python
{
  "summary": {
    "train": {"labels": 352, "molecules": 88},
    "val": {"labels": 44, "molecules": 11},
    "test": {"labels": 44, "molecules": 11}
  },
  "sizes": {"train": 3962, "val": 496, "test": 495},
  "leaks": [],
  "bad_sample_counts": {}
}
```

说明：`sizes` 是 SCF iteration 数，不是 label 文件数。

P1 110 molecule 正式 MLDFT force smoke：

```bash
TMPDIR=/mnt/afs/home/xiazhenhao/dft/_tmp \
.venv/bin/python scripts/qm9_force_mldft_smoke.py \
  --data-root /mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1 \
  --dataset-name QM9PBEForcePilot \
  --max-paths 16 \
  --batch-size 2 \
  --steps 10
```

结果：

```json
{
  "batch_size": 2,
  "dataset": "QM9PBEForcePilot",
  "force_loss_final": 0.011908184232301038,
  "force_loss_initial": 0.010019635893776156,
  "force_norm_max": 0.0385678529952547,
  "loss_final": 7.175328686857666,
  "loss_initial": 3.897392500526202,
  "paths": 16,
  "samples": 16,
  "steps": 8
}
```

结论：

- P1 datagen 已从 10 molecule 小切片扩展到 110 molecule，KS 和 labelgen 均可恢复、可分批运行。
- 440 个 labels 全部包含 finite force label。
- molecule-level grouped split 在 110 molecule 规模无泄漏。
- 正式 ML dataloader + `-dE/dR` force path 在更大 P1 labels 上 forward/backward 通过。

## 9. P1-4 110 Molecule 正式训练 Smoke

### 9.1 Cached transform 与 dataset statistics

正式 Graphformer 配置默认读取 `labels_local_frames_global_symmetric_natrep`。在未缓存时用 `data.transforms.use_cached_data=false` 在线计算 statistics 会失败，原因是 local frame 在重原子不足的小分子上会用随机 dummy atoms；statistics 三轮 pass 看到的局部基不完全一致，触发 `Mean gradient after AtomRef is not zero` 断言。

已处理方式：

- 使用 `mldft.datagen.transform_dataset` 将 440 个 P1 labels 缓存为 `labels_local_frames_global_symmetric_natrep`。
- 缓存目录：`_runtime/qm9_p1/QM9PBEForcePilot/labels_local_frames_global_symmetric_natrep`
- 缓存文件数：`440`
- 目录大小：`228M`
- force label 已确认保留在 `metadata/pbe_derivatives/forces`。

随后用 cached labels 成功生成 dataset statistics：

```text
_runtime/qm9_p1/QM9PBEForcePilot/dataset_statistics/dataset_statistics_labels_local_frames_global_symmetric_natrep_e_kin_plus_xc.zarr
```

statistics run 目录：

```text
_runtime/qm9_p1_models/statistics/runs/qm9_p1_110_stats_cached
```

本次 statistics 使用 `statistic_fitter_kwargs.n_batches=16` 做 smoke 级统计；1000 molecule pilot 完成后应重算全量 statistics。

### 9.2 EG/EGF 短训练结果

EG smoke：

- 配置：`experiment=str25/qm9_pbe_force_pilot_eg`
- batch size：`2`
- train batches：`2`
- val batches：`1`
- run dir：`_runtime/qm9_p1_models/train/runs/qm9_p1_110_eg_smoke`
- 结果：训练和 validation 均完成。

关键 validation metrics：

```text
val_loss/total            0.7208623886108398
val_loss/energy_loss      6.823925018310547
val_loss/gradient_loss    0.042744264006614685
val_loss/coefficient_loss 0.0029869971331208944
```

EGF smoke：

- 配置：`experiment=str25/qm9_pbe_force_pilot_egf`
- batch size：`1`
- train batches：`2`
- val batches：`1`
- run dir：`_runtime/qm9_p1_models/train/runs/qm9_p1_110_egf_smoke`
- 结果：训练和 validation 均完成，`force_loss` finite。

关键 validation metrics：

```text
val_loss/total            0.944225549697876
val_loss/energy_loss      9.098930358886719
val_loss/gradient_loss    0.04030338302254677
val_loss/force_loss       0.020898178219795227
val_loss/coefficient_loss 0.0018060185248032212
```

说明：

- `val_metrics_ground_state/*` 在 1 个 validation batch 的 smoke 中出现 `nan/inf`，原因是该 batch 没有 ground-state 样本；不影响本次训练链路验收。
- 本轮已验证正式 cached labels、dataset statistics、Graphformer、force label 加载、`pred_forces = -dE/dR`、`ForceLoss` 与 Lightning train/validate loop 可以串通。

## 10. P1-2 210 Molecule 批次结果

在 110 molecule 基础上继续追加 100 molecule，当前 P1 pilot 共完成 210 molecule。

本批次范围：

```text
start_idx=110
n_molecules=100
molecule ids: 111-210
samples per molecule: 4
```

### 10.1 Datagen 产物

KS 生成：

- log dir：`_runtime/qm9_p1/QM9PBEForcePilot/logs/2026-07-01_23-01-42`
- 运行时间：`3:30:28`
- `.chk` 文件数：`840`

Label 生成：

- log dir：`_runtime/qm9_p1/QM9PBEForcePilot/logs/2026-07-02_02-32-57`
- 运行时间：`2:09:06`
- label 文件数：`840`

当前目录体积：

| 目录 | 大小 |
|---|---:|
| `kohn_sham` | `2.2G` |
| `labels` | `482M` |

### 10.2 Force/NaN 检查

检查命令输出：

```json
{
  "checked_files": 840,
  "expected_molecules": 210,
  "expected_samples": 4,
  "failures": [],
  "force_labels": 840,
  "force_norm_max": 0.19067808023212568,
  "molecules": 210
}
```

summary 文件：

```text
_runtime/qm9_p1/QM9PBEForcePilot/qm9_p1_210_force_check.json
```

### 10.3 Grouped Split

grouped split 已重建并检查无泄漏：

```json
{
  "bad_sample_counts": {},
  "leaks": [],
  "sizes": {
    "test": 979,
    "train": 7804,
    "val": 986
  },
  "summary": {
    "test": {"labels": 84, "molecules": 21},
    "train": {"labels": 672, "molecules": 168},
    "val": {"labels": 84, "molecules": 21}
  }
}
```

说明：`sizes` 是 SCF iteration 数，不是 label 文件数。

### 10.4 Cached Transform 与 Statistics

cached transformed labels 已更新：

- 目录：`_runtime/qm9_p1/QM9PBEForcePilot/labels_local_frames_global_symmetric_natrep`
- 文件数：`840`
- 目录大小：`482M`
- transform run dir：`_runtime/qm9_p1_models/transform/runs/qm9_p1_210_local_frames_cache`

cached statistics 已刷新：

- dataset statistics：`_runtime/qm9_p1/QM9PBEForcePilot/dataset_statistics/dataset_statistics_labels_local_frames_global_symmetric_natrep_e_kin_plus_xc.zarr`
- statistics run dir：`_runtime/qm9_p1_models/statistics/runs/002_qm9_p1_210_stats_cached_batch_size-16__num_workers-0__qm9_pbe_force_pilot__enforce_tags-false__print_config-false__overwrite-true__n_batches-16`
- statistics 大小：`1.5M`
- 本轮仍使用 `statistic_fitter_kwargs.n_batches=16`，属于 smoke 级 statistics。

210 molecule 节点时的 P1-2 剩余规模：

```text
目标：1000 molecule / 4000 labels
已完成：210 molecule / 840 labels
剩余：790 molecule / 3160 labels
```

210 molecule 节点未重跑正式 EG/EGF training smoke；当前训练链路验收仍沿用 110 molecule 的 EG/EGF smoke。

## 11. P1-2 310 Molecule 批次结果

在 210 molecule 基础上继续追加 100 molecule，当前 P1 pilot 共完成 310 molecule。

本批次范围：

```text
start_idx=210
n_molecules=100
molecule ids: 211-310
samples per molecule: 4
```

### 11.1 Datagen 产物

KS 生成：

- log dir：`_runtime/qm9_p1/QM9PBEForcePilot/logs/2026-07-02_04-54-15`
- 运行时间：`4:24:33`
- `.chk` 文件数：`1240`

Label 生成：

- log dir：`_runtime/qm9_p1/QM9PBEForcePilot/logs/2026-07-02_09-20-04`
- 运行时间：`3:28:36`
- label 文件数：`1240`

当前目录体积：

| 目录 | 大小 |
|---|---:|
| `kohn_sham` | `3.9G` |
| `labels` | `784M` |

### 11.2 Force/NaN 检查

检查命令输出：

```json
{
  "checked_files": 1240,
  "expected_molecules": 310,
  "expected_samples": 4,
  "failures": [],
  "force_labels": 1240,
  "force_norm_max": 0.19067808023212568,
  "molecules": 310
}
```

summary 文件：

```text
_runtime/qm9_p1/QM9PBEForcePilot/qm9_p1_310_force_check.json
```

### 11.3 Grouped Split

grouped split 已重建并检查无泄漏：

```json
{
  "bad_sample_counts": {},
  "leaks": [],
  "sizes": {
    "test": 1447,
    "train": 11825,
    "val": 1430
  },
  "summary": {
    "test": {"labels": 124, "molecules": 31},
    "train": {"labels": 992, "molecules": 248},
    "val": {"labels": 124, "molecules": 31}
  }
}
```

说明：`sizes` 是 SCF iteration 数，不是 label 文件数。

### 11.4 Cached Transform 与 Statistics

cached transformed labels 已更新：

- 目录：`_runtime/qm9_p1/QM9PBEForcePilot/labels_local_frames_global_symmetric_natrep`
- 文件数：`1240`
- 目录大小：`782M`
- transform run dir：`_runtime/qm9_p1_models/transform/runs/qm9_p1_310_local_frames_cache`

cached statistics 已刷新：

- dataset statistics：`_runtime/qm9_p1/QM9PBEForcePilot/dataset_statistics/dataset_statistics_labels_local_frames_global_symmetric_natrep_e_kin_plus_xc.zarr`
- statistics run dir：`_runtime/qm9_p1_models/statistics/runs/004_qm9_p1_310_stats_cached_batch_size-16__num_workers-0__qm9_pbe_force_pilot__enforce_tags-false__print_config-false__overwrite-true__n_batches-16`
- statistics 大小：`1.5M`
- 本轮仍使用 `statistic_fitter_kwargs.n_batches=16`，属于 smoke 级 statistics。

当前 P1-2 剩余规模：

```text
目标：1000 molecule / 4000 labels
已完成：310 molecule / 1240 labels
剩余：690 molecule / 2760 labels
```

310 molecule 节点未重跑正式 EG/EGF training smoke；当前训练链路验收仍沿用 110 molecule 的 EG/EGF smoke。

## 12. P1-2 410 Molecule 批次结果

在 310 molecule 基础上继续追加 100 molecule，当前 P1 pilot 共完成 410 molecule。本批次 KS 阶段曾被中断，随后从现有 `.chk` 断点恢复。

本批次范围：

```text
start_idx=310
n_molecules=100
molecule ids: 311-410
samples per molecule: 4
```

### 12.1 Datagen 产物与恢复状态

KS 恢复：

- 初始 log dir：`_runtime/qm9_p1/QM9PBEForcePilot/logs/2026-07-02_13-02-33`
- 恢复 log dir：`_runtime/qm9_p1/QM9PBEForcePilot/logs/2026-07-02_16-28-52`
- 说明：13:02 运行期间误清理 `_tmp`，该运行出现一次临时文件 `FileNotFoundError` 并停在 `1536` 个 `.chk`；随后以相同命令恢复，恢复运行识别 `73` 个 molecule 已完成、补算 `27` 个 molecule，最终验证 `.chk` 到 `1640`。
- 恢复运行时间：`1:48:13`
- `.chk` 文件数：`1640`

Label 生成：

- log dir：`_runtime/qm9_p1/QM9PBEForcePilot/logs/2026-07-02_18-19-27`
- 运行时间：约 `3:10`
- label 文件数：`1640`

当前目录体积：

| 目录 | 大小 |
|---|---:|
| `kohn_sham` | `5.4G` |
| `labels` | `1.1G` |

### 12.2 Force/NaN 检查

检查命令输出：

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

summary 文件：

```text
_runtime/qm9_p1/QM9PBEForcePilot/qm9_p1_410_force_check.json
```

### 12.3 Grouped Split

grouped split 已重建并检查无泄漏：

```json
{
  "bad_sample_counts": {},
  "leaks": [],
  "sizes": {
    "test": 1985,
    "train": 15740,
    "val": 1902
  },
  "summary": {
    "test": {"labels": 164, "molecules": 41},
    "train": {"labels": 1312, "molecules": 328},
    "val": {"labels": 164, "molecules": 41}
  }
}
```

说明：`sizes` 是 SCF iteration 数，不是 label 文件数。

split 检查摘要已写入：

```text
_runtime/qm9_p1/QM9PBEForcePilot/reports/qm9_p1_410_split_check.json
```

### 12.4 Cached Transform 与 Statistics

cached transformed labels 已更新：

- 目录：`_runtime/qm9_p1/QM9PBEForcePilot/labels_local_frames_global_symmetric_natrep`
- 文件数：`1640`
- 目录大小：`1.1G`
- transform run dir：`_runtime/qm9_p1_models/transform/runs/qm9_p1_410_local_frames_cache`

cached statistics 已刷新：

- dataset statistics：`_runtime/qm9_p1/QM9PBEForcePilot/dataset_statistics/dataset_statistics_labels_local_frames_global_symmetric_natrep_e_kin_plus_xc.zarr`
- statistics run dir：`_runtime/qm9_p1_models/statistics/runs/006_qm9_p1_410_stats_cached_batch_size-16__num_workers-0__qm9_pbe_force_pilot__enforce_tags-false__print_config-false__overwrite-true__n_batches-16`
- statistics 大小：`1.5M`
- 本轮仍使用 `statistic_fitter_kwargs.n_batches=16`，属于 smoke 级 statistics。

当前 P1-2 剩余规模：

```text
目标：1000 molecule / 4000 labels
已完成：410 molecule / 1640 labels
剩余：590 molecule / 2360 labels
```

410 molecule 节点已重跑正式 EG/EGF 短训练 smoke，结果见下一节。

## 13. P1-4 410 Molecule 正式训练 Smoke

410 molecule 节点已完成短训练 smoke。详细记录见：

```text
docs/qm9_p1_410_training_smoke.md
```

数据前置检查：

- KS `.chk`：`1640`
- label `.zarr.zip`：`1640`
- cached transformed labels：`1640`
- force/NaN 检查：`failures=[]`
- grouped split：train/val/test = 328/41/41 molecule，无泄漏

EG baseline smoke：

- 配置：`experiment=str25/qm9_pbe_force_pilot_eg`
- batch size：`2`
- train batches：`2`
- val batches：`1`
- run dir：`_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_smoke`
- checkpoint：`epoch_000.ckpt`, `last.ckpt`
- validation total loss：`0.4906010329723358`
- 结果：训练和 validation 均完成。

EGF force smoke：

- 配置：`experiment=str25/qm9_pbe_force_pilot_egf`
- batch size：`1`
- train batches：`2`
- val batches：`1`
- run dir：`_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_smoke`
- checkpoint：`epoch_000.ckpt`, `last.ckpt`
- validation total loss：`0.741766631603241`
- validation force loss：`0.0016242226120084524`
- 结果：训练和 validation 均完成，`force_loss` finite。

说明：

- 本轮只验证 dataloader、forward/backward、loss logging、checkpoint 保存，不判断收敛。
- `val_metrics_ground_state/*` 仍可能在单个 validation batch 中出现 `nan/inf`，原因是该 batch 没有 ground-state 样本；主 validation losses 为 finite。
- 未启动新的 datagen/labelgen，未删除现有 `.chk` 或 labels。

## 14. 下一步

### 14.1 完成 P1-2 剩余 pilot labels

下一步应将 P1-2 从 410 molecule 扩展到 1000 molecule pilot datagen。

建议继续按 100-200 molecule 分批运行。下一批建议从 `start_idx=410` 开始：

```bash
TMPDIR=/mnt/afs/home/xiazhenhao/dft/_tmp \
DFT_DATA=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1 \
.venv/bin/python -m mldft.datagen.kohn_sham_dataset \
  preset=qm9_pbe_force_pilot \
  start_idx=410 \
  n_molecules=100 \
  num_processes=4 \
  num_threads_per_process=1 \
  max_memory_per_process=5000 \
  dataset.raw_data_dir=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p0/QM9/raw
```

随后运行对应 label generation：

```bash
TMPDIR=/mnt/afs/home/xiazhenhao/dft/_tmp \
DFT_DATA=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1 \
.venv/bin/python -m mldft.datagen.generate_labels_dataset \
  preset=qm9_pbe_force_pilot \
  start_idx=<START> \
  n_molecules=<BATCH_SIZE> \
  num_processes=4 \
  num_threads_per_process=1 \
  max_memory_per_process=5000 \
  dataset.raw_data_dir=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p0/QM9/raw
```

每批完成后立即运行：

- label count；
- `scripts/check_qm9_force_smoke.py`；
- cached transform 更新；
- grouped split rebuild；
- split leak check。

cached transform 可用以下命令重跑；脚本会跳过已转换文件：

```bash
TMPDIR=/mnt/afs/home/xiazhenhao/dft/_tmp \
DFT_DATA=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1 \
DFT_MODELS=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models \
.venv/bin/python -m mldft.datagen.transform_dataset \
  data=qm9_pbe_force_pilot \
  extras.enforce_tags=false \
  extras.print_config=false \
  +hydra.callbacks.git_logging.clean=false \
  +num_processes=4 \
  +num_threads_per_process=1 \
  +start_idx=0 \
  +num_molecules=999999
```

### 14.2 P1 training 下一步

当前 110 molecule 批次已经通过 ToyNet smoke、正式 EG smoke、正式 EGF smoke；410 molecule 节点已刷新 cached labels、grouped split、smoke 级 statistics，并通过正式 EG/EGF 短训练 smoke。下一步：

- 扫一个小范围 `lambda_F`；
- P1 labels 每扩展一个较大批次后重算 cached statistics；
- P1 labels 接近 1000 molecule 后再做完整 EG/EGF pilot training。
