# HORM -> STRUCTURES25 DFT 标签生成

更新时间：2026-07-01

## 路径

STRUCTURES25 代码目录：

```text
/mnt/afs/home/xiazhenhao/dft/structures25
```

HORM 本地数据：

```text
/mnt/afs/home/xiazhenhao/dft/archive/ts1x-val.lmdb
/mnt/afs/home/xiazhenhao/dft/archive/ts1x_hess_train.lmdb
/mnt/afs/home/xiazhenhao/dft/archive/RGD1.lmdb
```

还发现了 `/mnt/afs/home/xiazhenhao/dft/Transition1x.h5`。当前实现的 adapter 面向 HORM archive LMDB，因为这些文件已经确认包含 energy/forces/Hessian 字段；H5 反应数据没有纳入本次 pilot。

## 数据格式解析

新增 adapter：`mldft/datagen/datasets/horm.py`。

LMDB value 是 pickled `torch_geometric.data.Data` 对象。TS1x 样本包含：

```text
pos, ae, charges, rxn, hessian, energy, forces, one_hot, natoms
```

RGD1 样本包含：

```text
pos, charges, eig_values, force_constant, hessian, energy, forces, one_hot, freq, natoms
```

`charges` 映射为 `geometry/atomic_numbers`，`pos` 按 Angstrom 传给 PySCF，`hessian` reshape 为 `(3N, 3N)` 后只保存为 reference metadata。

HORM archive 没有 charge/spin 字段。本流程默认：

```text
charge = 0
spin = 0
multiplicity = 1
```

也就是 neutral closed-shell。奇电子或非闭壳层样本会在 restricted PBE single-point 中失败并写日志。

## DFT level

新生成的 STRUCTURES25 标签使用：

```text
PySCF RKS
PBE/6-31G(2df,p)
single-point only, no geometry optimization
grid_level = 3
initial guess = minao
density fitting = hartree+external_mofdft
OF basis = even_tempered_2.5
```

`mldft_ks` 保存 SCF 中间迭代到 `.chk`，供 `mldft_labelgen` 计算 density-gradient labels。

HORM archive 原始 force/Hessian 的来源在本地说明中标注为：

```text
omegaB97x/6-31G(d)
```

因此这些 archive energy/force/Hessian 不能和本流程生成的 `PBE/6-31G(2df,p)` energy/gradient 当作同一个训练目标混用。

## 生成标签

核心 STRUCTURES25/M-OFDFT 标签在 zarr 中：

```text
geometry/atomic_numbers
geometry/atom_pos
geometry/charge
geometry/spin
geometry/multiplicity

ks_labels/energies/e_kin
ks_labels/energies/e_xc
ks_labels/energies/e_hartree
ks_labels/energies/e_ext
ks_labels/energies/e_nuc_nuc
ks_labels/energies/e_tot

of_labels/spatial/coeffs
of_labels/energies/e_kin
of_labels/energies/e_xc
of_labels/energies/e_hartree
of_labels/energies/e_ext
of_labels/energies/e_tot
of_labels/energies/e_kin_plus_xc
of_labels/spatial/grad_kin
of_labels/spatial/grad_kin_plus_xc
of_labels/spatial/grad_tot
```

含义对应：

```text
p                 -> of_labels/spatial/coeffs
dTs/dp            -> of_labels/spatial/grad_kin
d(Ts+Exc)/dp      -> of_labels/spatial/grad_kin_plus_xc
dE_total/dp       -> of_labels/spatial/grad_tot
```

HORM archive 自带数据保存到：

```text
metadata/reference/energy
metadata/reference/forces
metadata/reference/hessian
metadata/reference/source_dft_level
```

这些是 reference metadata，不是 PBE 标签训练目标。

可选同水平 PBE force/Hessian 重算保存到：

```text
metadata/pbe_derivatives/forces
metadata/pbe_derivatives/nuclear_gradient
metadata/pbe_derivatives/hessian
metadata/pbe_derivatives/hessian_matrix
```

Hessian 默认关闭；本机 5 原子 pilot 的 PBE Hessian 单样本约 214 秒。

## Pilot 命令

默认 pilot 只跑 TS1x-val key `17304` 一个 5 原子样本：

```bash
cd /mnt/afs/home/xiazhenhao/dft/structures25
mkdir -p /mnt/afs/home/xiazhenhao/dft/logs

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYSCF_MAX_MEMORY=4000 \
./.venv/bin/mldft_ks preset=horm_pbe_labels \
  n_molecules=1 start_idx=0 num_threads_per_process=2 \
  hydra.run.dir=/mnt/afs/home/xiazhenhao/dft/logs/horm_pilot_ks \
  log_file=/mnt/afs/home/xiazhenhao/dft/logs/horm_pilot_ks.log \
  config_file=/mnt/afs/home/xiazhenhao/dft/logs/horm_pilot_ks_config.yaml

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYSCF_MAX_MEMORY=4000 \
./.venv/bin/mldft_labelgen preset=horm_pbe_labels \
  n_molecules=1 start_idx=0 num_threads_per_process=2 \
  hydra.run.dir=/mnt/afs/home/xiazhenhao/dft/logs/horm_pilot_labelgen \
  log_file=/mnt/afs/home/xiazhenhao/dft/logs/horm_pilot_labelgen.log \
  config_file=/mnt/afs/home/xiazhenhao/dft/logs/horm_pilot_labelgen_config.yaml
```

检查输出：

```bash
./.venv/bin/python scripts/check_horm_labels.py \
  /mnt/afs/home/xiazhenhao/dft/structures25/_runtime/horm_pbe_labels/HORM_TS1xVal/labels/0017304.zarr.zip
```

默认 pilot 输出：

```text
/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/horm_pbe_labels/HORM_TS1xVal/kohn_sham/horm_0017304.chk
/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/horm_pbe_labels/HORM_TS1xVal/labels/0017304.zarr.zip
```

## 开启 PBE force/Hessian

只建议在小 pilot 上开启 Hessian：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYSCF_MAX_MEMORY=4000 \
./.venv/bin/mldft_ks preset=horm_pbe_labels \
  dataset.name=HORM_TS1xVal_PBEHessPilot \
  kohn_sham.compute_forces=true kohn_sham.compute_hessian=true \
  n_molecules=1 start_idx=0 num_threads_per_process=2 \
  hydra.run.dir=/mnt/afs/home/xiazhenhao/dft/logs/horm_pilot_pbe_hess_ks \
  log_file=/mnt/afs/home/xiazhenhao/dft/logs/horm_pilot_pbe_hess_ks.log \
  config_file=/mnt/afs/home/xiazhenhao/dft/logs/horm_pilot_pbe_hess_ks_config.yaml

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYSCF_MAX_MEMORY=4000 \
./.venv/bin/mldft_labelgen preset=horm_pbe_labels \
  dataset.name=HORM_TS1xVal_PBEHessPilot \
  kohn_sham.compute_forces=true kohn_sham.compute_hessian=true \
  n_molecules=1 start_idx=0 num_threads_per_process=2 \
  hydra.run.dir=/mnt/afs/home/xiazhenhao/dft/logs/horm_pilot_pbe_hess_labelgen \
  log_file=/mnt/afs/home/xiazhenhao/dft/logs/horm_pilot_pbe_hess_labelgen.log \
  config_file=/mnt/afs/home/xiazhenhao/dft/logs/horm_pilot_pbe_hess_labelgen_config.yaml
```

如果 Hessian 或 force 失败，SCF `.chk` 会保留，错误写入日志；若 labelgen 能继续，则错误信息会进入 `metadata/pbe_derivative_errors`。

## 全量命令

不要在没有资源评估前直接全量跑。确认 pilot、并决定并行度和存储预算后，TS1x-val 全量可用：

```bash
./.venv/bin/mldft_ks preset=horm_pbe_labels \
  dataset.sample_ids=null n_molecules=-1 start_idx=0 \
  num_processes=<NPROC> num_threads_per_process=<NTHREADS> \
  hydra.run.dir=/mnt/afs/home/xiazhenhao/dft/logs/horm_ts1x_val_ks

./.venv/bin/mldft_labelgen preset=horm_pbe_labels \
  dataset.sample_ids=null n_molecules=-1 start_idx=0 \
  num_processes=<NPROC> num_threads_per_process=<NTHREADS> \
  hydra.run.dir=/mnt/afs/home/xiazhenhao/dft/logs/horm_ts1x_val_labelgen
```

TS1x train 全量要显式切换 archive 和输出 dataset name：

```bash
./.venv/bin/mldft_ks preset=horm_pbe_labels \
  dataset.name=HORM_TS1xTrain \
  dataset.archive_file=ts1x_hess_train.lmdb \
  dataset.dataset_name=ts1x_train \
  dataset.sample_ids=null n_molecules=-1 start_idx=0 \
  num_processes=<NPROC> num_threads_per_process=<NTHREADS> \
  hydra.run.dir=/mnt/afs/home/xiazhenhao/dft/logs/horm_ts1x_train_ks

./.venv/bin/mldft_labelgen preset=horm_pbe_labels \
  dataset.name=HORM_TS1xTrain \
  dataset.archive_file=ts1x_hess_train.lmdb \
  dataset.dataset_name=ts1x_train \
  dataset.sample_ids=null n_molecules=-1 start_idx=0 \
  num_processes=<NPROC> num_threads_per_process=<NTHREADS> \
  hydra.run.dir=/mnt/afs/home/xiazhenhao/dft/logs/horm_ts1x_train_labelgen
```

RGD1 可将 `dataset.archive_file=RGD1.lmdb`、`dataset.name=HORM_RGD1`、`dataset.dataset_name=rgd1`。

## 可用于训练的标签

可用于 STRUCTURES25/M-OFDFT 训练：

```text
geometry/*
ks_labels/energies/*
of_labels/energies/*
of_labels/spatial/coeffs
of_labels/spatial/grad_kin
of_labels/spatial/grad_kin_plus_xc
of_labels/spatial/grad_tot
```

仅作为 metadata/reference：

```text
metadata/reference/*
```

可选、同 PBE level 的 force/Hessian：

```text
metadata/pbe_derivatives/*
```

`metadata/reference/*` 来自 HORM archive 的 `omegaB97x/6-31G(d)`，不能和本流程生成的 `PBE/6-31G(2df,p)` labels 混成同一个 supervised target。
