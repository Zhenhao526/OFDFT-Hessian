# HORM 50 样本 Pilot 报告

更新时间：2026-07-01

## 运行配置

输入数据：`/mnt/afs/home/xiazhenhao/dft/archive/ts1x-val.lmdb`

KSDFT 输出：`/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/horm_pbe_labels/HORM_TS1xVal/kohn_sham`

标签输出：`/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/horm_pbe_labels/HORM_TS1xVal/labels`

NPROC=8, NTHREADS=2, PYSCF_MAX_MEMORY=4000 MB

KSDFT 命令：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYSCF_MAX_MEMORY=4000 ./.venv/bin/mldft_ks preset=horm_pbe_labels dataset.sample_ids=null n_molecules=50 start_idx=0 num_processes=8 num_threads_per_process=2 max_memory_per_process=4000 hydra.run.dir=/mnt/afs/home/xiazhenhao/dft/logs/horm_50_ks log_file=/mnt/afs/home/xiazhenhao/dft/logs/horm_50_ks.log
```

labelgen 命令：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYSCF_MAX_MEMORY=4000 ./.venv/bin/mldft_labelgen preset=horm_pbe_labels dataset.sample_ids=null n_molecules=50 start_idx=0 num_processes=8 num_threads_per_process=2 max_memory_per_process=4000 hydra.run.dir=/mnt/afs/home/xiazhenhao/dft/logs/horm_50_labelgen log_file=/mnt/afs/home/xiazhenhao/dft/logs/horm_50_labelgen.log
```

## 成功/失败统计

- 期望样本数：50
- 成功标签数：50
- 失败数：0
- failure CSV：`/mnt/afs/home/xiazhenhao/dft/logs/horm_50_failures.csv`
- 统计 JSON：`/mnt/afs/home/xiazhenhao/dft/logs/horm_50_summary.json`

## 数值健康检查

- NaN 数量：0
- Inf 数量：0
- `metadata/reference/forces` 覆盖：50/50
- `metadata/reference/hessian` 覆盖：50/50
- PBE force/Hessian 重算标签：0/50（本 pilot 默认关闭）

## 元素组成统计

| composition | count |
| --- | --- |
| H10C4O1 | 1 |
| H10C4O2 | 2 |
| H10C5O1 | 2 |
| H10C5O2 | 1 |
| H10C6 | 1 |
| H10C6O1 | 2 |
| H11C6N1 | 2 |
| H12C6 | 1 |
| H12C7 | 1 |
| H3C3N3O1 | 1 |
| H3C5N1O1 | 1 |
| H4C3N2O1 | 1 |
| H4C4N2 | 1 |
| H4C5N2 | 1 |
| H4C5O2 | 1 |
| H5C3N1O2 | 1 |
| H5C4N1O1 | 1 |
| H5C4N1O2 | 2 |
| H5C5N1O1 | 1 |
| H6C3N2O1 | 1 |
| H6C3O3 | 1 |
| H6C4N2O1 | 2 |
| H6C4O2 | 1 |
| H6C5O1 | 1 |
| H6C5O2 | 3 |
| H7C3N1O2 | 1 |
| H7C4N1O1 | 1 |
| H7C4N1O2 | 1 |
| H7C5N1 | 1 |
| H8C4N2O1 | 2 |
| H8C4O1 | 1 |
| H8C4O2 | 2 |
| H8C5 | 1 |
| H8C5N2 | 1 |
| H8C5O1 | 1 |
| H8C5O2 | 2 |
| H8C6 | 1 |
| H8C6O1 | 1 |
| H9C5N1O1 | 1 |

## 原子数分布

范围：10 - 19

| natoms | count |
| --- | --- |
| 10 | 4 |
| 11 | 4 |
| 12 | 7 |
| 13 | 10 |
| 14 | 5 |
| 15 | 7 |
| 16 | 6 |
| 17 | 3 |
| 18 | 3 |
| 19 | 1 |

## Density Coefficient 维度分布

范围：705 - 1003

| n_coeffs | count |
| --- | --- |
| 705 | 1 |
| 712 | 1 |
| 748 | 1 |
| 752 | 1 |
| 755 | 1 |
| 768 | 1 |
| 775 | 1 |
| 781 | 1 |
| 788 | 1 |
| 795 | 2 |
| 801 | 1 |
| 808 | 1 |
| 814 | 1 |
| 815 | 1 |
| 821 | 1 |
| 828 | 2 |
| 837 | 1 |
| 851 | 1 |
| 854 | 1 |
| 857 | 2 |
| 861 | 2 |
| 868 | 2 |
| 877 | 1 |
| 884 | 2 |
| 894 | 1 |
| 897 | 3 |
| 904 | 2 |
| 924 | 1 |
| 930 | 1 |
| 937 | 3 |
| 944 | 2 |
| 957 | 1 |
| 970 | 2 |
| 977 | 1 |
| 990 | 2 |
| 1003 | 1 |

## SCF Iteration 分布

范围：11 - 18

| n_scf_steps | count |
| --- | --- |
| 11 | 2 |
| 12 | 10 |
| 13 | 19 |
| 14 | 14 |
| 15 | 4 |
| 18 | 1 |

## Energy Label 范围

| label | min | max |
| --- | --- | --- |
| ks_labels/energies/e_electron | -687.931127027 | 0 |
| ks_labels/energies/e_ext | -1506.08419653 | 0 |
| ks_labels/energies/e_hartree | 0 | 504.348053925 |
| ks_labels/energies/e_kin | 0 | 362.451982349 |
| ks_labels/energies/e_nuc_nuc | 159.12002353 | 326.612497255 |
| ks_labels/energies/e_tot | -361.318629772 | 0 |
| ks_labels/energies/e_xc | -48.5091123828 | 0 |
| of_labels/energies/e_electron | -687.931127027 | 0 |
| of_labels/energies/e_ext | -1506.08419653 | 0 |
| of_labels/energies/e_hartree | 0 | 504.347988899 |
| of_labels/energies/e_kin | 0 | 362.45203836 |
| of_labels/energies/e_kin_minus_apbe | -0.610763681101 | 0.415192043544 |
| of_labels/energies/e_kin_plus_xc | 0 | 314.001004751 |
| of_labels/energies/e_kinapbe | 0 | 362.068674868 |
| of_labels/energies/e_tot | -361.318629772 | 0 |
| of_labels/energies/e_xc | -48.5091033679 | 0 |

## Gradient 统计

`grad_kin_plus_xc`：

```json
{
  "count": 574132,
  "min": -5.947393017289719,
  "max": 11.357173018650371,
  "mean": 0.6387211468951426,
  "std": 2.0573274495612983
}
```

`grad_tot`：

```json
{
  "count": 574132,
  "min": -8.00442194966719,
  "max": 5.140870165098249,
  "mean": -4.311074509936865e-05,
  "std": 0.1079632715987685
}
```

异常大梯度判断：未发现，max_abs <= 1e6。

## 失败样本

无失败样本。

## 扩大建议

- 建议进入 500/1000 样本阶段：是
- 建议直接全量跑：否

当前 50 样本 pilot 成功率、NaN/Inf 检查和 reference metadata 覆盖都支持进入 500/1000 样本阶段；但不建议直接全量跑，应先用 500/1000 样本验证长尾原子数、SCF 收敛、labelgen 内存和存储增长。

## DFT Level 警告

HORM archive 原始 force/Hessian 来源标注为 `omegaB97x/6-31G(d)`。本流程新生成的 STRUCTURES25/M-OFDFT 标签是 `PBE/6-31G(2df,p)`。`metadata/reference/*` 中的原始 HORM energy/force/Hessian 不能和 PBE 标签混合作为同一训练目标。
