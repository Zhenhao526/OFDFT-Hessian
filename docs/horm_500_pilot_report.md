# HORM 500 样本 Pilot 报告

更新时间：2026-07-01

## 运行配置

输入数据：`/mnt/afs/home/xiazhenhao/dft/archive/ts1x-val.lmdb`

KSDFT 输出：`/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/horm_pbe_labels/HORM_TS1xVal/kohn_sham`

标签输出：`/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/horm_pbe_labels/HORM_TS1xVal/labels`

NPROC=16, NTHREADS=2, PYSCF_MAX_MEMORY=4000 MB

KSDFT 命令：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYSCF_MAX_MEMORY=4000 ./.venv/bin/mldft_ks preset=horm_pbe_labels dataset.sample_ids=null n_molecules=500 start_idx=0 num_processes=16 num_threads_per_process=2 max_memory_per_process=4000 hydra.run.dir=/mnt/afs/home/xiazhenhao/dft/logs/horm_500_ks log_file=/mnt/afs/home/xiazhenhao/dft/logs/horm_500_ks.log
```

labelgen 命令：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYSCF_MAX_MEMORY=4000 ./.venv/bin/mldft_labelgen preset=horm_pbe_labels dataset.sample_ids=null n_molecules=500 start_idx=0 num_processes=16 num_threads_per_process=2 max_memory_per_process=4000 hydra.run.dir=/mnt/afs/home/xiazhenhao/dft/logs/horm_500_labelgen log_file=/mnt/afs/home/xiazhenhao/dft/logs/horm_500_labelgen.log
```

## 成功/失败统计

- 期望样本数：500
- 成功标签数：500
- 失败数：0
- failure CSV：`/mnt/afs/home/xiazhenhao/dft/logs/horm_500_failures.csv`
- 统计 JSON：`/mnt/afs/home/xiazhenhao/dft/logs/horm_500_summary.json`

## 存储占用与全量外推

统计口径：仅统计 sample id `0-499` 的 KS `.chk` 和 label `.zarr.zip` 文件；不包含早期 smoke test 样本或日志目录。

| item | 500 samples | avg/sample | TS1x-val 50,845 estimate |
| --- | ---: | ---: | ---: |
| KS `.chk` | 2.614 GB (2.434 GiB) | 5.228 MB | 265.817 GB (247.562 GiB) |
| labels `.zarr.zip` | 0.441 GB (0.411 GiB) | 0.883 MB | 44.896 GB (41.813 GiB) |
| total | 3.055 GB (2.846 GiB) | 6.111 MB | 310.713 GB (289.374 GiB) |

建议为完整 `ts1x-val.lmdb` 预留至少 350-400 GB，给文件系统开销、日志、失败重试和样本尺寸波动留余量。如果 labelgen 后确认不再需要 SCF `.chk`，只长期保留 labels 的占用约为 45 GB；但删除 `.chk` 会降低后续重算、审计和追加标签的便利性。

archive 目录中 `ts1x_hess_train.lmdb` 有 1,725,363 个样本。如果用本 500 样本均值线性外推，train archive 约需要 KS `.chk` 9.020 TB、labels 1.523 TB、合计 10.544 TB（约 9.589 TiB）。这是按 val 前 500 样本的单样本均值估算，正式扩大到 train 前应先抽样 train shard 复核尺寸和收敛率。

本次 500 pilot 复用了前 50 个已生成样本，新增计算 450 个样本。新增 KSDFT 用时约 21 分 52 秒，新增 labelgen 进度用时约 44 分 57 秒，尾部写入和收尾后正常退出。

## 数值健康检查

- NaN 数量：0
- Inf 数量：0
- `metadata/reference/forces` 覆盖：500/500
- `metadata/reference/hessian` 覆盖：500/500
- PBE force/Hessian 重算标签：0/500（本 pilot 默认关闭）

## 元素组成统计

| composition | count |
| --- | --- |
| C4N2 | 1 |
| H10C4O1 | 3 |
| H10C4O2 | 5 |
| H10C5N2 | 1 |
| H10C5O1 | 10 |
| H10C5O2 | 14 |
| H10C6 | 8 |
| H10C6O1 | 18 |
| H10C7 | 5 |
| H11C5N1 | 4 |
| H11C5N1O1 | 4 |
| H11C6N1 | 6 |
| H12C5O1 | 1 |
| H12C6 | 4 |
| H12C6O1 | 12 |
| H12C7 | 8 |
| H14C6O1 | 1 |
| H1C2N1O1 | 1 |
| H2C1N4O1 | 2 |
| H2C2N2O1 | 2 |
| H2C3O3 | 1 |
| H2C4O2 | 2 |
| H3C2N1O2 | 1 |
| H3C3N1O1 | 1 |
| H3C3N1O2 | 6 |
| H3C3N3O1 | 5 |
| H3C4N1O1 | 1 |
| H3C4N1O2 | 3 |
| H3C4N3 | 1 |
| H3C5N1O1 | 2 |
| H4C2N2O1 | 1 |
| H4C2N2O2 | 4 |
| H4C2N4 | 3 |
| H4C2O2 | 2 |
| H4C3N2 | 1 |
| H4C3N2O1 | 6 |
| H4C3N2O2 | 1 |
| H4C3O3 | 5 |
| H4C4N2 | 3 |
| H4C4N2O1 | 2 |
| H4C4O2 | 2 |
| H4C4O3 | 1 |
| H4C5N2 | 3 |
| H4C5O2 | 1 |
| H5C3N1O1 | 3 |
| H5C3N1O2 | 7 |
| H5C3N3 | 6 |
| H5C3N3O1 | 1 |
| H5C4N1O1 | 12 |
| H5C4N1O2 | 9 |
| H5C4N3 | 2 |
| H5C5N1 | 2 |
| H5C5N1O1 | 6 |
| H6C3N2 | 2 |
| H6C3N2O1 | 5 |
| H6C3O2 | 1 |
| H6C3O3 | 2 |
| H6C4N2 | 5 |
| H6C4N2O1 | 6 |
| H6C4O1 | 4 |
| H6C4O2 | 17 |
| H6C5N2 | 2 |
| H6C5O1 | 8 |
| H6C5O2 | 9 |
| H6C6O1 | 1 |
| H7C3N1O1 | 2 |
| H7C3N1O2 | 2 |
| H7C3N3 | 7 |
| H7C3N3O1 | 1 |
| H7C4N1O1 | 15 |
| H7C4N1O2 | 11 |
| H7C4N3 | 7 |
| H7C5N1 | 6 |
| H7C5N1O1 | 17 |
| H8C3 | 1 |
| H8C3N2O1 | 1 |
| H8C3O2 | 2 |
| H8C4N2 | 6 |
| H8C4N2O1 | 14 |
| H8C4O1 | 5 |
| H8C4O2 | 18 |
| H8C4O3 | 12 |
| H8C5 | 3 |
| H8C5N2 | 11 |
| H8C5O1 | 16 |
| H8C5O2 | 10 |
| H8C6 | 2 |
| H8C6O1 | 9 |
| H8C7 | 1 |
| H9C4N1O1 | 9 |
| H9C4N1O2 | 14 |
| H9C5N1 | 1 |
| H9C5N1O1 | 12 |
| H9C6N1 | 7 |

## 原子数分布

范围：5 - 21

| natoms | count |
| --- | --- |
| 5 | 1 |
| 6 | 1 |
| 7 | 2 |
| 8 | 9 |
| 9 | 9 |
| 10 | 37 |
| 11 | 43 |
| 12 | 57 |
| 13 | 58 |
| 14 | 79 |
| 15 | 70 |
| 16 | 56 |
| 17 | 42 |
| 18 | 15 |
| 19 | 20 |
| 21 | 1 |

## Density Coefficient 维度分布

范围：470 - 1050

| n_coeffs | count |
| --- | --- |
| 470 | 1 |
| 487 | 1 |
| 530 | 2 |
| 606 | 2 |
| 619 | 1 |
| 626 | 1 |
| 639 | 1 |
| 646 | 1 |
| 659 | 3 |
| 668 | 1 |
| 672 | 4 |
| 679 | 3 |
| 699 | 2 |
| 705 | 3 |
| 708 | 2 |
| 712 | 5 |
| 715 | 1 |
| 719 | 2 |
| 728 | 1 |
| 729 | 2 |
| 735 | 6 |
| 748 | 5 |
| 752 | 3 |
| 755 | 11 |
| 761 | 2 |
| 762 | 7 |
| 768 | 12 |
| 775 | 13 |
| 781 | 8 |
| 788 | 22 |
| 795 | 7 |
| 801 | 6 |
| 808 | 15 |
| 814 | 2 |
| 815 | 9 |
| 821 | 16 |
| 828 | 24 |
| 835 | 1 |
| 837 | 2 |
| 841 | 1 |
| 844 | 4 |
| 848 | 9 |
| 851 | 5 |
| 854 | 8 |
| 857 | 4 |
| 861 | 10 |
| 864 | 3 |
| 868 | 5 |
| 871 | 1 |
| 877 | 6 |
| 881 | 4 |
| 884 | 11 |
| 890 | 1 |
| 891 | 1 |
| 894 | 4 |
| 897 | 11 |
| 901 | 1 |
| 904 | 6 |
| 917 | 17 |
| 923 | 1 |
| 924 | 18 |
| 930 | 9 |
| 931 | 1 |
| 937 | 21 |
| 944 | 26 |
| 950 | 7 |
| 957 | 12 |
| 963 | 5 |
| 964 | 14 |
| 970 | 18 |
| 977 | 15 |
| 990 | 6 |
| 997 | 4 |
| 1003 | 8 |
| 1010 | 12 |
| 1050 | 1 |

## SCF Iteration 分布

范围：11 - 29

| n_scf_steps | count |
| --- | --- |
| 11 | 19 |
| 12 | 126 |
| 13 | 169 |
| 14 | 141 |
| 15 | 31 |
| 16 | 7 |
| 17 | 2 |
| 18 | 3 |
| 25 | 1 |
| 29 | 1 |

## Energy Label 范围

| label | min | max |
| --- | --- | --- |
| ks_labels/energies/e_electron | -709.696951394 | 0 |
| ks_labels/energies/e_ext | -1562.40344107 | 0 |
| ks_labels/energies/e_hartree | 0 | 521.361754848 |
| ks_labels/energies/e_kin | 0 | 386.336192784 |
| ks_labels/energies/e_nuc_nuc | 74.5026225161 | 341.797062796 |
| ks_labels/energies/e_tot | -382.477676709 | 0 |
| ks_labels/energies/e_xc | -50.9521962224 | 0 |
| of_labels/energies/e_electron | -709.696951394 | 0 |
| of_labels/energies/e_ext | -1562.40344107 | 0 |
| of_labels/energies/e_hartree | 0 | 521.361683391 |
| of_labels/energies/e_kin | 0 | 386.336873036 |
| of_labels/energies/e_kin_minus_apbe | -0.662516532898 | 0.724972164807 |
| of_labels/energies/e_kin_plus_xc | 0 | 335.384083912 |
| of_labels/energies/e_kinapbe | 0 | 386.131065565 |
| of_labels/energies/e_tot | -382.477676709 | 0 |
| of_labels/energies/e_xc | -50.952789124 | 0 |

## Gradient 统计

`grad_kin_plus_xc`：

```json
{
  "count": 5695702,
  "min": -6.351132960227128,
  "max": 11.367953673638274,
  "mean": 0.6390027424795968,
  "std": 2.0601511533980448
}
```

`grad_tot`：

```json
{
  "count": 5695702,
  "min": -18.33676480781179,
  "max": 10.688687828553142,
  "mean": -8.563342217514231e-05,
  "std": 0.11389906747953614
}
```

异常大梯度判断：未发现，max_abs <= 1e6。

## 失败样本

无失败样本。

## 扩大建议

- 建议进入 1000/5000 样本阶段：是
- 建议直接全量跑：否

当前 500 样本 pilot 成功率、NaN/Inf 检查和 reference metadata 覆盖都支持进入更大的分批任务。建议下一步先跑 1000 或 5000 样本 shard，继续观察长尾原子数、SCF 收敛、labelgen 内存和存储增长；全量任务应分 shard 跑并保留 failure CSV，不建议一次性无分片启动。

## DFT Level 警告

HORM archive 原始 force/Hessian 来源标注为 `omegaB97x/6-31G(d)`。本流程新生成的 STRUCTURES25/M-OFDFT 标签是 `PBE/6-31G(2df,p)`。`metadata/reference/*` 中的原始 HORM energy/force/Hessian 不能和 PBE 标签混合作为同一训练目标。
