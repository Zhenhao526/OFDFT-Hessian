# QM9 OFDFT Force/Hessian P1-P2 任务书

日期：2026-07-01

最近更新：2026-07-02 21:51 Asia/Singapore

## 1. 任务背景

本任务承接 `qm9_ofdft_force_hessian_task.md`，目标是在 STRUCTURES25/OFDFT 框架中验证：

```math
\mathrm{HessianError}(\mathrm{OFDFT\text{-}EGF})
<
\mathrm{HessianError}(\mathrm{OFDFT\text{-}EG})
```

其中：

- OFDFT-EG：energy + density-gradient baseline。
- OFDFT-EGF：energy + density-gradient + force。
- force 必须来自同一个标量能量泛函的坐标导数：

```math
F_{\mathrm{pred}} = -\frac{\partial E_{\mathrm{OFDFT}}}{\partial R}
```

本阶段不追求过渡态泛化，核心是验证同一 PBE level 下 force 监督是否能改善 OFDFT 对 Hessian/曲率的刻画。

## 2. 已完成结果：P0 Force Smoke

P0 已完成，详细报告见：

```text
structures25/docs/qm9_p0_force_smoke_report.md
```

### 2.1 数据与标签

- QM9 原始 `.xyz` 已下载并解压完成：`133885` 个文件。
- P0 使用前 100 个 QM9 molecule。
- 每个 molecule 生成 2 个 sample：
  - sample `0`：reference geometry；
  - sample `1`：deterministic Gaussian perturbation。
- DFT level：PBE/6-31G(2df,p)。
- Hessian：P0 未开启。
- Force label：已开启，保存于：

```text
metadata/pbe_derivatives/forces
```

### 2.2 P0 产物

运行目录：

```text
structures25/_runtime/qm9_p0
```

关键产物：

| 产物 | 数量 | 状态 |
|---|---:|---|
| QM9 raw xyz | 133885 | 已完成 |
| KS `.chk` | 200 | 已完成 |
| label `.zarr.zip` | 200 | 已完成 |
| force label | 200/200 | 已验证 |
| NaN/Inf 检查 | 200/200 | 通过 |

P0 检查结果：

```json
{
  "checked_files": 200,
  "expected_molecules": 100,
  "expected_samples": 2,
  "failures": [],
  "force_labels": 200,
  "force_norm_max": 0.09798267278000408,
  "molecules": 100
}
```

最小 force-supervision forward/backward smoke：

```json
{
  "bias": 0.0,
  "final_loss": 0.0006080805502358812,
  "initial_loss": 0.0008129049017408304,
  "samples": 8,
  "scale": 0.0070316326768313136,
  "steps": 3
}
```

### 2.3 P0 已建立的能力

- QM9 reader 已支持 `*^` 科学计数法读入，无需对全量 raw xyz 做原地转换。
- 已新增 QM9 geometry perturbation 数据集。
- datagen 已支持同一 molecule 的多 sample。
- `.chk` 与 `.zarr.zip` 已按 sample 写出，避免 reference/perturbation 覆盖。
- labelgen 已能为对应 sample 读取相同几何并写入 force label。
- 已有 force label 检查脚本和最小 forward/backward smoke 脚本。

### 2.4 当前限制

- `/tmp` 仍然满，P0 使用以下路径绕过：

```text
TMPDIR=/mnt/afs/home/xiazhenhao/dft/_tmp
```

- P0 的 forward/backward smoke 是最小独立脚本，不等价于正式 OFDFT 训练链路。
- P1 最大风险不是 label 生成，而是正式训练中：
  - dataloader 正确读取 force；
  - 坐标 `R` 支持 autograd；
  - `F_pred = -dE/dR` 能稳定求导；
  - force loss 不破坏原有 energy/density-gradient 训练。

## 3. 阶段定义

| 阶段 | 目标规模 | 样本设计 | 核心目的 | 状态 |
|---|---:|---|---|---|
| P0 Smoke | 100 molecule | reference + 1 perturbation | 验证 datagen + force label + 最小 backward | 已完成 |
| P1 Pilot | 1000 molecule | reference + 3 perturbations | 验证正式训练链路和 force loss 稳定性 | 进行中：P1-0/P1-1 已完成，P1-2 已完成 410/1000 molecule，P1-4 410 molecule smoke 已通过 |
| P2 Main-small | 10000 molecule | reference + 3-5 perturbations | EG vs EGF Hessian 定量对照 | 待执行 |

说明：原任务书中的“扰动数”指 perturbation 数。若保留 reference，则每个 molecule 的总 sample 数为 `1 + num_perturbations`。

## 4. P1 Pilot 任务

P1 目标是完成 1000 molecule 规模的 pilot，并把 force supervision 接入正式 OFDFT 训练链路。

当前状态（2026-07-02 21:40 Asia/Singapore）：

- P1-0 已完成：datagen preset、ML data config、EG/EGF experiment config、molecule-level split 均已接入。
- P1-1 已完成核心链路：正式 dataloader 可读 force label，`MLDFTLitModule` 可由 scalar energy 计算 `-dE/dR`，`ForceLoss` 可参与 backward。
- P1-2 进行中：已完成 410 molecule / 1640 labels，距离 1000 molecule pilot 还差 590 molecule / 2360 labels。
- P1-3 在 410 molecule 子集上通过，1000 molecule 全量完成后需重跑全局检查。
- P1-4 在 410 molecule 子集上已通过正式 EG/EGF 短训练 smoke；1000 molecule pilot 完成后需跑完整训练。
- P1-5 尚未开始。

### P1-0 配置与划分

目标：

- 新增 P1 pilot preset。
- 新增 molecule-level split。
- 确保同一 molecule 的所有 perturbation 只进入 train/val/test 之一。

建议配置：

```yaml
n_molecules: 1000
include_reference: true
num_perturbations: 3
perturbation_std_angstrom: 0.01-0.05
dft_level: PBE/6-31G(2df,p)
compute_forces: true
compute_hessian: false
```

预期 label 数：

```text
1000 molecule * 4 samples = 4000 labels
```

交付物：

- `configs/datagen/preset/qm9_pbe_force_pilot.yaml`
- molecule-level split 文件
- P1 runtime root，不能混入 P0 目录

验收：

- split 可复现；
- train/validation/test 无 molecule ID 泄漏；
- 配置能在小切片上正常启动。

当前结果：

- `configs/datagen/preset/qm9_pbe_force_pilot.yaml` 已新增。
- `configs/ml/data/qm9_pbe_force_pilot.yaml` 已新增。
- `configs/ml/experiment/str25/qm9_pbe_force_pilot_eg.yaml` 已新增。
- `configs/ml/experiment/str25/qm9_pbe_force_pilot_egf.yaml` 已新增。
- `mldft/utils/create_dataset_splits.py --group-by-molecule` 已新增。
- 410 molecule grouped split 已验证无泄漏：train/val/test = 328/41/41 molecule。

### P1-1 正式训练链路接入 force

目标：

- 在正式训练 dataloader 中读取 force label。
- 在正式 OFDFT 模型中通过 scalar energy 对坐标求导得到 predicted force。
- 添加 OFDFT-EGF 训练损失。

必须满足：

```math
F_{\mathrm{pred}} = -\frac{\partial E_{\mathrm{OFDFT}}}{\partial R}
```

不应新增独立 force head 作为主方案。

需要检查：

- 坐标单位是否与 force label 匹配；
- PySCF force label 的单位；
- zarr 中 geometry 的单位；
- batch 内不同原子数分子的 padding/mask；
- 坐标 tensor 是否 `requires_grad=True`；
- density optimization 或 energy evaluation 是否保留坐标梯度链路。

建议损失：

```math
L =
\lambda_E L_E
+ \lambda_p L_{\partial E/\partial p}
+ \lambda_F L_F
```

其中：

```math
L_F =
\left\|
F_{\mathrm{pred}} - F_{\mathrm{PBE}}
\right\|
```

交付物：

- force label dataloader 支持；
- OFDFT-EGF training config；
- force loss 实现；
- force loss 单元测试或 smoke 测试。

验收：

- 使用 P0 的 200 个 labels，正式训练代码能跑通 forward/backward；
- force loss finite；
- force loss 在小步训练中不发散；
- OFDFT-EG baseline 不受破坏。

当前结果：

- `OFData.from_file(load_force_label=True)` 可读取 `metadata/pbe_derivatives/forces`。
- `MLDFTLitModule(force_supervision=True)` 可输出 `pred_forces = -dE/dR`。
- `ForceLoss` 已实现并有测试覆盖。
- P0 真实 labels 和 P1 labels 上的正式 MLDFT force smoke 均已通过。

### P1-2 生成 1000 molecule pilot labels

目标：

- 生成 P1 所需 KS `.chk` 和 label `.zarr.zip`。
- 生成过程必须可恢复、可分批。

执行要求：

- 复用 P0 已下载 QM9 raw 数据。
- 每次运行显式设置：

```text
TMPDIR=/mnt/afs/home/xiazhenhao/dft/_tmp
```

- 建议按 100-200 molecule 分批运行。
- 每批完成后立即运行 label 检查。

当前进展：

| 项目 | 数量 | 状态 |
|---|---:|---|
| molecule | 410/1000 | 进行中 |
| KS `.chk` | 1640/4000 | 已完成子集 |
| label `.zarr.zip` | 1640/4000 | 已完成子集 |
| cached transformed label | 1640/4000 | 已完成子集 |
| force label | 1640/1640 | 子集已验证 |

410 molecule 检查结果：

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

410 molecule 节点的断点恢复结果：

- 本批从 `start_idx=310` 恢复并补齐 molecule ids 311-410。
- KS 初始运行因 `_tmp` 被清理出现一次临时文件 `FileNotFoundError`，停在 `1536` 个 `.chk`；恢复运行识别 `73` 个 molecule 已完成、补算 `27` 个 molecule，最终 `.chk` 为 `1640`。
- labelgen 最终产物为 `1640` labels，log dir 为 `_runtime/qm9_p1/QM9PBEForcePilot/logs/2026-07-02_18-19-27`。
- grouped split 已重建且无 molecule 泄漏：train/val/test = 1312/164/164 labels，对应 328/41/41 molecules。
- cached transform 已更新到 1640 files。
- cached smoke-level statistics 已刷新到 dataset 旁边的 `dataset_statistics_labels_local_frames_global_symmetric_natrep_e_kin_plus_xc.zarr`。

交付物：

- 约 4000 个 `.zarr.zip` labels；
- 对应 datagen 日志；
- 每批检查 summary。

验收：

- molecule 成功率 >95%；
- labels 无 NaN/Inf；
- force label 读取成功率 >95%；
- 失败 molecule 有明确日志和清单。

### P1-3 P1 label 质量检查

目标：

- 对 P1 labels 做自动检查，防止训练阶段才发现坏样本。

检查项：

- zarr 可打开；
- 数值数组无 NaN/Inf；
- force label 存在；
- force shape 与原子数一致；
- force norm 分布合理；
- reference 与 perturbation sample 数量符合预期。

交付物：

- `qm9_p1_force_pilot_check.json`
- `qm9_p1_failed_molecules.txt`

验收：

- 检查脚本返回 failures 数量可解释；
- 坏样本不会进入训练 split。

当前结果：

- 410 molecule 子集已通过 `scripts/check_qm9_force_smoke.py`。
- 当前 failures 为 `[]`。
- 全量 1000 molecule pilot 完成后仍需输出全局 `qm9_p1_force_pilot_check.json`。

### P1-4 训练 OFDFT-EG 与 OFDFT-EGF

目标：

- 在 P1 数据上训练两个模型。

模型：

| 模型 | 监督信号 | 用途 |
|---|---|---|
| OFDFT-EG | energy + density-gradient | baseline |
| OFDFT-EGF | energy + density-gradient + force | 验证 force supervision |

建议执行顺序：

1. 在 P0 200 labels 上跑正式训练 smoke。
2. 在 P1 train split 上跑短训练。
3. 调整 `lambda_F`，避免 force loss 主导全部梯度。
4. 跑完整 pilot training。

交付物：

- OFDFT-EG P1 checkpoint；
- OFDFT-EGF P1 checkpoint；
- training logs；
- validation metrics；
- force loss 曲线。

验收：

- 两个模型均能完整训练；
- EGF force loss 有下降趋势；
- EG baseline 指标没有因新代码退化；
- 训练过程无坐标 autograd/memory blocker。

当前结果：

- 1640 个 P1 labels 已缓存为 `labels_local_frames_global_symmetric_natrep`，用于正式 Graphformer 训练。
- cached labels force label 保留在 `metadata/pbe_derivatives/forces`。
- 410 molecule cached statistics 已生成：

```text
_runtime/qm9_p1/QM9PBEForcePilot/dataset_statistics/dataset_statistics_labels_local_frames_global_symmetric_natrep_e_kin_plus_xc.zarr
```

- statistics run dir：

```text
_runtime/qm9_p1_models/statistics/runs/006_qm9_p1_410_stats_cached_batch_size-16__num_workers-0__qm9_pbe_force_pilot__enforce_tags-false__print_config-false__overwrite-true__n_batches-16
```

- `str25/qm9_pbe_force_pilot_eg` 410 molecule 短训练 smoke 已通过：2 train batches + 1 val batch，`val_loss/total = 0.4906010329723358`。
- `str25/qm9_pbe_force_pilot_egf` 410 molecule 短训练 smoke 已通过：2 train batches + 1 val batch，`val_loss/force_loss = 0.0016242226120084524`。
- 详细记录：`docs/qm9_p1_410_training_smoke.md`。
- 在线 transform statistics 曾因 local frame dummy atom 随机性触发均值断言；缓存 transformed labels 后已解决。

### P1-5 Hessian mini-eval sanity check

目标：

- 在至少 100 个 held-out molecule 上跑 Hessian mini-eval。
- 先验证流程，不追求最终统计显著性。

推荐方式：

- 使用 predicted force 的有限差分：

```math
H_{ij} \approx
-\frac{F_i(R+\delta e_j)-F_i(R-\delta e_j)}{2\delta}
```

执行步骤：

1. 对每个测试结构的每个坐标方向做 `+delta/-delta` 位移；
2. 每个位移结构运行 OFDFT density optimization；
3. 得到 predicted force；
4. 由 force 差分得到 predicted Hessian；
5. 与 PBE Hessian 或可用 reference 对比。

交付物：

- Hessian mini-eval 脚本；
- 100 molecule mini-eval summary；
- EG vs EGF 初步对比表。

验收：

- 100 个测试样本至少大部分能完成；
- Hessian matrix 维度和质量检查通过；
- 能输出 EG vs EGF 的初步 Hessian error。

## 5. P2 Main-small 任务

P2 目标是把 P1 验证过的链路扩展到 10000 molecule，并完成 EG vs EGF 的 Hessian 定量比较。

### P2-0 存储与运行策略

目标：

- 在 10k 规模前明确 `.chk`、`.zarr.zip`、日志、checkpoint 的存储策略。

预估 label 数：

| 设置 | 总 label 数 |
|---|---:|
| reference + 3 perturbations | 40000 |
| reference + 5 perturbations | 60000 |

`.chk` 管理原则：

1. 生成 `.chk`；
2. 生成 `.zarr.zip`；
3. 检查 `.zarr.zip` 无 NaN/Inf 且 force 完整；
4. 确认无需重算后，再压缩、迁移或删除 `.chk`。

交付物：

- P2 runtime/storage plan；
- `.chk` cleanup policy；
- P2 batch manifest。

验收：

- P2 不依赖 `/tmp`；
- 每批产物可恢复；
- 已确认 `.chk` 删除不会破坏后续评估。

### P2-1 生成 10000 molecule main-small labels

目标：

- 使用 P1 验证后的配置和脚本生成 10k 规模 labels。

要求：

- 不在 P2 阶段大改 datagen 逻辑；
- 分批运行；
- 每批运行自动检查；
- 失败样本记录到 manifest。

交付物：

- 40k-60k 个 `.zarr.zip` labels；
- 每批检查报告；
- 全局 P2 label summary。

验收：

- molecule 成功率达到预设阈值；
- labels 无系统性 NaN/Inf；
- force norm 分布无明显异常；
- split 可复现且无泄漏。

### P2-2 训练 P2 OFDFT-EG 与 OFDFT-EGF

目标：

- 在 10k molecule 规模上训练正式对照模型。

至少训练：

- OFDFT-EG；
- OFDFT-EGF。

重点调参：

- `lambda_F`；
- batch size；
- density optimization 步数；
- learning rate；
- gradient clipping；
- force loss normalization。

可选策略：

- 先训练 EG；
- 使用 EG checkpoint 初始化 EGF；
- 再加入 force loss 做 fine-tuning。

交付物：

- P2 OFDFT-EG checkpoint；
- P2 OFDFT-EGF checkpoint；
- training/validation logs；
- energy/gradient/force 指标曲线。

验收：

- 两组模型都完成训练；
- EGF force MAE 优于 EG derived force；
- energy/density-gradient 指标没有不可接受退化。

### P2-3 Hessian test set

目标：

- 建立 200-1000 molecule 的 Hessian test set。

建议分两步：

1. 先做 200 molecule；
2. 流程稳定后扩展到 1000 molecule。

标签要求：

- Hessian reference 与训练 labels 保持同一 DFT level；
- 不使用 QM9 原始 B3LYP frequency 作为 PBE Hessian 监督；
- 不混用 HORM 的 omegaB97X force/Hessian。

交付物：

- Hessian test molecule list；
- PBE Hessian reference labels；
- Hessian label 检查报告。

验收：

- Hessian matrix shape 正确；
- mass-weighted Hessian 可计算；
- frequency/eigenvalue 后处理可运行。

### P2-4 Hessian 定量评估

目标：

- 输出核心科学结论：force supervision 是否改善 Hessian。

至少报告：

- energy MAE；
- force MAE；
- Hessian matrix MAE；
- mass-weighted Hessian MAE；
- Hessian eigenvalue MAE；
- vibrational frequency MAE；
- low-frequency mode error。

主表：

| 模型 | Energy MAE | Force MAE | Hessian MAE | MW-Hessian MAE | Frequency MAE | Low-freq Error |
|---|---:|---:|---:|---:|---:|---:|
| OFDFT-EG | TBD | TBD | TBD | TBD | TBD | TBD |
| OFDFT-EGF | TBD | TBD | TBD | TBD | TBD | TBD |

核心判据：

```math
\mathrm{HessianError}(\mathrm{OFDFT\text{-}EGF})
<
\mathrm{HessianError}(\mathrm{OFDFT\text{-}EG})
```

交付物：

- Hessian evaluation script；
- P2 evaluation table；
- P2 experiment report。

验收：

- 至少 200 个 Hessian test molecule 完成评估；
- EG vs EGF 对比表完整；
- 对是否改善 Hessian 给出明确结论；
- 若未改善，给出失败归因。

## 6. 优先级与立即执行顺序

### P1 优先级

P1-0/P1-1 已完成，410 molecule 子集上的正式 EG/EGF 训练 smoke 已通过；410 molecule 节点的 cached transform、grouped split 和 smoke 级 statistics 已刷新。当前最高优先级是完成 P1-2 的 1000 molecule pilot labels，并保持 cached transform/statistics/training smoke 可重复。

当前立即执行顺序：

1. 从 `start_idx=410` 开始，继续按 100-200 molecule 分批生成剩余 P1 labels。
2. 每批完成后运行 force/NaN 检查、更新 cached transformed labels、重建 grouped split、检查 molecule leakage。
3. labels 扩展到较大节点后重算 dataset statistics。
4. 先做小范围 `lambda_F` smoke，再在 P1 labels 接近 1000 molecule 后跑完整 pilot EG/EGF training。
5. 训练稳定后做 100 molecule Hessian mini-eval。

### P2 优先级

P2 只应在 P1 满足以下条件后启动：

- P1 label 成功率 >95%；
- 正式 EGF 训练稳定；
- force loss 有下降趋势；
- Hessian mini-eval 流程可运行；
- `.chk` 存储策略明确。

P2 推荐顺序：

1. 固化 P1 代码和配置。
2. 制定 P2 batch manifest 和存储策略。
3. 生成 10k labels。
4. 训练 EG/EGF。
5. 建立 200-1000 molecule Hessian test set。
6. 输出 Hessian 定量报告。

## 7. 风险清单

| 风险 | 影响 | 处理方式 |
|---|---|---|
| `/tmp` 满 | PySCF 或解压失败 | 所有命令显式设置 workspace `TMPDIR` |
| force 单位不一致 | force loss 无意义 | P1-1 必须确认 geometry 与 force 单位 |
| sample split 泄漏 | 测试集指标虚高 | 必须按 molecule ID split |
| 独立 force head | Hessian 不物理一致 | 主方案必须用 `-dE/dR` |
| `.chk` 占用过大 | P2 存储不可控 | label 验证后再压缩/迁移/删除 |
| force loss 权重过大 | energy/gradient 训练退化 | 系统扫 `lambda_F` |
| Hessian finite difference 太慢 | P2 评估阻塞 | 先 200 molecule，再扩到 1000 |

## 8. P1/P2 最小交付清单

P1 必须交付：

- P1 pilot datagen config；
- molecule-level split；
- 1000 molecule pilot labels；
- P1 label check report；
- 正式 OFDFT force dataloader；
- 正式 OFDFT-EGF force loss；
- OFDFT-EG / OFDFT-EGF P1 checkpoints；
- 100 molecule Hessian mini-eval report。

P2 必须交付：

- P2 main-small datagen config；
- 10000 molecule labels；
- P2 storage/check manifest；
- OFDFT-EG / OFDFT-EGF P2 checkpoints；
- 200-1000 molecule Hessian test set；
- Hessian evaluation scripts；
- EG vs EGF quantitative table；
- P2 experiment report。
