# WT-Al 熔点步骤 1：数据固化与审计

日期：2026-07-24

## 范围

本步骤仅固化和审计现有 WT-OFDFT 数据，不重新解释未通过门控的轨迹，也不启动新的熔点计算。

## 运行状态

- 计算节点：node01
- 检查时无 `abacus_wt`、焓差或直接共存主进程运行。
- 仅有 `wt_storage_wait` 存储等待会话。
- `/scratch/xzh/OFDFT-Hessian/runs/al/free_energy_wt` 可读取，但元数据访问存在明显延迟。

## 本地审计备份

- 归档：`node01_wt_reports_20260724.tar.gz`
- 文件数：17
- JSON 文件数：16
- JSON 语法检查：全部通过
- SHA-256：

  ```text
  30e80b5e44711fa91f68ffc1fbd1f5806debb08a2c4ad9c309d595d0ce63883f
  ```

归档只包含明确列出的 JSON、Markdown 审计结果，不包含大轨迹和 ABACUS 二进制。

## 已固化结果

### 900 K 自由能锚点

- 状态：`anchor_temperature_free_energy_verified`
- `Delta G_liq-solid = +9.3391195 meV/atom`
- 统计 RSS：`0.3048953 meV/atom`
- TI 保守误差：`0.9280610 meV/atom`
- 有限尺寸系统误差预算：`3.3622904 meV/atom`
- 合并保守误差：`4.2903515 meV/atom`
- 旧线性估计 `1003.8402 K` 仅作历史参考，不是正式熔点。

### 900 K 熔化焓

- d50：`82.6762834 meV/atom`
- 最大 block SE：`2.8100590 meV/atom`
- discard spread：`1.1562110 meV/atom`
- half drift：通过
- 状态：`verified`

### 975 K 和 1050 K 第二轮

- 两个温度的固相、液相都达到 3000 步。
- 最近邻均大于 2 A。
- 相态分别为 `solid_verified` 和 `liquid_verified`。
- 平均压力处于既定零压容差内。
- 当前归档中只有 `confirmation_summary.json`，尚无基于第二轮轨迹重建的统一 d25/d50/d75 焓差收敛报告。

### 1100 K

- 固、液两相轨迹结构完整，电子优化稳定，相态正确。
- 全局 d50：`82.371 meV/atom`，block SE `2.911 meV/atom`。
- 全局 d75：`78.722 meV/atom`，block SE `3.673 meV/atom`。
- d50-d75 spread：`3.649 meV/atom`。
- half drift：`7.30` 和 `9.09 meV/atom`。
- 状态：物理轨迹有效，但严格统计门控失败，不能进入正式 Gibbs-Helmholtz 输入。

## 已识别缺口

1. 必须从 975 K 和 1050 K 第二轮原始输出重建 d25/d50/d75 焓差报告。
2. 必须形成只包含已通过温度点的统一 `discard_convergence_summary`。
3. 在正式求根前必须复核能量定义、温度标签、外压项和零压体积来源。
4. 1100 K 是否补算应由 900/975/1050 K 首次正式根及其置信区间决定。

## 下一门控

步骤 2 将统一核对热力学定义，并在 node01 原地生成 975/1050 K 第二轮的正式焓差报告。任何原始轨迹缺失、相态失败或能量定义不一致都会阻止进入求根。
