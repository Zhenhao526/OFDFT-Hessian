# WT Al 熔点任务恢复审计

更新时间：2026-07-24  
恢复节点：`node01`  
状态：已整合代码、分析证据和重启结构；未启动计算。

## 1. node01 本地恢复区

恢复根目录：

```text
/home/shenwei01/wt_melting_restore_20260724
```

主要内容：

```text
integrated/repository/          Mac 保存的最新项目代码
integrated/canonical_scripts/   scripts、remote_staging 和恢复包脚本的并集
integrated/abacus_source/       可重新构建的 ABACUS WT 源码及 al.gga.psp
integrated/evidence/            188 个分析/日志/补丁文件、交接文档和重启结构
integrated/prepared_fallback/   从本地 step 1300 准备的 1100 K 续跑输入
```

所有内容均在 node01 本地 `/home`，不依赖当前降级的 BeeGFS。

## 2. 文件整合结果

- Mac 代码快照与 node01 `/tmp` 恢复包共有 13 个同名脚本或补丁。
- 其中 11 个字节级完全一致。
- `prepare_wt_zero_pressure_confirmation.py` 以 Mac 版本较新：
  增加经验证体积扫描来源、温度一致性及随机种子记录。
- `watch_wt_coexistence_validation_node01.sh` 以 Mac 版本较新：
  增加全轨迹最小最近邻必须大于 2 A 的门控。
- 暂停时使用的
  `watch_wt_enthalpy_recovery_node01.sh` 和
  `watch_wt_coexistence_after_recovery_node01.sh`
  只存在于 `remote_staging`；现已并入 canonical 脚本集合。
- canonical Python 脚本已通过 `compileall`。
- 新增 `prepare_wt_enthalpy_resume_from_stru.py`，用于从完整
  `STRU_MD_*` 和其中的速度生成全新 continuation 目录，不覆盖旧输出。
- 新脚本已通过两项单元测试，并用真实的固/液 `STRU_MD_1300`
  成功生成 1700 步的 1100 K fallback 输入；两相各有 108 组速度。

无法完成的比较：

```text
git -C /scratch/xzh/OFDFT-Hessian ...
```

会返回 `Communication error on send` 并可能进入不可中断 I/O。
因此在 Target 5/6 恢复前，不能安全完成本地快照与 `/scratch`
实际工作树的逐文件校验或覆盖。

## 3. 与暂停前任务进度的差距

已经正式完成：

1. 900 K 自由能锚点：
   `Delta G_liq-solid = +9.3391195 meV/atom`，
   保守不确定度 `4.2903515 meV/atom`。
2. 975 K 第二轮固/液各 3000 步，正式门控通过。
3. 1050 K 第二轮固/液各 3000 步，正式门控通过。

尚未完成：

1. 1100 K 第二轮只完成到共同完整帧 step 1350，目标为 3000。
2. 975/1050/1100 K 的最终 d25/d50/d75 焓差收敛汇总尚未重建。
3. `gibbs_helmholtz_melting.json` 尚未生成和验证。
4. 1728 原子直接固液共存验证尚未完成。
5. 因此尚无最终熔点及最终不确定度。

本地独立恢复证据只保存到 1100 K 的 `STRU_MD_1300`。它包含完整
坐标和速度，可作为 BeeGFS 无法恢复时的后备起点，但比正式暂停点
少 50 步。

## 4. 运行环境差距

- node01 本地没有 `abacus_wt_ti_cpu` 可执行文件。
- 原脚本使用：

```text
/scratch/xzh/OFDFT-Hessian/external/abacus-develop/build-wt-ti-cpu2/source/abacus_wt_ti_cpu
/scratch/xzh/conda/envs/abacus-gpu/bin/mpirun
```

- node01 本地 `.conda/pkgs` 中虽有 cmake/openmpi 文件，但缺少完整
  环境链接，当前不能独立运行。
- 已把完整 ABACUS WT 源码和 `al.gga.psp` 保存到本地恢复区，所以
  二进制可重建；但最快路线仍是 Target 恢复后复用原二进制和环境。

## 5. 最快恢复路线

### 首选：恢复 node04/BeeGFS 后原位续算

1. 确认 Target 5/6 均为 `Online / Good`。
2. 运行 `resume_wt_preflight_node01.sh`，确认 900 K 锚点、
   975/1050 K 正式 summary、1100 K step 1350、WT 二进制和 MPI
   环境全部可读。
3. 从固/液共同完整 `STRU_MD_1350` 新建 continuation；保留速度，
   各运行剩余 1650 步，不重跑已完成的 1350 步。
4. 将旧 0-1350 与新 continuation 合并做正式相态和热力学门控。
5. 重建 d25/d50/d75 及 discard convergence。
6. 通过后立即求 Gibbs-Helmholtz 根。
7. 只在根和误差门控通过后启动 1728 原子直接共存验证。

这是恢复原进度最快且统计上最干净的路线。

### 后备：Target 长期无法恢复

1. 使用本地 `STRU_MD_1300`，现成 fallback 输入还需运行 1700 步。
2. 在 node01 本地重新构建 ABACUS WT，并建立完整 MPI 运行环境。
3. 900 K 锚点可由独立 JSON 保留。
4. 975/1050 K 的正式原始轨迹没有独立本地副本；若永久丢失，必须
   重新制备或重新采样，不能仅用交接文档中的均值替代统计分析。

因此如果 node04 仍有恢复可能，优先修复存储远比立即重算更快。

## 6. 禁止事项

- Target 5/6 离线时不得读取、diff 或覆盖 `/scratch` 大目录。
- 不得直接运行旧 watcher；它会拒绝已有 `OUT.*`，且不能正确接续
  暂停中的 1100 K round2。
- 不得从仅有日志、无完整坐标的 step 1352/1356 续跑。
- 不得丢弃 1300/1350 帧的速度或重新随机化速度。
- 不得用 900 K 的线性估计 `1003.84 K` 作为最终熔点。

