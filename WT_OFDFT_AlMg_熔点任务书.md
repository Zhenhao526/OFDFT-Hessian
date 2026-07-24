# 基于 WT-OFDFT 的 Al/Mg 及 Al-Mg 合金熔点计算任务书

版本：v0.2  
日期：2026-07-24  
方向：使用 Wang-Teter kinetic energy density functional (WT-KEDF) 的 orbital-free DFT 方法，开发并验证纯 Al、纯 Mg 及后续 Al-Mg 合金熔点计算流程。

方法变更：自 v0.2 起，WT-KEDF 是本项目唯一主计算方法和验收对象。此前 MPN-KEDF 计算仅保留为历史探索与 KEDF 对照，不得与 WT 结果混合形成自由能曲线或最终熔点。

## 1. 项目背景与核心判断

Al-Mg 合金熔点、液相线、固相线和凝固行为对轻合金设计有直接意义。传统高精度第一性原理分子动力学基于 Kohn-Sham DFT，计算代价高，难以直接覆盖大尺寸固液共存界面、长时间熔化/凝固过程和多组分浓度扫描。OFDFT 通过显式密度泛函近似非相互作用动能，可以显著降低电子结构计算成本，但精度长期受 kinetic energy density functional (KEDF) 限制。

WT-KEDF 是基于均匀电子气线性响应构造的非局域动能泛函，形式明确、计算稳定、可直接用于周期性简单金属体系。对本项目而言，WT 的价值不是消除传统 OFDFT 的全部系统误差，而是建立一条可复现、可审计的 Al/Mg 自由能与固液共存基准线，定量判断 WT、局域赝势、有限尺寸和采样长度对熔点的综合偏差。

本任务书明确采用以下主路线：

> ABACUS + OFDFT + WT-KEDF + Python/ASE 自动化工作流，以绝对自由能和 Gibbs-Helmholtz 积分为定量主线，以大体系直接固液共存为独立验证，先完成纯 Al，再完成纯 Mg，最后决定是否扩展到 Al-Mg 合金。

## 2. 总体目标

开发一套可复现、可扩展的 WT-OFDFT 熔点计算程序和计算规范，首先完成纯 Al、纯 Mg 的绝对自由能、熔化焓和熔点计算与误差评估，判断 WT-OFDFT 对简单金属熔化行为的定量可靠性；在此基础上，为后续 Al-Mg 合金液相线/固相线计算建立技术路线。

## 3. 科学问题

1. WT-OFDFT 对 Al、Mg 熔点的系统偏差有多大，能否在明确不确定度后形成可靠基准？
2. WT-OFDFT 对 fcc Al 和 hcp Mg 的固相、液相、固液界面是否都能保持稳定收敛？
3. 熔点误差主要来自 KEDF、局域赝势、有限尺寸、采样时间，还是固液共存判据？
4. 纯 Al/Mg 验证通过后，WT-OFDFT 是否可作为 Al-Mg 合金熔点/相边界扫描的低成本第一性原理引擎？

## 4. 阶段性目标

### 阶段 A：软件与方法可运行性验证

目标：确认 ABACUS 的 WT-OFDFT 可以在本地或集群稳定运行 Al/Mg 固体、液体、自由能窗口与固液界面测试。

任务：

1. 获取并编译/安装支持 OFDFT、WT-KEDF、应力输出和自由能耦合路径的 ABACUS 版本。
2. 准备 Al、Mg 的 bulk-derived local pseudopotentials (BLPS) 或经静态与液体测试验证的局域赝势。
3. 运行 fcc Al、hcp Mg 的静态单点、体积扫描和结构优化。
4. 运行短程 NVT/NVE 液体 MD，确认电子密度优化、力和能量稳定。

交付物：

1. ABACUS 运行环境说明。
2. Al/Mg WT-OFDFT 最小输入模板。
3. 静态 benchmark 表：晶格常数、体模量、相对能量、力收敛表现。
4. 一组 5-10 ps 的液体稳定 MD 测试轨迹。

### 阶段 B：自动化代码开发

目标：开发一套 Python 工作流，自动生成结构、写 ABACUS 输入、提交计算、解析输出、判断固液状态并估算熔点。

建议代码结构：

```text
mpn_melting/
  config/
    al.yaml
    mg.yaml
    abacus_wt.yaml
  mpn_melting/
    structures.py        # 构建 fcc Al、hcp Mg、液体、固液共存结构
    abacus_input.py      # 写 INPUT、STRU、KPT、赝势路径等
    runner.py            # 本地/集群任务提交与状态管理
    parser.py            # 解析 ABACUS out、md、log、energy、force、stress
    observables.py       # RDF、MSD、CNA/PTM、Steinhardt Q_l、密度剖面
    references.py        # Einstein crystal、pair/sUF/WCA 参考体系
    free_energy.py       # λ 窗口、平均力、解析修正和自由能积分
    gibbs_helmholtz.py   # 多温度焓差拟合、温度积分和熔点求根
    coexistence.py       # 固液共存判据、界面追踪、熔点 bracket
    uncertainty.py       # 有限尺寸/时间采样/温度 bracket 误差估计
    report.py            # 自动生成图表与 Markdown/HTML 报告
  workflows/
    pure_al_free_energy.yaml
    pure_mg_free_energy.yaml
    pure_al_coexistence.yaml
    pure_mg_coexistence.yaml
  notebooks/
    01_static_benchmark.ipynb
    02_al_melting_analysis.ipynb
    03_mg_melting_analysis.ipynb
  tests/
    test_abacus_input.py
    test_parser.py
    test_order_parameters.py
```

说明：现有仓库暂时保留历史包名 `mpn_melting` 以兼容已生成数据和脚本；所有新配置、元数据和报告必须显式记录 `of_kinetic = wt`，后续可在不破坏数据溯源的前提下迁移为中性包名。

核心命令行接口建议：

```bash
mpn-melting init --element Al --method wt
mpn-melting make-static --element Al --volumes 0.94 0.97 1.00 1.03 1.06
mpn-melting make-liquid --element Al --temperature 1200 --steps 10000
mpn-melting make-ti --element Al --phase solid --temperature 900
mpn-melting analyze-ti runs/al/free_energy/T0900/solid
mpn-melting gibbs-helmholtz --element Al --anchor-temperature 900
mpn-melting make-coexist --element Al --temperature 930 --size 8x8x12
mpn-melting submit runs/al/coexist/T930
mpn-melting analyze runs/al/coexist/T930
mpn-melting bracket --element Al --tmin 850 --tmax 1100
```

交付物：

1. 可版本管理的 Python 包。
2. Al/Mg 输入模板自动生成器。
3. ABACUS 输出解析器。
4. 固液状态分析脚本。
5. 一键生成熔点分析报告的命令。

### 阶段 C：纯 Al 熔点计算

目标：用 WT-OFDFT 的绝对自由能/Gibbs-Helmholtz 主线计算纯 Al 熔点，并用大体系直接固液共存独立验证，再与实验值和已有 OFDFT/KSDFT/经典势结果比较。

参考实验熔点：

Al：约 933.47 K。

推荐计算路线：

1. 静态与有限温度体积扫描：确定 WT-OFDFT 下固相和液相的零压体积，压力必须经过有限差分或多体积回归验证。
2. 固相绝对自由能：使用 Einstein crystal 或经解析修正的谐参考体系，完成多 λ 热力学积分、质心约束修正和有限尺寸检查。
3. 液相绝对自由能：使用稳定的解析流体参考体系（优先 pair/sUF/WCA 路径），完成参考自由能、重叠检查和多 λ 热力学积分。
4. 在锚点温度得到 `Delta G_liq-solid`，并对正向/反向路径、λ 网格和块平均误差做门控。
5. 在多个温度分别采样零压固相和液相熔化焓，使用 Gibbs-Helmholtz 关系构建 `Delta G_liq-solid(T)` 并求唯一物理解。
6. 构建 512-2048 原子固液共存体系，在自由能根的下方、附近和上方运行独立长轨迹。
7. 通过固液界面移动方向验证自由能熔点：
   - 固相增长：T < Tm
   - 液相增长：T > Tm
   - 界面长期稳定：T 接近 Tm

分析指标：

1. RDF 峰形与液体短程有序。
2. 均方位移 MSD。
3. 局域结构分类：fcc/hcp/unknown/liquid-like。
4. Steinhardt order parameters，如 Q4、Q6。
5. 沿界面法向的密度剖面和固相分数剖面。
6. 总能、温度、压力、体积随时间的漂移。

验收标准：

1. 得到 Al 的 WT-OFDFT 熔点估计值和不确定度。
2. 给出与实验值 933.47 K 的相对误差。
3. 自由能根与直接固液共存 bracket 在各自不确定度内一致。
4. 与 MPN、其他传统 OFDFT、KSDFT 和经典势结果进行定性/定量对照。
5. 明确误差来源：参考自由能、λ 积分、KEDF/赝势、有限尺寸、采样时间和系综设置。

### 阶段 D：纯 Mg 熔点计算

目标：用同一套 WT-OFDFT 自由能和直接共存工作流计算纯 Mg 熔点，验证 hcp 金属体系的适用性。

参考实验熔点：

Mg：约 923 K。

Mg 的特殊注意事项：

1. Mg 为 hcp 结构，需要同时关注 a、c/a 和各向异性热膨胀。
2. 固液共存结构的晶向选择要避免过强界面取向偏差。
3. Mg 的局域赝势质量可能对体积、压力和熔点更敏感，需要做赝势敏感性检查。
4. 若 NPT 变胞 MD 不稳定，可先用 NPT/NVT 预平衡确定体积，再在固定体积下做 NVE/NVT 共存。

推荐温度 bracket：

870, 900, 930, 960, 1000 K。

验收标准：

1. 得到 Mg 的 WT-OFDFT 熔点估计值和不确定度。
2. 给出与实验值约 923 K 的相对误差。
3. 比较 Al 和 Mg 的误差模式，判断 WT 对不同简单金属结构的稳定性。

### 阶段 E：方法可靠性评估与是否进入合金阶段的决策

进入 Al-Mg 合金计算前，需要完成以下判据：

1. Al、Mg 两个纯元素熔点误差均在可接受范围内。
   - 理想目标：误差小于 5%。
   - 可接受探索目标：误差小于 8%，且误差来源清楚。
   - 若误差超过 10%，不建议直接进入合金熔点预测，应先做方法修正。
2. 液体结构 RDF 与实验/KSDFT/可靠势函数结果一致。
3. 固液共存界面在目标温度附近稳定，不出现非物理密度崩塌或异常结晶。
4. 小体系 KSDFT 对照显示 WT-OFDFT 的能量/力误差在熔化相关构型上可接受。

若通过，则进入 Al-Mg 合金阶段。若不通过，则优先检查 WT 参考密度与数值设置、局域赝势、有限尺寸和自由能参考路径，必要时再引入 MPN-KEDF 或 Δ-ML 作为对照/修正路线。

## 5. Al-Mg 合金扩展计划

合金阶段不作为第一阶段交付，但任务书需提前保留接口。

初始推荐成分：

1. Al-rich：Al-2at%Mg、Al-5at%Mg、Al-10at%Mg、Al-20at%Mg。
2. Mg-rich：Mg-2at%Al、Mg-5at%Al、Mg-10at%Al。
3. 视资源增加代表性中间成分和金属间化合物附近成分。

主要目标：

1. 计算液相线趋势，而不是一开始追求完整相图。
2. 与 CALPHAD Al-Mg 相图数据比较趋势。
3. 识别 WT-OFDFT 对混合焓、液体短程有序和固液界面偏析的预测能力。

合金方法注意事项：

1. 合金熔点不是单一温度，需区分 solidus、liquidus 和两相区。
2. 固液共存中可能存在成分偏析，必须跟踪局域 Mg/Al 浓度剖面。
3. 需要比纯元素更长的扩散采样时间。
4. 可能需要半巨正则或热力学积分方法辅助确定相边界。

## 6. 技术路线细节

### 6.1 电子结构设置

主设置：

1. 代码：ABACUS。
2. 电子结构方法：OFDFT。
3. KEDF：WT。
4. 赝势：bulk-derived local pseudopotential，优先使用经过固相、液相和压力验证的 ABACUS 设置。
5. 基组：plane wave / charge-density grid，按 ABACUS OFDFT 设置。
6. 交换关联：主线固定为当前自由能锚点所用的 PBE 设置；任何 LDA/PBE 变更必须另立支路，不得混入同一自由能积分。

需要记录的关键参数：

1. `of_kinetic = wt`
2. `of_method`
3. `of_conv`
4. `of_tole`
5. `of_tolp`
6. `of_wt_alpha`、`of_wt_beta` 和 `of_wt_rho0`
7. `ecutrho` 或等价密度网格参数
8. 赝势文件、版本、来源与 checksum
9. 时间步长、温控器/压控器、系综
10. ABACUS commit、编译选项和自由能耦合路径版本

注意：WT 路径允许计算应力，但零压体积不得仅依赖一次 NPT 外推。主线采用多体积 NVT 压力回归或经有限差分验证的各向同性压力确定零压体积；NPT 只作辅助校验。自由能窗口和焓差采样必须使用相同的 KEDF、赝势、交换关联、网格和能量定义。

### 6.2 熔点计算主方法：绝对自由能与 Gibbs-Helmholtz 积分

主线先分别计算固、液相的 Gibbs 自由能，再求解 `Delta G_liq-solid(Tm) = 0`。这样可以给出连续的自由能差、统计不确定度和明确的熔点根，不依赖单条界面轨迹的有限时间判断。

标准流程：

1. 在每个目标温度对固、液相分别做体积扫描，通过压力-体积回归确定零压体积。
2. 固相使用 Einstein crystal 或已验证的谐参考体系，液相使用 pair/sUF/WCA 等具有解析自由能的稳定流体参考体系。
3. 对参考势与 WT-OFDFT 势之间建立耦合势，并在多个 λ 窗口采样 `dU/dlambda`。
4. 对 λ 平均力数值积分，加入固相质心约束、有限尺寸和参考体系解析修正，得到同一锚点温度下的固、液绝对自由能。
5. 计算锚点自由能差 `Delta G(T0) = G_liq(T0) - G_solid(T0)`。
6. 在多个温度采样零压固、液相焓，得到 `Delta H(T) = H_liq(T) - H_solid(T)`。
7. 使用 Gibbs-Helmholtz 关系

   ```text
   d[Delta G(T)/T] / dT = -Delta H(T) / T^2
   ```

   将锚点传播到目标温区，并求解唯一物理根 `Delta G(Tm) = 0`。
8. 对锚点自由能、λ 积分、焓差拟合和温度积分联合传播不确定度，给出 `Tm` 的统计误差和系统误差。

自由能生产门控：

1. 固、液相在整个采样区间必须保持各自相态，且最近邻、RDF、MSD、CSP/Q6 无非物理异常。
2. 每个 λ 窗口电子优化收敛，平均力时间序列无持续漂移；端点重叠不足时必须加密 λ 或改参考路径。
3. 正向/反向路径或独立种子结果在合并不确定度内一致。
4. 焓差轨迹必须通过块平均、前后半程漂移和有效样本量门槛；不合格温度点不得进入 Gibbs-Helmholtz 积分。
5. 熔点根必须在采样温区内唯一，并对温度节点删减、拟合形式和积分方法保持稳定。

### 6.3 独立验证：直接固液共存法

直接共存不用于替代自由能主线，而用于验证自由能根附近界面迁移方向是否符合热力学预期。

标准流程：

1. 构建足够长的固体超胞，沿 z 方向或指定方向分区。
2. 固定一半为固体，在相同密度和周期边界下原位熔化另一半，避免独立硬拼导致界面原子重叠。
3. 在释放全部原子前完成短程界面松弛和全体系热化，并重新检查最近邻与两相结构对比。
4. 在自由能根下方、附近和上方分别运行 NPH/NVE 或弱耦合 NVT 长轨迹。
5. 使用区域 MSD、CSP/Q6、z 向密度/有序度剖面和界面位置判断固相增长、液相增长或近稳定共存。

温度验证：

1. 初始验证点可取自由能根上下各 20-40 K。
2. 若界面方向与自由能符号一致，再将间隔缩小到 10-20 K。
3. 直接共存不确定度由温度 bracket、轨迹长度、独立种子、有限尺寸和界面取向共同估计。

### 6.4 辅助方法：Z-method / 升温法

用途：

1. 快速扫描 WT-OFDFT 的熔化温区。
2. 检查固体过热极限。
3. 为固液共存法选择初始温度范围。

注意：

Z-method 和升温法不作为最终熔点主证据，因为简单金属可出现明显过热，尤其在有限尺寸周期体系中。

### 6.5 小体系 KSDFT 对照

目标：

1. 抽取固体、液体、界面构型。
2. 用 KSDFT 单点评估能量、力、压力差异。
3. 判断 WT-OFDFT 在熔化相关构型上的误差是否系统偏向固体或液体。

推荐对照集：

1. Al 固体 3-5 个温度构型。
2. Al 液体 3-5 个温度构型。
3. Al 固液界面 3-5 个构型。
4. Mg 同样数量。

评价指标：

1. 每原子能量差。
2. 力 RMSE。
3. 压力差。
4. 固液相对能量偏置。

## 7. 数据与结果管理

每个计算任务必须保存：

```text
runs/
  al/
    static/
    volume_scan/
    liquid/
    free_energy/
      T0900/
        solid/
        liquid/
    enthalpy/
      T0975/
      T1050/
      T1100/
    coexist/
      T0900/
        input/
        output/
        analysis/
        metadata.yaml
  mg/
    static/
    volume_scan/
    liquid/
    free_energy/
    enthalpy/
    coexist/
```

`metadata.yaml` 至少包含：

1. 元素/成分。
2. 结构类型。
3. 原子数。
4. 初始结构来源。
5. ABACUS commit/version。
6. WT 参数，包括 `of_wt_alpha`、`of_wt_beta`、`of_wt_rho0` 及其来源。
7. 赝势文件 checksum。
8. 系综、温度、压力、步长、总步数。
9. 提交脚本和运行机器。
10. 分析脚本版本。

所有图表必须可由脚本重新生成，避免手工处理不可复现。

## 8. 验收标准

第一阶段最终验收需要满足：

1. 代码可自动生成纯 Al 和纯 Mg 的 WT-OFDFT 输入，并完整记录 WT 参数和参考密度。
2. 代码可自动解析 ABACUS MD 输出并生成能量、温度、压力、RDF、MSD、固液分数图。
3. 完成 Al 和 Mg 固、液两相各自的零压体积确定、绝对自由能锚点、多温度熔化焓采样和 Gibbs-Helmholtz 求根。
4. 完成 Al 和 Mg 各至少一组大体系直接固液共存验证，并使其界面迁移方向与自由能根一致。
5. 给出 Al/Mg 熔点数值、统计误差、系统误差和与实验值的对比。
6. 给出有限尺寸、轨迹长度、赝势、WT 参考密度和自由能积分路径的敏感性分析。
7. 形成一份阶段报告，明确是否进入 Al-Mg 合金阶段。

推荐最终报告结构：

1. 方法与输入参数。
2. 零压体积和静态 benchmark。
3. 固、液相稳定性与采样质量。
4. 绝对自由能参考系、λ 窗口和积分收敛。
5. 多温度焓差与 Gibbs-Helmholtz 熔点。
6. Al/Mg 大体系直接共存验证。
7. KSDFT 小体系对照。
8. 误差来源分析。
9. 是否进入合金阶段的结论。
10. 下一步 Al-Mg 合金计划。

## 9. 里程碑与时间安排

建议 10 周完成第一阶段。

第 1 周：

1. ABACUS WT-OFDFT、WT 应力和自由能耦合环境搭建。
2. 收集 Al/Mg 局域赝势。
3. 跑通最小 fcc Al 和 hcp Mg 单点，并核对 WT 参数与能量定义。

第 2 周：

1. 完成 Python 项目骨架。
2. 完成 ABACUS 输入生成器。
3. 完成静态体积扫描和压力有限差分校验。

第 3 周：

1. 完成输出解析器。
2. 完成 RDF、MSD、Q_l、局域结构分类分析。
3. 完成 Al/Mg 固、液相短程 MD 和零压体积测试。

第 4-5 周：

1. 完成 Al 固相 Einstein crystal 和液相参考体系的绝对自由能计算。
2. 完成 Al 多温度熔化焓采样与 Gibbs-Helmholtz 求根。
3. 构建 Al 大体系固液共存结构并独立验证熔点方向。
4. 输出 Al 熔点及初步不确定度。

第 6-7 周：

1. 完成 Mg 固、液相绝对自由能和多温度熔化焓采样。
2. 完成 Mg Gibbs-Helmholtz 求根。
3. 构建 Mg 大体系固液共存结构并独立验证。
4. 输出 Mg 熔点及初步不确定度。

第 8 周：

1. 做自由能 λ 网格、有限尺寸和轨迹长度敏感性分析。
2. 对熔点附近关键温度做重复焓差轨迹和直接共存轨迹。

第 9 周：

1. 抽取 Al/Mg 固体、液体、界面构型。
2. 做小体系 KSDFT 对照。
3. 分析 WT-OFDFT 能量、力、压力和固液相对自由能误差。

第 10 周：

1. 完成阶段报告。
2. 给出是否进入 Al-Mg 合金阶段的决策。
3. 整理代码、输入模板、示例数据和复现实验说明。

## 10. 主要风险与应对方案

风险 1：WT 的线性响应近似和参考密度对液体或固液界面适用性不足。

应对：

1. 必须做液体和界面 KSDFT 小体系对照。
2. 对 `of_wt_rho0`、网格和赝势做独立敏感性分支，不得事后调参拟合目标熔点。
3. 若固液相对自由能存在稳定系统偏差，将 MPN-KEDF 或 Δ-ML 仅作为独立对照/修正路线，不能混入 WT 主曲线。

风险 2：局域赝势导致体积和压力误差。

应对：

1. 对 Al/Mg 至少测试一套备用 local pseudopotential。
2. 对比晶格常数、体模量、压力和液体 RDF。

风险 3：固液共存体系太小导致界面相互作用。

应对：

1. 从 512-1024 原子开始，关键温度用更大体系复核。
2. 比较不同界面方向和 cell 长度。

风险 4：温控器/压控器影响界面动力学。

应对：

1. 初始 bracket 可用 NVT，最终判断优先使用 NPH/NVE 或弱耦合方案。
2. 报告中必须说明系综选择对结果的影响。

风险 5：ABACUS WT-OFDFT MD 或热力学积分稳定性不足。

应对：

1. 先做短时间小体系液体测试。
2. 调整电子密度优化收敛阈值、时间步长、网格参数。
3. 对 λ 端点使用软核、分段参考势或更密 λ 网格，避免积分被端点奇异行为控制。
4. 如仍不稳定，先用静态 single-point + 外部 MD 框架评估是否可行，再决定是否修改 ABACUS 接口。

## 11. 参考文献与软件来源

1. L.-W. Wang and M. P. Teter, "Kinetic-energy functional of the electron density", Phys. Rev. B 45, 13196 (1992).  
   https://doi.org/10.1103/PhysRevB.45.13196

2. ABACUS open-source repository.  
   https://github.com/deepmodeling/abacus-develop

3. ABACUS documentation, OFDFT input keywords including `of_kinetic = wt`.  
   https://abacus.deepmodeling.com/en/latest/advanced/input_files/input-main.html

4. D. Frenkel and B. Smit, "Understanding Molecular Simulation: From Algorithms to Applications", 2nd ed., Academic Press (2002).  

5. Z. Bai et al., "Nucleation of Supercooled Liquid Aluminum by an Orbital-Free Density Functional Theory Molecular Dynamics Simulation".  
   https://arxiv.org/abs/2306.11933

6. Xuecheng Shao et al., "DFTpy: An efficient and object-oriented platform for orbital-free DFT simulations".  
   https://arxiv.org/abs/2002.02985

7. Shashikant Kumar et al., "Kohn-Sham accuracy from orbital-free density functional theory via Δ-machine learning".  
   https://arxiv.org/abs/2310.06598

8. Liang Sun and Mohan Chen, "Machine learning based nonlocal kinetic energy density functional for simple metals and alloys", Phys. Rev. B 109, 115135 (2024). 仅作为 KEDF 对照，不作为本任务主方法。  
   https://arxiv.org/abs/2310.15591

## 12. 第一阶段最终产物清单

1. 支持 WT 主线的 `mpn_melting` Python 代码包；包名仅为历史兼容。
2. ABACUS WT-OFDFT Al/Mg 输入模板。
3. Al/Mg 静态 benchmark、零压体积和压力验证数据。
4. Al/Mg 固、液相 MD 稳定性及熔化焓数据。
5. Al/Mg 绝对自由能、λ 积分和 Gibbs-Helmholtz 求根数据。
6. Al/Mg 大体系固液共存验证数据。
7. 自动分析图表和完整不确定度预算。
8. KSDFT 小体系对照数据。
9. 阶段报告：是否进入 Al-Mg 合金阶段。

## 13. 决策门槛

若满足以下条件，进入 Al-Mg 合金液相线/固相线计算：

1. Al 和 Mg 熔点误差均小于 8%，且至少一个元素小于 5%。
2. 液体 RDF、扩散行为和固液界面结构无明显非物理异常。
3. KSDFT 小体系对照未发现明显固液相对能量系统偏置。
4. Gibbs-Helmholtz 根可由独立采样复现，且与直接固液共存验证在合并不确定度内一致。

若不满足，则不直接进入合金预测，优先执行：

1. WT 参考密度、λ 路径、数值收敛和局域赝势的独立诊断。
2. 局域赝势重建/筛选。
3. 将 MPN-KEDF 作为独立 KEDF 对照。
4. OFDFT + Δ-ML energy/force correction。
5. 与 KSDFT/ML potential 混合的多层级 workflow。

## 14. 当前 WT-Al 基线（2026-07-24）

本节记录当前可审计进度，不替代最终验收。

1. 已获得 900 K 固液绝对自由能锚点：`ΔG_liq-solid = +9.339 meV/atom`，当前保守不确定度为 `4.290 meV/atom`。
2. 975 K 和 1050 K 的固、液两相 3000 步焓差轨迹已完成并通过相态检查。
3. 1100 K 的本地 `0-1300` 步轨迹与 node01 续跑至全局 3000 步的轨迹已完成拼接审计；固、液相均保持正确相态。
4. 1100 K 后半段熔化焓为 `82.371 meV/atom`，分块标准误为 `2.911 meV/atom`；后四分之三结果为 `78.722 ± 3.673 meV/atom`。
5. 1100 K 两种截断的半程漂移分别约为 `7.30` 和 `9.09 meV/atom`，超过 `5 meV/atom` 严格门槛，因此该温度点尚未通过生产验收。
6. 目前不得发布最终 Gibbs-Helmholtz 熔点；此前约 `1003.84 K` 的数值仅是未通过完整门控的预估。
7. 1728 原子直接固液共存验证尚未形成最终合格结果。

当前最近任务是改善或补充 1100 K 焓差采样，使其通过漂移门槛，然后使用通过门控的多温度数据求解唯一 Gibbs-Helmholtz 根，并以 1728 原子 WT 直接共存轨迹进行独立验证。
