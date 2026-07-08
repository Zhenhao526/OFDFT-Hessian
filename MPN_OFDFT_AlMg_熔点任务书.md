# 基于 MPN-OFDFT 的 Al/Mg 及 Al-Mg 合金熔点计算任务书

版本：v0.1  
日期：2026-07-07  
方向：使用 MPN kinetic energy density functional (MPN-KEDF) 的 orbital-free DFT 方法，开发并验证纯 Al、纯 Mg 及后续 Al-Mg 合金熔点计算流程。

## 1. 项目背景与核心判断

Al-Mg 合金熔点、液相线、固相线和凝固行为对轻合金设计有直接意义。传统高精度第一性原理分子动力学基于 Kohn-Sham DFT，计算代价高，难以直接覆盖大尺寸固液共存界面、长时间熔化/凝固过程和多组分浓度扫描。OFDFT 通过显式密度泛函近似非相互作用动能，可以显著降低电子结构计算成本，但精度长期受 kinetic energy density functional (KEDF) 限制。

MPN-KEDF 是 Liang Sun 和 Mohan Chen 提出的 machine-learning based physical-constrained nonlocal KEDF，已在 ABACUS 中实现，并针对 Li、Mg、Al 及多种简单金属合金进行了系统测试。其关键特点是同时引入非局域密度信息和物理约束，包括动能缩放律、自由电子气极限、Pauli 能量密度非负性。对本项目而言，MPN 相比传统 WT/LKT 等 KEDF 的优势在于：它已经面向简单金属和合金体系设计，且 Al/Mg 正是其验证对象之一。

本任务书明确采用以下主路线：

> ABACUS + OFDFT + MPN-KEDF + Python/ASE 自动化工作流，用固液共存法为主、Z-method/升温法为辅，先验证纯 Al 和纯 Mg 熔点，再扩展到 Al-Mg 合金。

## 2. 总体目标

开发一套可复现、可扩展的 MPN-OFDFT 熔点计算程序和计算规范，首先完成纯 Al、纯 Mg 的熔点计算与误差评估，判断 MPN-OFDFT 对简单金属熔化行为的定量可靠性；在此基础上，为后续 Al-Mg 合金液相线/固相线计算建立技术路线。

## 3. 科学问题

1. MPN-OFDFT 是否能显著改善传统 OFDFT 对 Al 熔点偏高的问题？
2. MPN-OFDFT 对 fcc Al 和 hcp Mg 的固相、液相、固液界面是否都能保持稳定收敛？
3. 熔点误差主要来自 KEDF、局域赝势、有限尺寸、采样时间，还是固液共存判据？
4. 纯 Al/Mg 验证通过后，MPN-OFDFT 是否可作为 Al-Mg 合金熔点/相边界扫描的低成本第一性原理引擎？

## 4. 阶段性目标

### 阶段 A：软件与方法可运行性验证

目标：确认 ABACUS 的 MPN-OFDFT 可以在本地或集群稳定运行 Al/Mg 固体与液体测试。

任务：

1. 获取并编译/安装支持 OFDFT 与 MPN-KEDF 的 ABACUS 版本。
2. 准备 Al、Mg 的 bulk-derived local pseudopotentials (BLPS) 或论文/ABACUS 推荐局域赝势。
3. 运行 fcc Al、hcp Mg 的静态单点、体积扫描和结构优化。
4. 运行短程 NVT/NVE 液体 MD，确认电子密度优化、力和能量稳定。

交付物：

1. ABACUS 运行环境说明。
2. Al/Mg MPN-OFDFT 最小输入模板。
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
    abacus_mpn.yaml
  mpn_melting/
    structures.py        # 构建 fcc Al、hcp Mg、液体、固液共存结构
    abacus_input.py      # 写 INPUT、STRU、KPT、赝势路径等
    runner.py            # 本地/集群任务提交与状态管理
    parser.py            # 解析 ABACUS out、md、log、energy、force、stress
    observables.py       # RDF、MSD、CNA/PTM、Steinhardt Q_l、密度剖面
    coexistence.py       # 固液共存判据、界面追踪、熔点 bracket
    uncertainty.py       # 有限尺寸/时间采样/温度 bracket 误差估计
    report.py            # 自动生成图表与 Markdown/HTML 报告
  workflows/
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

核心命令行接口建议：

```bash
mpn-melting init --element Al --method mpn
mpn-melting make-static --element Al --volumes 0.94 0.97 1.00 1.03 1.06
mpn-melting make-liquid --element Al --temperature 1200 --steps 10000
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

目标：用 MPN-OFDFT 计算纯 Al 熔点，并与实验值和已有 OFDFT/KSDFT/经典势结果比较。

参考实验熔点：

Al：约 933.47 K。

推荐计算路线：

1. 静态体积扫描：确定 MPN-OFDFT 下 fcc Al 平衡晶格常数。
2. 固体 NPT/NVT 预平衡：在 800-1100 K 范围获取热膨胀近似。
3. 液体制备：高温熔化后在目标温度附近平衡。
4. 固液共存结构构建：沿一个晶向拼接固体和液体区域，建议从 512-2048 原子开始。
5. 温度 bracket：例如 880, 920, 940, 980, 1020 K。
6. 每个温度运行多条短轨迹，再对关键温度运行长轨迹。
7. 通过固液界面移动方向判断温度高低：
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

1. 得到 Al 的 MPN-OFDFT 熔点估计值和不确定度。
2. 给出与实验值 933.47 K 的相对误差。
3. 与传统 OFDFT 结果进行定性/定量对照，特别关注是否优于已报道 Al OFDFT 熔点偏高约 10% 量级的问题。
4. 明确误差来源：方法误差、有限尺寸误差、模拟时间误差、温控/压控设置误差。

### 阶段 D：纯 Mg 熔点计算

目标：用同一套 MPN-OFDFT 工作流计算纯 Mg 熔点，验证 hcp 金属体系的适用性。

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

1. 得到 Mg 的 MPN-OFDFT 熔点估计值和不确定度。
2. 给出与实验值约 923 K 的相对误差。
3. 比较 Al 和 Mg 的误差模式，判断 MPN 对不同简单金属结构的稳定性。

### 阶段 E：方法可靠性评估与是否进入合金阶段的决策

进入 Al-Mg 合金计算前，需要完成以下判据：

1. Al、Mg 两个纯元素熔点误差均在可接受范围内。
   - 理想目标：误差小于 5%。
   - 可接受探索目标：误差小于 8%，且误差来源清楚。
   - 若误差超过 10%，不建议直接进入合金熔点预测，应先做方法修正。
2. 液体结构 RDF 与实验/KSDFT/可靠势函数结果一致。
3. 固液共存界面在目标温度附近稳定，不出现非物理密度崩塌或异常结晶。
4. 小体系 KSDFT 对照显示 MPN-OFDFT 的能量/力误差在熔化相关构型上可接受。

若通过，则进入 Al-Mg 合金阶段。若不通过，则启动 MPN-KEDF 微调、局域赝势重建或 Δ-ML 修正路线。

## 5. Al-Mg 合金扩展计划

合金阶段不作为第一阶段交付，但任务书需提前保留接口。

初始推荐成分：

1. Al-rich：Al-2at%Mg、Al-5at%Mg、Al-10at%Mg、Al-20at%Mg。
2. Mg-rich：Mg-2at%Al、Mg-5at%Al、Mg-10at%Al。
3. 视资源增加代表性中间成分和金属间化合物附近成分。

主要目标：

1. 计算液相线趋势，而不是一开始追求完整相图。
2. 与 CALPHAD Al-Mg 相图数据比较趋势。
3. 识别 MPN-OFDFT 对混合焓、液体短程有序和固液界面偏析的预测能力。

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
3. KEDF：MPN。
4. 赝势：bulk-derived local pseudopotential，优先使用 MPN 论文或 ABACUS 推荐设置。
5. 基组：plane wave / charge-density grid，按 ABACUS OFDFT 设置。
6. 交换关联：与 MPN 论文设置保持一致，优先 LDA/PBE 中论文采用者。

需要记录的关键参数：

1. `of_kinetic = mpn`
2. `of_method`
3. `of_conv`
4. `of_tole`
5. `of_tolp`
6. `ecutrho` 或等价密度网格参数
7. 赝势文件、版本、来源
8. 时间步长、温控器/压控器、系综

注意：当前 ABACUS 源码中 MPN-KEDF 暂不支持 stress，因此第一阶段默认关闭 `cal_stress`，优先采用 NVT/NVE 固定体积路线；不把 NPT 或需要可靠 stress 的压力耦合结果作为主判据。

### 6.2 熔点计算主方法：固液共存法

固液共存法是第一阶段主方法，因为它比简单升温法更接近平衡熔点，可避免过热和过冷导致的系统误差。

标准流程：

1. 构建固体超胞。
2. 复制为长条形 cell，沿 z 方向或指定方向分区。
3. 固定一半为固体，另一半高温熔化为液体。
4. 拼接后去除约束，在目标温度附近平衡。
5. 运行 NPH/NVE 或弱耦合 NPT/NVT。
6. 判断界面移动方向，逐步 bracket 熔点。

温度估计：

1. 初始温度间隔：40-50 K。
2. 找到 bracket 后缩小到 10-20 K。
3. 最终熔点不确定度由温度 bracket、轨迹长度、重复轨迹和有限尺寸估计共同给出。

### 6.3 辅助方法：Z-method / 升温法

用途：

1. 快速扫描 MPN-OFDFT 的熔化温区。
2. 检查固体过热极限。
3. 为固液共存法选择初始温度范围。

注意：

Z-method 和升温法不作为最终熔点主证据，因为简单金属可出现明显过热，尤其在有限尺寸周期体系中。

### 6.4 小体系 KSDFT 对照

目标：

1. 抽取固体、液体、界面构型。
2. 用 KSDFT 单点评估能量、力、压力差异。
3. 判断 MPN-OFDFT 在熔化相关构型上的误差是否系统偏向固体或液体。

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
    liquid/
    coexist/
      T0900/
        input/
        output/
        analysis/
        metadata.yaml
  mg/
    static/
    liquid/
    coexist/
```

`metadata.yaml` 至少包含：

1. 元素/成分。
2. 结构类型。
3. 原子数。
4. 初始结构来源。
5. ABACUS commit/version。
6. MPN 参数。
7. 赝势文件 checksum。
8. 系综、温度、压力、步长、总步数。
9. 提交脚本和运行机器。
10. 分析脚本版本。

所有图表必须可由脚本重新生成，避免手工处理不可复现。

## 8. 验收标准

第一阶段最终验收需要满足：

1. 代码可自动生成纯 Al 和纯 Mg 的 MPN-OFDFT 输入。
2. 代码可自动解析 ABACUS MD 输出并生成能量、温度、压力、RDF、MSD、固液分数图。
3. 完成 Al 和 Mg 各至少一组固液共存 bracket。
4. 给出 Al/Mg 熔点数值、误差条和与实验对比。
5. 给出有限尺寸、轨迹长度、赝势、KEDF 参数的敏感性初步分析。
6. 形成一份阶段报告，明确是否进入 Al-Mg 合金阶段。

推荐最终报告结构：

1. 方法与输入参数。
2. 静态 benchmark。
3. 液体稳定性测试。
4. Al 熔点结果。
5. Mg 熔点结果。
6. KSDFT 小体系对照。
7. 误差来源分析。
8. 是否进入合金阶段的结论。
9. 下一步 Al-Mg 合金计划。

## 9. 里程碑与时间安排

建议 10 周完成第一阶段。

第 1 周：

1. ABACUS MPN-OFDFT 环境搭建。
2. 收集 Al/Mg 局域赝势。
3. 跑通最小 fcc Al 和 hcp Mg 单点。

第 2 周：

1. 完成 Python 项目骨架。
2. 完成 ABACUS 输入生成器。
3. 完成静态体积扫描。

第 3 周：

1. 完成输出解析器。
2. 完成 RDF、MSD、Q_l、局域结构分类分析。
3. 完成 Al/Mg 液体短程 MD 测试。

第 4-5 周：

1. 构建 Al 固液共存结构。
2. 完成 Al 熔点温度 bracket。
3. 输出 Al 熔点初步结果。

第 6-7 周：

1. 构建 Mg 固液共存结构。
2. 完成 Mg 熔点温度 bracket。
3. 输出 Mg 熔点初步结果。

第 8 周：

1. 做有限尺寸和轨迹长度敏感性分析。
2. 对关键温度做重复轨迹。

第 9 周：

1. 抽取 Al/Mg 固体、液体、界面构型。
2. 做小体系 KSDFT 对照。
3. 分析 MPN-OFDFT 能量/力/压力误差。

第 10 周：

1. 完成阶段报告。
2. 给出是否进入 Al-Mg 合金阶段的决策。
3. 整理代码、输入模板、示例数据和复现实验说明。

## 10. 主要风险与应对方案

风险 1：MPN 在液体或固液界面上的训练覆盖不足。

应对：

1. 必须做液体和界面 KSDFT 小体系对照。
2. 若液体相对能量有系统偏差，考虑重新训练/微调 MPN-KEDF 或引入 Δ-ML 修正。

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

风险 5：ABACUS MPN-OFDFT MD 稳定性不足。

应对：

1. 先做短时间小体系液体测试。
2. 调整电子密度优化收敛阈值、时间步长、网格参数。
3. 如仍不稳定，先用静态 single-point + 外部 MD 框架评估是否可行，再决定是否修改 ABACUS 接口。

## 11. 参考文献与软件来源

1. Liang Sun and Mohan Chen, "Machine learning based nonlocal kinetic energy density functional for simple metals and alloys", Phys. Rev. B 109, 115135 (2024).  
   https://arxiv.org/abs/2310.15591

2. ABACUS open-source repository.  
   https://github.com/deepmodeling/abacus-develop

3. ABACUS documentation, OFDFT input keywords including `of_kinetic = mpn`.  
   https://abacus.deepmodeling.com/en/latest/advanced/input_files/input-main.html

4. Z. Bai et al., "Nucleation of Supercooled Liquid Aluminum by an Orbital-Free Density Functional Theory Molecular Dynamics Simulation".  
   https://arxiv.org/abs/2306.11933

5. Xuecheng Shao et al., "DFTpy: An efficient and object-oriented platform for orbital-free DFT simulations".  
   https://arxiv.org/abs/2002.02985

6. Shashikant Kumar et al., "Kohn-Sham accuracy from orbital-free density functional theory via Δ-machine learning".  
   https://arxiv.org/abs/2310.06598

## 12. 第一阶段最终产物清单

1. `mpn_melting` Python 代码包。
2. ABACUS MPN-OFDFT Al/Mg 输入模板。
3. Al/Mg 静态 benchmark 数据。
4. Al/Mg 液体 MD 稳定性数据。
5. Al/Mg 固液共存熔点计算数据。
6. 自动分析图表。
7. KSDFT 小体系对照数据。
8. 阶段报告：是否进入 Al-Mg 合金阶段。

## 13. 决策门槛

若满足以下条件，进入 Al-Mg 合金液相线/固相线计算：

1. Al 和 Mg 熔点误差均小于 8%，且至少一个元素小于 5%。
2. 液体 RDF、扩散行为和固液界面结构无明显非物理异常。
3. KSDFT 小体系对照未发现明显固液相对能量系统偏置。
4. 固液共存法结果可由重复轨迹复现。

若不满足，则不直接进入合金预测，优先执行：

1. MPN-KEDF 数据集补充与微调。
2. 局域赝势重建/筛选。
3. OFDFT + Δ-ML energy/force correction。
4. 与 KSDFT/ML potential 混合的多层级 workflow。
