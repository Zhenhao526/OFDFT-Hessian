# Al 自由能熔点方法阅读路线

本目录对应当前 MPN-KEDF 自由能熔点任务。目标是在零外压下分别计算
固相和液相的绝对 Gibbs 自由能，并由

`Delta G(T) = G_liquid(T) - G_solid(T) = 0`

确定熔点。

## 第一优先级：直接对应 Al 和第一性原理熔点

1. de Wijs, Kresse, Gillan, *First-order phase transitions by
   first-principles free-energy calculations: The melting of Al*,
   Phys. Rev. B **57**, 8223 (1998).
   DOI: https://doi.org/10.1103/PhysRevB.57.8223

   这是最直接的模板。固、液自由能均通过 coupling-constant integration
   获得；固相参考为准谐晶体，液相参考为 Lennard-Jones 流体，并讨论了
   k 点和数值精度对 Al 熔点的影响。

2. Vocadlo and Alfe, *Ab initio melting curve of the fcc phase of aluminum*,
   Phys. Rev. B **65**, 214105 (2002).
   DOI: https://doi.org/10.1103/PhysRevB.65.214105
   Open preprint: https://arxiv.org/abs/cond-mat/0108460

   给出完整 Al 熔化曲线。方法是准谐固体自由能，加固相非谐修正和液相
   热力学积分；适合学习如何把单个温度点扩展为熔化曲线。

3. Alfe, Gillan, Price, *Complementary approaches to the ab initio
   calculation of melting properties*, J. Chem. Phys. **116**, 6170 (2002).
   DOI: https://doi.org/10.1063/1.1460865
   Open preprint: https://arxiv.org/abs/cond-mat/0111510

   解释自由能法和直接共存法的关系、参考势修正和有限尺寸误差。用于把
   当前 1728 原子共存结果与 108 原子自由能结果放在同一热力学框架中。

## 第二优先级：参考势到 MPN 的热力学积分

4. Zhu, Grabowski, Neugebauer, *Efficient approach to compute melting
   properties fully from ab initio with application to Cu*,
   Phys. Rev. B **96**, 224202 (2017).
   DOI: https://doi.org/10.1103/PhysRevB.96.224202

   TOR-TILD 方法。分别为固相和液相拟合优化参考势，再积分到第一性原理
   Hamiltonian。它与当前 `pair reference -> MPN` 多 lambda 路径最接近。

5. Zhu, Koermann, Ruban, Neugebauer, Grabowski, *Performance of the standard
   exchange-correlation functionals in predicting melting properties fully
   from first principles: Application to Al and magnetic Ni*,
   Phys. Rev. B **101**, 144108 (2020).
   DOI: https://doi.org/10.1103/PhysRevB.101.144108

   将 TOR-TILD 真正应用到 Al，并给出熔点、熔化焓和熔化熵。它是完成
   9 个 lambda 试窗口后最应逐项对照的现代论文。

6. Kirkwood, *Statistical Mechanics of Fluid Mixtures*,
   J. Chem. Phys. **3**, 300 (1935).
   DOI: https://doi.org/10.1063/1.1749657

   coupling-parameter integration 的基础来源：
   `Delta F = integral_0^1 <dH(lambda)/dlambda>_lambda dlambda`。

## 第三优先级：固相绝对自由能

7. Frenkel and Ladd, *New Monte Carlo method to compute the free energy of
   arbitrary solids*, J. Chem. Phys. **81**, 3188 (1984).
   DOI: https://doi.org/10.1063/1.448024

   Einstein crystal 路径的基础论文。需要重点理解质心约束、谐振子强度和
   lambda 接近端点时的积分行为。

8. Polson, Trizac, Pronk, Frenkel, *Finite-size corrections to the free
   energies of crystalline solids*, J. Chem. Phys. **112**, 5339 (2000).
   DOI: https://doi.org/10.1063/1.481102
   Open preprint: https://arxiv.org/abs/cond-mat/9909162

   108 原子结果必须参考此文处理 `ln(N)/N` 和 `1/N` 有限尺寸项。

9. Vega and Noya, *Revisiting the Frenkel-Ladd method to compute the free
   energy of solids: The Einstein molecule approach*,
   J. Chem. Phys. **127**, 154113 (2007).
   DOI: https://doi.org/10.1063/1.2790426

   Frenkel-Ladd 的实现型补充，可用于交叉检查固定质心和固定单个原子的
   两种约束写法。

## 第四优先级：液相绝对自由能

10. Leite, Freitas, Azevedo, de Koning, *The Uhlenbeck-Ford model: Exact
   virial coefficients and application as a reference system in fluid-phase
   free-energy calculations*, J. Chem. Phys. **145**, 194101 (2016).
   DOI: https://doi.org/10.1063/1.4967775
   Open preprint: https://arxiv.org/abs/2204.06744

   scaled Uhlenbeck-Ford 流体具有已知自由能且仅稳定为流体，可避免直接
   关闭真实液体相互作用时的原子重叠和液-气相变问题。当前液相绝对自由能
   路径应优先采用此方法。

11. Sun, Brodholt, Li, Vocadlo, *Melting properties from ab initio free
    energy calculations: Iron at the Earth's inner-core boundary*,
    Phys. Rev. B **98**, 224301 (2018).
    DOI: https://doi.org/10.1103/PhysRevB.98.224301

    展示 WCA 纯排斥流体作为液相参考的另一条成熟路线，可与 sUF 比较。

## 建议对应到当前代码的热力学路径

固相：

`Einstein crystal -> solid pair reference -> MPN-KEDF`

液相：

`ideal gas -> scaled UF -> liquid pair reference -> MPN-KEDF`

其中最后一段均使用当前实现的

`U_lambda = (1-lambda) U_pair + lambda U_MPN`

在每个 lambda 窗口采样 `<U_MPN-U_pair>_lambda`。固、液最好分别拟合参考势；
若共用参考势导致曲率大或相邻窗口 ESS 低，应拆分模型，而不是只增加采样长度。

## 阅读时需要提取的实现参数

- 参考体系的解析自由能公式及单位约定。
- 质心或单粒子约束对应的自由能修正。
- lambda 变量变换、端点加密和积分公式。
- 每窗口热化长度、生产长度、自相关和误差估计。
- 液相参考势的参数选择与防重叠条件。
- 从 `F(T,V)` 到零压 `G(T,P=0)` 的体积/EOS处理。
- Gibbs-Helmholtz 温度积分和熔点根求解。
- 原子数、k 点、电子温度和有限尺寸收敛。
