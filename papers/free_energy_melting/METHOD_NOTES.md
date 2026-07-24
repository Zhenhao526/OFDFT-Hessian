# 文献学习后对当前 MPN-Al 自由能流程的修订

## 1. 正式计算应采用的路径

固相：

`Einstein/harmonic crystal -> solid-specific pair model -> MPN-KEDF`

液相：

`ideal gas -> scaled Uhlenbeck-Ford -> liquid-specific pair model -> MPN-KEDF`

当前统一的 `pair_reference_guarded_full` 保留作代码与 lambda 端点验证。
正式结果应分别拟合固相和液相参考势。

## 2. 参考势质量的真正判据

文献中的直接目标不是能量 RMSE 或力 RMSE 本身，而是最小化

`Var(U_MPN - U_ref)`。

常数能量偏移不影响热力学积分效率。波动越小，积分曲率、统计误差和所需
lambda 点越少。每个相应报告：

- `std(Delta U)/atom`
- 积分自相关时间
- 相邻 lambda 的双向 FEP/重加权 ESS
- 正向和反向自由能差的一致性

## 3. lambda 方案

先运行当前 9 点均匀网格作为 pilot：

`0, 0.125, ..., 0.875, 1`

随后按 integrand 曲率和相邻窗口重叠自适应加点。不得只因为某个窗口噪声大
就整体延长全部窗口。若 lambda 接近 0 时发散，应先改参考势硬核或变量变换。

## 4. 液相绝对自由能

原始 UF (`p=1`) 太软，真实势贡献会在 lambda 接近 0 时出现大正波动。
Leite 等展示 `p=50` 的 scaled UF 可使驱动力平滑；论文还给出
`p=25, 50, 75, 100` 的 EOS 和 excess Helmholtz free energy 数据。

建议：

1. 从 Al 液体最近邻分布选择 sUF 的 `sigma`。
2. 扫描 `p` 和 `sigma`，最小化 `Var(U_pair_liq-U_sUF)`。
3. 检查整个 sUF-to-pair 路径始终为液体。
4. 用文献 EOS 得到 `F_sUF-F_ideal`。
5. 再积分 `sUF -> pair_liq -> MPN`。

WCA 是备选。它短程、纯排斥、无液气相变且有成熟 EOS；若 sUF 参数实现或
复核困难，可用 Sun 等的 WCA 路径做独立交叉验证。

## 5. 固相绝对自由能

采用 Frenkel-Ladd/Einstein crystal，并显式处理：

- 固定质心或固定单原子的约束。
- 约束对应的解析自由能修正。
- 弹簧常数选择和 lambda 端点积分。
- `ln(N)/N` 主导有限尺寸项。
- 剩余 `1/N` 外推。

108 原子只适合 pilot。至少增加一个更大尺寸进行固相有限尺寸检查。

## 6. 精度目标

Al 的熔化熵约为 `1.4 k_B/atom`。Vocadlo-Alfe 指出自由能误差
`0.01 eV/atom` 可造成约 `80 K` 的熔点偏移。

因此建议误差预算：

- 每条参考路径统计误差：不高于 `1-2 meV/atom`
- lambda 数值积分误差：不高于 `1 meV/atom`
- k 点/网格/电子收敛修正：不高于 `1-2 meV/atom`
- 总随机误差目标：约 `3 meV/atom`

模型系统误差应单独报告，不能混入统计误差。

## 7. 体积与温度处理

单个零压体积只够得到一个锚点。完整 Gibbs 自由能应从多个 `(V,T)` 点拟合
`F(V,T)`，再最小化

`G(P,T) = min_V [F(V,T) + P V]`。

零压时取该极小值。随后可直接比较多个温度，或用 Gibbs-Helmholtz 关系结合
固、液焓差传播到其他温度。

## 8. 必须补充的电子自由能核查

有限电子温度下，MD 力对应 Mermin 电子自由能面。热力学积分中的
`U_MPN` 必须与生成力所用的电子热力学势一致。正式计算前应确认 ABACUS
输出中用于 `Delta U` 的量是电子自由能而非另一个未配套的内部能定义。

## 9. 对当前运行的判定

- 扩展 NVT 体积扫描仍是必须步骤。
- 9-lambda 运行定位为 pair-to-MPN pilot，不是完整熔点结果。
- pilot 后首先根据固、液各自的 `Delta U` 方差决定是否拆分参考势。
- 在 Einstein 和 sUF/WCA 绝对参考路径完成前，不得把积分结果称为绝对自由能。
