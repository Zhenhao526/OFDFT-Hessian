# Hessian / HVP 训练文献调研与 OFDFT-Hessian 路线建议

更新日期：2026-07-31  
范围：分子 Hessian、Hessian-vector product（HVP）、振动频率、二阶导数友好
的机器学习势、OFDFT 密度响应与密度优化。  
检索方法：直接检索期刊、会议、arXiv、作者代码页和数据集页面；优先采用论文
原文和出版社页面，不使用自动论文雷达。

## 1. 结论先行

目前最值得做的不是继续延长十分子 EGFH 训练，也不是用一个独立 Hessian 头
替代标量能量，而是把训练对象改成：

> **学习标量总能量 \(E_\theta(R,c)\)，同时用随机投影的、密度弛豫后的保守
> HVP 约束其二阶响应；新增密度响应模块只作为响应求解器/预条件器和辅助监督，
> 最终力与 Hessian 仍由同一个标量能量产生。**

文献与当前实验共同支持以下判断：

1. **Hessian 信息确实有用。** 正式发表的 JCTC 2025 和 Scientific Data
   HORM 工作均显示，加入曲率信息能显著改善非平衡构型、Hessian、频率和过渡态
   任务，并大幅提高数据效率。
2. **不需要每步构造完整 Hessian。** 随机 HVP、随机行/列采样可在保留大部分
   收益的同时避免 \(O(N^2)\) 输出与存储。Projected Hessian Learning（PHL）
   报告 HVP 训练相对完整 Hessian 训练超过 24 倍加速。
3. **“把位移构型当作普通 E/F 样本”不等于学习曲率。** PFT 的对照实验甚至
   发现，只用位移 E/F/stress 微调会使 Hessian 误差恶化；必须显式配对两个端点
   并约束方向曲率。
4. **当前项目的主要问题不是简单的训练步数。** 当前 step-1300 在一个训练分子
   上仍有 relative Frobenius 1.155、频率 MAE 1563 cm\(^{-1}\)、27 个虚频，
   且密度严格收敛从 step-100 的 77/90 降到 44/90。训练 loss 收敛而物理结果
   几乎不改善，说明目标定义、密度响应和采样覆盖比优化时长更关键。
5. **当前 H loss 与最终评估存在对象错位。** 代码中的
   `PairRelaxedForceSecantLoss` 使用模型在给定 KS 密度系数端点上的标量能量
   导数，匹配 PBE 端点力的割线；代码注释明确说明它“不是 implicit
   density-relaxed OFDFT HVP”。最终评估却在每个位移点优化模型自身的
   \(c_\theta^\*(R)\)。所以训练约束的是参考密度路径上的曲率代理，推理需要的是
   学习泛函自身极小值分支的曲率。
6. **直接 Hessian 头可以做，但不能作为主线的物理 Hessian。** HIP 一类
   SE(3)-等变对称 Hessian 头速度非常快，但一般不保证它是某个标量能量的二阶
   导数。它适合作为辅助读出、蒸馏目标或响应求解器初值，不应替换保守主线。
7. **若目标包含 IR 强度，还需要偶极矩响应。** Hessian 给出频率和正规模；
   双谐近似下的 IR 强度还依赖 \(\partial\mu/\partial Q\)。仅训练 Hessian
   无法得到完整 IR 强度。

## 2. 当前实现与目标之间的数学差异

令学习到的完整 OFDFT 标量能量为

\[
E_\theta(R,c),
\]

其中 \(R\) 是核坐标，\(c\) 是满足电子数约束的密度系数。推理时的密度和弛豫
能量为

\[
c_\theta^\*(R)=\arg\min_c E_\theta(R,c),\qquad
\bar E_\theta(R)=E_\theta(R,c_\theta^\*(R)).
\]

在约束密度切空间中，若 \(E_{cc}\) 可逆，则隐函数定理给出

\[
\frac{\partial c^\*}{\partial R}
=-E_{cc}^{-1}E_{cR},
\]

以及真正需要的密度弛豫 Hessian

\[
H_{\mathrm{relaxed}}
=\frac{\partial^2\bar E}{\partial R^2}
=E_{RR}-E_{Rc}E_{cc}^{-1}E_{cR}.
\]

因此一个方向 \(v\) 的 HVP 不需要完整 Hessian：

1. 计算 \(a=E_{cR}v\)；
2. 在电子数守恒切空间求解 \(E_{cc}x=a\)；
3. 返回
   \[
   H_{\mathrm{relaxed}}v=E_{RR}v-E_{Rc}x.
   \]

当前训练的端点力割线在形式上是合理的参考标签：

\[
H_{\mathrm{PBE}}v\approx
-\frac{F_{\mathrm{PBE}}(R+\epsilon v)
-F_{\mathrm{PBE}}(R-\epsilon v)}{2\epsilon}.
\]

问题在于模型侧当前并没有对 \(c_\theta^\*(R\pm\epsilon v)\) 重新求极小，也没有
显式约束上面的 Schur 补响应。只有当给定的 KS 密度在学习泛函下也足够接近
驻点、且学习到的 \(E_{cc}\) 和 \(E_{cR}\) 正确时，当前割线代理才会接近最终
需要的 relaxed HVP。step-1300 密度优化显著恶化，说明这个条件目前不成立。

## 3. 核心文献逐篇总结

以下将正式同行评审论文标为 **A**，顶级会议论文标为 **A-**，尚未完成同行
评审的近期预印本标为 **B**。B 类工作可提供方法线索，但不能单独作为最终结论
依据。

### 3.1 Rodriguez et al., Does Hessian Data Improve the Performance of ML Potentials?

- 出处：JCTC 2025，同行评审，**A**。
- 原文：[ACS / JCTC](https://pubs.acs.org/doi/10.1021/acs.jctc.5c00402)
- 方法：比较只训练能量 E、训练 E-F、以及训练 E-F-H 的 ANI 类势。
- 数据：稳定点、过渡态和反应路径；测试包含 IRC 与 normal-mode sampling
  的非平衡构型。
- 主要结果：Hessian 监督明显改善对非平衡构型的能量、力、Hessian、振动谱和
  反应路径预测。论文报告只用约 2% Hessian 数据的模型在部分能量任务上可超过
  使用 80% E/F 数据的模型，说明每个 Hessian 标签的信息密度很高。
- 代价：完整 Hessian 训练每个 epoch 约贵 25 倍。
- 局限：使用的是普通 MLIP 和完整 Hessian；没有电子密度内层优化。
- 对本项目的意义：确认二阶监督值得做，但直接完整 Hessian 训练不适合当前
  OFDFT；应转向 HVP。

### 3.2 Rodriguez et al., Projected Hessian Learning (PHL)

- 出处：arXiv 2026，预印本，**B**。
- 原文：[arXiv](https://arxiv.org/abs/2603.04523)
- 方法：用 Hutchinson 随机迹估计将
  \(\|H_\theta-H_{\rm ref}\|_F^2\) 变成随机向量 \(v\) 上的
  \(\|(H_\theta-H_{\rm ref})v\|^2\)。每个 minibatch 重新采样 Rademacher
  或 Gaussian probe。
- 主要结果：随机 HVP 与完整 Hessian 训练接近；在 normal-mode sampled
  外推集上，Hessian RMSE 相对 E-F 基线下降约 77%，力误差也下降约 48%；
  HVP 训练相对完整 Hessian 训练超过 24 倍加速。
- 重要细节：固定一个方向时，随机稠密 probe 优于只取一个坐标列；每批重采样
  时，两者在小分子上可接近。
- 局限：普通 MLIP 的 HVP 只需能量的二阶 AD；本项目还包含昂贵的密度响应
  线性求解，不能直接照搬其速度结论。
- 对本项目的意义：下一轮 H 监督应使用随机内部空间 probe，并把损失直接写成
  HVP 差，而不是继续固定一组有限方向长期拟合。

### 3.3 Cui et al., HORM: A Large Scale Molecular Hessian Database

- 出处：Scientific Data 2026，同行评审数据论文，**A**。
- 原文：[Nature / Scientific Data](https://www.nature.com/articles/s41597-025-06350-5)
- 数据：184 万个 \(\omega\)B97x/6-31G(d) Hessian，主要来自
  Transition1x；包含反应物、产物、过渡态和反应路径附近构型。
- 方法：每分子每 epoch 随机采样少量 Hessian 行，用 VJP 计算；成本对采样行数
  近似线性，而不是构造全矩阵。
- 主要结果：多种 autograd MLIP 的 Hessian / 特征值误差显著下降；对于直接
  预测力的 EquiformerV2，Hessian 约束还大幅降低不对称性，并显著改善过渡态
  搜索。
- 局限：只覆盖小型 CHON 反应体系，理论水平与本项目 PBE 不同，也没有密度
  系数或密度响应标签。
- 对本项目的意义：可借鉴随机行/HVP 训练和反应/非平衡采样；HORM 不适合作为
  最终 PBE-OFDFT 标签，但可用于预训练几何表征或辅助 Hessian 头。

### 3.4 Koker et al., PFT: Phonon Fine-tuning for MLIPs

- 出处：arXiv 2026，预印本，**B**。
- 原文：[arXiv](https://arxiv.org/abs/2601.07742)
- 方法：每结构采样一个 Hessian 列，即一次 HVP；用三重反向传播训练能量模型，
  并混合原有 E/F/stress 数据防止灾难性遗忘。
- 主要结果：报告平均约 55% 的声子热力学误差改善；上游任务退化通过 replay
  控制在约 1%。
- 最关键的对照：只在位移后的 E/F/stress 样本上普通微调，Hessian 误差反而
  变差。显式的成对曲率损失是必要的。
- 局限：材料声子而非分子 OFDFT；目前是预印本。
- 对本项目的意义：保留 E/G/F replay，逐步加入 HVP；不能把位移端点拆开当作
  普通训练样本。

### 3.5 Yin et al., Hessian INformed Training (HINT)

- 出处：arXiv 2026，预印本，**B**。
- 原文：[arXiv](https://arxiv.org/abs/2603.25373)
- 方法组合：低理论水平 Hessian 预训练、针对性构型采样、Hessian 权重 curriculum、
  Hutchinson 随机投影。
- 数据策略：大量 xTB Hessian 预训练，再用稀疏 DFT Hessian 微调；优先选择高能、
  高曲率或局部密度稀疏的构型。
- 主要结果：报告只用 1 万个 DFT Hessian 即接近约 170 万完整数据的性能，
  DFT Hessian 需求降低约 170 倍；AlphaNet Hessian MAE 从 0.415 降至
  0.105 eV/Å²，进一步预训练/微调达到约 0.075 eV/Å²。
- 局限：预印本，覆盖多个任务，部分结果尚需独立复现。
- 对本项目的意义：H 权重应渐增；PBE 标签应优先投到高曲率、密度优化困难和
  当前模型不确定的分子/方向，而非均匀生成完整 Hessian。

### 3.6 Burger et al., Shoot from the HIP

- 出处：arXiv 2025/2026 更新，预印本，**B**。
- 原文：[arXiv](https://arxiv.org/abs/2509.21624)
- 方法：在带 \(l\le2\) irreps 的 SE(3)-等变 backbone 上直接读出原子对
  \(3\times3\) Hessian block；由 \(l=0,1,2\) 张量构造并强制矩阵对称。
- 损失：除矩阵误差外，还加入最低特征值/特征向量的子空间损失，关注几何优化
  和过渡态最需要的低模。
- 主要结果：论文报告比 AD Hessian 快 10–100 倍、峰值内存低 2–3 倍，并在
  HORM 上得到更低的 Hessian、特征值和特征向量误差。
- 根本局限：直接读出的 Hessian 一般不保证等于同一模型标量能量的二阶导数，
  因而不保证全局可积性和与力的一致性。
- 对本项目的意义：可做快速辅助头，但应增加
  \(\|H_{\rm head}v-H_{\rm relaxed}v\|\) 一致性蒸馏；最终频率和动力学结论
  仍以标量能量的保守 Hessian 为准。

### 3.7 Yuan et al., Analytical ab initio Hessian from a deep learning potential

- 出处：Nature Communications 2024，同行评审，**A**。
- 原文：[Nature Communications](https://www.nature.com/articles/s41467-024-52481-5)
- 方法：使用二阶连续、可微的 NewtonNet，从标量能量经 AD 得到力和 Hessian。
- 数据策略：ANI-1 的两千多万构型预训练，加 Transition1x 的约 964 万反应
  构型；另外加入约 123 万压缩键构型补足正曲率和高能区域。
- 主要结果：240 个未见反应的过渡态优化中，ML Hessian 可减少约 2–3 倍优化
  步数，并比 DFT Hessian 快超过 1000 倍。
- 重要观察：只靠接近平衡的数据会系统性低估 Hessian 特征值；加入压缩构型后
  特征值 RMSE 和模态重合度改善。
- 局限：成功依赖巨量、任务匹配的数据；不能推出十个分子的 E/F 训练自然会有
  好 Hessian。
- 对本项目的意义：加入沿正常模、随机内部坐标和压缩键方向的非平衡构型；检查
  是否存在系统性“势能面软化”。

### 3.8 Gönnheimer et al., Beyond Numerical Hessians

- 出处：JCTC 2025，同行评审，**A**。
- 原文：[DOI / JCTC](https://doi.org/10.1021/acs.jctc.4c01790)
- 方法：在 MACE 中比较 AD Hessian 与力的有限差分 Hessian。
- 主要结果：单精度下 AD 的数值精度可比有限差分高两个数量级以上；完整 AD
  Hessian 还可比数值 Hessian 快约 1.5–2.7 倍。有限差分对位移步长和精度非常
  敏感。
- 局限：完整 AD 与有限差分最终仍是 \(O(N^2)\)；没有电子密度优化噪声。
- 对本项目的意义：当前 90 个端点只有 44 个严格收敛时，频率结果不能只归因于
  网络。必须做位移步长扫描，并将密度残差误差与 \(1/\epsilon\) 的噪声放大
  分开；在可行处用经过校验的隐式 HVP 代替外层有限差分。

### 3.9 Williams et al., Hessian QM9

- 出处：Scientific Data 2025，同行评审，**A**。
- 原文：[Nature / Scientific Data](https://www.nature.com/articles/s41597-024-04361-2)
- 数据：41,645 个 QM9 平衡分子，在真空和三种隐式溶剂中计算数值 Hessian，
  理论水平为 \(\omega\)B97x/6-31G*。
- 建模结果：论文中的 E(3)-等变 GNN 微调后，对高于 400 cm\(^{-1}\) 的频率
  报告约 9.49 cm\(^{-1}\) MAE；低于 400 cm\(^{-1}\) 的低频区改善有限。
- 局限：平衡构型、理论水平不匹配、无 \(c\) 或响应标签；低频小特征值对矩阵
  扰动天然敏感。
- 对本项目的意义：适合做外部 sanity check、几何 backbone 或辅助头预训练，
  但不能替代 PBE relaxed-density 标签；评价时应把低频与高频分开报告。

### 3.10 Deng et al., Systematic softening in universal MLIPs

- 出处：npj Computational Materials 2025，同行评审，**A**。
- 原文：[Nature / npj Computational Materials](https://www.nature.com/articles/s41524-024-01500-6)
- 发现：M3GNet、CHGNet、MACE-MP-0 在表面、缺陷、迁移势垒、高能构型和声子
  上普遍低估能量、力与曲率；229 个声子材料中超过 90% 出现频率软化。
- 原因：预训练数据主要来自靠近极小值的离子弛豫轨迹，对高曲率/高能区域覆盖
  不足。
- 修正：局部少量针对性高能数据即可明显校正系统偏差。
- 局限：固体材料，不是分子 OFDFT。
- 对本项目的意义：应增加“曲率斜率”诊断，即沿 probe 比较模型与 PBE 力割线
  的线性回归斜率；若整体小于 1，优先做采样修正而非盲目加模型容量。

### 3.11 Fu et al., Learning Smooth and Expressive Interatomic Potentials (eSEN)

- 出处：ICML 2025 oral，同行评审，**A-**。
- 原文：[ICML](https://icml.cc/virtual/2025/oral/47222)；
  [arXiv 全文](https://arxiv.org/abs/2502.12147)
- 核心观点：普通 E/F 测试误差低不保证声子、热导和长时间动力学好；模型必须在
  实际数值上保持保守、连续和高阶导数平滑。
- 关键设计：力从标量能量求导；使用光滑 envelope 使边消息及其高阶导数在
  cutoff 处趋零；避免最大邻居数导致的邻居集合跳变和离散网格伪影。
- 结果：eSEN 在约一万材料的 MDR phonon benchmark 达到当时 SOTA；直接力模型
  在小位移下出现虚频和不收敛的声子支，即使增大位移后某些积分热力学指标看似
  改善。
- 对本项目的意义：GraphFormer 的 GELU/SiLU 本身足够平滑，但必须专门审计
  动态局部 frame、邻居选择、cutoff、natural reparameterization 和 eig 路径
  的二阶连续性。直接力头不应成为主线。

### 3.12 Chmiela et al., sGDML / gradient-domain learning

- 出处：Science Advances 2017、Nature Communications 2018，同行评审，**A**。
- 原文：[Nature Communications](https://www.nature.com/articles/s41467-018-06169-2)
- 方法：直接在梯度域学习保守力场；核函数本身来自标量能量核的 Hessian，同时
  加入分子置换对称性。
- 结果：对单个分子可用很少的构型得到高精度保守力和稳定动力学。
- 局限：全局核方法随分子数和化学空间扩展困难，不适合替代当前 R+c
  GraphFormer。
- 对本项目的意义：物理原则非常重要——力、方向曲率和 Hessian 应当共享一个
  标量势，而不是独立预测后再希望彼此一致。

### 3.13 Remme et al., Stable and Accurate OFDFT Powered by ML

- 出处：JACS 2025，同行评审，**A**；当前项目的基础工作。
- 原文：[DOI / JACS](https://doi.org/10.1021/jacs.5c06219)；
  [arXiv](https://arxiv.org/abs/2503.00443)
- 方法：以核坐标 \(R\) 和密度系数 \(c\) 为输入，用 tensorial GraphFormer
  预测标量能量；对有效势做扰动，联合训练能量和对密度系数的泛函梯度。
- 关键结论：只用基态密度不足以定义基态附近的密度能量面；必须提供偏离极小值
  的 \(c\) 和相应梯度标签。
- 论文边界：主要是平衡构型，并明确指出对几何优化需要的高能构型泛化仍有限。
- 对本项目的意义：核坐标与密度响应必须联合采样。只沿 \(R\) 位移、却不系统
  约束 \(E_{cc}\) 与 \(E_{cR}\)，无法保证正确的 relaxed Hessian。

### 3.14 Meyer et al., Simultaneous Training on KEDF and Functional Derivative

- 出处：JCTC 2020，同行评审，**A**。
- 原文：[ACS / JCTC](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.0c00580)
- 发现：能量拟合准确并不意味着泛函导数准确；而 OFDFT 的密度极小化直接依赖
  泛函导数。因此需要同时训练泛函值和导数。
- 对本项目的意义：保留 G loss 是正确的。当前问题不是去掉 G，而是让 G 在
  实际参数更新中具有足够梯度规模，并进一步监督密度二阶响应。

### 3.15 Moldabekov et al., Imposing Correct Jellium Response Is Key

- 出处：Physical Review B 2023，同行评审，**A**。
- 原文：[APS / PRB](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.108.235168)
- 方法：比较多种 OFDFT KEDF 的线性和非线性密度响应；检查均匀电子气极限和
  真实材料。
- 发现：能否满足正确的均匀电子气/Lindhard 响应，与非均匀真实体系的密度响应
  精度有很强相关性。
- 局限：重点是周期材料和传统 KEDF，不是小分子 GraphFormer。
- 对本项目的意义：二阶泛函导数 \(E_{cc}\) 和密度响应不应只被当作产生核
  Hessian 的中间数值量，而应成为独立训练与验证对象；可加入物理响应先验或
  Rayleigh quotient 稳定性约束。

### 3.16 Remme & Hamprecht, Surrogate Functionals for ML-OFDFT

- 出处：Journal of Chemical Physics 2026，同行评审，**A**。
- 原文：[JCP / DOI](https://doi.org/10.1063/5.0336653)；
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/42467487/)
- 方法：提出 gradient-descent-improvement（GDI）损失，使固定密度优化算法的
  每一步都向真实基态密度收缩；训练时沿模型实际访问的密度优化轨迹自适应采样。
- 结果：在 QM9/QMugs 上用基态密度标签即可获得有竞争力的密度误差，并改善
  密度优化的缩放。
- 根本限制：surrogate functional 只需把优化器带到正确密度，不要求其能量是
  真实物理能量。因此不能直接拿它的标量能量计算本项目所需的物理力/Hessian。
- 对本项目的意义：可把 GDI 作为真实 E/G/F/H 泛函的辅助密度稳定项，或训练
  独立的密度优化器/预条件器；不能用纯 surrogate energy 替换物理主能量。

### 3.17 Pracht et al., Efficient Composite Infrared Spectroscopy

- 出处：JCTC 2024，同行评审，**A**。
- 原文：[ACS / JCTC](https://pubs.acs.org/doi/10.1021/acs.jctc.4c01157)
- 方法：在双谐近似下组合不同方法得到谐振频率、正规模和偶极矩导数，比较 xTB、
  MACE-OFF23 与专门的偶极矩模型。
- 关键物理关系：频率来自质量加权 Hessian；IR 强度来自偶极矩沿正规模的导数
  平方。二者可以来自不同的低成本模型。
- 对本项目的意义：若项目目标仅是“红外频率”，当前 Hessian 主线即可；若还要
  预测谱峰强度，需另加 E(3)-等变偶极矩向量头及偶极有限差分/导数监督。

## 4. 扩展读物与较低优先级路线

| 工作 | 价值 | 为什么不是当前主线 |
|---|---|---|
| Domenichini & Dellago, *Molecular Hessian matrices from random forest regression*, JCP 2023 | 用冗余内坐标和 Wilson B 矩阵预测 Hessian，说明内坐标/局部 block 有用 | 手工特征、直接 Hessian、跨化学空间能力有限 |
| Han et al., *Real-time interpretation of neutron vibrational spectra*, 2025 preprint | 直接对称等变 Hessian 可用于实时谱解释 | 非保守，证据仍主要是预印本 |
| Bhatia et al., *Benchmarking MLIPs for molecular IR*, 2026 preprint | 对 SchNet、PaiNN、MACE 等能量/力/偶极模型做 IR 基准，等变模型迁移更好 | 发表状态较弱，且没有 OFDFT 内层密度 |
| Fang et al., *Phonon predictions with E(3)-equivariant GNNs*, 2023/2024 | 展示从等变标量能量 AD Hessian，并讨论 IR/Raman 对称性 | 规模和证据弱于后来的 Hessian QM9、eSEN、PFT |
| THEMol, 2026 preprint | 同时包含扭转、Hessian、弛豫轨迹和 MBIS 数据 | 新数据集、尚未充分验证，理论水平不匹配 |

## 5. 对当前失败模式的诊断

### 5.1 训练与推理的密度分支不一致

当前 H loss 在每个端点使用数据集中给定的 KS 密度系数；最终 Hessian 则沿
学习泛函自身的 \(c_\theta^\*(R)\) 分支。G loss 只能约束一阶驻点误差，不能
单独保证：

- \(E_{cc}\) 的正定性与条件数；
- 核位移引起的 \(E_{cR}\)；
- \(\partial c^\*/\partial R\)；
- Schur 补响应项 \(E_{Rc}E_{cc}^{-1}E_{cR}\)。

step-1300 的密度优化平均迭代数和失败比例显著上升，是这一问题的直接证据。

仓库已有解析 relaxed-HVP 审计已经证明 Schur 补公式和参数梯度在数值上可做对，
但当时唯一满足严格残差门槛并支持参数伴随的 dense solver，单 probe 训练估算约
112 s，峰值 GPU 分配约 68 GB。也就是说，PHL 在普通 MLIP 上的“接近力计算
成本”不能直接套到当前 OFDFT。这里必须同时解决响应线性系统的条件数、矩阵自由
求解和伴随求导成本；这也是新增响应头应优先作为预条件器而非最终物理输出的原因。

### 5.2 目标梯度尺度和跨方向干扰

早期 EGFH10 记录中，E 的参数梯度范数比 G 大约四个数量级、比 H 大数百倍。
后续 MACE/GraphFormer 容量分支又观察到 F 与 HVP 梯度负余弦，以及多方向增加
后完整 Hessian 几乎不改善。这说明固定 loss weight 不能代表实际优化权重。

需要：

- 基于运行中梯度范数的尺度归一化；
- curriculum：先稳定 E/G/F，再逐渐提升 HVP；
- 每批重采样方向，避免对固定方向记忆；
- 对明显冲突的目标使用受约束更新或 PCGrad，但不能以牺牲 E/G 物理面为代价。

### 5.3 数据量和方向覆盖不足

十个分子和每分子一个方向只能验证代码通路，不能验证 Hessian 泛化。即使每个
分子增加到六个固定方向，也仍可能发生方向记忆。一个 \(3N-d\) 维振动空间的
Hessian 有 \(O(N^2)\) 自由度；随机 probe 的优势来自跨 epoch 持续重采样，
而不是一次性生成少数固定向量。

### 5.4 密度优化噪声被有限差分放大

若单个端点力误差为 \(\delta F\)，中心差分 Hessian 噪声量级约为
\(\delta F/\epsilon\)。step-1300 有 46/90 点未达到 \(10^{-8}\) 密度梯度门槛，
最大达到 \(2.14\times10^{-7}\)，同时使用 \(\epsilon=10^{-4}\) Bohr，误差
放大会非常明显。

应在同一 checkpoint、同一分子上同时测试
\(10^{-3},3\times10^{-4},10^{-4}\) Bohr，并比较：

- 固定密度 Hessian；
- 隐式响应 Hessian；
- 严格重优化有限差分 Hessian；
- 每个步长的密度残差与 Hessian 误差。

### 5.5 频率损失本身的病态性

频率满足 \(\omega_i^2=\lambda_i\)。接近零的低频模中，小的 Hessian 特征值
误差会产生很大的相对频率误差；简并或近简并模的单个特征向量也不唯一。因此：

- 不应一开始直接最小化按顺序匹配的频率；
- 先学习投影 Hessian/HVP；
- 对低频/简并部分使用子空间 projector loss，而非单个 eigenvector loss；
- 先做质量加权和 Eckart 投影，去掉 5/6 个刚体模；
- 分开报告 `<400 cm^-1` 和 `>=400 cm^-1`。

## 6. 推荐的新主线架构

### 6.1 保留不变的部分

- 输入仍为核坐标 \(R\) 与密度系数 \(c\)；
- 保留完整 GraphFormer 主体和标量总能量读出；
- 保留 E 和 Structures25 定义的 G；
- 最终力始终为
  \[
  F=-\partial E_{\rm total}/\partial R;
  \]
- 最终 Hessian/HVP 始终来自同一标量能量的密度弛豫导数。

### 6.2 新增密度响应头

新增一个条件响应模块

\[
\Delta c_{\phi}(R,c,v),
\]

输入 GraphFormer 的密度分支中间特征、核方向 \(v\) 和几何边特征，输出满足电子
数守恒的预测密度响应。它有三个用途：

1. 预测并监督参考响应
   \[
   \Delta c_{\rm ref}(v)\approx
   \frac{c_{\rm KS}(R+\epsilon v)-c_{\rm KS}(R-\epsilon v)}
        {2\epsilon};
   \]
2. 最小化响应方程残差
   \[
   L_{\rm response\ residual}
   =\|E_{cc}\Delta c_\phi+E_{cR}v\|^2;
   \]
3. 在严格推理时作为 \(E_{cc}x=E_{cR}v\) 的初值或预条件器。

重要边界：响应头不能直接决定最终 Hessian。严格 HVP 仍需把响应方程求到固定
残差，以保持标量能量的一致性；响应头的价值是降低迭代次数和提供可训练的密度
二阶信息。

### 6.3 可选的结构化 Hessian 头

若需要快速频率筛选，可再加 HIP 风格的 \(l\le2\) 对称 Hessian 头，但只作为：

- 快速近似/候选筛选；
- 严格隐式求解的初始 Hessian；
- 与保守 HVP 做蒸馏的一致性辅助项；
- 低特征值子空间的额外监督。

不应用它替代物理标量能量 Hessian，也不应从目前已失败的 final invariant
atom state 独立读出。应读取 G3D 中间边消息、显式方向特征和密度分支在求和前
的表示。

## 7. 建议的损失定义

在电子数守恒、去刚体运动的内部空间 \(Q\) 中采样 Rademacher 向量 \(z\)，令
\(v=Qz/\|Qz\|\)。建议总损失为：

\[
\begin{aligned}
L={}&
\lambda_E L_E+\lambda_G L_G+\lambda_F L_F\\
&+\lambda_{\rm HVP}
\frac{\|Q(H_{\theta,\rm relaxed}-H_{\rm ref})v\|_2^2}
     {3N-d}\\
&+\lambda_c L_{\Delta c}
+\lambda_r L_{\rm response\ residual}
+\lambda_{\rm traj} L_{\rm GDI/trajectory}\\
&+\lambda_{\rm replay}L_{\rm Structures25\ replay}.
\end{aligned}
\]

建议细节：

- 每个 molecule/batch 只用 1–2 个 probe，但每个 epoch 重新采样；
- 同时保留少量固定 validation probes，绝不用于训练；
- 使用质量加权 probe 的附加分支，使损失更贴近频率；
- HVP 权重从 0 逐步升高，不在第一步就与 E/G/F 同权；
- 通过 EMA 梯度范数使加权后的 E/G/F/HVP 参数梯度处在同一数量级；
- 不对 G/H 使用当前会饱和的 soft cap 作为唯一标度，至少同时记录未截断误差；
- 对 \(E_{cc}\) 在密度切空间的负 Rayleigh quotient 加软惩罚，但先排除真实的
  约束零模和数值规范自由度；
- 对近简并低频模，可在后期加入 HIP 式 spectral subspace loss。

## 8. 数据生成与采样方案

### 8.1 标签优先级

每个父分子的最低可用单元：

- 中心 \(R_0,c_0,E,G,F\)；
- 1–2 个随机内部空间方向 \(v\)；
- 每个方向的 \(R_\pm,c_\pm,F_\pm\)；
- \(\Delta c_v\) 与 \(H_{\rm ref}v\)；
- SCF/密度收敛残差、步长和数值精度元数据。

单个 HVP 标签只需要两个端点力，而不是 \(6N\) 个完整 Hessian 端点。已有完整
PBE Hessian 的分子可以离线乘任意新 probe，不必重复量化计算。

### 8.2 构型分布

不要只采平衡点附近的单一小位移。训练池应混合：

- 平衡点；
- 正常模采样；
- 随机内部坐标位移；
- 适量压缩键/高曲率构型；
- 当前模型密度优化慢或失败的构型；
- 当前 HVP ensemble 分歧大的构型。

### 8.3 理论水平混合

- PBE/项目目标水平的 E/G/F/HVP 是最终监督；
- Hessian QM9、HORM、xTB Hessian 只能用于 backbone 或辅助头预训练；
- 微调时必须保留理论水平标识，不能直接把不同 Hessian 当成同一标签；
- 最终模型选择只看 PBE molecule-disjoint validation。

## 9. 分阶段实验建议

### P0：先证明响应问题，暂不训练

在 `0016298` 和另外 2–4 个已有 PBE Hessian 的 train-only 分子上，对文章原始
checkpoint、step-100、step-1300 做：

1. 计算 \(H_{\rm fixed}=E_{RR}\)；
2. 计算响应修正
   \(-E_{Rc}E_{cc}^{-1}E_{cR}\)；
3. 报告 \(E_{cc}\) 投影谱、条件数、负特征值数和线性求解残差；
4. 比较隐式 relaxed HVP 与三种步长的严格力差分；
5. 比较三个 checkpoint 的密度优化迭代数。

判据：若 step-1300 的 \(E_{cc}\) 条件数、负曲率或响应项明显恶化，而文章
checkpoint 较稳定，就可以确认主瓶颈是密度曲率漂移，而非几何 backbone 容量。

### P1：十至五十分子的响应头/HVP 单元实验

- 从文章 checkpoint 初始化，不再从 scratch 开始主线实验；
- 80% parent train、20% parent held-out；
- 每个训练 parent 生成 6–8 个随机内部 probe，但每步只采 1 个；
- 至少保留每个 held parent 2 个从未训练的方向；
- 比较三个 arm：
  1. E/G/F replay；
  2. E/G/F + 当前端点割线；
  3. E/G/F + \(\Delta c\)/响应残差 + relaxed HVP；
- 用相同训练步数和相同计算预算；
- 每 50–100 步只在 1–2 个冻结 held-out 分子上做完整 Hessian。

只有 arm 3 同时改善 held-out HVP、密度收敛和完整 Hessian，才扩大规模。

### P2：100–500 parent 的随机投影训练

- molecule-disjoint train/validation；
- 随机 probe 跨 epoch 重采样；
- curriculum 提升 HVP 权重；
- replay Structures25 的广泛 \(R,c\) 扰动数据；
- 主动选择高曲率、失败密度点和不确定性高的样本；
- 以 held-out HVP + 密度收敛为 checkpoint 选择指标，不能用训练总 loss。

### P3：完整评估

在冻结的 20–100 个 molecule-disjoint 分子上报告：

- Hessian relative Frobenius、element MAE/RMSE；
- 随机 HVP RMSE 和曲率斜率；
- 质量加权特征值误差；
- 频率 MAE/RMSE，分 `<400` 与 `>=400 cm^-1`；
- 虚频数量；
- 模态 overlap 与简并子空间 overlap；
- 平移/旋转零模、对称性；
- 密度严格收敛比例、平均迭代数、wall time；
- 三种位移步长的一致性。

如果要报告 IR 强度，再加入偶极矩、偶极导数和光谱强度误差。

## 10. 明确不建议继续的路线

| 路线 | 决策 | 原因 |
|---|---|---|
| 十分子、同一 loss、继续加训练步数 | 停止 | step-1300 已显示训练收敛但 Hessian 几乎不改善，密度优化反而恶化 |
| scratch 作为主线 | 停止 | 文章 checkpoint 相比 scratch-100 将 relative Frobenius 从约 24 降到约 1.26 |
| 每分子固定一个方向长期训练 | 停止 | 容易方向记忆，无法估计 Frobenius；PHL 支持跨 batch 重采样 |
| 只增加固定方向数量 | 不足 | 当前多方向实验已出现跨方向干扰；需随机化、held-out 方向和响应一致性 |
| 独立 force 头作为最终力 | 停止 | 现有 force head 比标量能量导数差约 2.29 倍，且失去全局保守性 |
| 独立 Hessian 头作为最终 Hessian | 不建议 | 可快但不保证与能量/力可积一致 |
| 完整 Hessian 每步训练 | 不建议 | 成本过高；HVP/随机行已有更强证据 |
| 只优化频率 loss | 不建议 | 低频和简并模病态，先稳定矩阵/HVP 与密度响应 |
| 纯 GDI surrogate energy 作为物理能量 | 禁止 | 它只保证优化到正确密度，不保证能量和核导数物理正确 |

## 11. 最小可执行决定

下一次真正值得运行的实验应当满足五个条件：

1. 文章 checkpoint warm-start；
2. train/held-out 按父分子隔离，并有 held-out directions；
3. 随机内部空间 HVP probe，而非固定坐标列；
4. 显式密度响应监督或响应方程残差；
5. checkpoint 选择同时看 held-out HVP 与密度收敛，而不是训练总 loss。

如果只能先做一件事，应先完成 P0 的 Schur 补分解和
\(E_{cc}\) 条件数审计。它最便宜，而且能直接回答：当前 Hessian 差究竟主要来自
固定密度几何曲率、密度响应项，还是密度优化数值噪声。

## 12. 与仓库现有证据的对应

- 当前 EGFH10、step-1300 与频率结果：
  `docs/source/qm9_egfh10_scratch_pilot_20260730.md`
- 当前力割线损失实现：
  `mldft/ml/models/components/loss_function.py`
- 已完成的解析 relaxed-HVP 正确性审计：
  `docs/qm9_graphformer_complete_total_analytic_relaxed_hvp_refactor.md`
- 结构化 Hessian/密度头试验：
  `docs/qm9_structured_density_hessian_head_pilot_v1.md`
- 冻结 GraphFormer force/Hessian attention 头试验：
  `docs/qm9_graphformer_frozen_force_attention_pilot_v1.md`、
  `docs/qm9_graphformer_frozen_hessian_attention_head_pilot_v1.md`
- 项目总交接：
  `docs/qm9_force_hessian_project_handoff.md`
