# 安德森局域化模拟 (1D/2D/3D)

## 项目简介

本项目用Python实现一维、二维、三维安德森局域化的数值模拟，包括：
- 传输矩阵法求解本征态（Gram-Schmidt稳定算法）
- 有限尺寸标度分析
- 迁移率边缘和金属-绝缘体转变计算

## 物理背景

安德森局域化（Anderson Localization）是指在无序系统中，由于多重散射干涉效应，电子波函数从扩展态变为局域态的现象。

**维度与局域化的关系：**
- **1D**: 任意小的无序 → 所有态局域化
- **2D**: 任意小的无序 → 所有态局域化（弱局域化）
- **3D**: 存在迁移率边缘，W < W_c 为扩展态，W > W_c 为局域态

## 主要功能

### 一维系统 (anderson_localization.py)
1. **稳定传输矩阵法**：使用Gram-Schmidt正交化防止数值溢出，支持长链系统（10000+格点）
2. **直接传输矩阵法**：传统递推方法，用于对比和短链计算
3. **本征态求解**：使用稀疏矩阵对角化计算哈密顿量的本征值和本征态
4. **逆参与比(IPR)**：定量描述波函数的局域化程度
5. **局域化长度计算**：分析局域化长度随无序强度的变化关系

### 二维/三维系统 (anderson_2d3d.py)
1. **二维/三维哈密顿量构造**：稀疏矩阵表示，支持周期性边界条件
2. **有限尺寸标度分析**：IPR、IPR指数、能级间距统计的尺寸标度
3. **迁移率边缘估计**：利用不同尺寸IPR曲线的相交点估计临界无序强度W_c
4. **波函数可视化**：二维波函数空间分布和概率密度
5. **金属-绝缘体转变分析**：能级间距分布从GOE（金属）到泊松（绝缘）的转变

## 依赖安装

```bash
pip install -r requirements.txt
```

或手动安装：
```bash
pip install numpy scipy matplotlib tqdm
```

## 运行方式

### 一维模拟
```bash
python anderson_localization.py
```

### 二维/三维模拟
```bash
python anderson_2d3d.py
```

## 输出文件

### 一维模拟输出
- `wavefunctions.png`：不同无序强度下的波函数分布图
- `localization_length.png`：局域化长度随无序强度的变化曲线
- `ipr.png`：逆参与比随无序强度的变化曲线

### 二维/三维模拟输出
- `anderson_2d_fss.png`：二维系统有限尺寸标度分析图
- `anderson_3d_fss.png`：三维系统有限尺寸标度分析图
- `mobility_edge.png`：三维迁移率边缘估计图
- `wavefunction_2d_W2.png`：W=2时二维波函数分布图
- `wavefunction_2d_W8.png`：W=8时二维波函数分布图

## 核心函数说明

### 一维系统函数

#### `anderson_hamiltonian(N, W, t=1.0)`
构造一维安德森模型的三对角哈密顿量
- `N`: 格点数
- `W`: 无序强度（势能随机分布范围 [-W/2, W/2]）
- `t`: 最近邻跃迁能

#### `transfer_matrix_method(N, W, E, t=1.0, method='gram_schmidt')`
传输矩阵法统一接口
- `method='gram_schmidt'`: 稳定的Gram-Schmidt正交化方法（推荐）
- `method='direct'`: 直接递推方法（易溢出，仅用于对比）
- 返回：波函数、局域化长度ξ、李雅普诺夫指数λ

#### `transfer_matrix_gram_schmidt(N, W, E, t=1.0, ortho_interval=5)`
**Gram-Schmidt稳定传输矩阵法**（核心改进）

算法原理：
1. 用两个线性无关的初始向量构造2维基
2. 每步应用2×2传输矩阵：M = [[(E-ε)/t, -1], [1, 0]]
3. 定期（每ortho_interval格点）进行Gram-Schmidt正交化和归一化
4. 累积所有归一化因子计算李雅普诺夫指数

优点：
- 完全避免数值溢出，支持N>10000的长链计算
- 李雅普诺夫指数收敛更稳定
- 正交化间隔可调（默认为5，兼顾精度与效率）

### 二维/三维系统函数

#### `anderson_hamiltonian_2d(L, W, t=1.0, periodic=False)`
构造二维安德森模型哈密顿量
- `L`: 每维格点数 (总格点数 N = L×L)
- 支持周期性边界条件

#### `anderson_hamiltonian_3d(L, W, t=1.0, periodic=False)`
构造三维安德森模型哈密顿量
- `L`: 每维格点数 (总格点数 N = L×L×L)
- 支持周期性边界条件

#### `ipr_exponent(psi, dim)`
计算IPR指数 α = -log(IPR)/log(N)
- α ≈ 1: 扩展态（金属相）
- α ≈ 0: 局域态（绝缘相）

#### `compute_level_spacing(eigenvalues)`
计算能级间距比 <r>
- <r> ≈ 0.5307: GOE分布（金属相，量子混沌）
- <r> ≈ 0.3863: 泊松分布（绝缘相，可积系统）

#### `finite_size_scaling_2d/3d(L_values, W_values, ...)`
有限尺寸标度分析
- 计算不同尺寸下IPR、α、<r>随无序强度的变化

#### `estimate_mobility_edge_3d(L_values, W_values, results)`
估计三维迁移率边缘 W_c
- 利用不同尺寸IPR曲线的相交点确定临界点

### 通用函数

#### `inverse_participation_ratio(psi)`
计算逆参与比 IPR = Σ|ψ|⁴ / (Σ|ψ|²)²
- IPR ~ 1/N 表示扩展态
- IPR ~ 1 表示强局域态

## 数值稳定性对比（1D）

| 格点数 N | 直接法状态 | Gram-Schmidt法状态 |
|---------|-----------|-------------------|
| 100     | 稳定      | 稳定              |
| 500     | 稳定      | 稳定              |
| 1000    | 可能溢出  | 稳定              |
| 5000    | 溢出      | 稳定              |
| 10000   | 溢出      | 稳定              |

## 物理结论

| 维度 | 局域化行为 | 临界现象 |
|------|-----------|---------|
| 1D   | 任意无序→全部局域化 | 无金属-绝缘体转变 |
| 2D   | 任意无序→全部局域化 | 弱局域化，无真正转变 |
| 3D   | W < W_c: 扩展态<br>W > W_c: 局域态 | 金属-绝缘体转变，W_c ≈ 16-18（能带中心） |

## 理论公式

- 波函数包络：|ψ(n)| ~ exp(-|n-n₀|/ξ)
- 李雅普诺夫指数：λ = 1/ξ
- 一维标度理论：ξ(W) ~ W⁻² (强无序区)
- 三维标度理论：ξ(W) ~ |W - W_c|^(-ν)，临界指数 ν ≈ 1.5

## 参考资料

1. Anderson, P. W. (1958). Absence of diffusion in certain random lattices.
2. Abrahams, E., et al. (1979). Scaling theory of localization.
3. Lee, P. A., & Ramakrishnan, T. V. (1985). Disordered electronic systems.
4. Evers, F., & Mirlin, A. D. (2008). Anderson transitions.
