# 变系数KdV方程与Boussinesq模型 - 技术说明

## 问题分析

### 标准KdV方程的局限性

标准KdV方程形式为：
```
∂u/∂t + 6u∂u/∂x + ∂³u/∂x³ = 0
```

**问题**：该方程推导时假设**水深恒定**。当应用于变水深地形时：
1. 简单地用h(x)缩放不能保证能量守恒
2. 产生虚假的数值反射
3. 波与地形的相互作用物理上不一致

---

## 改进的数学模型

### 1. 变系数KdV方程 (vKdV) - 守恒形式

#### 物理推导

从浅水波方程出发，考虑变水深h(x)：

**连续性方程**：
```
∂(hη)/∂t + ∂/∂x [ (h + η)u ] = 0
```

**动量方程**：
```
∂u/∂t + u∂u/∂x + g∂η/∂x + (g/3)h²∂³η/∂x³ = 0
```

消去u得到变系数KdV方程的**守恒形式**：

```
∂η/∂t + (1/h) ∂/∂x [ (1/2)g h² η + c η² - (g/3) h³ ∂²η/∂x² ] = 0
```

其中：
- c = √(g h) 为线性波速
- η为波面位移
- 守恒形式确保数值计算时的能量守恒性质

#### 离散形式

使用**通量差分方法**：
```
η^(n+1) = η^n - Δt/h * (F_{i+1/2} - F_{i-1/2})
```

其中数值通量 F 包含：
1. 非线性对流项：(1/2) c η²
2. 压力项：(1/2) g h² η
3. 色散项：-(g/3) h³ ∂²η/∂x²

---

### 2. Boussinesq类模型

#### 控制方程

对于变水深情形，采用修正的Boussinesq方程：

```
∂η/∂t + ∂(h u)/∂x = 0

∂u/∂t + u∂u/∂x + g∂η/∂x - (h/2) ∂/∂x [ ∂²(h u)/∂x∂t ] + (h²/6) ∂³u/∂x³ = 0
```

#### 简化形式（我们的实现）

为提高计算效率，采用弱非线性近似：

```
∂η/∂t = -c(x) [ ∂η/∂x + (η/h) ∂η/∂x + (h_x/h) η ] 
         + (c/6) h² ∂³η/∂x³ + (c/2) h h_x ∂²η/∂x²
```

其中：
- c(x) = √(g h(x)) 为局部波速
- h_x = dh/dx 为地形斜率

---

### 3. 非线性浅水波方程 (SWE)

作为参考模型，实现无色散的浅水波方程：

```
∂η/∂t + (1/h) ∂/∂x [ c h η (1 + η/(2h)) ] = 0
```

---

## 数值方法改进

### 1. 去混叠滤波 (De-aliasing)

**问题**：伪谱方法中，非线性项会产生高频波数分量，混叠到低频。

**解决方案**：2/3 规则
```python
dealias_filter = np.ones(N)
dealias_filter[np.abs(k) > 2/3 * k_max] = 0
```

将波数空间中超过2/3最大波数的分量置零。

### 2. 守恒量诊断

实时计算并监控：
- **质量守恒**：M = ∫ η dx
- **能量守恒**：E = (1/2) ∫ h η² dx

用于验证数值方法的正确性。

### 3. 空间导数计算

使用傅里叶谱方法：
```
∂u/∂x ↔ ℱ⁻¹ [ ik û ]
∂³u/∂x³ ↔ ℱ⁻¹ [ -i k³ û ]
```

---

## 模型对比分析

| 特性 | 标准KdV | 变系数KdV | Boussinesq | 浅水波 |
|------|---------|-----------|------------|--------|
| **能量守恒** | ✗ 差 | ✓ 好 | ✓ 很好 | ✓ 最好 |
| **色散效应** | ✓ | ✓ | ✓ | ✗ |
| **变水深** | ✗ 虚假反射 | ✓ 物理一致 | ✓ 更准确 | ✓ |
| **计算效率** | ✓ 高 | ✓ 高 | ☐ 中 | ✓ 很高 |
| **适用范围** | 平底 | 缓变地形 | 一般地形 | 长波 |

---

## 理论分析：波-地形相互作用

### 绝热不变量

对于缓变地形，存在绝热不变量：
```
I = ∫ (η² / c) dx ≈ 常数
```

这意味着：
- 水深减小时（h↓ → c↓），振幅必须增大（η↑）
- 能量不完全守恒，但绝热不变量守恒
- 地形变化越缓，绝热近似越好

### 反射系数

对于台阶地形 h₁ → h₂，线性理论反射系数：
```
|r| = |(√h₁ - √h₂) / (√h₁ + √h₂)|
```

我们的数值模型应该接近这个理论值。

---

## 参数选择指南

### 网格分辨率

- **N**：空间网格点数，建议取2的幂次（256, 512, 1024...）
- **收敛性**：谱方法具有指数收敛性
- **典型值**：N = 512-2048

### 时间步长

根据CFL条件：
```
Δt < CFL * Δx / c_max
```

其中CFL数建议取0.1-0.5。

### 地形变化尺度

为避免显著反射，地形变化宽度应满足：
```
L_terrain > λ_typical
```

其中λ为典型波长。

---

## 数值验证方法

### 1. 守恒性测试

- 平坦地形上，能量误差应 < 0.1%
- 缓变地形上，能量误差应 < 1%

### 2. 收敛性测试

```
error ~ N^(-p)
```

谱方法应显示指数收敛。

### 3. 解析解对比

对于平底孤立波解：
```
η(x,t) = A sech²[ √(3A/(4h³)) (x - √(g(h+A)) t) ]
```

数值解应与解析解吻合。

---

## 扩展方向

### 1. 高阶Boussinesq模型

- Nwogu模型
- 完全非线性Boussinesq方程
- 包含垂向速度变化

### 2. 旋转效应

添加科里奥利力：
```
f = 2Ω sin(latitude)
```

适用于大尺度海洋内波。

### 3. 密度层结

考虑连续密度剖面：
```
N(z) = √[ -g/ρ dρ/dz ] （浮力频率）
```

### 4. 并行计算

- 域分解并行化
- GPU加速（CuPy）
- 自适应时间步长

---

## 参考文献

1. **Johnson, R. S. (1997)**. A Modern Introduction to the Mathematical Theory of Water Waves. Cambridge University Press.

2. **Ostrovsky, L. A., & Stepanyants, Y. A. (1989)**. Nonlinear Waves in Stratified Fluids. American Institute of Physics.

3. **Peregrine, D. H. (1967)**. Long waves on a beach. Journal of Fluid Mechanics, 27(4), 815-827.

4. **Madsen, P. A., Bingham, H. B., & Liu, H. (2003)**. Boussinesq-type formulations for fully nonlinear and extremely dispersive water waves. Proceedings of the Royal Society of London.

5. **Grilli, S. T., et al. (1997)**. Numerical simulation of wave interaction with submerged breakwaters using a fully nonlinear Boussinesq model. Coastal Engineering.

---

## 代码结构

```
kdv_improved.py
├── VariableCoefficientKdV  # 主求解器类
│   ├── rhs_conservation_vkdv  # 守恒形式vKdV右端项
│   ├── rhs_boussinesq         # Boussinesq模型
│   ├── rhs_shallow_water      # 浅水波方程
│   ├── compute_diagnostics    # 质量/能量诊断
│   └── plot_diagnostics       # 诊断结果可视化
└── compare_models              # 模型对比函数
```
