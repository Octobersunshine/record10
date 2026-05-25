# 电化学阻抗谱（EIS）数据分析工具

**完整的EIS分析工具包，包含等效电路拟合 + 分布弛豫时间（DRT）分析**

## ✨ 功能亮点

| 功能 | 说明 |
|------|------|
| **加权CNLS拟合** | 支持Modulus权重、Variance权重的复数非线性最小二乘 |
| **DRT分析** | 无需预设等效电路，直接提取弛豫时间分布 |
| **多模型支持** | Randles电路、带CPE的Randles电路 |
| **自动参数估计** | 智能初始化拟合参数 |
| **交叉验证** | 自动优化DRT正则化参数 |

---

## 📁 文件说明

| 文件名 | 说明 |
|--------|------|
| `eis_fitting_advanced.py` | **主程序** - 包含所有功能（推荐使用） |
| `eis_fitting.py` | 基础版本，仅包含等效电路拟合 |
| `fit_my_data.py` | 使用示例数据进行拟合的脚本 |
| `sample_eis_data.csv` | 示例EIS数据文件 |

---

## 🎯 两种分析方法对比

### 方法1：等效电路拟合（CNLS）

**需要预设电路模型**，如Randles电路：

```
    Rs
----/\/\/\----+----+
              |    |
             Rct  Cdl
              |    |
--------------+----+
```

**优点：**
- 参数物理意义明确
- 拟合结果可直接用于定量分析

**缺点：**
- 需要预先知道电路结构
- 复杂系统难以选择正确模型

---

### 方法2：分布弛豫时间（DRT）分析 ⭐ NEW

**无需预设等效电路**，直接从阻抗谱提取弛豫时间分布：

**原理：**
阻抗可以表示为无穷多个弛豫过程的叠加：
```
Z(ω) = Rs + ∫ [γ(lnτ) / (1 + jωτ)] d(lnτ)
```

其中γ(lnτ)就是**分布弛豫时间（DRT）**，每个峰代表一个电化学过程。

**优点：**
- ✅ 无需预设等效电路
- ✅ 自动识别多步动力学过程
- ✅ 直观展示各个弛豫过程的时间常数
- ✅ 帮助选择合适的等效电路模型

**缺点：**
- 正则化参数需要谨慎选择
- 难以区分非常接近的弛豫过程

---

## ⚖️ 加权方法（CNLS拟合）

| 方法 | 权重公式 | 说明 | 推荐程度 |
|------|---------|------|---------|
| **None** | wᵢ = 1 | 等权重 | ❌ 不推荐 |
| **Modulus** | wᵢ = 1/\|Zᵢ\|² | 比例权重 | ✅ **默认推荐** |
| **Variance** | wᵢ = 1/σᵢ² | 基于测量方差 | ⭐ 已知噪声时 |

---

## 🔧 安装依赖

```bash
pip install numpy scipy matplotlib
```

---

## 🚀 快速开始

### 运行完整演示

```bash
python eis_fitting_advanced.py
```

这将演示：
1. 三种加权方法的CNLS拟合对比
2. 单弛豫过程的DRT分析（Randles电路）
3. 双弛豫过程的DRT分析
4. 交叉验证优化正则化参数

---

## 📖 使用指南

### 1. 等效电路拟合

```python
from eis_fitting_advanced import fit_eis, load_eis_from_csv

# 加载数据
f, Z = load_eis_from_csv('your_data.csv')

# 使用Modulus权重拟合Randles电路（推荐）
result = fit_eis(f, Z, model='randles', weighting='modulus')

# 获取拟合参数
Rs, Rct, Cdl = result.x
print(f"Rs = {Rs:.2f} Ω")
print(f"Rct = {Rct:.2f} Ω")
print(f"Cdl = {Cdl:.2e} F")
```

**可用模型：**
- `'randles'`: Rs + (Rct || Cdl)
- `'randles_cpe'`: Rs + (Rct || CPE)

---

### 2. DRT分析 ⭐ NEW

```python
from eis_fitting_advanced import compute_drt, find_drt_peaks, plot_drt

# 计算DRT
drt_result = compute_drt(f, Z, lambda_reg=1e-3)

# 结果包含：
# drt_result['tau']: 弛豫时间轴
# drt_result['gamma']: DRT分布γ(lnτ)
# drt_result['Rs']: 估计的溶液电阻
# drt_result['R_total']: 总极化电阻（DRT曲线下面积）

# 自动识别弛豫过程（峰检测）
peaks = find_drt_peaks(drt_result['tau'], drt_result['gamma'])
for i, peak in enumerate(peaks):
    print(f"过程 {i+1}:")
    print(f"  弛豫时间 τ = {peak['tau']:.2e} s")
    print(f"  特征频率 f = {peak['f_peak']:.1f} Hz")
    print(f"  峰面积（电阻） = {peak['area']:.2f} Ω")

# 可视化
plot_drt(drt_result, title='你的数据DRT分析')
```

---

### 3. 自动优化DRT正则化参数

```python
from eis_fitting_advanced import optimize_lambda_cv, compute_drt

# 使用交叉验证找到最优λ
best_lambda, lambda_values, cv_errors = optimize_lambda_cv(
    f, Z, 
    lambda_values=np.logspace(-6, 0, 20),  # 搜索范围
    cv_folds=5  # 5折交叉验证
)

print(f"最优正则化参数 λ = {best_lambda:.2e}")

# 使用最优λ计算DRT
drt_result = compute_drt(f, Z, lambda_reg=best_lambda)
```

**正则化参数选择指南：**
- λ太小：过拟合，DRT曲线出现虚假峰
- λ太大：过平滑，真实峰被合并
- 推荐使用交叉验证自动选择

---

### 4. 生成多弛豫测试数据

```python
from eis_fitting_advanced import generate_multi_relaxation_data

f = np.logspace(5, -1, 50)

# 生成包含2个弛豫过程的阻抗谱
Z_noisy, Z_true = generate_multi_relaxation_data(
    f,
    Rs=10,              # 溶液电阻
    R_list=[80, 120],   # 两个过程的电阻
    tau_list=[1e-3, 1e-1],  # 两个过程的弛豫时间
    noise_level=0.02
)
```

---

## 📊 输出文件

运行 `python eis_fitting_advanced.py` 后会生成：

### CNLS拟合相关
- `weighting_comparison.png` - 不同加权方法对比
- `nyquist_plot_weighted.png` - Nyquist图
- `bode_plot_weighted.png` - Bode图
- `residuals_plot_weighted.png` - 残差图

### DRT分析相关 ⭐ NEW
- `drt_single_relaxation.png` - 单弛豫DRT分析（Nyquist图 + DRT谱）
- `drt_multi_relaxation.png` - 双弛豫DRT分析
- `drt_cv_optimization.png` - 交叉验证曲线

---

## 🧮 DRT数学原理

### 基本公式

阻抗的频域表达式可以表示为：

```
Z(ω) = Rs + ∫₀^∞ [g(τ) / (1 + jωτ)] dτ
```

其中 τ = 1/(2πf) 是弛豫时间，g(τ) 是弛豫时间分布。

变量替换后使用 γ(lnτ) = τg(τ)：

```
Z(ω) = Rs + ∫_{-∞}^{+∞} [γ(lnτ) / (1 + jωτ)] d(lnτ)
```

### 离散化

将积分离散化为线性方程组：
```
Z_pol = A · γ
```

其中 A 是离散核矩阵，γ 是待求的DRT向量。

### Tikhonov正则化

由于问题是病态的，使用正则化求解：

```
min ||Aγ - b||² + λ||Lγ||²
```

其中：
- λ 是正则化参数
- L 是二阶差分矩阵（保证解的光滑性）

解析解：
```
γ = (AᵀA + λLᵀL)⁻¹Aᵀb
```

---

## 📝 实际应用建议

### 何时使用DRT？

1. **未知系统**：不确定使用哪种等效电路时
2. **复杂系统**：涉及多步电化学反应时
3. **过程识别**：需要识别有几个动力学过程时
4. **模型验证**：验证等效电路假设是否合理时

### DRT结果解读

- **峰的数量** = 弛豫过程的数量
- **峰位置（τ）** = 各过程的时间常数
- **峰面积** = 各过程的电阻贡献
- **峰宽度** = 弛豫时间的分布宽度

### 注意事项

1. **频率范围**：确保频率范围覆盖所有感兴趣的弛豫过程
2. **数据质量**：DRT对噪声敏感，高质量数据很重要
3. **正则化**：始终检查正则化参数的影响
4. **峰重叠**：时间常数接近的过程难以完全分离

---

## 🔍 故障排除

### DRT出现虚假峰
- 增大正则化参数 λ
- 检查数据是否有噪声或异常点

### DRT峰太宽或合并
- 减小正则化参数 λ
- 确保频率范围足够宽

### 拟合不收敛
- 尝试手动提供初始参数
- 检查数据格式是否正确
- 考虑使用CPE模型代替理想电容

---

## 📚 参考资料

1. Macdonald, J. R., & Johnson, W. B. (1990). Fundamentals of impedance spectroscopy.
2. Boukamp, B. A. (1989). A nonlinear least squares fit procedure for analysis of immittance data.
3. Orazem, M. E., & Tribollet, B. (2008). Electrochemical impedance spectroscopy.
4. Wan, T. H., Saccoccio, M., Chen, C., & Ciucci, F. (2015). Optimal regularization in distribution of relaxation times.
5. Kulikovsky, A. A. (2019). Distribution of relaxation times in impedance spectroscopy.

---

## 📧 联系与反馈

如有问题或建议，欢迎反馈！

---

**版本信息：** v2.0  
**更新内容：** 新增DRT分布弛豫时间分析功能  
**更新日期：** 2026-05-25
