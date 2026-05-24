# 白矮星结构方程求解器 (Lane-Emden方程)

## 简介

本项目使用Python数值求解白矮星的结构方程（Lane-Emden方程），计算给定中心密度和化学组成时的质量-半径关系和密度轮廓。

## 物理背景

白矮星是简并电子气支撑的致密天体，其结构可以用多方球模型（polytrope model）描述：

- **非相对论简并**：n = 1.5（适用于低质量白矮星）
- **极端相对论简并**：n = 3.0（适用于大质量白矮星，趋近钱德拉塞卡极限）

Lane-Emden方程形式为：
```
d²θ/dξ² + (2/ξ) dθ/dξ + θ^n = 0
```

## 代码功能

### 1. Lane-Emden方程求解
- 使用SciPy的`solve_ivp`进行数值积分
- 自动检测恒星表面（θ = 0处）
- 支持任意多方指数n

### 2. 白矮星物理性质计算
- 半径计算
- 质量计算
- 密度轮廓
- 钱德拉塞卡极限质量

### 3. 可视化输出
- Lane-Emden方程解 θ(ξ)
- 归一化密度轮廓
- 质量-半径关系图
- 物理坐标下的密度轮廓

## 使用方法

### 环境要求
```bash
pip install numpy scipy matplotlib
```

### 运行程序
```bash
python white_dwarf_structure.py
```

### 自定义参数

```python
# 示例：计算特定中心密度的白矮星性质
from white_dwarf_structure import white_dwarf_properties

n = 1.5  # 多方指数
rho_c = 1e9  # 中心密度 (kg/m³)
mu_e = 2.0   # 电子平均分子量

R, M, r, rho = white_dwarf_properties(n, rho_c, mu_e)
print(f"质量: {M/1.989e30:.3f} M☉")
print(f"半径: {R/6.371e6:.3f} R⊕")
```

## 输出结果

程序运行后将：
1. 在控制台打印示例白矮星性质表
2. 显示包含4个子图的可视化窗口
3. 保存图像为 `white_dwarf_structure.png`

## 关键函数说明

| 函数 | 功能 |
|------|------|
| `solve_lane_emden(n)` | 求解Lane-Emden方程 |
| `white_dwarf_properties(n, rho_c, mu_e)` | 计算白矮星物理性质 |
| `chandrasekhar_mass(mu_e)` | 计算钱德拉塞卡极限质量 |
| `mass_radius_relation(...)` | 计算质量-半径关系 |
| `plot_results()` | 绘制所有结果图 |

## 参考值

- 钱德拉塞卡极限（μ_e=2.0）：约 1.44 M☉
- 典型白矮星中心密度：10⁸ - 10¹² kg/m³
- 天狼星B：质量 ~1.02 M☉，半径 ~0.008 R☉ (~0.88 R⊕)
