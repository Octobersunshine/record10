# SIMP 拓扑优化 - 棋盘格问题修复指南

## 问题描述

棋盘格现象（Checkerboard Pattern）是拓扑优化中最常见的数值不稳定性问题，表现为：

- 单元密度交替出现 0 和 1
- 形成类似国际象棋棋盘的模式
- 通常没有实际物理意义，是数值伪影
- 导致结果难以制造和解释

## 产生原因

1. **有限元离散化误差** - 低阶单元的数值不稳定性
2. **刚度矩阵奇异性** - 低密度单元导致病态系统
3. **灵敏度数值振荡** - 相邻单元灵敏度符号交替
4. **没有足够的正则化** - 缺少空间平滑约束

## 修复方案

本实现采用 **三层防护** 策略彻底解决棋盘格问题：

---

### 🛡️ 第一层：标准灵敏度滤波（Sigmund 2001）

**原理**：对目标函数灵敏度 `dc` 和体积灵敏度 `dv` 同时进行空间加权平均。

**数学公式**：
```
dc'_i = (Σ w_ij * dc_j) / (Σ w_ij)
dv'_i = (Σ w_ij * dv_j) / (Σ w_ij)
```
其中权重 `w_ij = max(0, rmin - distance(i,j))`

**代码实现**：
```python
dcn = np.zeros((nely, nelx))
dvy = np.zeros((nely, nelx))
for i in range(nelx):
    for j in range(nely):
        sum_ = 0.0
        sum_dc = 0.0
        sum_dv = 0.0
        for k in range(...):
            for l in range(...):
                fac = rmin - np.sqrt((i-k)**2 + (j-l)**2)
                if fac > 0:
                    sum_ += fac
                    sum_dc += fac * dc[l, k]
                    sum_dv += fac * dv[l, k]
        dcn[j, i] = sum_dc / sum_
        dvy[j, i] = sum_dv / sum_
dc, dv = dcn, dvy
```

**关键改进**：**同时滤波 dc 和 dv**（很多实现只滤波 dc）

**参数建议**：
- `rmin = 1.5 ~ 3.0`（单元数的 2-5%）
- 网格越密，rmin 应适当增大

---

### 🛡️ 第二层：Heaviside 投影滤波

**原理**：对密度场进行非线性投影，使边界更锐利，抑制中间密度扩散。

**数学公式**：
```
x_phys = [tanh(β·η) + tanh(β·(x - η))] / [tanh(β·η) + tanh(β·(1-η))]
```

**链式法则修正灵敏度**：
```
dc' = dc * d(x_phys)/dx
```

**代码实现**：
```python
def heaviside_projection(x, beta, eta):
    numerator = np.tanh(beta * eta) + np.tanh(beta * (x - eta))
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return numerator / denominator

# 灵敏度修正
if use_projection:
    dproj = beta * (1 - np.tanh(beta * (x - eta))**2) / denominator
    dc = dc * dproj
```

**参数建议**：
- `beta = 4 ~ 16`（越大越锐利，建议从 8 开始）
- `eta = 0.5`（阈值，通常取 0.5）
- 迭代过程中可逐步增大 beta（继续锐化）

---

### 🛡️ 第三层：灰度惩罚项

**原理**：在目标函数中添加惩罚项，主动抑制中间密度单元。

**惩罚函数**：
```
P(x) = Σ 4·w·x_i·(1 - x_i)
```
这是一个抛物线函数，在 x=0 和 x=1 时取最小值 0，在 x=0.5 时取最大值 w。

**灵敏度修正**：
```
dc' = dc + dP/dx = dc + 4·w·(1 - 2·x)
```

**代码实现**：
```python
if use_gray_penalty:
    dc_gray = 4 * gray_weight * (1 - 2 * x_phys) / (nelx * nely)
    dc = dc + dc_gray
```

**参数建议**：
- `gray_weight = 0.05 ~ 0.2`
- 权重过大可能影响优化收敛性

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `simp_fixed_checkerboard.py` | 完整修复版（带GUI可视化） |
| `simp_fixed_no_gui.py` | 无GUI修复版（带诊断功能） |
| `simp_example.py` | 原始版本（对比用） |

## 使用方法

### 快速运行修复版

```bash
python simp_fixed_checkerboard.py
```

### 无GUI版本（推荐用于测试）

```bash
python simp_fixed_no_gui.py
```

### 自定义参数

```python
from simp_fixed_no_gui import simp_topology_optimization

# 强力去棋盘格配置
x = simp_topology_optimization(
    nelx=60, nely=30,
    volfrac=0.4,
    penal=3.0,
    rmin=2.5,              # 增大滤波半径
    use_projection=True,   # 开启Heaviside投影
    beta=12.0,             # 更强的锐化
    use_gray_penalty=True, # 开启灰度惩罚
    gray_weight=0.15,      # 适中的惩罚权重
    max_iter=100
)
```

## 参数调优指南

| 症状 | 解决方案 |
|------|----------|
| 明显棋盘格 | 增大 `rmin`（如 1.5 → 2.5） |
| 边界太模糊 | 增大 `beta`（如 8 → 16） |
| 中间密度太多 | 增大 `gray_weight`（如 0.1 → 0.2） |
| 收敛不稳定 | 减小 `move`（如 0.2 → 0.1） |
| 优化太慢 | 减小 `rmin` 或 `beta` |

## 诊断功能

代码内置棋盘格诊断函数：

```python
from simp_fixed_no_gui import diagnose_checkerboard

# 诊断密度场 x
checker_ratio = diagnose_checkerboard(x)
```

**判定标准**：
- `< 0.01`：优秀，基本无棋盘格
- `0.01 ~ 0.05`：良好，轻微棋盘格可接受
- `> 0.05`：需调整参数进一步去棋盘格

## 收敛监控

每 5 次迭代输出：
```
Iter  50 | Compliance=123.4567 | Volume=0.4998 | Change=0.0032 | Gray=0.045
```

- **Compliance**：柔顺度（目标函数）
- **Volume**：当前体积分数
- **Change**：密度最大变化量（收敛判据）
- **Gray**：中间密度单元比例（0.05 < x < 0.95）

## 常见问题

**Q: 为什么滤波后结构细节丢失了？**

A: 这是滤波的副作用。可以：
1. 适当减小 rmin
2. 使用更大的 beta 进行后处理锐化
3. 采用多尺度或延续方法

**Q: Heaviside投影导致收敛振荡？**

A:
1. 减小 beta 值
2. 在迭代过程中逐步增大 beta
3. 减小移动限制 move

**Q: 灰度惩罚使柔顺度变差？**

A: 正常现象。灰度惩罚是一种正则化，会牺牲少量性能换取更清晰的黑白设计。可以适当减小 gray_weight。

## 参考文献

1. Sigmund, O. (2001). A 99 line topology optimization code written in Matlab.
2. Guest, J. K., et al. (2004). Achieving minimum length scale in topology optimization.
3. Wang, F., et al. (2011). Heaviside projection based continuum topology optimization.
