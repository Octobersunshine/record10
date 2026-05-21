# SIMP 拓扑优化算法

基于 Python 实现的 SIMP (Solid Isotropic Material with Penalization) 拓扑优化算法，用于最小化柔顺度，满足体积约束。

## 算法简介

SIMP 方法是拓扑优化中最经典的方法之一，通过引入惩罚因子，使得中间密度（0 < x < 1）的单元在优化过程中被推向 0 或 1，从而得到清晰的黑白拓扑结构。

### 核心公式

**材料插值模型（SIMP）：**
```
E(x) = E_min + x^p * (E_0 - E_min)
```
其中：
- x 是单元密度（0 ≤ x ≤ 1）
- p 是惩罚因子（通常取 3）
- E_0 是实体材料的弹性模量
- E_min 是空隙材料的弹性模量（很小值，避免刚度矩阵奇异）

**目标函数（柔顺度）：**
```
C(x) = U^T K(x) U
```
其中 U 是位移向量，K 是刚度矩阵。

**约束：**
```
V(x) / V_0 = volfrac
```
即最终体积占总体积的比例为 volfrac。

## 文件说明

- `simp_example.py` - 主程序，包含完整的 SIMP 算法实现和悬臂梁示例
- `simp_topology_optimization.py` - 面向对象版本的实现
- `requirements.txt` - Python 依赖包列表

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行示例

```bash
python simp_example.py
```

或

```bash
python simp_topology_optimization.py
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `nelx` | x 方向单元数量 | 60 |
| `nely` | y 方向单元数量 | 30 |
| `volfrac` | 体积约束分数 | 0.5 |
| `penal` | SIMP 惩罚因子 | 3.0 |
| `rmin` | 密度滤波器半径 | 1.5 |
| `max_iter` | 最大迭代次数 | 100 |

## 算法流程

1. **初始化**：所有单元密度初始化为体积约束值
2. **有限元分析**：构建刚度矩阵，求解位移
3. **灵敏度分析**：计算目标函数对密度的导数
4. **灵敏度滤波**：使用密度滤波器避免棋盘格现象
5. **OC 优化**：使用最优准则法更新密度
6. **收敛检查**：检查密度变化是否小于阈值
7. **可视化**：实时显示拓扑结构和收敛历史

## 边界条件和载荷

示例中使用的是悬臂梁问题：
- 左端（x=0）完全固定
- 右端（x=nelx）顶部（y=0）施加向下的单位载荷

## 输出

程序会实时显示：
- 当前迭代次数
- 柔顺度值
- 当前体积分数
- 密度变化量

同时显示两个图形：
1. 拓扑结构（黑色为实体，白色为空隙）
2. 柔顺度收敛历史曲线

## 自定义使用

```python
from simp_example import simp_topology_optimization

# 自定义参数
x_final, c_history = simp_topology_optimization(
    nelx=80,
    nely=40,
    volfrac=0.4,
    penal=3.0,
    rmin=2.0,
    max_iter=150
)
```

## 注意事项

1. 滤波器半径 `rmin` 越大，结果越平滑，但可能丢失细节
2. 惩罚因子 `penal` 一般取 3，过小会导致中间密度过多
3. 单元数量越多，计算越慢，但结果越精细
4. 如果出现棋盘格现象，增大 `rmin` 值
