# Delaunay三角剖分网格生成器

本项目实现了用于有限元分析的Delaunay三角剖分算法，可以给定二维区域边界（多边形），生成非结构化三角网格。

## 文件说明

### 1. `delaunay_mesh.py` - 基于SciPy的高效实现
**推荐使用版本**，依赖numpy、scipy、matplotlib库。

**特性：**
- 使用scipy.spatial.Delaunay（Qhull库）实现，性能优异
- 支持多种预设边界形状（正方形、矩形、圆形、三角形、六边形）
- 自动生成内部点
- 多边形内外判断（射线法）
- 单元过滤（只保留边界内部的三角形）
- 边界边提取
- 网格信息输出到文件
- matplotlib可视化

### 2. `delaunay_pure_python.py` - 纯Python实现
**学习算法原理使用**，不依赖任何第三方库。

**特性：**
- 完整实现Bowyer-Watson算法
- 包含Point和Triangle类
- 外接圆计算和空圆性质判断
- 适合学习Delaunay三角剖分原理

## 安装依赖

### 对于 `delaunay_mesh.py`：
```bash
pip install numpy scipy matplotlib
```

### 对于 `delaunay_pure_python.py`：
无需安装额外依赖，使用Python标准库即可。

## 使用方法

### 方法一：使用预设边界

```python
from delaunay_mesh import DelaunayMesh, generate_polygon_boundary

# 生成正方形边界
boundary = generate_polygon_boundary('square', size=1.0)

# 创建网格生成器，设置内部点数量
mesh = DelaunayMesh(boundary, num_internal_points=50)

# 生成网格
nodes, elements = mesh.generate_mesh()

# 过滤掉边界外的单元
mesh.filter_elements()

# 输出到文件
mesh.output_mesh("mesh_output.txt")

# 可视化
mesh.plot_mesh()
```

### 方法二：使用自定义多边形边界

```python
import numpy as np
from delaunay_mesh import DelaunayMesh

# 自定义边界点（按顺时针或逆时针顺序排列）
boundary = np.array([
    [0.0, 0.0],    # 点0
    [2.0, 0.0],    # 点1
    [2.0, 1.0],    # 点2
    [1.0, 1.5],    # 点3
    [0.0, 1.0]     # 点4
])

# 生成网格
mesh = DelaunayMesh(boundary, num_internal_points=30)
nodes, elements = mesh.generate_mesh()
mesh.filter_elements()
```

## 输出格式说明

### 节点信息
```
节点编号         x坐标         y坐标
     0        0.000000        0.000000
     1        1.000000        0.000000
     ...
```

### 单元信息（每个单元由3个节点索引组成）
```
单元编号      节点1      节点2      节点3
     0          0          5          7
     1          1          8          3
     ...
```

## 算法原理

### Delaunay三角剖分特性
1. **空圆性质**：每个三角形的外接圆内不包含其他点
2. **最大化最小角**：避免出现过小的内角
3. **唯一性**：点集不共圆时结果唯一

### Bowyer-Watson算法流程
1. 创建一个包含所有点的超级三角形
2. 逐个插入点：
   - 找出外接圆包含该点的所有三角形（坏三角形）
   - 移除这些三角形，形成多边形空洞
   - 用该点与空洞边界连接，形成新的三角形
3. 移除包含超级三角形顶点的单元

## 有限元应用

生成的网格可直接用于：
- 二维弹性力学分析
- 热传导分析
- 流体动力学（CFD）
- 电磁场分析

### 示例：刚度矩阵组装伪代码
```python
for element in elements:
    # 获取单元节点坐标
    x = nodes[element, 0]
    y = nodes[element, 1]
    
    # 计算单元刚度矩阵
    Ke = compute_element_stiffness(x, y)
    
    # 组装到整体刚度矩阵
    assemble_global_matrix(K, Ke, element)
```

## 支持的边界类型

| 类型 | 说明 | 参数 |
|------|------|------|
| `square` | 正方形 | size |
| `rectangle` | 矩形 | width, height |
| `circle` | 圆形（多边形近似） | radius, num_points |
| `triangle` | 等边三角形 | size |
| `hexagon` | 正六边形 | size |

## 注意事项

1. 边界点必须按顺序（顺时针或逆时针）排列
2. 内部点数量越多，网格越精细
3. 对于复杂边界，建议增加边界点密度
4. 纯Python版本仅用于学习，大规模计算请使用SciPy版本

## 运行示例

在命令行中运行：
```bash
# 运行SciPy版本
python delaunay_mesh.py

# 运行纯Python版本
python delaunay_pure_python.py
```
