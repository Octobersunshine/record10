# 边界元法求解声场散射问题 (BEM for Acoustic Scattering)

本项目使用Python实现了三维边界元法，用于求解声场散射问题。本实现包含了**奇异积分的精确处理**和**快速多极子方法（FMM）**，支持大规模问题求解。

## 主要功能

### 1. 网格数据结构 (`TriangleMesh`)
- 支持三角形网格
- 自动计算面中心、法向量和面积

### 2. 格林函数
- 三维Helmholtz方程格林函数
- 格林函数的法向导数

### 3. 数值积分（含奇异积分处理）
- **标准三角形高斯积分**（3、4、7个积分点）
- **Duffy坐标变换**：处理1/r型奇异性，将三角形映射到单位正方形
- **细分积分法**：将近奇异单元递归细分为更小的子单元
- **自动检测**：基于距离/单元尺寸比自动识别近奇异单元
- 单层位势积分
- 双层位势积分

### 4. 标准边界元法求解 (`bem_acoustic.py`)
- Burton-Miller公式（解决非唯一性问题）
- Neumann边界条件（刚性散射体）
- Dirichlet边界条件（软边界）
- GMRES迭代求解器
- 自动应用奇异积分修正

### 5. 多层快速多极子BEM (`mlfma.py`)
- **八叉树空间分层**：将计算域递归分割
- **平面波展开**：V+（多极展开）和W（局部展开）
- **向上传递 (M2M)**：从叶节点到根节点聚合多极展开
- **多极到局部变换 (M2L)**：非相邻组的相互作用
- **向下传递 (L2L)**：从根节点到叶节点传播局部展开
- **近场/远场分离**：平衡计算精度和效率
- **O(N)计算复杂度**：适合10万+自由度大规模问题

### 6. 入射波
- 平面波
- 平面波法向导数

### 7. 场计算
- 表面声压计算
- 任意点声场外推

## 使用方法

### 基本示例

```python
import numpy as np
from bem_acoustic import (
    create_sphere_mesh,
    plane_wave,
    plane_wave_normal_derivative,
    solve_acoustic_scattering,
    evaluate_field
)

# 参数设置
radius = 1.0
frequency = 100.0
c = 343.0
k = 2 * np.pi * frequency / c

# 创建网格
mesh = create_sphere_mesh(radius=radius, refinement_level=2)
print(f"网格: {mesh.n_vertices} 顶点, {mesh.n_faces} 单元")

# 计算入射波
p_inc = plane_wave(mesh.face_centers, k)
dp_inc_dn = plane_wave_normal_derivative(mesh.face_centers, mesh.face_normals, k)

# 求解表面声压（刚性散射体，Neumann边界条件）
p_total, dp_total_dn = solve_acoustic_scattering(
    mesh, k, p_inc, dp_inc_dn, boundary_condition="neumann"
)

print(f"表面声压幅值范围: [{np.min(np.abs(p_total)):.4f}, {np.max(np.abs(p_total)):.4f}]")

# 计算远场声压
theta = np.linspace(0, np.pi, 37)
r_far = 10.0
eval_points = np.array([
    [r_far * np.sin(t), 0, r_far * np.cos(t)] for t in theta
])

p_scat = evaluate_field(mesh, p_total - p_inc, dp_total_dn - dp_inc_dn, eval_points, k)
p_far_total = p_scat + plane_wave(eval_points, k)
```

## 理论基础

### Helmholtz方程
简谐声场满足Helmholtz方程：
$$\nabla^2 p + k^2 p = 0$$
其中 $k = \omega/c$ 为波数，$\omega = 2\pi f$ 为角频率，$c$ 为声速。

### 边界积分方程
对于外部问题，边界积分方程为：
$$\frac{1}{2} p(x) = \int_S \left[ p(y) \frac{\partial G(x,y)}{\partial n(y)} - \frac{\partial p(y)}{\partial n(y)} G(x,y) \right] dS(y) + p_{inc}(x)$$
其中格林函数：
$$G(x,y) = \frac{e^{ik|x-y|}}{4\pi|x-y|}$$

### Burton-Miller公式
为了解决特征频率处的非唯一性问题，使用Burton-Miller公式：
$$(H + i\eta G) p = -G \frac{\partial p}{\partial n} + i\eta p_{inc}$$
通常取 $\eta = 1/k$。

## 边界条件

### Neumann边界条件（刚性散射体）
表面法向速度为零：
$$\frac{\partial p}{\partial n} = -\frac{\partial p_{inc}}{\partial n}$$

### Dirichlet边界条件（软边界）
表面声压为零：
$$p = -p_{inc}$$

## 依赖项

- numpy
- scipy

## 文件说明

- `bem_acoustic.py`: 主要实现文件
- `README.md`: 说明文档

## 奇异积分处理详解

### 问题背景

在边界元法中，当场点和源点重合或非常接近时，格林函数$G(x,y) = \frac{e^{ikr}}{4\pi r}$中的$1/r$项会产生（近）奇异性。使用标准高斯积分会导致精度显著下降。

### Duffy坐标变换

**原理**：将三角形通过坐标变换映射到单位正方形，从而消除1/r型奇异性。

对于以顶点$v_0$为公共顶点的三角形：
- 标准三角形坐标：$y = v_0 + u(v_1-v_0) + v(v_2-v_0),\ u,v \ge 0,\ u+v \le 1$
- Duffy变换后：奇异被积函数变为非奇异

**特点**：
- 精确处理1/r型奇异性
- 计算效率高
- 适用于场点位于单元顶点附近的情况

### 细分积分法

**原理**：将近奇异单元递归细分为多个小子单元，在每个子单元上使用高斯积分。随着细分层数增加，精度不断提高。

**特点**：
- 实现简单，通用性强
- 计算量随细分层数指数增长
- 适用于各种奇异情况

### 自动检测机制

通过距离-单元尺寸比自动检测近奇异单元：

```
is_near_singular = min_distance < threshold * h
```

其中：
- `min_distance`：场点到单元各顶点的最小距离
- `h`：单元特征尺寸（面积的平方根）
- `threshold`：近奇异阈值（默认2.0，可调节）

### 使用方法

```python
# 启用奇异积分修正（默认）
p_total, dp_total_dn = solve_acoustic_scattering(
    mesh, k, use_singular_correction=True
)

# 禁用奇异积分修正（用于对比）
p_simple, _ = solve_acoustic_scattering(
    mesh, k, use_singular_correction=False
)

# 自定义近奇异阈值
p_custom, _ = solve_acoustic_scattering(
    mesh, k, 
    use_singular_correction=True,
    near_singular_threshold=3.0
)
```

### 运行对比测试

```bash
python bem_acoustic.py test
```

测试程序会输出：
- 无修正和有修正的结果对比
- 绝对误差和相对误差的统计值
- 奇异积分修正的应用次数

## 多层快速多极子方法 (MLFMA)

### 工作原理

MLFMA通过空间分层和平面波展开将计算复杂度从O(N²)降低到O(N)：

```
第L层（最细）      叶节点：直接计算近场相互作用
   |                    ↑
   | 向上传递 (M2M)    |  V+ = 多极展开
   ↓                    |
...                   ...
   |                    |
   |  M2L变换          |  W = 局部展开
   ↓                    |
第0层（最粗）      根节点：聚合所有远场贡献
   |                    ↑
   | 向下传递 (L2L)    |
   ↓                    |
计算表面声压           GMRES迭代求解器
```

### 关键算法步骤

**1. 向上传递 (Multipole-to-Multipole, M2M)**
- 从叶节点到根节点，逐层聚合子节点的多极展开
- 利用平移加法定理将子节点的V+转换到父节点

**2. 多极到局部变换 (Multipole-to-Local, M2L)**
- 在每个层级，计算非相邻节点之间的相互作用
- 将远场贡献转换为局部展开W

**3. 向下传递 (Local-to-Local, L2L)**
- 从根节点到叶节点，逐层传播局部展开
- 将父节点的W转换到子节点

**4. 近场直接计算**
- 相邻节点之间的贡献直接计算

### MLFMA使用示例

```python
from mlfma import MLFMABEMSolver, create_mesh_by_size

# 创建5000单元网格
mesh, _ = create_mesh_by_size(5000)

# 初始化MLFMA求解器
solver = MLFMABEMSolver(
    mesh, 
    k=2*np.pi*100/343,  # 波数
    p=8,                  # 展开阶数
    max_elements_per_leaf=50
)

# 求解
p_scat = solver.solve(rhs, tol=1e-3, maxiter=200)
```

### 复杂度对比

| 方法 | 计算复杂度 | 内存复杂度 | 适用规模 |
|-----|-----------|-----------|---------|
| 直接BEM | O(N²) | O(N²) | < 1000 |
| FMM-BEM | O(N) | O(N) | > 10000 |

**内存估算示例**：
- N = 10,000
  - 直接BEM: 10⁸ × 16字节 ≈ 1.6 GB
  - FMM-BEM: ~10⁶ × 16字节 ≈ 16 MB
- N = 100,000
  - 直接BEM: 10¹⁰ × 16字节 ≈ 160 GB
  - FMM-BEM: ~10⁷ × 16字节 ≈ 160 MB

### 性能优化建议

1. **网格密度与波数匹配**
   - 每波长至少6-10个单元
   - 过密网格增加计算量但精度提高有限

2. **叶节点大小选择**
   - 推荐每叶节点30-100个单元
   - 平衡近场和远场计算量

3. **展开阶数p**
   - 推荐p=6-10
   - p增大提高精度但增加内存和计算量

4. **GMRES参数**
   - restart=30-50
   - tol=1e-3~1e-4对于大多数工程应用足够

## 文件清单

| 文件 | 功能 | 适用场景 |
|-----|------|---------|
| `bem_acoustic.py` | 标准BEM + 奇异积分处理 | 中小规模问题 (< 5000) |
| `fmm_bem.py` | 基础FMM实现 | 教学演示 |
| `mlfma.py` | 完整MLFMA实现 | 大规模问题 (> 5000) |
| `README.md` | 文档 | 参考 |

## 注意事项

1. 本实现使用分段常数近似（每个单元一个自由度）
2. 对于大问题（>5000单元），必须使用MLFMA加速
3. 奇异积分已通过Duffy坐标变换精确处理，显著提高了计算精度
4. MLFMA的实际性能取决于网格质量和波数大小
5. 对于非常精细的网格，可适当调小`near_singular_threshold`参数以平衡精度和计算效率

## 参考资料

1. Song, J. M., & Chew, W. C. (1995). Multilevel fast multipole algorithm for electromagnetic scattering by large complex objects. IEEE Transactions on Antennas and Propagation.
2. Coifman, R., et al. (1993). The fast multipole method for the wave equation: A pedestrian prescription. IEEE Antennas and Propagation Magazine.
3. Ochmann, M. (1999). The fast multipole boundary element method for potential problems: A tutorial. Engineering Analysis with Boundary Elements.
4. Duffy, M. G. (1982). Quadrature over a pyramid or cube of integrands with a singularity at a vertex. SIAM Journal on Numerical Analysis.
