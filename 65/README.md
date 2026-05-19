# 基因调控网络的Gillespie随机模拟算法（SSA）实现

## 概述

本项目使用Python实现了化学主方程（Chemical Master Equation）的Gillespie随机模拟算法（Stochastic Simulation Algorithm, SSA），用于模拟基因调控网络的动力学行为。

## 目录结构

- `gene_regulation_ssa.py` - 主程序文件，包含完整的实现

## 核心组件

### 1. Reaction类
表示单个化学反应，包含：
- `propensity`: 反应倾向函数（反应速率）
- `stoichiometry`: 化学计量向量
- `name`: 反应名称

### 2. GillespieSSA类
Gillespie随机模拟算法的核心实现，包含：
- `add_reaction()`: 添加反应到网络
- `simulate()`: 执行SSA模拟
- `plot_results()`: 可视化模拟结果

## Gillespie SSA算法原理

Gillespie算法是一种精确的蒙特卡洛方法，用于模拟化学主方程。算法步骤：

1. **计算所有反应的倾向值**
2. **计算总倾向值**
3. **采样下一个反应发生的时间**：τ ~ Exp(1/总倾向值)
4. **采样哪个反应发生**：按倾向值概率加权
5. **更新系统状态**：应用选中反应的化学计量
6. **重复直到达到最大时间**

## 内置网络模型

### 1. 基本基因表达网络

包含4个物种：
- DNA: 基因模板
- mRNA: 信使RNA
- Protein: 蛋白质
- Protein_Dimer: 蛋白质二聚体

包含6个反应：
1. **转录**: DNA → DNA + mRNA
2. **mRNA降解**: mRNA → ∅
3. **翻译**: mRNA → mRNA + Protein
4. **蛋白质降解**: Protein → ∅
5. **二聚化**: 2 Protein → Protein_Dimer
6. **二聚解离**: Protein_Dimer → 2 Protein

### 2. 阻遏振荡网络（Repressilator）

经典的合成生物学振荡网络，包含3个基因相互抑制：
- X, Y, Z: 三种蛋白质
- X_mRNA, Y_mRNA, Z_mRNA: 三种mRNA

调控关系：
- X抑制Y的转录
- Y抑制Z的转录
- Z抑制X的转录

使用Hill函数模拟抑制效应：
```
rate = α₀ + (α - α₀) / (1 + (repressor/Kd)ⁿ)
```

## 使用方法

### 基本用法

```python
import numpy as np
from gene_regulation_ssa import GillespieSSA, Reaction

# 创建模拟实例
species_names = ['A', 'B', 'C']
ssa = GillespieSSA(species_names)

# 添加反应
ssa.add_reaction(Reaction(
    propensity=lambda s, t: 0.1 * s[0],
    stoichiometry=np.array([-1, 1, 0]),
    name='A → B'
))

# 运行模拟
initial_state = np.array([100, 0, 0])
times, states = ssa.simulate(initial_state, t_max=50.0)

# 可视化结果
ssa.plot_results()
```

### 使用内置网络

```python
from gene_regulation_ssa import create_gene_expression_network, create_repressilator_network
import numpy as np

# 基因表达网络
ssa1 = create_gene_expression_network()
initial_state1 = np.array([1, 0, 0, 0])  # DNA=1, 其他=0
times1, states1 = ssa1.simulate(initial_state1, t_max=100.0)
ssa1.plot_results()

# 阻遏振荡网络
ssa2 = create_repressilator_network()
initial_state2 = np.array([0, 2, 0, 10, 0, 10])
times2, states2 = ssa2.simulate(initial_state2, t_max=200.0)
ssa2.plot_results()
```

### 运行主程序

```bash
python gene_regulation_ssa.py
```

## 依赖要求

- Python 3.7+
- NumPy
- Matplotlib

安装依赖：
```bash
pip install numpy matplotlib
```

## 扩展自定义网络

要创建自定义基因调控网络：

1. 定义物种名称列表
2. 创建GillespieSSA实例
3. 为每个反应定义倾向函数和化学计量向量
4. 添加所有反应
5. 设置初始状态并运行模拟

示例：创建一个简单的自调控网络

```python
def create_auto_regulation_network():
    species_names = ['DNA', 'DNA_Protein', 'mRNA', 'Protein']
    ssa = GillespieSSA(species_names)
    
    # 蛋白质结合到DNA（抑制）
    ssa.add_reaction(Reaction(
        propensity=lambda s, t: 0.01 * s[0] * s[3],
        stoichiometry=np.array([-1, 1, 0, -1]),
        name='Protein binding'
    ))
    
    # 蛋白质从DNA解离
    ssa.add_reaction(Reaction(
        propensity=lambda s, t: 0.1 * s[1],
        stoichiometry=np.array([1, -1, 0, 1]),
        name='Protein unbinding'
    ))
    
    # 从游离DNA转录
    ssa.add_reaction(Reaction(
        propensity=lambda s, t: 0.5 * s[0],
        stoichiometry=np.array([0, 0, 1, 0]),
        name='Transcription (active)'
    ))
    
    # 从抑制态DNA转录（基础水平）
    ssa.add_reaction(Reaction(
        propensity=lambda s, t: 0.01 * s[1],
        stoichiometry=np.array([0, 0, 1, 0]),
        name='Transcription (repressed)'
    ))
    
    return ssa
```

## 参考文献

1. Gillespie, D. T. (1977). Exact stochastic simulation of coupled chemical reactions. Journal of Physical Chemistry.
2. Elowitz, M. B., & Leibler, S. (2000). A synthetic oscillatory network of transcriptional regulators. Nature.
