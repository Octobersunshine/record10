# 柯西分布参数估计器

使用 Python + SciPy 实现柯西分布的参数估计、分布检验和异常值稳健性分析。

## 版本说明

| 版本 | 文件 | 功能 |
|------|------|------|
| 基础版 | `cauchy_estimator.py` | 基本的柯西分布参数估计 |
| 改进版 | `cauchy_estimator_improved.py` | 解决似然多峰问题，多种优化算法 |
| **增强版** | **`cauchy_estimator_enhanced.py`** | **含分布检验和异常值稳健性分析（推荐）** |

## 核心功能

### 1. 参数估计（解决似然多峰问题）

提供5种拟合方法：
- `default`: SciPy默认方法（最快但易陷入局部最优）
- `robust`: 使用中位数和四分位距作为初始值
- `multistart`: **多起点局部优化（默认推荐）**
- `basinhopping`: Basin-Hopping全局优化
- `de`: 差分进化全局优化（最可靠但较慢）

### 2. 分布拟合检验

#### KS检验（Kolmogorov-Smirnov Test）
- 检验样本数据是否符合柯西分布或正态分布
- 通过p值判断是否拒绝原假设

#### 信息准则
- **AIC** (Akaike Information Criterion): 赤池信息准则
- **BIC** (Bayesian Information Criterion): 贝叶斯信息准则
- 值越小表示模型拟合越好

#### QQ图统计量
- 计算样本分位数与理论分位数的相关系数
- 计算拟合线的斜率、截距和RMSE
- 量化分位数的匹配程度

#### 智能推荐
- 综合AIC、BIC、QQ相关系数投票
- 返回推荐分布：`cauchy` / `normal` / `uncertain`

### 3. 异常值稳健性检验

对比柯西分布与正态分布对异常值的敏感性：
- 自动添加指定数量和大小的异常值
- 计算加入异常值前后的参数相对变化
- 返回稳健性比率（正态变化/柯西变化）
- 量化评估柯西分布的重尾优势

## 安装依赖

```bash
pip install numpy scipy
```

## 快速开始

### 基本用法

```python
from cauchy_estimator_enhanced import CauchyEstimator

estimator = CauchyEstimator()

# 拟合数据（默认使用多起点优化）
estimator.fit(data, method='multistart')

# 获取参数
params = estimator.get_params()
print(f"x0 = {params['x0']}, γ = {params['gamma']}")
```

### 分布检验

```python
# 检验数据分布（柯西 vs 正态）
test_result = estimator.test_distribution(data, alpha=0.05)

# 获取推荐分布
recommended = test_result['comparison']['recommended_distribution']
print(f"推荐分布: {recommended}")

# 查看详细信息
print(f"柯西AIC: {test_result['cauchy']['aic']}")
print(f"正态AIC: {test_result['normal']['aic']}")
print(f"KS p值(柯西): {test_result['cauchy']['ks_test']['p_value']}")
print(f"QQ相关系数(柯西): {test_result['cauchy']['qq_stats']['correlation']}")
```

### 异常值稳健性检验

```python
# 检验异常值稳健性
robustness = estimator.outlier_robustness_test(
    data,
    n_outliers=5,           # 异常值数量
    outlier_magnitude=10    # 异常值大小（倍数）
)

# 查看稳健性比率
print(f"位置参数稳健性比率: {robustness['changes']['robustness_ratio_x0']:.2f}x")
print(f"尺度参数稳健性比率: {robustness['changes']['robustness_ratio_gamma']:.2f}x")

# 查看结论
print(f"柯西位置参数更稳健: {robustness['conclusion']['cauchy_more_robust_x0']}")
print(f"柯西尺度参数更稳健: {robustness['conclusion']['cauchy_more_robust_gamma']}")
```

### 其他功能

```python
# 计算概率密度
pdf_value = estimator.pdf(x)

# 计算对数似然
log_likelihood = estimator.log_likelihood(data)

# 尝试所有方法，自动选择最优
results, best_method = estimator.fit_all_methods(data)
```

### 生成测试数据

```python
from cauchy_estimator_enhanced import generate_sample_data, generate_normal_data

# 生成柯西分布样本
cauchy_data = generate_sample_data(
    true_location=0.0,
    true_scale=1.0,
    size=1000,
    random_seed=42
)

# 生成正态分布样本
normal_data = generate_normal_data(
    mu=0.0,
    sigma=1.0,
    size=1000,
    random_seed=42
)
```

## API 详细说明

### CauchyEstimator 类

#### 方法

| 方法 | 说明 | 参数 | 返回值 |
|------|------|------|--------|
| `fit(data, method='multistart')` | 拟合数据 | data: 样本数据, method: 优化方法 | (location, scale) |
| `fit_all_methods(data)` | 尝试所有方法并选择最优 | data: 样本数据 | (results_dict, best_method) |
| `pdf(x)` | 计算概率密度 | x: 输入值 | 概率密度值 |
| `get_params()` | 获取拟合参数 | 无 | {'x0', 'gamma', 'method'} |
| `log_likelihood(data)` | 计算对数似然 | data: 样本数据 | 对数似然值 |
| `test_distribution(data, alpha=0.05)` | 分布拟合检验 | data: 样本数据, alpha: 显著性水平 | 见下文 |
| `outlier_robustness_test(data, n_outliers=5, outlier_magnitude=10)` | 异常值稳健性检验 | 见下文 | 见下文 |

#### test_distribution 返回结构

```python
{
    'cauchy': {
        'params': {'x0', 'gamma'},
        'ks_test': {'statistic', 'p_value'},
        'log_likelihood', 'aic', 'bic',
        'qq_stats': {'correlation', 'slope', 'intercept', 'rmse'}
    },
    'normal': {
        # 同上，包含正态分布参数和检验结果
    },
    'comparison': {
        'is_cauchy_by_ks', 'is_normal_by_ks',
        'preferred_by_aic', 'preferred_by_bic', 'preferred_by_qq_correlation',
        'recommended_distribution',
        'aic_diff', 'bic_diff', 'qq_correlation_diff'
    },
    'qq_data': {
        # QQ图的原始数据，用于绘制
    }
}
```

#### outlier_robustness_test 返回结构

```python
{
    'original': {          # 原始数据拟合结果
        'cauchy': {'x0', 'gamma'},
        'normal': {'mu', 'sigma'}
    },
    'with_outliers': {     # 加入异常值后的拟合结果
        # 同上结构
    },
    'changes': {           # 参数相对变化
        'cauchy_x0_relative_change',
        'cauchy_gamma_relative_change',
        'normal_mu_relative_change',
        'normal_sigma_relative_change',
        'robustness_ratio_x0',       # 正态变化/柯西变化
        'robustness_ratio_gamma'     # >1表示柯西更稳健
    },
    'outlier_info': {      # 异常值信息
        'n_outliers', 'outlier_magnitude',
        'outlier_positions', 'outlier_values'
    },
    'conclusion': {
        'cauchy_more_robust_x0',
        'cauchy_more_robust_gamma',
        'robustness_ratio_x0',
        'robustness_ratio_gamma'
    }
}
```

## 测试文件

- `test_enhanced.py`: 完整功能测试，展示所有新特性
- `test_multimodal.py`: 多峰问题测试
- `test_import.py`: 基础功能测试

## 典型应用场景

### 1. 金融数据建模
- 收益率通常具有重尾特性
- 极端事件（黑天鹅）更符合柯西分布
- 使用 `test_distribution()` 判断是否应使用柯西分布

### 2. 稳健统计分析
- 数据中存在异常值时
- 比较柯西与正态的参数稳定性
- 使用 `outlier_robustness_test()` 量化评估

### 3. 模型选择
- 在柯西和正态之间选择更合适的分布
- 基于AIC/BIC/KS检验综合判断

## 数学背景

### 柯西分布
概率密度函数：
```
f(x; x0, γ) = 1 / (πγ(1 + ((x - x0)/γ)²))
```

特点：
- 均值和方差不存在（无定义）
- 中位数 = x0（位置参数）
- 半宽度半高 = γ（尺度参数）
- 重尾分布，对异常值更稳健

### 为什么柯西对异常值更稳健？
- 正态分布：似然函数随距离平方衰减，异常值影响大
- 柯西分布：似然函数随距离一次方衰减，异常值权重较低
- 本质上：MLE对于柯西相当于中位数估计，对于正态相当于均值估计

## 注意事项

1. **样本量**：KS检验在小样本下功效较低，建议样本量 > 50
2. **参数解释**：柯西没有均值和方差，使用中位数(x0)和四分位距
3. **计算成本**：全局优化（`de`, `basinhopping`）计算较慢
4. **异常值检验**：稳健性检验基于模拟，结果有随机性

## 文件列表

| 文件 | 说明 |
|------|------|
| `cauchy_estimator_enhanced.py` | **主文件（推荐使用）** |
| `cauchy_estimator_improved.py` | 改进版（仅参数估计） |
| `cauchy_estimator.py` | 基础版 |
| `test_enhanced.py` | 增强功能测试 |
| `test_multimodal.py` | 多峰问题测试 |
| `README.md` | 本文档 |

## 版本历史

- **v3.0 (增强版)**: 新增KS检验、AIC/BIC、QQ图统计量、分布推荐、异常值稳健性检验
- **v2.0 (改进版)**: 解决似然多峰问题，新增多种优化算法
- **v1.0 (基础版)**: 基本的柯西分布参数估计
