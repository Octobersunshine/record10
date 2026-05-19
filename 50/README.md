# 空间计量模型 Python 实现

实现空间自回归模型（SAR）、空间误差模型（SEM），并通过拉格朗日乘数检验（LM test）自动选择最优模型。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 核心功能

### 1. 模型拟合

#### SAR 模型（空间自回归模型）
- 模型形式: `y = ρWy + Xβ + ε`
- 函数: `fit_sar_model(W, y, X, auto_standardize=True, verbose=True)`

#### SEM 模型（空间误差模型）
- 模型形式: `y = Xβ + μ, μ = λWμ + ε`
- 函数: `fit_sem_model(W, y, X, auto_standardize=True, verbose=True)`

### 2. 拉格朗日乘数检验（LM Test）

四种检验统计量：
- **LMlag**: 检验是否存在空间滞后相关性（SAR）
- **LMerr**: 检验是否存在空间误差相关性（SEM）
- **RLMlag**: 稳健LMlag检验（控制空间误差影响）
- **RLMerr**: 稳健LMerr检验（控制空间滞后影响）

函数: `spatial_lm_test(W, y, X)`

### 3. 自动模型选择

通过LM检验自动选择最优模型（OLS / SAR / SEM）

选择策略：
1. LMlag和LMerr都不显著 → 选择OLS
2. 只有LMlag显著 → 选择SAR
3. 只有LMerr显著 → 选择SEM
4. 两者都显著 → 比较RLMlag和RLMerr的p值，选择更小的

函数: `auto_select_model(W, y, X, alpha=0.05)`

### 4. 模型比较

并排比较SAR和SEM的拟合优度

函数: `print_model_comparison(sar_results, sem_results)`

### 5. 权重矩阵标准化

- 自动行标准化: 默认开启，确保空间系数可比性
- 手动标准化: `row_standardize(W_array)`
- 检查是否已标准化: `check_row_standardized(W_array)`

## 快速使用

```python
import numpy as np
import libpysal
from sar_model import (
    fit_sar_model, fit_sem_model, print_sar_results, print_sem_results,
    auto_select_model, print_lm_results, print_model_comparison
)

# 准备数据
n = 49
W = libpysal.weights.lat2W(7, 7)
W.transform = 'r'

X = np.random.randn(n, 1)
X = np.hstack([np.ones((n, 1)), X])

# 生成模拟数据（或使用真实数据）
W_matrix = W.full()[0]
I = np.eye(n)
rho_true = 0.5
beta_true = np.array([1.0, 2.0])
epsilon = np.random.randn(n, 1) * 0.5
inv_I_rhoW = np.linalg.inv(I - rho_true * W_matrix)
y = inv_I_rhoW @ (X @ beta_true.reshape(-1, 1) + epsilon)

# 方法1: 自动选择模型
selected_model, model_results, lm_results = auto_select_model(W, y, X)
print(f"选择的模型: {selected_model}")

# 方法2: 分别拟合并比较
sar_results = fit_sar_model(W, y, X, verbose=False)
sem_results = fit_sem_model(W, y, X, verbose=False)
print_model_comparison(sar_results, sem_results)

# 方法3: 先做LM检验再决定
lm_results = spatial_lm_test(W, y, X)
print_lm_results(lm_results)
```

## 运行示例

```bash
# 运行主文件演示
python sar_model.py

# 运行更多示例
python example_usage.py
```

## API 参考

### fit_sar_model(W, y, X=None, auto_standardize=True, verbose=True)

拟合空间自回归模型。

**参数:**
- `W`: 空间权重矩阵 (libpysal.weights.W 对象 或 numpy数组)
- `y`: 因变量 (n×1 或 n 维数组)
- `X`: 自变量矩阵 (n×k)，默认包含截距
- `auto_standardize`: 是否自动行标准化，默认True
- `verbose`: 是否打印标准化信息

**返回:** ML_Lag 模型结果对象

### fit_sem_model(W, y, X=None, auto_standardize=True, verbose=True)

拟合空间误差模型。参数同上。

**返回:** ML_Error 模型结果对象

### spatial_lm_test(W, y, X=None, auto_standardize=True)

执行拉格朗日乘数检验。

**返回:** 包含四个检验统计量和对应p值的字典

### auto_select_model(W, y, X=None, alpha=0.05, auto_standardize=True, verbose=True)

自动选择最优空间模型。

**返回:** `(selected_model, model_results, lm_results)`
- `selected_model`: 'OLS', 'SAR', 'SEM' 之一
- `model_results`: 拟合的模型结果对象
- `lm_results`: LM检验结果字典

## 模型结果属性

### SAR 模型
- `results.rho`: 空间自回归系数 ρ
- `results.betas`: 回归系数 β（包括截距）
- `results.se_rho`: ρ的标准误
- `results.std_err`: β的标准误
- `results.logll`: 对数似然值
- `results.aic`: AIC信息准则
- `results.sic`: SIC信息准则
- `results.pr2`: 伪R²

### SEM 模型
- `results.lam`: 空间误差系数 λ
- 其余属性同上

## 注意事项

1. **权重矩阵标准化**: 默认自动行标准化，确保空间系数跨模型可比。如确需关闭，设置 `auto_standardize=False`。

2. **孤岛观测值**: 如有邻居数为0的观测值，会产生警告。

3. **显著性水平**: LM检验默认使用 α=0.05，可通过参数调整。

4. **小样本**: 样本量较小时，LM检验的检验效能可能下降。

## 文件说明

- `sar_model.py` - 核心实现，包含所有模型和检验函数
- `example_usage.py` - 详细使用示例
- `requirements.txt` - 依赖包列表
- `README.md` - 本文档
