# GARCH(1,1) & Realized GARCH 波动率预测工具包

使用Python + arch库拟合金融收益率序列的波动率模型。包含两个核心改进：

1. ✅ **厚尾分布修正**：学生t分布/偏t分布，避免正态分布低估尾部风险
2. ✅ **高频数据利用**：Realized GARCH模型，利用日内高频数据显著提高预测精度

## 安装依赖

```bash
pip install -r requirements.txt
```

## 核心功能概览

| 模块 | 功能 | 关键特性 |
|------|------|----------|
| `fit_garch_and_predict` | 标准GARCH拟合 | t分布/skewt分布，厚尾修正 |
| `compare_distributions` | 分布对比分析 | AIC/BIC自动选优，VaR对比 |
| `fit_realized_garch` | Realized GARCH | 利用高频已实现波动率 |
| `compare_garch_realized_garch` | 模型对比 | 量化高频数据的价值 |
| `calculate_var_comparison` | 风险分析 | 尾部风险低估程度量化 |

## 文件说明

- `garch_model.py` - 完整功能版本，包含所有高级功能
- `simple_garch.py` - 简化版本，专注于核心功能
- `realized_volatility_example.py` - 高频数据与Realized GARCH完整示例
- `requirements.txt` - Python依赖包列表

---

## 第一部分：厚尾分布修正（解决尾部风险低估）

### 为什么必须用厚尾分布？

**重要修正**：
- ❌ **正态分布假设**：严重低估尾部风险（黑天鹅事件），VaR计算偏差可达30-50%
- ✅ **学生t分布**：通过自由度参数ν捕捉厚尾特性，是金融数据的标准选择（ν通常在3-8之间）
- ✅ **偏t分布**：同时捕捉厚尾和偏度，最适合真实金融数据（股票收益率通常左偏）

### 快速开始

```bash
python garch_model.py
```

这将运行完整演示，包括：
1. 标准GARCH(1,1)-t模型拟合
2. VaR尾部风险对比分析
3. 三种分布假设对比
4. Realized GARCH高频数据演示

### 使用自定义数据

```python
import pandas as pd
from garch_model import fit_garch_and_predict, compare_distributions

# 加载你的收益率数据
returns = pd.read_csv('returns.csv', index_col=0, parse_dates=True)['return']

# 选项1: 使用学生t分布（推荐，默认）
result, forecast, cond_vol = fit_garch_and_predict(
    returns=returns,
    forecast_horizon=10,
    plot=True,
    dist='t'
)

# 选项2: 对比所有分布并自动选择最优
results, forecasts = compare_distributions(returns, forecast_horizon=5)
```

---

## 第二部分：Realized GARCH - 利用高频数据提高预测精度

### 什么是Realized GARCH？

标准GARCH仅使用日度收益率：
```
σ²(t) = ω + α · ε²(t-1) + β · σ²(t-1)
```

Realized GARCH加入日内高频信息（已实现波动率RV）：
```
σ²(t) = ω + α · σ²(t-1) + β · RV(t-1) + γ · z(t-1)
```

**核心优势**：利用日内高频信息（5分钟、1分钟数据），预测精度通常提高10-30%

### 使用Realized GARCH

```python
import pandas as pd
from garch_model import fit_realized_garch, compare_garch_realized_garch

# 方法1: 直接从高频数据计算已实现波动率
from realized_volatility_example import calculate_realized_volatility

# 加载高频数据 (timestamp, return)
intraday_df = pd.read_csv('5min_returns.csv', index_col=0, parse_dates=True)
rv = calculate_realized_volatility(intraday_df['return'], method='rv')

# 计算日收益率
daily_returns = intraday_df['return'].resample('D').sum()

# 拟合Realized GARCH
result, forecast, cond_vol = fit_realized_garch(
    returns=daily_returns,
    realized_measure=rv,
    forecast_horizon=5,
    plot=True,
    dist='t'
)

# 方法2: 自动对比标准GARCH和Realized GARCH
comparison = compare_garch_realized_garch(
    returns=daily_returns,
    realized_measure=rv,
    forecast_horizon=5
)
```

### 运行Realized GARCH完整示例

```bash
python realized_volatility_example.py
```

### 已实现波动率测度选择

| 测度 | 公式 | 特点 | 适用场景 |
|------|------|------|----------|
| **RV** (已实现方差) | Σr² | 计算简单，标准选择 | 流动性好、噪声低的资产 |
| **RK** (已实现核) | 加权自相关 | 对微观结构噪声稳健 | 一般推荐使用 |
| **BV** (双幂变差) | (π/2)Σ\|rᵢrᵢ₋₁\| | 稳健于跳跃 | 波动大、跳跃多的资产 |

### 实践建议

**数据频率选择**：
- ✅ 5分钟：标准选择，平衡精度与噪声
- ⚠️ 1小时：信息损失较大
- ⚠️ 1分钟：微观结构噪声增大，需要RK降噪

**样本大小**：
- 至少1年（250交易日）
- 2-3年数据效果更佳

**典型应用**：
- 期权定价：需要精确波动率预测
- 风险管理：VaR计算更准确
- 组合优化：协方差矩阵估计更精确

---

## 模型公式

### 标准GARCH(1,1)

```
σ²(t) = ω + α · ε²(t-1) + β · σ²(t-1)
```

### Realized GARCH(1,1)

```
σ²(t) = ω + α · σ²(t-1) + β · RV(t-1) + γ · z(t-1)
```

### 残差分布选项

- **正态分布**：ε ~ N(0, σ²) ⚠️ 仅用于对比
- **学生t分布**：ε ~ t(ν, σ²) ✅ 推荐
- **偏t分布**：ε ~ skewt(ν, λ, σ²) ✅ 最推荐

### 参数解释

- ω (omega): 常数项
- α (alpha): GARCH项系数（波动率持续性）
- β (beta): 已实现波动率项系数（高频信息权重）
- γ (gamma): 杠杆效应项（可选）
- ν (nu): t分布自由度（越小，尾部越厚）
- λ (lambda): 偏度参数（负表示左偏）

---

## 如何解读结果？

### 分布选择判断（AIC/BIC准则）
- AIC/BIC值越小越好
- 真实金融数据几乎总是t分布 > skewt分布 > 正态分布

### 自由度ν的含义
- ν > 10: 接近正态分布，厚尾不明显
- 5 < ν < 10: 中等厚尾，典型成熟市场
- 3 < ν < 5: 显著厚尾，新兴市场或危机期
- ν < 3: 极端厚尾，风险极高

### Realized GARCH改进判断
- AIC改进 > 2: 显著改进
- Log-likelihood提升: 模型拟合更好
- 预测偏差更小: 样本外预测更准

---

## 简化版快速使用

```python
from simple_garch import simple_garch_forecast
import numpy as np

# 你的收益率数据
returns = np.array([0.01, -0.02, 0.015, ...])

# 使用学生t分布预测
forecast, result = simple_garch_forecast(returns, horizon=5, dist='t')
print(f"估计的自由度 ν: {result.params['nu']:.2f}")
print(forecast)
```

---

## 输出说明

运行后会输出：
- 收益率序列统计（均值、标准差、偏度、峰度）
- 残差分布假设说明
- 模型参数估计值（包括厚尾参数ν和偏度λ）
- 厚尾程度警告和解释
- AIC/BIC对比和最优分布推荐
- Realized GARCH vs 标准GARCH对比
- 条件波动率序列
- 未来N期波动率预测
- 尾部风险VaR低估程度量化
