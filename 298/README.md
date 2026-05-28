# 树模型特征重要性计算工具

## 功能说明

支持计算以下树模型的特征重要性：
- **随机森林** (RandomForestClassifier / RandomForestRegressor)
- **XGBoost** (XGBClassifier / XGBRegressor)

### 特征重要性方法
1. **Gini重要性**（基于不纯度减少，有高基数偏向）
2. **排列重要性**（Permutation Importance，公平）
3. **SHAP重要性**（SHapley Additive exPlanations，基于博弈论，最公平）

### 可视化工具
1. **PDP** (Partial Dependence Plot) - 部分依赖图，显示特征对预测的平均边际效应
2. **ICE** (Individual Conditional Expectation) - 个体条件期望图，显示每个样本的特征效应曲线
3. **2D PDP** - 二维部分依赖图，显示两个特征的交互效应

## 快速开始

### 基本用法

```python
from feature_importance import calculate_feature_importance

# 排列重要性（默认推荐）
permutation_importance = calculate_feature_importance(
    model=trained_model,
    X=X_test,
    y=y_test,
    method='permutation'
)

# SHAP重要性（最公平）
shap_importance = calculate_feature_importance(
    model=trained_model,
    X=X_test,
    y=y_test,
    method='shap'
)
```

### 完整示例

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from feature_importance import calculate_feature_importance

# 加载数据
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# 训练模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 计算Gini重要性
gini_imp = calculate_feature_importance(
    model, X_test, y_test, method='gini',
    feature_names=data.feature_names
)
print("Gini重要性排名:")
print(gini_imp)

# 计算排列重要性
perm_imp = calculate_feature_importance(
    model, X_test, y_test, method='permutation',
    feature_names=data.feature_names, n_repeats=5
)
print("\n排列重要性排名:")
print(perm_imp)
```

### PDP + ICE 可视化

```python
from feature_importance import plot_pdp_ice, plot_top_features_pdp, plot_2d_pdp

# 单个特征的 PDP + ICE
fig, ax = plot_pdp_ice(
    model, X_test, 'feature_name', feature_names,
    kind='both',  # 'pdp', 'ice', 或 'both'
    save_path='pdp_ice.png'
)

# Top-K 特征的 PDP + ICE
fig, axes = plot_top_features_pdp(
    model, X_test, y_test, feature_names,
    top_k=4, method='shap',
    save_path='top4_pdp_ice.png'
)

# 二维 PDP（特征交互）
fig, ax = plot_2d_pdp(
    model, X_test, ('feat1', 'feat2'), feature_names,
    save_path='pdp_2d.png'
)
```

### 偏向检测与方法对比

```python
from feature_importance import detect_cardinality_bias, compare_methods

# 检测高基数偏向
bias_df = detect_cardinality_bias(X_test, feature_names, gini_imp)
print(bias_df.attrs.get('cardinality_gini_correlation'))  # 相关系数 > 0.5 表示有偏向

# 对比多种方法
comparison = compare_methods(model, X_test, y_test, feature_names)
print(comparison.attrs.get('avg_rank_shift'))  # 排名偏移 > 3 表示Gini可能有偏向
```

## API 文档

### calculate_feature_importance

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | TreeModel | 必填 | 训练好的树模型（随机森林/XGBoost） |
| `X` | DataFrame / ndarray | 必填 | 特征数据 |
| `y` | Series / ndarray | 必填 | 目标变量 |
| `method` | str | `'permutation'` | 计算方法：`'gini'` / `'permutation'` / `'shap'` |
| `feature_names` | list | None | 特征名称列表（None则自动推断） |
| `n_repeats` | int | 10 | 排列重要性的重复次数 |
| `random_state` | int | 42 | 随机种子 |
| `scoring` | str | None | 排列重要性的评分指标 |
| `approximate` | bool | False | SHAP快速近似计算 |
| `check_additivity` | bool | True | SHAP叠加性检查 |

**返回：**
- `pd.DataFrame`：按重要性降序排列的特征重要性表

### plot_pdp_ice

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | TreeModel | 必填 | 训练好的树模型 |
| `X` | DataFrame / ndarray | 必填 | 特征数据 |
| `feature_index` | int / str | 必填 | 特征索引或名称 |
| `feature_names` | list | None | 特征名称列表 |
| `kind` | str | `'both'` | `'pdp'` / `'ice'` / `'both'` |
| `n_ice_samples` | int | 50 | ICE曲线采样数量（避免过密） |
| `ice_alpha` | float | 0.1 | ICE曲线透明度 |
| `grid_resolution` | int | 50 | 特征取值网格点数 |
| `figsize` | tuple | (8, 5) | 图像尺寸 |
| `save_path` | str | None | 保存路径（None则不保存） |

### plot_top_features_pdp

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | TreeModel | 必填 | 训练好的树模型 |
| `X` | DataFrame / ndarray | 必填 | 特征数据 |
| `y` | Series / ndarray | 必填 | 目标变量（用于计算重要性） |
| `top_k` | int | 4 | 展示前K个重要特征 |
| `method` | str | `'permutation'` | 用于排序的重要性方法 |
| `ncols` | int | 2 | 子图列数 |
| `figsize_per_plot` | tuple | (6, 4) | 每个子图尺寸 |
| `save_path` | str | None | 保存路径 |

### plot_2d_pdp

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | TreeModel | 必填 | 训练好的树模型 |
| `X` | DataFrame / ndarray | 必填 | 特征数据 |
| `feature_pair` | tuple | 必填 | 两个特征的索引或名称 |
| `grid_resolution` | int | 30 | 网格分辨率 |
| `figsize` | tuple | (8, 6) | 图像尺寸 |
| `save_path` | str | None | 保存路径 |

## 三种重要性方法对比

| 特性 | Gini重要性 | 排列重要性 | SHAP重要性 |
|------|-----------|-----------|-----------|
| **计算速度** | 快（模型内置） | 慢（需多次评估） | 中等（TreeExplainer优化） |
| **对测试集依赖** | 否（基于训练） | 是 | 是 |
| **抗过拟合** | 较差 | 好 | 最好 |
| **高基数偏向** | 有 | 无 | 无 |
| **理论基础** | 启发式 | 经验性 | 博弈论（严谨） |
| **支持交互检测** | 否 | 否 | 是（SHAP交互值） |

## PDP/ICE 读图指南

| 现象 | 含义 |
|------|------|
| **向上倾斜曲线** | 特征值增加 → 预测值增加（正相关） |
| **向下倾斜曲线** | 特征值增加 → 预测值减少（负相关） |
| **S型曲线** | 存在阈值效应（超过某值后效应突变） |
| **平行线（ICE）** | 特征效应一致，无交互 |
| **交叉线（ICE）** | 存在特征交互，不同样本的效应方向不同 |
| **平坦曲线** | 该特征对模型预测影响很小 |

## 依赖安装

```bash
# 核心依赖
pip install scikit-learn numpy pandas

# 可选依赖
pip install xgboost      # XGBoost模型支持
pip install shap         # SHAP重要性计算
pip install matplotlib   # PDP/ICE可视化
```
