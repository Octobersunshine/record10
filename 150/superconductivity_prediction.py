#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超导临界温度Tc预测模型 - 小样本优化版
使用材料特征（化学成分、晶体结构、带隙、德拜温度）预测超导临界温度
优化策略：
- 留一法交叉验证(LOOCV) - 适用于小样本(~1000个)
- SHAP特征选择 - 防止特征过多导致泛化差
- 正则化和早停 - 防止过拟合
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost未安装，将仅使用随机森林模型")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP未安装，将使用内置特征重要性进行特征选择")

import joblib

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_synthetic_data(n_samples=1000):
    """
    生成模拟超导材料数据集（小样本：约1000个）
    包含：化学成分、晶体结构、带隙、德拜温度等特征
    """
    np.random.seed(42)
    
    elements = ['Cu', 'Fe', 'O', 'La', 'Ba', 'Sr', 'Ca', 'Y', 'Bi', 'Ti', 'Nb', 'H', 'C', 'S', 'Se']
    composition = np.random.dirichlet(np.ones(len(elements)) * 0.5, n_samples) * 100
    
    crystal_structure = np.random.choice(['cubic', 'tetragonal', 'orthorhombic', 'hexagonal'], n_samples)
    lattice_constant = np.random.uniform(3.0, 15.0, n_samples)
    atomic_density = np.random.uniform(0.01, 0.1, n_samples)
    coordination_number = np.random.randint(4, 13, n_samples)
    
    band_gap = np.random.exponential(0.5, n_samples)
    band_gap = np.clip(band_gap, 0, 5.0)
    fermi_energy = np.random.uniform(-5.0, 5.0, n_samples)
    carrier_density = np.random.exponential(1e21, n_samples)
    carrier_density = np.clip(carrier_density, 1e18, 1e23)
    
    debye_temperature = np.random.uniform(100, 800, n_samples)
    melting_point = np.random.uniform(500, 3000, n_samples)
    
    bulk_modulus = np.random.uniform(50, 500, n_samples)
    shear_modulus = np.random.uniform(20, 200, n_samples)
    
    df = pd.DataFrame(composition, columns=[f'comp_{el}' for el in elements])
    df['crystal_structure'] = crystal_structure
    df['lattice_constant'] = lattice_constant
    df['atomic_density'] = atomic_density
    df['coordination_number'] = coordination_number
    df['band_gap'] = band_gap
    df['fermi_energy'] = fermi_energy
    df['log_carrier_density'] = np.log10(carrier_density)
    df['debye_temperature'] = debye_temperature
    df['melting_point'] = melting_point
    df['bulk_modulus'] = bulk_modulus
    df['shear_modulus'] = shear_modulus
    
    lambda_ep = 0.5 + 0.3 * np.random.randn(n_samples)
    lambda_ep = np.clip(lambda_ep, 0.1, 2.5)
    mu_star = 0.1 + 0.05 * np.random.randn(n_samples)
    mu_star = np.clip(mu_star, 0.05, 0.2)
    
    Tc = (debye_temperature / 1.45) * np.exp(-1.04 * (1 + lambda_ep) / (lambda_ep - mu_star * (1 + 0.62 * lambda_ep)))
    
    structure_factor = pd.get_dummies(crystal_structure).values @ np.array([1.2, 0.9, 1.1, 0.8])
    gap_effect = np.exp(-band_gap / 0.5)
    
    Tc = Tc * structure_factor * gap_effect
    Tc = Tc + np.random.normal(0, 3.0, n_samples)
    Tc = np.clip(Tc, 0, 150)
    
    df['Tc'] = Tc
    
    return df


def preprocess_data(df):
    """
    数据预处理
    """
    X = df.drop('Tc', axis=1)
    y = df['Tc']
    
    X = pd.get_dummies(X, columns=['crystal_structure'], drop_first=True)
    
    return X, y


def shap_feature_selection(model, X, y, n_features=15, feature_names=None):
    """
    使用SHAP值进行特征选择
    """
    print(f"\n{'='*60}")
    print("SHAP特征选择...")
    print(f"{'='*60}")
    
    if not SHAP_AVAILABLE:
        print("SHAP不可用，使用模型内置特征重要性...")
        importances = model.feature_importances_
        indices = np.argsort(importances)[-n_features:]
        selected_features = [feature_names[i] for i in indices]
        print(f"基于特征重要性选择了 {n_features} 个特征")
        return selected_features
    
    print(f"计算SHAP值（样本数: {X.shape[0]}）...")
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        shap_importance = np.abs(shap_values).mean(axis=0)
        
        indices = np.argsort(shap_importance)[-n_features:]
        selected_features = [feature_names[i] for i in indices]
        
        print(f"\nSHAP特征重要性 (Top {n_features}):")
        for i in reversed(indices):
            print(f"  {feature_names[i]:25s}: {shap_importance[i]:.4f}")
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names, 
                         max_display=n_features, show=False, plot_size=(12, 8))
        plt.tight_layout()
        plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\nSHAP摘要图已保存为: shap_summary.png")
        
    except Exception as e:
        print(f"SHAP计算出错: {e}")
        print("使用模型内置特征重要性...")
        importances = model.feature_importances_
        indices = np.argsort(importances)[-n_features:]
        selected_features = [feature_names[i] for i in indices]
    
    print(f"\n最终选择了 {len(selected_features)} 个特征")
    return selected_features


def train_random_forest_loocv(X, y, param_grid=None):
    """
    使用留一法交叉验证训练随机森林
    """
    print(f"\n{'='*60}")
    print("训练随机森林模型 (LOOCV)...")
    print(f"{'='*60}")
    
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 150],
            'max_depth': [8, 12, None],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [3, 5],
            'max_features': ['sqrt', 0.5]
        }
    
    base_rf = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
        oob_score=True
    )
    
    loo = LeaveOneOut()
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        base_rf, param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证MSE: {-grid_search.best_score_:.4f}")
    print(f"OOB得分: {grid_search.best_estimator_.oob_score_:.4f}")
    
    best_rf = grid_search.best_estimator_
    
    print(f"\n执行留一法交叉验证 (LOOCV)...")
    loo_predictions = []
    loo_true = []
    
    for train_idx, test_idx in loo.split(X):
        X_train_loo, X_test_loo = X.iloc[train_idx], X.iloc[test_idx]
        y_train_loo, y_test_loo = y.iloc[train_idx], y.iloc[test_idx]
        
        model_loo = RandomForestRegressor(**grid_search.best_params_, random_state=42, n_jobs=-1)
        model_loo.fit(X_train_loo, y_train_loo)
        
        pred = model_loo.predict(X_test_loo)
        loo_predictions.append(pred[0])
        loo_true.append(y_test_loo.values[0])
    
    loo_mse = mean_squared_error(loo_true, loo_predictions)
    loo_rmse = np.sqrt(loo_mse)
    loo_mae = mean_absolute_error(loo_true, loo_predictions)
    loo_r2 = r2_score(loo_true, loo_predictions)
    
    print(f"\nLOOCV性能:")
    print(f"  MSE:  {loo_mse:.4f}")
    print(f"  RMSE: {loo_rmse:.4f}")
    print(f"  MAE:  {loo_mae:.4f}")
    print(f"  R²:   {loo_r2:.4f}")
    
    return best_rf, np.array(loo_predictions), np.array(loo_true)


def train_xgboost_loocv(X, y, param_grid=None):
    """
    使用留一法交叉验证训练XGBoost（带正则化和早停）
    """
    print(f"\n{'='*60}")
    print("训练XGBoost模型 (LOOCV + 正则化)...")
    print(f"{'='*60}")
    
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'reg_alpha': [0.1, 1.0],
            'reg_lambda': [1.0, 2.0],
            'min_child_weight': [3, 5]
        }
    
    base_xgb = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        base_xgb, param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证MSE: {-grid_search.best_score_:.4f}")
    
    best_xgb = grid_search.best_estimator_
    
    print(f"\n执行留一法交叉验证 (LOOCV)...")
    loo_predictions = []
    loo_true = []
    
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X):
        X_train_loo, X_test_loo = X.iloc[train_idx], X.iloc[test_idx]
        y_train_loo, y_test_loo = y.iloc[train_idx], y.iloc[test_idx]
        
        X_train_inner, X_val, y_train_inner, y_val = train_test_split(
            X_train_loo, y_train_loo, test_size=0.1, random_state=42
        )
        
        model_loo = xgb.XGBRegressor(
            **grid_search.best_params_,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        model_loo.fit(
            X_train_inner, y_train_inner,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        pred = model_loo.predict(X_test_loo)
        loo_predictions.append(pred[0])
        loo_true.append(y_test_loo.values[0])
    
    loo_mse = mean_squared_error(loo_true, loo_predictions)
    loo_rmse = np.sqrt(loo_mse)
    loo_mae = mean_absolute_error(loo_true, loo_predictions)
    loo_r2 = r2_score(loo_true, loo_predictions)
    
    print(f"\nLOOCV性能:")
    print(f"  MSE:  {loo_mse:.4f}")
    print(f"  RMSE: {loo_rmse:.4f}")
    print(f"  MAE:  {loo_mae:.4f}")
    print(f"  R²:   {loo_r2:.4f}")
    
    return best_xgb, np.array(loo_predictions), np.array(loo_true)


def evaluate_model(model, X_test, y_test, model_name):
    """
    评估模型性能
    """
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name} 测试集性能:")
    print("-" * 40)
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    
    return y_pred, {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}


def plot_learning_curves(model, X, y, model_name):
    """
    绘制学习曲线，检测过拟合
    """
    from sklearn.model_selection import learning_curve
    
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42
    )
    
    train_scores_mean = -train_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)
    val_scores_mean = -val_scores.mean(axis=1)
    val_scores_std = val_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='训练误差')
    plt.plot(train_sizes, val_scores_mean, 'o-', color='g', label='验证误差')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1, color='g')
    plt.xlabel('训练样本数量')
    plt.ylabel('MSE')
    plt.title(f'{model_name} 学习曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'learning_curve_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"学习曲线已保存为: learning_curve_{model_name}.png")


def plot_results(y_test, y_pred_rf, y_pred_xgb, feature_names, rf_model, xgb_model,
                 loo_true_rf, loo_pred_rf, loo_true_xgb, loo_pred_xgb,
                 selected_features_rf, selected_features_xgb):
    """
    可视化结果
    """
    n_rows = 4
    n_cols = 2
    fig = plt.figure(figsize=(16, 20))
    
    # 1. LOOCV预测值 vs 真实值 - 随机森林
    ax1 = plt.subplot(n_rows, n_cols, 1)
    ax1.scatter(loo_true_rf, loo_pred_rf, alpha=0.6, s=40, edgecolors='k', linewidth=0.5)
    min_val = min(loo_true_rf.min(), loo_pred_rf.min())
    max_val = max(loo_true_rf.max(), loo_pred_rf.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax1.set_xlabel('真实 Tc (K)')
    ax1.set_ylabel('LOOCV预测 Tc (K)')
    ax1.set_title('随机森林: LOOCV预测值 vs 真实值')
    ax1.grid(True, alpha=0.3)
    
    # 2. LOOCV预测值 vs 真实值 - XGBoost
    if loo_pred_xgb is not None:
        ax2 = plt.subplot(n_rows, n_cols, 2)
        ax2.scatter(loo_true_xgb, loo_pred_xgb, alpha=0.6, s=40, color='orange', edgecolors='k', linewidth=0.5)
        min_val = min(loo_true_xgb.min(), loo_pred_xgb.min())
        max_val = max(loo_true_xgb.max(), loo_pred_xgb.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax2.set_xlabel('真实 Tc (K)')
        ax2.set_ylabel('LOOCV预测 Tc (K)')
        ax2.set_title('XGBoost: LOOCV预测值 vs 真实值')
        ax2.grid(True, alpha=0.3)
    
    # 3. 特征重要性 - 随机森林 (已选特征)
    if selected_features_rf:
        ax3 = plt.subplot(n_rows, n_cols, 3)
        rf_importances = rf_model.feature_importances_
        rf_feature_indices = [list(feature_names).index(f) for f in selected_features_rf if f in feature_names]
        rf_importances_selected = rf_importances[rf_feature_indices]
        rf_sorted_idx = np.argsort(rf_importances_selected)
        
        ax3.barh(range(len(rf_sorted_idx)), rf_importances_selected[rf_sorted_idx], color='skyblue')
        ax3.set_yticks(range(len(rf_sorted_idx)))
        ax3.set_yticklabels([selected_features_rf[i] for i in rf_sorted_idx])
        ax3.set_xlabel('重要性')
        ax3.set_title(f'随机森林: 选中特征重要性 (n={len(selected_features_rf)})')
    
    # 4. 特征重要性 - XGBoost (已选特征)
    if xgb_model is not None and selected_features_xgb:
        ax4 = plt.subplot(n_rows, n_cols, 4)
        xgb_importances = xgb_model.feature_importances_
        xgb_feature_indices = [list(feature_names).index(f) for f in selected_features_xgb if f in feature_names]
        xgb_importances_selected = xgb_importances[xgb_feature_indices]
        xgb_sorted_idx = np.argsort(xgb_importances_selected)
        
        ax4.barh(range(len(xgb_sorted_idx)), xgb_importances_selected[xgb_sorted_idx], color='peachpuff')
        ax4.set_yticks(range(len(xgb_sorted_idx)))
        ax4.set_yticklabels([selected_features_xgb[i] for i in xgb_sorted_idx])
        ax4.set_xlabel('重要性')
        ax4.set_title(f'XGBoost: 选中特征重要性 (n={len(selected_features_xgb)})')
    
    # 5. LOOCV残差图 - 随机森林
    ax5 = plt.subplot(n_rows, n_cols, 5)
    residuals_rf = loo_true_rf - loo_pred_rf
    ax5.scatter(loo_pred_rf, residuals_rf, alpha=0.6, s=40, edgecolors='k', linewidth=0.5)
    ax5.axhline(y=0, color='r', linestyle='--')
    ax5.set_xlabel('LOOCV预测 Tc (K)')
    ax5.set_ylabel('残差 (K)')
    ax5.set_title('随机森林: LOOCV残差图')
    ax5.grid(True, alpha=0.3)
    
    # 6. LOOCV残差图 - XGBoost
    if loo_pred_xgb is not None:
        ax6 = plt.subplot(n_rows, n_cols, 6)
        residuals_xgb = loo_true_xgb - loo_pred_xgb
        ax6.scatter(loo_pred_xgb, residuals_xgb, alpha=0.6, s=40, color='orange', edgecolors='k', linewidth=0.5)
        ax6.axhline(y=0, color='r', linestyle='--')
        ax6.set_xlabel('LOOCV预测 Tc (K)')
        ax6.set_ylabel('残差 (K)')
        ax6.set_title('XGBoost: LOOCV残差图')
        ax6.grid(True, alpha=0.3)
    
    # 7. Tc分布
    ax7 = plt.subplot(n_rows, n_cols, 7)
    ax7.hist(loo_true_rf, bins=30, alpha=0.7, label='真实分布', color='gray', edgecolor='black')
    ax7.hist(loo_pred_rf, bins=30, alpha=0.5, label='RF预测分布', color='blue')
    if loo_pred_xgb is not None:
        ax7.hist(loo_pred_xgb, bins=30, alpha=0.5, label='XGB预测分布', color='orange')
    ax7.set_xlabel('Tc (K)')
    ax7.set_ylabel('频数')
    ax7.set_title('Tc分布对比')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. 误差分布
    ax8 = plt.subplot(n_rows, n_cols, 8)
    ax8.hist(residuals_rf, bins=30, alpha=0.6, label='RF残差', color='blue', edgecolor='black')
    if loo_pred_xgb is not None:
        residuals_xgb = loo_true_xgb - loo_pred_xgb
        ax8.hist(residuals_xgb, bins=30, alpha=0.6, label='XGB残差', color='orange')
    ax8.axvline(x=0, color='r', linestyle='--')
    ax8.set_xlabel('残差 (K)')
    ax8.set_ylabel('频数')
    ax8.set_title('残差分布')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_results_loocv.png', dpi=300, bbox_inches='tight')
    print("\n结果图已保存为: model_results_loocv.png")


def main():
    """
    主函数 - 小样本优化版
    """
    print("="*70)
    print("超导临界温度Tc预测 - 小样本优化版")
    print("优化策略: LOOCV + SHAP特征选择 + 正则化")
    print("="*70)
    
    print("\n[1/6] 生成模拟超导材料数据集 (n~1000)...")
    df = generate_synthetic_data(n_samples=1000)
    print(f"数据集形状: {df.shape}")
    print(f"特征列: {list(df.columns[:-1])}")
    print(f"\nTc统计信息:")
    print(df['Tc'].describe())
    
    print("\n[2/6] 数据预处理...")
    X, y = preprocess_data(df)
    feature_names = X.columns
    print(f"预处理后特征数量: {X.shape[1]}")
    
    print("\n[3/6] 划分训练集和测试集 (测试集用于最终验证)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    print("\n[4/6] 训练基准模型用于特征选择...")
    rf_base = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    rf_base.fit(X_train, y_train)
    
    xgb_base = None
    if XGBOOST_AVAILABLE:
        xgb_base = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            reg_alpha=0.1, reg_lambda=1.0, min_child_weight=3,
            random_state=42, n_jobs=-1, verbosity=0
        )
        xgb_base.fit(X_train, y_train)
    
    print("\n[5/6] SHAP特征选择...")
    n_selected = 12
    selected_features_rf = shap_feature_selection(rf_base, X_train, y_train, n_features=n_selected, feature_names=feature_names)
    
    selected_features_xgb = None
    if xgb_base is not None:
        selected_features_xgb = shap_feature_selection(xgb_base, X_train, y_train, n_features=n_selected, feature_names=feature_names)
    
    X_train_rf_selected = X_train[selected_features_rf]
    X_test_rf_selected = X_test[selected_features_rf]
    
    X_train_xgb_selected = None
    X_test_xgb_selected = None
    if selected_features_xgb is not None:
        X_train_xgb_selected = X_train[selected_features_xgb]
        X_test_xgb_selected = X_test[selected_features_xgb]
    
    print("\n[6/6] 训练最终模型 (LOOCV + 正则化)...")
    rf_model, loo_pred_rf, loo_true_rf = train_random_forest_loocv(X_train_rf_selected, y_train)
    
    xgb_model = None
    loo_pred_xgb = None
    loo_true_xgb = None
    if XGBOOST_AVAILABLE and X_train_xgb_selected is not None:
        xgb_model, loo_pred_xgb, loo_true_xgb = train_xgboost_loocv(X_train_xgb_selected, y_train)
    
    print("\n" + "="*60)
    print("最终测试集评估")
    print("="*60)
    y_pred_rf, metrics_rf = evaluate_model(rf_model, X_test_rf_selected, y_test, "随机森林")
    
    y_pred_xgb = None
    metrics_xgb = None
    if xgb_model is not None:
        y_pred_xgb, metrics_xgb = evaluate_model(xgb_model, X_test_xgb_selected, y_test, "XGBoost")
    
    print("\n" + "="*60)
    print("模型性能对比")
    print("="*60)
    comparison = pd.DataFrame({
        '随机森林': metrics_rf,
    })
    if metrics_xgb is not None:
        comparison['XGBoost'] = metrics_xgb
    print(comparison.round(4))
    
    print("\n" + "="*60)
    print("生成可视化结果...")
    print("="*60)
    plot_results(y_test, y_pred_rf, y_pred_xgb, feature_names, rf_model, xgb_model,
                 loo_true_rf, loo_pred_rf, loo_true_xgb, loo_pred_xgb,
                 selected_features_rf, selected_features_xgb)
    
    print("\n绘制学习曲线 (检测过拟合)...")
    plot_learning_curves(rf_model, X_train_rf_selected, y_train, "RandomForest")
    if xgb_model is not None:
        plot_learning_curves(xgb_model, X_train_xgb_selected, y_train, "XGBoost")
    
    joblib.dump(rf_model, 'random_forest_model_optimized.pkl')
    joblib.dump(selected_features_rf, 'selected_features_rf.pkl')
    if xgb_model is not None:
        joblib.dump(xgb_model, 'xgboost_model_optimized.pkl')
        joblib.dump(selected_features_xgb, 'selected_features_xgb.pkl')
    print(f"\n模型已保存！")
    print(f"  - random_forest_model_optimized.pkl")
    print(f"  - selected_features_rf.pkl")
    if xgb_model is not None:
        print(f"  - xgboost_model_optimized.pkl")
        print(f"  - selected_features_xgb.pkl")
    
    print("\n" + "="*70)
    print("项目完成！")
    print("="*70)
    print("\n防止过拟合的关键优化:")
    print("  1. LOOCV: 留一法交叉验证，最大化利用小样本")
    print(f"  2. 特征选择: 从 {X.shape[1]} 个特征降至 {n_selected} 个")
    print("  3. 正则化: L1/L2正则化 + 子采样 + 树剪枝")
    print("  4. 早停: XGBoost使用验证集早停")
    print("  5. OOB估计: 随机森林使用袋外估计")


if __name__ == "__main__":
    main()
