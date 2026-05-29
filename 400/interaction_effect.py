import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor, _tree
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def calculate_vif(X):
    """
    计算方差膨胀因子(VIF)来检测多重共线性
    
    参数:
        X: 特征矩阵
    
    返回:
        dict: 各特征的VIF值
    """
    vif = {}
    for i in range(X.shape[1]):
        y_vif = X[:, i]
        X_vif = np.delete(X, i, axis=1)
        r_squared = sm.OLS(y_vif, X_vif).fit().rsquared
        if r_squared < 1:
            vif[i] = 1 / (1 - r_squared)
        else:
            vif[i] = float('inf')
    return vif


def detect_interaction_effect(x1, x2, y, significance_level=0.05, center=True):
    """
    检测线性模型中两个特征之间的交互效应
    通过中心化特征降低多重共线性，提高系数估计的稳定性
    
    参数:
        x1: 第一个特征数组
        x2: 第二个特征数组
        y: 因变量数组
        significance_level: 显著性水平，默认0.05
        center: 是否对特征进行中心化处理，默认True
    
    返回:
        dict: 包含交互效应检验结果的字典
    """
    x1 = np.asarray(x1).flatten()
    x2 = np.asarray(x2).flatten()
    y = np.asarray(y).flatten()
    
    if len(x1) != len(x2) or len(x1) != len(y):
        raise ValueError("x1, x2, y的长度必须相同")
    
    x1_mean = np.mean(x1)
    x2_mean = np.mean(x2)
    
    if center:
        x1_centered = x1 - x1_mean
        x2_centered = x2 - x2_mean
    else:
        x1_centered = x1.copy()
        x2_centered = x2.copy()
    
    interaction_term_raw = x1 * x2
    interaction_term_centered = x1_centered * x2_centered
    
    if center:
        x1_model = x1_centered
        x2_model = x2_centered
        interaction_model = interaction_term_centered
    else:
        x1_model = x1
        x2_model = x2
        interaction_model = interaction_term_raw
    
    X_raw = np.column_stack([x1, x2, interaction_term_raw])
    X_raw_with_const = sm.add_constant(X_raw)
    vif_raw = calculate_vif(X_raw_with_const)
    
    X = np.column_stack([x1_model, x2_model, interaction_model])
    X_with_const = sm.add_constant(X)
    vif_centered = calculate_vif(X_with_const)
    
    corr_x1_interaction_raw = np.corrcoef(x1, interaction_term_raw)[0, 1]
    corr_x2_interaction_raw = np.corrcoef(x2, interaction_term_raw)[0, 1]
    corr_x1_interaction_centered = np.corrcoef(x1_centered, interaction_term_centered)[0, 1]
    corr_x2_interaction_centered = np.corrcoef(x2_centered, interaction_term_centered)[0, 1]
    
    model = sm.OLS(y, X_with_const).fit()
    
    interaction_coef = model.params[3]
    interaction_pvalue = model.pvalues[3]
    interaction_tvalue = model.tvalues[3]
    
    has_interaction = interaction_pvalue < significance_level
    
    r2_full = model.rsquared
    
    X_reduced = np.column_stack([x1_model, x2_model])
    X_reduced = sm.add_constant(X_reduced)
    model_reduced = sm.OLS(y, X_reduced).fit()
    r2_reduced = model_reduced.rsquared
    
    r2_change = r2_full - r2_reduced
    
    f_stat, p_value_f, _ = model.compare_f_test(model_reduced)
    
    x1_std = np.std(x1, ddof=1)
    x2_std = np.std(x2, ddof=1)
    y_std = np.std(y, ddof=1)
    
    if y_std > 0 and x1_std > 0 and x2_std > 0:
        standardized_coef = interaction_coef * (x1_std * x2_std) / y_std
    else:
        standardized_coef = np.nan
    
    abs_coef = abs(interaction_coef)
    if abs_coef < 0.1:
        effect_size_label = "非常小"
    elif abs_coef < 0.3:
        effect_size_label = "小"
    elif abs_coef < 0.5:
        effect_size_label = "中等"
    else:
        effect_size_label = "大"
    
    condition_number = np.linalg.cond(X_with_const)
    
    return {
        "has_interaction": has_interaction,
        "interaction_coefficient": interaction_coef,
        "interaction_pvalue": interaction_pvalue,
        "interaction_tvalue": interaction_tvalue,
        "significance_level": significance_level,
        "centered": center,
        "x1_mean": x1_mean,
        "x2_mean": x2_mean,
        "r2_full_model": r2_full,
        "r2_reduced_model": r2_reduced,
        "r2_change": r2_change,
        "f_statistic": f_stat,
        "f_test_pvalue": p_value_f,
        "standardized_coefficient": standardized_coef,
        "effect_size_magnitude": effect_size_label,
        "vif_raw": {
            "const": vif_raw[0],
            "x1": vif_raw[1],
            "x2": vif_raw[2],
            "x1*x2": vif_raw[3]
        },
        "vif_centered": {
            "const": vif_centered[0],
            "x1_centered": vif_centered[1],
            "x2_centered": vif_centered[2],
            "x1*x2_centered": vif_centered[3]
        },
        "correlation_raw": {
            "x1_vs_interaction": corr_x1_interaction_raw,
            "x2_vs_interaction": corr_x2_interaction_raw
        },
        "correlation_centered": {
            "x1_vs_interaction": corr_x1_interaction_centered,
            "x2_vs_interaction": corr_x2_interaction_centered
        },
        "condition_number": condition_number,
        "model_summary": model.summary().as_text()
    }


def detect_interaction_effect_dataframe(df, x1_col, x2_col, y_col, significance_level=0.05, center=True):
    """
    从DataFrame中检测交互效应
    
    参数:
        df: pandas DataFrame
        x1_col: 第一个特征列名
        x2_col: 第二个特征列名
        y_col: 因变量列名
        significance_level: 显著性水平，默认0.05
        center: 是否对特征进行中心化处理，默认True
    
    返回:
        dict: 包含交互效应检验结果的字典
    """
    x1 = df[x1_col].values
    x2 = df[x2_col].values
    y = df[y_col].values
    
    result = detect_interaction_effect(x1, x2, y, significance_level, center)
    result["x1_column"] = x1_col
    result["x2_column"] = x2_col
    result["y_column"] = y_col
    
    return result


def print_interaction_report(result):
    """
    打印交互效应检验报告
    """
    print("=" * 70)
    print("交互效应检验报告")
    print("=" * 70)
    
    if "x1_column" in result:
        print(f"特征1: {result['x1_column']}")
        print(f"特征2: {result['x2_column']}")
        print(f"因变量: {result['y_column']}")
        print("-" * 70)
    
    print(f"是否中心化: {'是' if result['centered'] else '否'}")
    if result['centered']:
        print(f"x1均值: {result['x1_mean']:.6f}, x2均值: {result['x2_mean']:.6f}")
    print("-" * 70)
    
    print(f"交互项系数: {result['interaction_coefficient']:.6f}")
    print(f"标准化系数: {result['standardized_coefficient']:.6f}")
    print(f"效应强度: {result['effect_size_magnitude']}")
    print("-" * 70)
    print(f"t统计量: {result['interaction_tvalue']:.6f}")
    print(f"P值: {result['interaction_pvalue']:.6e}")
    print(f"显著性水平: {result['significance_level']}")
    print(f"存在显著交互效应: {'是' if result['has_interaction'] else '否'}")
    print("-" * 70)
    print(f"完整模型 R²: {result['r2_full_model']:.6f}")
    print(f"简化模型 R²: {result['r2_reduced_model']:.6f}")
    print(f"R² 变化量: {result['r2_change']:.6f}")
    print(f"F统计量: {result['f_statistic']:.6f}")
    print(f"F检验 P值: {result['f_test_pvalue']:.6e}")
    print("-" * 70)
    print("多重共线性诊断:")
    print(f"  条件数 (Condition Number): {result['condition_number']:.2f}")
    if result['condition_number'] < 10:
        print("    程度: 无明显共线性")
    elif result['condition_number'] < 30:
        print("    程度: 中等共线性")
    else:
        print("    程度: 严重共线性")
    print()
    print(f"  中心化前 VIF:")
    print(f"    x1: {result['vif_raw']['x1']:.2f}, x2: {result['vif_raw']['x2']:.2f}, x1*x2: {result['vif_raw']['x1*x2']:.2f}")
    print(f"  中心化后 VIF:")
    print(f"    x1_c: {result['vif_centered']['x1_centered']:.2f}, x2_c: {result['vif_centered']['x2_centered']:.2f}, x1*x2_c: {result['vif_centered']['x1*x2_centered']:.2f}")
    print()
    print(f"  中心化前相关系数:")
    print(f"    x1与x1*x2: {result['correlation_raw']['x1_vs_interaction']:.4f}")
    print(f"    x2与x1*x2: {result['correlation_raw']['x2_vs_interaction']:.4f}")
    print(f"  中心化后相关系数:")
    print(f"    x1_c与x1*x2_c: {result['correlation_centered']['x1_vs_interaction']:.4f}")
    print(f"    x2_c与x1*x2_c: {result['correlation_centered']['x2_vs_interaction']:.4f}")
    print("=" * 70)


def detect_interaction_tree_based(X, y, feature_names=None, n_estimators=100, max_depth=10, random_state=42):
    """
    基于树模型检测特征间的交互效应
    通过分析决策树的连续分裂特征对来估计交互强度
    
    参数:
        X: 特征矩阵 (n_samples, n_features)
        y: 因变量数组
        feature_names: 特征名称列表，可选
        n_estimators: 随机森林的树数量
        max_depth: 树的最大深度
        random_state: 随机种子
    
    返回:
        dict: 包含交互效应检测结果的字典
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn 未安装，请安装后使用此功能")
    
    X = np.asarray(X)
    y = np.asarray(y).flatten()
    
    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f'x{i+1}' for i in range(n_features)]
    
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                               random_state=random_state, oob_score=True)
    rf.fit(X, y)
    
    pair_counts = np.zeros((n_features, n_features))
    triple_counts = {}
    pair_gain = np.zeros((n_features, n_features))
    triple_gain = {}
    
    for tree in rf.estimators_:
        tree_ = tree.tree_
        feature_idx = tree_.feature
        threshold = tree_.threshold
        children_left = tree_.children_left
        children_right = tree_.children_right
        impurity = tree_.impurity
        n_node_samples = tree_.n_node_samples
        
        def traverse(node_id, depth, path_features, path_gains):
            if feature_idx[node_id] != _tree.TREE_UNDEFINED:
                current_feature = feature_idx[node_id]
                
                if len(path_features) >= 1:
                    prev_feature = path_features[-1]
                    f1, f2 = min(prev_feature, current_feature), max(prev_feature, current_feature)
                    pair_counts[f1, f2] += 1
                    pair_gain[f1, f2] += path_gains[-1] + (impurity[node_id] * n_node_samples[node_id]) / len(y)
                
                if len(path_features) >= 2:
                    f_list = path_features[-2:] + [current_feature]
                    f_unique = list(set(f_list))
                    if len(f_unique) >= 3:
                        f_sorted = sorted(f_unique)[:3]
                        triple_key = tuple(f_sorted)
                        triple_counts[triple_key] = triple_counts.get(triple_key, 0) + 1
                        total_gain = sum(path_gains[-2:]) + (impurity[node_id] * n_node_samples[node_id]) / len(y)
                        triple_gain[triple_key] = triple_gain.get(triple_key, 0) + total_gain
                
                new_path_features = path_features + [current_feature]
                new_gain = (impurity[node_id] - (impurity[children_left[node_id]] * n_node_samples[children_left[node_id]] + 
                                                  impurity[children_right[node_id]] * n_node_samples[children_right[node_id]]) / n_node_samples[node_id]) * n_node_samples[node_id] / len(y)
                new_path_gains = path_gains + [new_gain]
                
                traverse(children_left[node_id], depth + 1, new_path_features, new_path_gains)
                traverse(children_right[node_id], depth + 1, new_path_features, new_path_gains)
        
        traverse(0, 0, [], [])
    
    feature_importance = rf.feature_importances_
    
    interaction_matrix = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if pair_counts[i, j] > 0:
                base_importance = feature_importance[i] + feature_importance[j]
                interaction_strength = pair_gain[i, j] / pair_counts[i, j]
                interaction_matrix[i, j] = interaction_strength
                interaction_matrix[j, i] = interaction_strength
    
    interaction_normalized = interaction_matrix.copy()
    max_interaction = np.max(interaction_matrix) if np.max(interaction_matrix) > 0 else 1
    interaction_normalized = interaction_matrix / max_interaction
    
    pairwise_interactions = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if pair_counts[i, j] > 0:
                pairwise_interactions.append({
                    'feature_pair': (feature_names[i], feature_names[j]),
                    'cooccurrence_count': int(pair_counts[i, j]),
                    'interaction_strength': interaction_matrix[i, j],
                    'normalized_strength': interaction_normalized[i, j]
                })
    
    pairwise_interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
    
    triple_interactions = []
    for triple_key, count in triple_counts.items():
        if count > 0 and len(triple_key) == 3:
            f1, f2, f3 = triple_key
            triple_interactions.append({
                'feature_triple': (feature_names[f1], feature_names[f2], feature_names[f3]),
                'cooccurrence_count': count,
                'interaction_strength': triple_gain[triple_key] / count if count > 0 else 0
            })
    
    triple_interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
    
    return {
        'method': 'random_forest_split_analysis',
        'feature_names': feature_names,
        'feature_importance': dict(zip(feature_names, feature_importance)),
        'interaction_matrix': interaction_matrix,
        'interaction_matrix_normalized': interaction_normalized,
        'pairwise_interactions': pairwise_interactions,
        'triple_interactions': triple_interactions,
        'oob_score': rf.oob_score_,
        'model': rf
    }


def detect_three_way_interaction(x1, x2, x3, y, significance_level=0.05, center=True):
    """
    检测三因素交互效应 (x1 * x2 * x3)
    
    参数:
        x1, x2, x3: 三个特征数组
        y: 因变量数组
        significance_level: 显著性水平
        center: 是否中心化
    
    返回:
        dict: 三因素交互检验结果
    """
    x1 = np.asarray(x1).flatten()
    x2 = np.asarray(x2).flatten()
    x3 = np.asarray(x3).flatten()
    y = np.asarray(y).flatten()
    
    if center:
        x1 = x1 - np.mean(x1)
        x2 = x2 - np.mean(x2)
        x3 = x3 - np.mean(x3)
    
    two_way_12 = x1 * x2
    two_way_13 = x1 * x3
    two_way_23 = x2 * x3
    three_way = x1 * x2 * x3
    
    X_full = np.column_stack([x1, x2, x3, two_way_12, two_way_13, two_way_23, three_way])
    X_full = sm.add_constant(X_full)
    model_full = sm.OLS(y, X_full).fit()
    
    three_way_coef = model_full.params[7]
    three_way_pvalue = model_full.pvalues[7]
    three_way_tvalue = model_full.tvalues[7]
    
    X_reduced = np.column_stack([x1, x2, x3, two_way_12, two_way_13, two_way_23])
    X_reduced = sm.add_constant(X_reduced)
    model_reduced = sm.OLS(y, X_reduced).fit()
    
    f_stat, f_pvalue, _ = model_full.compare_f_test(model_reduced)
    
    r2_full = model_full.rsquared
    r2_reduced = model_reduced.rsquared
    r2_change = r2_full - r2_reduced
    
    has_three_way_interaction = three_way_pvalue < significance_level
    
    two_way_results = []
    for i, (name, idx) in enumerate([('x1*x2', 4), ('x1*x3', 5), ('x2*x3', 6)]):
        two_way_results.append({
            'interaction': name,
            'coefficient': model_full.params[idx],
            'pvalue': model_full.pvalues[idx],
            'significant': model_full.pvalues[idx] < significance_level
        })
    
    return {
        'has_three_way_interaction': has_three_way_interaction,
        'three_way_coefficient': three_way_coef,
        'three_way_pvalue': three_way_pvalue,
        'three_way_tvalue': three_way_tvalue,
        'significance_level': significance_level,
        'centered': center,
        'f_statistic': f_stat,
        'f_test_pvalue': f_pvalue,
        'r2_full_model': r2_full,
        'r2_reduced_model': r2_reduced,
        'r2_change': r2_change,
        'two_way_interactions': two_way_results,
        'model_summary': model_full.summary().as_text()
    }


def plot_interaction_heatmap(interaction_result, figsize=(10, 8), cmap='YlOrRd', 
                             annotate=True, save_path=None, show_plot=True):
    """
    绘制交互效应热力图
    
    参数:
        interaction_result: detect_interaction_tree_based 返回的结果
        figsize: 图大小
        cmap: 颜色映射
        annotate: 是否标注数值
        save_path: 保存路径，可选
        show_plot: 是否显示图
    
    返回:
        matplotlib.figure.Figure: 热力图对象
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib/seaborn 未安装，请安装后使用此功能")
    
    feature_names = interaction_result['feature_names']
    interaction_matrix = interaction_result['interaction_matrix_normalized']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    mask = np.zeros_like(interaction_matrix, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    
    sns.heatmap(interaction_matrix, 
                mask=mask,
                annot=annotate, 
                fmt='.3f',
                cmap=cmap,
                xticklabels=feature_names,
                yticklabels=feature_names,
                cbar_kws={'label': '标准化交互强度'},
                ax=ax,
                vmin=0,
                vmax=1)
    
    ax.set_title('特征交互效应热力图 (标准化)', fontsize=14, pad=20)
    ax.set_xlabel('特征', fontsize=12)
    ax.set_ylabel('特征', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    
    return fig


def print_tree_interaction_report(result, top_n=5):
    """
    打印树模型交互检测报告
    """
    print("=" * 70)
    print("树模型交互效应检测报告")
    print("=" * 70)
    print(f"方法: {result['method']}")
    print(f"OOB R² 得分: {result['oob_score']:.4f}")
    print("-" * 70)
    
    print("特征重要性:")
    for feat, imp in sorted(result['feature_importance'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat}: {imp:.4f}")
    print("-" * 70)
    
    print(f"Top {top_n} 二阶交互效应:")
    for i, inter in enumerate(result['pairwise_interactions'][:top_n]):
        feat1, feat2 = inter['feature_pair']
        print(f"  {i+1}. {feat1} × {feat2}:")
        print(f"     共现次数: {inter['cooccurrence_count']}, 强度: {inter['interaction_strength']:.6f}")
    print("-" * 70)
    
    if result['triple_interactions']:
        print(f"Top {top_n} 三阶交互效应:")
        for i, inter in enumerate(result['triple_interactions'][:top_n]):
            f1, f2, f3 = inter['feature_triple']
            print(f"  {i+1}. {f1} × {f2} × {f3}:")
            print(f"     共现次数: {inter['cooccurrence_count']}, 强度: {inter['interaction_strength']:.6f}")
    print("=" * 70)


def print_three_way_interaction_report(result):
    """
    打印三因素交互效应报告
    """
    print("=" * 70)
    print("三因素交互效应检验报告")
    print("=" * 70)
    print(f"是否中心化: {'是' if result['centered'] else '否'}")
    print("-" * 70)
    print(f"三阶交互项系数 (x1*x2*x3): {result['three_way_coefficient']:.6f}")
    print(f"t统计量: {result['three_way_tvalue']:.6f}")
    print(f"P值: {result['three_way_pvalue']:.6e}")
    print(f"存在显著三阶交互: {'是' if result['has_three_way_interaction'] else '否'}")
    print("-" * 70)
    print(f"完整模型 R²: {result['r2_full_model']:.6f}")
    print(f"简化模型 R²: {result['r2_reduced_model']:.6f}")
    print(f"R² 变化量: {result['r2_change']:.6f}")
    print(f"F统计量: {result['f_statistic']:.6f}")
    print(f"F检验 P值: {result['f_test_pvalue']:.6e}")
    print("-" * 70)
    print("二阶交互项检验结果:")
    for inter in result['two_way_interactions']:
        print(f"  {inter['interaction']}: coef={inter['coefficient']:.6f}, "
              f"p={inter['pvalue']:.4f}, 显著={'是' if inter['significant'] else '否'}")
    print("=" * 70)


if __name__ == "__main__":
    np.random.seed(42)
    n = 300
    
    print("=" * 70)
    print("示例1: 线性模型 - 二阶交互效应检测")
    print("=" * 70)
    
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    
    true_interaction = 0.5
    y = 2 + 1.5 * x1 + 1.0 * x2 + true_interaction * x1 * x2 + np.random.normal(0, 0.5, n)
    
    result = detect_interaction_effect(x1, x2, y)
    print_interaction_report(result)
    
    print("\n" + "=" * 70)
    print("示例2: 线性模型 - 三因素交互效应检测")
    print("=" * 70)
    
    x3 = np.random.normal(0, 1, n)
    y_3way = 2 + 1.0 * x1 + 1.0 * x2 + 1.0 * x3 + 0.3 * x1 * x2 + 0.2 * x1 * x2 * x3 + np.random.normal(0, 0.5, n)
    
    result_3way = detect_three_way_interaction(x1, x2, x3, y_3way)
    print_three_way_interaction_report(result_3way)
    
    print("\n" + "=" * 70)
    print("示例3: 树模型 - 多特征交互效应检测")
    print("=" * 70)
    
    x4 = np.random.normal(0, 1, n)
    X_multi = np.column_stack([x1, x2, x3, x4])
    
    y_tree = (2 + 1.5 * x1 + 1.0 * x2 + 0.8 * x3 + 0.5 * x4 
              + 0.6 * x1 * x2 + 0.4 * x2 * x3 + 0.3 * x1 * x2 * x3
              + np.random.normal(0, 0.3, n))
    
    feature_names = ['温度', '湿度', '风速', '气压']
    tree_result = detect_interaction_tree_based(X_multi, y_tree, feature_names=feature_names, 
                                                 n_estimators=100, max_depth=8)
    print_tree_interaction_report(tree_result, top_n=5)
    
    print("\n" + "=" * 70)
    print("示例4: 生成交互效应热力图")
    print("=" * 70)
    
    if MATPLOTLIB_AVAILABLE:
        try:
            heatmap_path = 'interaction_heatmap.png'
            plot_interaction_heatmap(tree_result, figsize=(10, 8), 
                                     save_path=heatmap_path, show_plot=False)
            print(f"热力图已保存至: {heatmap_path}")
            print("热力图显示各特征对之间的标准化交互强度 (0-1)")
        except Exception as e:
            print(f"生成热力图时出错: {e}")
    else:
        print("matplotlib/seaborn 未安装，跳过热力图生成")
    
    print("\n" + "=" * 70)
    print("示例5: 中心化改善多重共线性验证")
    print("=" * 70)
    
    x1_nonzero = np.random.normal(5, 1, n)
    x2_nonzero = np.random.normal(3, 1, n)
    y_nonzero = 2 + 1.5 * x1_nonzero + 1.0 * x2_nonzero + true_interaction * x1_nonzero * x2_nonzero + np.random.normal(0, 0.5, n)
    
    print("\n--- 非中心化结果 ---")
    result_no_center = detect_interaction_effect(x1_nonzero, x2_nonzero, y_nonzero, center=False)
    print(f"条件数: {result_no_center['condition_number']:.2f} (严重共线性: >30)")
    print(f"交互项 VIF: {result_no_center['vif_raw']['x1*x2']:.2f}")
    
    print("\n--- 中心化结果 ---")
    result_centered = detect_interaction_effect(x1_nonzero, x2_nonzero, y_nonzero, center=True)
    print(f"条件数: {result_centered['condition_number']:.2f}")
    print(f"交互项 VIF: {result_centered['vif_centered']['x1*x2_centered']:.2f}")
    
    print(f"\n>>> 条件数改善率: {(1 - result_centered['condition_number']/result_no_center['condition_number']) * 100:.1f}%")
    
    print("\n" + "=" * 70)
    print("所有功能演示完成!")
    print("=" * 70)
