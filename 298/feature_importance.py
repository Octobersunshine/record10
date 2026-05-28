import warnings
import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, List, Tuple
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance, partial_dependence

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.inspection import PartialDependenceDisplay
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


TreeModel = Union[
    RandomForestClassifier, RandomForestRegressor,
    'xgb.XGBClassifier', 'xgb.XGBRegressor'
]


def gini_importance(model: TreeModel, feature_names: list) -> pd.DataFrame:
    warnings.warn(
        "Gini (impurity-based) importance is biased towards high-cardinality "
        "features (continuous or high-cardinality categorical features). "
        "Consider using method='permutation' or method='shap' for a fairer "
        "assessment of feature importance.",
        UserWarning,
        stacklevel=2
    )
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        raise ValueError("Model does not have feature_importances_ attribute.")

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    return importance_df


def permutation_importance_calc(
    model: TreeModel,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    feature_names: list,
    n_repeats: int = 10,
    random_state: int = 42,
    scoring: Optional[str] = None
) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = X

    result = permutation_importance(
        model, X_values, y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    })
    importance_df = importance_df.sort_values('importance_mean', ascending=False).reset_index(drop=True)
    return importance_df


def _resolve_shap_values(shap_values) -> np.ndarray:
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            return np.array(shap_values[1])
        return np.mean(np.abs(np.stack(shap_values, axis=-1)), axis=-1)
    return np.asarray(shap_values)


def shap_importance(
    model: TreeModel,
    X: Union[pd.DataFrame, np.ndarray],
    feature_names: list,
    approximate: bool = False,
    check_additivity: bool = True
) -> pd.DataFrame:
    if not SHAP_AVAILABLE:
        raise ImportError(
            "shap package is required for SHAP importance. "
            "Install it with: pip install shap"
        )

    if isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = np.array(X, dtype=np.float64)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(
        X_values,
        approximate=approximate,
        check_additivity=check_additivity
    )

    shap_values = _resolve_shap_values(shap_values)

    if shap_values.ndim == 3:
        mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 2))
        std_abs_shap = np.std(np.abs(shap_values), axis=(0, 2))
    elif shap_values.ndim == 2:
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        std_abs_shap = np.std(np.abs(shap_values), axis=0)
    elif shap_values.ndim == 1:
        mean_abs_shap = np.abs(shap_values)
        std_abs_shap = np.zeros_like(mean_abs_shap)
    else:
        raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")

    mean_abs_shap = np.asarray(mean_abs_shap).ravel()
    std_abs_shap = np.asarray(std_abs_shap).ravel()

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': mean_abs_shap,
        'importance_std': std_abs_shap
    })
    importance_df = importance_df.sort_values('importance_mean', ascending=False).reset_index(drop=True)
    return importance_df


def detect_cardinality_bias(
    X: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[list] = None,
    gini_importance_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = list(X.columns)
        X_values = X.values
    else:
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_values = np.asarray(X)

    cardinalities = []
    for i in range(X_values.shape[1]):
        col = X_values[:, i]
        try:
            cardinalities.append(len(np.unique(col[~pd.isna(col)])))
        except TypeError:
            cardinalities.append(len(set(col)))

    bias_df = pd.DataFrame({
        'feature': feature_names,
        'n_unique': cardinalities
    })

    if gini_importance_df is not None:
        gini_map = dict(zip(gini_importance_df['feature'], gini_importance_df['importance']))
        bias_df['gini_importance'] = bias_df['feature'].map(gini_map)
        valid = bias_df.dropna(subset=['gini_importance'])
        if len(valid) > 2:
            corr = np.corrcoef(
                np.log1p(valid['n_unique'].values),
                valid['gini_importance'].values
            )[0, 1]
            bias_df.attrs['cardinality_gini_correlation'] = corr
            if abs(corr) > 0.5:
                warnings.warn(
                    f"Strong correlation (r={corr:.3f}) detected between feature "
                    f"cardinality and Gini importance. This indicates high-cardinality "
                    f"bias. Use method='permutation' or method='shap' for fairer results.",
                    UserWarning,
                    stacklevel=2
                )

    return bias_df.sort_values('n_unique', ascending=False).reset_index(drop=True)


def compare_methods(
    model: TreeModel,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    feature_names: Optional[list] = None,
    methods: Optional[List[str]] = None,
    n_repeats: int = 10,
    random_state: int = 42,
    scoring: Optional[str] = None
) -> pd.DataFrame:
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    if methods is None:
        methods = ['gini', 'permutation']
        if SHAP_AVAILABLE:
            methods.append('shap')

    results: Dict[str, pd.Series] = {}

    for method in methods:
        method_lower = method.lower()
        if method_lower == 'gini':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = gini_importance(model, feature_names)
            col_name = 'gini_importance'
            results[col_name] = df.set_index('feature')['importance']
        elif method_lower == 'permutation':
            df = permutation_importance_calc(
                model, X, y, feature_names, n_repeats, random_state, scoring
            )
            col_name = 'permutation_importance'
            results[col_name] = df.set_index('feature')['importance_mean']
        elif method_lower == 'shap':
            if not SHAP_AVAILABLE:
                warnings.warn(
                    "shap not installed, skipping SHAP method.",
                    UserWarning,
                    stacklevel=2
                )
                continue
            df = shap_importance(model, X, feature_names)
            col_name = 'shap_importance'
            results[col_name] = df.set_index('feature')['importance_mean']
        else:
            raise ValueError(f"Unknown method: {method}")

    comparison = pd.DataFrame(results)
    for col in comparison.columns:
        comparison[f'{col}_rank'] = comparison[col].rank(ascending=False).astype(int)

    comparison = comparison.sort_values(
        comparison.columns[0], ascending=False
    ).reset_index().rename(columns={'index': 'feature'} if 'feature' not in comparison.columns else {})

    if 'feature' not in comparison.columns:
        comparison = comparison.reset_index().rename(columns={'index': 'feature'})

    bias_cols = [c for c in comparison.columns if c.endswith('_rank')]
    if len(bias_cols) >= 2:
        rank_cols = [comparison[c] for c in bias_cols]
        rank_diffs = []
        for i in range(len(rank_cols)):
            for j in range(i + 1, len(rank_cols)):
                rank_diffs.append(np.abs(rank_cols[i] - rank_cols[j]))
        avg_rank_shift = np.mean(rank_diffs)
        comparison.attrs['avg_rank_shift'] = avg_rank_shift
        if avg_rank_shift > 3.0:
            warnings.warn(
                f"Large average rank shift ({avg_rank_shift:.1f}) detected between "
                f"methods. This suggests impurity-based importance may be biased. "
                f"Trust permutation/SHAP results over Gini.",
                UserWarning,
                stacklevel=2
            )

    return comparison


def compute_pdp(
    model: TreeModel,
    X: Union[pd.DataFrame, np.ndarray],
    feature_index: Union[int, str],
    feature_names: Optional[list] = None,
    grid_resolution: int = 50,
    percentiles: Tuple[float, float] = (0.05, 0.95)
) -> Dict:
    if isinstance(feature_index, str):
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = list(X.columns)
            else:
                raise ValueError("feature_names must be provided when feature_index is a string and X is ndarray.")
        feature_index = feature_names.index(feature_index)

    X_input = X if isinstance(X, pd.DataFrame) else np.asarray(X)

    result = partial_dependence(
        model, X_input,
        features=[feature_index],
        kind='average',
        grid_resolution=grid_resolution,
        percentiles=percentiles
    )

    avg = result['average']
    if avg.ndim == 2 and avg.shape[0] == 1:
        avg = avg[0]
    elif avg.ndim == 3:
        avg = np.mean(avg, axis=-1)[0] if avg.shape[2] > 1 else avg[:, :, 0][0]

    return {
        'grid_values': result['grid_values'][0],
        'average': avg,
        'feature_index': feature_index
    }


def compute_ice(
    model: TreeModel,
    X: Union[pd.DataFrame, np.ndarray],
    feature_index: Union[int, str],
    feature_names: Optional[list] = None,
    grid_resolution: int = 50,
    percentiles: Tuple[float, float] = (0.05, 0.95)
) -> Dict:
    if isinstance(feature_index, str):
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = list(X.columns)
            else:
                raise ValueError("feature_names must be provided when feature_index is a string and X is ndarray.")
        feature_index = feature_names.index(feature_index)

    X_input = X if isinstance(X, pd.DataFrame) else np.asarray(X)

    result = partial_dependence(
        model, X_input,
        features=[feature_index],
        kind='individual',
        grid_resolution=grid_resolution,
        percentiles=percentiles
    )

    indiv = result['individual']
    if indiv.ndim == 3 and indiv.shape[0] == 1:
        indiv = indiv[0]
    elif indiv.ndim == 4:
        indiv = np.mean(indiv, axis=-1)[0] if indiv.shape[3] > 1 else indiv[:, :, :, 0][0]

    return {
        'grid_values': result['grid_values'][0],
        'individual': indiv,
        'feature_index': feature_index
    }


def plot_pdp_ice(
    model: TreeModel,
    X: Union[pd.DataFrame, np.ndarray],
    feature_index: Union[int, str],
    feature_names: Optional[list] = None,
    kind: str = 'both',
    n_ice_samples: int = 50,
    ice_alpha: float = 0.1,
    ice_color: str = 'steelblue',
    pdp_color: str = 'red',
    pdp_linewidth: int = 2,
    grid_resolution: int = 50,
    percentiles: Tuple[float, float] = (0.05, 0.95),
    figsize: Tuple[float, float] = (8, 5),
    title: Optional[str] = None,
    save_path: Optional[str] = None
):
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

    if isinstance(feature_index, str):
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = list(X.columns)
            else:
                raise ValueError("feature_names must be provided when feature_index is a string and X is ndarray.")
        feature_name = feature_index
        feature_index = feature_names.index(feature_index)
    else:
        if feature_names is not None and feature_index < len(feature_names):
            feature_name = feature_names[feature_index]
        elif isinstance(X, pd.DataFrame):
            feature_name = list(X.columns)[feature_index]
        else:
            feature_name = f'feature_{feature_index}'

    X_input = X if isinstance(X, pd.DataFrame) else np.asarray(X)

    if kind in ('both', 'ice'):
        ice_result = compute_ice(
            model, X_input, feature_index, feature_names,
            grid_resolution, percentiles
        )
        grid = ice_result['grid_values']
        ice_lines = ice_result['individual']

        if n_ice_samples < ice_lines.shape[0]:
            rng = np.random.RandomState(42)
            idx = rng.choice(ice_lines.shape[0], size=n_ice_samples, replace=False)
            ice_lines = ice_lines[idx]

        pdp_line = np.mean(ice_result['individual'], axis=0)
    else:
        pdp_result = compute_pdp(
            model, X_input, feature_index, feature_names,
            grid_resolution, percentiles
        )
        grid = pdp_result['grid_values']
        pdp_line = pdp_result['average']
        ice_lines = None

    fig, ax = plt.subplots(figsize=figsize)

    if ice_lines is not None:
        for i in range(ice_lines.shape[0]):
            ax.plot(grid, ice_lines[i], color=ice_color, alpha=ice_alpha, linewidth=0.8)
        ax.plot([], [], color=ice_color, alpha=0.4, linewidth=0.8, label='ICE curves')

    ax.plot(grid, pdp_line, color=pdp_color, linewidth=pdp_linewidth, label='PDP (average)')

    ax.set_xlabel(feature_name)
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'PDP + ICE: {feature_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.close(fig)

    return fig, ax


def plot_top_features_pdp(
    model: TreeModel,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    feature_names: Optional[list] = None,
    top_k: int = 4,
    method: str = 'permutation',
    kind: str = 'both',
    n_ice_samples: int = 30,
    ice_alpha: float = 0.1,
    grid_resolution: int = 50,
    ncols: int = 2,
    figsize_per_plot: Tuple[float, float] = (6, 4),
    save_path: Optional[str] = None
):
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imp_df = calculate_feature_importance(
            model, X, y, method=method, feature_names=feature_names
        )

    top_features = imp_df['feature'].head(top_k).tolist()

    nrows = int(np.ceil(top_k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * figsize_per_plot[0], nrows * figsize_per_plot[1]))
    if top_k == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    X_input = X if isinstance(X, pd.DataFrame) else np.asarray(X)

    for i, feat_name in enumerate(top_features):
        ax = axes[i]
        feat_idx = feature_names.index(feat_name)

        if kind in ('both', 'ice'):
            ice_result = compute_ice(
                model, X_input, feat_idx, feature_names, grid_resolution
            )
            grid = ice_result['grid_values']
            ice_lines = ice_result['individual']
            if n_ice_samples < ice_lines.shape[0]:
                rng = np.random.RandomState(42)
                idx = rng.choice(ice_lines.shape[0], size=n_ice_samples, replace=False)
                ice_lines = ice_lines[idx]
            pdp_line = np.mean(ice_result['individual'], axis=0)

            for j in range(ice_lines.shape[0]):
                ax.plot(grid, ice_lines[j], color='steelblue', alpha=ice_alpha, linewidth=0.6)
            ax.plot([], [], color='steelblue', alpha=0.4, linewidth=0.6, label='ICE')
        else:
            pdp_result = compute_pdp(
                model, X_input, feat_idx, feature_names, grid_resolution
            )
            grid = pdp_result['grid_values']
            pdp_line = pdp_result['average']

        ax.plot(grid, pdp_line, color='red', linewidth=2, label='PDP')
        ax.set_xlabel(feat_name)
        ax.set_title(f'{feat_name} (rank {i + 1})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(top_k, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f'Top-{top_k} Features PDP + ICE (method={method})', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.close(fig)

    return fig, axes


def plot_2d_pdp(
    model: TreeModel,
    X: Union[pd.DataFrame, np.ndarray],
    feature_pair: Tuple[Union[int, str], Union[int, str]],
    feature_names: Optional[list] = None,
    grid_resolution: int = 30,
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None
):
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    indices = []
    names = []
    for feat in feature_pair:
        if isinstance(feat, str):
            names.append(feat)
            indices.append(feature_names.index(feat))
        else:
            indices.append(feat)
            names.append(feature_names[feat] if feat < len(feature_names) else f'feature_{feat}')

    X_input = X if isinstance(X, pd.DataFrame) else np.asarray(X)

    result = partial_dependence(
        model, X_input,
        features=indices,
        kind='average',
        grid_resolution=grid_resolution
    )

    fig, ax = plt.subplots(figsize=figsize)

    xx = result['grid_values'][0]
    yy = result['grid_values'][1]
    zz = result['average'][0]

    if zz.ndim == 3:
        zz = zz[:, :, 0] if zz.shape[2] == 1 else zz[:, :, -1]

    cm = ax.contourf(xx, yy, zz, levels=20, cmap='viridis')
    fig.colorbar(cm, ax=ax, label='Predicted response')

    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'2D PDP: {names[0]} x {names[1]}')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.close(fig)

    return fig, ax


def calculate_feature_importance(
    model: TreeModel,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    method: str = 'permutation',
    feature_names: Optional[list] = None,
    n_repeats: int = 10,
    random_state: int = 42,
    scoring: Optional[str] = None,
    approximate: bool = False,
    check_additivity: bool = True
) -> pd.DataFrame:
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    method = method.lower()
    if method == 'gini':
        return gini_importance(model, feature_names)
    elif method == 'permutation':
        return permutation_importance_calc(
            model, X, y, feature_names, n_repeats, random_state, scoring
        )
    elif method == 'shap':
        return shap_importance(
            model, X, feature_names, approximate, check_additivity
        )
    else:
        raise ValueError("method must be one of: 'gini', 'permutation', 'shap'")


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    print("=" * 70)
    print("[1] Cardinality Bias Detection")
    print("=" * 70)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gini_imp = calculate_feature_importance(
            rf_model, X_test, y_test, method='gini', feature_names=feature_names
        )
    bias_df = detect_cardinality_bias(X_test, feature_names, gini_imp)
    print(bias_df.to_string(index=False))
    if 'cardinality_gini_correlation' in bias_df.attrs:
        print(f"\nCardinality-Gini correlation: {bias_df.attrs['cardinality_gini_correlation']:.3f}")

    print("\n" + "=" * 70)
    print("[2] Gini Importance (biased towards high-cardinality features)")
    print("=" * 70)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gini_imp = calculate_feature_importance(
            rf_model, X_test, y_test, method='gini', feature_names=feature_names
        )
        if w:
            print(f"[WARNING] {w[0].message}\n")
    print(gini_imp.head(10))

    print("\n" + "=" * 70)
    print("[3] Permutation Importance (fairer assessment)")
    print("=" * 70)
    perm_imp = calculate_feature_importance(
        rf_model, X_test, y_test, method='permutation',
        feature_names=feature_names, n_repeats=5
    )
    print(perm_imp.head(10))

    if SHAP_AVAILABLE:
        print("\n" + "=" * 70)
        print("[4] SHAP Importance (most fair, game-theory based)")
        print("=" * 70)
        shap_imp = calculate_feature_importance(
            rf_model, X_test, y_test, method='shap', feature_names=feature_names
        )
        print(shap_imp.head(10))

    print("\n" + "=" * 70)
    print("[5] Cross-Method Comparison (rank shift = bias indicator)")
    print("=" * 70)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        comparison = compare_methods(
            rf_model, X_test, y_test, feature_names=feature_names, n_repeats=5
        )
        for warning in w:
            print(f"[WARNING] {warning.message}\n")
    print(comparison.to_string(index=False))
    if 'avg_rank_shift' in comparison.attrs:
        print(f"\nAverage rank shift: {comparison.attrs['avg_rank_shift']:.1f}")

    if XGBOOST_AVAILABLE:
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, random_state=42, eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)

        print("\n" + "=" * 70)
        print("[6] XGBoost - Permutation Importance")
        print("=" * 70)
        xgb_perm_imp = calculate_feature_importance(
            xgb_model, X_test, y_test, method='permutation',
            feature_names=feature_names, n_repeats=5
        )
        print(xgb_perm_imp.head(10))

        if SHAP_AVAILABLE:
            print("\n" + "=" * 70)
            print("[7] XGBoost - SHAP Importance")
            print("=" * 70)
            xgb_shap_imp = calculate_feature_importance(
                xgb_model, X_test, y_test, method='shap',
                feature_names=feature_names
            )
            print(xgb_shap_imp.head(10))
    else:
        print("\n[INFO] xgboost not installed. Install with: pip install xgboost")

    print("\n" + "=" * 70)
    print("[8] PDP + ICE Visualization (Feature Effect Analysis)")
    print("=" * 70)
    if MATPLOTLIB_AVAILABLE:
        top_feature = gini_imp['feature'].iloc[0]
        print(f"Generating PDP + ICE for top feature: {top_feature}")

        import os
        output_dir = os.path.join(os.path.dirname(__file__), 'plots')
        os.makedirs(output_dir, exist_ok=True)

        single_pdp_path = os.path.join(output_dir, 'pdp_ice_single.png')
        fig1, _ = plot_pdp_ice(
            rf_model, X_test, top_feature, feature_names,
            kind='both', save_path=single_pdp_path
        )
        print(f"  Single PDP+ICE saved to: {single_pdp_path}")

        top4_pdp_path = os.path.join(output_dir, 'pdp_ice_top4.png')
        fig2, _ = plot_top_features_pdp(
            rf_model, X_test, y_test, feature_names,
            top_k=4, method='shap' if SHAP_AVAILABLE else 'permutation',
            save_path=top4_pdp_path
        )
        print(f"  Top-4 features PDP+ICE saved to: {top4_pdp_path}")

        try:
            two_features = (gini_imp['feature'].iloc[0], gini_imp['feature'].iloc[1])
            pdp2d_path = os.path.join(output_dir, 'pdp_2d.png')
            fig3, _ = plot_2d_pdp(
                rf_model, X_test, two_features, feature_names,
                save_path=pdp2d_path
            )
            print(f"  2D PDP ({two_features[0]} x {two_features[1]}) saved to: {pdp2d_path}")
        except Exception as e:
            print(f"  2D PDP skipped: {e}")

        print("\n[HINT] PDP shows AVERAGE marginal effect of a feature on predictions.")
        print("  ICE shows INDIVIDUAL effect curves (one line per sample).")
        print("  - Upward slope = feature increases prediction as value rises")
        print("  - Downward slope = feature decreases prediction as value rises")
        print("  - Flat line = feature has little/no effect")
        print("  - Crossing ICE lines = interaction effects may exist")
    else:
        print("[INFO] matplotlib not installed. Install with: pip install matplotlib")
