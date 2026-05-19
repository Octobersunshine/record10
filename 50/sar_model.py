import numpy as np
import warnings
import libpysal
from spreg import ML_Lag, ML_Error, OLS
from scipy import stats


def row_standardize(W_array):
    """
    对权重矩阵进行行标准化
    
    参数:
        W_array: numpy数组形式的权重矩阵
    
    返回:
        W_row_std: 行标准化后的权重矩阵
    """
    W_row_std = W_array.copy().astype(float)
    row_sums = W_row_std.sum(axis=1)
    
    zero_rows = np.where(row_sums == 0)[0]
    if len(zero_rows) > 0:
        warnings.warn(f"警告: 有 {len(zero_rows)} 行的行和为0（孤岛观测值），这些行无法标准化。")
    
    for i in range(W_row_std.shape[0]):
        if row_sums[i] != 0:
            W_row_std[i, :] = W_row_std[i, :] / row_sums[i]
    
    return W_row_std


def check_row_standardized(W_array, tol=1e-10):
    """
    检查权重矩阵是否已行标准化
    
    参数:
        W_array: numpy数组形式的权重矩阵
        tol: 容差
    
    返回:
        is_std: 是否已行标准化
    """
    row_sums = W_array.sum(axis=1)
    
    non_zero_rows = row_sums != 0
    if not np.any(non_zero_rows):
        return False
    
    return np.allclose(row_sums[non_zero_rows], 1.0, atol=tol)


def fit_sar_model(W, y, X=None, auto_standardize=True, verbose=True):
    """
    拟合空间自回归模型（SAR）：y = ρWy + Xβ + ε
    
    为确保空间自回归系数ρ具有可比性（不依赖权重尺度），权重矩阵必须进行行标准化。
    行标准化后，Wy表示邻居的平均值，ρ的解释更直观且跨模型可比。
    
    参数:
        W: 空间权重矩阵，可以是libpysal.weights.W对象或numpy数组
        y: 因变量，形状为(n, 1)或(n,)的numpy数组
        X: 自变量矩阵，形状为(n, k)的numpy数组，默认为None（只含截距项）
        auto_standardize: 是否自动进行行标准化（默认True，强烈建议保留）
        verbose: 是否打印标准化信息
    
    返回:
        results: 模型拟合结果对象，附加W标准化信息
    """
    is_standardized = False
    original_transform = None
    
    if isinstance(W, np.ndarray):
        W_array = W.astype(float)
        is_standardized = check_row_standardized(W_array)
        
        if auto_standardize and not is_standardized:
            if verbose:
                print(f"权重矩阵未行标准化，正在进行行标准化...")
                print(f"  行和范围: [{W_array.sum(axis=1).min():.2f}, {W_array.sum(axis=1).max():.2f}]")
            W_array = row_standardize(W_array)
            is_standardized = True
        
        W = libpysal.weights.W.from_array(W_array)
        
    elif isinstance(W, libpysal.weights.W):
        original_transform = W.transform
        
        W_temp = W.full()[0]
        is_standardized = check_row_standardized(W_temp)
        
        if auto_standardize and W.transform != 'r':
            if verbose:
                if not is_standardized:
                    print(f"权重矩阵未行标准化，正在进行行标准化...")
                else:
                    print(f"设置权重矩阵变换为行标准化('r')")
            W.transform = 'r'
            is_standardized = True
    else:
        raise ValueError("W必须是libpysal.weights.W对象或numpy数组")
    
    y = np.asarray(y).reshape(-1, 1)
    
    if X is None:
        X = np.ones((y.shape[0], 1))
    else:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X和y的样本数必须匹配")
    
    model = ML_Lag(y, X, W, name_y='y', name_x=['x{}'.format(i) for i in range(X.shape[1])])
    
    model.was_standardized = is_standardized
    model.original_transform = original_transform
    model.w_transform = W.transform
    
    return model


def print_sar_results(results):
    """
    打印SAR模型的结果
    """
    print("=" * 60)
    print("空间自回归模型（SAR）拟合结果")
    print("=" * 60)
    print("\n模型形式: y = ρWy + Xβ + ε")
    print("\n样本数:", results.n)
    print("自变量个数:", results.k)
    
    w_status = "✓ 已行标准化" if getattr(results, 'was_standardized', False) else "⚠ 未行标准化"
    print(f"权重矩阵状态: {w_status}")
    print(f"权重变换方式: {getattr(results, 'w_transform', 'unknown')}")
    print("=" * 60)
    
    print("\n系数估计:")
    print("-" * 40)
    print(f"ρ (空间自回归系数): {results.rho:.6f}")
    print(f"ρ的标准误: {results.se_rho:.6f}")
    print(f"ρ的z值: {results.z_rho:.6f}")
    print(f"ρ的p值: {results.p_rho:.6f}")
    print()
    
    print("β系数:")
    for i, (beta, se, z, p) in enumerate(zip(results.betas, results.std_err, results.z_stat, results.pvalues)):
        if i < len(results.betas) - 1:
            print(f"  β_{i}: {beta[0]:.6f} (se={se:.6f}, z={z:.6f}, p={p:.6f})")
    
    print("\n模型拟合优度:")
    print("-" * 40)
    print(f"对数似然值: {results.logll:.6f}")
    print(f"AIC: {results.aic:.6f}")
    print(f"SIC: {results.sic:.6f}")
    print(f"伪R²: {results.pr2:.6f}")
    
    if not getattr(results, 'was_standardized', False):
        print("\n⚠ 警告: 权重矩阵未行标准化!")
        print("  ρ的解释依赖于权重尺度，跨模型不可比。")
        print("  建议设置 auto_standardize=True (默认值)。")
    
    print("=" * 60)


def fit_sem_model(W, y, X=None, auto_standardize=True, verbose=True):
    """
    拟合空间误差模型（SEM）：y = Xβ + μ, μ = λWμ + ε
    
    为确保空间自回归系数λ具有可比性（不依赖权重尺度），权重矩阵必须进行行标准化。
    
    参数:
        W: 空间权重矩阵，可以是libpysal.weights.W对象或numpy数组
        y: 因变量，形状为(n, 1)或(n,)的numpy数组
        X: 自变量矩阵，形状为(n, k)的numpy数组，默认为None（只含截距项）
        auto_standardize: 是否自动进行行标准化（默认True，强烈建议保留）
        verbose: 是否打印标准化信息
    
    返回:
        results: 模型拟合结果对象，附加W标准化信息
    """
    is_standardized = False
    original_transform = None
    
    if isinstance(W, np.ndarray):
        W_array = W.astype(float)
        is_standardized = check_row_standardized(W_array)
        
        if auto_standardize and not is_standardized:
            if verbose:
                print(f"权重矩阵未行标准化，正在进行行标准化...")
                print(f"  行和范围: [{W_array.sum(axis=1).min():.2f}, {W_array.sum(axis=1).max():.2f}]")
            W_array = row_standardize(W_array)
            is_standardized = True
        
        W = libpysal.weights.W.from_array(W_array)
        
    elif isinstance(W, libpysal.weights.W):
        original_transform = W.transform
        
        W_temp = W.full()[0]
        is_standardized = check_row_standardized(W_temp)
        
        if auto_standardize and W.transform != 'r':
            if verbose:
                if not is_standardized:
                    print(f"权重矩阵未行标准化，正在进行行标准化...")
                else:
                    print(f"设置权重矩阵变换为行标准化('r')")
            W.transform = 'r'
            is_standardized = True
    else:
        raise ValueError("W必须是libpysal.weights.W对象或numpy数组")
    
    y = np.asarray(y).reshape(-1, 1)
    
    if X is None:
        X = np.ones((y.shape[0], 1))
    else:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X和y的样本数必须匹配")
    
    model = ML_Error(y, X, W, name_y='y', name_x=['x{}'.format(i) for i in range(X.shape[1])])
    
    model.was_standardized = is_standardized
    model.original_transform = original_transform
    model.w_transform = W.transform
    
    return model


def print_sem_results(results):
    """
    打印SEM模型的结果
    """
    print("=" * 60)
    print("空间误差模型（SEM）拟合结果")
    print("=" * 60)
    print("\n模型形式: y = Xβ + μ, μ = λWμ + ε")
    print("\n样本数:", results.n)
    print("自变量个数:", results.k)
    
    w_status = "✓ 已行标准化" if getattr(results, 'was_standardized', False) else "⚠ 未行标准化"
    print(f"权重矩阵状态: {w_status}")
    print(f"权重变换方式: {getattr(results, 'w_transform', 'unknown')}")
    print("=" * 60)
    
    print("\n系数估计:")
    print("-" * 40)
    print(f"λ (空间误差系数): {results.lam:.6f}")
    print(f"λ的标准误: {results.se_lam:.6f}")
    print(f"λ的z值: {results.z_lam:.6f}")
    print(f"λ的p值: {results.p_lam:.6f}")
    print()
    
    print("β系数:")
    for i, (beta, se, z, p) in enumerate(zip(results.betas, results.std_err, results.z_stat, results.pvalues)):
        if i < len(results.betas) - 1:
            print(f"  β_{i}: {beta[0]:.6f} (se={se:.6f}, z={z:.6f}, p={p:.6f})")
    
    print("\n模型拟合优度:")
    print("-" * 40)
    print(f"对数似然值: {results.logll:.6f}")
    print(f"AIC: {results.aic:.6f}")
    print(f"SIC: {results.sic:.6f}")
    print(f"伪R²: {results.pr2:.6f}")
    
    if not getattr(results, 'was_standardized', False):
        print("\n⚠ 警告: 权重矩阵未行标准化!")
        print("  λ的解释依赖于权重尺度，跨模型不可比。")
        print("  建议设置 auto_standardize=True (默认值)。")
    
    print("=" * 60)


def spatial_lm_test(W, y, X=None, auto_standardize=True):
    """
    拉格朗日乘数检验（LM test）用于诊断空间相关性
    
    检验包括：
    - LMlag: 检验是否存在空间滞后相关性（SAR）
    - LMerr: 检验是否存在空间误差相关性（SEM）
    - RLMlag: 稳健LMlag检验（控制空间误差）
    - RLMerr: 稳健LMerr检验（控制空间滞后）
    
    参数:
        W: 空间权重矩阵
        y: 因变量
        X: 自变量
        auto_standardize: 是否自动行标准化
    
    返回:
        lm_results: 包含LM检验统计量和p值的字典
    """
    if isinstance(W, np.ndarray):
        W_array = W.astype(float)
        if auto_standardize and not check_row_standardized(W_array):
            W_array = row_standardize(W_array)
        W = libpysal.weights.W.from_array(W_array)
    elif isinstance(W, libpysal.weights.W):
        if auto_standardize and W.transform != 'r':
            W.transform = 'r'
    
    y = np.asarray(y).reshape(-1, 1)
    if X is None:
        X = np.ones((y.shape[0], 1))
    else:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
    
    ols = OLS(y, X)
    
    e = ols.u
    n = len(e)
    
    W_matrix = W.full()[0]
    
    sigma2 = np.sum(e**2) / n
    
    WX = W_matrix @ X
    M = np.eye(n) - X @ np.linalg.inv(X.T @ X) @ X.T
    MWXM = M @ WX @ np.linalg.inv(X.T @ X) @ X.T
    trace_WtW = np.trace(W_matrix.T @ W_matrix)
    
    We = W_matrix @ e
    WWe = W_matrix @ We
    
    eWe = e.T @ We
    
    J = np.trace(W_matrix @ W_matrix + W_matrix.T @ W_matrix) / 2
    
    D = (J + (We.T @ M @ We) / sigma2) / n
    
    LMlag = (eWe / (sigma2 * D)) ** 2
    p_LMlag = 1 - stats.chi2.cdf(LMlag, 1)
    
    LMerr = (eWe / sigma2) ** 2 / J * n
    p_LMerr = 1 - stats.chi2.cdf(LMerr, 1)
    
    T1 = np.trace(W_matrix @ W_matrix + W_matrix.T @ W_matrix)
    T2 = np.trace(W_matrix ** 2)
    
    eWWe = e.T @ WWe
    eWWe_eWe_sq = eWWe / sigma2 - (eWe / sigma2) ** 2
    
    RLMlag_num = (eWe / sigma2 - eWWe_eWe_sq / J * n * D)
    RLMlag = RLMlag_num ** 2 / (n * (D - n * D**2 / J))
    p_RLMlag = 1 - stats.chi2.cdf(RLMlag, 1)
    
    RLMerr_num = (eWe / sigma2 - D * eWe / (sigma2 * D))
    RLMerr = RLMerr_num ** 2 / (J - n * D**2) * n
    p_RLMerr = 1 - stats.chi2.cdf(RLMerr, 1)
    
    return {
        'LMlag': float(LMlag),
        'p_LMlag': float(p_LMlag),
        'LMerr': float(LMerr),
        'p_LMerr': float(p_LMerr),
        'RLMlag': float(RLMlag),
        'p_RLMlag': float(p_RLMlag),
        'RLMerr': float(RLMerr),
        'p_RLMerr': float(p_RLMerr)
    }


def print_lm_results(lm_results, alpha=0.05):
    """
    打印LM检验结果
    """
    print("=" * 60)
    print("拉格朗日乘数检验（LM Test）结果")
    print("=" * 60)
    print(f"\n显著性水平 α = {alpha}")
    print("-" * 60)
    
    print(f"\nLMlag (检验空间滞后SAR):")
    print(f"  统计量 = {lm_results['LMlag']:.6f}")
    print(f"  p值    = {lm_results['p_LMlag']:.6f}")
    print(f"  结论   = {'✓ 存在空间滞后相关性' if lm_results['p_LMlag'] < alpha else '✗ 无显著空间滞后相关性'}")
    
    print(f"\nLMerr (检验空间误差SEM):")
    print(f"  统计量 = {lm_results['LMerr']:.6f}")
    print(f"  p值    = {lm_results['p_LMerr']:.6f}")
    print(f"  结论   = {'✓ 存在空间误差相关性' if lm_results['p_LMerr'] < alpha else '✗ 无显著空间误差相关性'}")
    
    print(f"\nRLMlag (稳健LMlag检验):")
    print(f"  统计量 = {lm_results['RLMlag']:.6f}")
    print(f"  p值    = {lm_results['p_RLMlag']:.6f}")
    print(f"  结论   = {'✓ 存在空间滞后相关性' if lm_results['p_RLMlag'] < alpha else '✗ 无显著空间滞后相关性'}")
    
    print(f"\nRLMerr (稳健LMerr检验):")
    print(f"  统计量 = {lm_results['RLMerr']:.6f}")
    print(f"  p值    = {lm_results['p_RLMerr']:.6f}")
    print(f"  结论   = {'✓ 存在空间误差相关性' if lm_results['p_RLMerr'] < alpha else '✗ 无显著空间误差相关性'}")
    
    print("=" * 60)


def auto_select_model(W, y, X=None, alpha=0.05, auto_standardize=True, verbose=True):
    """
    通过LM检验自动选择空间模型（OLS / SAR / SEM）
    
    选择策略：
    1. 如果LMlag和LMerr都不显著 → 选择OLS
    2. 如果只有LMlag显著 → 选择SAR
    3. 如果只有LMerr显著 → 选择SEM
    4. 如果两者都显著 → 比较RLMlag和RLMerr，选择p值更小的模型
    
    参数:
        W: 空间权重矩阵
        y: 因变量
        X: 自变量
        alpha: 显著性水平
        auto_standardize: 是否自动行标准化
        verbose: 是否打印详细信息
    
    返回:
        selected_model: 选择的模型类型 ('OLS', 'SAR', 'SEM')
        model_results: 拟合的模型结果（如果是OLS则为OLS结果对象）
        lm_results: LM检验结果
    """
    lm_results = spatial_lm_test(W, y, X, auto_standardize)
    
    if verbose:
        print_lm_results(lm_results, alpha)
    
    LMlag_sig = lm_results['p_LMlag'] < alpha
    LMerr_sig = lm_results['p_LMerr'] < alpha
    
    selected_model = 'OLS'
    model_results = None
    
    if not LMlag_sig and not LMerr_sig:
        if verbose:
            print("\n✓ 无显著空间相关性，选择OLS模型")
        
        y_arr = np.asarray(y).reshape(-1, 1)
        if X is None:
            X_arr = np.ones((y_arr.shape[0], 1))
        else:
            X_arr = np.asarray(X).reshape(y_arr.shape[0], -1)
        model_results = OLS(y_arr, X_arr)
    
    elif LMlag_sig and not LMerr_sig:
        if verbose:
            print("\n✓ 检测到空间滞后相关性，选择SAR模型")
        selected_model = 'SAR'
        model_results = fit_sar_model(W, y, X, auto_standardize, verbose=False)
    
    elif not LMlag_sig and LMerr_sig:
        if verbose:
            print("\n✓ 检测到空间误差相关性，选择SEM模型")
        selected_model = 'SEM'
        model_results = fit_sem_model(W, y, X, auto_standardize, verbose=False)
    
    else:
        RLMlag_sig = lm_results['p_RLMlag'] < alpha
        RLMerr_sig = lm_results['p_RLMerr'] < alpha
        
        if RLMlag_sig and not RLMerr_sig:
            if verbose:
                print("\n✓ 稳健检验显示空间滞后更显著，选择SAR模型")
            selected_model = 'SAR'
            model_results = fit_sar_model(W, y, X, auto_standardize, verbose=False)
        elif RLMerr_sig and not RLMlag_sig:
            if verbose:
                print("\n✓ 稳健检验显示空间误差更显著，选择SEM模型")
            selected_model = 'SEM'
            model_results = fit_sem_model(W, y, X, auto_standardize, verbose=False)
        else:
            if lm_results['p_RLMlag'] < lm_results['p_RLMerr']:
                if verbose:
                    print("\n✓ 比较稳健p值，空间滞后更显著，选择SAR模型")
                selected_model = 'SAR'
                model_results = fit_sar_model(W, y, X, auto_standardize, verbose=False)
            else:
                if verbose:
                    print("\n✓ 比较稳健p值，空间误差更显著，选择SEM模型")
                selected_model = 'SEM'
                model_results = fit_sem_model(W, y, X, auto_standardize, verbose=False)
    
    return selected_model, model_results, lm_results


def print_model_comparison(sar_results, sem_results):
    """
    打印SAR和SEM模型的比较结果
    """
    print("=" * 60)
    print("SAR vs SEM 模型比较")
    print("=" * 60)
    
    print(f"\n{'指标':<15} {'SAR':<15} {'SEM':<15} {'更优模型':<10}")
    print("-" * 55)
    
    metrics = [
        ('对数似然', sar_results.logll, sem_results.logll, 'max'),
        ('AIC', sar_results.aic, sem_results.aic, 'min'),
        ('SIC', sar_results.sic, sem_results.sic, 'min'),
        ('伪R²', sar_results.pr2, sem_results.pr2, 'max')
    ]
    
    for name, sar_val, sem_val, direction in metrics:
        if direction == 'max':
            better = 'SAR' if sar_val > sem_val else 'SEM'
        else:
            better = 'SAR' if sar_val < sem_val else 'SEM'
        print(f"{name:<15} {sar_val:<15.4f} {sem_val:<15.4f} {better:<10}")
    
    print("\n空间系数:")
    print(f"  SAR ρ = {sar_results.rho:.6f} (p={sar_results.p_rho:.6f})")
    print(f"  SEM λ = {sem_results.lam:.6f} (p={sem_results.p_lam:.6f})")
    
    print("=" * 60)


def generate_sar_data(W_matrix, rho, beta, X, sigma=0.5):
    """生成SAR模型数据"""
    n = len(W_matrix)
    I = np.eye(n)
    epsilon = np.random.randn(n, 1) * sigma
    inv_I_rhoW = np.linalg.inv(I - rho * W_matrix)
    y = inv_I_rhoW @ (X @ beta.reshape(-1, 1) + epsilon)
    return y


def generate_sem_data(W_matrix, lam, beta, X, sigma=0.5):
    """生成SEM模型数据"""
    n = len(W_matrix)
    I = np.eye(n)
    epsilon = np.random.randn(n, 1) * sigma
    inv_I_lamW = np.linalg.inv(I - lam * W_matrix)
    mu = inv_I_lamW @ epsilon
    y = X @ beta.reshape(-1, 1) + mu
    return y


if __name__ == "__main__":
    np.random.seed(42)
    
    n = 49
    grid_size = int(np.sqrt(n))
    W = libpysal.weights.lat2W(grid_size, grid_size)
    W.transform = 'r'
    W_matrix = W.full()[0]
    
    beta = np.array([1.0, 2.0])
    X = np.random.randn(n, 1)
    X = np.hstack([np.ones((n, 1)), X])
    
    print("\n" + "=" * 60)
    print("演示1: SAR数据生成与自动模型选择")
    print("=" * 60)
    print(f"真实模型: SAR (ρ=0.5)")
    y_sar = generate_sar_data(W_matrix, rho=0.5, beta=beta, X=X)
    
    selected_model, model_results, lm_results = auto_select_model(W, y_sar, X)
    
    print(f"\n自动选择结果: {selected_model}")
    if selected_model == 'SAR':
        print_sar_results(model_results)
    elif selected_model == 'SEM':
        print_sem_results(model_results)
    
    print("\n" + "=" * 60)
    print("演示2: SEM数据生成与自动模型选择")
    print("=" * 60)
    print(f"真实模型: SEM (λ=0.6)")
    y_sem = generate_sem_data(W_matrix, lam=0.6, beta=beta, X=X)
    
    selected_model2, model_results2, lm_results2 = auto_select_model(W, y_sem, X)
    
    print(f"\n自动选择结果: {selected_model2}")
    if selected_model2 == 'SAR':
        print_sar_results(model_results2)
    elif selected_model2 == 'SEM':
        print_sem_results(model_results2)
    
    print("\n" + "=" * 60)
    print("演示3: 同时拟合SAR和SEM并比较")
    print("=" * 60)
    sar_results = fit_sar_model(W, y_sar, X, verbose=False)
    sem_results = fit_sem_model(W, y_sar, X, verbose=False)
    
    print_model_comparison(sar_results, sem_results)
    
    print("\n真实参数值 (SAR数据):")
    print(f"ρ = 0.5")
    print(f"β = {beta}")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("演示4: 同时拟合SEM并比较")
    print("=" * 60)
    sar_results_sem = fit_sar_model(W, y_sem, X, verbose=False)
    sem_results_sem = fit_sem_model(W, y_sem, X, verbose=False)
    
    print_model_comparison(sar_results_sem, sem_results_sem)
    
    print("\n真实参数值 (SEM数据):")
    print(f"λ = 0.6")
    print(f"β = {beta}")
    print("=" * 60)
