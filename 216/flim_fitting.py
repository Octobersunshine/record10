"""
光子计数探测器直方图拟合 - 多指数衰减模型 (IRF卷积)
用于荧光寿命成像 (FLIM) 数据分析

模型: I(t) = IRF(t) * Σ [a_i * exp(-t/τ_i)] + b
其中:
  - IRF: 仪器响应函数 (高斯分布)
  - τ_i: 寿命分量
  - a_i: 各分量的振幅
  - b: 背景噪声

拟合方法:
  - poisson_mle : 泊松最大似然估计 (推荐，低光子数区域无偏)
  - pearson_chi2: Pearson卡方拟合 (以模型为权重)
  - neyman_chi2 : Neyman卡方拟合 (以数据为权重，传统方法)
"""

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.signal import convolve
from scipy.stats import poisson
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Tuple, Dict, List, Optional, Union


def gaussian_irf(t: np.ndarray, t0: float, sigma: float) -> np.ndarray:
    """
    高斯仪器响应函数 (IRF)
    
    参数:
        t: 时间轴 (纳秒)
        t0: IRF中心位置
        sigma: IRF宽度 (标准差)
    
    返回:
        归一化的IRF分布
    """
    irf = np.exp(-0.5 * ((t - t0) / sigma) ** 2)
    return irf / np.sum(irf)


def multi_exponential_decay(t: np.ndarray, params: np.ndarray, 
                            n_components: int = 2) -> np.ndarray:
    """
    多指数衰减模型 (未卷积IRF)
    
    参数:
        t: 时间轴
        params: 模型参数 [a1, tau1, a2, tau2, ..., background]
        n_components: 指数分量数量
    
    返回:
        衰减曲线
    """
    decay = np.zeros_like(t, dtype=float)
    t_start_idx = np.argmin(np.abs(t))
    
    for i in range(n_components):
        amp = params[2 * i]
        tau = params[2 * i + 1]
        decay[t_start_idx:] += amp * np.exp(-(t[t_start_idx:]) / tau)
    
    decay += params[-1]
    
    return decay


def model_with_irf(t: np.ndarray, decay_params: np.ndarray, 
                   irf: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    IRF卷积后的多指数衰减模型
    
    参数:
        t: 时间轴
        decay_params: 衰减参数 [a1, tau1, a2, tau2, ..., background]
        irf: 仪器响应函数
        n_components: 指数分量数量
    
    返回:
        卷积后的模型曲线
    """
    decay = np.zeros_like(t, dtype=float)
    t_start_idx = np.argmin(np.abs(t))
    
    for i in range(n_components):
        amp = decay_params[2 * i]
        tau = decay_params[2 * i + 1]
        decay[t_start_idx:] += amp * np.exp(-(t[t_start_idx:]) / tau)
    
    decay += decay_params[-1]
    
    model = convolve(irf, decay, mode='same')
    
    return model


def _compute_model_from_params(params: np.ndarray, t: np.ndarray, 
                               irf: np.ndarray, n_components: int,
                               fit_irf: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    从参数计算模型曲线 (内部辅助函数)
    
    参数:
        params: 完整参数向量
        t: 时间轴
        irf: 仪器响应函数
        n_components: 指数分量数量
        fit_irf: 是否拟合IRF参数
    
    返回:
        (模型曲线, 当前IRF)
    """
    if fit_irf:
        t0 = params[-3]
        sigma_irf = params[-2]
        current_irf = gaussian_irf(t, t0, sigma_irf)
        decay_and_bg = np.concatenate([params[:-3], [params[-1]]])
    else:
        current_irf = irf
        decay_and_bg = params
    
    model = model_with_irf(t, decay_and_bg, current_irf, n_components)
    
    return model, current_irf


def cost_poisson_mle(params: np.ndarray, t: np.ndarray, 
                     data: np.ndarray, irf: np.ndarray,
                     n_components: int = 2,
                     fit_irf: bool = True) -> np.ndarray:
    """
    泊松MLE残差 (使用近似变换用于least_squares)
    
    使用 Freeman-Tukey 变换:
    r_i = sqrt(y_i + 3/8) - sqrt(mu_i + 3/8)
    
    这是泊松数据的方差稳定变换，渐近等价于MLE。
    
    参数:
        params: 模型参数
        t: 时间轴
        data: 实验数据 (光子计数)
        irf: 仪器响应函数
        n_components: 指数分量数量
        fit_irf: 是否拟合IRF参数
    
    返回:
        残差向量
    """
    model, _ = _compute_model_from_params(params, t, irf, n_components, fit_irf)
    
    model_safe = np.maximum(model, 0.0)
    
    residuals = np.sqrt(data + 0.375) - np.sqrt(model_safe + 0.375)
    
    return residuals


def poisson_nll_scalar(params: np.ndarray, t: np.ndarray, 
                       data: np.ndarray, irf: np.ndarray,
                       n_components: int = 2,
                       fit_irf: bool = True) -> float:
    """
    泊松负对数似然 (标量形式，用于minimize)
    
    NLL = Σ [μ_i - y_i * log(μ_i)]
    
    参数:
        params: 模型参数
        t: 时间轴
        data: 实验数据 (光子计数)
        irf: 仪器响应函数
        n_components: 指数分量数量
        fit_irf: 是否拟合IRF参数
    
    返回:
        负对数似然值
    """
    model, _ = _compute_model_from_params(params, t, irf, n_components, fit_irf)
    
    model_safe = np.maximum(model, 1e-15)
    
    nll = np.sum(model_safe - data * np.log(model_safe))
    
    return nll


def cost_pearson_chi2(params: np.ndarray, t: np.ndarray, 
                      data: np.ndarray, irf: np.ndarray,
                      n_components: int = 2,
                      fit_irf: bool = True) -> np.ndarray:
    """
    Pearson卡方拟合成本函数
    
    χ² = Σ [(y_i - μ_i)² / μ_i]
    
    优点:
      - 以模型预期值为权重，低光子数区域稳定
      - 渐近等价于泊松MLE
    
    参数:
        params: 模型参数
        t: 时间轴
        data: 实验数据 (光子计数)
        irf: 仪器响应函数
        n_components: 指数分量数量
        fit_irf: 是否拟合IRF参数
    
    返回:
        残差向量
    """
    model, _ = _compute_model_from_params(params, t, irf, n_components, fit_irf)
    
    model_safe = np.maximum(model, 1e-10)
    
    residuals = (data - model) / np.sqrt(model_safe)
    
    return residuals


def cost_neyman_chi2(params: np.ndarray, t: np.ndarray, 
                     data: np.ndarray, irf: np.ndarray,
                     n_components: int = 2,
                     fit_irf: bool = True) -> np.ndarray:
    """
    Neyman卡方拟合成本函数 (传统方法)
    
    χ² = Σ [(y_i - μ_i)² / y_i]
    
    缺点:
      - 低光子数区域 (y_i=0) 权重无穷大，产生偏差
    
    参数:
        params: 模型参数
        t: 时间轴
        data: 实验数据 (光子计数)
        irf: 仪器响应函数
        n_components: 指数分量数量
        fit_irf: 是否拟合IRF参数
    
    返回:
        残差向量
    """
    model, _ = _compute_model_from_params(params, t, irf, n_components, fit_irf)
    
    weights = np.sqrt(np.maximum(data, 1))
    residuals = (data - model) / weights
    
    return residuals


def fit_flim_histogram(t: np.ndarray, data: np.ndarray,
                       n_components: int = 2,
                       initial_params: Optional[np.ndarray] = None,
                       irf: Optional[np.ndarray] = None,
                       irf_t0: float = 0.0,
                       irf_sigma: float = 0.1,
                       fit_irf: bool = True,
                       method: str = "poisson_mle",
                       bounds: Optional[Tuple] = None,
                       verbose: bool = True) -> Dict:
    """
    拟合FLIM直方图
    
    参数:
        t: 时间轴 (纳秒)
        data: 实验数据 (光子计数)
        n_components: 指数分量数量
        initial_params: 初始参数 [a1, tau1, a2, tau2, ..., t0, sigma_irf, background]
        irf: 仪器响应函数 (如果fit_irf=False则使用此IRF)
        irf_t0: IRF初始中心位置
        irf_sigma: IRF初始宽度
        fit_irf: 是否同时拟合IRF参数
        method: 拟合方法
            - "poisson_mle" : 泊松MLE (推荐，低光子数无偏，使用minimize)
            - "pearson_chi2": Pearson卡方 (以模型为权重，使用least_squares)
            - "neyman_chi2" : Neyman卡方 (以数据为权重，使用least_squares)
        bounds: 参数边界 ((低边界), (高边界))
        verbose: 是否打印详细信息
    
    返回:
        拟合结果字典
    """
    t = np.asarray(t, dtype=float)
    data = np.asarray(data, dtype=float)
    
    if method not in ("poisson_mle", "pearson_chi2", "neyman_chi2"):
        raise ValueError(f"Unknown method: {method}, options: poisson_mle, pearson_chi2, neyman_chi2")
    
    if irf is None:
        irf = gaussian_irf(t, irf_t0, irf_sigma)
    
    if initial_params is None:
        initial_params = []
        for i in range(n_components):
            initial_params.extend([np.max(data) / n_components, 1.0 + i])
        if fit_irf:
            initial_params.extend([irf_t0, irf_sigma])
        initial_params.append(np.min(data) * 0.5)
        initial_params = np.array(initial_params)
    
    if bounds is None:
        lower = []
        upper = []
        for i in range(n_components):
            lower.extend([0, 0.01])
            upper.extend([np.inf, 100])
        if fit_irf:
            lower.extend([-2, 0.01])
            upper.extend([2, 2.0])
        lower.append(0)
        upper.append(np.inf)
        bounds = (lower, upper)
    
    if method == "poisson_mle":
        scipy_bounds = list(zip(bounds[0], bounds[1]))
        result = minimize(
            poisson_nll_scalar,
            initial_params,
            args=(t, data, irf, n_components, fit_irf),
            method='L-BFGS-B',
            bounds=scipy_bounds,
            options={'maxiter': 10000, 'ftol': 1e-15}
        )
        fitted_params = result.x
        success = result.success
    else:
        cost_functions = {
            "pearson_chi2": cost_pearson_chi2,
            "neyman_chi2": cost_neyman_chi2
        }
        cost_func = cost_functions[method]
        
        result = least_squares(
            cost_func,
            initial_params,
            args=(t, data, irf, n_components, fit_irf),
            bounds=bounds,
            method='trf',
            max_nfev=10000
        )
        fitted_params = result.x
        success = result.success
    
    if fit_irf:
        t0_fit = fitted_params[-3]
        sigma_fit = fitted_params[-2]
        irf_fit = gaussian_irf(t, t0_fit, sigma_fit)
    else:
        t0_fit = irf_t0
        sigma_fit = irf_sigma
        irf_fit = irf
    
    decay_params = np.concatenate([fitted_params[:-3], [fitted_params[-1]]]) if fit_irf else fitted_params
    model_fit = model_with_irf(t, decay_params, irf_fit, n_components)
    
    residuals = data - model_fit
    chi_sq = np.sum(residuals**2 / np.maximum(model_fit, 1))
    n_data = len(data)
    n_params = len(fitted_params)
    reduced_chi_sq = chi_sq / (n_data - n_params)
    
    lifetimes = []
    amplitudes = []
    for i in range(n_components):
        amp = fitted_params[2 * i]
        tau = fitted_params[2 * i + 1]
        lifetimes.append(tau)
        amplitudes.append(amp)
    
    total_amp = np.sum(amplitudes)
    fractions = [amp / total_amp for amp in amplitudes]
    
    if verbose:
        method_names = {
            "poisson_mle": "Poisson MLE",
            "pearson_chi2": "Pearson Chi2",
            "neyman_chi2": "Neyman Chi2"
        }
        print("=" * 60)
        print(f"FLIM Multi-Exponential Fit [{method_names[method]}]")
        print("=" * 60)
        print(f"Components: {n_components}")
        print(f"Data points: {n_data}")
        print(f"Fitted parameters: {n_params}")
        print(f"Degrees of freedom: {n_data - n_params}")
        print(f"chi2 (Pearson): {chi_sq:.2f}")
        print(f"Reduced chi2: {reduced_chi_sq:.4f}")
        print("-" * 60)
        
        if fit_irf:
            print(f"IRF center t0 = {t0_fit:.4f} ns")
            print(f"IRF width sigma = {sigma_fit:.4f} ns")
            print("-" * 60)
        
        for i in range(n_components):
            print(f"Component {i+1}:")
            print(f"  Lifetime tau{i+1} = {lifetimes[i]:.4f} ns")
            print(f"  Amplitude a{i+1} = {amplitudes[i]:.2f}")
            print(f"  Fraction f{i+1} = {fractions[i]*100:.1f}%")
        
        print(f"Background = {fitted_params[-1]:.2f}")
        print("=" * 60)
    
    return {
        'params': fitted_params,
        'lifetimes': lifetimes,
        'amplitudes': amplitudes,
        'fractions': fractions,
        'irf_t0': t0_fit,
        'irf_sigma': sigma_fit,
        'irf': irf_fit,
        'model': model_fit,
        'residuals': residuals,
        'chi_square': chi_sq,
        'reduced_chi_square': reduced_chi_sq,
        'n_components': n_components,
        'method': method,
        'success': success,
        'optimize_result': result
    }


def generate_synthetic_flim_data(t: np.ndarray, 
                                 true_params: Dict,
                                 irf: np.ndarray,
                                 noise: bool = True,
                                 seed: Optional[int] = None) -> np.ndarray:
    """
    生成模拟FLIM数据
    
    参数:
        t: 时间轴
        true_params: 真实参数 {'lifetimes': [...], 'amplitudes': [...], 'background': ...}
        irf: 仪器响应函数
        noise: 是否添加泊松噪声
        seed: 随机种子
    
    返回:
        模拟数据
    """
    if seed is not None:
        np.random.seed(seed)
    
    lifetimes = true_params['lifetimes']
    amplitudes = true_params['amplitudes']
    background = true_params.get('background', 0)
    n_components = len(lifetimes)
    
    decay = np.zeros_like(t, dtype=float)
    t_start_idx = np.argmin(np.abs(t))
    
    for i in range(n_components):
        decay[t_start_idx:] += amplitudes[i] * np.exp(-(t[t_start_idx:]) / lifetimes[i])
    
    decay += background
    
    data = convolve(irf, decay, mode='same')
    
    if noise:
        data = np.random.poisson(np.maximum(data, 0.5))
    
    return data


def plot_fit_results(t: np.ndarray, data: np.ndarray, 
                     fit_result: Dict,
                     irf: Optional[np.ndarray] = None,
                     title: str = "FLIM 多指数衰减拟合",
                     save_path: Optional[str] = None) -> None:
    """
    可视化拟合结果
    
    参数:
        t: 时间轴
        data: 实验数据
        fit_result: fit_flim_histogram() 的返回结果
        irf: 原始IRF (可选)
        title: 图表标题
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    ax1.semilogy(t, data, 'o', color='steelblue', markersize=4, 
                label='Data', alpha=0.7)
    
    ax1.semilogy(t, fit_result['model'], 'r-', linewidth=2.5, 
                label='Fit')
    
    t_start_idx = np.argmin(np.abs(t))
    for i, (tau, amp, frac) in enumerate(zip(
        fit_result['lifetimes'], 
        fit_result['amplitudes'],
        fit_result['fractions'])):
        decay_comp = np.zeros_like(t)
        decay_comp[t_start_idx:] = amp * np.exp(-(t[t_start_idx:]) / tau)
        decay_comp = convolve(fit_result['irf'], decay_comp, mode='same')
        ax1.semilogy(t, decay_comp, '--', linewidth=1.5, alpha=0.8,
                    label=f'tau{i+1}={tau:.2f}ns ({frac*100:.1f}%)')
    
    if irf is not None:
        ax1.semilogy(t, irf * np.max(fit_result['model']), ':', 
                    color='gray', linewidth=1, alpha=0.5, label='IRF')
    
    ax1.semilogy(t, fit_result['irf'] * np.max(fit_result['model']), ':', 
                color='orange', linewidth=1.5, label='Fitted IRF')
    
    ax1.set_xlabel('Time (ns)', fontsize=12)
    ax1.set_ylabel('Photon Counts', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(t[0], t[-1])
    
    residuals = fit_result['residuals']
    ax2.bar(t, residuals, width=t[1]-t[0], color='steelblue', alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_xlabel('Time (ns)', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(t[0], t[-1])
    
    weighted_residuals = residuals / np.sqrt(np.maximum(fit_result['model'], 1))
    ax3.bar(t, weighted_residuals, width=t[1]-t[0], color='coral', alpha=0.7)
    ax3.axhline(y=0, color='black', linewidth=0.8)
    ax3.set_xlabel('Time (ns)', fontsize=12)
    ax3.set_ylabel('Weighted Res.', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(t[0], t[-1])
    
    method_names = {
        "poisson_mle": "Poisson MLE",
        "pearson_chi2": "Pearson Chi2",
        "neyman_chi2": "Neyman Chi2"
    }
    method_label = method_names.get(fit_result.get('method', ''), '')
    
    info_text = (f"{method_label}  chi2={fit_result['chi_square']:.2f}, "
                f"chi2_red={fit_result['reduced_chi_square']:.4f}\n")
    for i, (tau, frac) in enumerate(zip(
        fit_result['lifetimes'], fit_result['fractions'])):
        info_text += f"tau{i+1}={tau:.3f}ns ({frac*100:.1f}%)  "
    
    fig.text(0.5, 0.01, info_text, ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_method_comparison(t: np.ndarray, data: np.ndarray,
                           results: Dict[str, Dict],
                           irf: Optional[np.ndarray] = None,
                           save_path: Optional[str] = None) -> None:
    """
    对比不同拟合方法的结果
    
    参数:
        t: 时间轴
        data: 实验数据
        results: {方法名: 拟合结果} 字典
        irf: 仪器响应函数
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax1 = axes[0, 0]
    ax1.semilogy(t, data, 'o', color='steelblue', markersize=3, alpha=0.5, label='Data')
    colors = ['r-', 'g-', 'b-']
    for (name, res), color in zip(results.items(), colors):
        ax1.semilogy(t, res['model'], color, linewidth=2, label=name)
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Photon Counts')
    ax1.set_title('Fit Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    for (name, res), color in zip(results.items(), colors):
        ax2.plot(t, res['residuals'], color, linewidth=1, label=name, alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    for (name, res), color in zip(results.items(), colors):
        weighted = res['residuals'] / np.sqrt(np.maximum(res['model'], 1))
        ax3.plot(t, weighted, color, linewidth=1, label=name, alpha=0.7)
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_xlabel('Time (ns)')
    ax3.set_ylabel('Weighted Residuals')
    ax3.set_title('Weighted Residual Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    names = list(results.keys())
    x_pos = np.arange(len(names))
    chi2_vals = [results[n]['reduced_chi_square'] for n in names]
    bars = ax4.bar(x_pos, chi2_vals, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax4.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Ideal')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(names, rotation=15)
    ax4.set_ylabel('Reduced Chi-Square')
    ax4.set_title('Goodness of Fit')
    ax4.legend()
    for bar, val in zip(bars, chi2_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('FLIM Fitting Method Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison chart saved to: {save_path}")
    
    plt.show()


def compare_models(t: np.ndarray, data: np.ndarray,
                   max_components: int = 4,
                   irf: Optional[np.ndarray] = None,
                   method: str = "poisson_mle",
                   verbose: bool = True) -> List[Dict]:
    """
    比较不同分量数的拟合结果 (使用AIC/BIC准则)
    
    参数:
        t: 时间轴
        data: 实验数据
        max_components: 最大分量数
        irf: 仪器响应函数
        method: 拟合方法
        verbose: 是否打印结果
    
    返回:
        各分量数的拟合结果列表
    """
    results = []
    n_data = len(data)
    
    for n_comp in range(1, max_components + 1):
        fit = fit_flim_histogram(t, data, n_components=n_comp, 
                                irf=irf, method=method, verbose=False)
        
        k = 2 * n_comp + 3
        chi_sq = fit['chi_square']
        
        aic = n_data * np.log(chi_sq / n_data) + 2 * k
        bic = n_data * np.log(chi_sq / n_data) + k * np.log(n_data)
        
        fit['aic'] = aic
        fit['bic'] = bic
        results.append(fit)
        
        if verbose:
            print(f"Components={n_comp}: chi2={chi_sq:.2f}, chi2_red={fit['reduced_chi_square']:.4f}, "
                  f"AIC={aic:.2f}, BIC={bic:.2f}")
    
    return results


def estimate_initial_tau(t: np.ndarray, data: np.ndarray,
                         n_components: int = 2) -> np.ndarray:
    """
    估算初始寿命值 (基于对数线性拟合)
    
    参数:
        t: 时间轴
        data: 实验数据
        n_components: 指数分量数量
    
    返回:
        初始寿命估计值
    """
    t_start_idx = np.argmin(np.abs(t))
    t_tail = t[t_start_idx:]
    data_tail = data[t_start_idx:]
    
    mask = data_tail > 0
    t_valid = t_tail[mask]
    data_valid = data_tail[mask]
    
    if len(t_valid) < 10:
        return np.ones(n_components)
    
    log_data = np.log(data_valid)
    
    taus = []
    for i in range(n_components):
        if i == 0:
            slope, _ = np.polyfit(t_valid[-len(t_valid)//3:], 
                                log_data[-len(log_data)//3:], 1)
        else:
            segment = len(t_valid) // (n_components + 1)
            start = segment * i
            end = segment * (i + 1)
            slope, _ = np.polyfit(t_valid[start:end], 
                                log_data[start:end], 1)
        
        tau = max(-1.0 / slope, 0.01)
        taus.append(tau)
    
    return np.sort(taus)[::-1]


def fit_with_automatic_initial(t: np.ndarray, data: np.ndarray,
                               n_components: int = 2,
                               irf: Optional[np.ndarray] = None,
                               fit_irf: bool = True,
                               method: str = "poisson_mle",
                               verbose: bool = True) -> Dict:
    """
    使用自动估计的初始参数进行拟合
    
    参数:
        t: 时间轴
        data: 实验数据
        n_components: 指数分量数量
        irf: 仪器响应函数
        fit_irf: 是否拟合IRF参数
        method: 拟合方法 (poisson_mle/pearson_chi2/neyman_chi2)
        verbose: 是否打印信息
    
    返回:
        拟合结果字典
    """
    estimated_taus = estimate_initial_tau(t, data, n_components)
    
    initial_params = []
    max_count = np.max(data)
    for i in range(n_components):
        initial_params.extend([max_count / n_components, estimated_taus[i]])
    
    if fit_irf:
        initial_params.extend([0.0, 0.1])
    initial_params.append(np.min(data) * 0.1)
    
    if verbose:
        print("Auto-estimated initial parameters:")
        for i in range(n_components):
            print(f"  tau{i+1}_init = {estimated_taus[i]:.4f} ns")
    
    return fit_flim_histogram(t, data, n_components=n_components,
                             initial_params=np.array(initial_params),
                             irf=irf, fit_irf=fit_irf, method=method, verbose=verbose)


def phasor_analysis(t: np.ndarray, data: np.ndarray,
                    frequency: float = 80.0) -> Dict:
    """
    相量分析 (Phasor Plot) - 频域寿命估计
    
    将时域直方图通过傅里叶变换映射到相量空间 (g, s):
      - g = Σ[y(t) * cos(ωt)] / Σ[y(t)]
      - s = Σ[y(t) * sin(ωt)] / Σ[y(t)]
    
    单指数衰减在相量空间上的轨迹是通用圆 (universal circle):
      (g - 0.5)² + s² = 0.25
    
    多指数衰减位于通用圆内部。
    
    参数:
        t: 时间轴 (纳秒)
        data: 光子计数数据
        frequency: 调制频率 (MHz)
    
    返回:
        相量分析结果字典
    """
    omega = 2 * np.pi * frequency
    
    norm = np.sum(data)
    if norm == 0:
        return {'g': 0, 's': 0, 'tau_phase': 0, 'tau_mod': 0, 
                'frequency': frequency, 'norm': 0}
    
    g = np.sum(data * np.cos(omega * t)) / norm
    s = np.sum(data * np.sin(omega * t)) / norm
    
    tau_phase = (1 / omega) * (s / g) if g > 0 else 0
    tau_mod = (1 / omega) * np.sqrt(max(0, 1 / (g**2 + s**2) - 1)) if (g**2 + s**2) > 0 else 0
    
    return {
        'g': g,
        's': s,
        'tau_phase': tau_phase,
        'tau_mod': tau_mod,
        'frequency': frequency,
        'norm': norm
    }


def phasor_analysis_irf_corrected(t: np.ndarray, data: np.ndarray,
                                  irf: np.ndarray,
                                  frequency: float = 80.0) -> Dict:
    """
    IRF校正的相量分析
    
    对数据和IRF分别进行相量分析，然后通过除法消除IRF影响:
      g_corr = (g_data * g_irf + s_data * s_irf) / (g_irf² + s_irf²)
      s_corr = (s_data * g_irf - g_data * s_irf) / (g_irf² + s_irf²)
    
    参数:
        t: 时间轴 (纳秒)
        data: 光子计数数据
        irf: 仪器响应函数数据
        frequency: 调制频率 (MHz)
    
    返回:
        校正后的相量分析结果
    """
    phasor_data = phasor_analysis(t, data, frequency)
    phasor_irf = phasor_analysis(t, irf, frequency)
    
    g_irf = phasor_irf['g']
    s_irf = phasor_irf['s']
    
    denom = g_irf**2 + s_irf**2
    if denom < 1e-15:
        return phasor_data
    
    g_corr = (phasor_data['g'] * g_irf + phasor_data['s'] * s_irf) / denom
    s_corr = (phasor_data['s'] * g_irf - phasor_data['g'] * s_irf) / denom
    
    omega = 2 * np.pi * frequency
    tau_phase = (1 / omega) * (s_corr / g_corr) if g_corr > 0 else 0
    tau_mod = (1 / omega) * np.sqrt(max(0, 1 / (g_corr**2 + s_corr**2) - 1)) if (g_corr**2 + s_corr**2) > 0 else 0
    
    return {
        'g': g_corr,
        's': s_corr,
        'tau_phase': tau_phase,
        'tau_mod': tau_mod,
        'frequency': frequency,
        'g_raw': phasor_data['g'],
        's_raw': phasor_data['s'],
        'norm': phasor_data['norm']
    }


def multi_frequency_phasor(t: np.ndarray, data: np.ndarray,
                           frequencies: List[float] = None) -> List[Dict]:
    """
    多频率相量分析
    
    在多个频率上进行相量分析，用于区分不同的寿命组分。
    
    参数:
        t: 时间轴 (纳秒)
        data: 光子计数数据
        frequencies: 频率列表 (MHz)，默认 [20, 40, 80, 160]
    
    返回:
        各频率的相量分析结果列表
    """
    if frequencies is None:
        frequencies = [20, 40, 80, 160]
    
    results = []
    for freq in frequencies:
        phasor = phasor_analysis(t, data, freq)
        results.append(phasor)
    
    return results


def compute_phasor_from_tau(tau: float, frequency: float) -> Tuple[float, float]:
    """
    从寿命计算相量坐标
    
    对于单指数衰减 exp(-t/τ):
      g = 1 / (1 + (ωτ)²)
      s = ωτ / (1 + (ωτ)²)
    
    参数:
        tau: 寿命 (ns)
        frequency: 频率 (MHz)
    
    返回:
        (g, s) 坐标
    """
    omega = 2 * np.pi * frequency
    denom = 1 + (omega * tau) ** 2
    g = 1 / denom
    s = omega * tau / denom
    return g, s


def compute_tau_from_phasor(g: float, s: float, frequency: float) -> Dict:
    """
    从相量坐标反演寿命
    
    参数:
        g: g坐标
        s: s坐标
        frequency: 频率 (MHz)
    
    返回:
        {'tau_phase': 相寿命, 'tau_mod': 调制寿命}
    """
    omega = 2 * np.pi * frequency
    
    tau_phase = (1 / omega) * (s / g) if abs(g) > 1e-15 else 0
    r2 = g**2 + s**2
    tau_mod = (1 / omega) * np.sqrt(max(0, 1 / r2 - 1)) if r2 > 1e-15 else 0
    
    return {
        'tau_phase': tau_phase,
        'tau_mod': tau_mod
    }


def phasor_fraction_estimate(g: float, s: float, 
                             tau1: float, tau2: float,
                             frequency: float) -> Dict:
    """
    估计双组分的分数
    
    假设两个组分，在线上插值估计分数:
      f1 * phasor(tau1) + f2 * phasor(tau2) = phasor(data)
    
    参数:
        g: 数据的g坐标
        s: 数据的s坐标
        tau1: 组分1的寿命 (ns)
        tau2: 组分2的寿命 (ns)
        frequency: 频率 (MHz)
    
    返回:
        {'f1': 组分1分数, 'f2': 组分2分数}
    """
    g1, s1 = compute_phasor_from_tau(tau1, frequency)
    g2, s2 = compute_phasor_from_tau(tau2, frequency)
    
    dg = g2 - g1
    ds = s2 - s1
    
    denom = dg**2 + ds**2
    if denom < 1e-15:
        return {'f1': 0.5, 'f2': 0.5}
    
    f2 = ((g - g1) * dg + (s - s1) * ds) / denom
    f2 = np.clip(f2, 0, 1)
    f1 = 1 - f2
    
    return {'f1': f1, 'f2': f2}


def plot_phasor(results: List[Dict],
                data: Optional[np.ndarray] = None,
                t: Optional[np.ndarray] = None,
                title: str = "Phasor Plot",
                save_path: Optional[str] = None) -> None:
    """
    绘制相量图
    
    显示通用圆、单指数寿命线、以及数据点。
    
    参数:
        results: 相量分析结果列表 (单元素或多元素)
        data: 原始数据直方图 (可选，用于显示IRF点)
        t: 时间轴 (可选)
        title: 图表标题
        save_path: 保存路径
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    theta = np.linspace(0, np.pi, 200)
    g_circle = 0.5 + 0.5 * np.cos(theta)
    s_circle = 0.5 * np.sin(theta)
    ax.plot(g_circle, s_circle, 'k-', linewidth=2, label='Universal Circle')
    
    frequency = results[0]['frequency']
    tau_range = np.logspace(-1, 2, 100)
    for tau in [0.1, 0.5, 1, 2, 5, 10, 20, 50]:
        g_tau, s_tau = compute_phasor_from_tau(tau, frequency)
        ax.plot(g_tau, s_tau, 'o', color='gray', markersize=8)
        ax.annotate(f'{tau}ns', (g_tau, s_tau), textcoords="offset points", 
                   xytext=(10, 10), fontsize=9, color='gray')
    
    ax.plot([0, 1], [0, 0], 'k--', linewidth=0.5, alpha=0.5)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(results)))
    
    for i, result in enumerate(results):
        g = result['g']
        s = result['s']
        color = colors[i]
        
        ax.plot(g, s, 'o', color=color, markersize=12, markeredgecolor='black',
               markeredgewidth=2, zorder=5)
        
        freq = result.get('frequency', 80)
        label = f'{freq} MHz'
        if 'tau_phase' in result:
            label += f'\nτφ={result["tau_phase"]:.2f}ns'
        if 'tau_mod' in result:
            label += f'\nτM={result["tau_mod"]:.2f}ns'
        
        ax.annotate(label, (g, s), textcoords="offset points",
                   xytext=(15, 15), fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax.plot(0, 0, 's', color='blue', markersize=10, label='τ=0 (IRF)')
    ax.plot(1, 0, 's', color='green', markersize=10, label='τ=∞')
    
    if data is not None and t is not None:
        irf_phasor = phasor_analysis(t, np.maximum(data, 0), frequency)
        ax.plot(irf_phasor['g'], irf_phasor['s'], 's', color='orange', 
               markersize=10, label='Data IRF')
    
    ax.set_xlabel('g = <cos(ωt)>', fontsize=14)
    ax.set_ylabel('s = <sin(ωt)>', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 0.6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Phasor plot saved to: {save_path}")
    
    plt.show()


def plot_phasor_multifrequency(t: np.ndarray, data: np.ndarray,
                               frequencies: List[float] = None,
                               save_path: Optional[str] = None) -> None:
    """
    绘制多频率相量图
    
    在同一相量空间中显示多个频率的数据点和通用圆。
    
    参数:
        t: 时间轴 (纳秒)
        data: 光子计数数据
        frequencies: 频率列表 (MHz)
        save_path: 保存路径
    """
    if frequencies is None:
        frequencies = [20, 40, 80, 160]
    
    results = multi_frequency_phasor(t, data, frequencies)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    ax1 = axes[0]
    for freq, result in zip(frequencies, results):
        theta = np.linspace(0, np.pi, 200)
        g_circle = 0.5 + 0.5 * np.cos(theta)
        s_circle = 0.5 * np.sin(theta)
        ax1.plot(g_circle, s_circle, '-', linewidth=1, alpha=0.3, color='gray')
        
        g, s = result['g'], result['s']
        ax1.plot(g, s, 'o', markersize=12, markeredgecolor='black',
                markeredgewidth=2, label=f'{freq} MHz')
    
    ax1.set_xlabel('g', fontsize=12)
    ax1.set_ylabel('s', fontsize=12)
    ax1.set_title('Multi-Frequency Phasor Plot', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 0.6)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    taus_phase = [r['tau_phase'] for r in results]
    taus_mod = [r['tau_mod'] for r in results]
    
    x = np.arange(len(frequencies))
    width = 0.35
    ax2.bar(x - width/2, taus_phase, width, label='τ_phase', color='steelblue')
    ax2.bar(x + width/2, taus_mod, width, label='τ_mod', color='coral')
    
    ax2.set_xlabel('Frequency (MHz)', fontsize=12)
    ax2.set_ylabel('Lifetime (ns)', fontsize=12)
    ax2.set_title('Lifetime vs Frequency', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(frequencies)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Multi-frequency phasor plot saved to: {save_path}")
    
    plt.show()


def phasor_histogram_2d(t: np.ndarray, data: np.ndarray,
                        frequency: float = 80.0,
                        bins: int = 50,
                        save_path: Optional[str] = None) -> None:
    """
    绘制相量空间中的直方图分布
    
    对数据的每个时间通道进行相量变换，统计分布。
    
    参数:
        t: 时间轴 (纳秒)
        data: 光子计数数据
        frequency: 频率 (MHz)
        bins: 直方图bin数
        save_path: 保存路径
    """
    omega = 2 * np.pi * frequency
    
    g_vals = np.cos(omega * t)
    s_vals = np.sin(omega * t)
    
    g_mean = np.sum(g_vals * data) / np.sum(data)
    s_mean = np.sum(s_vals * data) / np.sum(data)
    
    g_weighted = g_vals
    s_weighted = s_vals
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    ax1 = axes[0]
    ax1.hist2d(g_weighted, s_weighted, bins=bins, weights=data, 
               cmap='hot_r', norm=plt.Normalize(vmin=0))
    ax1.plot(g_mean, s_mean, 'wo', markersize=15, markeredgecolor='red',
            markeredgewidth=3)
    ax1.set_xlabel('g')
    ax1.set_ylabel('s')
    ax1.set_title('Data Distribution in Phasor Space')
    
    ax2 = axes[1]
    ax2.plot(t, g_vals, 'g-', label='cos(ωt)')
    ax2.plot(t, s_vals, 'b-', label='sin(ωt)')
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Value')
    ax2.set_title('Reference Functions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    theta = np.linspace(0, np.pi, 200)
    g_circle = 0.5 + 0.5 * np.cos(theta)
    s_circle = 0.5 * np.sin(theta)
    ax3.plot(g_circle, s_circle, 'k-', linewidth=2, label='Universal Circle')
    ax3.plot(g_mean, s_mean, 'ro', markersize=15, label='Data Point')
    ax3.set_xlabel('g')
    ax3.set_ylabel('s')
    ax3.set_title('Phasor Plot')
    ax3.legend()
    ax3.set_aspect('equal')
    ax3.set_xlim(-0.1, 1.1)
    ax3.set_ylim(-0.1, 0.6)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Phasor histogram saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("FLIM Multi-Exponential Decay Fitting Demo")
    print("=" * 70)
    
    t = np.linspace(-2, 20, 512)
    
    irf_t0_true = 0.5
    irf_sigma_true = 0.15
    irf_true = gaussian_irf(t, irf_t0_true, irf_sigma_true)
    
    true_params = {
        'lifetimes': [0.5, 2.5],
        'amplitudes': [3000, 5000],
        'background': 20
    }
    
    print("\n[1] Generate synthetic FLIM data...")
    data = generate_synthetic_flim_data(t, true_params, irf_true, 
                                       noise=True, seed=42)
    print(f"    Total photons: {np.sum(data)}")
    
    print("\n[2] Phasor analysis (quick lifetime estimate)...")
    phasor_80 = phasor_analysis(t, data, frequency=80)
    print(f"    At 80 MHz:")
    print(f"      g = {phasor_80['g']:.4f}, s = {phasor_80['s']:.4f}")
    print(f"      Phase lifetime tau_phi = {phasor_80['tau_phase']:.4f} ns")
    print(f"      Modulation lifetime tau_M = {phasor_80['tau_mod']:.4f} ns")
    
    print("\n[3] IRF-corrected phasor analysis...")
    phasor_corr = phasor_analysis_irf_corrected(t, data, irf_true, frequency=80)
    print(f"    Corrected:")
    print(f"      g_corr = {phasor_corr['g']:.4f}, s_corr = {phasor_corr['s']:.4f}")
    print(f"      tau_phi = {phasor_corr['tau_phase']:.4f} ns")
    print(f"      tau_M = {phasor_corr['tau_mod']:.4f} ns")
    
    print("\n[4] Multi-frequency phasor analysis...")
    freqs = [20, 40, 80, 160]
    multi_phasor = multi_frequency_phasor(t, data, freqs)
    for freq, ph in zip(freqs, multi_phasor):
        print(f"    {freq:4d} MHz: g={ph['g']:.4f}, s={ph['s']:.4f}, "
              f"tau_phi={ph['tau_phase']:.4f} ns, tau_M={ph['tau_mod']:.4f} ns")
    
    print("\n[5] Estimate fractions from phasor...")
    frac = phasor_fraction_estimate(phasor_80['g'], phasor_80['s'], 
                                   tau1=0.5, tau2=2.5, frequency=80)
    print(f"    If tau1=0.5ns, tau2=2.5ns:")
    print(f"      f1 = {frac['f1']*100:.1f}%, f2 = {frac['f2']*100:.1f}%")
    
    print("\n[6] Two-exponential fit with Poisson MLE...")
    fit_mle = fit_with_automatic_initial(t, data, n_components=2, 
                                         irf=irf_true, fit_irf=True,
                                         method="poisson_mle")
    
    print("\n[7] Plot phasor diagram...")
    plot_phasor([phasor_80], data=data, t=t,
               title="FLIM Phasor Plot (80 MHz)",
               save_path="flim_phasor.png")
    
    print("\n[8] Plot multi-frequency phasor diagram...")
    plot_phasor_multifrequency(t, data, frequencies=freqs,
                              save_path="flim_phasor_multifreq.png")
    
    print("\n[9] Plot phasor histogram...")
    phasor_histogram_2d(t, data, frequency=80,
                       save_path="flim_phasor_histogram.png")
    
    print("\n[10] Plot fit results (Poisson MLE)...")
    plot_fit_results(t, data, fit_mle, irf=irf_true,
                    title="FLIM Two-Exponential Fit [Poisson MLE]",
                    save_path="flim_fit_mle.png")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
    
    print("""
Usage Examples:
---------
from flim_fitting import *

# 1. Prepare data
t = np.linspace(-2, 20, 512)
data = your_histogram_data  # photon counts

# 2. Define IRF (or measure experimentally)
irf = gaussian_irf(t, t0=0.5, sigma=0.15)

# ===== Phasor Analysis (Model-Free) =====
# Quick phasor analysis at 80 MHz
phasor = phasor_analysis(t, data, frequency=80)
print(f"g={phasor['g']:.4f}, s={phasor['s']:.4f}")
print(f"tau_phi={phasor['tau_phase']:.2f} ns")
print(f"tau_M={phasor['tau_mod']:.2f} ns")

# IRF-corrected phasor analysis
phasor_corr = phasor_analysis_irf_corrected(t, data, irf, frequency=80)

# Multi-frequency analysis
results = multi_frequency_phasor(t, data, frequencies=[20, 40, 80, 160])

# Estimate fractions from phasor
frac = phasor_fraction_estimate(phasor['g'], phasor['s'], 
                               tau1=0.5, tau2=2.5, frequency=80)

# Plot phasor diagram
plot_phasor([phasor], title="Phasor Plot")

# ===== Model-Based Fitting =====
# Fit with Poisson MLE (recommended for low photon counts)
result = fit_with_automatic_initial(t, data, n_components=2, 
                                     irf=irf, method="poisson_mle")

# View results
print(f"Lifetimes: {result['lifetimes']}")
print(f"Fractions: {result['fractions']}")

# Visualize
plot_fit_results(t, data, result, irf=irf)
""")
