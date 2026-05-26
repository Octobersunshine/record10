import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares

R = 8.314
T = 353.15
F = 96485
n = 2

def cpe_impedance(f, Q, n_cpe):
    """
    常相位元件（CPE）阻抗
    Z_CPE = 1 / [(jω)^n * Q]
    """
    omega = 2 * np.pi * f
    return 1.0 / ((1j * omega)**n_cpe * Q)

def warburg_impedance(f, sigma):
    """
    无限Warburg阻抗（半无限扩散）
    Z_W = sigma * (1 - j) / sqrt(omega)
    """
    omega = 2 * np.pi * f
    return sigma * (1 - 1j) / np.sqrt(omega)

def finite_warburg_impedance(f, sigma, delta, D_eff):
    """
    有限Warburg阻抗（有边界扩散）
    Z_W = sigma * (1 - j) * tanh(sqrt(j * omega * delta^2 / D_eff)) / sqrt(omega)
    """
    omega = 2 * np.pi * f
    tau = delta**2 / D_eff
    return sigma * (1 - 1j) * np.tanh(np.sqrt(1j * omega * tau)) / np.sqrt(omega)

def eis_equivalent_circuit(f, R_ohm, R_ct, Q_dl, n_dl, sigma_w):
    """
    PEMFC典型EIS等效电路：RΩ + (Rct || CPEdl) + W
    - R_ohm: 欧姆电阻（膜+接触电阻）
    - R_ct: 电荷转移电阻（催化剂活性）
    - Q_dl, n_dl: 双电层CPE参数
    - sigma_w: Warburg系数（质量传输）
    """
    Z_ohm = R_ohm
    Z_cpe = cpe_impedance(f, Q_dl, n_dl)
    Z_parallel = (R_ct * Z_cpe) / (R_ct + Z_cpe)
    Z_w = warburg_impedance(f, sigma_w)
    
    return Z_ohm + Z_parallel + Z_w

def eis_impedance_real_imag(f, R_ohm, R_ct, Q_dl, n_dl, sigma_w):
    """返回实部和虚部，用于拟合"""
    Z = eis_equivalent_circuit(f, R_ohm, R_ct, Q_dl, n_dl, sigma_w)
    return np.concatenate([Z.real, Z.imag])

def generate_eis_data(R_ohm=0.10, R_ct=0.20, Q_dl=0.5, n_dl=0.85, sigma_w=0.03, 
                      f_min=0.1, f_max=1e4, n_points=50, noise=0.02):
    """生成带噪声的示例EIS数据"""
    f = np.logspace(np.log10(f_min), np.log10(f_max), n_points)
    Z = eis_equivalent_circuit(f, R_ohm, R_ct, Q_dl, n_dl, sigma_w)
    
    noise_real = np.random.normal(0, noise * R_ct, size=len(f))
    noise_imag = np.random.normal(0, noise * R_ct, size=len(f))
    
    Z_noisy = Z + noise_real + 1j * noise_imag
    
    return f, Z_noisy

def smooth_log(x, epsilon=1e-6):
    """平滑对数函数，避免x接近0时发散"""
    return np.log(np.maximum(x, epsilon))

def concentration_loss_smooth(i, j_L, A_conc, transition_ratio=0.6):
    """
    稳定的浓差极化损失函数，基于对数模型但在高电流区使用平滑渐近
    
    当 i/j_L <= transition_ratio 时:
        η_conc = -A_conc * ln(1 - i/j_L)
    当 i/j_L > transition_ratio 时:
        使用四阶多项式平滑过渡到线性渐近，保证斜率连续且有界
    
    参数:
    transition_ratio: 对数模型的最大适用比例 (0, 1)
    """
    i_rel = np.clip(i / j_L, 1e-6, 0.999)
    
    eta = np.zeros_like(i_rel)
    
    mask_log = i_rel <= transition_ratio
    eta[mask_log] = -A_conc * smooth_log(1.0 - i_rel[mask_log])
    
    mask_asym = i_rel > transition_ratio
    if np.any(mask_asym):
        x_asym = i_rel[mask_asym]
        x_t = transition_ratio
        
        eta_t = -A_conc * smooth_log(1.0 - x_t)
        d_eta_dx_t = A_conc / (1.0 - x_t)
        d2_eta_dx2_t = A_conc / (1.0 - x_t)**2
        
        max_slope = d_eta_dx_t * 2.0
        
        x_end = 0.99
        target_eta_end = eta_t + max_slope * (x_end - x_t)
        
        a = np.array([
            [x_t**4, x_t**3, x_t**2, x_t, 1],
            [4*x_t**3, 3*x_t**2, 2*x_t, 1, 0],
            [12*x_t**2, 6*x_t, 2, 0, 0],
            [x_end**4, x_end**3, x_end**2, x_end, 1],
            [4*x_end**3, 3*x_end**2, 2*x_end, 1, 0]
        ])
        b = np.array([eta_t, d_eta_dx_t, d2_eta_dx2_t, target_eta_end, max_slope])
        
        try:
            coeffs = np.linalg.solve(a, b)
            
            eta_poly = (coeffs[0] * x_asym**4 + coeffs[1] * x_asym**3 + 
                       coeffs[2] * x_asym**2 + coeffs[3] * x_asym + coeffs[4])
            
            mask_very_high = x_asym > x_end
            if np.any(mask_very_high):
                x_vh = x_asym[mask_very_high]
                eta_poly[mask_very_high] = target_eta_end + max_slope * (x_vh - x_end)
            
            if len(x_asym) > 1:
                d_poly = np.gradient(eta_poly, x_asym)
                if np.any(d_poly > max_slope * 1.1):
                    scale = max_slope * 1.1 / np.max(d_poly)
                    eta_poly = eta_t + (eta_poly - eta_t) * scale
            
            eta[mask_asym] = eta_poly
        except np.linalg.LinAlgError:
            eta[mask_asym] = eta_t + d_eta_dx_t * (x_asym - x_t)
    
    return eta

def pemfc_voltage(i, E_ocv, b, R_ohm, j_L, A_conc):
    """
    改进的PEMFC极化曲线半经验模型
    V = E_ocv - b*log10(i/i0) - R_ohm*i - eta_conc
    
    浓差极化使用平滑的 ln(1 - i/j_L) 模型，在高电流区不会发散
    
    参数:
    i: 电流密度 (A/cm²)
    E_ocv: 开路电压 (V)
    b: Tafel斜率 (V/decade)
    R_ohm: 面积比电阻 (Ω·cm²)
    j_L: 极限电流密度 (A/cm²)
    A_conc: 浓差极化系数
    """
    i_safe = np.maximum(i, 1e-6)
    
    eta_act = b * np.log10(i_safe / 0.0001)
    eta_ohm = R_ohm * i_safe
    eta_conc = concentration_loss_smooth(i_safe, j_L, A_conc, transition_ratio=0.6)
    
    return E_ocv - eta_act - eta_ohm - eta_conc

def pemfc_voltage_improved(i, E_ocv, A, B, C):
    """
    改进的三参数模型，稳定性更好
    V = E_ocv - A*ln(i) - B*i - C*i^2
    """
    i_safe = np.maximum(i, 1e-6)
    return E_ocv - A * np.log(i_safe) - B * i_safe - C * i_safe**2

def calculate_all_losses(i, E_ocv, b, R_ohm, j_L, A_conc):
    """计算各损失项"""
    i_safe = np.maximum(i, 1e-6)
    eta_act = b * np.log10(i_safe / 0.0001)
    eta_ohm = R_ohm * i_safe
    eta_conc = concentration_loss_smooth(i_safe, j_L, A_conc, transition_ratio=0.6)
    return eta_act, eta_ohm, eta_conc

def load_experimental_data(filename=None):
    """
    加载实验数据
    如果没有提供文件名，则生成示例数据
    """
    if filename is None:
        print("  使用内置示例数据...")
        i_data = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 
                           0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
        
        params_true = [0.95, 0.06, 0.10, 2.5, 0.05]
        v_true = pemfc_voltage(i_data, *params_true)
        noise = np.random.normal(0, 0.01, size=len(v_true))
        v_data = np.maximum(v_true + noise, 0.2)
        return i_data, v_data
    else:
        print(f"  从文件加载数据: {filename}")
        data = np.loadtxt(filename, skiprows=1)
        return data[:, 0], data[:, 1]

def estimate_initial_params(i_data, v_data):
    """智能估计初始参数"""
    E_ocv_est = np.max(v_data) * 1.05
    E_ocv_est = np.clip(E_ocv_est, 0.9, 1.15)
    
    mid_idx = len(i_data) // 2
    i_mid = i_data[mid_idx:mid_idx+3]
    v_mid = v_data[mid_idx:mid_idx+3]
    coeffs = np.polyfit(i_mid, v_mid, 1)
    R_ohm_est = abs(coeffs[0])
    R_ohm_est = np.clip(R_ohm_est, 0.02, 0.3)
    
    b_est = 0.06
    
    max_i = np.max(i_data)
    j_L_est = max_i * 3.0
    j_L_est = np.clip(j_L_est, max_i * 2.5, max_i * 6.0)
    
    A_conc_est = 0.05
    
    return [E_ocv_est, b_est, R_ohm_est, j_L_est, A_conc_est]

def residuals(params, i, v, lambda_reg=1e-4):
    """
    残差函数，包含L2正则化项以提高稳定性
    对浓差极化参数施加更强的正则化约束
    """
    E_ocv, b, R_ohm, j_L, A_conc = params
    max_i = np.max(i)
    
    v_pred = pemfc_voltage(i, E_ocv, b, R_ohm, j_L, A_conc)
    residuals = v - v_pred
    
    j_L_target = max_i * 3.0
    j_L_penalty = max(0.0, j_L_target - j_L) * 10.0
    
    reg_terms = np.array([
        lambda_reg * (E_ocv - 1.0),
        lambda_reg * (b - 0.06) * 10.0,
        lambda_reg * (R_ohm - 0.1) * 5.0,
        lambda_reg * (j_L - j_L_target) * 2.0 + j_L_penalty,
        lambda_reg * (A_conc - 0.05) * 10.0
    ])
    
    return np.concatenate([residuals, reg_terms])

def huber_residuals(params, i, v, delta=0.02):
    """
    Huber损失函数，对异常值不敏感
    """
    E_ocv, b, R_ohm, j_L, A_conc = params
    v_pred = pemfc_voltage(i, E_ocv, b, R_ohm, j_L, A_conc)
    r = v - v_pred
    
    huber = np.where(np.abs(r) <= delta,
                     0.5 * r**2,
                     delta * (np.abs(r) - 0.5 * delta))
    return huber

def compute_covariance(result, i, v):
    """
    从最小二乘结果计算协方差矩阵
    """
    y_pred = pemfc_voltage(i, *result.x)
    residuals = v - y_pred
    sse = np.sum(residuals**2)
    dof = len(v) - len(result.x)
    mse = sse / dof if dof > 0 else sse
    
    J = result.jac
    try:
        H = J.T @ J
        pcov = np.linalg.inv(H) * mse
    except np.linalg.LinAlgError:
        pcov = np.linalg.pinv(H) * mse
    
    return pcov

def fit_pemfc_data(i_data, v_data):
    """
    拟合PEMFC极化曲线，使用鲁棒的最小二乘法
    包含正则化和Huber损失以提高稳定性
    """
    
    mask = (i_data > 0) & (v_data > 0.2)
    i_fit = i_data[mask]
    v_fit = v_data[mask]
    
    try:
        initial_guess = estimate_initial_params(i_fit, v_fit)
        max_i = np.max(i_fit)
        bounds_lower = [0.85, 0.02, 0.02, max_i * 2.0, 0.005]
        bounds_upper = [1.15, 0.12, 0.4, max_i * 10.0, 0.2]
        
        result = least_squares(
            residuals,
            x0=initial_guess,
            args=(i_fit, v_fit, 1e-4),
            bounds=(bounds_lower, bounds_upper),
            method='trf',
            max_nfev=200000,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            loss='huber',
            f_scale=0.02
        )
        
        if result.success or result.status in [1, 2, 3, 4]:
            popt = result.x
            pcov = compute_covariance(result, i_fit, v_fit)
            perr = np.sqrt(np.diag(np.abs(pcov)))
            model_type = 'full'
            
            v_pred = pemfc_voltage(i_fit, *popt)
            r2 = 1 - np.sum((v_fit - v_pred)**2) / np.sum((v_fit - np.mean(v_fit))**2)
            
            if r2 < 0.95:
                print("  初始拟合质量不足，使用Huber损失重试...")
                result_huber = least_squares(
                    huber_residuals,
                    x0=popt,
                    args=(i_fit, v_fit),
                    bounds=(bounds_lower, bounds_upper),
                    method='trf',
                    max_nfev=100000
                )
                if result_huber.success or result_huber.status in [1, 2, 3, 4]:
                    popt = result_huber.x
                    pcov = compute_covariance(result_huber, i_fit, v_fit)
                    perr = np.sqrt(np.diag(np.abs(pcov)))
            
            return popt, perr, model_type
        else:
            raise RuntimeError("Fit failed")
    except Exception as e:
        print(f"  完整模型拟合失败 ({str(e)})，使用简化模型...")
        initial_guess_simple = [1.0, 0.05, 0.10, 0.02]
        bounds_simple = ([0.8, 0.01, 0.02, 0.001],
                         [1.2, 0.15, 0.3, 0.1])
        
        popt, pcov = curve_fit(pemfc_voltage_improved, i_fit, v_fit,
                               p0=initial_guess_simple, bounds=bounds_simple, maxfev=50000)
        perr = np.sqrt(np.diag(pcov))
        model_type = 'simple'
        return popt, perr, model_type

def eis_residuals(params, f, Z_exp):
    """EIS拟合残差函数"""
    R_ohm, R_ct, Q_dl, n_dl, sigma_w = params
    Z_model = eis_equivalent_circuit(f, R_ohm, R_ct, Q_dl, n_dl, sigma_w)
    return np.concatenate([Z_exp.real - Z_model.real, Z_exp.imag - Z_model.imag])

def fit_eis_data(f, Z_data):
    """拟合EIS数据"""
    Z_high_freq_real = Z_data.real[np.argmax(f)]
    R_ohm_est = max(0.02, Z_high_freq_real * 0.95)
    
    Z_low_freq_real = np.max(Z_data.real)
    R_ct_est = max(0.05, Z_low_freq_real - R_ohm_est)
    
    f_mid = f[len(f)//2]
    Q_dl_est = 1.0 / (2 * np.pi * f_mid * R_ct_est)
    n_dl_est = 0.85
    
    sigma_w_est = 0.05
    
    initial_guess = [R_ohm_est, R_ct_est, Q_dl_est, n_dl_est, sigma_w_est]
    
    bounds_lower = [0.01, 0.01, 1e-4, 0.5, 0.001]
    bounds_upper = [0.5, 2.0, 10.0, 1.0, 1.0]
    
    try:
        result = least_squares(
            eis_residuals,
            x0=initial_guess,
            args=(f, Z_data),
            bounds=(bounds_lower, bounds_upper),
            method='trf',
            max_nfev=100000,
            ftol=1e-10,
            xtol=1e-10
        )
        
        if result.success or result.status in [1, 2, 3, 4]:
            popt = result.x
            J = result.jac
            sse = np.sum(result.fun**2)
            dof = len(f) * 2 - len(popt)
            mse = sse / dof if dof > 0 else sse
            try:
                pcov = np.linalg.inv(J.T @ J) * mse
            except np.linalg.LinAlgError:
                pcov = np.linalg.pinv(J.T @ J) * mse
            perr = np.sqrt(np.diag(np.abs(pcov)))
            return popt, perr, True
        else:
            raise RuntimeError("EIS fit failed")
    except Exception as e:
        print(f"  EIS拟合警告: {str(e)}")
        return initial_guess, np.zeros_like(initial_guess), False

def joint_fit_objective(params, i_data, v_data, f_data, Z_data, weight_pol=1.0, weight_eis=1.0):
    """
    极化曲线和EIS联合拟合目标函数
    params: [E_ocv, b, R_ohm, j_L, A_conc, R_ct, Q_dl, n_dl, sigma_w]
    """
    E_ocv, b, R_ohm_pol, j_L, A_conc, R_ct, Q_dl, n_dl, sigma_w = params
    
    v_pred = pemfc_voltage(i_data, E_ocv, b, R_ohm_pol, j_L, A_conc)
    pol_residuals = (v_data - v_pred) * weight_pol
    
    Z_model = eis_equivalent_circuit(f_data, R_ohm_pol, R_ct, Q_dl, n_dl, sigma_w)
    eis_residuals_real = (Z_data.real - Z_model.real) * weight_eis
    eis_residuals_imag = (Z_data.imag - Z_model.imag) * weight_eis
    
    reg_R_ohm = 0.01 * (R_ohm_pol - 0.1)
    
    return np.concatenate([pol_residuals, eis_residuals_real, eis_residuals_imag, [reg_R_ohm]])

def fit_joint_pemfc_eis(i_data, v_data, f_data, Z_data):
    """极化曲线和EIS联合拟合"""
    print("\n  执行极化曲线与EIS联合拟合...")
    
    pol_popt, _, _ = fit_pemfc_data(i_data, v_data)
    eis_popt, _, eis_success = fit_eis_data(f_data, Z_data)
    
    if len(pol_popt) >= 3 and eis_success:
        initial_guess = [
            pol_popt[0], pol_popt[1], pol_popt[2], pol_popt[3], pol_popt[4],
            eis_popt[1], eis_popt[2], eis_popt[3], eis_popt[4]
        ]
    else:
        initial_guess = [0.95, 0.06, 0.10, 3.0, 0.05, 0.20, 0.5, 0.85, 0.03]
    
    max_i = np.max(i_data)
    bounds_lower = [0.85, 0.02, 0.02, max_i * 2.0, 0.005, 0.01, 1e-4, 0.5, 0.001]
    bounds_upper = [1.15, 0.12, 0.4, max_i * 10.0, 0.2, 2.0, 10.0, 1.0, 1.0]
    
    try:
        result = least_squares(
            joint_fit_objective,
            x0=initial_guess,
            args=(i_data, v_data, f_data, Z_data, 1.0, 0.5),
            bounds=(bounds_lower, bounds_upper),
            method='trf',
            max_nfev=200000,
            ftol=1e-10,
            xtol=1e-10
        )
        
        if result.success or result.status in [1, 2, 3, 4]:
            popt = result.x
            pol_params = popt[:5]
            eis_params = np.array([popt[2], popt[5], popt[6], popt[7], popt[8]])
            
            J = result.jac
            sse = np.sum(result.fun**2)
            dof = len(i_data) + len(f_data) * 2 - len(popt)
            mse = sse / dof if dof > 0 else sse
            try:
                pcov = np.linalg.inv(J.T @ J) * mse
            except np.linalg.LinAlgError:
                pcov = np.linalg.pinv(J.T @ J) * mse
            perr = np.sqrt(np.diag(np.abs(pcov)))
            
            pol_perr = perr[:5]
            eis_perr = np.array([perr[2], perr[5], perr[6], perr[7], perr[8]])
            
            return pol_params, pol_perr, eis_params, eis_perr, True
        else:
            raise RuntimeError("Joint fit failed")
    except Exception as e:
        print(f"  联合拟合失败 ({str(e)})，使用单独拟合结果...")
        return pol_popt, np.zeros_like(pol_popt), eis_popt, np.zeros_like(eis_popt), False

def calculate_degradation_metrics(eis_params, reference_params=None):
    """
    计算催化剂老化诊断指标
    
    参考基准值（典型新电池）:
    - R_ohm_ref: 0.08-0.12 Ω·cm²
    - R_ct_ref: 0.15-0.25 Ω·cm²
    - sigma_w_ref: 0.02-0.05 V·s^0.5/cm
    - n_dl_ref: 0.85-0.95 (越接近1表示双电层越理想)
    """
    R_ohm, R_ct, Q_dl, n_dl, sigma_w = eis_params
    
    if reference_params is None:
        reference_params = {
            'R_ohm': 0.10,
            'R_ct': 0.20,
            'sigma_w': 0.03,
            'n_dl': 0.90,
            'Q_dl': 0.5
        }
    
    metrics = {}
    
    metrics['R_ohm_increase'] = ((R_ohm - reference_params['R_ohm']) / reference_params['R_ohm']) * 100
    metrics['R_ct_increase'] = ((R_ct - reference_params['R_ct']) / reference_params['R_ct']) * 100
    metrics['sigma_w_increase'] = ((sigma_w - reference_params['sigma_w']) / reference_params['sigma_w']) * 100
    metrics['n_dl_decrease'] = ((reference_params['n_dl'] - n_dl) / reference_params['n_dl']) * 100
    
    metrics['ECSA_loss'] = ((reference_params['Q_dl'] - Q_dl) / reference_params['Q_dl']) * 100
    
    if metrics['R_ct_increase'] < 20:
        metrics['catalyst_activity'] = 'EXCELLENT'
    elif metrics['R_ct_increase'] < 50:
        metrics['catalyst_activity'] = 'GOOD'
    elif metrics['R_ct_increase'] < 100:
        metrics['catalyst_activity'] = 'MODERATE'
    else:
        metrics['catalyst_activity'] = 'SEVERE'
    
    if metrics['R_ohm_increase'] < 15:
        metrics['membrane_condition'] = 'GOOD'
    elif metrics['R_ohm_increase'] < 40:
        metrics['membrane_condition'] = 'MODERATE'
    else:
        metrics['membrane_condition'] = 'DEGRADED'
    
    if metrics['sigma_w_increase'] < 30:
        metrics['mass_transport'] = 'GOOD'
    elif metrics['sigma_w_increase'] < 80:
        metrics['mass_transport'] = 'MODERATE'
    else:
        metrics['mass_transport'] = 'POOR'
    
    overall_score = 100 - 0.3 * metrics['R_ct_increase'] - 0.2 * metrics['R_ohm_increase'] - 0.2 * metrics['sigma_w_increase']
    overall_score = max(0, min(100, overall_score))
    metrics['overall_health'] = overall_score
    
    if overall_score > 80:
        metrics['health_level'] = 'HEALTHY'
    elif overall_score > 60:
        metrics['health_level'] = 'GOOD'
    elif overall_score > 40:
        metrics['health_level'] = 'MODERATE'
    elif overall_score > 20:
        metrics['health_level'] = 'POOR'
    else:
        metrics['health_level'] = 'SEVERE'
    
    return metrics

def print_degradation_diagnosis(metrics, eis_params):
    """打印催化剂老化诊断报告"""
    R_ohm, R_ct, Q_dl, n_dl, sigma_w = eis_params
    
    print("\n" + "="*70)
    print("           PEMFC CATALYST DEGRADATION DIAGNOSIS")
    print("="*70)
    
    print(f"\n  FITTED EIS PARAMETERS")
    print("-" * 50)
    print(f"  Ohmic Resistance (RΩ):     {R_ohm:.4f} Ω·cm²")
    print(f"  Charge Transfer R (Rct):   {R_ct:.4f} Ω·cm²")
    print(f"  Double Layer C (Q_dl):     {Q_dl:.4f} F/cm²·s^(n-1)")
    print(f"  CPE Exponent (n_dl):       {n_dl:.4f}")
    print(f"  Warburg Coefficient (σ_w): {sigma_w:.4f} V·s^0.5/cm")
    
    print(f"\n  DEGRADATION METRICS (vs. Reference)")
    print("-" * 50)
    print(f"  RΩ Increase:               {metrics['R_ohm_increase']:+.1f}%")
    print(f"  Rct Increase:              {metrics['R_ct_increase']:+.1f}%")
    print(f"  σ_w Increase:              {metrics['sigma_w_increase']:+.1f}%")
    print(f"  n_dl Decrease:             {metrics['n_dl_decrease']:+.1f}%")
    print(f"  ECSA Loss Estimate:        {metrics['ECSA_loss']:+.1f}%")
    
    print(f"\n  DIAGNOSIS RESULTS")
    print("-" * 50)
    print(f"  Catalyst Activity:         {metrics['catalyst_activity']}")
    print(f"  Membrane Condition:        {metrics['membrane_condition']}")
    print(f"  Mass Transport:            {metrics['mass_transport']}")
    print(f"  Overall Health Score:      {metrics['overall_health']:.1f}/100")
    print(f"  Health Level:              {metrics['health_level']}")
    
    print(f"\n  RECOMMENDATIONS")
    print("-" * 50)
    if metrics['R_ct_increase'] > 50:
        print("  ⚠ Catalyst degradation detected - Consider catalyst regeneration")
    if metrics['R_ohm_increase'] > 30:
        print("  ⚠ Membrane degradation detected - Check humidity and contaminants")
    if metrics['sigma_w_increase'] > 50:
        print("  ⚠ Mass transport issues detected - Check gas diffusion layer")
    if metrics['health_level'] == 'HEALTHY':
        print("  ✓ Cell in good condition - Continue normal operation")
    elif metrics['health_level'] == 'GOOD':
        print("  ✓ Cell performing well - Continue monitoring")
    
    print("="*70)

def plot_eis_results(f, Z_data, eis_params, fig=None):
    """绘制EIS奈奎斯特图和伯德图"""
    Z_fit = eis_equivalent_circuit(f, *eis_params)
    
    if fig is None:
        fig = plt.figure(figsize=(14, 6))
    
    gs = fig.add_gridspec(1, 2, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(Z_data.real, -Z_data.imag, color='navy', s=60, edgecolor='black', 
                alpha=0.8, zorder=5, label='Experimental')
    ax1.plot(Z_fit.real, -Z_fit.imag, 'crimson', linewidth=2.5, label='Fitted')
    
    for idx in [0, len(f)//4, len(f)//2, 3*len(f)//4, -1]:
        freq_label = f"{f[idx]:.1f} Hz"
        ax1.annotate(freq_label, (Z_data.real[idx], -Z_data.imag[idx]),
                    textcoords="offset points", xytext=(5, 10), fontsize=8)
    
    ax1.set_xlabel("Z' (Ω·cm²)")
    ax1.set_ylabel("-Z'' (Ω·cm²)")
    ax1.set_title("Nyquist Plot")
    ax1.legend(frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2_twin = ax2.twinx()
    
    mag_data = np.abs(Z_data)
    phase_data = np.angle(Z_data, deg=True)
    mag_fit = np.abs(Z_fit)
    phase_fit = np.angle(Z_fit, deg=True)
    
    ax2.semilogx(f, mag_data, 'o', color='navy', markersize=6, label='|Z| Data')
    ax2.semilogx(f, mag_fit, 'navy', linewidth=2, label='|Z| Fit')
    ax2_twin.semilogx(f, phase_data, 's', color='forestgreen', markersize=6, label='Phase Data')
    ax2_twin.semilogx(f, phase_fit, 'forestgreen', linewidth=2, label='Phase Fit')
    
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("|Z| (Ω·cm²)", color='navy')
    ax2_twin.set_ylabel("Phase (°)", color='forestgreen')
    ax2.set_title("Bode Plot")
    ax2.grid(True, alpha=0.3)
    
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
    
    return fig

def plot_final_results(i_data, v_data, popt, model_type):
    """绘制最终结果"""
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    
    if model_type == 'full':
        i_max_plot = np.max(i_data) * 1.2
        i_fit = np.linspace(0.01, i_max_plot, 300)
        v_fit = pemfc_voltage(i_fit, *popt)
        eta_act, eta_ohm, eta_conc = calculate_all_losses(i_fit, *popt)
        
        i_verify = np.linspace(np.max(i_data) * 0.9, i_max_plot, 50)
        v_verify = pemfc_voltage(i_verify, *popt)
        dv_di = np.gradient(v_verify, i_verify)
        has_unphysical_drop = np.any(dv_di < -0.5)
    else:
        i_fit = np.linspace(0.01, np.max(i_data) * 1.05, 200)
        v_fit = pemfc_voltage_improved(i_fit, *popt)
        eta_act = popt[1] * np.log(i_fit)
        eta_ohm = popt[2] * i_fit
        eta_conc = popt[3] * i_fit**2
    
    power = i_fit * v_fit
    max_power_idx = np.argmax(power)
    max_power = power[max_power_idx]
    i_mp = i_fit[max_power_idx]
    v_mp = v_fit[max_power_idx]
    
    ax1.scatter(i_data, v_data, color='navy', s=70, edgecolor='black', 
                alpha=0.8, zorder=5, label='Experimental Data')
    ax1.plot(i_fit, v_fit, 'crimson', linewidth=3, label='Fitted Curve')
    ax1.set_xlabel('Current Density (A/cm$^2$)')
    ax1.set_ylabel('Cell Voltage (V)')
    ax1.set_title('PEMFC Polarization Curve')
    ax1.legend(frameon=True, shadow=True)
    ax1.grid(True)
    ax1.set_ylim([0, 1.1])
    
    residuals = v_data - pemfc_voltage(i_data, *popt) if model_type == 'full' else \
                v_data - pemfc_voltage_improved(i_data, *popt)
    
    ax2.scatter(i_data, residuals, color='purple', s=50, edgecolor='black', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Current Density (A/cm$^2$)')
    ax2.set_ylabel('Residuals (V)')
    ax2.set_title('Fit Residuals')
    ax2.grid(True)
    
    ax3_twin = ax3.twinx()
    ax3.plot(i_fit, v_fit, 'navy', linewidth=2.5, label='Voltage')
    ax3_twin.plot(i_fit, power, 'forestgreen', linewidth=2.5, label='Power Density')
    ax3.scatter([i_mp], [v_mp], color='red', s=150, zorder=6, edgecolor='black', 
                label=f'MPP: {max_power:.3f} W/cm$^2$')
    ax3_twin.scatter([i_mp], [max_power], color='red', s=150, zorder=6, edgecolor='black')
    ax3.axvline(x=i_mp, color='gray', linestyle='--', alpha=0.6)
    
    ax3.fill_between(i_fit, 0, v_fit, alpha=0.15, color='navy')
    ax3_twin.fill_between(i_fit, 0, power, alpha=0.15, color='forestgreen')
    
    ax3.set_xlabel('Current Density (A/cm$^2$)')
    ax3.set_ylabel('Voltage (V)', color='navy')
    ax3_twin.set_ylabel('Power Density (W/cm$^2$)', color='forestgreen')
    ax3.set_title('Polarization Curve and Power Density with Maximum Power Point')
    ax3.grid(True)
    ax3.set_ylim([0, 1.1])
    
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', 
               frameon=True, shadow=True)
    
    ax3_inset = ax3.inset_axes([0.05, 0.55, 0.35, 0.4])
    ax3_inset.plot(i_fit, eta_act, 'orange', linewidth=2, label='Activation')
    ax3_inset.plot(i_fit, eta_ohm, 'blue', linewidth=2, label='Ohmic')
    ax3_inset.plot(i_fit, eta_conc, 'purple', linewidth=2, label='Concentration')
    ax3_inset.set_title('Loss Contributions', fontsize=10)
    ax3_inset.set_xlabel('i (A/cm$^2$)', fontsize=8)
    ax3_inset.set_ylabel('η (V)', fontsize=8)
    ax3_inset.legend(fontsize=7)
    ax3_inset.tick_params(axis='both', labelsize=7)
    ax3_inset.grid(True, alpha=0.3)
    
    plt.savefig('pemfc_final_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return max_power, i_mp, v_mp, power, v_fit

def verify_model_stability(popt, model_type, i_data):
    """验证模型在高电流区的稳定性"""
    if model_type != 'full':
        return "N/A (simplified model)"
    
    max_i = np.max(i_data)
    i_test = np.linspace(max_i * 0.9, max_i * 1.5, 100)
    v_test = pemfc_voltage(i_test, *popt)
    
    dv_di = np.gradient(v_test, i_test)
    max_negative_slope = np.min(dv_di)
    
    is_finite = np.all(np.isfinite(v_test))
    is_monotonic = np.all(np.diff(v_test) <= 0.01)
    slope_reasonable = max_negative_slope > -5.0
    
    is_stable = is_finite and is_monotonic and slope_reasonable
    
    if is_stable:
        if max_negative_slope > -0.5:
            stability_msg = "EXCELLENT"
        elif max_negative_slope > -1.0:
            stability_msg = "GOOD"
        else:
            stability_msg = "STABLE"
    else:
        stability_msg = "POTENTIALLY UNSTABLE"
    
    details = []
    if not is_finite:
        details.append("non-finite values")
    if not is_monotonic:
        details.append("non-monotonic behavior")
    if not slope_reasonable:
        details.append(f"excessive slope ({max_negative_slope:.2f})")
    
    detail_str = f" [{'; '.join(details)}]" if details else ""
    
    return f"{stability_msg} (max dV/di = {max_negative_slope:.4f} V·cm²/A){detail_str}"

def print_summary(popt, perr, model_type, max_power, i_mp, v_mp, i_data, v_data):
    """打印拟合结果摘要"""
    print("\n" + "="*70)
    print("           PEMFC POLARIZATION CURVE FITTING RESULTS")
    print("="*70)
    
    if model_type == 'full':
        print(f"\n  Model Type: Full Semi-Empirical Model (Smooth ln(1-i/j_L))")
        print("-" * 50)
        print(f"  Open Circuit Voltage (E_ocv): {popt[0]:.4f} ± {perr[0]:.4f} V")
        print(f"  Tafel Slope (b):             {popt[1]:.4f} ± {perr[1]:.4f} V/dec")
        print(f"  Ohmic Resistance (R_ohm):    {popt[2]:.4f} ± {perr[2]:.4f} Ω·cm²")
        print(f"  Limiting Current (j_L):      {popt[3]:.4f} ± {perr[3]:.4f} A/cm²")
        print(f"  Concentration Coefficient:   {popt[4]:.4f} ± {perr[4]:.4f} V")
        
        j_L, A_conc = popt[3], popt[4]
        max_i = np.max(i_data)
        print(f"\n  High Current Stability Check:")
        print(f"    j_L / max(i):            {j_L/max_i:.2f}")
        print(f"    Model Status:            {verify_model_stability(popt, model_type, i_data)}")
    else:
        print(f"\n  Model Type: Simplified Model")
        print("-" * 50)
        print(f"  Open Circuit Voltage (E_ocv): {popt[0]:.4f} ± {perr[0]:.4f} V")
        print(f"  Activation Coefficient (A):   {popt[1]:.4f} ± {perr[1]:.4f} V")
        print(f"  Ohmic Coefficient (B):        {popt[2]:.4f} ± {perr[2]:.4f} Ω·cm²")
        print(f"  Concentration Coefficient (C):{popt[3]:.4f} ± {perr[3]:.4f} Ω·cm^4/A")
    
    print("\n" + "-" * 50)
    print("  PERFORMANCE METRICS")
    print("-" * 50)
    print(f"  Peak Power Density:    {max_power:.4f} W/cm²")
    print(f"  Current Density @ MPP: {i_mp:.4f} A/cm²")
    print(f"  Voltage @ MPP:         {v_mp:.4f} V")
    
    v_pred = pemfc_voltage(i_data, *popt) if model_type == 'full' else \
             pemfc_voltage_improved(i_data, *popt)
    r2 = 1 - np.sum((v_data - v_pred)**2) / np.sum((v_data - np.mean(v_data))**2)
    rmse = np.sqrt(np.mean((v_data - v_pred)**2))
    
    print("\n" + "-" * 50)
    print("  FIT QUALITY")
    print("-" * 50)
    print(f"  R-squared (R²):  {r2:.6f}")
    print(f"  RMSE:            {rmse:.6f} V")
    print("=" * 70)

def load_eis_data(filename=None):
    """加载或生成EIS数据"""
    if filename is None:
        print("  使用内置示例EIS数据...")
        return generate_eis_data(
            R_ohm=0.12, R_ct=0.35, Q_dl=0.35, n_dl=0.78, sigma_w=0.06,
            noise=0.03
        )
    else:
        print(f"  从文件加载EIS数据: {filename}")
        data = np.loadtxt(filename, skiprows=1)
        f = data[:, 0]
        Z = data[:, 1] + 1j * data[:, 2]
        return f, Z

def plot_combined_results(i_data, v_data, popt_pol, f_eis, Z_eis, popt_eis, model_type):
    """绘制综合分析结果：极化曲线+EIS"""
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.linewidth': 1.2,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])
    ax7 = fig.add_subplot(gs[2, 2])
    
    if model_type == 'full':
        i_fit = np.linspace(0.01, np.max(i_data) * 1.2, 200)
        v_fit = pemfc_voltage(i_fit, *popt_pol)
        eta_act, eta_ohm, eta_conc = calculate_all_losses(i_fit, *popt_pol)
    else:
        i_fit = np.linspace(0.01, np.max(i_data) * 1.05, 200)
        v_fit = pemfc_voltage_improved(i_fit, *popt_pol)
        eta_act = popt_pol[1] * np.log(i_fit)
        eta_ohm = popt_pol[2] * i_fit
        eta_conc = popt_pol[3] * i_fit**2
    
    power = i_fit * v_fit
    max_power_idx = np.argmax(power)
    max_power = power[max_power_idx]
    i_mp = i_fit[max_power_idx]
    v_mp = v_fit[max_power_idx]
    
    ax1.scatter(i_data, v_data, color='navy', s=50, edgecolor='black', alpha=0.8, zorder=5)
    ax1.plot(i_fit, v_fit, 'crimson', linewidth=2.5)
    ax1.set_xlabel('Current Density (A/cm²)')
    ax1.set_ylabel('Cell Voltage (V)')
    ax1.set_title('Polarization Curve')
    ax1.grid(True)
    ax1.set_ylim([0, 1.1])
    
    residuals = v_data - (pemfc_voltage(i_data, *popt_pol) if model_type == 'full' else 
                          pemfc_voltage_improved(i_data, *popt_pol))
    ax2.scatter(i_data, residuals, color='purple', s=40, edgecolor='black', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Current Density (A/cm²)')
    ax2.set_ylabel('Residuals (V)')
    ax2.set_title('Fit Residuals')
    ax2.grid(True)
    
    ax3.plot(i_fit, eta_act, 'orange', linewidth=2, label='Activation')
    ax3.plot(i_fit, eta_ohm, 'blue', linewidth=2, label='Ohmic')
    ax3.plot(i_fit, eta_conc, 'purple', linewidth=2, label='Concentration')
    ax3.set_xlabel('Current Density (A/cm²)')
    ax3.set_ylabel('Overpotential (V)')
    ax3.set_title('Loss Contributions')
    ax3.legend(fontsize=9)
    ax3.grid(True)
    
    ax3_twin = ax4.twinx()
    ax4.plot(i_fit, v_fit, 'navy', linewidth=2.5, label='Voltage')
    ax3_twin.plot(i_fit, power, 'forestgreen', linewidth=2.5, label='Power Density')
    ax4.scatter([i_mp], [v_mp], color='red', s=120, zorder=6, edgecolor='black',
                label=f'MPP: {max_power:.3f} W/cm²')
    ax3_twin.scatter([i_mp], [max_power], color='red', s=120, zorder=6, edgecolor='black')
    ax4.axvline(x=i_mp, color='gray', linestyle='--', alpha=0.6)
    ax4.fill_between(i_fit, 0, v_fit, alpha=0.15, color='navy')
    ax3_twin.fill_between(i_fit, 0, power, alpha=0.15, color='forestgreen')
    ax4.set_xlabel('Current Density (A/cm²)')
    ax4.set_ylabel('Voltage (V)', color='navy')
    ax3_twin.set_ylabel('Power Density (W/cm²)', color='forestgreen')
    ax4.set_title('Polarization Curve and Power Density')
    ax4.grid(True)
    ax4.set_ylim([0, 1.1])
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
    
    Z_fit = eis_equivalent_circuit(f_eis, *popt_eis)
    ax5.scatter(Z_eis.real, -Z_eis.imag, color='navy', s=40, edgecolor='black', alpha=0.8, label='Data')
    ax5.plot(Z_fit.real, -Z_fit.imag, 'crimson', linewidth=2.5, label='Fit')
    for idx in [0, len(f_eis)//4, len(f_eis)//2, 3*len(f_eis)//4, -1]:
        ax5.annotate(f"{f_eis[idx]:.1f} Hz", (Z_eis.real[idx], -Z_eis.imag[idx]),
                    textcoords="offset points", xytext=(5, 8), fontsize=7)
    ax5.set_xlabel("Z' (Ω·cm²)")
    ax5.set_ylabel("-Z'' (Ω·cm²)")
    ax5.set_title("Nyquist Plot")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal', adjustable='box')
    
    ax6_twin = ax6.twinx()
    ax6.semilogx(f_eis, np.abs(Z_eis), 'o', color='navy', markersize=5, label='|Z| Data')
    ax6.semilogx(f_eis, np.abs(Z_fit), 'navy', linewidth=2, label='|Z| Fit')
    ax6_twin.semilogx(f_eis, np.angle(Z_eis, deg=True), 's', color='forestgreen', markersize=5, label='Phase Data')
    ax6_twin.semilogx(f_eis, np.angle(Z_fit, deg=True), 'forestgreen', linewidth=2, label='Phase Fit')
    ax6.set_xlabel("Frequency (Hz)")
    ax6.set_ylabel("|Z| (Ω·cm²)", color='navy')
    ax6_twin.set_ylabel("Phase (°)", color='forestgreen')
    ax6.set_title("Bode Plot")
    ax6.grid(True, alpha=0.3)
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=7)
    
    metrics = calculate_degradation_metrics(popt_eis)
    categories = ['Catalyst\nActivity', 'Membrane\nCondition', 'Mass\nTransport', 'Overall\nHealth']
    scores = [
        100 - min(100, metrics['R_ct_increase']),
        100 - min(100, metrics['R_ohm_increase'] * 2),
        100 - min(100, metrics['sigma_w_increase'] * 1.5),
        metrics['overall_health']
    ]
    colors = ['#2ecc71' if s >= 70 else '#f39c12' if s >= 40 else '#e74c3c' for s in scores]
    bars = ax7.bar(categories, scores, color=colors, edgecolor='black', linewidth=1.5)
    ax7.set_ylim([0, 100])
    ax7.set_ylabel("Health Score (%)")
    ax7.set_title("Degradation Diagnosis Dashboard")
    ax7.grid(True, axis='y', alpha=0.3)
    for bar, score in zip(bars, scores):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{score:.0f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.savefig('pemfc_eis_combined_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return max_power, i_mp, v_mp

def main():
    print("="*70)
    print("  PEMFC Polarization Curve + EIS Joint Analysis Program")
    print("          Catalyst Degradation Diagnosis")
    print("="*70)
    
    print("\n[1] Loading experimental data...")
    i_data, v_data = load_experimental_data()
    f_eis, Z_eis = load_eis_data()
    
    print("\n[2] Polarization Data Points:")
    print("-" * 50)
    print(f"  {'No.':>3} {'i (A/cm2)':>10} {'V (V)':>8}")
    print("-" * 50)
    for idx, (i, v) in enumerate(zip(i_data, v_data), 1):
        print(f"  {idx:>3} {i:>10.3f} {v:>8.4f}")
    
    print(f"\n[3] EIS Data Summary:")
    print("-" * 50)
    print(f"  Frequency range: {f_eis[0]:.2f} - {f_eis[-1]:.0f} Hz")
    print(f"  Number of points: {len(f_eis)}")
    print(f"  |Z| range: {np.min(np.abs(Z_eis)):.4f} - {np.max(np.abs(Z_eis)):.4f} Ω·cm²")
    
    print("\n[4] Performing joint fitting (Polarization + EIS)...")
    pol_popt, pol_perr, eis_popt, eis_perr, joint_success = \
        fit_joint_pemfc_eis(i_data, v_data, f_eis, Z_eis)
    
    model_type = 'full'
    if joint_success:
        print("  ✓ Joint fitting completed successfully!")
    else:
        print("  ⚠ Using separate fitting results")
    
    print("\n[5] Generating combined analysis plots...")
    max_power, i_mp, v_mp = plot_combined_results(
        i_data, v_data, pol_popt, f_eis, Z_eis, eis_popt, model_type
    )
    
    print_summary(pol_popt, pol_perr, model_type, max_power, i_mp, v_mp, i_data, v_data)
    
    print_degradation_diagnosis(calculate_degradation_metrics(eis_popt), eis_popt)
    
    print("\n[6] Saving results...")
    results_pol = np.column_stack([i_data, v_data])
    np.savetxt('experimental_data.txt', results_pol,
               header='Current_Density(A/cm2) Voltage(V)', fmt='%.6f')
    
    results_eis = np.column_stack([f_eis, Z_eis.real, Z_eis.imag])
    np.savetxt('eis_data.txt', results_eis,
               header='Frequency(Hz) Z_real(Ohm.cm2) Z_imag(Ohm.cm2)', fmt='%.6f')
    
    np.savetxt('fitted_eis_params.txt',
               np.column_stack([eis_popt, eis_perr]),
               header='EIS_Parameter Std_Error\nR_ohm R_ct Q_dl n_dl sigma_w',
               fmt='%.6f')
    
    print("\n" + "="*70)
    print("  ✓ Combined analysis completed successfully!")
    print("  Files saved:")
    print("    - pemfc_eis_combined_analysis.png")
    print("    - experimental_data.txt")
    print("    - eis_data.txt")
    print("    - fitted_eis_params.txt")
    print("="*70)

if __name__ == "__main__":
    main()
