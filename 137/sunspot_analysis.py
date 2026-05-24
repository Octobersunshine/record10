import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, integrate
from scipy.integrate import odeint
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class AlphaOmegaDynamo:
    """
    α-Ω太阳发电机模型（简化版）
    
    物理原理:
    - α效应: 对流层中的螺旋运动产生环向磁场→极向磁场
    - Ω效应: 较差自转拉伸极向磁场→环向磁场
    - 磁扩散: 欧姆耗散
    
    状态变量:
    - Bp: 极向磁场强度 (rms)
    - Bt: 环向磁场强度 (rms)
    - phase: 周期相位
    
    控制方程:
    dBp/dt = α*Bt - η*Bp
    dBt/dt = Ω*Bp - η*Bt
    """
    
    def __init__(self, alpha=0.8, Omega=1.1, eta=0.15, Bp0=0.5, Bt0=1.0):
        """
        初始化发电机模型参数
        
        参数:
            alpha: α效应系数 (螺旋运动强度)
            Omega: Ω效应系数 (较差自转强度)
            eta: 磁扩散系数
            Bp0: 初始极向磁场强度
            Bt0: 初始环向磁场强度
        """
        self.alpha = alpha
        self.Omega = Omega
        self.eta = eta
        self.Bp0 = Bp0
        self.Bt0 = Bt0
        
    def dynamo_equations(self, state, t):
        """
        α-Ω发电机控制方程
        
        d(Bp)/dt = α*Bt - η*Bp - γ*Bp*Bt² (非线性饱和)
        d(Bt)/dt = Ω*Bp - η*Bt - γ*Bt*Bp² (非线性饱和)
        """
        Bp, Bt = state
        gamma = 0.02
        
        dBp_dt = self.alpha * Bt - self.eta * Bp - gamma * Bp * Bt**2
        dBt_dt = self.Omega * Bp - self.eta * Bt - gamma * Bt * Bp**2
        
        return [dBp_dt, dBt_dt]
    
    def simulate(self, n_years, dt=0.1):
        """
        数值模拟发电机模型
        
        参数:
            n_years: 模拟年数
            dt: 时间步长
            
        返回:
            t: 时间数组
            Bp: 极向磁场
            Bt: 环向磁场
            sunspot_proxy: 太阳黑子数代理 (与环向磁场平方成正比)
        """
        t = np.arange(0, n_years, dt)
        initial_state = [self.Bp0, self.Bt0]
        
        solution = odeint(self.dynamo_equations, initial_state, t)
        
        Bp = solution[:, 0]
        Bt = solution[:, 1]
        
        sunspot_proxy = 160 * Bt**2
        sunspot_proxy = np.maximum(sunspot_proxy, 0)
        
        return t, Bp, Bt, sunspot_proxy
    
    def get_period(self):
        """
        计算发电机周期 (理论近似)
        T ≈ 2π / sqrt(alpha*Omega - eta²)
        """
        growth_rate = self.alpha * self.Omega - self.eta**2
        if growth_rate > 0:
            return 2 * np.pi / np.sqrt(growth_rate)
        else:
            return np.nan

class KalmanFilterDynamo:
    """
    卡尔曼滤波数据同化: 结合α-Ω发电机模型与观测数据
    
    状态向量: [Bp, Bt, alpha, Omega, eta]
    观测: 太阳黑子数 → 环向磁场 Bt
    """
    
    def __init__(self, dynamo_model, process_noise=1e-3, obs_noise=25.0):
        """
        初始化卡尔曼滤波
        
        参数:
            dynamo_model: AlphaOmegaDynamo实例
            process_noise: 过程噪声方差
            obs_noise: 观测噪声方差
        """
        self.dynamo = dynamo_model
        self.Q = process_noise * np.eye(2)
        self.R = obs_noise
        
        self.x = np.array([dynamo_model.Bp0, dynamo_model.Bt0])
        self.P = np.eye(2) * 0.1
        
        self.params = np.array([dynamo_model.alpha, dynamo_model.Omega, dynamo_model.eta])
        self.param_noise = 1e-4
        
    def predict(self, dt=1.0):
        """
        预测步骤: 使用发电机模型预测下一状态
        """
        Bp, Bt = self.x
        alpha, Omega, eta = self.params
        gamma = 0.02
        
        dBp = alpha * Bt - eta * Bp - gamma * Bp * Bt**2
        dBt = Omega * Bp - eta * Bt - gamma * Bt * Bp**2
        
        x_pred = np.array([Bp + dBp * dt, Bt + dBt * dt])
        
        F = self._jacobian(Bp, Bt, alpha, Omega, eta, gamma, dt)
        
        P_pred = F @ self.P @ F.T + self.Q
        
        return x_pred, P_pred
    
    def _jacobian(self, Bp, Bt, alpha, Omega, eta, gamma, dt):
        """
        计算雅可比矩阵 (线性化)
        """
        dBp_dBp = -eta - gamma * Bt**2
        dBp_dBt = alpha - 2 * gamma * Bp * Bt
        dBt_dBp = Omega - 2 * gamma * Bt * Bp
        dBt_dBt = -eta - gamma * Bp**2
        
        F = np.eye(2) + dt * np.array([
            [dBp_dBp, dBp_dBt],
            [dBt_dBp, dBt_dBt]
        ])
        
        return F
    
    def update(self, observation, x_pred, P_pred):
        """
        更新步骤: 使用观测修正预测
        """
        H = np.array([0, 1.0 / np.sqrt(160)])
        
        y_pred = H @ x_pred
        y_obs = np.sqrt(max(observation / 160, 0.01))
        
        innovation = y_obs - y_pred
        S = H @ P_pred @ H.T + self.R
        
        K = P_pred @ H.T / S
        
        self.x = x_pred + K * innovation
        self.P = (np.eye(2) - np.outer(K, H)) @ P_pred
        
        return innovation
    
    def assimilate_data(self, observations, years, dt=1.0):
        """
        完整的数据同化过程
        
        参数:
            observations: 观测的太阳黑子数
            years: 对应的年份
            
        返回:
            assimilated_states: 同化后的状态序列
            innovations: 新息序列
            params_history: 参数演化历史
        """
        n = len(observations)
        assimilated_states = np.zeros((n, 2))
        innovations = np.zeros(n)
        params_history = np.zeros((n, 3))
        
        for i in range(n):
            x_pred, P_pred = self.predict(dt)
            
            if not np.isnan(observations[i]):
                innov = self.update(observations[i], x_pred, P_pred)
                innovations[i] = innov
            else:
                self.x = x_pred
                self.P = P_pred
            
            assimilated_states[i] = self.x
            params_history[i] = self.params.copy()
            
            if i % 22 == 0 and i > 0:
                self._update_params(observations[max(0, i-22):i], assimilated_states[max(0, i-22):i])
        
        return assimilated_states, innovations, params_history
    
    def _update_params(self, obs_window, state_window):
        """
        自适应更新模型参数 (简化版)
        """
        if len(obs_window) < 11:
            return
        
        obs_mean = np.mean(obs_window[obs_window > 0])
        model_mean = 160 * np.mean(state_window[:, 1]**2)
        
        if model_mean > 0:
            ratio = obs_mean / model_mean
            if 0.7 < ratio < 1.3:
                self.params[0] *= ratio**0.1
                self.params[1] *= ratio**0.05

def run_dynamo_assimilation(data, forecast_years=30):
    """
    运行α-Ω发电机模型 + 卡尔曼滤波数据同化
    
    参数:
        data: 观测数据DataFrame
        forecast_years: 预测年数
    """
    print("\n" + "="*60)
    print("α-Ω太阳发电机模型 + 卡尔曼滤波数据同化")
    print("="*60)
    
    sunspots = data['SunspotNumber'].values
    years = data['Year'].values
    last_year = years[-1]
    
    print("\n" + "-"*60)
    print("1. 初始化发电机模型")
    print("-"*60)
    
    dynamo = AlphaOmegaDynamo(alpha=0.75, Omega=1.15, eta=0.12)
    theo_period = dynamo.get_period()
    print(f"模型参数:")
    print(f"  α效应系数: {dynamo.alpha:.3f} (螺旋运动)")
    print(f"  Ω效应系数: {dynamo.Omega:.3f} (较差自转)")
    print(f"  磁扩散系数: {dynamo.eta:.3f} (欧姆耗散)")
    print(f"  理论周期: {theo_period:.2f} 年")
    
    print("\n" + "-"*60)
    print("2. 卡尔曼滤波数据同化")
    print("-"*60)
    
    kf = KalmanFilterDynamo(dynamo, process_noise=5e-3, obs_noise=20.0)
    assimilated_states, innovations, params_history = kf.assimilate_data(sunspots, years)
    
    assimilated_Bp = assimilated_states[:, 0]
    assimilated_Bt = assimilated_states[:, 1]
    assimilated_ss = 160 * assimilated_Bt**2
    
    rmse = np.sqrt(np.nanmean((assimilated_ss - sunspots)**2))
    print(f"同化结果:")
    print(f"  同化RMSE: {rmse:.2f}")
    print(f"  新息(innovation)均值: {np.nanmean(innovations):.4f}")
    print(f"  新息标准差: {np.nanstd(innovations):.4f}")
    
    print("\n" + "-"*60)
    print("3. 模型参数演化分析")
    print("-"*60)
    print(f"  初始α: {params_history[0, 0]:.3f} → 最终α: {params_history[-1, 0]:.3f}")
    print(f"  初始Ω: {params_history[0, 1]:.3f} → 最终Ω: {params_history[-1, 1]:.3f}")
    
    print("\n" + "-"*60)
    print(f"4. 预测未来{forecast_years}年")
    print("-"*60)
    
    dynamo_final = AlphaOmegaDynamo(
        alpha=kf.params[0],
        Omega=kf.params[1],
        eta=kf.params[2],
        Bp0=kf.x[0],
        Bt0=kf.x[1]
    )
    
    t_future, Bp_future, Bt_future, ss_future = dynamo_final.simulate(forecast_years, dt=1.0)
    future_years = np.arange(last_year + 1, last_year + forecast_years + 1)
    
    peak_idx = np.argmax(ss_future[:20])
    peak_year = future_years[peak_idx]
    peak_value = ss_future[peak_idx]
    
    print(f"下一太阳活动周预测:")
    print(f"  峰值年份: {peak_year}")
    print(f"  峰值太阳黑子数: {peak_value:.1f}")
    print(f"  峰值极向磁场: {Bp_future[peak_idx]:.3f} G")
    print(f"  峰值环向磁场: {Bt_future[peak_idx]:.3f} G")
    
    print("\n" + "-"*60)
    print("5. 生成可视化图表")
    print("-"*60)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    ax = axes[0]
    ax.plot(years, sunspots, 'bo', label='观测数据', markersize=3, alpha=0.6)
    ax.plot(years, assimilated_ss, 'r-', label='同化结果', linewidth=2)
    ax.plot(future_years, ss_future, 'g--', label='模型预测', linewidth=2)
    ax.scatter(peak_year, peak_value, color='gold', s=200, zorder=5, edgecolor='black', linewidth=2)
    ax.annotate(f'预测峰值\n{peak_year}年\n{peak_value:.0f}', 
                xy=(peak_year, peak_value), 
                xytext=(peak_year + 3, peak_value + 20),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    ax.set_ylabel('太阳黑子数', fontsize=11)
    ax.set_title('α-Ω发电机模型同化与预测', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1990, peak_year + 10)
    
    ax = axes[1]
    ax.plot(years, assimilated_Bp, 'b-', label='极向磁场 Bp', linewidth=1.5)
    ax.plot(years, assimilated_Bt, 'r-', label='环向磁场 Bt', linewidth=1.5)
    ax.plot(future_years, Bp_future, 'b--', alpha=0.7)
    ax.plot(future_years, Bt_future, 'r--', alpha=0.7)
    ax.set_ylabel('磁场强度 (归一化)', fontsize=11)
    ax.set_title('磁场所演化 (极向场 vs 环向场)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1990, peak_year + 10)
    
    ax = axes[2]
    ax.plot(years, assimilated_Bp, assimilated_Bt, 'b-', linewidth=0.5, alpha=0.5)
    ax.plot(assimilated_Bp[-100:], assimilated_Bt[-100:], 'r-', linewidth=2, label='最近200年')
    ax.plot([assimilated_Bp[-1], Bp_future[0]], [assimilated_Bt[-1], Bt_future[0]], 'g--', alpha=0.5)
    ax.plot(Bp_future, Bt_future, 'g--', linewidth=2, label='预测轨迹')
    ax.scatter(assimilated_Bp[0], assimilated_Bt[0], c='blue', s=100, label='初始状态')
    ax.scatter(Bp_future[0], Bt_future[0], c='green', s=100, label='当前状态')
    ax.set_xlabel('极向磁场 Bp', fontsize=11)
    ax.set_ylabel('环向磁场 Bt', fontsize=11)
    ax.set_title('发电机吸引子 (Bp-Bt相空间)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('dynamo_assimilation.png', dpi=150, bbox_inches='tight')
    print("图表已保存: dynamo_assimilation.png")
    
    return {
        'peak_year': peak_year,
        'peak_value': peak_value,
        'assimilated': assimilated_ss,
        'forecast': ss_future,
        'future_years': future_years,
        'Bp': assimilated_Bp,
        'Bt': assimilated_Bt,
        'Bp_future': Bp_future,
        'Bt_future': Bt_future,
        'innovations': innovations,
        'params': kf.params
    }

def load_sunspot_data():
    """加载太阳黑子数据（从公开数据源或本地生成模拟数据）"""
    try:
        url = "https://www.sidc.be/SILSO/DATA/SN_y_tot_V2.0.txt"
        data = pd.read_csv(url, sep='\s+', header=None, names=['Year', 'SunspotNumber', 'Std', 'N'], usecols=['Year', 'SunspotNumber'])
        data['Year'] = data['Year'].astype(int)
        data = data[data['Year'] >= 1700]
        return data
    except:
        print("无法获取在线数据，使用模拟数据演示...")
        return generate_simulated_data()

def generate_simulated_data():
    """生成模拟的太阳黑子数据（基于11年周期）"""
    years = np.arange(1700, 2025)
    np.random.seed(42)
    
    cycle = 11.1
    phase = 2.0
    amplitude = 80
    trend = 0.05 * (years - 1700)
    noise = np.random.normal(0, 15, len(years))
    
    sunspots = amplitude * (0.5 + 0.5 * np.sin(2 * np.pi * (years - phase) / cycle)) + trend + noise
    sunspots = np.maximum(sunspots, 0)
    
    return pd.DataFrame({'Year': years, 'SunspotNumber': sunspots})

def spectral_analysis(data):
    """频谱分析检测周期"""
    print("\n" + "="*60)
    print("频谱分析结果")
    print("="*60)
    
    years = data['Year'].values
    sunspots = data['SunspotNumber'].values
    n = len(sunspots)
    
    freqs, psd = signal.welch(sunspots, fs=1.0, nperseg=min(256, n))
    
    periods = 1 / freqs[freqs > 0]
    psd_pos = psd[freqs > 0]
    
    peak_idx = np.argmax(psd_pos)
    dominant_period = periods[peak_idx]
    
    print(f"检测到的主导周期: {dominant_period:.2f} 年")
    print(f"理论太阳活动周期: 约 11 年")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    axes[0].plot(years, sunspots, 'b-', linewidth=1)
    axes[0].set_xlabel('年份')
    axes[0].set_ylabel('太阳黑子数')
    axes[0].set_title('太阳黑子数年平均值 (1700-至今)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].semilogx(periods, psd_pos, 'r-', linewidth=1.5)
    axes[1].axvline(dominant_period, color='g', linestyle='--', 
                    label=f'主导周期: {dominant_period:.1f}年')
    axes[1].axvline(11, color='orange', linestyle=':', label='理论周期: 11年')
    axes[1].set_xlabel('周期 (年)')
    axes[1].set_ylabel('功率谱密度')
    axes[1].set_title('太阳黑子数功率谱分析')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(2, 50)
    
    plt.tight_layout()
    plt.savefig('spectral_analysis.png', dpi=150, bbox_inches='tight')
    print("频谱分析图已保存: spectral_analysis.png")
    
    return dominant_period

def check_stationarity(series, diff_level=0):
    """检查时间序列平稳性"""
    result = adfuller(series)
    if diff_level == 0:
        print("\nADF平稳性检验 (原始序列):")
    else:
        print(f"\nADF平稳性检验 ({diff_level}阶差分后):")
    print(f"ADF统计量: {result[0]:.4f}")
    print(f"p值: {result[1]:.4f}")
    print(f"临界值:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.4f}")
    is_stationary = result[1] < 0.05
    print(f"结论: {'平稳 ✓' if is_stationary else '非平稳 ✗'}")
    return is_stationary

def determine_d(sunspots, max_d=2):
    """确定差分阶数d"""
    print("\n" + "-"*60)
    print("确定差分阶数d (ADF检验)")
    print("-"*60)
    
    current_series = sunspots.copy()
    for d in range(max_d + 1):
        is_stationary = check_stationarity(current_series, diff_level=d)
        if is_stationary:
            print(f"\n选择差分阶数 d = {d}")
            return d
        if d < max_d:
            current_series = np.diff(current_series)
    
    print(f"\n达到最大差分阶数，选择 d = {max_d}")
    return max_d

def arima_cross_validation(sunspots, order, n_splits=5, test_size=20):
    """时间序列交叉验证，返回平均RMSE"""
    n = len(sunspots)
    rmse_list = []
    
    for i in range(n_splits):
        split_idx = n - test_size * (n_splits - i)
        if split_idx < 10:
            continue
            
        train = sunspots[:split_idx]
        test = sunspots[split_idx:split_idx + test_size]
        
        if len(test) == 0:
            continue
            
        try:
            model = ARIMA(train, order=order)
            results = model.fit()
            forecast = results.get_forecast(steps=len(test)).predicted_mean
            rmse = np.sqrt(np.mean((forecast - test) ** 2))
            rmse_list.append(rmse)
        except:
            continue
    
    if len(rmse_list) == 0:
        return np.inf
    
    return np.mean(rmse_list)

def select_arima_order(sunspots, d, max_p=4, max_q=4):
    """使用BIC准则和交叉验证选择ARIMA阶数，防止过拟合"""
    print("\n" + "-"*60)
    print("ARIMA定阶: BIC准则 + 交叉验证 (防止过拟合)")
    print("-"*60)
    
    bic_results = []
    cv_results = []
    
    print("\n1. BIC准则筛选候选模型:")
    print(f"{'(p,d,q)':<12} {'BIC':<12} {'收敛':<8}")
    print("-" * 35)
    
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            try:
                model = ARIMA(sunspots, order=(p, d, q))
                results = model.fit()
                if results.mle_retvals['converged']:
                    bic_results.append({
                        'order': (p, d, q),
                        'bic': results.bic,
                        'converged': True
                    })
                    print(f"({p},{d},{q}):    {results.bic:<10.2f} ✓")
                else:
                    bic_results.append({
                        'order': (p, d, q),
                        'bic': np.inf,
                        'converged': False
                    })
                    print(f"({p},{d},{q}):    {'-':<10} ✗ (未收敛)")
            except:
                bic_results.append({
                    'order': (p, d, q),
                    'bic': np.inf,
                    'converged': False
                })
                print(f"({p},{d},{q}):    {'-':<10} ✗ (失败)")
    
    bic_results.sort(key=lambda x: x['bic'])
    top_candidates = [r for r in bic_results if r['converged']][:5]
    
    print(f"\n2. BIC排名前5的模型:")
    for i, r in enumerate(top_candidates):
        print(f"  {i+1}. {r['order']}, BIC = {r['bic']:.2f}")
    
    print("\n3. 交叉验证验证预测能力 (滚动窗口):")
    print(f"{'(p,d,q)':<12} {'CV-RMSE':<12}")
    print("-" * 25)
    
    for r in top_candidates:
        cv_rmse = arima_cross_validation(sunspots, r['order'])
        cv_results.append({
            'order': r['order'],
            'bic': r['bic'],
            'cv_rmse': cv_rmse
        })
        print(f"{r['order']}:    {cv_rmse:.2f}")
    
    cv_results.sort(key=lambda x: x['cv_rmse'])
    best_order = cv_results[0]['order']
    
    print(f"\n4. 最优模型选择:")
    print(f"   BIC最优: {bic_results[0]['order']}, BIC = {bic_results[0]['bic']:.2f}")
    print(f"   CV最优:  {cv_results[0]['order']}, CV-RMSE = {cv_results[0]['cv_rmse']:.2f}")
    print(f"   最终选择: {best_order} (基于交叉验证)")
    
    return best_order, bic_results[0]['bic']

def residual_diagnostics(results, order):
    """残差诊断，验证模型拟合质量"""
    print("\n" + "-"*60)
    print("模型残差诊断")
    print("-" * 60)
    
    residuals = results.resid
    
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    p_value = lb_test['lb_pvalue'].values[0]
    
    print(f"Ljung-Box检验 (滞后10期):")
    print(f"  统计量: {lb_test['lb_stat'].values[0]:.4f}")
    print(f"  p值: {p_value:.4f}")
    
    if p_value > 0.05:
        print("  结论: 残差无显著自相关 ✓ (模型充分)")
    else:
        print("  结论: 残差存在自相关 ✗ (模型可能不充分)")
    
    print(f"\n残差统计:")
    print(f"  均值: {np.mean(residuals):.4f}")
    print(f"  标准差: {np.std(residuals):.4f}")

def fit_arima_model(data, forecast_years=25):
    """拟合ARIMA模型并预测（改进版：BIC+CV防止过拟合）"""
    print("\n" + "="*60)
    print("ARIMA模型预测 (改进版：BIC+交叉验证防止过拟合)")
    print("="*60)
    
    sunspots = data['SunspotNumber'].values
    years = data['Year'].values
    last_year = years[-1]
    
    d = determine_d(sunspots, max_d=2)
    
    best_order, best_bic = select_arima_order(sunspots, d, max_p=4, max_q=4)
    
    print(f"\n" + "-"*60)
    print("拟合最终模型")
    print("-" * 60)
    
    model = ARIMA(sunspots, order=best_order)
    results = model.fit()
    
    print(f"模型: ARIMA{best_order}")
    print(f"BIC: {results.bic:.2f}")
    print(f"AIC: {results.aic:.2f}")
    print(f"对数似然: {results.llf:.2f}")
    
    residual_diagnostics(results, best_order)
    
    forecast = results.get_forecast(steps=forecast_years)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    
    future_years = np.arange(last_year + 1, last_year + forecast_years + 1)
    
    peak_idx = np.argmax(forecast_mean)
    peak_year = future_years[peak_idx]
    peak_value = forecast_mean[peak_idx]
    
    print(f"\n" + "-"*60)
    print("预测结果")
    print("-" * 60)
    print(f"下一太阳活动周峰值年份: {peak_year}")
    print(f"预测峰值太阳黑子数: {peak_value:.1f}")
    print(f"95%置信区间: [{forecast_ci[peak_idx, 0]:.1f}, {forecast_ci[peak_idx, 1]:.1f}]")
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(years, sunspots, 'b-', label='历史数据', linewidth=1.5, alpha=0.7)
    ax.plot(future_years, forecast_mean, 'r-', label='预测值', linewidth=2)
    ax.fill_between(future_years, 
                    forecast_ci[:, 0], 
                    forecast_ci[:, 1], 
                    color='red', alpha=0.2, label='95%置信区间')
    
    ax.scatter(peak_year, peak_value, color='gold', s=200, zorder=5, edgecolor='black', linewidth=2)
    ax.annotate(f'预测峰值\n{peak_year}年\n{peak_value:.0f}', 
                xy=(peak_year, peak_value), 
                xytext=(peak_year + 3, peak_value + 20),
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    model_text = f'ARIMA{best_order}\nBIC={best_bic:.0f}'
    ax.text(0.02, 0.98, model_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('年份', fontsize=12)
    ax.set_ylabel('太阳黑子数', fontsize=12)
    ax.set_title('太阳黑子数时间序列预测 (ARIMA模型 - BIC+CV定阶)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1990, peak_year + 10)
    
    plt.tight_layout()
    plt.savefig('arima_forecast.png', dpi=150, bbox_inches='tight')
    print("\n预测图已保存: arima_forecast.png")
    
    return peak_year, peak_value, forecast_mean, future_years

def cycle_analysis(data):
    """分析历史太阳活动周特征"""
    print("\n" + "="*60)
    print("历史太阳活动周分析")
    print("="*60)
    
    years = data['Year'].values
    sunspots = data['SunspotNumber'].values
    
    peaks = []
    for i in range(5, len(sunspots) - 5):
        if sunspots[i] == max(sunspots[i-5:i+6]) and sunspots[i] > 50:
            peaks.append((years[i], sunspots[i]))
    
    peaks = np.array(peaks)
    if len(peaks) > 1:
        intervals = np.diff(peaks[:, 0])
        avg_interval = np.mean(intervals)
        avg_amplitude = np.mean(peaks[:, 1])
        
        print(f"检测到 {len(peaks)} 个历史峰值")
        print(f"平均周期间隔: {avg_interval:.2f} 年")
        print(f"平均峰值强度: {avg_amplitude:.1f}")
        print(f"\n最近几个太阳活动周:")
        for i in range(max(0, len(peaks)-5), len(peaks)):
            print(f"  峰值年份: {int(peaks[i, 0])}, 峰值: {peaks[i, 1]:.1f}")
    
    return peaks

def compare_models(arima_result, dynamo_result, years, sunspots):
    """
    对比ARIMA模型和α-Ω发电机模型的预测结果
    """
    print("\n" + "="*60)
    print("模型对比: ARIMA vs α-Ω发电机模型")
    print("="*60)
    
    print(f"\n{'模型':<20} {'峰值年份':<12} {'峰值强度':<12}")
    print("-" * 45)
    print(f"{'ARIMA (统计)':<20} {arima_result[0]:<12} {arima_result[1]:<12.1f}")
    print(f"{'α-Ω发电机 (物理)':<20} {dynamo_result['peak_year']:<12} {dynamo_result['peak_value']:<12.1f}")
    print(f"{'综合预测 (平均)':<20} {int(round((arima_result[0] + dynamo_result['peak_year'])/2)):<12}")
    print(f"                    {round((arima_result[1] + dynamo_result['peak_value'])/2, 1):<12.1f}")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    ax = axes[0]
    ax.plot(years, sunspots, 'ko', label='观测数据', markersize=2, alpha=0.5)
    ax.plot(arima_result[2], arima_result[3], 'r-', label='ARIMA预测', linewidth=2, alpha=0.8)
    ax.plot(dynamo_result['future_years'], dynamo_result['forecast'], 'g--', label='α-Ω发电机预测', linewidth=2, alpha=0.8)
    
    ax.scatter(arima_result[0], arima_result[1], color='red', s=150, zorder=5, label='ARIMA峰值', edgecolor='black')
    ax.scatter(dynamo_result['peak_year'], dynamo_result['peak_value'], color='green', s=150, zorder=5, label='发电机峰值', edgecolor='black')
    
    ax.set_xlabel('年份', fontsize=11)
    ax.set_ylabel('太阳黑子数', fontsize=11)
    ax.set_title('ARIMA vs α-Ω发电机模型 预测对比', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1990, max(arima_result[0], dynamo_result['peak_year']) + 10)
    
    ax = axes[1]
    model_names = ['ARIMA (统计)', 'α-Ω发电机 (物理)']
    peak_years = [arima_result[0], dynamo_result['peak_year']]
    peak_values = [arima_result[1], dynamo_result['peak_value']]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax2 = ax.twinx()
    rects1 = ax.bar(x - width/2, peak_years, width, label='峰值年份', color='steelblue', alpha=0.7)
    rects2 = ax2.bar(x + width/2, peak_values, width, label='峰值强度', color='coral', alpha=0.7)
    
    ax.set_ylabel('峰值年份', fontsize=11, color='steelblue')
    ax2.set_ylabel('峰值太阳黑子数', fontsize=11, color='coral')
    ax.set_title('模型预测结果对比', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='coral')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print("\n对比图已保存: model_comparison.png")

def main():
    print("="*60)
    print("太阳黑子数时间序列分析与预测")
    print("  统计模型(ARIMA) + 物理模型(α-Ω发电机)")
    print("="*60)
    
    data = load_sunspot_data()
    print(f"\n数据范围: {data['Year'].min()} - {data['Year'].max()} 年")
    print(f"数据点数: {len(data)}")
    print(f"平均太阳黑子数: {data['SunspotNumber'].mean():.2f}")
    print(f"最大太阳黑子数: {data['SunspotNumber'].max():.2f} ({data.loc[data['SunspotNumber'].idxmax(), 'Year']}年)")
    
    cycle_analysis(data)
    spectral_analysis(data)
    
    arima_peak_year, arima_peak_value, arima_forecast, arima_future_years = fit_arima_model(data, forecast_years=30)
    arima_result = (arima_peak_year, arima_peak_value, arima_future_years, arima_forecast)
    
    dynamo_result = run_dynamo_assimilation(data, forecast_years=30)
    
    compare_models(arima_result, dynamo_result, data['Year'].values, data['SunspotNumber'].values)
    
    print("\n" + "="*60)
    print("最终总结")
    print("="*60)
    print(f"1. 频谱分析确认太阳活动周主导周期: ~11年")
    print(f"\n2. ARIMA统计模型预测:")
    print(f"   峰值年份: {arima_peak_year}, 峰值: {arima_peak_value:.1f}")
    print(f"\n3. α-Ω发电机物理模型预测:")
    print(f"   峰值年份: {dynamo_result['peak_year']}, 峰值: {dynamo_result['peak_value']:.1f}")
    print(f"   模型参数: α={dynamo_result['params'][0]:.3f}, Ω={dynamo_result['params'][1]:.3f}")
    print(f"\n4. 综合预测 (模型平均):")
    print(f"   峰值年份: {int(round((arima_peak_year + dynamo_result['peak_year'])/2))}")
    print(f"   峰值太阳黑子数: {round((arima_peak_value + dynamo_result['peak_value'])/2, 1)}")
    print(f"\n生成的图表:")
    print(f"   ✓ spectral_analysis.png  - 频谱分析")
    print(f"   ✓ arima_forecast.png     - ARIMA预测")
    print(f"   ✓ dynamo_assimilation.png - 发电机模型同化")
    print(f"   ✓ model_comparison.png    - 模型对比")
    print("\n分析完成！")
    print("="*60)

if __name__ == "__main__":
    main()
