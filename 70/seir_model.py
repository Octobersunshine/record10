import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class Intervention:
    """干预措施基类"""
    def apply(self, t, state, params):
        raise NotImplementedError

class LockdownIntervention(Intervention):
    """
    封控干预措施
    
    通过降低传播率 beta 来模拟封控效果
    """
    def __init__(self, start_day, end_day, reduction_factor=0.6, smooth=True):
        """
        参数:
            start_day: 封控开始天数
            end_day: 封控结束天数
            reduction_factor: 传播率降低系数 (0-1), 0.6表示降低60%
            smooth: 是否使用平滑过渡
        """
        self.start_day = start_day
        self.end_day = end_day
        self.reduction_factor = reduction_factor
        self.smooth = smooth
    
    def get_effectiveness(self, t):
        """获取t时刻的干预效果 (0-1), 返回的是保留的传播率比例"""
        if t < self.start_day:
            return 1.0
        elif t > self.end_day:
            return 1.0
        
        if self.smooth:
            phase_in = min((t - self.start_day) / 3, 1.0)
            phase_out = min((self.end_day - t) / 3, 1.0)
            smooth_factor = phase_in * phase_out
            return 1.0 - smooth_factor * self.reduction_factor
        else:
            return 1.0 - self.reduction_factor
    
    def apply(self, t, state, params):
        beta_multiplier = self.get_effectiveness(t)
        params['beta'] = params['beta'] * beta_multiplier
        return params

class VaccinationIntervention(Intervention):
    """
    疫苗接种干预措施
    
    将易感人群(S)转移到接种人群(V), 再转移到免疫保护人群(P)
    """
    def __init__(self, start_day, end_day, daily_rate, efficacy=0.95, incubation_days=14):
        """
        参数:
            start_day: 开始接种天数
            end_day: 结束接种天数
            daily_rate: 每日接种率 (占总人口的比例)
            efficacy: 疫苗保护效力 (0-1)
            incubation_days: 疫苗产生保护的天数
        """
        self.start_day = start_day
        self.end_day = end_day
        self.daily_rate = daily_rate
        self.efficacy = efficacy
        self.vaccine_rate = 1.0 / incubation_days
    
    def get_daily_vaccinations(self, t, N):
        """获取t时刻的每日接种人数"""
        if t < self.start_day or t > self.end_day:
            return 0
        return self.daily_rate * N
    
    def apply(self, t, state, params):
        params['vaccination_rate'] = self.get_daily_vaccinations(t, params['N'])
        params['vaccine_efficacy'] = self.efficacy
        params['vaccine_protection_rate'] = self.vaccine_rate
        return params

def seir_intervention_model(y, t, params):
    """
    带有时变干预措施的SEIR-VP模型
    
    舱室: S(易感), E1...En(暴露), I(感染), R(康复), V(接种未保护), P(接种保护)
    """
    n_E = params['n_E']
    N = params['N']
    sigma = params['sigma']
    gamma = params['gamma']
    
    S = y[0]
    E = y[1:1+n_E]
    I = y[1+n_E]
    R = y[2+n_E]
    V = y[3+n_E]
    P = y[4+n_E]
    
    current_params = {'beta': params['beta0'], 'N': N}
    for intervention in params.get('interventions', []):
        current_params = intervention.apply(t, y, current_params)
    
    beta = current_params['beta']
    
    vaccination_rate = current_params.get('vaccination_rate', 0)
    vaccine_efficacy = current_params.get('vaccine_efficacy', 0.95)
    vaccine_protection_rate = current_params.get('vaccine_protection_rate', 1/14)
    
    force_of_infection = beta * I / N
    
    dSdt = -force_of_infection * S - vaccination_rate * S
    
    rate_per_stage = n_E * sigma
    
    dEdt = np.zeros(n_E)
    dEdt[0] = force_of_infection * S + force_of_infection * V * (1 - vaccine_efficacy) - rate_per_stage * E[0]
    for i in range(1, n_E):
        dEdt[i] = rate_per_stage * E[i-1] - rate_per_stage * E[i]
    
    dIdt = rate_per_stage * E[-1] - gamma * I
    dRdt = gamma * I
    
    dVdt = vaccination_rate * S - vaccine_protection_rate * V - force_of_infection * V * (1 - vaccine_efficacy)
    dPdt = vaccine_protection_rate * V
    
    return [dSdt] + list(dEdt) + [dIdt, dRdt, dVdt, dPdt]

def calculate_Rt(t, S, V, I, R0, interventions, N, vaccine_efficacy=0.95):
    """
    计算有效再生数 Rt(t)
    
    Rt(t) = R0 * intervention_effect(t) * (S(t) + V(t)*(1-efficacy)) / N
    """
    Rt = np.zeros_like(t)
    
    for i, time in enumerate(t):
        intervention_factor = 1.0
        for intervention in interventions:
            if hasattr(intervention, 'get_effectiveness'):
                intervention_factor *= intervention.get_effectiveness(time)
        
        effective_susceptible = S[i] + V[i] * (1 - vaccine_efficacy)
        Rt[i] = R0 * intervention_factor * effective_susceptible / N
    
    return Rt

def calculate_instantaneous_Rt(t, I, generation_time=5.2, window=7):
    """
    通过感染病例增长率估算瞬时 Rt
    
    使用滑动窗口估计增长率 r, 然后 Rt = exp(r * Tg)
    """
    Rt_inst = np.zeros_like(t)
    half_window = window // 2
    
    for i in range(len(t)):
        start = max(0, i - half_window)
        end = min(len(t), i + half_window + 1)
        
        if end - start < 3:
            Rt_inst[i] = np.nan
            continue
        
        log_I = np.log(I[start:end] + 1)
        t_window = t[start:end]
        
        r = np.polyfit(t_window, log_I, 1)[0]
        Rt_inst[i] = np.exp(r * generation_time)
    
    return Rt_inst

def simulate_seir_with_interventions(R0, sigma, gamma, N, I0, E0, days, n_E=4, interventions=None):
    """
    带干预措施的SEIR模型模拟
    """
    if interventions is None:
        interventions = []
    
    beta0 = R0 * gamma
    
    params = {
        'beta0': beta0,
        'sigma': sigma,
        'gamma': gamma,
        'n_E': n_E,
        'N': N,
        'interventions': interventions
    }
    
    E_per_compartment = E0 // n_E
    E0_array = [E_per_compartment] * n_E
    remaining = E0 % n_E
    for i in range(remaining):
        E0_array[i] += 1
    
    S0 = N - I0 - E0
    V0 = 0
    P0 = 0
    R0_val = 0
    
    y0 = [S0] + E0_array + [I0, R0_val, V0, P0]
    
    t = np.linspace(0, days, days)
    solution = odeint(seir_intervention_model, y0, t, args=(params,))
    
    S = solution[:, 0]
    E_total = np.sum(solution[:, 1:1+n_E], axis=1)
    I = solution[:, 1+n_E]
    R = solution[:, 2+n_E]
    V = solution[:, 3+n_E]
    P = solution[:, 4+n_E]
    
    vaccine_efficacy = 0.95
    for intervention in interventions:
        if hasattr(intervention, 'efficacy'):
            vaccine_efficacy = intervention.efficacy
            break
    
    Rt = calculate_Rt(t, S, V, I, R0, interventions, N, vaccine_efficacy)
    Rt_inst = calculate_instantaneous_Rt(t, I)
    
    return t, S, E_total, I, R, V, P, Rt, Rt_inst, solution[:, 1:1+n_E]

def analyze_latent_distribution(sigma, n_E_values):
    """分析不同舱室数下的潜伏期分布特性"""
    print("=" * 60)
    print("潜伏期分布特性分析 (Gamma分布)")
    print("=" * 60)
    print(f"sigma = {sigma:.4f} (1/平均潜伏期)")
    print(f"目标平均潜伏期: {1/sigma:.2f} 天")
    print()
    
    for n_E in n_E_values:
        mean = 1 / sigma
        var = 1 / (n_E * sigma**2)
        std = np.sqrt(var)
        cv = std / mean
        
        print(f"n_E = {n_E}:")
        print(f"  ├─ 分布类型: Gamma(k={n_E}, θ={1/(n_E*sigma):.4f})")
        print(f"  ├─ 均值: {mean:.2f} 天")
        print(f"  ├─ 方差: {var:.2f} 天²")
        print(f"  ├─ 标准差: {std:.2f} 天")
        print(f"  └─ 变异系数: {cv:.3f}")
        print()

def plot_results_with_interventions(t, S, E, I, R, V, P, Rt, Rt_inst, interventions, n_E=None):
    """绘制带干预和Rt的完整结果"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, S, label='易感者 (S)', color='blue', linewidth=2)
    ax1.plot(t, V, label='接种未保护 (V)', color='purple', linewidth=2)
    ax1.plot(t, P, label='接种保护 (P)', color='green', linewidth=2, linestyle='--')
    ax1.set_xlabel('天数', fontsize=11)
    ax1.set_ylabel('人数', fontsize=11)
    ax1.set_title('易感与疫苗保护人群', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, E, label='暴露者 (E)', color='orange', linewidth=2)
    ax2.plot(t, I, label='感染者 (I)', color='red', linewidth=2)
    ax2.set_xlabel('天数', fontsize=11)
    ax2.set_ylabel('人数', fontsize=11)
    ax2.set_title('暴露与感染人群', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, R, label='康复者 (R)', color='green', linewidth=2)
    ax3.set_xlabel('天数', fontsize=11)
    ax3.set_ylabel('人数', fontsize=11)
    ax3.set_title('康复人群', fontsize=13)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t, Rt, label='理论 Rt', color='darkblue', linewidth=2)
    ax4.plot(t, Rt_inst, label='瞬时 Rt (病例增长率)', color='crimson', 
             linewidth=2, alpha=0.7, linestyle='--')
    ax4.axhline(y=1, color='black', linestyle=':', linewidth=2, label='Rt=1 阈值')
    ax4.set_xlabel('天数', fontsize=11)
    ax4.set_ylabel('有效再生数 Rt', fontsize=11)
    ax4.set_title('有效再生数 Rt 变化', fontsize=13)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[2, :])
    intervention_days = np.zeros_like(t)
    for intervention in interventions:
        if hasattr(intervention, 'get_effectiveness'):
            for i, time in enumerate(t):
                intervention_days[i] += (1 - intervention.get_effectiveness(time)) * 100
    
    ax5.fill_between(t, 0, intervention_days, color='red', alpha=0.3, label='干预强度')
    ax5.plot(t, intervention_days, color='red', linewidth=2)
    ax5.set_xlabel('天数', fontsize=11)
    ax5.set_ylabel('传播率降低 (%)', fontsize=11)
    ax5.set_title('干预措施强度变化', fontsize=13)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    for intervention in interventions:
        if hasattr(intervention, 'start_day'):
            ax5.axvline(x=intervention.start_day, color='black', linestyle='--', alpha=0.5)
            ax5.axvline(x=intervention.end_day, color='black', linestyle='--', alpha=0.5)
    
    if n_E:
        fig.suptitle(f'SEIR-VP疫情动态与干预效果 (E舱室数: {n_E})', fontsize=16, y=0.98)
    else:
        fig.suptitle('SEIR-VP疫情动态与干预效果', fontsize=16, y=0.98)
    
    plt.show()

def plot_intervention_comparison(R0, sigma, gamma, N, I0, E0, days, n_E):
    """对比不同干预措施的效果"""
    scenarios = {
        '无干预': [],
        '封控 (第30-60天)': [
            LockdownIntervention(start_day=30, end_day=60, reduction_factor=0.6)
        ],
        '疫苗接种 (第50-150天)': [
            VaccinationIntervention(start_day=50, end_day=150, daily_rate=0.005)
        ],
        '封控+疫苗': [
            LockdownIntervention(start_day=30, end_day=60, reduction_factor=0.6),
            VaccinationIntervention(start_day=50, end_day=150, daily_rate=0.005)
        ]
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = ['gray', 'blue', 'green', 'red']
    
    for idx, (scenario_name, interventions) in enumerate(scenarios.items()):
        t, S, E, I, R, V, P, Rt, Rt_inst, _ = simulate_seir_with_interventions(
            R0, sigma, gamma, N, I0, E0, days, n_E, interventions
        )
        
        ax = axes[idx]
        ax.plot(t, I, label=f'{scenario_name} - 感染', color=colors[idx], linewidth=2)
        ax.fill_between(t, 0, I, color=colors[idx], alpha=0.2)
        ax.set_xlabel('天数', fontsize=11)
        ax.set_ylabel('感染人数', fontsize=11)
        ax.set_title(f'{scenario_name}\n峰值: {int(np.max(I))}人, 总感染: {int(N - S[-1] - V[-1]*(1-0.95))}人', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('不同干预措施效果对比', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()

def print_intervention_summary(t, S, E, I, R, V, P, Rt, interventions, N):
    """打印干预措施效果摘要"""
    print("\n" + "=" * 60)
    print("干预措施效果评估")
    print("=" * 60)
    
    print(f"\n疫情峰值:")
    print(f"  感染峰值: 第{np.argmax(I)}天, {int(np.max(I))}人")
    print(f"  暴露峰值: 第{np.argmax(E)}天, {int(np.max(E))}人")
    
    rt_under_1_day = np.argmax(Rt < 1)
    if Rt[-1] < 1:
        print(f"\n  Rt首次低于1: 第{rt_under_1_day}天")
    else:
        print(f"\n  模拟期内Rt始终高于1")
    
    print(f"\n最终状态 (第{int(t[-1])}天):")
    print(f"  易感人群: {int(S[-1])}人 ({S[-1]/N*100:.1f}%)")
    print(f"  接种未保护: {int(V[-1])}人 ({V[-1]/N*100:.1f}%)")
    print(f"  接种保护: {int(P[-1])}人 ({P[-1]/N*100:.1f}%)")
    print(f"  感染人群: {int(I[-1])}人 ({I[-1]/N*100:.2f}%)")
    print(f"  康复人群: {int(R[-1])}人 ({R[-1]/N*100:.1f}%)")
    print(f"  曾感染人数: {int(N - S[-1])}人 ({(N - S[-1])/N*100:.1f}%)")
    
    print(f"\n疫苗效果:")
    total_vaccinated = int(V[-1] + P[-1])
    print(f"  总接种人数: {total_vaccinated}人 ({total_vaccinated/N*100:.1f}%)")
    print(f"  完全保护人数: {int(P[-1])}人 ({P[-1]/N*100:.1f}%)")
    
    print(f"\n有效再生数 Rt:")
    print(f"  初始 R0: {Rt[0]:.2f}")
    print(f"  最低 Rt: {np.min(Rt):.2f} (第{np.argmin(Rt)}天)")
    print(f"  最终 Rt: {Rt[-1]:.2f}")
    
    print("\n干预措施详情:")
    for i, intervention in enumerate(interventions):
        if hasattr(intervention, 'reduction_factor'):
            print(f"  {i+1}. 封控措施: 第{intervention.start_day}-{intervention.end_day}天, "
                  f"传播率降低{intervention.reduction_factor*100:.0f}%")
        elif hasattr(intervention, 'daily_rate'):
            print(f"  {i+1}. 疫苗接种: 第{intervention.start_day}-{intervention.end_day}天, "
                  f"日接种率{intervention.daily_rate*100:.2f}%, 效力{intervention.efficacy*100:.0f}%")

if __name__ == "__main__":
    R0 = 3.0
    sigma = 1/5.2
    gamma = 1/7
    N = 1000000
    I0 = 10
    E0 = 50
    days = 300
    n_E = 4
    
    analyze_latent_distribution(sigma, [1, 2, 4, 8])
    
    print("=" * 60)
    print(f"SEIR-VP模型模拟 - 带干预措施")
    print("=" * 60)
    print(f"基本再生数 R0 = {R0}")
    print(f"平均潜伏期 = {1/sigma:.1f} 天")
    print(f"E舱室数量 n_E = {n_E}")
    print(f"总人口 N = {N}")
    print(f"初始感染人数 I0 = {I0}")
    print(f"初始暴露人数 E0 = {E0}")
    print(f"模拟天数: {days}天")
    
    interventions = [
        LockdownIntervention(start_day=40, end_day=90, reduction_factor=0.65, smooth=True),
        VaccinationIntervention(start_day=60, end_day=180, daily_rate=0.006, efficacy=0.95)
    ]
    
    print(f"\n干预措施:")
    for i, intervention in enumerate(interventions):
        if hasattr(intervention, 'reduction_factor'):
            print(f"  {i+1}. 封控: 第{intervention.start_day}-{intervention.end_day}天, "
                  f"传播率降低{intervention.reduction_factor*100:.0f}%")
        elif hasattr(intervention, 'daily_rate'):
            print(f"  {i+1}. 疫苗接种: 第{intervention.start_day}-{intervention.end_day}天, "
                  f"日接种率{intervention.daily_rate*100:.2f}%")
    
    t, S, E, I, R, V, P, Rt, Rt_inst, E_compartments = simulate_seir_with_interventions(
        R0, sigma, gamma, N, I0, E0, days, n_E, interventions
    )
    
    print_intervention_summary(t, S, E, I, R, V, P, Rt, interventions, N)
    
    plot_results_with_interventions(t, S, E, I, R, V, P, Rt, Rt_inst, interventions, n_E)
    
    print("\n" + "=" * 60)
    print("生成不同干预措施对比图...")
    print("=" * 60)
    plot_intervention_comparison(R0, sigma, gamma, N, I0, E0, days, n_E)
