import numpy as np
import matplotlib.pyplot as plt

def seir_discrete_gamma(R0, sigma, gamma, N, I0, E0, days, n_E=4):
    beta = R0 * gamma
    
    S = np.zeros(days)
    E = np.zeros((days, n_E))
    I = np.zeros(days)
    R = np.zeros(days)
    
    S[0] = N - I0 - E0
    I[0] = I0
    R[0] = 0
    
    E_per_compartment = E0 // n_E
    remaining = E0 % n_E
    for i in range(n_E):
        E[0, i] = E_per_compartment
    for i in range(remaining):
        E[0, i] += 1
    
    transition_rate = n_E * sigma
    
    for t in range(days - 1):
        new_infections = beta * S[t] * I[t] / N
        
        E_transitions = transition_rate * E[t, :]
        
        S[t+1] = S[t] - new_infections
        
        E[t+1, 0] = E[t, 0] + new_infections - E_transitions[0]
        
        for i in range(1, n_E):
            E[t+1, i] = E[t, i] + E_transitions[i-1] - E_transitions[i]
        
        I[t+1] = I[t] + E_transitions[-1] - gamma * I[t]
        
        R[t+1] = R[t] + gamma * I[t]
    
    E_total = np.sum(E, axis=1)
    
    return np.arange(days), S, E_total, I, R, E

def seir_continuous_correct(y, t, N, beta, sigma, gamma, n_E):
    S = y[0]
    E = y[1:1+n_E]
    I = y[1+n_E]
    R = y[2+n_E]
    
    dSdt = -beta * S * I / N
    
    rate_per_stage = n_E * sigma
    
    dEdt = np.zeros(n_E)
    dEdt[0] = beta * S * I / N - rate_per_stage * E[0]
    for i in range(1, n_E):
        dEdt[i] = rate_per_stage * E[i-1] - rate_per_stage * E[i]
    
    dIdt = rate_per_stage * E[-1] - gamma * I
    dRdt = gamma * I
    
    return [dSdt] + list(dEdt) + [dIdt, dRdt]

def simulate_seir_continuous(R0, sigma, gamma, N, I0, E0, days, n_E=4):
    from scipy.integrate import odeint
    
    beta = R0 * gamma
    R0_val = 0
    
    E_per_compartment = E0 // n_E
    E0_array = [E_per_compartment] * n_E
    remaining = E0 % n_E
    for i in range(remaining):
        E0_array[i] += 1
    
    S0 = N - I0 - E0
    y0 = [S0] + E0_array + [I0, R0_val]
    
    t = np.linspace(0, days, days)
    solution = odeint(seir_continuous_correct, y0, t, args=(N, beta, sigma, gamma, n_E))
    
    S = solution[:, 0]
    E_total = np.sum(solution[:, 1:1+n_E], axis=1)
    I = solution[:, 1+n_E]
    R = solution[:, 2+n_E]
    
    return t, S, E_total, I, R, solution[:, 1:1+n_E]

def plot_comparison(t_disc, S_disc, E_disc, I_disc, R_disc, t_cont, S_cont, E_cont, I_cont, R_cont, n_E):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.plot(t_disc, S_disc, label='离散时间 S', linestyle='--', color='blue')
    ax.plot(t_cont, S_cont, label='连续时间 S', color='blue', alpha=0.6)
    ax.set_xlabel('天数')
    ax.set_ylabel('人数')
    ax.set_title('易感者 (S)')
    ax.legend()
    ax.grid(True)
    
    ax = axes[0, 1]
    ax.plot(t_disc, E_disc, label='离散时间 E', linestyle='--', color='orange')
    ax.plot(t_cont, E_cont, label='连续时间 E', color='orange', alpha=0.6)
    ax.set_xlabel('天数')
    ax.set_ylabel('人数')
    ax.set_title('暴露者 (E)')
    ax.legend()
    ax.grid(True)
    
    ax = axes[1, 0]
    ax.plot(t_disc, I_disc, label='离散时间 I', linestyle='--', color='red')
    ax.plot(t_cont, I_cont, label='连续时间 I', color='red', alpha=0.6)
    ax.set_xlabel('天数')
    ax.set_ylabel('人数')
    ax.set_title('感染者 (I)')
    ax.legend()
    ax.grid(True)
    
    ax = axes[1, 1]
    ax.plot(t_disc, R_disc, label='离散时间 R', linestyle='--', color='green')
    ax.plot(t_cont, R_cont, label='连续时间 R', color='green', alpha=0.6)
    ax.set_xlabel('天数')
    ax.set_ylabel('人数')
    ax.set_title('康复者 (R)')
    ax.legend()
    ax.grid(True)
    
    plt.suptitle(f'离散时间 vs 连续时间 SEIR模型对比 (E舱室数: {n_E})', fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_results(t, S, E, I, R, n_E=None, title_suffix=''):
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='易感者 (S)', color='blue')
    plt.plot(t, E, label='暴露者 (E)', color='orange')
    plt.plot(t, I, label='感染者 (I)', color='red')
    plt.plot(t, R, label='康复者 (R)', color='green')
    plt.xlabel('天数')
    plt.ylabel('人数')
    if n_E:
        plt.title(f'SEIR模型疫情动态模拟 {title_suffix}(E舱室数: {n_E})')
    else:
        plt.title(f'SEIR模型疫情动态模拟 {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.show()

def analyze_latent_distribution(sigma, n_E_values):
    print("=== 潜伏期分布分析 ===")
    print(f"sigma = {sigma:.4f} (1/平均潜伏期)")
    print(f"理论平均潜伏期: {1/sigma:.2f} 天")
    print()
    
    for n_E in n_E_values:
        mean = 1 / sigma
        var = 1 / (n_E * sigma**2)
        cv = np.sqrt(var) / mean
        
        print(f"n_E = {n_E}:")
        print(f"  均值: {mean:.2f} 天")
        print(f"  方差: {var:.2f} 天²")
        print(f"  标准差: {np.sqrt(var):.2f} 天")
        print(f"  变异系数: {cv:.3f}")
        print(f"  分布: Gamma(k={n_E}, θ={1/(n_E*sigma):.4f})")
        print()

if __name__ == "__main__":
    R0 = 3.0
    sigma = 1/5.2
    gamma = 1/7
    N = 1000000
    I0 = 10
    E0 = 50
    days = 200
    n_E = 4
    
    analyze_latent_distribution(sigma, [1, 2, 4, 8, 16])
    
    print(f"=== Gamma分布 SEIR 模型 (离散时间) ===")
    print(f"基本再生数 R0 = {R0}")
    print(f"平均潜伏期 = {1/sigma:.1f} 天")
    print(f"E舱室数量 n_E = {n_E}")
    print(f"潜伏期分布: Gamma(k={n_E}, θ={1/(n_E*sigma):.4f})")
    print(f"  - 均值: {1/sigma:.1f} 天")
    print(f"  - 方差: {1/(n_E*sigma**2):.2f} 天²")
    print(f"  - 变异系数: {1/np.sqrt(n_E):.3f}")
    print(f"总人口 N = {N}")
    print(f"初始感染人数 I0 = {I0}")
    print(f"初始暴露人数 E0 = {E0}")
    print(f"\n模拟天数: {days}天")
    
    t_disc, S_disc, E_disc, I_disc, R_disc, E_compartments_disc = seir_discrete_gamma(
        R0, sigma, gamma, N, I0, E0, days, n_E
    )
    
    t_cont, S_cont, E_cont, I_cont, R_cont, E_compartments_cont = simulate_seir_continuous(
        R0, sigma, gamma, N, I0, E0, days, n_E
    )
    
    print(f"\n=== 离散时间模型结果 ===")
    peak_day_disc = np.argmax(I_disc)
    peak_infected_disc = int(np.max(I_disc))
    print(f"  感染峰值出现在第 {peak_day_disc} 天")
    print(f"  峰值感染人数: {peak_infected_disc} 人")
    print(f"  最终感染率: {((N - S_disc[-1])/N)*100:.2f}%")
    
    print(f"\n=== 连续时间模型结果 ===")
    peak_day_cont = np.argmax(I_cont)
    peak_infected_cont = int(np.max(I_cont))
    print(f"  感染峰值出现在第 {peak_day_cont} 天")
    print(f"  峰值感染人数: {peak_infected_cont} 人")
    print(f"  最终感染率: {((N - S_cont[-1])/N)*100:.2f}%")
    
    plot_results(t_disc, S_disc, E_disc, I_disc, R_disc, n_E, title_suffix='(离散时间) ')
    plot_comparison(t_disc, S_disc, E_disc, I_disc, R_disc, 
                   t_cont, S_cont, E_cont, I_cont, R_cont, n_E)
