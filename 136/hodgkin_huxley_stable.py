import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

C_m = 1.0
g_Na = 120.0
g_K = 36.0
g_L = 0.3
E_Na = 50.0
E_K = -77.0
E_L = -54.387

def alpha_m(V):
    return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

def beta_m(V):
    return 4.0 * np.exp(-(V + 65.0) / 18.0)

def alpha_h(V):
    return 0.07 * np.exp(-(V + 65.0) / 20.0)

def beta_h(V):
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

def alpha_n(V):
    return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

def beta_n(V):
    return 0.125 * np.exp(-(V + 65.0) / 80.0)

def tau_m(V):
    return 1.0 / (alpha_m(V) + beta_m(V))

def tau_h(V):
    return 1.0 / (alpha_h(V) + beta_h(V))

def tau_n(V):
    return 1.0 / (alpha_n(V) + beta_n(V))

def m_inf(V):
    return alpha_m(V) / (alpha_m(V) + beta_m(V))

def h_inf(V):
    return alpha_h(V) / (alpha_h(V) + beta_h(V))

def n_inf(V):
    return alpha_n(V) / (alpha_n(V) + beta_n(V))

def hodgkin_huxley(t, y, I_stim):
    V, m, h, n = y
    I_Na = g_Na * (m**3) * h * (V - E_Na)
    I_K = g_K * (n**4) * (V - E_K)
    I_L = g_L * (V - E_L)
    dVdt = (I_stim(t) - I_Na - I_K - I_L) / C_m
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n
    return [dVdt, dmdt, dhdt, dndt]

class ExponentialEulerHH:
    def __init__(self, I_stim, dt=0.01):
        self.I_stim = I_stim
        self.dt = dt
        
    def step(self, V, m, h, n, t):
        I_Na = g_Na * (m**3) * h * (V - E_Na)
        I_K = g_K * (n**4) * (V - E_K)
        I_L = g_L * (V - E_L)
        dVdt = (self.I_stim(t) - I_Na - I_K - I_L) / C_m
        V_new = V + dVdt * self.dt
        
        tau_m_val = tau_m(V)
        tau_h_val = tau_h(V)
        tau_n_val = tau_n(V)
        m_inf_val = m_inf(V)
        h_inf_val = h_inf(V)
        n_inf_val = n_inf(V)
        
        m_new = m_inf_val + (m - m_inf_val) * np.exp(-self.dt / tau_m_val)
        h_new = h_inf_val + (h - h_inf_val) * np.exp(-self.dt / tau_h_val)
        n_new = n_inf_val + (n - n_inf_val) * np.exp(-self.dt / tau_n_val)
        
        return V_new, m_new, h_new, n_new
    
    def simulate(self, t_span, y0):
        t0, tf = t_span
        num_steps = int((tf - t0) / self.dt) + 1
        t = np.linspace(t0, tf, num_steps)
        V = np.zeros(num_steps)
        m = np.zeros(num_steps)
        h = np.zeros(num_steps)
        n = np.zeros(num_steps)
        
        V[0], m[0], h[0], n[0] = y0
        
        for i in range(num_steps - 1):
            V[i+1], m[i+1], h[i+1], n[i+1] = self.step(V[i], m[i], h[i], n[i], t[i])
        
        return t, np.column_stack([V, m, h, n])

def stimulus_current(t, start=10, duration=50, amplitude=10):
    return amplitude if (start <= t < start + duration) else 0.0

def detect_spikes(V, t, threshold=0):
    spikes = []
    for i in range(1, len(V)):
        if V[i-1] < threshold and V[i] >= threshold:
            spikes.append(t[i])
    return spikes

def analyze_time_constants():
    V_range = np.linspace(-80, 50, 200)
    tau_m_vals = [tau_m(v) for v in V_range]
    tau_h_vals = [tau_h(v) for v in V_range]
    tau_n_vals = [tau_n(v) for v in V_range]
    
    print("=" * 60)
    print("门控变量时间常数分析")
    print("=" * 60)
    print(f"τ_m 范围: {min(tau_m_vals):.4f} ~ {max(tau_m_vals):.4f} ms")
    print(f"τ_h 范围: {min(tau_h_vals):.4f} ~ {max(tau_h_vals):.4f} ms")
    print(f"τ_n 范围: {min(tau_n_vals):.4f} ~ {max(tau_n_vals):.4f} ms")
    print(f"最大时间常数比: {max(max(tau_m_vals), max(tau_h_vals), max(tau_n_vals)) / min(min(tau_m_vals), min(tau_h_vals), min(tau_n_vals)):.1f}:1")
    print("=" * 60)
    
    return V_range, tau_m_vals, tau_h_vals, tau_n_vals

def compare_solvers():
    V0 = -65.0
    m0 = m_inf(V0)
    h0 = h_inf(V0)
    n0 = n_inf(V0)
    y0 = [V0, m0, h0, n0]
    t_span = (0, 100)
    I_amp = 15
    I_stim = lambda t: stimulus_current(t, start=10, duration=50, amplitude=I_amp)
    
    print("\n" + "=" * 60)
    print("求解器对比测试")
    print("=" * 60)
    
    import time
    
    solvers = [
        ('RK45 (非刚性)', 'RK45', False),
        ('Radau (刚性)', 'Radau', True),
        ('BDF (刚性)', 'BDF', True),
    ]
    
    results = {}
    
    for name, method, is_stiff in solvers:
        print(f"\n正在使用 {name}...")
        start = time.time()
        sol = solve_ivp(hodgkin_huxley, t_span, y0, method=method, 
                       args=(I_stim,), rtol=1e-8, atol=1e-10)
        elapsed = time.time() - start
        print(f"  完成时间: {elapsed:.4f} 秒")
        print(f"  步数: {len(sol.t)}")
        print(f"  成功: {sol.success}")
        if not sol.success:
            print(f"  消息: {sol.message}")
        results[name] = sol
    
    print(f"\n正在使用 指数积分法...")
    start = time.time()
    ee_solver = ExponentialEulerHH(I_stim, dt=0.01)
    t_ee, sol_ee = ee_solver.simulate(t_span, y0)
    elapsed = time.time() - start
    print(f"  完成时间: {elapsed:.4f} 秒")
    print(f"  步数: {len(t_ee)}")
    results['指数积分法'] = (t_ee, sol_ee)
    
    return results

def plot_results(results, V_range, tau_m_vals, tau_h_vals, tau_n_vals):
    fig = plt.figure(figsize=(15, 12))
    
    ax1 = plt.subplot(3, 2, 1)
    ax1.semilogy(V_range, tau_m_vals, label='τ_m')
    ax1.semilogy(V_range, tau_h_vals, label='τ_h')
    ax1.semilogy(V_range, tau_n_vals, label='τ_n')
    ax1.set_xlabel('膜电位 V (mV)')
    ax1.set_ylabel('时间常数 (ms)')
    ax1.set_title('门控变量时间常数 (对数坐标)')
    ax1.legend()
    ax1.grid(True)
    
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(V_range, tau_m_vals, label='τ_m')
    ax2.plot(V_range, tau_h_vals, label='τ_h')
    ax2.plot(V_range, tau_n_vals, label='τ_n')
    ax2.set_xlabel('膜电位 V (mV)')
    ax2.set_ylabel('时间常数 (ms)')
    ax2.set_title('门控变量时间常数 (线性坐标)')
    ax2.legend()
    ax2.grid(True)
    
    ax3 = plt.subplot(3, 1, 2)
    colors = ['b', 'g', 'r', 'm']
    for i, (name, data) in enumerate(results.items()):
        if name == '指数积分法':
            t, sol = data
            V = sol[:, 0]
        else:
            t = data.t
            V = data.y[0]
        ax3.plot(t, V, label=name, color=colors[i], linewidth=1, alpha=0.7)
    ax3.set_ylabel('膜电位 V (mV)')
    ax3.set_title('不同求解器的膜电位对比')
    ax3.legend()
    ax3.grid(True)
    
    ax4 = plt.subplot(3, 1, 3)
    for i, (name, data) in enumerate(results.items()):
        if name == '指数积分法':
            t, sol = data
            V = sol[:, 0]
        else:
            t = data.t
            V = data.y[0]
        spikes = detect_spikes(V, t)
        if spikes:
            ax4.scatter([name] * len(spikes), spikes, s=50, color=colors[i], alpha=0.7)
    ax4.set_ylabel('动作电位时间 (ms)')
    ax4.set_title('动作电位检测对比')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('hh_solver_comparison.png', dpi=150)
    print("\n对比图已保存为 hh_solver_comparison.png")
    
    fig2 = plt.figure(figsize=(12, 8))
    t_ee, sol_ee = results['指数积分法']
    V = sol_ee[:, 0]
    m = sol_ee[:, 1]
    h = sol_ee[:, 2]
    n = sol_ee[:, 3]
    
    plt.subplot(2, 2, 1)
    plt.plot(t_ee, V)
    plt.ylabel('V (mV)')
    plt.title('膜电位 (指数积分法)')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(t_ee, m, label='m')
    plt.plot(t_ee, h, label='h')
    plt.plot(t_ee, n, label='n')
    plt.ylabel('门控变量')
    plt.title('门控变量')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    I_Na = g_Na * (m**3) * h * (V - E_Na)
    I_K = g_K * (n**4) * (V - E_K)
    plt.plot(t_ee, I_Na, label='I_Na')
    plt.plot(t_ee, I_K, label='I_K')
    plt.ylabel('电流 (uA/cm^2)')
    plt.xlabel('时间 (ms)')
    plt.title('离子电流')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(t_ee, tau_m(V), label='τ_m')
    plt.plot(t_ee, tau_h(V), label='τ_h')
    plt.plot(t_ee, tau_n(V), label='τ_n')
    plt.ylabel('时间常数 (ms)')
    plt.xlabel('时间 (ms)')
    plt.title('动态时间常数')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('hh_exponential_euler.png', dpi=150)
    print("详细结果图已保存为 hh_exponential_euler.png")
    
    spikes = detect_spikes(V, t_ee)
    print(f"\n" + "=" * 60)
    print("动作电位分析 (指数积分法)")
    print("=" * 60)
    print(f"检测到 {len(spikes)} 个动作电位")
    if spikes:
        print(f"动作电位时间: {[f'{s:.2f}ms' for s in spikes]}")
        if len(spikes) >= 2:
            intervals = np.diff(spikes)
            print(f"发放间隔: {[f'{i:.2f}ms' for i in intervals]}")
            print(f"最小不应期: {min(intervals):.2f} ms")
    print("=" * 60)

if __name__ == "__main__":
    V_range, tau_m_vals, tau_h_vals, tau_n_vals = analyze_time_constants()
    results = compare_solvers()
    plot_results(results, V_range, tau_m_vals, tau_h_vals, tau_n_vals)
    plt.show()
