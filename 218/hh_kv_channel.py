import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time

class HodgkinHuxleyKv:
    def __init__(self, T=6.3):
        self.T = T
        self.phi = 3 ** ((T - 6.3) / 10)
        self.g_K = 36.0
        self.E_K = -77.0
        self.C_m = 1.0
        
    def alpha_n(self, V):
        V_shifted = V + 65
        return 0.01 * (10 - V_shifted) / (np.exp((10 - V_shifted) / 10) - 1)
    
    def beta_n(self, V):
        V_shifted = V + 65
        return 0.125 * np.exp(-V_shifted / 80)
    
    def n_inf(self, V):
        alpha = self.alpha_n(V)
        beta = self.beta_n(V)
        return alpha / (alpha + beta)
    
    def tau_n(self, V):
        alpha = self.alpha_n(V)
        beta = self.beta_n(V)
        return 1.0 / (alpha + beta) / self.phi
    
    def dvdt(self, n, V, I_inj):
        I_K = self.g_K * (n ** 4) * (V - self.E_K)
        dVdt = (I_inj - I_K) / self.C_m
        return dVdt
    
    def dndt(self, n, V):
        return self.phi * (self.alpha_n(V) * (1 - n) - self.beta_n(V) * n)
    
    def model(self, y, t, V_clamp, is_voltage_clamp=True):
        n = y[0]
        
        if is_voltage_clamp:
            V = V_clamp(t)
            dVdt = 0
        else:
            V = y[1]
            dVdt = self.dvdt(n, V, 0)
        
        dndt_val = self.dndt(n, V)
        
        if is_voltage_clamp:
            return [dndt_val]
        else:
            return [dndt_val, dVdt]
    
    def voltage_clamp_protocol(self, t, V_hold=-70, V_step=20, t_start=5, t_end=25):
        if t < t_start:
            return V_hold
        elif t < t_end:
            return V_step
        else:
            return V_hold
    
    def simulate_voltage_clamp_odeint(self, t_start=0, t_end=50, dt=0.01, 
                              V_hold=-70, V_step=20, step_start=5, step_end=25):
        t = np.arange(t_start, t_end, dt)
        
        V_clamp_func = lambda t: self.voltage_clamp_protocol(t, V_hold, V_step, step_start, step_end)
        
        n0 = self.n_inf(V_hold)
        
        y0 = [n0]
        
        y = odeint(self.model, y0, t, args=(V_clamp_func, True))
        
        n = y[:, 0]
        V = np.array([V_clamp_func(ti) for ti in t])
        I_K = self.g_K * (n ** 4) * (V - self.E_K)
        
        return t, V, n, I_K
    
    def simulate_voltage_clamp_etd(self, t_start=0, t_end=50, dt=0.01,
                                 V_hold=-70, V_step=20, step_start=5, step_end=25):
        t = np.arange(t_start, t_end, dt)
        n_steps = len(t)
        
        V_clamp_func = lambda t: self.voltage_clamp_protocol(t, V_hold, V_step, step_start, step_end)
        
        V = np.array([V_clamp_func(ti) for ti in t])
        
        n = np.zeros(n_steps)
        n[0] = self.n_inf(V_hold)
        
        for i in range(1, n_steps):
            n_inf_i = self.n_inf(V[i])
            tau_i = self.tau_n(V[i])
            decay = np.exp(-dt / tau_i)
            n[i] = n_inf_i + (n[i-1] - n_inf_i) * decay
        
        I_K = self.g_K * (n ** 4) * (V - self.E_K)
        
        return t, V, n, I_K
    
    def simulate_voltage_clamp_crank_nicolson(self, t_start=0, t_end=50, dt=0.01,
                                    V_hold=-70, V_step=20, step_start=5, step_end=25):
        t = np.arange(t_start, t_end, dt)
        n_steps = len(t)
        
        V_clamp_func = lambda t: self.voltage_clamp_protocol(t, V_hold, V_step, step_start, step_end)
        
        V = np.array([V_clamp_func(ti) for ti in t])
        
        n = np.zeros(n_steps)
        n[0] = self.n_inf(V_hold)
        
        for i in range(1, n_steps):
            V_prev = V[i-1]
            V_curr = V[i]
            
            alpha_prev = self.alpha_n(V_prev)
            beta_prev = self.beta_n(V_prev)
            alpha_curr = self.alpha_n(V_curr)
            beta_curr = self.beta_n(V_curr)
            
            a_prev = self.phi * alpha_prev
            b_prev = self.phi * (alpha_prev + beta_prev)
            a_curr = self.phi * alpha_curr
            b_curr = self.phi * (alpha_curr + beta_curr)
            
            numerator = n[i-1] * (1 - dt/2 * b_prev) + dt/2 * (a_prev + a_curr)
            denominator = 1 + dt/2 * b_curr
            
            n[i] = numerator / denominator
        
        I_K = self.g_K * (n ** 4) * (V - self.E_K)
        
        return t, V, n, I_K
    
    def simulate_voltage_clamp(self, t_start=0, t_end=50, dt=0.01,
                                V_hold=-70, V_step=20, step_start=5, step_end=25,
                                method='etd'):
        
        if method == 'odeint':
            return self.simulate_voltage_clamp_odeint(
                t_start, t_end, dt, V_hold, V_step, step_start, step_end
            )
        elif method == 'etd':
            return self.simulate_voltage_clamp_etd(
                t_start, t_end, dt, V_hold, V_step, step_start, step_end
            )
        elif method == 'crank_nicolson':
            return self.simulate_voltage_clamp_crank_nicolson(
                t_start, t_end, dt, V_hold, V_step, step_start, step_end
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'odeint', 'etd', or 'crank_nicolson'")
    
    def compare_methods(self, t_start=0, t_end=50, dt_values=[0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
                       V_hold=-70, V_step=20, step_start=5, step_end=25):
        methods = ['odeint', 'etd', 'crank_nicolson']
        results = {}
        
        for method in methods:
            results[method] = {}
            for dt in dt_values:
                start_time = time.time()
                t, V, n, I_K = self.simulate_voltage_clamp(
                    t_start, t_end, dt, V_hold, V_step, step_start, step_end, method
                )
                elapsed = time.time() - start_time
                
                peak_I = np.max(np.abs(I_K))
                ss_I = I_K[-1]
                final_n = n[-1]
                
                results[method][dt] = {
                    't': t,
                    'V': V,
                    'n': n,
                    'I_K': I_K,
                    'time': elapsed,
                    'peak_I': peak_I,
                    'ss_I': ss_I,
                    'final_n': final_n
                }
        
        return results
    
    def plot_results(self, t, V, n, I_K, title_suffix=''):
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        axes[0].plot(t, V, 'b-', linewidth=2)
        axes[0].set_ylabel('Membrane Potential (mV)', fontsize=12)
        axes[0].set_title(f'Voltage Clamp Stimulus {title_suffix}', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(t, n, 'r-', linewidth=2)
        axes[1].set_ylabel('Activation Gating n', fontsize=12)
        axes[1].set_title('n Gating Variable Dynamics', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(t, I_K, 'g-', linewidth=2)
        axes[2].set_ylabel('K+ Current (μA/cm²)', fontsize=12)
        axes[2].set_xlabel('Time (ms)', fontsize=12)
        axes[2].set_title('Potassium Channel Current', fontsize=14)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'hh_kv_simulation{title_suffix.replace(' ', '_')}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_method_comparison(self, results, dt_ref=0.01):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        methods = list(results.keys())
        colors = {'odeint': 'b-', 'etd': 'r-', 'crank_nicolson': 'g-'}
        labels = {'odeint': 'scipy.odeint (LSODA)', 'etd': 'ETD (Exact)', 'crank_nicolson': 'Crank-Nicolson'}
        
        for method in methods:
            ref_data = results[method][dt_ref]
            axes[0, 0].plot(ref_data['t'], ref_data['n'], colors[method],
                          label=labels[method], linewidth=2, alpha=0.7)
            axes[0, 1].plot(ref_data['t'], ref_data['I_K'], colors[method],
                           label=labels[method], linewidth=2, alpha=0.7)
        
        axes[0, 0].set_ylabel('n Gating Variable', fontsize=12)
        axes[0, 0].set_xlabel('Time (ms)', fontsize=12)
        axes[0, 0].set_title(f'n Dynamics (dt = {dt_ref} ms)', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(fontsize=10)
        
        axes[0, 1].set_ylabel('K+ Current (μA/cm²)', fontsize=12)
        axes[0, 1].set_xlabel('Time (ms)', fontsize=12)
        axes[0, 1].set_title(f'Current Dynamics (dt = {dt_ref} ms)', fontsize=14)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(fontsize=10)
        
        dt_values = sorted(results['etd'].keys())
        dts = []
        times = []
        peak_errors = []
        ss_errors = []
        
        ref_peak = results['etd'][dt_ref]['peak_I']
        ref_ss = results['etd'][dt_ref]['ss_I']
        
        for dt in dt_values:
            dts.append(dt)
            times.append(results['etd'][dt]['time'])
            peak_errors.append(abs(results['etd'][dt]['peak_I'] - ref_peak) / ref_peak * 100)
            ss_errors.append(abs(results['etd'][dt]['ss_I'] - ref_ss) / ref_ss * 100)
        
        axes[1, 0].semilogx(dts, times, 'ro-', markersize=8, linewidth=2)
        axes[1, 0].set_xlabel('Time Step dt (ms)', fontsize=12)
        axes[1, 0].set_ylabel('Computation Time (s)', fontsize=12)
        axes[1, 0].set_title('ETD Method Performance', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].semilogx(dts, peak_errors, 'bo-', markersize=8, linewidth=2, label='Peak Current Error')
        axes[1, 1].semilogx(dts, ss_errors, 'go-', markersize=8, linewidth=2, label='Steady-state Error')
        axes[1, 1].set_xlabel('Time Step dt (ms)', fontsize=12)
        axes[1, 1].set_ylabel('Relative Error (%)', fontsize=12)
        axes[1, 1].set_title('ETD Accuracy vs Time Step', fontsize=14)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig('method_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def simulate_multiple_steps(self, t_start=0, t_end=50, dt=0.01,
                                 V_hold=-70, V_steps=np.arange(-40, 60, 10),
                                 step_start=5, step_end=25,
                                 method='etd'):
        results = []
        
        for V_step in V_steps:
            t, V, n, I_K = self.simulate_voltage_clamp(
                t_start, t_end, dt, V_hold, V_step, step_start, step_end,
                method
            )
            results.append((t, V, n, I_K, V_step))
        
        return results
    
    def plot_iv_curve(self, results):
        V_steps = []
        I_ss = []
        
        for t, V, n, I_K, V_step in results:
            step_mask = (t >= 5) & (t <= 25)
            step_I = I_K[step_mask]
            
            V_steps.append(V_step)
            I_ss.append(step_I[-1])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(V_steps, I_ss, 'bo-', linewidth=2, markersize=8, label='Steady-state Current')
        ax.set_xlabel('Clamp Potential (mV)', fontsize=12)
        ax.set_ylabel('K+ Current (μA/cm²)', fontsize=12)
        ax.set_title('Kv Channel I-V Curve', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig('kv_iv_curve.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    print("Hodgkin-Huxley Kv Channel Simulation")
    print("=" * 50)
    
    hh_kv = HodgkinHuxleyKv(T=6.3)
    
    print("\n1. Single-step voltage clamp simulation (ETD method)...")
    t, V, n, I_K = hh_kv.simulate_voltage_clamp(
        t_start=0, t_end=50, dt=0.01,
        V_hold=-70, V_step=20,
        step_start=5, step_end=25,
        method='etd'
    )
    
    print(f"   Holding potential: -70 mV")
    print(f"   Depolarization potential: 20 mV")
    print(f"   Peak current: {np.max(np.abs(I_K)):.2f} μA/cm²")
    print(f"   Steady-state current: {I_K[-1]:.2f} μA/cm²")
    
    hh_kv.plot_results(t, V, n, I_K, title_suffix='(ETD Method)')
    print("   Results saved to hh_kv_simulation_(ETD_Method).png")
    
    print("\n2. Method comparison with different time steps...")
    dt_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    results = hh_kv.compare_methods(
        dt_values=dt_values)
    
    print("\n   Performance Comparison:")
    print("   " + "-" * 70)
    print(f"   {'Method':<20} {'dt (ms)':<10} {'Time (s)':<12} {'Peak I':<15} {'SS I':<15}")
    print("   " + "-" * 70)
    
    for method in ['odeint', 'etd', 'crank_nicolson']:
        for dt in dt_values[:3]:
            res = results[method][dt]
            print(f"   {method:<20} {dt:<10.4f} {res['time']:<12.4f} {res['peak_I']:<15.2f} {res['ss_I']:<15.2f}")
    
    hh_kv.plot_method_comparison(results, dt_ref=0.01)
    print("\n   Comparison plot saved to method_comparison.png")
    
    print("\n3. Large time step test (dt=2.0 ms)...")
    t_large, V_large, n_large, I_large = hh_kv.simulate_voltage_clamp(
        t_start=0, t_end=50, dt=2.0,
        V_hold=-70, V_step=20,
        step_start=5, step_end=25,
        method='etd'
    )
    print(f"   Peak current (dt=2.0ms): {np.max(np.abs(I_large)):.2f} μA/cm²")
    print(f"   Steady-state current: {I_large[-1]:.2f} μA/cm²")
    
    hh_kv.plot_results(t_large, V_large, n_large, I_large, title_suffix='(Large dt=2ms ETD)')
    print("   Large dt results saved to hh_kv_simulation_(Large_dt=2ms_ETD).png")
    
    print("\n4. Multi-step voltage clamp simulation and I-V curve...")
    V_steps = np.arange(-40, 60, 10)
    results_iv = hh_kv.simulate_multiple_steps(
        V_hold=-70, V_steps=V_steps,
        step_start=5, step_end=25,
        method='etd'
    )
    
    hh_kv.plot_iv_curve(results_iv)
    print("   I-V curve saved to kv_iv_curve.png")
    
    print("\nSimulation completed!")
    print("\nKey Features:")
    print("  - ETD method: Exact solution for linear gating dynamics")
    print("  - Unconditionally stable for any time step size")
    print("  - Crank-Nicolson: Second-order accurate, A-stable")
    print("  - Enables 100-500x speedup with large dt (from 0.01ms to 2-5ms)")
    return hh_kv, t, V, n, I_K


if __name__ == "__main__":
    hh_kv, t, V, n, I_K = main()
