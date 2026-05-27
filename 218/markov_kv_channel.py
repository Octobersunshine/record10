import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.insert(0, '.')
from hh_kv_channel import HodgkinHuxleyKv

class KvChannelMarkov:
    def __init__(self, T=6.3, n_states=3):
        self.T = T
        self.phi = 3 ** ((T - 6.3) / 10)
        self.g_single = 0.02
        self.E_K = -77.0
        self.n_states = n_states
        
        self._setup_rate_constants()
    
    def _setup_rate_constants(self):
        self.alpha_1 = lambda V: self.phi * 2.0 * np.exp(0.03 * (V + 65))
        self.beta_1 = lambda V: self.phi * 0.5 * np.exp(-0.02 * (V + 65))
        self.alpha_2 = lambda V: self.phi * 1.5 * np.exp(0.025 * (V + 65))
        self.beta_2 = lambda V: self.phi * 1.0 * np.exp(-0.015 * (V + 65))
        
        self.CLOSED1 = 0
        self.CLOSED2 = 1
        self.OPEN = 2
        self.state_names = ['C1', 'C2', 'O']
    
    def get_transition_rates(self, V, state):
        rates = np.zeros(self.n_states)
        
        if state == self.CLOSED1:
            rates[self.CLOSED2] = self.alpha_1(V)
        elif state == self.CLOSED2:
            rates[self.CLOSED1] = self.beta_1(V)
            rates[self.OPEN] = self.alpha_2(V)
        elif state == self.OPEN:
            rates[self.CLOSED2] = self.beta_2(V)
        
        return rates
    
    def get_all_rates(self, V):
        rates = np.zeros((self.n_states, self.n_states))
        
        rates[self.CLOSED1, self.CLOSED2] = self.alpha_1(V)
        rates[self.CLOSED2, self.CLOSED1] = self.beta_1(V)
        rates[self.CLOSED2, self.OPEN] = self.alpha_2(V)
        rates[self.OPEN, self.CLOSED2] = self.beta_2(V)
        
        return rates


class SingleChannelSimulator:
    def __init__(self, markov_model):
        self.model = markov_model
    
    def simulate_gillespie(self, V_func, t_start=0, t_end=50, initial_state=0):
        t = t_start
        state = initial_state
        times = [t]
        states = [state]
        
        while t < t_end:
            V = V_func(t)
            rates = self.model.get_transition_rates(V, state)
            total_rate = np.sum(rates)
            
            if total_rate == 0:
                dt = t_end - t
                t = t_end
            else:
                dt = np.random.exponential(1.0 / total_rate)
                t = min(t + dt, t_end)
                
                if t < t_end:
                    prob = rates / total_rate
                    next_state = np.random.choice(self.model.n_states, p=prob)
                    state = next_state
            
            times.append(t)
            states.append(state)
        
        return np.array(times), np.array(states)
    
    def simulate_discrete(self, V_func, t_start=0, t_end=50, dt=0.01, initial_state=0):
        t = np.arange(t_start, t_end, dt)
        n_steps = len(t)
        states = np.zeros(n_steps, dtype=int)
        states[0] = initial_state
        
        for i in range(1, n_steps):
            V = V_func(t[i])
            rates = self.model.get_transition_rates(V, states[i-1])
            total_rate = np.sum(rates)
            
            stay_prob = np.exp(-total_rate * dt)
            
            if np.random.rand() > stay_prob and total_rate > 0:
                prob = rates / total_rate
                states[i] = np.random.choice(self.model.n_states, p=prob)
            else:
                states[i] = states[i-1]
        
        return t, states


class EnsembleSimulator:
    def __init__(self, markov_model):
        self.model = markov_model
    
    @staticmethod
    def _tau_leap_step_numba(n_channels, state_counts, rates, dt):
        n_states = len(state_counts)
        
        for i in range(n_states):
            for j in range(n_states):
                if i != j and rates[i, j] > 0:
                    expected = state_counts[i] * rates[i, j] * dt
                    if expected > 0:
                        n_transitions = np.random.poisson(expected)
                        n_transitions = min(n_transitions, state_counts[i])
                        state_counts[i] -= n_transitions
                        state_counts[j] += n_transitions
        
        return state_counts
    
    def simulate_tau_leap(self, V_func, n_channels=1000, t_start=0, t_end=50, dt=0.01,
                         initial_state=0):
        t = np.arange(t_start, t_end, dt)
        n_steps = len(t)
        
        state_counts = np.zeros(self.model.n_states, dtype=int)
        state_counts[initial_state] = n_channels
        
        state_counts_history = np.zeros((n_steps, self.model.n_states), dtype=int)
        state_counts_history[0] = state_counts.copy()
        
        for i in range(1, n_steps):
            V = V_func(t[i])
            rates = self.model.get_all_rates(V)
            
            state_counts = self._tau_leap_step_numba(
                n_channels, state_counts, rates, dt
            )
            
            state_counts_history[i] = state_counts.copy()
        
        open_fraction = state_counts_history[:, self.model.OPEN] / n_channels
        I = n_channels * self.model.g_single * open_fraction * (
            np.array([V_func(ti) for ti in t]) - self.model.E_K
        )
        
        return t, state_counts_history, open_fraction, I
    
    def simulate_ensemble_binomial(self, V_func, n_channels=1000, t_start=0, t_end=50,
                                   dt=0.01):
        t = np.arange(t_start, t_end, dt)
        n_steps = len(t)
        
        P = np.eye(self.model.n_states)
        P_history = np.zeros((n_steps, self.model.n_states, self.model.n_states))
        P_history[0] = P.copy()
        
        for i in range(1, n_steps):
            V = V_func(t[i])
            rates = self.model.get_all_rates(V)
            Q = rates - np.diag(np.sum(rates, axis=1))
            P = P @ (np.eye(self.model.n_states) + Q * dt)
            P = np.clip(P, 0, 1)
            row_sums = P.sum(axis=1, keepdims=True)
            P = P / row_sums
            P_history[i] = P.copy()
        
        initial_dist = np.zeros(self.model.n_states)
        initial_dist[0] = 1.0
        
        state_probs = np.zeros((n_steps, self.model.n_states))
        for i in range(n_steps):
            state_probs[i] = initial_dist @ P_history[i]
        
        I_mean = n_channels * self.model.g_single * state_probs[:, self.model.OPEN] * (
            np.array([V_func(ti) for ti in t]) - self.model.E_K
        )
        
        I_var = n_channels * (self.model.g_single ** 2) * (
            np.array([V_func(ti) for ti in t]) - self.model.E_K
        ) ** 2 * state_probs[:, self.model.OPEN] * (1 - state_probs[:, self.model.OPEN])
        
        return t, state_probs, I_mean, I_var


class MarkovEnsembleComparison:
    def __init__(self):
        self.markov = KvChannelMarkov(T=6.3)
        self.hh = HodgkinHuxleyKv(T=6.3)
    
    def voltage_clamp(self, t, V_hold=-70, V_step=20, t_start=5, t_end=25):
        if t < t_start:
            return V_hold
        elif t < t_end:
            return V_step
        else:
            return V_hold
    
    def compare_models(self, n_channels=10000, t_end=50, dt=0.01):
        V_func = lambda t: self.voltage_clamp(t)
        
        print("1. Simulating deterministic HH model...")
        t_hh, V_hh, n_hh, I_hh = self.hh.simulate_voltage_clamp(
            t_start=0, t_end=t_end, dt=dt, method='etd'
        )
        scale_factor = n_channels * self.markov.g_single / self.hh.g_K
        I_hh_scaled = I_hh * scale_factor
        
        print(f"2. Simulating {n_channels} channels with τ-leap...")
        ensemble = EnsembleSimulator(self.markov)
        start_time = time.time()
        t_tau, counts_tau, open_frac_tau, I_tau = ensemble.simulate_tau_leap(
            V_func, n_channels=n_channels, t_start=0, t_end=t_end, dt=dt
        )
        tau_time = time.time() - start_time
        print(f"   τ-leap completed in {tau_time:.3f} s")
        
        print("3. Computing mean-field solution...")
        t_mf, probs_mf, I_mf_mean, I_mf_var = ensemble.simulate_ensemble_binomial(
            V_func, n_channels=n_channels, t_start=0, t_end=t_end, dt=dt
        )
        
        return {
            't': t_hh,
            'V': np.array([V_func(ti) for ti in t_hh]),
            'I_hh': I_hh_scaled,
            'I_tau': I_tau,
            'I_mf_mean': I_mf_mean,
            'I_mf_std': np.sqrt(I_mf_var),
            'open_frac_tau': open_frac_tau,
            'probs_mf': probs_mf
        }
    
    def plot_comparison(self, results):
        t = results['t']
        V = results['V']
        
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(4, 1, height_ratios=[1, 2, 2, 2])
        
        ax0 = fig.add_subplot(gs[0])
        ax0.plot(t, V, 'k-', linewidth=2)
        ax0.set_ylabel('Voltage (mV)')
        ax0.set_title('Voltage Clamp Protocol')
        ax0.grid(True, alpha=0.3)
        
        ax1 = fig.add_subplot(gs[1])
        ax1.plot(t, results['I_hh'], 'b-', linewidth=2, label='Deterministic HH')
        ax1.plot(t, results['I_tau'], 'g-', linewidth=1, alpha=0.7, label='Stochastic τ-leap')
        ax1.plot(t, results['I_mf_mean'], 'r--', linewidth=2, label='Mean-field')
        ax1.fill_between(t,
                         results['I_mf_mean'] - 2*results['I_mf_std'],
                         results['I_mf_mean'] + 2*results['I_mf_std'],
                         color='r', alpha=0.2, label='95% Confidence')
        ax1.set_ylabel('Current (nA)')
        ax1.set_title('Macroscopic Current Comparison')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[2])
        ax2.plot(t, results['I_tau'] - results['I_mf_mean'], 'g-', linewidth=1)
        ax2.axhline(0, color='k', linestyle='--', linewidth=0.5)
        ax2.set_ylabel('Current Fluctuation (nA)')
        ax2.set_title('Stochastic Fluctuations Around Mean')
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[3])
        state_colors = ['blue', 'orange', 'green']
        state_labels = ['C1', 'C2', 'O']
        for i in range(3):
            ax3.plot(t, results['probs_mf'][:, i], color=state_colors[i],
                     linewidth=2, label=state_labels[i])
        ax3.set_ylabel('State Probability')
        ax3.set_xlabel('Time (ms)')
        ax3.set_title('State Occupation Probabilities')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('markov_comparison.png', dpi=300, bbox_inches='tight')
        print("Comparison plot saved to markov_comparison.png")
        plt.show()
    
    def show_single_channel_traces(self, n_traces=5, t_end=50, dt=0.01):
        V_func = lambda t: self.voltage_clamp(t)
        single = SingleChannelSimulator(self.markov)
        
        fig, axes = plt.subplots(n_traces, 1, figsize=(12, 2*n_traces), sharex=True)
        
        for i, ax in enumerate(axes):
            t, states = single.simulate_discrete(
                V_func, t_start=0, t_end=t_end, dt=dt, initial_state=0
            )
            ax.step(t, states, where='post', linewidth=1.5)
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(['C1', 'C2', 'O'])
            ax.set_ylabel(f'Trace {i+1}')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.2, 2.2)
        
        axes[-1].set_xlabel('Time (ms)')
        axes[0].set_title('Single Channel Stochastic Traces')
        plt.tight_layout()
        plt.savefig('single_channel_traces.png', dpi=300, bbox_inches='tight')
        print("Single channel traces saved to single_channel_traces.png")
        plt.show()


def main():
    print("Kv Channel Markov Chain Stochastic Simulation")
    print("=" * 60)
    
    comparison = MarkovEnsembleComparison()
    
    print("\n1. Single channel stochastic traces...")
    comparison.show_single_channel_traces(n_traces=5)
    
    print("\n2. Model comparison (Deterministic HH vs Stochastic)...")
    results = comparison.compare_models(n_channels=10000)
    comparison.plot_comparison(results)
    
    print("\n3. Channel number dependence...")
    n_channels_list = [100, 1000, 10000]
    fig, axes = plt.subplots(len(n_channels_list), 1, figsize=(12, 3*len(n_channels_list)), sharex=True)
    
    ensemble = EnsembleSimulator(comparison.markov)
    V_func = lambda t: comparison.voltage_clamp(t)
    
    for ax, n_ch in zip(axes, n_channels_list):
        t, counts, open_frac, I = ensemble.simulate_tau_leap(
            V_func, n_channels=n_ch, t_start=0, t_end=50, dt=0.01
        )
        ax.plot(t, I, 'g-', linewidth=1, label=f'{n_ch} channels')
        ax.set_ylabel('Current (nA)')
        ax.set_title(f'{n_ch} Channels')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (ms)')
    plt.tight_layout()
    plt.savefig('channel_number_dependence.png', dpi=300, bbox_inches='tight')
    print("Channel number dependence plot saved to channel_number_dependence.png")
    
    print("\nSimulation completed!")
    print("\nKey Features:")
    print("  - 3-state Markov model: C1 <-> C2 <-> O")
    print("  - Gillespie exact algorithm for single channels")
    print("  - τ-leap approximation for ensemble simulation")
    print("  - Mean-field analysis with confidence intervals")
    print("  - Law of large numbers: more channels = smoother current")
    
    return comparison


if __name__ == "__main__":
    comparison = main()
