import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time

C_m = 1.0
g_Na = 120.0
g_K = 36.0
g_L = 0.3
E_Na = 50.0
E_K = -77.0
E_L = -54.387
V_rest = -65.0

E_AMPA = 0.0
E_NMDA = 0.0
E_GABA = -70.0

tau_r_AMPA = 0.5
tau_d_AMPA = 5.0
tau_r_NMDA = 2.0
tau_d_NMDA = 150.0
tau_r_GABA = 0.5
tau_d_GABA = 10.0

Mg_conc = 1.0

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

def mg_block(V):
    return 1.0 / (1.0 + np.exp(-0.062 * V) * (Mg_conc / 3.57))

class Synapse:
    def __init__(self, syn_type, g_max=0.1, delay=0.0):
        self.syn_type = syn_type
        self.g_max = g_max
        self.delay = delay
        self.r = 0.0
        self.s = 0.0
        self.spike_queue = []
        
    def step(self, dt, V_post):
        if self.syn_type == 'AMPA':
            tau_r, tau_d = tau_r_AMPA, tau_d_AMPA
            E_syn = E_AMPA
            mg_factor = 1.0
        elif self.syn_type == 'NMDA':
            tau_r, tau_d = tau_r_NMDA, tau_d_NMDA
            E_syn = E_NMDA
            mg_factor = mg_block(V_post)
        elif self.syn_type == 'GABA':
            tau_r, tau_d = tau_r_GABA, tau_d_GABA
            E_syn = E_GABA
            mg_factor = 1.0
        else:
            return 0.0, 0.0
        
        drdt = -self.r / tau_r
        dsdt = (self.r - self.s) / tau_d
        
        self.r += drdt * dt
        self.s += dsdt * dt
        
        self.r = np.clip(self.r, 0, 1)
        self.s = np.clip(self.s, 0, 1)
        
        g_syn = self.g_max * self.s * mg_factor
        I_syn = g_syn * (V_post - E_syn)
        
        return I_syn, g_syn
    
    def receive_spike(self, t):
        self.spike_queue.append(t + self.delay)
    
    def process_delayed_spikes(self, t):
        to_process = [spike_t for spike_t in self.spike_queue if spike_t <= t]
        self.spike_queue = [spike_t for spike_t in self.spike_queue if spike_t > t]
        
        for _ in to_process:
            if self.syn_type == 'AMPA':
                self.r += 1.0
            elif self.syn_type == 'NMDA':
                self.r += 1.0
            elif self.syn_type == 'GABA':
                self.r += 1.0

class HHNeuron:
    def __init__(self, neuron_id, neuron_type='excitatory'):
        self.id = neuron_id
        self.neuron_type = neuron_type
        
        self.V = V_rest
        self.m = m_inf(V_rest)
        self.h = h_inf(V_rest)
        self.n = n_inf(V_rest)
        
        self.synapses = []
        self.spike_times = []
        self.last_spike_time = -np.inf
        self.refractory_period = 2.0
        self.threshold = 0.0
        
        self.I_ext = 0.0
        
    def add_synapse(self, synapse):
        self.synapses.append(synapse)
    
    def step(self, dt, t):
        if t - self.last_spike_time < self.refractory_period:
            return self.V, False
        
        I_syn_total = 0.0
        for syn in self.synapses:
            syn.process_delayed_spikes(t)
            I_syn, _ = syn.step(dt, self.V)
            I_syn_total += I_syn
        
        I_Na = g_Na * (self.m**3) * self.h * (self.V - E_Na)
        I_K = g_K * (self.n**4) * (self.V - E_K)
        I_L = g_L * (self.V - E_L)
        
        dVdt = (self.I_ext - I_Na - I_K - I_L - I_syn_total) / C_m
        
        tau_m_val = tau_m(self.V)
        tau_h_val = tau_h(self.V)
        tau_n_val = tau_n(self.V)
        
        m_inf_val = m_inf(self.V)
        h_inf_val = h_inf(self.V)
        n_inf_val = n_inf(self.V)
        
        self.V += dVdt * dt
        self.m = m_inf_val + (self.m - m_inf_val) * np.exp(-dt / tau_m_val)
        self.h = h_inf_val + (self.h - h_inf_val) * np.exp(-dt / tau_h_val)
        self.n = n_inf_val + (self.n - n_inf_val) * np.exp(-dt / tau_n_val)
        
        spiked = False
        if self.V >= self.threshold and (t - self.last_spike_time) > self.refractory_period:
            self.spike_times.append(t)
            self.last_spike_time = t
            spiked = True
        
        return self.V, spiked

class NeuralNetwork:
    def __init__(self, num_excitatory, num_inhibitory):
        self.num_excitatory = num_excitatory
        self.num_inhibitory = num_inhibitory
        self.num_neurons = num_excitatory + num_inhibitory
        
        self.neurons = []
        for i in range(num_excitatory):
            self.neurons.append(HHNeuron(i, 'excitatory'))
        for i in range(num_excitatory, self.num_neurons):
            self.neurons.append(HHNeuron(i, 'inhibitory'))
        
        self.connections = {}
        self.spike_matrix = None
        self.time_array = None
    
    def connect_random(self, p_ee=0.1, p_ei=0.1, p_ie=0.1, p_ii=0.1, 
                       g_ee=0.05, g_ei=0.05, g_ie=0.2, g_ii=0.2, delay=1.0):
        for i, pre in enumerate(self.neurons):
            for j, post in enumerate(self.neurons):
                if i == j:
                    continue
                
                pre_type = pre.neuron_type
                post_type = post.neuron_type
                
                if pre_type == 'excitatory' and post_type == 'excitatory':
                    p, g, syn_type = p_ee, g_ee, 'AMPA'
                elif pre_type == 'excitatory' and post_type == 'inhibitory':
                    p, g, syn_type = p_ei, g_ei, 'AMPA'
                elif pre_type == 'inhibitory' and post_type == 'excitatory':
                    p, g, syn_type = p_ie, g_ie, 'GABA'
                else:
                    p, g, syn_type = p_ii, g_ii, 'GABA'
                
                if np.random.random() < p:
                    syn = Synapse(syn_type, g_max=g, delay=delay)
                    post.add_synapse(syn)
                    
                    if i not in self.connections:
                        self.connections[i] = []
                    self.connections[i].append((j, syn))
    
    def connect_forward(self, input_neurons, output_neurons, syn_type='AMPA', g_max=0.1, delay=1.0):
        for i in input_neurons:
            for j in output_neurons:
                syn = Synapse(syn_type, g_max=g_max, delay=delay)
                self.neurons[j].add_synapse(syn)
                if i not in self.connections:
                    self.connections[i] = []
                self.connections[i].append((j, syn))
    
    def set_external_current(self, neuron_ids, current):
        for nid in neuron_ids:
            self.neurons[nid].I_ext = current
    
    def simulate(self, t_total, dt=0.05, verbose=True):
        num_steps = int(t_total / dt)
        self.time_array = np.linspace(0, t_total, num_steps)
        self.spike_matrix = np.zeros((self.num_neurons, num_steps), dtype=bool)
        self.V_matrix = np.zeros((self.num_neurons, num_steps))
        
        start_time = time.time()
        
        for step_idx, t in enumerate(self.time_array):
            spiked_neurons = []
            
            for i, neuron in enumerate(self.neurons):
                V, spiked = neuron.step(dt, t)
                self.V_matrix[i, step_idx] = V
                if spiked:
                    self.spike_matrix[i, step_idx] = True
                    spiked_neurons.append(i)
            
            for pre_idx in spiked_neurons:
                if pre_idx in self.connections:
                    for (post_idx, syn) in self.connections[pre_idx]:
                        syn.receive_spike(t)
            
            if verbose and step_idx % int(100 / dt) == 0:
                elapsed = time.time() - start_time
                print(f"  进度: {t:.0f}/{t_total} ms | 已发放: {np.sum(self.spike_matrix)} 次 | 耗时: {elapsed:.1f}s")
        
        total_time = time.time() - start_time
        if verbose:
            print(f"  模拟完成! 总耗时: {total_time:.2f}s, 平均: {total_time/t_total*1000:.2f}ms/100ms")
        
        return self.time_array, self.V_matrix, self.spike_matrix

def detect_oscillations(spike_matrix, dt, freq_range=(10, 100)):
    num_neurons = spike_matrix.shape[0]
    population_rate = np.sum(spike_matrix, axis=0) / (num_neurons * dt * 1e-3)
    
    from scipy.signal import welch
    nperseg = min(len(population_rate), 2048)
    freqs, psd = welch(population_rate, fs=1/dt*1000, nperseg=nperseg)
    
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if np.any(freq_mask):
        peak_idx = np.argmax(psd[freq_mask])
        peak_freq = freqs[freq_mask][peak_idx]
        peak_power = psd[freq_mask][peak_idx]
    else:
        peak_freq = 0
        peak_power = 0
    
    return freqs, psd, peak_freq, peak_power

def calculate_synchrony(spike_matrix, dt):
    population_rate = np.sum(spike_matrix, axis=0).astype(float)
    
    mean_rate = np.mean(population_rate)
    std_rate = np.std(population_rate)
    
    cv = std_rate / mean_rate if mean_rate > 0 else 0
    
    spike_times_all = []
    for i in range(spike_matrix.shape[0]):
        spike_times = np.where(spike_matrix[i])[0] * dt
        if len(spike_times) > 0:
            spike_times_all.append(spike_times)
    
    if len(spike_times_all) < 2:
        return cv, 0.0, 0.0
    
    all_spikes = np.concatenate(spike_times_all)
    all_spikes.sort()
    
    isi = np.diff(all_spikes)
    cv_isi = np.std(isi) / np.mean(isi) if len(isi) > 0 and np.mean(isi) > 0 else 0
    
    return cv, cv_isi, std_rate

def compute_firing_rates(spike_matrix, t_array, window_size=20.0):
    num_neurons = spike_matrix.shape[0]
    dt = t_array[1] - t_array[0]
    window_steps = int(window_size / dt)
    
    firing_rates = np.zeros((num_neurons, len(t_array)))
    
    for i in range(num_neurons):
        spikes = spike_matrix[i].astype(float)
        kernel = np.ones(window_steps) / (window_size * 1e-3)
        firing_rates[i] = np.convolve(spikes, kernel, mode='same') / dt
    
    return firing_rates

def plot_network_results(t_array, V_matrix, spike_matrix, network, title="神经网络模拟结果"):
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 3, figure=fig)
    
    ax_raster = fig.add_subplot(gs[0:2, :])
    num_neurons = spike_matrix.shape[0]
    num_exc = network.num_excitatory
    
    exc_spikes_y = []
    exc_spikes_x = []
    inh_spikes_y = []
    inh_spikes_x = []
    
    for i in range(num_neurons):
        spike_times = t_array[spike_matrix[i]]
        if i < num_exc:
            exc_spikes_x.extend(spike_times)
            exc_spikes_y.extend([i] * len(spike_times))
        else:
            inh_spikes_x.extend(spike_times)
            inh_spikes_y.extend([i] * len(spike_times))
    
    ax_raster.scatter(exc_spikes_x, exc_spikes_y, s=2, c='red', alpha=0.6, label='兴奋性')
    ax_raster.scatter(inh_spikes_x, inh_spikes_y, s=2, c='blue', alpha=0.6, label='抑制性')
    ax_raster.set_ylabel('神经元编号')
    ax_raster.set_title(f'{title} - 脉冲光栅图')
    ax_raster.legend()
    ax_raster.axhline(y=num_exc - 0.5, color='k', linestyle='--', alpha=0.5)
    ax_raster.text(0, num_exc / 2, '兴奋性', rotation=90, va='center')
    ax_raster.text(0, num_exc + (num_neurons - num_exc) / 2, '抑制性', rotation=90, va='center')
    
    ax_v = fig.add_subplot(gs[2, 0])
    for i in range(min(3, num_exc)):
        ax_v.plot(t_array, V_matrix[i], label=f'Exc {i}', alpha=0.7)
    for i in range(num_exc, min(num_exc + 2, num_neurons)):
        ax_v.plot(t_array, V_matrix[i], label=f'Inh {i - num_exc}', alpha=0.7, linestyle='--')
    ax_v.set_ylabel('膜电位 (mV)')
    ax_v.set_xlabel('时间 (ms)')
    ax_v.set_title('样本神经元膜电位')
    ax_v.legend(fontsize=8)
    ax_v.grid(True, alpha=0.3)
    
    ax_rate = fig.add_subplot(gs[2, 1])
    firing_rates = compute_firing_rates(spike_matrix, t_array)
    exc_rate = np.mean(firing_rates[:num_exc], axis=0)
    inh_rate = np.mean(firing_rates[num_exc:], axis=0)
    ax_rate.plot(t_array, exc_rate, 'r', label='兴奋性平均')
    ax_rate.plot(t_array, inh_rate, 'b', label='抑制性平均')
    ax_rate.set_ylabel('发放率 (Hz)')
    ax_rate.set_xlabel('时间 (ms)')
    ax_rate.set_title('群体发放率')
    ax_rate.legend()
    ax_rate.grid(True, alpha=0.3)
    
    ax_fft = fig.add_subplot(gs[2, 2])
    freqs, psd, peak_freq, peak_power = detect_oscillations(spike_matrix, t_array[1] - t_array[0])
    ax_fft.plot(freqs, psd)
    ax_fft.axvline(x=peak_freq, color='r', linestyle='--', alpha=0.5, label=f'峰值: {peak_freq:.1f} Hz')
    ax_fft.set_xlabel('频率 (Hz)')
    ax_fft.set_ylabel('功率谱密度')
    ax_fft.set_title('发放率频谱分析')
    ax_fft.legend()
    ax_fft.grid(True, alpha=0.3)
    ax_fft.set_xlim(0, 100)
    
    ax_sync = fig.add_subplot(gs[3, 0])
    cv, cv_isi, std_rate = calculate_synchrony(spike_matrix, t_array[1] - t_array[0])
    sync_metrics = ['发放率 CV', 'ISI CV', '发放率标准差']
    sync_values = [cv, cv_isi, std_rate]
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    bars = ax_sync.bar(sync_metrics, sync_values, color=colors)
    ax_sync.set_ylabel('数值')
    ax_sync.set_title('同步性指标')
    for bar, val in zip(bars, sync_values):
        ax_sync.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{val:.3f}', ha='center', va='bottom')
    ax_sync.grid(True, alpha=0.3, axis='y')
    
    ax_dist = fig.add_subplot(gs[3, 1])
    exc_counts = [np.sum(spike_matrix[i]) for i in range(num_exc)]
    inh_counts = [np.sum(spike_matrix[i]) for i in range(num_exc, num_neurons)]
    ax_dist.hist(exc_counts, alpha=0.5, label='兴奋性', color='red', bins=15)
    ax_dist.hist(inh_counts, alpha=0.5, label='抑制性', color='blue', bins=15)
    ax_dist.set_xlabel('脉冲数')
    ax_dist.set_ylabel('神经元数')
    ax_dist.set_title('脉冲发放分布')
    ax_dist.legend()
    ax_dist.grid(True, alpha=0.3)
    
    ax_heatmap = fig.add_subplot(gs[3, 2])
    rate_heatmap = firing_rates[:, ::10]
    im = ax_heatmap.imshow(rate_heatmap, aspect='auto', cmap='hot', 
                          extent=[t_array[0], t_array[-1], num_neurons, 0])
    ax_heatmap.set_xlabel('时间 (ms)')
    ax_heatmap.set_ylabel('神经元编号')
    ax_heatmap.set_title('发放率热力图')
    plt.colorbar(im, ax=ax_heatmap, label='Hz')
    
    plt.tight_layout()
    plt.savefig('hh_network_results.png', dpi=150, bbox_inches='tight')
    print(f"结果图已保存为 hh_network_results.png")
    
    return fig

def demo_async_irregular():
    print("\n" + "=" * 70)
    print("演示1: 异步不规则发放 (AI 状态)")
    print("=" * 70)
    
    nn = NeuralNetwork(num_excitatory=80, num_inhibitory=20)
    nn.connect_random(p_ee=0.1, p_ei=0.1, p_ie=0.1, p_ii=0.1,
                      g_ee=0.02, g_ei=0.02, g_ie=0.1, g_ii=0.1, delay=1.0)
    nn.set_external_current(range(80), 7.0)
    nn.set_external_current(range(80, 100), 8.0)
    
    print("开始模拟 (1000 ms)...")
    t_array, V_matrix, spike_matrix = nn.simulate(1000, dt=0.05)
    
    fig = plot_network_results(t_array, V_matrix, spike_matrix, nn, 
                              title="异步不规则发放 (AI 状态)")
    plt.close(fig)
    
    return nn, t_array, V_matrix, spike_matrix

def demo_synchronous_oscillation():
    print("\n" + "=" * 70)
    print("演示2: 同步振荡 (Gamma 振荡)")
    print("=" * 70)
    
    nn = NeuralNetwork(num_excitatory=80, num_inhibitory=20)
    nn.connect_random(p_ee=0.2, p_ei=0.3, p_ie=0.3, p_ii=0.2,
                      g_ee=0.08, g_ei=0.08, g_ie=0.4, g_ii=0.2, delay=1.5)
    nn.set_external_current(range(80), 12.0)
    nn.set_external_current(range(80, 100), 10.0)
    
    print("开始模拟 (1000 ms)...")
    t_array, V_matrix, spike_matrix = nn.simulate(1000, dt=0.05)
    
    fig = plot_network_results(t_array, V_matrix, spike_matrix, nn,
                              title="同步振荡 (Gamma 振荡)")
    plt.close(fig)
    
    return nn, t_array, V_matrix, spike_matrix

def demo_rate_coding():
    print("\n" + "=" * 70)
    print("演示3: 发放率编码 (刺激强度编码)")
    print("=" * 70)
    
    nn = NeuralNetwork(num_excitatory=100, num_inhibitory=25)
    nn.connect_random(p_ee=0.05, p_ei=0.1, p_ie=0.1, p_ii=0.05,
                      g_ee=0.03, g_ei=0.03, g_ie=0.15, g_ii=0.1, delay=1.0)
    
    group1 = range(0, 25)
    group2 = range(25, 50)
    group3 = range(50, 75)
    group4 = range(75, 100)
    
    current_levels = [5, 10, 15, 20]
    groups = [group1, group2, group3, group4]
    
    for g, curr in zip(groups, current_levels):
        nn.set_external_current(g, curr)
    
    nn.set_external_current(range(100, 125), 8.0)
    
    print("开始模拟 (800 ms)...")
    t_array, V_matrix, spike_matrix = nn.simulate(800, dt=0.05)
    
    firing_rates = compute_firing_rates(spike_matrix, t_array)
    
    fig = plt.figure(figsize=(14, 8))
    
    ax1 = plt.subplot(2, 2, 1)
    for g, curr, color in zip(groups, current_levels, ['r', 'g', 'b', 'm']):
        rates = np.mean(firing_rates[list(g)], axis=0)
        ax1.plot(t_array, rates, color=color, label=f'{curr} uA/cm²')
    ax1.set_ylabel('平均发放率 (Hz)')
    ax1.set_xlabel('时间 (ms)')
    ax1.set_title('不同刺激强度的发放率响应')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 2, 2)
    mean_rates = [np.mean(firing_rates[list(g), -200:]) for g in groups]
    ax2.plot(current_levels, mean_rates, 'o-', linewidth=2, markersize=8)
    ax2.set_xlabel('刺激电流 (uA/cm²)')
    ax2.set_ylabel('稳态发放率 (Hz)')
    ax2.set_title('发放率-电流 (f-I) 曲线')
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(2, 2, 3)
    for i, (g, curr) in enumerate(zip(groups, current_levels)):
        rates = firing_rates[list(g)].flatten()
        ax3.scatter([curr] * len(rates), rates, alpha=0.1, s=5)
    ax3.boxplot([firing_rates[list(g)].flatten() for g in groups], 
                positions=current_levels, widths=2)
    ax3.set_xlabel('刺激电流 (uA/cm²)')
    ax3.set_ylabel('发放率分布 (Hz)')
    ax3.set_title('发放率分布')
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(2, 2, 4)
    num_spikes = [np.sum(spike_matrix[list(g)]) for g in groups]
    ax4.bar([f'{c} μA' for c in current_levels], num_spikes, 
            color=['r', 'g', 'b', 'm'], alpha=0.7)
    ax4.set_ylabel('总脉冲数')
    ax4.set_title('各神经元群体总发放数')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('hh_rate_coding.png', dpi=150)
    print("发放率编码结果已保存为 hh_rate_coding.png")
    plt.close()
    
    return nn, t_array, V_matrix, spike_matrix

if __name__ == "__main__":
    print("=" * 70)
    print("Hodgkin-Huxley 神经网络模拟器")
    print("=" * 70)
    print("包含: AMPA, NMDA, GABA 突触 | 随机连接网络")
    print("分析: 同步振荡 | 发放率编码 | 脉冲光栅图")
    print("=" * 70)
    
    np.random.seed(42)
    
    nn_ai, t_ai, V_ai, spikes_ai = demo_async_irregular()
    nn_sync, t_sync, V_sync, spikes_sync = demo_synchronous_oscillation()
    nn_rate, t_rate, V_rate, spikes_rate = demo_rate_coding()
    
    print("\n" + "=" * 70)
    print("所有演示完成!")
    print("=" * 70)
