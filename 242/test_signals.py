import numpy as np
from signal_generator import generate_signal, _flattop_window

np.random.seed(42)

print("=" * 70)
print("=== 数字信号生成完整测试 ===")
print("=" * 70)

sample_rate = 1000
amp = 1.0
phase = 0.0
duration = 1.0

print("\n" + "=" * 50)
print("--- 测试 1: 基础波形 ---")
print("=" * 50)
basic_waves = ['sine', 'square', 'triangle', 'sawtooth']
basic_names = ['正弦波', '方波', '三角波', '锯齿波']

for wave_type, wave_name in zip(basic_waves, basic_names):
    t, s = generate_signal(wave_type, frequency=5, sample_rate=sample_rate,
                           amplitude=amp, phase=phase, duration=duration)
    print(f"{wave_name}: 长度={len(s)}, 范围=[{np.min(s):.3f}, {np.max(s):.3f}], "
          f"均值={np.mean(s):.4f}, 标准差={np.std(s):.4f}")

print("\n" + "=" * 50)
print("--- 测试 2: 调幅波 (AM) ---")
print("=" * 50)

t, s_am = generate_signal('am', frequency=5, sample_rate=sample_rate,
                          amplitude=amp, phase=phase, duration=duration,
                          carrier_freq=50, mod_freq=5, mod_index=0.8)
envelope = amp * (1 + 0.8 * np.sin(2 * np.pi * 5 * t))
env_max = np.max(np.abs(s_am))
env_min = np.min(np.abs(s_am))
print(f"AM信号: 长度={len(s_am)}")
print(f"  载波频率=50Hz, 调制频率=5Hz, 调制度=0.8")
print(f"  包络范围: 预期[{amp*(1-0.8):.1f}, {amp*(1+0.8):.1f}], "
      f"实际[{env_min:.3f}, {env_max:.3f}]")
print(f"  最大幅度正确: {np.isclose(env_max, amp*(1+0.8), rtol=0.05)}")

print("\n" + "=" * 50)
print("--- 测试 3: 调频波 (FM) ---")
print("=" * 50)

t, s_fm = generate_signal('fm', frequency=5, sample_rate=sample_rate,
                          amplitude=amp, phase=phase, duration=duration,
                          carrier_freq=50, mod_freq=5, mod_index=3)
inst_freq = np.diff(np.unwrap(np.angle(s_fm + 1j * np.abs(np.gradient(s_fm))))) / (2 * np.pi) * sample_rate
print(f"FM信号: 长度={len(s_fm)}")
print(f"  载波频率=50Hz, 调制频率=5Hz, 调制指数=3")
print(f"  频偏= {3*5:.1f}Hz, 预期频率范围=[{50-15:.1f}, {50+15:.1f}]Hz")
print(f"  幅度范围: [{np.min(s_fm):.3f}, {np.max(s_fm):.3f}] (恒幅)")
print(f"  幅度恒定: {np.isclose(np.std(s_fm), np.std(np.sin(2*np.pi*50*t)), rtol=0.1)}")

print("\n" + "=" * 50)
print("--- 测试 4: 扫频信号 (Chirp) ---")
print("=" * 50)

t, s_chirp_lin = generate_signal('chirp', frequency=1, sample_rate=sample_rate,
                                 amplitude=amp, phase=phase, duration=2.0,
                                 freq_start=1, freq_end=21, chirp_method='linear')
print(f"线性扫频: 起始频率=1Hz, 终止频率=21Hz, 时长=2s")
print(f"  扫频速率=10Hz/s, 长度={len(s_chirp_lin)}")
print(f"  幅度范围: [{np.min(s_chirp_lin):.3f}, {np.max(s_chirp_lin):.3f}]")

t, s_chirp_exp = generate_signal('chirp', frequency=1, sample_rate=sample_rate,
                                 amplitude=amp, phase=phase, duration=2.0,
                                 freq_start=1, freq_end=100, chirp_method='exponential')
print(f"\n指数扫频: 起始频率=1Hz, 终止频率=100Hz, 时长=2s")
print(f"  长度={len(s_chirp_exp)}")
print(f"  幅度范围: [{np.min(s_chirp_exp):.3f}, {np.max(s_chirp_exp):.3f}]")

print("\n" + "=" * 50)
print("--- 测试 5: 高斯白噪声 (SNR) ---")
print("=" * 50)

t, s_clean = generate_signal('sine', frequency=5, sample_rate=sample_rate,
                             amplitude=amp, phase=phase, duration=2.0)

for snr_dB in [0, 10, 20, 30]:
    np.random.seed(42)
    t, s_noisy = generate_signal('sine', frequency=5, sample_rate=sample_rate,
                                 amplitude=amp, phase=phase, duration=2.0, snr=snr_dB)
    noise = s_noisy - s_clean
    signal_power = np.mean(s_clean**2)
    noise_power = np.mean(noise**2)
    actual_snr = 10 * np.log10(signal_power / noise_power)
    print(f"SNR设置={snr_dB}dB: 实际SNR={actual_snr:.2f}dB, "
          f"误差={abs(actual_snr - snr_dB):.2f}dB, "
          f"误差<0.5dB: {abs(actual_snr - snr_dB) < 0.5}")

print("\n" + "=" * 50)
print("--- 测试 6: 组合功能 (加窗+噪声) ---")
print("=" * 50)

t, s_combo = generate_signal('am', frequency=5, sample_rate=sample_rate,
                             amplitude=amp, phase=phase, duration=duration,
                             carrier_freq=50, mod_freq=5, mod_index=0.7,
                             window='hann', snr=15)
print(f"AM信号 + 汉宁窗 + SNR=15dB:")
print(f"  端点值: s[0]={s_combo[0]:.4e}, s[-1]={s_combo[-1]:.4e}")
print(f"  端点接近0: {np.abs(s_combo[0]) < 0.01 and np.abs(s_combo[-1]) < 0.01}")

noise_est = s_combo[len(s_combo)//2:]
clean_est = generate_signal('am', frequency=5, sample_rate=sample_rate,
                            amplitude=amp, phase=phase, duration=duration,
                            carrier_freq=50, mod_freq=5, mod_index=0.7,
                            window='hann')[1]
actual_noise = s_combo - clean_est
signal_power = np.mean(clean_est**2)
noise_power = np.mean(actual_noise**2)
actual_snr = 10 * np.log10(signal_power / noise_power)
print(f"  实际SNR: {actual_snr:.2f}dB (预期≈15dB)")

print("\n" + "=" * 50)
print("--- 测试 7: 参数验证 (错误处理) ---")
print("=" * 50)

try:
    generate_signal('am', frequency=5, sample_rate=sample_rate,
                    amplitude=amp, phase=phase, duration=duration,
                    carrier_freq=50, mod_freq=5, mod_index=1.5)
    print("AM调制度>1 应该报错: ❌ 未报错")
except ValueError as e:
    print(f"AM调制度>1 正确报错: ✅ {e}")

try:
    generate_signal('unknown', frequency=5, sample_rate=sample_rate,
                    amplitude=amp, phase=phase, duration=duration)
    print("未知波形类型应该报错: ❌ 未报错")
except ValueError as e:
    print(f"未知波形类型正确报错: ✅ {e}")

try:
    generate_signal('chirp', frequency=5, sample_rate=sample_rate,
                    amplitude=amp, phase=phase, duration=duration,
                    freq_start=1, freq_end=10, chirp_method='unknown')
    print("未知扫频方式应该报错: ❌ 未报错")
except ValueError as e:
    print(f"未知扫频方式正确报错: ✅ {e}")

try:
    generate_signal('sine', frequency=5, sample_rate=sample_rate,
                    amplitude=amp, phase=phase, duration=duration, window='unknown')
    print("未知窗函数应该报错: ❌ 未报错")
except ValueError as e:
    print(f"未知窗函数正确报错: ✅ {e}")

print("\n" + "=" * 50)
print("--- 测试 8: 默认参数测试 ---")
print("=" * 50)

t, s_am_default = generate_signal('am', frequency=5, sample_rate=sample_rate,
                                  amplitude=amp, phase=phase, duration=duration)
print(f"AM默认参数: 载波频率={5*10}Hz, 调制频率=5Hz, 调制度=1.0")
print(f"  幅度范围: [{np.min(s_am_default):.3f}, {np.max(s_am_default):.3f}]")

t, s_chirp_default = generate_signal('chirp', frequency=10, sample_rate=sample_rate,
                                     amplitude=amp, phase=phase, duration=duration)
print(f"\nChirp默认参数: 起始频率=10Hz, 终止频率={10*10}Hz, 线性扫频")
print(f"  幅度范围: [{np.min(s_chirp_default):.3f}, {np.max(s_chirp_default):.3f}]")

print("\n" + "=" * 70)
print("=== 所有测试完成 ===")
print("=" * 70)
