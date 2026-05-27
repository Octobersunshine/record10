"""快速测试矩张量反演功能"""
import numpy as np
from focal_mechanism import (
    mt_from_sdr, mt_decompose, mt_combine,
    mt_p_radiation, mt_inversion_amplitude,
    sph_to_cart
)

np.random.seed(42)

print("=" * 60)
print("测试 1: 纯双力偶 (DC) 源")
print("=" * 60)
M_dc = mt_from_sdr(120.0, 50.0, 30.0, M0=1.0)
dec = mt_decompose(M_dc)
print(f"真值 DC 源: ISO={dec['iso_percent']:+6.1f}%, DC={dec['dc_percent']:5.1f}%, CLVD={dec['clvd_percent']:+6.1f}%")
print(f"期望:       ISO≈0%, DC≈100%, CLVD≈0%")

print("\n" + "=" * 60)
print("测试 2: 火山地震 (含 ISO 膨胀分量)")
print("=" * 60)
M_volc = mt_combine(iso_frac=0.30, dc_frac=0.70, clvd_frac=0.0,
                    strike=120.0, dip=50.0, rake=30.0, M0=1.0)
dec_v = mt_decompose(M_volc)
print(f"构造火山源: ISO输入=+30%, DC输入=70%")
print(f"分解结果:   ISO={dec_v['iso_percent']:+6.1f}%, DC={dec_v['dc_percent']:5.1f}%, CLVD={dec_v['clvd_percent']:+6.1f}%")
print(f"(分解比例与输入比例略有差异是正常的, 因为分解是非线性的)")

print("\n" + "=" * 60)
print("测试 3: 诱发地震 (含 CLVD 分量)")
print("=" * 60)
M_ind = mt_combine(iso_frac=0.0, dc_frac=0.70, clvd_frac=0.30,
                   strike=120.0, dip=50.0, rake=30.0,
                   clvd_axis_az=90.0, clvd_axis_plunge=0.0, M0=1.0)
dec_i = mt_decompose(M_ind)
print(f"构造诱发源: DC输入=70%, CLVD输入=+30%")
print(f"分解结果:   ISO={dec_i['iso_percent']:+6.1f}%, DC={dec_i['dc_percent']:5.1f}%, CLVD={dec_i['clvd_percent']:+6.1f}%")
print(f"(CLVD 输入符号为正表示拉伸型)")

print("\n" + "=" * 60)
print("测试 4: 线性反演 — 纯 DC 源")
print("=" * 60)
N = 30
azs = np.random.uniform(0, 360, N)
toas = np.random.uniform(20, 160, N)

obs_dc = []
for az, toa in zip(azs, toas):
    g = sph_to_cart(az, toa)
    Fp = mt_p_radiation(M_dc, g)
    pol = 1 if Fp >= 0 else -1
    amp = abs(Fp) * np.random.lognormal(0, 0.10)
    obs_dc.append({'az': float(az), 'toa': float(toa), 'pol': pol, 'amp_P': float(amp)})

res = mt_inversion_amplitude(obs_dc, phase='P', allow_iso=True)
dec_inv = res['decomposition']
print(f"反演结果: ISO={dec_inv['iso_percent']:+6.1f}%, DC={dec_inv['dc_percent']:5.1f}%, CLVD={dec_inv['clvd_percent']:+6.1f}%")
print(f"振幅 RMS: {res['residual']:.4f}, 观测数: {res['n_obs']}")

print("\n" + "=" * 60)
print("测试 5: 线性反演 — 火山源 (含 ISO)")
print("=" * 60)
obs_volc = []
for az, toa in zip(azs, toas):
    g = sph_to_cart(az, toa)
    Fp = mt_p_radiation(M_volc, g)
    pol = 1 if Fp >= 0 else -1
    amp = abs(Fp) * np.random.lognormal(0, 0.10)
    obs_volc.append({'az': float(az), 'toa': float(toa), 'pol': pol, 'amp_P': float(amp)})

res_v = mt_inversion_amplitude(obs_volc, phase='P', allow_iso=True)
dec_v_inv = res_v['decomposition']
print(f"反演结果: ISO={dec_v_inv['iso_percent']:+6.1f}%, DC={dec_v_inv['dc_percent']:5.1f}%, CLVD={dec_v_inv['clvd_percent']:+6.1f}%")
print(f"振幅 RMS: {res_v['residual']:.4f}, 观测数: {res_v['n_obs']}")

# 错误地强制纯偏差
res_v_wrong = mt_inversion_amplitude(obs_volc, phase='P', allow_iso=False)
dec_vw = res_v_wrong['decomposition']
print(f"强制纯偏差: ISO={dec_vw['iso_percent']:+6.1f}%, DC={dec_vw['dc_percent']:5.1f}%, CLVD={dec_vw['clvd_percent']:+6.1f}%")
print(f"振幅 RMS: {res_v_wrong['residual']:.4f}")
if res_v_wrong['residual'] > res_v['residual'] * 1.5:
    print("→ 强制纯偏差导致拟合显著变差!")

print("\n" + "=" * 60)
print("所有测试完成!")
print("=" * 60)
