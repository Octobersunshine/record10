import numpy as np
from ebm_1d import EnergyBalanceModel1D

def test_radiation_calibration():
    print("=" * 60)
    print("长波辐射校准测试")
    print("=" * 60)
    
    model = EnergyBalanceModel1D(n_lat=18, dt=86400.0 * 10, n_years=10)
    
    T_test = np.linspace(250, 310, 61)
    
    print("\n" + "=" * 60)
    print("长波辐射参数验证 (OLR vs 温度)")
    print("=" * 60)
    print(f"{'T(K)':>8} {'T(°C)':>8} {'OLR(W/m²)':>12} {'sigma*T^4':>12} {'ε_eff':>10}")
    print("-" * 60)
    
    for T in [255, 270, 288, 300, 310]:
        olr = model.compute_olr_calibrated(np.array([T]))[0]
        sigma_T4 = model.sigma * T**4
        eps_eff = olr / sigma_T4
        print(f"{T:>8.0f} {T-273.15:>8.1f} {olr:>12.2f} {sigma_T4:>12.2f} {eps_eff:>10.3f}")
    
    print("\n" + "=" * 60)
    print("温室气体透过率验证")
    print("=" * 60)
    
    T_ref = 288.0
    olr_ref = model.compute_olr_calibrated(np.array([T_ref]))[0]
    sigma_T4_ref = model.sigma * T_ref**4
    tau_eff = olr_ref / sigma_T4_ref
    
    print(f"参考温度 T_ref = {T_ref} K")
    print(f"黑体辐射 σT⁴ = {sigma_T4_ref:.2f} W/m²")
    print(f"OLR = {olr_ref:.2f} W/m²")
    print(f"有效大气透过率 τ_eff = {tau_eff:.3f}")
    print(f"温室效应捕获 = {sigma_T4_ref - olr_ref:.2f} W/m²")
    
    return tau_eff

def test_model():
    print("=" * 60)
    print("一维能量平衡模型测试")
    print("=" * 60)
    
    model = EnergyBalanceModel1D(n_lat=18, dt=86400.0 * 10, n_years=20)
    
    print(f"\n网格配置:")
    print(f"  纬度带数量: {model.n_lat}")
    print(f"  纬度范围: {model.lat[0]:.1f}° ~ {model.lat[-1]:.1f}°")
    print(f"  时间步长: {model.dt / 86400:.1f} 天")
    print(f"  总步数: {model.n_steps}")
    
    print(f"\n初始温度:")
    print(f"  赤道: {model.T[0, model.n_lat//2] - 273.15:.1f}°C")
    print(f"  极地: {model.T[0, 0] - 273.15:.1f}°C")
    
    print(f"\n物理参数:")
    print(f"  太阳常数 Q0 = {model.Q0} W/m²")
    print(f"  热容量 C = {model.C:.2e} J/m²/K")
    print(f"  扩散系数 D = {model.D}")
    print(f"  Stefan-Boltzmann σ = {model.sigma:.2e} W/m²/K⁴")
    print(f"  有效发射率 ε = {model.epsilon:.3f}")
    
    print("\n" + "=" * 60)
    print("运行模型...")
    print("=" * 60)
    
    T_final = model.run()
    
    print(f"\n最终平衡态温度:")
    print(f"  赤道: {T_final[-1, model.n_lat//2] - 273.15:.1f}°C")
    print(f"  极地: {T_final[-1, 0] - 273.15:.1f}°C")
    
    weights = np.cos(model.phi)
    T_global = np.average(T_final[-1, :], weights=weights)
    print(f"  全球平均: {T_global - 273.15:.1f}°C")
    
    print(f"\n温度变化:")
    delta_T = T_final[-1, :] - T_final[0, :]
    print(f"  最大变化: {np.max(delta_T):.1f} K")
    print(f"  最小变化: {np.min(delta_T):.1f} K")
    
    print("\n" + "=" * 60)
    print("能量平衡检查（最后时刻）")
    print("=" * 60)
    
    balance = model.verify_energy_balance(T_final[-1, :])
    print(f"\n各能量项的全球平均值 (W/m²):")
    print(f"  短波吸收: {balance['shortwave']:.2f}")
    print(f"  长波发射: {balance['longwave']:.2f}")
    print(f"  扩散通量: {balance['diffusion']:.2f}")
    print(f"  净能量不平衡: {balance['net']:.6f}")
    print(f"  全球平均温度: {balance['T_global']:.1f}°C")
    
    print("\n" + "=" * 60)
    print("温度偏差验证")
    print("=" * 60)
    
    expected_T = 15.0
    T_dev = balance['T_global'] - expected_T
    
    print(f"预期全球平均温度: {expected_T:.1f}°C")
    print(f"模拟全球平均温度: {balance['T_global']:.1f}°C")
    print(f"温度偏差: {T_dev:.1f}°C")
    
    if abs(T_dev) > 2.0:
        print("\n⚠️  警告: 温度偏差超过2°C，需要检查参数校准!")
    else:
        print("\n✓ 温度偏差在合理范围内 (< 2°C)")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    
    return T_final

if __name__ == "__main__":
    test_radiation_calibration()
    print("\n" + "="*60 + "\n")
    T = test_model()
