import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

R = 8.314
T = 353.15
F = 96485
n = 2

def pemfc_model(i, E_ocv, A, R_ohm, iL):
    """
    改进的PEMFC极化曲线模型
    V = E_ocv - A*ln(i) - R_ohm*i - (R*T/(n*F))*ln(1 - i/iL)
    
    将A和i0合并为一个参数A = R*T/(n*F*alpha)，避免i0的数值问题
    """
    i_safe = np.maximum(i, 1e-6)
    eta_act = A * np.log(i_safe)
    eta_ohm = R_ohm * i_safe
    eta_conc = (R * T / (n * F)) * np.log(np.maximum(1 - i_safe / iL, 1e-6))
    return E_ocv - eta_act - eta_ohm - eta_conc

def calculate_losses(i, E_ocv, A, R_ohm, iL):
    """计算各损失项"""
    i_safe = np.maximum(i, 1e-6)
    eta_act = A * np.log(i_safe)
    eta_ohm = R_ohm * i_safe
    eta_conc = (R * T / (n * F)) * np.log(np.maximum(1 - i_safe / iL, 1e-6))
    return eta_act, eta_ohm, eta_conc

def generate_realistic_data():
    """生成更真实的PEMFC实验数据"""
    i_data = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 
                       0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
    
    E_ocv_true = 1.0
    A_true = 0.06
    R_ohm_true = 0.12
    iL_true = 1.8
    
    v_true = pemfc_model(i_data, E_ocv_true, A_true, R_ohm_true, iL_true)
    noise = np.random.normal(0, 0.015, size=len(v_true))
    v_data = np.maximum(v_true + noise, 0.1)
    
    return i_data, v_data, [E_ocv_true, A_true, R_ohm_true, iL_true]

def fit_with_estimation(i_data, v_data):
    """使用线性估计初始化参数，提高拟合稳定性"""
    
    mask = (i_data > 0.05) & (i_data < 1.0)
    i_lin = i_data[mask]
    v_lin = v_data[mask]
    
    slope, intercept = np.polyfit(i_lin, v_lin, 1)
    R_ohm_est = -slope
    E_ocv_est = intercept
    
    A_est = 0.05
    iL_est = 2.0
    
    initial_guess = [E_ocv_est, A_est, R_ohm_est, iL_est]
    bounds = ([0.8, 0.02, 0.05, 1.5], 
              [1.2, 0.15, 0.3, 2.5])
    
    popt, pcov = curve_fit(pemfc_model, i_data, v_data, 
                           p0=initial_guess, bounds=bounds, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    
    return popt, perr, initial_guess

def plot_comprehensive(i_data, v_data, popt, true_params=None):
    """绘制综合分析图"""
    fig = plt.figure(figsize=(16, 10))
    
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    i_fit = np.linspace(0.01, popt[3] * 0.98, 300)
    v_fit = pemfc_model(i_fit, *popt)
    eta_act, eta_ohm, eta_conc = calculate_losses(i_fit, *popt)
    power = i_fit * v_fit
    
    ax1.scatter(i_data, v_data, color='blue', label='实验数据', s=60, 
                edgecolor='black', zorder=5, alpha=0.8)
    ax1.plot(i_fit, v_fit, 'r-', linewidth=3, label='拟合曲线')
    ax1.set_xlabel('Current Density (A/cm$^2$)', fontsize=12)
    ax1.set_ylabel('Voltage (V)', fontsize=12)
    ax1.set_title('PEMFC Polarization Curve', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 1.1])
    
    ax2.plot(i_fit, eta_act, 'g-', linewidth=2.5, label='Activation Loss')
    ax2.plot(i_fit, eta_ohm, 'orange', linewidth=2.5, label='Ohmic Loss')
    ax2.plot(i_fit, eta_conc, 'purple', linewidth=2.5, label='Concentration Loss')
    ax2.set_xlabel('Current Density (A/cm$^2$)', fontsize=12)
    ax2.set_ylabel('Overpotential (V)', fontsize=12)
    ax2.set_title('Loss Breakdown Analysis', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    ax3_twin = ax3.twinx()
    ax3.plot(i_fit, v_fit, 'b-', linewidth=2.5, label='Voltage')
    ax3_twin.plot(i_fit, power, 'r-', linewidth=2.5, label='Power Density')
    
    max_power_idx = np.argmax(power)
    max_power = power[max_power_idx]
    i_at_max = i_fit[max_power_idx]
    v_at_max = v_fit[max_power_idx]
    
    ax3.scatter([i_at_max], [v_at_max], color='green', s=100, zorder=5, 
                label=f'Max Power Point')
    ax3_twin.scatter([i_at_max], [max_power], color='green', s=100, zorder=5)
    
    ax3.axhline(y=v_at_max, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=i_at_max, color='gray', linestyle='--', alpha=0.5)
    
    ax3.set_xlabel('Current Density (A/cm$^2$)', fontsize=12)
    ax3.set_ylabel('Voltage (V)', fontsize=12, color='b')
    ax3_twin.set_ylabel('Power Density (W/cm$^2$)', fontsize=12, color='r')
    ax3.set_title('Voltage and Power Density', fontsize=14, fontweight='bold')
    
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='lower left')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_ylim([0, 1.1])
    
    ax4.axis('off')
    param_text = f"""
    FITTED PARAMETERS:
    ──────────────────────────────
    Open Circuit Voltage (E_ocv):
        {popt[0]:.4f} ± {perr[0]:.4f} V
    
    Tafel Coefficient (A):
        {popt[1]:.4f} ± {perr[1]:.4f} V
    
    Ohmic Resistance (R_ohm):
        {popt[2]:.4f} ± {perr[2]:.4f} Ω·cm²
    
    Limiting Current Density (iL):
        {popt[3]:.4f} ± {perr[3]:.4f} A/cm²
    
    PERFORMANCE METRICS:
    ──────────────────────────────
    Peak Power Density:
        {max_power:.4f} W/cm²
    
    @ Current Density:
        {i_at_max:.4f} A/cm²
    
    @ Voltage:
        {v_at_max:.4f} V
    """
    ax4.text(0.05, 0.95, param_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig('pemfc_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return max_power, i_at_max, v_at_max

def print_detailed_results(popt, perr, max_power, i_at_max, v_at_max):
    """打印详细结果"""
    print("\n" + "="*70)
    print("           PEMFC POLARIZATION CURVE FITTING RESULTS")
    print("="*70)
    print("\n【Fitted Parameters】")
    print("-" * 70)
    print(f"  Open Circuit Voltage (E_ocv):  {popt[0]:.4f} ± {perr[0]:.4f} V")
    print(f"  Tafel Coefficient (A):         {popt[1]:.4f} ± {perr[1]:.4f} V")
    print(f"  Ohmic Resistance (R_ohm):      {popt[2]:.4f} ± {perr[2]:.4f} Ω·cm²")
    print(f"  Limiting Current Density (iL): {popt[3]:.4f} ± {perr[3]:.4f} A/cm²")
    
    alpha_est = R * T / (n * F * popt[1])
    print(f"\n  Estimated Transfer Coefficient (α): {alpha_est:.4f}")
    
    print("\n【Performance Metrics】")
    print("-" * 70)
    print(f"  Peak Power Density:    {max_power:.4f} W/cm²")
    print(f"  Current Density @ MPP: {i_at_max:.4f} A/cm²")
    print(f"  Voltage @ MPP:         {v_at_max:.4f} V")
    
    print("\n【Loss Analysis at Operating Point (0.6 A/cm²)】")
    print("-" * 70)
    i_op = 0.6
    eta_act_op, eta_ohm_op, eta_conc_op = calculate_losses(i_op, *popt)
    total_loss = eta_act_op + eta_ohm_op + eta_conc_op
    print(f"  Activation Loss:    {eta_act_op:.4f} V ({eta_act_op/total_loss*100:.1f}%)")
    print(f"  Ohmic Loss:         {eta_ohm_op:.4f} V ({eta_ohm_op/total_loss*100:.1f}%)")
    print(f"  Concentration Loss: {eta_conc_op:.4f} V ({eta_conc_op/total_loss*100:.1f}%)")
    print(f"  Total Loss:         {total_loss:.4f} V")
    print("=" * 70)

if __name__ == "__main__":
    print("PEMFC Polarization Curve Fitting - Advanced Version")
    print("=" * 60)
    
    print("\n[1] Generating realistic PEMFC experimental data...")
    i_data, v_data, true_params = generate_realistic_data()
    
    print("\n[2] Experimental Data Points:")
    print("-" * 40)
    print(f"{'i (A/cm²)':>12} {'V (V)':>10}")
    print("-" * 40)
    for i, v in zip(i_data, v_data):
        print(f"{i:>12.2f} {v:>10.4f}")
    
    print("\n[3] Performing curve fitting with parameter estimation...")
    popt, perr, initial_guess = fit_with_estimation(i_data, v_data)
    
    print("\n[4] Plotting comprehensive analysis...")
    max_power, i_at_max, v_at_max = plot_comprehensive(i_data, v_data, popt)
    
    print_detailed_results(popt, perr, max_power, i_at_max, v_at_max)
    
    print("\n✓ Analysis complete! Results saved to:")
    print("  - pemfc_comprehensive_analysis.png")
