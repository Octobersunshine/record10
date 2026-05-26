import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

def pemfc_polarization_model(i, E_ocv, a, b, i0, iL):
    """
    PEMFC极化曲线半经验模型
    V = E_ocv - a*ln(i/i0) - b*i - (R*T/(2*F))*ln(1 - i/iL)
    
    参数:
    i: 电流密度 (A/cm²)
    E_ocv: 开路电压 (V)
    a: Tafel斜率相关系数
    b: 欧姆电阻 (Ω·cm²)
    i0: 交换电流密度 (A/cm²)
    iL: 极限电流密度 (A/cm²)
    """
    eta_act = a * np.log(i / i0)
    eta_ohm = b * i
    eta_conc = (8.314 * 353.15 / (2 * 96485)) * np.log(1 - i / iL)
    return E_ocv - eta_act - eta_ohm - eta_conc

def simplified_model(i, E_ocv, A, B, C):
    """
    简化的极化曲线模型
    V = E_ocv - A*ln(i) - B*i - C*ln(1 - i/iL_guess)
    """
    iL_guess = 2.0
    return E_ocv - A * np.log(i + 1e-10) - B * i - C * np.log(1 - i / iL_guess + 1e-10)

def generate_sample_data():
    """
    生成模拟的PEMFC实验数据
    """
    i_data = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
                       0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8])
    
    E_ocv_true = 1.23
    a_true = 0.05
    b_true = 0.15
    i0_true = 1e-4
    iL_true = 2.0
    
    v_true = pemfc_polarization_model(i_data, E_ocv_true, a_true, b_true, i0_true, iL_true)
    noise = np.random.normal(0, 0.02, size=len(v_true))
    v_data = v_true + noise
    
    return i_data, v_data

def fit_polarization_curve(i_data, v_data):
    """
    拟合极化曲线
    """
    i_data = np.array(i_data)
    v_data = np.array(v_data)
    
    mask = (i_data > 0) & (i_data < 1.9)
    i_fit = i_data[mask]
    v_fit = v_data[mask]
    
    initial_guess = [1.0, 0.05, 0.1, 1e-4, 2.0]
    bounds = ([0.5, 0.01, 0.01, 1e-6, 1.5], 
              [1.5, 0.2, 0.5, 1e-2, 3.0])
    
    try:
        popt, pcov = curve_fit(pemfc_polarization_model, i_fit, v_fit, 
                               p0=initial_guess, bounds=bounds, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except:
        print("使用简化模型拟合...")
        i_data_safe = i_data[i_data > 0]
        v_data_safe = v_data[i_data > 0]
        initial_guess_simple = [1.0, 0.1, 0.15, 0.02]
        popt, pcov = curve_fit(simplified_model, i_data_safe, v_data_safe, 
                               p0=initial_guess_simple, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr

def plot_results(i_data, v_data, popt):
    """
    绘制拟合结果
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    i_fit = np.linspace(0.01, 1.9, 200)
    v_fit = pemfc_polarization_model(i_fit, *popt)
    
    eta_act = popt[1] * np.log(i_fit / popt[3])
    eta_ohm = popt[2] * i_fit
    eta_conc = (8.314 * 353.15 / (2 * 96485)) * np.log(1 - i_fit / popt[4])
    power = i_fit * v_fit
    
    ax1.scatter(i_data, v_data, color='blue', label='实验数据', s=50, zorder=5)
    ax1.plot(i_fit, v_fit, 'r-', linewidth=2.5, label='拟合曲线')
    ax1.set_xlabel('电流密度 (A/cm²)', fontsize=12)
    ax1.set_ylabel('电压 (V)', fontsize=12)
    ax1.set_title('PEMFC极化曲线拟合', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.3])
    
    ax2.plot(i_fit, eta_act, 'g--', linewidth=2, label='活化过电位')
    ax2.plot(i_fit, eta_ohm, 'orange', linewidth=2, label='欧姆损失')
    ax2.plot(i_fit, eta_conc, 'purple', linewidth=2, label='浓差过电位')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(i_fit, power, 'k-', linewidth=2.5, label='功率密度')
    ax2_twin.set_ylabel('功率密度 (W/cm²)', fontsize=12)
    
    ax2.set_xlabel('电流密度 (A/cm²)', fontsize=12)
    ax2.set_ylabel('过电位 (V)', fontsize=12)
    ax2.set_title('各损失项与功率密度', fontsize=14, fontweight='bold')
    
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pemfc_polarization_fit.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return i_fit, v_fit, eta_act, eta_ohm, eta_conc, power

def print_fit_results(popt, perr):
    """
    打印拟合参数结果
    """
    print("\n" + "="*60)
    print("PEMFC极化曲线拟合结果")
    print("="*60)
    print(f"开路电压 (E_ocv): {popt[0]:.4f} ± {perr[0]:.4f} V")
    print(f"Tafel系数 (a): {popt[1]:.4f} ± {perr[1]:.4f} V")
    print(f"欧姆电阻 (b): {popt[2]:.4f} ± {perr[2]:.4f} Ω·cm²")
    print(f"交换电流密度 (i0): {popt[3]:.2e} ± {perr[3]:.2e} A/cm²")
    print(f"极限电流密度 (iL): {popt[4]:.4f} ± {perr[4]:.4f} A/cm²")
    print("="*60)

def main():
    print("PEMFC极化曲线拟合程序")
    print("-"*50)
    
    print("\n1. 生成模拟实验数据...")
    i_data, v_data = generate_sample_data()
    
    print("\n2. 数据点:")
    for i, v in zip(i_data, v_data):
        print(f"   i = {i:.2f} A/cm², V = {v:.4f} V")
    
    print("\n3. 进行曲线拟合...")
    popt, perr = fit_polarization_curve(i_data, v_data)
    
    print_fit_results(popt, perr)
    
    print("\n4. 绘制结果并保存图片...")
    i_fit, v_fit, eta_act, eta_ohm, eta_conc, power = plot_results(i_data, v_data, popt)
    
    max_power_idx = np.argmax(power)
    print(f"\n5. 峰值功率密度: {power[max_power_idx]:.4f} W/cm²")
    print(f"   对应电流密度: {i_fit[max_power_idx]:.4f} A/cm²")
    print(f"   对应电压: {v_fit[max_power_idx]:.4f} V")
    
    print("\n拟合完成！图片已保存为 pemfc_polarization_fit.png")

if __name__ == "__main__":
    main()
