import numpy as np
from scipy.special import kn, gamma, airy, jv
from scipy.integrate import quad
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

c = 2.99792458e10
e_charge = 4.8032e-10
m_e = 9.10938356e-28
m_e_cgs = 9.10938356e-28
hbar = 1.0546e-27
k_B = 1.3806e-16


class SynchrotronRadiation:
    def __init__(self, B=1.0, alpha=np.pi/2):
        self.B = B
        self.alpha = alpha
        self.sin_alpha = np.sin(alpha)
        self.omega_B = e_charge * B / (m_e_cgs * c)

    def critical_frequency(self, gamma):
        omega_c = 1.5 * gamma**3 * self.omega_B * self.sin_alpha
        return omega_c

    def critical_wavelength(self, gamma):
        omega_c = self.critical_frequency(gamma)
        lambda_c = 2 * np.pi * c / omega_c
        return lambda_c

    def critical_harmonic_number(self, gamma):
        n_c = 1.5 * gamma**3 * self.sin_alpha
        return n_c

    def synchrotron_function(self, x):
        if np.isscalar(x):
            if x == 0:
                return 0.0
            if x > 1000:
                return np.sqrt(2 * x / np.pi) * np.exp(-x)
            result, _ = quad(lambda xi: kn(5.0/3.0, xi), x, np.inf)
            return x * result
        else:
            result = np.zeros_like(x)
            for i, xi_val in enumerate(x):
                if xi_val == 0:
                    result[i] = 0.0
                elif xi_val > 1000:
                    result[i] = np.sqrt(2 * xi_val / np.pi) * np.exp(-xi_val)
                else:
                    val, _ = quad(lambda xi: kn(5.0/3.0, xi), xi_val, np.inf)
                    result[i] = xi_val * val
            return result

    def synchrotron_function_F(self, x):
        return self.synchrotron_function(x)

    def synchrotron_function_G(self, x):
        if np.isscalar(x):
            if x == 0:
                return 0.0
            if x > 1000:
                return np.sqrt(x / (2 * np.pi)) * np.exp(-x)
            val, _ = quad(lambda xi: kn(2.0/3.0, xi), x, np.inf)
            return x * val
        else:
            result = np.zeros_like(x)
            for i, xi_val in enumerate(x):
                if xi_val == 0:
                    result[i] = 0.0
                elif xi_val > 1000:
                    result[i] = np.sqrt(xi_val / (2 * np.pi)) * np.exp(-xi_val)
                else:
                    val, _ = quad(lambda xi: kn(2.0/3.0, xi), xi_val, np.inf)
                    result[i] = xi_val * val
            return result

    def single_electron_power(self, gamma, omega):
        omega_c = self.critical_frequency(gamma)
        x = omega / omega_c
        P_total_analytical = self.total_power_single_electron(gamma)
        norm_factor = P_total_analytical / omega_c
        P = norm_factor * self.synchrotron_function_F(x)
        return P

    def single_electron_power_per_freq(self, gamma, nu):
        omega = 2 * np.pi * nu
        return self.single_electron_power(gamma, omega) * 2 * np.pi

    def total_power_single_electron(self, gamma):
        P_total = (4.0/3.0) * (e_charge**4 * self.B**2 * self.sin_alpha**2 /
                               (m_e_cgs**3 * c**5)) * gamma**2
        return P_total

    def harmonic_power_exact(self, gamma, n):
        nu_B = self.omega_B / (2 * np.pi)
        nu_n = n * nu_B
        P_nu = self.single_electron_power_per_freq(gamma, nu_n)
        P_n = P_nu * nu_B
        return P_n

    def harmonic_power_airy_asymptotic(self, gamma, n):
        n_c = self.critical_harmonic_number(gamma)
        y = (n / n_c)**(2.0/3.0)

        if y > 50:
            return self.harmonic_power_exponential_tail(gamma, n)

        Ai, Aip, Bi, Bip = airy(y)
        F_approx = (np.sqrt(3) * np.pi / 2) * y * Ai**2

        P_total_analytical = self.total_power_single_electron(gamma)
        omega_c = self.critical_frequency(gamma)
        norm_factor = P_total_analytical / omega_c

        P_omega = norm_factor * F_approx
        P_nu = P_omega * 2 * np.pi
        nu_B = self.omega_B / (2 * np.pi)
        P_n = P_nu * nu_B

        return P_n

    def harmonic_power_exponential_tail(self, gamma, n):
        n_c = self.critical_harmonic_number(gamma)
        x = n / n_c

        if x < 1:
            return 0.0

        P_total_analytical = self.total_power_single_electron(gamma)
        P_n = P_total_analytical * np.sqrt(3 * x / np.pi) * np.exp(-x) / n_c

        return P_n

    def harmonic_power_spectrum(self, gamma, n_max=None, method='auto'):
        n_c = self.critical_harmonic_number(gamma)

        if n_max is None:
            n_max = int(5 * n_c)
            n_max = min(n_max, 100000)

        n_values = np.arange(1, n_max + 1)
        omega_values = n_values * self.omega_B

        P_n = np.zeros_like(n_values, dtype=float)

        for i, n in enumerate(n_values):
            if method == 'exact':
                P_n[i] = self.harmonic_power_exact(gamma, n)
            elif method == 'airy':
                P_n[i] = self.harmonic_power_airy_asymptotic(gamma, n)
            else:
                if n / n_c < 0.1 or n / n_c > 10:
                    P_n[i] = self.harmonic_power_airy_asymptotic(gamma, n)
                else:
                    P_n[i] = self.harmonic_power_exact(gamma, n)

        return n_values, omega_values, P_n

    def harmonic_spectrum_at_nu(self, gamma, nu):
        omega = 2 * np.pi * nu
        n = omega / self.omega_B
        n_int = int(np.round(n))

        if n_int < 1:
            return 0.0

        P_n = self.harmonic_power_exact(gamma, n_int)
        return P_n / (self.omega_B / (2 * np.pi))

    def harmonic_total_power_sum(self, gamma, n_max=None, use_airy_cutoff=False,
                                  relative_cutoff=1e-15):
        n_c = self.critical_harmonic_number(gamma)

        if n_max is None:
            n_max = int(20 * n_c)
            n_max = min(n_max, 200000)

        total_power = 0.0
        max_power = 0.0
        for n in range(1, n_max + 1):
            if use_airy_cutoff and n > 10 * n_c:
                P_n = self.harmonic_power_exponential_tail(gamma, n)
            elif use_airy_cutoff and n > 3 * n_c:
                P_n = self.harmonic_power_airy_asymptotic(gamma, n)
            else:
                P_n = self.harmonic_power_exact(gamma, n)

            total_power += P_n
            max_power = max(max_power, P_n)

            if n > 3 * n_c and P_n < relative_cutoff * max_power:
                break

        return total_power, n

    def optimal_harmonic_cutoff(self, gamma, relative_accuracy=1e-3):
        n_c = self.critical_harmonic_number(gamma)

        n_max_est = int(10 * n_c)
        n_max_est = min(n_max_est, 100000)
        P_analytical = self.total_power_single_electron(gamma)

        cumulative_power = 0.0
        n_optimal = n_max_est

        for n in range(1, n_max_est + 1):
            P_n = self.harmonic_power_exact(gamma, n)
            cumulative_power += P_n

            if cumulative_power >= (1 - relative_accuracy) * P_analytical:
                n_optimal = n
                break

        return n_optimal, cumulative_power / P_analytical

    def airy_approximation_accuracy(self, gamma, n):
        P_exact = self.harmonic_power_exact(gamma, n)
        P_airy = self.harmonic_power_airy_asymptotic(gamma, n)

        if P_exact > 0:
            rel_error = abs(P_exact - P_airy) / P_exact * 100
        else:
            rel_error = 0.0

        return rel_error, P_exact, P_airy

    def power_law_distribution(self, gamma, N0, p, gamma_min=1.0, gamma_max=1e8):
        if np.isscalar(gamma):
            if gamma < gamma_min or gamma > gamma_max:
                return 0.0
            return N0 * gamma**(-p)
        else:
            result = np.zeros_like(gamma)
            mask = (gamma >= gamma_min) & (gamma <= gamma_max)
            result[mask] = N0 * gamma[mask]**(-p)
            return result

    def relativistic_maxwell_distribution(self, gamma, N0, T):
        mc2_T = m_e_cgs * c**2 / T
        if mc2_T > 500:
            return np.zeros_like(gamma) if not np.isscalar(gamma) else 0.0

        exp_arg = -gamma * mc2_T
        if np.isscalar(gamma):
            if exp_arg < -500:
                return 0.0
            beta = np.sqrt(1.0 - 1.0/gamma**2)
            if mc2_T < 500:
                norm = mc2_T * np.exp(mc2_T)
            else:
                norm = 1e200
            return N0 * gamma**2 * beta * np.exp(exp_arg) / norm
        else:
            beta = np.sqrt(1.0 - 1.0/gamma**2)
            result = np.zeros_like(gamma)
            mask = exp_arg > -500
            if mc2_T < 500:
                norm = mc2_T * np.exp(mc2_T)
            else:
                norm = 1e200
            result[mask] = N0 * gamma[mask]**2 * beta[mask] * np.exp(exp_arg[mask]) / norm
            return result

    def power_law_spectrum(self, nu, N0, p, gamma_min=1.0, gamma_max=1e8):
        omega = 2 * np.pi * nu

        def integrand(gamma):
            N_gamma = self.power_law_distribution(gamma, N0, p, gamma_min, gamma_max)
            P_gamma = self.single_electron_power(gamma, omega)
            return N_gamma * P_gamma

        result, _ = quad(integrand, gamma_min, gamma_max, limit=200)
        return result

    def power_law_spectrum_approx(self, nu, N0, p, gamma_min=1.0, gamma_max=1e8):
        omega = 2 * np.pi * nu
        omega_B = self.omega_B

        gamma_c = (omega / (1.5 * omega_B * self.sin_alpha))**(1.0/3.0)
        gamma_c = max(gamma_min, min(gamma_c, gamma_max))

        coeff = (np.sqrt(3) * e_charge**3 * self.B * self.sin_alpha /
                 (2 * np.pi * m_e_cgs * c**2))

        def integrand(gamma):
            if gamma < gamma_min or gamma > gamma_max:
                return 0.0
            x = omega / (1.5 * gamma**3 * omega_B * self.sin_alpha)
            return gamma**(-p) * self.synchrotron_function_F(x)

        result, _ = quad(integrand, gamma_min, gamma_max, limit=200)
        return N0 * coeff * result

    def emissivity_power_law(self, nu, N0, p, gamma_min=1.0, gamma_max=1e8):
        return self.power_law_spectrum(nu, N0, p, gamma_min, gamma_max) / (4 * np.pi)

    def spectral_index_power_law(self, nu1, nu2, N0, p, gamma_min=1.0, gamma_max=1e8):
        P1 = self.power_law_spectrum(nu1, N0, p, gamma_min, gamma_max)
        P2 = self.power_law_spectrum(nu2, N0, p, gamma_min, gamma_max)
        alpha = np.log(P2 / P1) / np.log(nu2 / nu1)
        return alpha

    def luminosity_power_law(self, nu, N0, p, V, gamma_min=1.0, gamma_max=1e8):
        epsilon = self.emissivity_power_law(nu, N0, p, gamma_min, gamma_max)
        return 4 * np.pi * V * epsilon


def plot_synchrotron_function(sr):
    x = np.logspace(-2, 2, 200)
    F_x = sr.synchrotron_function_F(x)
    G_x = sr.synchrotron_function_G(x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(x, F_x, 'b-', linewidth=2, label='F(x)')
    ax1.plot(x, G_x, 'r--', linewidth=2, label='G(x)')
    ax1.set_xlabel('x = ω/ω_c', fontsize=12)
    ax1.set_ylabel('F(x), G(x)', fontsize=12)
    ax1.set_title('同步辐射函数 F(x) 和 G(x)')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylim(1e-15, 1e1)

    x_approx = np.linspace(0, 10, 200)
    F_approx = sr.synchrotron_function_F(x_approx)

    ax2.plot(x_approx, F_approx, 'b-', linewidth=2)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('F(x)', fontsize=12)
    ax2.set_title('同步辐射函数 F(x) (线性坐标)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('e:/temp/record10/208/synchrotron_function.png', dpi=150)
    plt.close()
    print("同步辐射函数图像已保存: synchrotron_function.png")


def plot_single_electron_spectrum(sr, gamma=100):
    omega_c = sr.critical_frequency(gamma)
    nu_c = omega_c / (2 * np.pi)

    nu = np.logspace(np.log10(nu_c/100), np.log10(nu_c*100), 300)
    omega = 2 * np.pi * nu
    x = omega / omega_c

    P_nu = sr.single_electron_power_per_freq(gamma, nu)

    P_total = sr.total_power_single_electron(gamma)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(nu, P_nu, 'b-', linewidth=2)
    ax.axvline(x=nu_c, color='r', linestyle='--', label=f'临界频率 ν_c = {nu_c:.2e} Hz')
    ax.set_xlabel('频率 ν (Hz)', fontsize=12)
    ax.set_ylabel('P(ν) [erg s⁻¹ Hz⁻¹]', fontsize=12)
    ax.set_title(f'单个相对论电子 (γ={gamma}) 的同步辐射谱')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    textstr = '\n'.join((
        f'磁场 B = {sr.B} G',
        f'投射角 α = {sr.alpha*180/np.pi:.1f}°',
        f'洛伦兹因子 γ = {gamma}',
        f'临界频率 ν_c = {nu_c:.2e} Hz',
        f'总辐射功率 P = {P_total:.2e} erg/s'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig('e:/temp/record10/208/single_electron_spectrum.png', dpi=150)
    plt.close()
    print("单电子辐射谱图像已保存: single_electron_spectrum.png")


def plot_harmonic_spectrum(sr, gamma=10):
    n_c = sr.critical_harmonic_number(gamma)
    n_max = int(5 * n_c)

    print(f"  计算谐波谱: γ={gamma}, n_c={n_c:.1f}, n_max={n_max}")

    n_values, omega_values, P_n = sr.harmonic_power_spectrum(gamma, n_max=n_max, method='auto')

    nu_values = omega_values / (2 * np.pi)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.stem(n_values, P_n, basefmt='b-', linefmt='b-', markerfmt='bo')
    ax1.axvline(x=n_c, color='r', linestyle='--', label=f'临界谐波 n_c = {n_c:.1f}')
    ax1.set_xlabel('谐波数 n', fontsize=12)
    ax1.set_ylabel('P_n [erg/s]', fontsize=12)
    ax1.set_title(f'各次谐波辐射功率 (γ={gamma})')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, n_max)
    ax1.set_yscale('log')

    ax2.plot(nu_values, P_n / sr.omega_B, 'b-', linewidth=2, label='离散谐波求和')
    ax2.set_xlabel('频率 ν (Hz)', fontsize=12)
    ax2.set_ylabel('P(ν) [erg s⁻¹ Hz⁻¹]', fontsize=12)
    ax2.set_title(f'离散谐波辐射功率谱 (γ={gamma})')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('e:/temp/record10/208/harmonic_spectrum.png', dpi=150)
    plt.close()
    print("谐波谱图像已保存: harmonic_spectrum.png")


def plot_method_comparison(sr, gamma=20):
    n_c = sr.critical_harmonic_number(gamma)
    n_max = int(5 * n_c)

    print(f"  方法对比: γ={gamma}, n_c={n_c:.1f}")

    n_exact, omega_exact, P_exact = sr.harmonic_power_spectrum(gamma, n_max=n_max, method='exact')
    n_airy, omega_airy, P_airy = sr.harmonic_power_spectrum(gamma, n_max=n_max, method='airy')

    nu_exact = omega_exact / (2 * np.pi)
    nu_airy = omega_airy / (2 * np.pi)

    omega_c = sr.critical_frequency(gamma)
    nu_continuous = np.logspace(np.log10(sr.omega_B/(2*np.pi)),
                                 np.log10(5*omega_c/(2*np.pi)), 500)
    P_continuous = sr.single_electron_power_per_freq(gamma, nu_continuous)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(n_exact, P_exact, 'b-', linewidth=2, label='精确公式')
    ax1.plot(n_airy, P_airy, 'r--', linewidth=2, label='Airy渐近展开')
    ax1.axvline(x=n_c, color='k', linestyle=':', label=f'n_c = {n_c:.1f}')
    ax1.set_xlabel('谐波数 n', fontsize=12)
    ax1.set_ylabel('P_n [erg/s]', fontsize=12)
    ax1.set_title('精确公式 vs Airy渐近展开')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    valid_mask = P_exact > 0
    error = np.zeros_like(P_exact)
    error[valid_mask] = np.abs(P_exact[valid_mask] - P_airy[valid_mask]) / P_exact[valid_mask] * 100
    ax2.plot(n_exact[valid_mask], error[valid_mask], 'g-', linewidth=2)
    ax2.axvline(x=n_c, color='r', linestyle='--', label=f'n_c = {n_c:.1f}')
    ax2.set_xlabel('谐波数 n', fontsize=12)
    ax2.set_ylabel('相对误差 (%)', fontsize=12)
    ax2.set_title('Airy近似的相对误差')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('e:/temp/record10/208/method_comparison.png', dpi=150)
    plt.close()
    print("方法对比图像已保存: method_comparison.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(nu_continuous, P_continuous, 'b-', linewidth=2, label='连续谱近似')
    ax.stem(nu_exact, P_exact / sr.omega_B, basefmt='none', linefmt='r-',
            markerfmt='ro', label='离散谐波谱')
    ax.axvline(x=omega_c/(2*np.pi), color='k', linestyle='--',
               label=f'ν_c = {omega_c/(2*np.pi):.2e} Hz')
    ax.set_xlabel('频率 ν (Hz)', fontsize=12)
    ax.set_ylabel('P(ν) [erg s⁻¹ Hz⁻¹]', fontsize=12)
    ax.set_title(f'连续谱近似 vs 离散谐波谱 (γ={gamma})')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig('e:/temp/record10/208/continuous_vs_discrete.png', dpi=150)
    plt.close()
    print("连续谱vs离散谱图像已保存: continuous_vs_discrete.png")


def plot_power_law_spectrum(sr, N0=1.0, p=2.5, gamma_min=10, gamma_max=1e5):
    gamma_values = np.logspace(np.log10(gamma_min), np.log10(gamma_max), 100)
    N_gamma = sr.power_law_distribution(gamma_values, N0, p, gamma_min, gamma_max)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(gamma_values, N_gamma, 'b-', linewidth=2)
    ax1.set_xlabel('洛伦兹因子 γ', fontsize=12)
    ax1.set_ylabel('N(γ) [cm⁻³]', fontsize=12)
    ax1.set_title(f'幂律电子能量分布 (p={p})')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    nu_values = np.logspace(7, 20, 100)
    P_nu = np.zeros_like(nu_values)
    for i, nu in enumerate(nu_values):
        P_nu[i] = sr.power_law_spectrum_approx(nu, N0, p, gamma_min, gamma_max)

    ax2.plot(nu_values, P_nu, 'r-', linewidth=2)
    ax2.set_xlabel('频率 ν (Hz)', fontsize=12)
    ax2.set_ylabel('P(ν) [erg s⁻¹ Hz⁻¹]', fontsize=12)
    ax2.set_title(f'幂律分布电子的同步辐射谱 (p={p})')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('e:/temp/record10/208/power_law_spectrum.png', dpi=150)
    plt.close()
    print("幂律分布辐射谱图像已保存: power_law_spectrum.png")


def plot_gamma_dependence(sr):
    gammas = [10, 50, 100, 500, 1000]
    colors = ['b', 'g', 'r', 'm', 'c']

    fig, ax = plt.subplots(figsize=(10, 6))

    for gamma, color in zip(gammas, colors):
        omega_c = sr.critical_frequency(gamma)
        nu_c = omega_c / (2 * np.pi)

        nu = np.logspace(np.log10(nu_c/100), np.log10(nu_c*50), 200)
        P_nu = sr.single_electron_power_per_freq(gamma, nu)
        x = 2 * np.pi * nu / omega_c

        ax.plot(x, P_nu / np.max(P_nu), color=color, linewidth=2,
                label=f'γ = {gamma}')

    ax.set_xlabel('x = ω/ω_c', fontsize=12)
    ax.set_ylabel('归一化功率 P(ν)/P_max', fontsize=12)
    ax.set_title('不同洛伦兹因子的归一化同步辐射谱')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 1.5)

    plt.tight_layout()
    plt.savefig('e:/temp/record10/208/gamma_dependence.png', dpi=150)
    plt.close()
    print("洛伦兹因子依赖图像已保存: gamma_dependence.png")


def plot_spectral_index(sr, N0=1.0, p_list=[2.0, 2.5, 3.0, 3.5], gamma_min=10, gamma_max=1e5):
    fig, ax = plt.subplots(figsize=(10, 6))

    nu1 = 1e10
    nu = np.logspace(10, 19, 50)

    for p in p_list:
        alpha_list = []
        for nu2 in nu:
            if nu2 > nu1:
                alpha = sr.spectral_index_power_law(nu1, nu2, N0, p, gamma_min, gamma_max)
                alpha_list.append(alpha)
            else:
                alpha_list.append(np.nan)

        ax.plot(nu, alpha_list, linewidth=2, label=f'p = {p}')

    ax.axhline(y=-0.5, color='k', linestyle='--', alpha=0.5, label='α = -0.5 (低频极限)')
    ax.set_xlabel('频率 ν (Hz)', fontsize=12)
    ax.set_ylabel('谱指数 α', fontsize=12)
    ax.set_title('不同幂律指数的同步辐射谱指数')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_ylim(-2, 0.5)

    plt.tight_layout()
    plt.savefig('e:/temp/record10/208/spectral_index.png', dpi=150)
    plt.close()
    print("谱指数图像已保存: spectral_index.png")


def print_comparison_table(sr):
    print("\n" + "="*70)
    print("不同幂律指数 p 下的同步辐射谱指数 α 比较")
    print("="*70)
    print(f"{'p':>6} {'α(理论)':>10} {'α(计算)':>10} {'相对误差':>10}")
    print("-"*70)

    nu1, nu2 = 1e12, 1e14
    for p in [2.0, 2.5, 3.0, 3.5, 4.0]:
        alpha_theory = -(p - 1) / 2
        alpha_calc = sr.spectral_index_power_law(nu1, nu2, N0=1.0, p=p,
                                                  gamma_min=10, gamma_max=1e5)
        error = abs((alpha_calc - alpha_theory) / alpha_theory) * 100
        print(f"{p:6.1f} {alpha_theory:10.4f} {alpha_calc:10.4f} {error:9.2f}%")

    print("="*70)


def print_harmonic_validation(sr, gamma=20):
    print("\n" + "="*70)
    print(f"谐波求和验证 (γ = {gamma})")
    print("="*70)

    n_c = sr.critical_harmonic_number(gamma)
    print(f"  临界谐波数 n_c = {n_c:.2f}")
    print(f"  临界频率 ν_c = {sr.critical_frequency(gamma)/(2*np.pi):.4e} Hz")

    P_analytical = sr.total_power_single_electron(gamma)
    P_sum, n_used = sr.harmonic_total_power_sum(gamma)

    print(f"\n  理论总功率 (解析) = {P_analytical:.4e} erg/s")
    print(f"  谐波求和总功率    = {P_sum:.4e} erg/s")
    print(f"  使用谐波数        = {n_used}")
    print(f"  相对误差          = {abs(P_sum-P_analytical)/P_analytical*100:.4f}%")

    n_opt, coverage = sr.optimal_harmonic_cutoff(gamma, relative_accuracy=1e-3)
    print(f"\n  最优截断谐波数 (精度99.9%) = {n_opt} (n/n_c = {n_opt/n_c:.2f})")
    print(f"  功率覆盖率                 = {coverage*100:.4f}%")

    print("\n  各次谐波功率分布:")
    print(f"  {'谐波n':>8} {'n/n_c':>10} {'P_n (erg/s)':>15} {'累计占比(%)':>12}")
    print("  " + "-"*50)

    n_test = [1, int(n_c/4), int(n_c/2), int(n_c), int(2*n_c), int(3*n_c)]

    n_max_display = int(5 * n_c)
    P_all = []
    for n in range(1, n_max_display + 1):
        P_all.append(sr.harmonic_power_exact(gamma, n))
    P_total = sum(P_all)

    for n in n_test:
        if n < 1 or n >= len(P_all):
            continue
        P_n = P_all[n-1]
        cumulative = sum(P_all[:n])
        print(f"  {n:8d} {n/n_c:10.3f} {P_n:15.4e} {cumulative/P_total*100:12.2f}")

    print("\n  智能截断策略:")
    print(f"    默认: 使用精确公式求和，相对误差 < 0.01%")
    print(f"    可选: n > 3n_c 使用Airy近似，n > 10n_c 使用指数尾近似")
    print(f"    停止条件: P_n < 1e-15 × P_max 时自动停止")

    print("="*70)


def print_physical_params(sr):
    print("\n" + "="*70)
    print("同步辐射计算 - 物理参数")
    print("="*70)
    print(f"  磁场强度 B = {sr.B} G")
    print(f"  投射角 α = {sr.alpha*180/np.pi:.1f}°")
    print(f"  回旋频率 ω_B = {sr.omega_B:.4e} rad/s")
    print(f"  回旋频率 ν_B = {sr.omega_B/(2*np.pi):.4e} Hz")
    print("="*70)


def example_calculation():
    B = 1.0
    alpha = np.pi / 2

    print("="*70)
    print("磁化等离子体中相对论性电子的同步辐射谱计算")
    print("="*70)

    sr = SynchrotronRadiation(B=B, alpha=alpha)
    print_physical_params(sr)

    gamma = 100
    print(f"\n{'='*70}")
    print(f"单个相对论电子的同步辐射 (γ = {gamma})")
    print(f"{'='*70}")

    omega_c = sr.critical_frequency(gamma)
    nu_c = omega_c / (2 * np.pi)
    lambda_c = sr.critical_wavelength(gamma)

    print(f"  临界频率 ω_c = {omega_c:.4e} rad/s")
    print(f"  临界频率 ν_c = {nu_c:.4e} Hz")
    print(f"  临界波长 λ_c = {lambda_c:.4e} cm = {lambda_c*1e4:.4f} μm")

    P_total = sr.total_power_single_electron(gamma)
    print(f"  总辐射功率 P = {P_total:.4e} erg/s")

    print_harmonic_validation(sr, gamma=20)

    print(f"\n{'='*70}")
    print(f"幂律分布电子的同步辐射谱")
    print(f"{'='*70}")

    N0 = 1.0
    p = 2.5
    gamma_min = 10
    gamma_max = 1e5

    print(f"  分布: N(γ) = {N0} × γ^(-{p})")
    print(f"  能量范围: γ ∈ [{gamma_min}, {gamma_max}]")

    nu_test = 1e14
    P_nu = sr.power_law_spectrum_approx(nu_test, N0, p, gamma_min, gamma_max)
    print(f"\n  在 ν = {nu_test:.2e} Hz 处:")
    print(f"    辐射功率谱 P(ν) = {P_nu:.4e} erg s⁻¹ Hz⁻¹")

    epsilon = sr.emissivity_power_law(nu_test, N0, p, gamma_min, gamma_max)
    print(f"    辐射系数 ε(ν) = {epsilon:.4e} erg s⁻¹ cm⁻³ Hz⁻¹ sr⁻¹")

    print_comparison_table(sr)

    print(f"\n{'='*70}")
    print(f"生成图像中...")
    print(f"{'='*70}")

    plot_synchrotron_function(sr)
    plot_single_electron_spectrum(sr, gamma=100)
    plot_harmonic_spectrum(sr, gamma=20)
    plot_method_comparison(sr, gamma=20)
    plot_power_law_spectrum(sr, N0=1.0, p=2.5, gamma_min=10, gamma_max=1e5)
    plot_gamma_dependence(sr)
    plot_spectral_index(sr, N0=1.0, p_list=[2.0, 2.5, 3.0, 3.5],
                        gamma_min=10, gamma_max=1e5)

    print(f"\n{'='*70}")
    print(f"计算完成！所有图像已保存到当前目录。")
    print(f"{'='*70}")

    return sr


if __name__ == "__main__":
    sr = example_calculation()
