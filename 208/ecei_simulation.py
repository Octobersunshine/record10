import numpy as np
from scipy.special import kn, gamma
from scipy.integrate import quad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

c = 2.99792458e10
e_charge = 4.8032e-10
m_e_cgs = 9.10938356e-28
k_B = 1.3806e-16
m_kg = 9.10938356e-31
e_si = 1.602176634e-19
epsilon_0 = 8.8541878128e-12
hbar = 1.0545718e-27


class PlasmaDispersion:
    def __init__(self, n_e, B, omega, theta=0):
        self.n_e = n_e
        self.B = B
        self.omega = omega
        self.theta = theta

        self.omega_pe2 = 4 * np.pi * e_charge**2 * n_e / m_e_cgs
        self.omega_p = np.sqrt(self.omega_pe2)
        self.omega_ce = e_charge * B / (m_e_cgs * c)

        self.X = self.omega_pe2 / omega**2
        self.Y = self.omega_ce / omega
        self.Y_l = self.Y * np.cos(theta)
        self.Y_t = self.Y * np.sin(theta)

    def refractive_index_O_mode(self):
        X = self.X
        if X >= 1:
            return 0.0
        return np.sqrt(1 - X)

    def refractive_index_X_mode(self):
        X = self.X
        Y = self.Y
        Y_l = self.Y_l
        Y_t = self.Y_t

        denom = 2 * (1 - X)
        disc = Y_t**4 + 4 * (1 - X)**2 * Y_l**2

        if disc < 0:
            return np.nan

        N2 = 1 - X * (1 - X) / (1 - X - Y_t**2/2 + np.sqrt(disc)/2)

        if N2 < 0:
            return 0.0

        return np.sqrt(N2)

    def group_velocity_O_mode(self):
        N = self.refractive_index_O_mode()
        if N == 0:
            return 0.0
        return c / N

    def group_velocity_X_mode(self):
        N = self.refractive_index_X_mode()
        if N == 0 or np.isnan(N):
            return 0.0
        return c / N

    def cutoff_frequency_O_mode(self):
        return self.omega_p

    def cutoff_frequency_X_mode_high(self):
        Y = self.omega_ce
        return (Y + np.sqrt(Y**2 + 4*self.omega_pe2)) / 2

    def cutoff_frequency_X_mode_low(self):
        Y = self.omega_ce
        return (-Y + np.sqrt(Y**2 + 4*self.omega_pe2)) / 2


class RadiationTransfer:
    def __init__(self, sr_calculator):
        self.sr = sr_calculator

    def emissivity(self, nu, n_e, T_e, gamma_min=1.0, gamma_max=1e8):
        gamma_opt = T_e / (m_e_cgs * c**2)
        if gamma_opt < 1.0:
            gamma_opt = 1.0

        N_opt = self.sr.relativistic_maxwell_distribution(gamma_opt, n_e, T_e)
        P_opt = self.sr.single_electron_power_per_freq(gamma_opt, nu)

        width = max(0.5 * gamma_opt, 1.0)
        gamma_int = np.logspace(np.log10(max(gamma_opt - width, 1.0)),
                                np.log10(gamma_opt + width), 20)

        integrand = np.array([self.sr.relativistic_maxwell_distribution(g, n_e, T_e) *
                              self.sr.single_electron_power_per_freq(g, nu)
                              for g in gamma_int])

        integral = np.trapezoid(integrand, gamma_int)
        return integral / (4 * np.pi)

    def absorption_coefficient(self, nu, n_e, T_e, gamma_min=1.0, gamma_max=1e8):
        gamma_opt = T_e / (m_e_cgs * c**2)
        if gamma_opt < 1.0:
            gamma_opt = 1.0

        width = max(0.5 * gamma_opt, 1.0)
        gamma_int = np.logspace(np.log10(max(gamma_opt - width, 1.0)),
                                np.log10(gamma_opt + width), 20)

        integrand = np.array([self.sr.relativistic_maxwell_distribution(g, n_e, T_e) *
                              self.sr.single_electron_power_per_freq(g, nu) / (g * m_e_cgs * c**2)
                              for g in gamma_int])

        integral = np.trapezoid(integrand, gamma_int)
        prefactor = e_charge**2 * np.pi * nu**2 / (m_e_cgs * c**3)
        return prefactor * integral

    def source_function(self, nu, T_e):
        x = hbar * 2 * np.pi * nu / (k_B * T_e)
        if np.isscalar(x):
            if x < 1e-10:
                return k_B * T_e * nu**2 / c**2
            elif x > 100:
                return 0.0
            else:
                return (2 * hbar * nu**3 / c**2) / (np.exp(x) - 1)
        else:
            S = np.zeros_like(x)
            mask1 = x < 1e-10
            mask2 = x > 100
            mask3 = ~mask1 & ~mask2
            S[mask1] = k_B * T_e * nu**2 / c**2
            S[mask3] = (2 * hbar * nu**3 / c**2) / (np.exp(x[mask3]) - 1)
            return S

    def brightness_temperature(self, nu, tau, T_e):
        if np.isscalar(tau):
            if tau < 1e-10:
                return 0.0
            elif tau > 100:
                return T_e
            else:
                return T_e * (1 - np.exp(-tau))
        else:
            T_b = np.zeros_like(tau)
            mask1 = tau < 1e-10
            mask2 = tau > 100
            mask3 = ~mask1 & ~mask2
            T_b[mask2] = T_e
            T_b[mask3] = T_e * (1 - np.exp(-tau[mask3]))
            return T_b

    def transfer_equation_homogeneous(self, nu, n_e, T_e, path_length,
                                       gamma_min=1.0, gamma_max=1e8):
        epsilon = self.emissivity(nu, n_e, T_e, gamma_min, gamma_max)
        alpha = self.absorption_coefficient(nu, n_e, T_e, gamma_min, gamma_max)

        if alpha < 1e-30:
            return epsilon * path_length, 0.0

        tau = alpha * path_length
        S = self.source_function(nu, T_e)

        I_out = (1 - np.exp(-tau)) * S

        return I_out, tau

    def transfer_equation_inhomogeneous(self, nu, n_e_profile, T_e_profile,
                                         B_profile, z_values, gamma_min=1.0,
                                         gamma_max=1e8):
        n_points = len(z_values)
        tau_array = np.zeros(n_points)
        I_array = np.zeros(n_points)

        for i in range(1, n_points):
            dz = z_values[i] - z_values[i-1]

            n_e_avg = (n_e_profile[i] + n_e_profile[i-1]) / 2
            T_e_avg = (T_e_profile[i] + T_e_profile[i-1]) / 2
            B_avg = (B_profile[i] + B_profile[i-1]) / 2

            self.sr.B = B_avg

            alpha = self.absorption_coefficient(nu, n_e_avg, T_e_avg,
                                                gamma_min, gamma_max)
            epsilon = self.emissivity(nu, n_e_avg, T_e_avg, gamma_min, gamma_max)

            S = self.source_function(nu, T_e_avg)
            d_tau = alpha * dz

            tau_array[i] = tau_array[i-1] + d_tau

            if d_tau < 1e-10:
                I_array[i] = I_array[i-1] + epsilon * dz
            else:
                I_array[i] = I_array[i-1] * np.exp(-d_tau) + S * (1 - np.exp(-d_tau))

        return I_array, tau_array


class TokamakProfile:
    def __init__(self, R0=6.2, a=2.0, B0=5.3):
        self.R0 = R0
        self.a = a
        self.B0 = B0
        self.aspect_ratio = R0 / a

    def magnetic_field(self, R, Z=0):
        B_t = self.B0 * self.R0 / R
        return B_t

    def magnetic_field_normalized(self, rho):
        R = self.R0 + rho * self.a
        return self.magnetic_field(R)

    def electron_density_parabolic(self, rho, n_e0=1e20):
        return n_e0 * (1 - rho**2)

    def electron_temperature_parabolic(self, rho, T_e0=2.0):
        T0_keV = T_e0
        T0_erg = T0_keV * 1.602e-9
        return T0_erg * (1 - rho**2)

    def electron_temperature_H_mode(self, rho, T_e0=5.0, T_edge=0.5):
        T0_erg = T_e0 * 1.602e-9
        T_edge_erg = T_edge * 1.602e-9

        if rho < 0.8:
            return T0_erg * (1 - rho**2)
        else:
            return T_edge_erg + (T0_erg * 0.36 - T_edge_erg) * (1 - (rho - 0.8) / 0.2)**2

    def safety_factor(self, rho, q0=1.0, q95=4.0):
        return q0 + (q95 - q0) * rho**2

    def cyclotron_freq(self, R, harmonic=1):
        B_tesla = self.magnetic_field(R)
        B_gauss = B_tesla * 1e4
        omega_ce = e_charge * B_gauss / (m_e_cgs * c)
        return harmonic * omega_ce / (2 * np.pi)

    def cyclotron_freq_from_rho(self, rho, harmonic=1):
        R = self.R0 + rho * self.a
        return self.cyclotron_freq(R, harmonic)

    def major_radius_from_freq(self, nu, harmonic=1):
        omega = 2 * np.pi * nu
        B_gauss = m_e_cgs * c * omega / (harmonic * e_charge)
        B_tesla = B_gauss / 1e4
        R = self.B0 * self.R0 / B_tesla
        return R


class ECEIDiagnostic:
    def __init__(self, tokamak, rt):
        self.tokamak = tokamak
        self.rt = rt

    def line_of_sight(self, R_start, R_end, n_points=100):
        R_values = np.linspace(R_start, R_end, n_points)
        rho_values = (R_values - self.tokamak.R0) / self.tokamak.a
        return R_values, rho_values

    def measure_ecei(self, nu, n_e0, T_e0, harmonic=2,
                     R_start=None, R_end=None, n_points=100):
        if R_start is None:
            R_start = self.tokamak.R0 - self.tokamak.a
        if R_end is None:
            R_end = self.tokamak.R0 + self.tokamak.a

        R_values, rho_values = self.line_of_sight(R_start, R_end, n_points)

        n_e_profile = self.tokamak.electron_density_parabolic(rho_values, n_e0)
        T_e_profile = self.tokamak.electron_temperature_parabolic(rho_values, T_e0)
        B_profile_tesla = self.tokamak.magnetic_field_normalized(rho_values)
        B_profile_gauss = B_profile_tesla * 1e4

        z_values = np.linspace(0, 1, n_points)

        self.rt.sr.B = B_profile_gauss[0]

        I_array, tau_array = self.rt.transfer_equation_inhomogeneous(
            nu, n_e_profile, T_e_profile, B_profile_gauss, z_values)

        T_b_array = self.rt.brightness_temperature(nu, tau_array, T_e_profile[-1])

        return R_values, rho_values, I_array, T_b_array, tau_array

    def ecei_image(self, frequencies, n_e0, T_e0, harmonic=2,
                   R_start=None, R_end=None, n_radial=50):
        n_freq = len(frequencies)

        R_mid = self.tokamak.R0
        if R_start is None:
            R_start = R_mid - self.tokamak.a * 0.8
        if R_end is None:
            R_end = R_mid + self.tokamak.a * 0.8

        R_values, rho_values = self.line_of_sight(R_start, R_end, n_radial)

        image_data = np.zeros((n_freq, n_radial))

        for i, nu in enumerate(frequencies):
            _, _, I_array, T_b_array, _ = self.measure_ecei(
                nu, n_e0, T_e0, harmonic, R_start, R_end, n_radial)
            image_data[i, :] = T_b_array

        return frequencies, R_values, rho_values, image_data


def plot_plasma_dispersion(tokamak):
    n_e0 = 1e20
    B0 = tokamak.B0

    freq_range = np.logspace(10, 12, 500)

    N_O = np.zeros_like(freq_range)
    N_X = np.zeros_like(freq_range)

    for i, nu in enumerate(freq_range):
        omega = 2 * np.pi * nu
        pd = PlasmaDispersion(n_e0, B0, omega, theta=np.pi/2)
        N_O[i] = pd.refractive_index_O_mode()
        N_X[i] = pd.refractive_index_X_mode()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(freq_range, N_O, 'b-', linewidth=2, label='O模')
    ax1.axvline(x=pd.cutoff_frequency_O_mode()/(2*np.pi), color='r',
                linestyle='--', label='O模截止频率')
    ax1.set_xlabel('频率 ν (Hz)', fontsize=12)
    ax1.set_ylabel('折射率 N', fontsize=12)
    ax1.set_title('O模折射率')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_xlim(1e10, 1e12)
    ax1.set_ylim(0, 1.1)

    ax2.plot(freq_range, N_X, 'r-', linewidth=2, label='X模')
    ax2.axvline(x=pd.cutoff_frequency_X_mode_high()/(2*np.pi), color='b',
                linestyle='--', label='X模上截止')
    ax2.axvline(x=pd.cutoff_frequency_X_mode_low()/(2*np.pi), color='g',
                linestyle='--', label='X模下截止')
    ax2.set_xlabel('频率 ν (Hz)', fontsize=12)
    ax2.set_ylabel('折射率 N', fontsize=12)
    ax2.set_title('X模折射率')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_xlim(1e10, 1e12)
    ax2.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig('e:/temp/record10/208/plasma_dispersion.png', dpi=150)
    plt.close()
    print("等离子体色散图像已保存: plasma_dispersion.png")


def plot_absorption_emission(sr, rt):
    n_e = 1e19
    T_e_keV = 1.0
    T_e_erg = T_e_keV * 1.602e-9

    freq_range = np.logspace(10, 12, 100)

    epsilon = np.zeros_like(freq_range)
    alpha = np.zeros_like(freq_range)

    for i, nu in enumerate(freq_range):
        epsilon[i] = rt.emissivity(nu, n_e, T_e_erg)
        alpha[i] = rt.absorption_coefficient(nu, n_e, T_e_erg)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(freq_range, epsilon, 'b-', linewidth=2)
    ax1.set_xlabel('频率 ν (Hz)', fontsize=12)
    ax1.set_ylabel('发射系数 ε [erg cm⁻³ s⁻¹ Hz⁻¹ sr⁻¹]', fontsize=12)
    ax1.set_title('发射系数')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax2.plot(freq_range, alpha, 'r-', linewidth=2)
    ax2.set_xlabel('频率 ν (Hz)', fontsize=12)
    ax2.set_ylabel('吸收系数 α [cm⁻¹]', fontsize=12)
    ax2.set_title('吸收系数')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('e:/temp/record10/208/absorption_emission.png', dpi=150)
    plt.close()
    print("吸收发射系数图像已保存: absorption_emission.png")


def plot_optical_depth(sr, rt):
    n_e = 1e19
    T_e_keV = 1.0
    T_e_erg = T_e_keV * 1.602e-9
    path_length = 100.0

    freq_range = np.logspace(10, 12, 100)
    tau = np.zeros_like(freq_range)
    T_b = np.zeros_like(freq_range)

    for i, nu in enumerate(freq_range):
        _, tau_i = rt.transfer_equation_homogeneous(nu, n_e, T_e_erg, path_length)
        tau[i] = tau_i
        T_b[i] = rt.brightness_temperature(nu, tau_i, T_e_erg)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(freq_range, tau, 'b-', linewidth=2)
    ax1.axhline(y=1, color='r', linestyle='--', label='τ = 1')
    ax1.set_xlabel('频率 ν (Hz)', fontsize=12)
    ax1.set_ylabel('光学厚度 τ', fontsize=12)
    ax1.set_title('光学厚度')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax2.plot(freq_range, T_b/1.602e-9, 'r-', linewidth=2)
    ax2.axhline(y=T_e_keV, color='b', linestyle='--', label='电子温度')
    ax2.set_xlabel('频率 ν (Hz)', fontsize=12)
    ax2.set_ylabel('亮温度 T_b [keV]', fontsize=12)
    ax1.set_title('亮温度')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')

    plt.tight_layout()
    plt.savefig('e:/temp/record10/208/optical_depth.png', dpi=150)
    plt.close()
    print("光学厚度图像已保存: optical_depth.png")


def plot_tokamak_profile(tokamak):
    rho = np.linspace(0, 1, 100)

    n_e0 = 1e20
    T_e0 = 2.0

    B_profile = tokamak.magnetic_field_normalized(rho)
    n_e_profile = tokamak.electron_density_parabolic(rho, n_e0)
    T_e_profile = tokamak.electron_temperature_parabolic(rho, T_e0)
    q_profile = tokamak.safety_factor(rho)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(rho, B_profile, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('ρ', fontsize=12)
    axes[0, 0].set_ylabel('B [T]', fontsize=12)
    axes[0, 0].set_title('磁场剖面')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(rho, n_e_profile/1e20, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('ρ', fontsize=12)
    axes[0, 1].set_ylabel('n_e [10²⁰ m⁻³]', fontsize=12)
    axes[0, 1].set_title('电子密度剖面')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(rho, T_e_profile/1.602e-9, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('ρ', fontsize=12)
    axes[1, 0].set_ylabel('T_e [keV]', fontsize=12)
    axes[1, 0].set_title('电子温度剖面')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(rho, q_profile, 'm-', linewidth=2)
    axes[1, 1].set_xlabel('ρ', fontsize=12)
    axes[1, 1].set_ylabel('q(ρ)', fontsize=12)
    axes[1, 1].set_title('安全因子剖面')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('e:/temp/record10/208/tokamak_profile.png', dpi=150)
    plt.close()
    print("托卡马克剖面图像已保存: tokamak_profile.png")


def plot_ecei_measurement(ecei):
    n_e0 = 1e20
    T_e0_keV = 2.0

    harmonic = 2
    nu_center = ecei.tokamak.cyclotron_freq(ecei.tokamak.R0, harmonic)

    freq_range = np.linspace(nu_center * 0.8, nu_center * 1.2, 30)

    T_b_vs_freq = np.zeros_like(freq_range)

    for i, nu in enumerate(freq_range):
        _, _, _, T_b, _ = ecei.measure_ecei(nu, n_e0, T_e0_keV, harmonic, n_points=30)
        T_b_vs_freq[i] = np.mean(T_b)/1.602e-9

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(freq_range/1e9, T_b_vs_freq, 'b-', linewidth=2)
    ax.set_xlabel('频率 ν (GHz)', fontsize=12)
    ax.set_ylabel('亮温度 T_b [keV]', fontsize=12)
    ax.set_title(f'ECEI测量 (第{harmonic}次谐波)')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=nu_center/1e9, color='r', linestyle='--',
               label=f'中心频率 {nu_center/1e9:.1f} GHz')
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('e:/temp/record10/208/ecei_measurement.png', dpi=150)
    plt.close()
    print("ECEI测量图像已保存: ecei_measurement.png")


def plot_ecei_image(ecei):
    n_e0 = 1e20
    T_e0_keV = 2.0

    harmonic = 2
    nu_center = ecei.tokamak.cyclotron_freq(ecei.tokamak.R0, harmonic)

    frequencies = np.linspace(nu_center * 0.85, nu_center * 1.15, 20)

    _, R_values, rho_values, image_data = ecei.ecei_image(
        frequencies, n_e0, T_e0_keV, harmonic, n_radial=20)

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.pcolormesh(rho_values, frequencies/1e9, image_data/1.602e-9,
                       cmap='hot', shading='auto')
    ax.set_xlabel('ρ', fontsize=12)
    ax.set_ylabel('频率 ν (GHz)', fontsize=12)
    ax.set_title(f'ECEI图像 (第{harmonic}次谐波)')
    plt.colorbar(im, ax=ax, label='亮温度 T_b [keV]')

    plt.tight_layout()
    plt.savefig('e:/temp/record10/208/ecei_image.png', dpi=150)
    plt.close()
    print("ECEI图像已保存: ecei_image.png")


def plot_radial_profile(ecei):
    n_e0 = 1e20
    T_e0_keV = 2.0

    harmonic = 2

    nu_list = [
        ecei.tokamak.cyclotron_freq(ecei.tokamak.R0 - 0.5*ecei.tokamak.a, harmonic),
        ecei.tokamak.cyclotron_freq(ecei.tokamak.R0, harmonic),
        ecei.tokamak.cyclotron_freq(ecei.tokamak.R0 + 0.5*ecei.tokamak.a, harmonic)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, nu in enumerate(nu_list):
        R_values, rho_values, _, T_b_array, tau_array = ecei.measure_ecei(
            nu, n_e0, T_e0_keV, harmonic, n_points=30)

        axes[i].plot(rho_values, T_b_array/1.602e-9, 'b-', linewidth=2, label='T_b')
        axes[i].plot(rho_values, tau_array, 'r--', linewidth=2, label='τ')
        axes[i].set_xlabel('ρ', fontsize=12)
        axes[i].set_title(f'ν = {nu/1e9:.1f} GHz')
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)

    axes[0].set_ylabel('T_b [keV] / τ', fontsize=12)

    plt.tight_layout()
    plt.savefig('e:/temp/record10/208/radial_profile.png', dpi=150)
    plt.close()
    print("径向剖面图像已保存: radial_profile.png")


def print_ecei_parameters(tokamak):
    print("\n" + "="*70)
    print("托卡马克ECEI诊断参数")
    print("="*70)
    print(f"  大半径 R0 = {tokamak.R0} m")
    print(f"  小半径 a = {tokamak.a} m")
    print(f"  中心磁场 B0 = {tokamak.B0} T")
    print(f"  纵横比 A = {tokamak.aspect_ratio:.1f}")

    nu_ce_center = tokamak.cyclotron_freq(tokamak.R0, harmonic=1)
    print(f"\n  中心回旋频率:")
    print(f"    第1次谐波: {nu_ce_center/1e9:.2f} GHz")
    print(f"    第2次谐波: {2*nu_ce_center/1e9:.2f} GHz")
    print(f"    第3次谐波: {3*nu_ce_center/1e9:.2f} GHz")

    R_range = np.linspace(tokamak.R0 - tokamak.a, tokamak.R0 + tokamak.a, 5)
    print(f"\n  径向回旋频率范围 (第2次谐波):")
    for R in R_range:
        rho = (R - tokamak.R0) / tokamak.a
        nu_ce = tokamak.cyclotron_freq(R, harmonic=2)
        print(f"    ρ = {rho:.1f}: ν = {nu_ce/1e9:.2f} GHz")

    print("="*70)


def main():
    from synchrotron_radiation import SynchrotronRadiation

    B_tesla = 5.3
    B_gauss = B_tesla * 1e4
    sr = SynchrotronRadiation(B=B_gauss, alpha=np.pi/2)

    rt = RadiationTransfer(sr)

    tokamak = TokamakProfile(R0=6.2, a=2.0, B0=B_tesla)

    ecei = ECEIDiagnostic(tokamak, rt)

    print_ecei_parameters(tokamak)

    print("\n" + "="*70)
    print("生成图像中...")
    print("="*70)

    plot_plasma_dispersion(tokamak)
    plot_absorption_emission(sr, rt)
    plot_optical_depth(sr, rt)
    plot_tokamak_profile(tokamak)
    plot_ecei_measurement(ecei)
    plot_ecei_image(ecei)
    plot_radial_profile(ecei)

    print("\n" + "="*70)
    print("ECEI模拟完成！")
    print("="*70)


if __name__ == "__main__":
    main()
