import numpy as np
from scipy import special
from scipy.optimize import root, minimize
from scipy.integrate import quad


class PlasmaSpecies:
    """等离子体物种类"""
    def __init__(self, name, mass, charge, density, temperature, relativistic=False):
        self.name = name
        self.mass = mass
        self.charge = charge
        self.density = density
        self.temperature = temperature
        self.relativistic = relativistic
        
        self.e = 1.602176634e-19
        self.c = 299792458.0
        
        self.T_J = temperature * self.e
        self.kT = self.T_J
        self.mc2 = mass * self.c**2
        
        if relativistic:
            self.theta = self.kT / self.mc2
        else:
            self.v_th = np.sqrt(2 * self.kT / mass)
    
    def __repr__(self):
        rel_str = " (相对论)" if self.relativistic else ""
        return f"PlasmaSpecies({self.name}, n={self.density:.2e}, T={self.temperature:.1f} eV{rel_str})"


def maxwellian_distribution(v, v_th):
    """非相对论麦克斯韦分布函数"""
    return (1.0 / (np.sqrt(np.pi) * v_th)) * np.exp(-(v / v_th) ** 2)


def juettner_distribution(gamma, theta):
    """相对论麦克斯韦分布（Jüttner分布）
    
    参数:
        gamma: 洛伦兹因子
        theta: kT/(mc²)，无量纲温度
    """
    beta = np.sqrt(1 - 1/gamma**2)
    K2 = special.kv(2, 1/theta)
    norm = (theta * K2)
    return (gamma**2 * beta * np.exp(-gamma/theta)) / norm


def juettner_momentum_distribution(p, mc, theta):
    """动量空间的Jüttner分布"""
    gamma = np.sqrt(1 + (p/mc)**2)
    K2 = special.kv(2, 1/theta)
    norm = 4 * np.pi * (mc)**3 * theta * K2
    return (p**2 * np.exp(-gamma/theta)) / norm


def plasma_dispersion_function(z, method='wofz'):
    """高精度等离子体色散函数 Z(z)"""
    z = np.asarray(z, dtype=np.complex128)
    
    if method == 'wofz':
        return 1j * np.sqrt(np.pi) * special.wofz(z)
    elif method == 'asymptotic':
        z_squared = z ** 2
        series = 1 + 1/(2*z_squared) + 3/(4*z_squared**2) + 15/(8*z_squared**3)
        return -1/z * series
    else:
        return 1j * np.sqrt(np.pi) * special.wofz(z)


def plasma_dispersion_function_derivative(z):
    """等离子体色散函数的导数 Z'(z)"""
    z = np.asarray(z, dtype=np.complex128)
    return -2 * (1 + z * plasma_dispersion_function(z))


def relativistic_plasma_dispersion_function(z, theta, method='integral'):
    """相对论等离子体色散函数
    
    对于相对论麦克斯韦分布，需要计算速度空间积分
    """
    z = np.asarray(z, dtype=np.complex128)
    c = 299792458.0
    
    if method == 'weakly_relativistic' and theta < 0.1:
        Z_nonrel = plasma_dispersion_function(z)
        correction = 1 + (15/4) * theta
        return Z_nonrel * correction
    
    def integrand_real(beta):
        gamma = 1 / np.sqrt(1 - beta**2)
        f_beta = juettner_distribution(gamma, theta)
        v = beta * c
        return f_beta * v / (v - z.real)
    
    def integrand_imag(beta):
        gamma = 1 / np.sqrt(1 - beta**2)
        f_beta = juettner_distribution(gamma, theta)
        v = beta * c
        
        delta = np.exp(-((v - z.real)/z.imag)**2) / (np.sqrt(np.pi) * abs(z.imag))
        return f_beta * delta * np.pi
    
    try:
        beta_max = min(0.999, 1 - 1e-6)
        real_part, _ = quad(integrand_real, 0, beta_max)
        imag_part, _ = quad(integrand_imag, 0, beta_max)
        return real_part + 1j * imag_part
    except:
        return plasma_dispersion_function(z)


def relativistic_plasma_dispersion_function_approx(z, theta):
    """相对论等离子体色散函数的近似"""
    z = np.asarray(z, dtype=np.complex128)
    
    if theta < 0.01:
        return plasma_dispersion_function(z)
    
    Z_nonrel = plasma_dispersion_function(z)
    
    correction = 1 + (15/4) * theta - (105/16) * theta**2 + (1155/64) * theta**3
    
    return Z_nonrel * correction


def plasma_frequency(species, epsilon0=8.854187817e-12):
    """计算等离子体频率"""
    return np.sqrt(species.density * species.charge**2 / (species.mass * epsilon0))


def thermal_velocity(species):
    """计算热速度（考虑相对论修正）"""
    if species.relativistic:
        theta = species.theta
        v_th_nonrel = np.sqrt(2 * species.kT / species.mass)
        correction = np.sqrt(1 + (5/2) * theta)
        return v_th_nonrel / correction
    else:
        return np.sqrt(2 * species.kT / species.mass)


def dielectric_function_multispecies(omega, k, species_list, epsilon0=8.854187817e-12):
    """多物种静电波介电函数
    
    支持相对论和非相对论物种混合
    """
    c = 299792458.0
    eps = 1.0 + 0j
    
    if abs(k) < 1e-15:
        k = 1e-15
    
    for species in species_list:
        omega_p = plasma_frequency(species, epsilon0)
        v_th = thermal_velocity(species)
        
        if species.relativistic:
            theta = species.theta
            z = omega / (k * v_th)
            Z_rel = relativistic_plasma_dispersion_function_approx(z, theta)
            term = (omega_p**2 / (k**2 * v_th**2)) * Z_rel
        else:
            z = omega / (k * v_th)
            Z_nonrel = plasma_dispersion_function(z)
            term = (omega_p**2 / (k**2 * v_th**2)) * Z_nonrel
        
        eps -= term
    
    return eps


def dielectric_jacobian_multispecies(omega, k, species_list, epsilon0=8.854187817e-12):
    """多物种介电函数的雅可比矩阵"""
    c = 299792458.0
    d_eps_domega = 0 + 0j
    
    if abs(k) < 1e-15:
        k = 1e-15
    
    for species in species_list:
        omega_p = plasma_frequency(species, epsilon0)
        v_th = thermal_velocity(species)
        
        z = omega / (k * v_th)
        
        if species.relativistic:
            theta = species.theta
            dZ = plasma_dispersion_function_derivative(z)
        else:
            dZ = plasma_dispersion_function_derivative(z)
        
        d_term = (omega_p**2 / (k**3 * v_th**3)) * dZ
        d_eps_domega -= d_term
    
    return d_eps_domega


def solve_dispersion_newton_multispecies(k, species_list, omega_guess, 
                                         max_iter=100, tol=1e-12):
    """多物种牛顿法求解色散关系"""
    omega = omega_guess + 0j
    
    for i in range(max_iter):
        eps = dielectric_function_multispecies(omega, k, species_list)
        d_eps = dielectric_jacobian_multispecies(omega, k, species_list)
        
        if abs(d_eps) < 1e-30:
            break
        
        delta = -eps / d_eps
        omega = omega + delta
        
        if abs(delta) < tol * max(1.0, abs(omega)):
            return omega, i+1, True
    
    return omega, max_iter, False


def solve_dispersion_robust_multispecies(k, species_list, omega_guess):
    """鲁棒的多物种色散关系求解器"""
    omega, iters, converged = solve_dispersion_newton_multispecies(
        k, species_list, omega_guess, max_iter=200, tol=1e-14
    )
    
    if converged:
        eps_final = dielectric_function_multispecies(omega, k, species_list)
        if abs(eps_final) < 1e-10:
            return omega
    
    def objective(x):
        omega = x[0] + 1j * x[1]
        eps = dielectric_function_multispecies(omega, k, species_list)
        return abs(eps) ** 2
    
    x0 = [np.real(omega_guess), np.imag(omega_guess)]
    
    bounds = [
        (0.1 * abs(x0[0]), 10.0 * abs(x0[0])),
        (-5.0 * abs(x0[0]), 0.0)
    ]
    
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'ftol': 1e-15, 'maxiter': 500})
    
    if result.success:
        omega = result.x[0] + 1j * result.x[1]
        eps_final = dielectric_function_multispecies(omega, k, species_list)
        if abs(eps_final) < 1e-8:
            return omega
    
    def objective_root(x):
        omega = x[0] + 1j * x[1]
        eps = dielectric_function_multispecies(omega, k, species_list)
        return [np.real(eps), np.imag(eps)]
    
    result = root(objective_root, x0, method='lm', 
                  options={'ftol': 1e-12, 'maxfev': 1000})
    
    if result.success:
        omega = result.x[0] + 1j * result.x[1]
        return omega
    
    return None


def estimate_ion_sound_velocity(species_list):
    """估计离子声波速度"""
    e = 1.602176634e-19
    
    electrons = [s for s in species_list if s.charge < 0]
    ions = [s for s in species_list if s.charge > 0]
    
    if not electrons or not ions:
        return None
    
    electron = electrons[0]
    total_ion_density = sum(ion.density for ion in ions)
    avg_ion_mass = sum(ion.mass * ion.density for ion in ions) / total_ion_density
    
    c_s = np.sqrt(electron.kT / avg_ion_mass)
    return c_s


def estimate_initial_omega(k, species_list, wave_type='ion_sound'):
    """估计初始频率"""
    e = 1.602176634e-19
    c = 299792458.0
    
    electrons = [s for s in species_list if s.charge < 0]
    ions = [s for s in species_list if s.charge > 0]
    
    if wave_type == 'ion_sound' and electrons and ions:
        electron = electrons[0]
        total_ion_density = sum(ion.density for ion in ions)
        avg_ion_mass = sum(ion.mass * ion.density for ion in ions) / total_ion_density
        
        c_s = np.sqrt(electron.kT / avg_ion_mass)
        
        omega_r = k * c_s
        
        v_the = thermal_velocity(electron)
        z_e = omega_r / (k * v_the)
        
        gamma = -np.sqrt(np.pi/8) * (avg_ion_mass/electron.mass)**0.5 * \
                z_e**3 * np.exp(-z_e**2 - 3/2) * omega_r
        
        return omega_r + 1j * gamma
    
    elif wave_type == 'electron_plasma' and electrons:
        electron = electrons[0]
        omega_pe = plasma_frequency(electron)
        
        v_the = thermal_velocity(electron)
        omega_r = np.sqrt(omega_pe**2 + 3 * k**2 * v_the**2)
        
        z_e = omega_r / (k * v_the)
        gamma = -np.sqrt(np.pi/8) * omega_pe * (omega_pe/(k*v_the))**3 * \
                np.exp(-(omega_pe/(k*v_the))**2 - 3/2)
        
        return omega_r + 1j * gamma
    
    else:
        if electrons:
            omega_pe = plasma_frequency(electrons[0])
            return omega_pe + 1j * (-0.01 * omega_pe)
        elif ions:
            omega_pi = plasma_frequency(ions[0])
            return omega_pi + 1j * (-0.01 * omega_pi)
        return 1e9 + 1j * (-1e7)


def verify_solution_multispecies(omega, k, species_list):
    """验证多物种解的精度"""
    eps = dielectric_function_multispecies(omega, k, species_list)
    residual = abs(eps)
    
    return {
        'omega': omega,
        'residual': residual,
        'accurate': residual < 1e-8
    }


def print_species_info(species_list):
    """打印物种信息"""
    print("=" * 80)
    print("等离子体物种信息:")
    print("=" * 80)
    print(f"{'物种':<12} {'质量(kg)':<15} {'电荷(C)':<12} {'密度(m^-3)':<15} {'温度(eV)':<12} {'相对论':<8}")
    print("-" * 80)
    
    for species in species_list:
        rel_str = "是" if species.relativistic else "否"
        print(f"{species.name:<12} {species.mass:<15.2e} {species.charge:<12.2e} "
              f"{species.density:<15.2e} {species.temperature:<12.1f} {rel_str:<8}")
    print()


if __name__ == "__main__":
    e = 1.602176634e-19
    m_e = 9.1093837015e-31
    m_p = 1.67262192369e-27
    m_D = 2.014 * 1.66053906660e-27
    
    species_list = [
        PlasmaSpecies("电子", m_e, -e, 1e20, 100.0, relativistic=False),
        PlasmaSpecies("氢离子", m_p, e, 0.5e20, 10.0, relativistic=False),
        PlasmaSpecies("氘离子", m_D, e, 0.5e20, 10.0, relativistic=False),
    ]
    
    print_species_info(species_list)
    
    k_values = np.logspace(-3, 1, 25)
    omega_results = []
    
    print("=" * 80)
    print("多物种静电波色散关系求解")
    print("=" * 80)
    print()
    
    for k in k_values:
        omega_guess = estimate_initial_omega(k, species_list, wave_type='ion_sound')
        
        omega = solve_dispersion_robust_multispecies(k, species_list, omega_guess)
        
        if omega is not None:
            verification = verify_solution_multispecies(omega, k, species_list)
            omega_results.append((k, omega, verification))
            
            print(f"k = {k:.4e} m^-1")
            print(f"  ω_r = {omega.real:.4e} rad/s, γ = {omega.imag:.4e} rad/s")
            print(f"  初始猜测: ω_r0 = {omega_guess.real:.4e} rad/s, γ0 = {omega_guess.imag:.4e} rad/s")
            print(f"  残差 |ε| = {verification['residual']:.2e}")
            
            if abs(omega_guess.imag) > 1e-20:
                error = abs((omega.imag - omega_guess.imag) / omega_guess.imag) * 100
                print(f"  阻尼率与初始猜测误差: {error:.2f}%")
            print()
    
    print("=" * 80)
    print(f"成功求解 {len(omega_results)} 个波数")
    print("=" * 80)
    
    if len(omega_results) > 0:
        print("\n" + "=" * 80)
        print("结果摘要:")
        print("=" * 80)
        print(f"{'k (m^-1)':<15} {'ω_r (rad/s)':<18} {'γ (rad/s)':<18} {'γ/ω_r (%)':<12} {'残差':<10}")
        print("-" * 80)
        for k, omega, verif in omega_results:
            gamma_over_omega = abs(omega.imag / omega.real) * 100 if omega.real != 0 else 0
            print(f"{k:<15.4e} {omega.real:<18.4e} {omega.imag:<18.4e} {gamma_over_omega:<12.4f} {verif['residual']:<10.2e}")
    
    print("\n" + "=" * 80)
    print("相对论效应测试:")
    print("=" * 80)
    
    species_list_rel = [
        PlasmaSpecies("电子(相对论)", m_e, -e, 1e20, 500000.0, relativistic=True),
        PlasmaSpecies("质子", m_p, e, 1e20, 1000.0, relativistic=False),
    ]
    
    print_species_info(species_list_rel)
    
    k_test = 100.0
    omega_guess_rel = estimate_initial_omega(k_test, species_list_rel, wave_type='ion_sound')
    omega_rel = solve_dispersion_robust_multispecies(k_test, species_list_rel, omega_guess_rel)
    
    if omega_rel is not None:
        verif_rel = verify_solution_multispecies(omega_rel, k_test, species_list_rel)
        print(f"\nk = {k_test:.4e} m^-1 时的相对论结果:")
        print(f"  ω_r = {omega_rel.real:.4e} rad/s")
        print(f"  γ = {omega_rel.imag:.4e} rad/s")
        print(f"  残差 |ε| = {verif_rel['residual']:.2e}")
        print(f"  θ_e = kT_e/mc² = {species_list_rel[0].theta:.4e}")
    else:
        print("相对论测试求解失败")
