import numpy as np
from scipy.integrate import quad, simps
from scipy.constants import hbar, k, pi


class PhononBranch:
    def __init__(self, name, polarization, v_sound, gamma, scaling_factor=1.0):
        self.name = name
        self.polarization = polarization
        self.v_sound = v_sound
        self.gamma = gamma
        self.scaling_factor = scaling_factor


class EnhancedPhononBTE:
    def __init__(self, material='Si', L=None, use_branch_resolved=True, 
                 use_dispersion_correction=True):
        self.material = material
        self.L = L
        self.use_branch_resolved = use_branch_resolved
        self.use_dispersion_correction = use_dispersion_correction
        
        self.branches = []
        self._setup_material(material)
        
        self.first_principles_data = None
    
    def _setup_material(self, material):
        if material == 'Si':
            self._setup_silicon()
        elif material == 'Ge':
            self._setup_germanium()
        elif material == 'GaAs':
            self._setup_gaas()
        else:
            self._setup_generic()
    
    def _setup_silicon(self):
        self.rho = 2330.0
        self.n_atoms = 5e28
        self.theta_D = 640.0
        self.a_lattice = 5.431e-10
        
        self.A_iso = 1.08e-43
        
        self.B_U_LA = 1.4e-19
        self.B_U_TA = 3.2e-18
        
        self.B_N_LA = 2.0e-20
        self.B_N_TA = 5.0e-19
        
        self.gruneisen = 1.5
        
        self.branches = [
            PhononBranch('LA', 'longitudinal', 8433.0, 1.5, 1.0),
            PhononBranch('TA1', 'transverse', 5845.0, 1.5, 1.0),
            PhononBranch('TA2', 'transverse', 5845.0, 1.5, 1.0),
        ]
        
        self.omega_max_branch = {
            'LA': 9.6e13,
            'TA1': 7.6e13,
            'TA2': 7.6e13
        }
    
    def _setup_germanium(self):
        self.rho = 5323.0
        self.n_atoms = 4.42e28
        self.theta_D = 374.0
        self.a_lattice = 5.658e-10
        
        self.A_iso = 8.0e-43
        
        self.B_U_LA = 2.0e-19
        self.B_U_TA = 5.0e-18
        
        self.B_N_LA = 3.0e-20
        self.B_N_TA = 8.0e-19
        
        self.gruneisen = 1.5
        
        self.branches = [
            PhononBranch('LA', 'longitudinal', 5410.0, 1.5, 1.0),
            PhononBranch('TA1', 'transverse', 3350.0, 1.5, 1.0),
            PhononBranch('TA2', 'transverse', 3350.0, 1.5, 1.0),
        ]
        
        self.omega_max_branch = {
            'LA': 6.0e13,
            'TA1': 4.5e13,
            'TA2': 4.5e13
        }
    
    def _setup_gaas(self):
        self.rho = 5317.0
        self.n_atoms = 4.42e28
        self.theta_D = 360.0
        self.a_lattice = 5.653e-10
        
        self.A_iso = 5.0e-42
        
        self.B_U_LA = 3.0e-19
        self.B_U_TA = 6.0e-18
        
        self.B_N_LA = 4.0e-20
        self.B_N_TA = 1.0e-18
        
        self.gruneisen = 1.4
        
        self.branches = [
            PhononBranch('LA', 'longitudinal', 4730.0, 1.4, 1.0),
            PhononBranch('TA1', 'transverse', 3340.0, 1.4, 1.0),
            PhononBranch('TA2', 'transverse', 3340.0, 1.4, 1.0),
            PhononBranch('LO', 'longitudinal_optical', 5.0e13/1e10, 1.0, 1.0),
            PhononBranch('TO1', 'transverse_optical', 5.0e13/1e10, 1.0, 1.0),
            PhononBranch('TO2', 'transverse_optical', 5.0e13/1e10, 1.0, 1.0),
        ]
        
        self.omega_max_branch = {
            'LA': 5.0e13,
            'TA1': 3.8e13,
            'TA2': 3.8e13,
            'LO': 8.5e13,
            'TO1': 8.0e13,
            'TO2': 8.0e13
        }
    
    def _setup_generic(self):
        self.rho = 2330.0
        self.n_atoms = 5e28
        self.theta_D = 500.0
        self.a_lattice = 5.43e-10
        
        self.A_iso = 1.0e-42
        
        self.B_U_LA = 2.0e-19
        self.B_U_TA = 4.0e-18
        
        self.B_N_LA = 3.0e-20
        self.B_N_TA = 6.0e-19
        
        self.gruneisen = 1.5
        
        self.branches = [
            PhononBranch('LA', 'longitudinal', 6000.0, 1.5, 1.0),
            PhononBranch('TA1', 'transverse', 4000.0, 1.5, 1.0),
            PhononBranch('TA2', 'transverse', 4000.0, 1.5, 1.0),
        ]
        
        self.omega_max_branch = {
            'LA': 7.0e13,
            'TA1': 5.5e13,
            'TA2': 5.5e13
        }
    
    def load_first_principles_data(self, omega_array, tau_dict, v_g_dict, dos_dict):
        self.first_principles_data = {
            'omega': omega_array,
            'tau': tau_dict,
            'v_g': v_g_dict,
            'dos': dos_dict
        }
        print("第一性原理数据已加载")
    
    def debye_wavevector(self):
        return (6 * pi**2 * self.n_atoms)**(1/3)
    
    def debye_frequency(self, branch=None):
        if branch is None:
            v_avg = self.average_velocity()
            k_D = self.debye_wavevector()
            return v_avg * k_D
        else:
            branch_obj = self._get_branch(branch)
            k_D = self.debye_wavevector()
            return branch_obj.v_sound * k_D
    
    def _get_branch(self, branch_name):
        for b in self.branches:
            if b.name == branch_name:
                return b
        return None
    
    def average_velocity(self):
        v_inv_sum = 0.0
        count = 0
        for b in self.branches:
            if 'acoustic' in b.polarization or b.polarization in ['longitudinal', 'transverse']:
                v_inv_sum += 1.0 / (b.v_sound**3)
                count += 1
        return (count / v_inv_sum)**(1/3)
    
    def dispersion_relation(self, k, branch='LA'):
        branch_obj = self._get_branch(branch)
        if branch_obj is None:
            return 0
        
        v_s = branch_obj.v_sound
        
        if self.use_dispersion_correction and 'acoustic' in branch_obj.polarization or branch_obj.polarization in ['longitudinal', 'transverse']:
            k_D = self.debye_wavevector()
            omega_max = self.omega_max_branch.get(branch, v_s * k_D)
            omega_linear = v_s * k
            
            sin_factor = np.sin(pi * k / (2 * k_D))
            if sin_factor > 0:
                omega_corrected = omega_max * sin_factor
                alpha = 0.3
                return (1 - alpha) * omega_linear + alpha * omega_corrected
            else:
                return omega_linear
        else:
            return v_s * k
    
    def group_velocity(self, k, branch='LA'):
        branch_obj = self._get_branch(branch)
        if branch_obj is None:
            return 0
        
        v_s = branch_obj.v_sound
        
        if self.use_dispersion_correction and 'acoustic' in branch_obj.polarization or branch_obj.polarization in ['longitudinal', 'transverse']:
            k_D = self.debye_wavevector()
            omega_max = self.omega_max_branch.get(branch, v_s * k_D)
            
            if k < 1e-10:
                return v_s
            
            dk = 1e-3 * k_D
            omega1 = self.dispersion_relation(k - dk, branch)
            omega2 = self.dispersion_relation(k + dk, branch)
            return (omega2 - omega1) / (2 * dk)
        else:
            return v_s
    
    def bose_einstein(self, omega, T):
        if T == 0 or omega == 0:
            return 0.0
        x = hbar * omega / (k * T)
        if x > 500:
            return 0.0
        return 1.0 / (np.exp(x) - 1.0)
    
    def mode_heat_capacity(self, omega, T):
        if T == 0 or omega == 0:
            return 0.0
        x = hbar * omega / (k * T)
        if x > 500:
            return 0.0
        exp_x = np.exp(x)
        return k * x**2 * exp_x / (exp_x - 1)**2
    
    def tau_normal(self, omega, T, branch='LA'):
        if T == 0:
            return np.inf
        
        branch_obj = self._get_branch(branch)
        if branch_obj is None:
            return np.inf
        
        if 'longitudinal' in branch_obj.polarization:
            B_N = self.B_N_LA
        elif 'transverse' in branch_obj.polarization:
            B_N = self.B_N_TA
        else:
            B_N = self.B_N_LA
        
        if omega == 0:
            return np.inf
        
        return 1.0 / (B_N * omega**2 * T**3)
    
    def tau_umklapp(self, omega, T, branch='LA'):
        if T == 0:
            return np.inf
        
        branch_obj = self._get_branch(branch)
        if branch_obj is None:
            return np.inf
        
        if 'longitudinal' in branch_obj.polarization:
            B_U = self.B_U_LA
        elif 'transverse' in branch_obj.polarization:
            B_U = self.B_U_TA
        else:
            B_U = self.B_U_LA
        
        if omega == 0:
            return np.inf
        
        if T < self.theta_D / 4:
            return 1.0 / (B_U * omega**2 * T * np.exp(-self.theta_D / (3 * T)))
        else:
            return 1.0 / (B_U * omega**2 * T)
    
    def tau_isotope(self, omega, branch='LA'):
        if omega == 0:
            return np.inf
        return 1.0 / (self.A_iso * omega**4)
    
    def tau_boundary(self, k, branch='LA'):
        if self.L is None:
            return np.inf
        v_g = self.group_velocity(k, branch)
        if v_g == 0:
            return np.inf
        return self.L / v_g
    
    def tau_optical(self, omega, T):
        if 'optical' not in self._get_branch('LA').polarization:
            return 1e-12
        if T == 0:
            return np.inf
        return 1e-13
    
    def relaxation_time(self, k, omega, T, branch='LA'):
        branch_obj = self._get_branch(branch)
        if branch_obj is None:
            return 1e-15
        
        if 'optical' in branch_obj.polarization:
            return self.tau_optical(omega, T)
        
        tau_n = self.tau_normal(omega, T, branch)
        tau_u = self.tau_umklapp(omega, T, branch)
        tau_iso = self.tau_isotope(omega, branch)
        tau_b = self.tau_boundary(k, branch)
        
        tau_inv = 1.0/tau_n + 1.0/tau_u + 1.0/tau_iso + 1.0/tau_b
        
        return 1.0 / tau_inv
    
    def phonon_dos_k(self, k, branch='LA'):
        branch_obj = self._get_branch(branch)
        if branch_obj is None:
            return 0
        return (k**2) / (2 * pi**2)
    
    def thermal_conductivity_integrand_k(self, k, T, branch='LA'):
        omega = self.dispersion_relation(k, branch)
        v_g = self.group_velocity(k, branch)
        tau = self.relaxation_time(k, omega, T, branch)
        C = self.mode_heat_capacity(omega, T)
        g_k = self.phonon_dos_k(k, branch)
        
        return C * v_g**2 * tau * g_k
    
    def thermal_conductivity_branch(self, T, branch='LA', k_points=200):
        if self.first_principles_data is not None:
            return self._thermal_conductivity_from_first_principles(T, branch)
        
        k_D = self.debye_wavevector()
        k_array = np.linspace(0, k_D, k_points)
        
        integrand_values = []
        for k in k_array:
            if k == 0:
                integrand_values.append(0)
            else:
                integrand_values.append(self.thermal_conductivity_integrand_k(k, T, branch))
        
        kappa = simps(integrand_values, k_array)
        return kappa / 3.0
    
    def _thermal_conductivity_from_first_principles(self, T, branch):
        data = self.first_principles_data
        omega = data['omega']
        tau = data['tau'][branch]
        v_g = data['v_g'][branch]
        dos = data['dos'][branch]
        
        C = np.array([self.mode_heat_capacity(o, T) for o in omega])
        integrand = C * v_g**2 * tau * dos
        
        return simps(integrand, omega) / 3.0
    
    def thermal_conductivity(self, T, k_points=200):
        if not self.use_branch_resolved:
            return self._thermal_conductivity_average(T)
        
        kappa_total = 0.0
        for branch in self.branches:
            if 'optical' in branch.polarization:
                continue
            kappa_branch = self.thermal_conductivity_branch(T, branch.name, k_points)
            kappa_total += kappa_branch
        
        return kappa_total
    
    def _thermal_conductivity_average(self, T):
        v_avg = self.average_velocity()
        omega_D = self.debye_frequency()
        
        def integrand(omega):
            C = self.mode_heat_capacity(omega, T)
            tau_u = 1.0 / (2e-19 * omega**2 * T * np.exp(-self.theta_D / (3 * T)) if T > 0 else np.inf)
            tau_iso = 1.0 / (self.A_iso * omega**4) if omega > 0 else np.inf
            
            tau_inv = 1.0/tau_u + 1.0/tau_iso
            if self.L is not None:
                tau_inv += v_avg / self.L
            tau = 1.0 / tau_inv
            
            g = 3 * omega**2 / (2 * pi**2 * v_avg**3)
            
            return C * v_avg**2 * tau * g
        
        kappa, _ = quad(integrand, 0, omega_D, epsabs=1e-10, epsrel=1e-8)
        return kappa / 3.0
    
    def thermal_conductivity_spectral(self, omega, T, branch='LA'):
        branch_obj = self._get_branch(branch)
        if branch_obj is None:
            return 0
        
        k = omega / branch_obj.v_sound
        v_g = self.group_velocity(k, branch)
        tau = self.relaxation_time(k, omega, T, branch)
        C = self.mode_heat_capacity(omega, T)
        g = omega**2 / (2 * pi**2 * v_g**2 * branch_obj.v_sound) if v_g > 0 else 0
        
        return C * v_g**2 * tau * g / 3.0
    
    def mean_free_path_branch(self, k, T, branch='LA'):
        omega = self.dispersion_relation(k, branch)
        v_g = self.group_velocity(k, branch)
        tau = self.relaxation_time(k, omega, T, branch)
        return v_g * tau
    
    def cumulative_thermal_conductivity(self, T, max_lambda=None, k_points=200):
        if max_lambda is None:
            return self.thermal_conductivity(T, k_points)
        
        k_D = self.debye_wavevector()
        k_array = np.linspace(0, k_D, k_points)
        
        kappa_cum = 0.0
        for branch in self.branches:
            if 'optical' in branch.polarization:
                continue
            
            for i, k in enumerate(k_array[1:], 1):
                dk = k_array[i] - k_array[i-1]
                mfp = self.mean_free_path_branch(k, T, branch.name)
                
                if mfp <= max_lambda:
                    integrand = self.thermal_conductivity_integrand_k(k, T, branch.name)
                    kappa_cum += integrand * dk / 3.0
        
        return kappa_cum
    
    def get_branch_contributions(self, T, k_points=200):
        contributions = {}
        for branch in self.branches:
            if 'optical' in branch.polarization:
                continue
            kappa = self.thermal_conductivity_branch(T, branch.name, k_points)
            contributions[branch.name] = kappa
        return contributions
    
    def size_effect_thermal_conductivity(self, T, L_array, k_points=200):
        original_L = self.L
        kappas = []
        
        for L in L_array:
            self.L = L
            kappa = self.thermal_conductivity(T, k_points)
            kappas.append(kappa)
        
        self.L = original_L
        return np.array(kappas)
    
    def compare_models(self, T=300):
        kappa_branch = self.thermal_conductivity(T)
        
        original_use_branch = self.use_branch_resolved
        self.use_branch_resolved = False
        kappa_average = self.thermal_conductivity(T)
        self.use_branch_resolved = original_use_branch
        
        return {
            'mode_resolved': kappa_branch,
            'gray_body': kappa_average,
            'difference': (kappa_branch - kappa_average) / kappa_average * 100
        }
