import numpy as np
from scipy.integrate import quad
from scipy.constants import hbar, k, pi

class PhononBTE:
    def __init__(self, material='Si', L=None):
        self.material = material
        self.L = L
        
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
        self.v_long = 8433.0
        self.v_trans = 5845.0
        self.theta_D = 640.0
        self.gamma = 1.5
        self.A = 3.02e-45
        self.B = 1.48e-19
        self.n_atoms = 5e28
    
    def _setup_germanium(self):
        self.rho = 5323.0
        self.v_long = 5410.0
        self.v_trans = 3350.0
        self.theta_D = 374.0
        self.gamma = 1.5
        self.A = 8.02e-45
        self.B = 2.9e-19
        self.n_atoms = 4.42e28
    
    def _setup_gaas(self):
        self.rho = 5317.0
        self.v_long = 4730.0
        self.v_trans = 3340.0
        self.theta_D = 360.0
        self.gamma = 1.4
        self.A = 1.0e-44
        self.B = 2.0e-19
        self.n_atoms = 4.42e28
    
    def _setup_generic(self):
        self.rho = 2330.0
        self.v_long = 6000.0
        self.v_trans = 4000.0
        self.theta_D = 500.0
        self.gamma = 1.5
        self.A = 3.0e-45
        self.B = 1.5e-19
        self.n_atoms = 5e28
    
    def average_velocity(self):
        return (2/self.v_trans**3 + 1/self.v_long**3)**(-1/3)
    
    def debye_frequency(self):
        v_avg = self.average_velocity()
        return (6 * pi**2 * self.n_atoms)**(1/3) * v_avg
    
    def debye_wavevector(self):
        v_avg = self.average_velocity()
        omega_D = self.debye_frequency()
        return omega_D / v_avg
    
    def bose_einstein(self, omega, T):
        if T == 0:
            return 0.0
        x = hbar * omega / (k * T)
        return 1.0 / (np.exp(x) - 1.0)
    
    def bose_einstein_derivative(self, omega, T):
        if T == 0:
            return 0.0
        x = hbar * omega / (k * T)
        exp_x = np.exp(x)
        return -exp_x * x / (k * T * (exp_x - 1)**2)
    
    def tau_umklapp(self, omega, T):
        if T == 0:
            return np.inf
        return 1.0 / (self.A * omega**2 * T * np.exp(-self.theta_D / (3 * T)))
    
    def tau_boundary(self, omega):
        if self.L is None:
            return np.inf
        v_avg = self.average_velocity()
        return self.L / v_avg
    
    def tau_impurity(self, omega):
        return 1.0 / (self.B * omega**4)
    
    def relaxation_time(self, omega, T):
        tau_u = self.tau_umklapp(omega, T)
        tau_b = self.tau_boundary(omega)
        tau_i = self.tau_impurity(omega)
        
        tau_inv = 1.0/tau_u + 1.0/tau_b + 1.0/tau_i
        return 1.0 / tau_inv
    
    def phonon_dos(self, omega):
        v_avg = self.average_velocity()
        omega_D = self.debye_frequency()
        
        if omega > omega_D:
            return 0.0
        
        return 3 * omega**2 / (2 * pi**2 * v_avg**3)
    
    def mode_heat_capacity(self, omega, T):
        if T == 0:
            return 0.0
        return k * (hbar * omega / (k * T))**2 * np.exp(hbar * omega / (k * T)) / \
               (np.exp(hbar * omega / (k * T)) - 1)**2
    
    def thermal_conductivity_integrand(self, omega, T):
        v_avg = self.average_velocity()
        C = self.mode_heat_capacity(omega, T)
        tau = self.relaxation_time(omega, T)
        g = self.phonon_dos(omega)
        
        return (1.0/3.0) * C * v_avg**2 * tau * g
    
    def thermal_conductivity(self, T):
        omega_D = self.debye_frequency()
        
        kappa, _ = quad(
            self.thermal_conductivity_integrand,
            0, omega_D,
            args=(T,),
            epsabs=1e-10,
            epsrel=1e-8
        )
        
        return kappa
    
    def thermal_conductivity_k(self, T, k_points=100):
        k_D = self.debye_wavevector()
        v_avg = self.average_velocity()
        
        k_array = np.linspace(0, k_D, k_points)
        kappa = 0.0
        
        for i, k in enumerate(k_array[1:], 1):
            dk = k_array[i] - k_array[i-1]
            omega = v_avg * k
            
            C = self.mode_heat_capacity(omega, T)
            tau = self.relaxation_time(omega, T)
            
            kappa += (1.0/3.0) * C * v_avg**2 * tau * (k**2 / (2 * pi**2)) * dk
        
        return kappa * 3
    
    def spectral_thermal_conductivity(self, omega, T):
        v_avg = self.average_velocity()
        C = self.mode_heat_capacity(omega, T)
        tau = self.relaxation_time(omega, T)
        g = self.phonon_dos(omega)
        
        return (1.0/3.0) * C * v_avg**2 * tau * g
    
    def mean_free_path(self, omega, T):
        v_avg = self.average_velocity()
        tau = self.relaxation_time(omega, T)
        return v_avg * tau
    
    def cumulative_thermal_conductivity(self, T, max_lambda=None):
        omega_D = self.debye_frequency()
        v_avg = self.average_velocity()
        
        if max_lambda is None:
            max_lambda = self.mean_free_path(omega_D * 0.01, T)
        
        def integrand(omega):
            lambda_val = self.mean_free_path(omega, T)
            if lambda_val > max_lambda:
                return 0.0
            return self.spectral_thermal_conductivity(omega, T)
        
        kappa_cum, _ = quad(integrand, 0, omega_D, epsabs=1e-10, epsrel=1e-8)
        return kappa_cum
    
    def size_effect_thermal_conductivity(self, T, L_array):
        kappa_bulk = self.thermal_conductivity(T)
        kappas = []
        
        for L in L_array:
            self.L = L
            kappa = self.thermal_conductivity(T)
            kappas.append(kappa)
        
        return np.array(kappas)
