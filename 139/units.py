import numpy as np


class UnitConverter:
    def __init__(self, 
                 mass_per_bead: float = 110.0,
                 length_scale: float = 3.8,
                 energy_scale: float = 1.0):
        
        self.mass_Da_per_bead = mass_per_bead
        self.length_A_per_sigma = length_scale
        self.energy_kcal_per_epsilon = energy_scale
        
        self.kB = 1.9872041e-3
        self.kB_SI = 1.38064852e-23
        
        self.Da_to_kg = 1.660539040e-27
        self.A_to_m = 1e-10
        self.kcal_to_J = 4184.0
        self.ps_to_s = 1e-12
        
        self._calculate_time_scale()
        self._calculate_viscosity_scale()
        
    def _calculate_time_scale(self):
        mass_kg = self.mass_Da_per_bead * self.Da_to_kg
        length_m = self.length_A_per_sigma * self.A_to_m
        energy_J = self.energy_kcal_per_epsilon * self.kcal_to_J
        
        self.time_s_per_tau = np.sqrt(mass_kg * length_m**2 / energy_J)
        self.time_ps_per_tau = self.time_s_per_tau / self.ps_to_s
        
    def _calculate_viscosity_scale(self):
        length_m = self.length_A_per_sigma * self.A_to_m
        energy_J = self.energy_kcal_per_epsilon * self.kcal_to_J
        time_s = self.time_s_per_tau
        
        self.viscosity_SI_per_eta = energy_J * time_s / length_m**3
        
        self.water_viscosity_SI = 0.001
        self.water_viscosity_reduced = self.water_viscosity_SI / self.viscosity_SI_per_eta
    
    def stokes_einstein_gamma(self, 
                               radius_A: float, 
                               viscosity_SI: float = None) -> float:
        if viscosity_SI is None:
            viscosity_SI = self.water_viscosity_SI
        
        radius_m = radius_A * self.A_to_m
        gamma_SI = 6 * np.pi * viscosity_SI * radius_m
        
        mass_kg = self.mass_Da_per_bead * self.Da_to_kg
        time_s = self.time_s_per_tau
        
        gamma_reduced = gamma_SI * time_s / mass_kg
        
        return gamma_reduced
    
    def estimate_protein_radius(self, num_beads: int) -> float:
        return 2.0 + 1.5 * (num_beads ** (1.0/3.0))
    
    def get_reduced_gamma(self, 
                           num_beads: int, 
                           viscosity_SI: float = None) -> float:
        radius_A = self.estimate_protein_radius(num_beads)
        radius_A_per_bead = radius_A / (num_beads ** (1.0/3.0))
        return self.stokes_einstein_gamma(radius_A_per_bead, viscosity_SI)


def calibrate_gamma_by_folding_rate(experimental_kf: float,
                                     simulated_kf: float,
                                     current_gamma: float,
                                     time_ps_per_tau: float = None) -> tuple:
    correction_factor = experimental_kf / simulated_kf
    
    new_gamma = current_gamma / correction_factor
    
    return new_gamma, correction_factor


def get_optimal_gamma(num_beads: int, 
                       temperature_K: float = 300.0,
                       model: str = 'stokes_einstein') -> float:
    
    converter = UnitConverter()
    
    if model == 'stokes_einstein':
        gamma = converter.get_reduced_gamma(num_beads)
        
    elif model == 'empirical':
        gamma = 0.1 + 0.005 * num_beads
        
    elif model == 'literature':
        gamma = 0.15
        
    else:
        raise ValueError(f"Unknown model: {model}")
    
    T_ref = 300.0
    gamma *= (T_ref / temperature_K) ** 0.5
    
    return max(0.05, min(2.0, gamma))


def folding_rate_from_mfpt(mfpt_steps: float, 
                            dt: float = 0.005,
                            time_ps_per_tau: float = None) -> float:
    if time_ps_per_tau is None:
        converter = UnitConverter()
        time_ps_per_tau = converter.time_ps_per_tau
    
    total_ps = mfpt_steps * dt * time_ps_per_tau
    total_s = total_ps * 1e-12
    
    kf = 1.0 / total_s
    
    return kf


class FoldingRateCalibrator:
    def __init__(self, num_beads: int):
        self.num_beads = num_beads
        self.converter = UnitConverter()
        
        self.experimental_data = {
            'protein_g': {'N': 56, 'kf': 1.5e4, 'temp': 298},
            'ubiquitin': {'N': 76, 'kf': 3.0e3, 'temp': 298},
            'ci2': {'N': 64, 'kf': 4.0e3, 'temp': 298},
            'sh3': {'N': 57, 'kf': 8.0e2, 'temp': 298},
        }
    
    def estimate_kf_from_size(self) -> float:
        log_N = np.log10(self.num_beads)
        log_kf = 7.5 - 2.5 * log_N
        return 10 ** log_kf
    
    def get_calibrated_gamma(self, 
                              simulated_mfpt: float,
                              dt: float = 0.005,
                              target_kf: float = None) -> tuple:
        
        if target_kf is None:
            target_kf = self.estimate_kf_from_size()
        
        simulated_kf = folding_rate_from_mfpt(
            simulated_mfpt, 
            dt, 
            self.converter.time_ps_per_tau
        )
        
        gamma_guess = get_optimal_gamma(self.num_beads, model='stokes_einstein')
        
        corrected_gamma, correction = calibrate_gamma_by_folding_rate(
            target_kf,
            simulated_kf,
            gamma_guess,
            self.converter.time_ps_per_tau
        )
        
        return {
            'target_kf': target_kf,
            'simulated_kf': simulated_kf,
            'initial_gamma': gamma_guess,
            'corrected_gamma': corrected_gamma,
            'correction_factor': correction,
            'time_scale_ps': self.converter.time_ps_per_tau
        }
    
    def print_calibration_report(self, results: dict):
        print("\n" + "="*60)
        print("摩擦系数校准报告")
        print("="*60)
        print(f"\n蛋白质大小: {self.num_beads} 个残基")
        print(f"目标折叠速率 kf = {results['target_kf']:.2e} s^-1")
        print(f"模拟折叠速率 kf = {results['simulated_kf']:.2e} s^-1")
        print(f"\n时间尺度: τ = {results['time_scale_ps']:.3f} ps")
        print(f"\n初始摩擦系数 (Stokes-Einstein): γ = {results['initial_gamma']:.4f}")
        print(f"校正后摩擦系数: γ = {results['corrected_gamma']:.4f}")
        print(f"校正因子: {results['correction_factor']:.2f}x")
        print("="*60 + "\n")


def get_water_viscosity(T: float) -> float:
    T0 = 293.15
    eta0 = 0.001002
    return eta0 * np.exp(2.565 - 0.0215 * (T - 273.15)) / 1000
