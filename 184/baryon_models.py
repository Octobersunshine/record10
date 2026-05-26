import numpy as np
from scipy.special import i0, i1, k0, k1


class BaryonicModels:
    def __init__(self):
        self.G = 4.30091e-6

    def stellar_disk(self, r, M_disk, R_disk):
        y = r / (2 * R_disk)
        term = i0(y) * k0(y) - i1(y) * k1(y)
        v_disk = np.sqrt(2 * np.pi * self.G * (M_disk * 1e10) * y**2 * term / R_disk)
        return v_disk

    def stellar_bulge(self, r, M_bulge, r_bulge):
        M_enclosed = M_bulge * 1e10 * (1 - np.exp(-r / r_bulge) * (1 + r / r_bulge))
        v_bulge = np.sqrt(self.G * M_enclosed / r)
        return v_bulge

    def hi_gas_disk(self, r, M_hi, R_hi):
        sigma_0 = (M_hi * 1e10) / (2 * np.pi * R_hi**2)
        sigma_gas = sigma_0 * np.exp(-r / R_hi)
        M_enclosed = np.zeros_like(r)
        for i, ri in enumerate(r):
            if ri == 0:
                M_enclosed[i] = 0
            else:
                r_grid = np.linspace(0, ri, 1000)
                sigma_grid = sigma_0 * np.exp(-r_grid / R_hi)
                M_enclosed[i] = 2 * np.pi * np.trapezoid(sigma_grid * r_grid, r_grid)
        v_gas = np.sqrt(self.G * M_enclosed / r)
        return v_gas

    def molecular_gas(self, r, M_h2, R_h2, alpha_h2=1.36):
        sigma_0 = (M_h2 * 1e10 * alpha_h2) / (2 * np.pi * R_h2**2)
        M_enclosed = np.zeros_like(r)
        for i, ri in enumerate(r):
            if ri == 0:
                M_enclosed[i] = 0
            else:
                r_grid = np.linspace(0, ri, 1000)
                sigma_grid = sigma_0 * np.exp(-r_grid / R_h2)
                M_enclosed[i] = 2 * np.pi * np.trapezoid(sigma_grid * r_grid, r_grid)
        v_h2 = np.sqrt(self.G * M_enclosed / r)
        return v_h2

    def total_baryonic_v(self, r, params):
        M_disk, R_disk, M_bulge, r_bulge, M_hi, R_hi, M_h2, R_h2 = params
        
        v_disk = self.stellar_disk(r, M_disk, R_disk)
        v_bulge = self.stellar_bulge(r, M_bulge, r_bulge)
        v_hi = self.hi_gas_disk(r, M_hi, R_hi)
        v_h2 = self.molecular_gas(r, M_h2, R_h2)
        
        v_total = np.sqrt(v_disk**2 + v_bulge**2 + v_hi**2 + v_h2**2)
        return v_total, {'disk': v_disk, 'bulge': v_bulge, 'hi': v_hi, 'h2': v_h2}

    def compute_mass_to_light_ratio(self, color, population='old'):
        if population == 'old':
            ML_b = 1.5 + 2.0 * (color - 0.8)
            ML_v = 1.0 + 1.2 * (color - 0.8)
        elif population == 'young':
            ML_b = 0.3 + 0.5 * (color - 0.6)
            ML_v = 0.2 + 0.4 * (color - 0.6)
        else:
            ML_b = 1.0
            ML_v = 0.8
        
        return {'B': ML_b, 'V': ML_v}

    def stellar_mass_from_luminosity(self, L_b, L_v, color):
        ML = self.compute_mass_to_light_ratio(color)
        M_star_b = L_b * ML['B']
        M_star_v = L_v * ML['V']
        return 0.5 * (M_star_b + M_star_v), ML

    def gas_mass_from_hi(self, hi_flux, distance):
        M_hi = 2.36e5 * hi_flux * distance**2
        return M_hi / 1e10


class MONDModel:
    def __init__(self):
        self.G = 4.30091e-6
        self.a0 = 1.2e-10

    def standard_interpolating_function(self, x):
        return x / np.sqrt(1 + x**2)

    def simple_interpolating_function(self, x):
        return np.where(x < 1, x, 1)

    def mond_v_circular(self, r, baryonic_params, baryon_model, 
                        interpolating='standard'):
        v_baryon, components = baryon_model.total_baryonic_v(r, baryonic_params)
        a_newton = v_baryon**2 / r
        
        a_newton_si = a_newton * 3.24078e-14
        
        if interpolating == 'standard':
            mu = self.standard_interpolating_function(a_newton_si / self.a0)
        else:
            mu = self.simple_interpolating_function(a_newton_si / self.a0)
        
        a_mond = a_newton_si / mu
        a_mond_astro = a_mond / 3.24078e-14
        
        v_mond = np.sqrt(a_mond_astro * r)
        
        return v_mond, v_baryon, components

    def log_prior(self, params, param_names):
        for i, name in enumerate(param_names):
            if 'M_' in name:
                if not (0.1 <= params[i] <= 50):
                    return -np.inf
            elif 'R_' in name or 'r_' in name:
                if not (0.1 <= params[i] <= 30):
                    return -np.inf
        return 0.0

    def log_likelihood(self, params, r, v_obs, v_err, 
                      baryon_model, param_names):
        v_pred, _, _ = self.mond_v_circular(r, params, baryon_model)
        if not np.all(np.isfinite(v_pred)):
            return -np.inf
        chi2 = np.sum(((v_obs - v_pred) / v_err)**2)
        return -0.5 * chi2

    def log_probability(self, params, r, v_obs, v_err, 
                       baryon_model, param_names):
        lp = self.log_prior(params, param_names)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(params, r, v_obs, v_err, 
                                       baryon_model, param_names)


def generate_sample_rotation_curve(r_min=1, r_max=30, n_points=25, noise=5):
    r = np.linspace(r_min, r_max, n_points)
    
    baryon_params = [0.05, 3.0, 0.01, 0.5, 0.015, 8.0, 0.005, 4.0]
    
    dm_params = [0.05, 15.0]
    
    baryon_model = BaryonicModels()
    v_baryon, baryon_components = baryon_model.total_baryonic_v(r, baryon_params)
    
    G = 4.30091e-6
    rho_s, r_s = dm_params
    rho_s_converted = rho_s * 1e9
    x = r / r_s
    M_enclosed_dm = 4 * np.pi * rho_s_converted * r_s**3 * (np.log(1 + x) - x / (1 + x))
    v_dm = np.sqrt(G * M_enclosed_dm / r)
    
    v_total = np.sqrt(v_baryon**2 + v_dm**2)
    v_obs = v_total + np.random.normal(0, noise, n_points)
    v_err = np.full_like(v_obs, noise)
    
    true_params = {
        'baryons': baryon_params,
        'dm': dm_params,
        'baryon_components': baryon_components,
        'v_dm': v_dm,
        'v_baryon': v_baryon
    }
    
    return r, v_obs, v_err, true_params
