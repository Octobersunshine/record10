import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate, optimize
from scipy.stats import norm


spectral_lines = [
    {'element': 'Fe', 'ion': 'Fe XVIII', 'wavelength': 10.62, 'energy': 1.167, 'logT_max': 6.8, 'Ab': 1.0},
    {'element': 'Fe', 'ion': 'Fe XIX', 'wavelength': 10.82, 'energy': 1.146, 'logT_max': 6.9, 'Ab': 1.0},
    {'element': 'Fe', 'ion': 'Fe XX', 'wavelength': 12.80, 'energy': 0.969, 'logT_max': 7.0, 'Ab': 1.0},
    {'element': 'Fe', 'ion': 'Fe XXI', 'wavelength': 11.27, 'energy': 1.100, 'logT_max': 7.1, 'Ab': 1.0},
    {'element': 'Fe', 'ion': 'Fe XXII', 'wavelength': 11.77, 'energy': 1.054, 'logT_max': 7.2, 'Ab': 1.0},
    {'element': 'Fe', 'ion': 'Fe XXIII', 'wavelength': 13.26, 'energy': 0.935, 'logT_max': 7.3, 'Ab': 1.0},
    {'element': 'Fe', 'ion': 'Fe XXIV', 'wavelength': 15.18, 'energy': 0.817, 'logT_max': 7.4, 'Ab': 1.0},
    {'element': 'Fe', 'ion': 'Fe XXV', 'wavelength': 1.85, 'energy': 6.70, 'logT_max': 7.5, 'Ab': 1.0},
    {'element': 'Ca', 'ion': 'Ca XIX', 'wavelength': 3.17, 'energy': 3.91, 'logT_max': 7.0, 'Ab': 0.01},
    {'element': 'Ca', 'ion': 'Ca XX', 'wavelength': 3.20, 'energy': 3.87, 'logT_max': 7.1, 'Ab': 0.01},
    {'element': 'O', 'ion': 'O VII', 'wavelength': 21.60, 'energy': 0.574, 'logT_max': 6.3, 'Ab': 0.6},
    {'element': 'O', 'ion': 'O VIII', 'wavelength': 18.97, 'energy': 0.654, 'logT_max': 6.5, 'Ab': 0.6},
    {'element': 'Mg', 'ion': 'Mg XI', 'wavelength': 9.17, 'energy': 1.352, 'logT_max': 6.6, 'Ab': 0.08},
    {'element': 'Mg', 'ion': 'Mg XII', 'wavelength': 8.42, 'energy': 1.472, 'logT_max': 6.7, 'Ab': 0.08},
    {'element': 'Si', 'ion': 'Si XIII', 'wavelength': 6.65, 'energy': 1.864, 'logT_max': 6.8, 'Ab': 0.05},
    {'element': 'Si', 'ion': 'Si XIV', 'wavelength': 6.18, 'energy': 2.006, 'logT_max': 6.9, 'Ab': 0.05},
    {'element': 'S', 'ion': 'S XV', 'wavelength': 5.04, 'energy': 2.460, 'logT_max': 6.9, 'Ab': 0.03},
    {'element': 'S', 'ion': 'S XVI', 'wavelength': 4.73, 'energy': 2.622, 'logT_max': 7.0, 'Ab': 0.03},
    {'element': 'Ar', 'ion': 'Ar XVII', 'wavelength': 3.95, 'energy': 3.139, 'logT_max': 7.0, 'Ab': 0.008},
    {'element': 'Ar', 'ion': 'Ar XVIII', 'wavelength': 3.73, 'energy': 3.324, 'logT_max': 7.1, 'Ab': 0.008},
]


abundance_models = {
    'photospheric': {
        'H': 1.0, 'He': 0.1, 'C': 0.0004, 'N': 0.0001, 'O': 0.0006,
        'Ne': 0.0001, 'Na': 2e-6, 'Mg': 4e-5, 'Al': 3e-6, 'Si': 3e-5,
        'S': 1e-5, 'Ar': 3e-6, 'Ca': 2e-6, 'Fe': 3e-5, 'Ni': 2e-6
    },
    'coronal': {
        'H': 1.0, 'He': 0.1, 'C': 0.004, 'N': 0.001, 'O': 0.006,
        'Ne': 0.001, 'Na': 2e-5, 'Mg': 4e-4, 'Al': 3e-5, 'Si': 3e-4,
        'S': 1e-4, 'Ar': 3e-5, 'Ca': 2e-5, 'Fe': 3e-4, 'Ni': 2e-5
    },
    'flare': {
        'H': 1.0, 'He': 0.1, 'C': 0.002, 'N': 0.0005, 'O': 0.003,
        'Ne': 0.0005, 'Na': 1e-5, 'Mg': 2e-4, 'Al': 1.5e-5, 'Si': 1.5e-4,
        'S': 5e-5, 'Ar': 1.5e-5, 'Ca': 1e-5, 'Fe': 1.5e-4, 'Ni': 1e-5
    },
    'flaring': {
        'H': 1.0, 'He': 0.1, 'C': 0.0008, 'N': 0.0002, 'O': 0.0012,
        'Ne': 0.0002, 'Na': 4e-6, 'Mg': 8e-5, 'Al': 6e-6, 'Si': 6e-5,
        'S': 2e-5, 'Ar': 6e-6, 'Ca': 4e-6, 'Fe': 6e-5, 'Ni': 4e-6
    }
}


def gaussian_response(x, x0, sigma):
    return np.exp(-(x - x0)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def line_emissivity(logT, line):
    logT0 = line['logT_max']
    sigma_T = 0.25
    emiss = np.exp(-(logT - logT0)**2 / (2 * sigma_T**2))
    return emiss


def calculate_line_intensity(logT, EM, line, abundance_model='coronal'):
    abund = abundance_models[abundance_model]
    element_abund = abund.get(line['element'], 1e-5)
    
    emiss = line_emissivity(logT, line)
    
    const = 1.0
    intensity = const * element_abund * line['Ab'] * EM * emiss
    
    return intensity


def synthesize_spectrum(wavelengths, logT_array, DEM_array, 
                        abundance_model='coronal', resolution=0.01):
    spectrum = np.zeros_like(wavelengths)
    
    for logT, dem in zip(logT_array, DEM_array):
        for line in spectral_lines:
            intensity = calculate_line_intensity(logT, dem, line, abundance_model)
            sigma = resolution * line['wavelength'] / 2.355
            spectrum += intensity * gaussian_response(wavelengths, line['wavelength'], sigma)
    
    return spectrum


def add_continuum(wavelengths, spectrum, T_eff, EM):
    energy_kev = 12.398 / wavelengths
    const = 1e-40
    continuum = const * EM * np.sqrt(1.0 / T_eff) * np.exp(-energy_kev / T_eff) / energy_kev
    return spectrum + continuum


def calculate_line_ratios(line_intensities, ratio_pairs):
    ratios = {}
    for name, (line1, line2) in ratio_pairs.items():
        if line_intensities[line2] > 0:
            ratios[name] = line_intensities[line1] / line_intensities[line2]
        else:
            ratios[name] = np.nan
    return ratios


def temperature_from_line_ratio(ratio_name, ratio_value):
    ratio_calibration = {
        'FeXXI/FeXXIII': {'T': [6.8, 7.0, 7.2, 7.4, 7.6], 'ratio': [0.1, 0.5, 2.0, 5.0, 10.0]},
        'CaXIX/FeXXV': {'T': [6.5, 6.8, 7.0, 7.2, 7.5], 'ratio': [0.01, 0.1, 0.5, 2.0, 10.0]},
        'OVII/OVIII': {'T': [5.8, 6.0, 6.2, 6.4, 6.6], 'ratio': [10.0, 5.0, 2.0, 0.8, 0.3]},
    }
    
    if ratio_name not in ratio_calibration:
        return np.nan
    
    cal = ratio_calibration[ratio_name]
    f = interpolate.interp1d(np.log(cal['ratio']), cal['T'], kind='linear', fill_value='extrapolate')
    
    try:
        return f(np.log(ratio_value))
    except:
        return np.nan


class DEMInverter:
    def __init__(self, logT_min=5.5, logT_max=8.0, n_bins=20):
        self.logT_min = logT_min
        self.logT_max = logT_max
        self.n_bins = n_bins
        self.logT_centers = np.linspace(logT_min, logT_max, n_bins)
        self.logT_edges = np.linspace(logT_min - 0.5*(self.logT_centers[1]-self.logT_centers[0]),
                                       logT_max + 0.5*(self.logT_centers[1]-self.logT_centers[0]),
                                       n_bins + 1)
        self.dlogT = np.diff(self.logT_edges)[0]
        
    def construct_response_matrix(self, lines, abundance_model='coronal'):
        n_lines = len(lines)
        n_temp = self.n_bins
        
        RM = np.zeros((n_lines, n_temp))
        
        for i, line in enumerate(lines):
            for j, logT in enumerate(self.logT_centers):
                RM[i, j] = calculate_line_intensity(logT, 1e44, line, abundance_model)
        
        self.response_matrix = RM
        self.lines = lines
        return RM
    
    def invert_regularized(self, line_intensities, errors, lambda_reg=1.0, method='positivity'):
        A = self.response_matrix
        b = np.array(line_intensities)
        sigma = np.array(errors)
        
        A_weighted = A / sigma[:, np.newaxis]
        b_weighted = b / sigma
        
        L = np.eye(self.n_bins)
        L = np.dot(L.T, L)
        
        ATA = np.dot(A_weighted.T, A_weighted) + lambda_reg * L
        ATb = np.dot(A_weighted.T, b_weighted)
        
        if method == 'positivity':
            def objective(x):
                res = np.dot(A_weighted, x) - b_weighted
                return 0.5 * np.sum(res**2) + lambda_reg * np.sum(x**2)
            
            def constraint(x):
                return x
            
            bounds = [(0, None) for _ in range(self.n_bins)]
            
            x0 = np.ones(self.n_bins) * 1e44
            result = optimize.minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
            DEM = result.x
        else:
            DEM = np.linalg.solve(ATA, ATb)
        
        self.DEM = DEM
        self.chi2 = np.sum((np.dot(A, DEM) - b)**2 / sigma**2)
        
        return DEM, self.chi2
    
    def invert_mcmc(self, line_intensities, errors, n_walkers=32, n_steps=5000, burnin=1000):
        import emcee
        
        def log_prior(log_dem):
            if np.any(log_dem < 40) or np.any(log_dem > 55):
                return -np.inf
            
            smoothness = -0.5 * np.sum(np.diff(log_dem)**2) / 0.1
            return smoothness
        
        def log_likelihood(log_dem):
            dem = 10**log_dem
            model = np.dot(self.response_matrix, dem)
            chi2 = np.sum((model - line_intensities)**2 / errors**2)
            return -0.5 * chi2
        
        def log_posterior(log_dem):
            lp = log_prior(log_dem)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(log_dem)
        
        ndim = self.n_bins
        pos = []
        for _ in range(n_walkers):
            p = 44 + np.random.randn(ndim) * 0.5
            pos.append(p)
        pos = np.array(pos)
        
        sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior)
        sampler.run_mcmc(pos, n_steps, progress=True)
        
        samples = sampler.get_chain(discard=burnin, flat=True)
        self.DEM_samples = 10**samples
        self.DEM_median = np.median(self.DEM_samples, axis=0)
        self.DEM_low = np.percentile(self.DEM_samples, 16, axis=0)
        self.DEM_high = np.percentile(self.DEM_samples, 84, axis=0)
        
        return self.DEM_median, self.DEM_low, self.DEM_high
    
    def calculate_total_em(self):
        try:
            trapz = np.trapezoid
        except AttributeError:
            trapz = np.trapz
        
        if hasattr(self, 'DEM_median'):
            return trapz(self.DEM_median, self.logT_centers)
        elif hasattr(self, 'DEM'):
            return trapz(self.DEM, self.logT_centers)
        else:
            return 0
    
    def plot_DEM(self, save_path='dem_plot.png'):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if hasattr(self, 'DEM_samples'):
            ax.fill_between(self.logT_centers, self.DEM_low, self.DEM_high, 
                           alpha=0.3, color='steelblue', label='68% credible interval')
            ax.plot(self.logT_centers, self.DEM_median, 'r-', linewidth=2, label='Median DEM')
        else:
            ax.plot(self.logT_centers, self.DEM, 'r-', linewidth=2, label='DEM')
        
        ax.set_xlabel('log Temperature (K)', fontsize=12)
        ax.set_ylabel('DEM (cm$^{-5}$ K$^{-1}$)', fontsize=12)
        ax.set_yscale('log')
        ax.set_ylim(1e42, 1e47)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_title('Differential Emission Measure', fontsize=14)
        
        total_em = self.calculate_total_em()
        ax.text(0.02, 0.95, f'Total EM = {total_em:.2e} cm$^{-3}$', 
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"DEM plot saved to: {save_path}")
        
        return fig


def get_abundance_factor(element, model1, model2):
    ab1 = abundance_models[model1].get(element, 1e-5)
    ab2 = abundance_models[model2].get(element, 1e-5)
    return ab1 / ab2


def fit_abundances(line_intensities, lines, DEM, logT_array, reference_model='photospheric'):
    elements = list(set([line['element'] for line in lines]))
    abundance_factors = {}
    
    for element in elements:
        element_lines = [i for i, l in enumerate(lines) if l['element'] == element]
        
        if len(element_lines) >= 1:
            model_intensities = []
            for idx in element_lines:
                line = lines[idx]
                intensity = 0
                for logT, dem in zip(logT_array, DEM):
                    intensity += calculate_line_intensity(logT, dem, line, reference_model)
                model_intensities.append(intensity)
            
            obs_intensities = [line_intensities[i] for i in element_lines]
            factor = np.mean(np.array(obs_intensities) / np.array(model_intensities))
            abundance_factors[element] = factor
    
    return abundance_factors


def generate_demo_spectrum(n_wavelength=1000, lambda_min=1.0, lambda_max=25.0):
    wavelengths = np.linspace(lambda_min, lambda_max, n_wavelength)
    
    logT_centers = np.array([6.5, 6.8, 7.0, 7.2, 7.3, 7.4])
    DEM = np.array([1e43, 5e44, 2e45, 8e44, 2e44, 5e43])
    
    spectrum = synthesize_spectrum(wavelengths, logT_centers, DEM, abundance_model='coronal')
    
    T_eff_kev = 2.0
    spectrum = add_continuum(wavelengths, spectrum, T_eff_kev, np.sum(DEM))
    
    return {
        'wavelengths': wavelengths,
        'spectrum': spectrum,
        'logT_centers': logT_centers,
        'DEM': DEM
    }


def identify_lines(wavelengths, spectrum, threshold=0.1):
    from scipy.signal import find_peaks
    
    peaks, properties = find_peaks(spectrum, height=threshold * np.max(spectrum))
    
    identified_lines = []
    for peak_idx in peaks:
        wave_peak = wavelengths[peak_idx]
        for line in spectral_lines:
            if abs(line['wavelength'] - wave_peak) < 0.1:
                identified_lines.append({
                    'line': line,
                    'wavelength_obs': wave_peak,
                    'intensity': spectrum[peak_idx]
                })
                break
    
    return identified_lines
