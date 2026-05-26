import numpy as np
from scipy.special import erfc


def bremsstrahlung(energy, temperature, emission_measure):
    kT = temperature
    const = 8.1e-43
    sqrt_term = np.sqrt(8.0 / (9.0 * np.pi * kT))
    exp_term = np.exp(-energy / kT)
    return const * emission_measure * sqrt_term * exp_term / energy


def thermal_spectrum(energy, params):
    T1, log_EM1, T2, log_EM2 = params[:4]
    EM1 = 10 ** log_EM1
    EM2 = 10 ** log_EM2
    spec1 = bremsstrahlung(energy, T1, EM1)
    spec2 = bremsstrahlung(energy, T2, EM2)
    return spec1 + spec2


def power_law(energy, log_normalization, index, e_cut=1000.0):
    normalization = 10 ** log_normalization
    return normalization * (energy ** (-index)) * np.exp(-energy / e_cut)


def nonthermal_spectrum(energy, params):
    log_norm, index = params
    return power_law(energy, log_norm, index)


def combined_model(energy, params):
    T1, log_EM1, T2, log_EM2, log_norm_pl, index_pl = params
    
    thermal = thermal_spectrum(energy, [T1, log_EM1, T2, log_EM2])
    nonthermal = power_law(energy, log_norm_pl, index_pl)
    
    return thermal + nonthermal


def get_param_names():
    return ['T1 (keV)', 'log10(EM1)', 'T2 (keV)', 'log10(EM2)', 
            'log10(norm_pl)', 'index_pl']


def get_default_bounds():
    lower = [0.5, 40.0, 5.0, 40.0, -50.0, 1.0]
    upper = [20.0, 50.0, 50.0, 50.0, -25.0, 8.0]
    return (lower, upper)


def get_initial_guess(energy_centers, counts, energy_widths=None, exposure=1.0):
    lower, upper = get_default_bounds()
    
    if energy_widths is None:
        energy_widths = np.ones_like(energy_centers)
    
    flux = counts / energy_widths / exposure
    
    low_mask = energy_centers < 10.0
    mid_mask = (energy_centers >= 10.0) & (energy_centers < 50.0)
    high_mask = (energy_centers >= 100.0) & (energy_centers < 300.0)
    
    low_energy = np.mean(energy_centers[low_mask]) if np.any(low_mask) else energy_centers[0]
    mid_energy = np.mean(energy_centers[mid_mask]) if np.any(mid_mask) else energy_centers[len(energy_centers)//2]
    
    T1_guess = 10.0
    T1_guess = min(max(T1_guess, lower[0]), upper[0])
    
    T2_guess = 25.0
    T2_guess = min(max(T2_guess, lower[2]), upper[2])
    
    const = 8.1e-43
    low_flux = np.mean(flux[low_mask]) if np.any(low_mask) else flux[0]
    EM1_guess = low_flux * low_energy * np.sqrt(9.0 * np.pi * T1_guess / 8.0) / (const * np.exp(-low_energy / T1_guess))
    log_EM1_guess = np.log10(max(EM1_guess, 1e40))
    log_EM1_guess = min(max(log_EM1_guess, lower[1]), upper[1])
    
    log_EM2_guess = log_EM1_guess - 0.7
    log_EM2_guess = min(max(log_EM2_guess, lower[3]), upper[3])
    
    if np.any(high_mask) and np.sum(flux[high_mask]) > 0:
        high_energy = np.mean(energy_centers[high_mask])
        high_flux = np.mean(flux[high_mask])
        norm_pl_guess = high_flux * (high_energy ** 3.0) * 1e-36
        log_norm_pl_guess = np.log10(max(norm_pl_guess, 1e-50))
    else:
        log_norm_pl_guess = -37.0
    log_norm_pl_guess = min(max(log_norm_pl_guess, lower[4]), upper[4])
    
    index_pl_guess = 2.5
    index_pl_guess = min(max(index_pl_guess, lower[5]), upper[5])
    
    return [T1_guess, log_EM1_guess, T2_guess, log_EM2_guess, log_norm_pl_guess, index_pl_guess]


def calculate_chi2(observed, expected, errors):
    mask = (errors > 0) & (expected > 0) & (observed > 0)
    chi2 = np.sum(((observed[mask] - expected[mask]) / errors[mask]) ** 2)
    dof = np.sum(mask) - 6
    return chi2, dof


def convert_to_physical(params):
    T1, log_EM1, T2, log_EM2, log_norm_pl, index_pl = params
    return {
        'T1': T1,
        'EM1': 10 ** log_EM1,
        'T2': T2,
        'EM2': 10 ** log_EM2,
        'norm_pl': 10 ** log_norm_pl,
        'index_pl': index_pl
    }
