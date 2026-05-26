import numpy as np
from spectrum_models import combined_model


def generate_rhessi_energy_bins():
    edges = np.logspace(np.log10(3.0), np.log10(800.0), 25)
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = edges[1:] - edges[:-1]
    return edges, centers, widths


def generate_goes_energy_bins():
    edges = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = edges[1:] - edges[:-1]
    return edges, centers, widths


def generate_drm(energy_centers, energy_widths, response_width=0.1):
    n_bins = len(energy_centers)
    drm = np.zeros((n_bins, n_bins))
    
    for i in range(n_bins):
        for j in range(n_bins):
            sigma = response_width * energy_centers[j]
            drm[i, j] = np.exp(-(energy_centers[i] - energy_centers[j]) ** 2 / (2 * sigma ** 2))
        drm[i, :] /= np.sum(drm[i, :]) if np.sum(drm[i, :]) > 0 else 1.0
    
    return drm


def apply_response(true_flux, drm, energy_widths, exposure=1.0):
    photon_flux = true_flux * energy_widths
    count_rate = np.dot(drm, photon_flux)
    return count_rate * exposure


def simulate_spectrum(true_params, energy_centers, energy_widths, 
                      exposure=1.0, noise=True, response=True,
                      response_width=0.1, poisson=True):
    true_flux = combined_model(energy_centers, true_params)
    
    if response:
        drm = generate_drm(energy_centers, energy_widths, response_width)
        counts = apply_response(true_flux, drm, energy_widths, exposure)
    else:
        counts = true_flux * energy_widths * exposure
    
    if noise and poisson:
        counts = np.random.poisson(np.maximum(counts, 0))
    
    errors = np.sqrt(np.maximum(counts, 1.0))
    
    return counts, errors, true_flux


def load_rhessi_data(filename=None):
    if filename is None:
        print("没有提供RHESSI数据文件，使用模拟数据")
        return None, None, None, None, None
    
    try:
        data = np.loadtxt(filename)
        energy_edges_low = data[:, 0]
        energy_edges_high = data[:, 1]
        counts = data[:, 2]
        errors = data[:, 3] if data.shape[1] > 3 else np.sqrt(np.maximum(counts, 1.0))
        
        energy_centers = (energy_edges_low + energy_edges_high) / 2.0
        energy_widths = energy_edges_high - energy_edges_low
        
        return energy_centers, energy_widths, counts, errors, None
        
    except Exception as e:
        print(f"加载RHESSI数据失败: {e}")
        return None, None, None, None, None


def load_goes_data(filename=None):
    if filename is None:
        print("没有提供GOES数据文件，使用模拟数据")
        return None, None, None, None, None
    
    try:
        data = np.loadtxt(filename)
        channels = data[:, 0] if data.shape[1] > 1 else np.arange(data.shape[0])
        fluxes = data[:, 1]
        errors = data[:, 2] if data.shape[1] > 2 else fluxes * 0.1
        
        goes_energies = {
            1: (0.5, 1.0),
            2: (1.0, 2.0),
            3: (2.0, 4.0),
            4: (4.0, 8.0),
            5: (8.0, 16.0),
        }
        
        energy_centers = []
        energy_widths = []
        for ch in channels:
            el, eh = goes_energies.get(int(ch), (0, 1))
            energy_centers.append((el + eh) / 2.0)
            energy_widths.append(eh - el)
        
        energy_centers = np.array(energy_centers)
        energy_widths = np.array(energy_widths)
        
        return energy_centers, energy_widths, fluxes, errors, None
        
    except Exception as e:
        print(f"加载GOES数据失败: {e}")
        return None, None, None, None, None


def create_demo_data():
    true_params = [8.0, np.log10(1e45), 30.0, np.log10(2e44), np.log10(5e-35), 2.5]
    
    energy_edges, energy_centers, energy_widths = generate_rhessi_energy_bins()
    
    counts, errors, true_flux = simulate_spectrum(
        true_params, energy_centers, energy_widths,
        exposure=100.0, noise=True, response=False,
        response_width=0.15, poisson=True
    )
    
    return {
        'energy_centers': energy_centers,
        'energy_widths': energy_widths,
        'energy_edges': energy_edges,
        'counts': counts,
        'errors': errors,
        'true_flux': true_flux,
        'true_params': true_params
    }
