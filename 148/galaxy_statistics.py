import numpy as np
from scipy.spatial import cKDTree

class StellarMassFunction:
    def __init__(self):
        pass
    
    def compute_smf(self, stellar_masses, box_size, n_bins=15, logM_min=8.0, logM_max=12.0):
        stellar_masses = np.array(stellar_masses)
        logMs = np.log10(stellar_masses)
        
        mass_bins = np.logspace(logM_min, logM_max, n_bins + 1)
        logM_bins = np.log10(mass_bins)
        dlogM = logM_bins[1:] - logM_bins[:-1]
        
        n_galaxies, _ = np.histogram(stellar_masses, bins=mass_bins)
        
        volume = box_size**3
        phi = n_galaxies / (volume * dlogM)
        
        logM_mid = 0.5 * (logM_bins[1:] + logM_bins[:-1])
        
        return logM_mid, phi, mass_bins
    
    def schechter_function(self, logMs, logMs_star=10.7, phi_star=1e-2, alpha=-1.25):
        logMs = np.atleast_1d(logMs)
        
        x = 10**(logMs - logMs_star)
        phi = np.log(10.0) * phi_star * x**(alpha + 1) * np.exp(-x)
        
        return phi


class GalaxyCorrelationFunction:
    def __init__(self):
        pass
    
    def compute_xi_landy_szalay(self, galaxies, box_size, n_bins=20, r_min=0.1, r_max=50.0):
        pos = np.array([g['pos'] for g in galaxies])
        n_data = len(pos)
        
        r_bins = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)
        r_mid = np.sqrt(r_bins[1:] * r_bins[:-1])
        
        tree_data = cKDTree(pos, boxsize=box_size)
        
        DD = np.zeros(n_bins)
        for i in range(n_data):
            distances, _ = tree_data.query(pos[i], k=n_data, distance_upper_bound=r_max)
            distances = distances[np.isfinite(distances)]
            
            for j in range(1, len(distances)):
                r = distances[j]
                bin_idx = np.searchsorted(r_bins, r) - 1
                if 0 <= bin_idx < n_bins:
                    DD[bin_idx] += 1
        
        n_rand = n_data * 5
        pos_rand = np.random.rand(n_rand, 3) * box_size
        
        tree_rand = cKDTree(pos_rand, boxsize=box_size)
        
        DR = np.zeros(n_bins)
        RR = np.zeros(n_bins)
        
        for i in range(n_data):
            distances, _ = tree_rand.query(pos[i], k=n_rand, distance_upper_bound=r_max)
            distances = distances[np.isfinite(distances)]
            
            for j in range(len(distances)):
                r = distances[j]
                bin_idx = np.searchsorted(r_bins, r) - 1
                if 0 <= bin_idx < n_bins:
                    DR[bin_idx] += 1
        
        for i in range(n_rand):
            distances, _ = tree_rand.query(pos_rand[i], k=n_rand, distance_upper_bound=r_max)
            distances = distances[np.isfinite(distances)]
            
            for j in range(1, len(distances)):
                r = distances[j]
                bin_idx = np.searchsorted(r_bins, r) - 1
                if 0 <= bin_idx < n_bins:
                    RR[bin_idx] += 1
        
        DD = DD / (n_data * (n_data - 1) / 2.0)
        DR = DR / (n_data * n_rand)
        RR = RR / (n_rand * (n_rand - 1) / 2.0)
        
        xi = np.zeros(n_bins)
        mask = RR > 0
        xi[mask] = (DD[mask] - 2.0 * DR[mask] + RR[mask]) / RR[mask]
        
        return r_mid, xi, r_bins
    
    def compute_xi_simple(self, galaxies, box_size, n_bins=20, r_min=0.1, r_max=50.0):
        pos = np.array([g['pos'] for g in galaxies])
        n_data = len(pos)
        
        r_bins = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)
        r_mid = np.sqrt(r_bins[1:] * r_bins[:-1])
        
        tree = cKDTree(pos, boxsize=box_size)
        
        pairs = np.zeros(n_bins)
        for i in range(n_data):
            distances = tree.query(pos[i], k=1000, distance_upper_bound=r_max)[0]
            distances = distances[np.isfinite(distances)]
            
            hist, _ = np.histogram(distances[1:], bins=r_bins)
            pairs += hist
        
        n_pairs_total = n_data * (n_data - 1) / 2.0
        
        volume_shells = 4.0 * np.pi / 3.0 * (r_bins[1:]**3 - r_bins[:-1]**3)
        
        n_random = n_pairs_total * volume_shells / box_size**3
        
        xi = np.zeros(n_bins)
        mask = n_random > 0
        xi[mask] = pairs[mask] / n_random[mask] - 1.0
        
        return r_mid, xi, r_bins
    
    def power_law(self, r, r0=5.0, gamma=1.8):
        r = np.atleast_1d(r)
        xi = (r / r0)**(-gamma)
        return xi


def compute_galaxy_bias(galaxies, box_size, dark_matter_pos=None, n_bins=20, r_min=1.0, r_max=30.0):
    from scipy.spatial import cKDTree
    
    if dark_matter_pos is None:
        n_dm = 100000
        dark_matter_pos = np.random.rand(n_dm, 3) * box_size
    
    pos_gal = np.array([g['pos'] for g in galaxies])
    
    r_bins = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)
    r_mid = np.sqrt(r_bins[1:] * r_bins[:-1])
    
    def compute_xi_data(pos_data, box_size, r_bins):
        n = len(pos_data)
        tree = cKDTree(pos_data, boxsize=box_size)
        
        DD = np.zeros(len(r_bins) - 1)
        max_dist = r_bins[-1]
        
        for i in range(n):
            distances = tree.query(pos_data[i], k=min(n, 1000), distance_upper_bound=max_dist)[0]
            distances = distances[np.isfinite(distances)]
            hist, _ = np.histogram(distances[1:], bins=r_bins)
            DD += hist
        
        volume = box_size**3
        volume_shells = 4.0 * np.pi / 3.0 * (r_bins[1:]**3 - r_bins[:-1]**3)
        
        n_random = n * (n - 1) / 2.0 * volume_shells / volume
        
        xi = np.zeros_like(DD)
        mask = n_random > 0
        xi[mask] = DD[mask] / n_random[mask] - 1.0
        
        return xi
    
    xi_gal = compute_xi_data(pos_gal, box_size, r_bins)
    xi_dm = compute_xi_data(dark_matter_pos, box_size, r_bins)
    
    bias = np.zeros_like(xi_gal)
    mask = xi_dm > 0
    bias[mask] = np.sqrt(xi_gal[mask] / xi_dm[mask])
    
    return r_mid, bias, xi_gal, xi_dm
