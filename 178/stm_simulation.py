import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class TipModel:
    """
    Tip state model based on spherical harmonic (multipole) expansion.

    The tunneling conductance is proportional to the sum over multipole
    components: dI/dV ∝ Σ_lm |c_lm|² · (D_lm ρ)², where D_lm is the
    spatial derivative operator corresponding to Y_lm.

    Supported multipoles:
        s       (l=0, m=0)  → ρ
        px      (l=1, m=-1) → ∂ρ/∂x
        py      (l=1, m=1)  → ∂ρ/∂y
        pz      (l=1, m=0)  → ∂ρ/∂z
        dx2y2   (l=2, m=-2) → (∂²/∂x² - ∂²/∂y²)ρ
        dxy     (l=2, m=2)  → ∂²ρ/∂x∂y
        dz2     (l=2, m=0)  → (3∂²/∂z² - ∂²/∂x² - ∂²/∂y²)ρ
        dxz     (l=2, m=-1) → ∂²ρ/∂x∂z
        dyz     (l=2, m=1)  → ∂²ρ/∂y∂z
    """
    VALID_MULTIPOLES = {'s', 'px', 'py', 'pz', 'dx2y2', 'dxy', 'dz2', 'dxz', 'dyz'}

    def __init__(self, **multipoles):
        self.coefficients = {'s': 1.0}
        if multipoles:
            self.coefficients = {}
            for k, v in multipoles.items():
                if k not in self.VALID_MULTIPOLES:
                    raise ValueError(f"Unknown multipole: {k}")
                self.coefficients[k] = float(v)

    def set(self, **multipoles):
        self.coefficients = {}
        for k, v in multipoles.items():
            if k not in self.VALID_MULTIPOLES:
                raise ValueError(f"Unknown multipole: {k}")
            self.coefficients[k] = float(v)

    def normalized(self):
        norm = np.sqrt(sum(v ** 2 for v in self.coefficients.values()))
        if norm == 0:
            return self.coefficients
        return {k: v / norm for k, v in self.coefficients.items()}

    def __repr__(self):
        parts = [f"{k}={v:.2f}" for k, v in self.normalized().items()]
        return f"TipModel({', '.join(parts)})"


class MolecularVibrations:
    """
    Database of molecular vibrational modes for IETS simulation.

    Vibrational energies (in meV) and relative inelastic coupling strengths
    for common molecules on surfaces. Each vibrational mode contributes a
    step in dI/dV and a peak in d²I/dV² at V = ℏω/e.
    """
    VIBRATION_DATABASE = {
        'CO': {
            'name': 'Carbon Monoxide',
            'modes': [
                {'energy': 45, 'intensity': 0.3, 'symmetry': 'frustration', 'label': 'Cu-CO stretch'},
                {'energy': 58, 'intensity': 0.2, 'symmetry': 'frustration', 'label': 'CO bend'},
                {'energy': 265, 'intensity': 0.8, 'symmetry': 'stretch', 'label': 'C-O stretch'},
            ]
        },
        'C2H2': {
            'name': 'Acetylene',
            'modes': [
                {'energy': 85, 'intensity': 0.4, 'symmetry': 'bend', 'label': 'C-C-H bend'},
                {'energy': 125, 'intensity': 0.3, 'symmetry': 'bend', 'label': 'skeletal bend'},
                {'energy': 190, 'intensity': 0.5, 'symmetry': 'stretch', 'label': 'C-C stretch'},
                {'energy': 360, 'intensity': 0.9, 'symmetry': 'stretch', 'label': 'C-H stretch'},
            ]
        },
        'C2H4': {
            'name': 'Ethylene',
            'modes': [
                {'energy': 75, 'intensity': 0.25, 'symmetry': 'bend', 'label': 'CH2 rock'},
                {'energy': 110, 'intensity': 0.35, 'symmetry': 'bend', 'label': 'CH2 scissor'},
                {'energy': 160, 'intensity': 0.4, 'symmetry': 'stretch', 'label': 'C-C stretch'},
                {'energy': 370, 'intensity': 0.85, 'symmetry': 'stretch', 'label': 'C-H stretch'},
            ]
        },
        'C6H6': {
            'name': 'Benzene',
            'modes': [
                {'energy': 50, 'intensity': 0.2, 'symmetry': 'bend', 'label': 'ring torsion'},
                {'energy': 100, 'intensity': 0.3, 'symmetry': 'bend', 'label': 'C-H bend'},
                {'energy': 150, 'intensity': 0.35, 'symmetry': 'bend', 'label': 'ring deformation'},
                {'energy': 190, 'intensity': 0.5, 'symmetry': 'stretch', 'label': 'C-C stretch'},
                {'energy': 380, 'intensity': 0.9, 'symmetry': 'stretch', 'label': 'C-H stretch'},
            ]
        },
        'H2O': {
            'name': 'Water',
            'modes': [
                {'energy': 75, 'intensity': 0.3, 'symmetry': 'libration', 'label': 'librational mode'},
                {'energy': 205, 'intensity': 0.6, 'symmetry': 'bend', 'label': 'H-O-H bend'},
                {'energy': 450, 'intensity': 0.95, 'symmetry': 'stretch', 'label': 'O-H stretch'},
            ]
        },
        'NH3': {
            'name': 'Ammonia',
            'modes': [
                {'energy': 120, 'intensity': 0.35, 'symmetry': 'bend', 'label': 'umbrella mode'},
                {'energy': 165, 'intensity': 0.4, 'symmetry': 'bend', 'label': 'H-N-H bend'},
                {'energy': 410, 'intensity': 0.9, 'symmetry': 'stretch', 'label': 'N-H stretch'},
            ]
        },
        'CH3OH': {
            'name': 'Methanol',
            'modes': [
                {'energy': 40, 'intensity': 0.15, 'symmetry': 'torsion', 'label': 'CH3 torsion'},
                {'energy': 110, 'intensity': 0.3, 'symmetry': 'bend', 'label': 'C-O bend'},
                {'energy': 200, 'intensity': 0.45, 'symmetry': 'stretch', 'label': 'C-O stretch'},
                {'energy': 280, 'intensity': 0.35, 'symmetry': 'stretch', 'label': 'O-H bend'},
                {'energy': 370, 'intensity': 0.8, 'symmetry': 'stretch', 'label': 'C-H stretch'},
                {'energy': 450, 'intensity': 0.9, 'symmetry': 'stretch', 'label': 'O-H stretch'},
            ]
        },
        'O2': {
            'name': 'Oxygen',
            'modes': [
                {'energy': 60, 'intensity': 0.25, 'symmetry': 'frustration', 'label': 'surface-O2 stretch'},
                {'energy': 100, 'intensity': 0.3, 'symmetry': 'stretch', 'label': 'O-O stretch (chemisorbed)'},
                {'energy': 180, 'intensity': 0.7, 'symmetry': 'stretch', 'label': 'O-O stretch (physisorbed)'},
            ]
        },
    }

    ADSORPTION_SITES = ['top', 'bridge', 'hollow', 'tilted']

    @classmethod
    def available_molecules(cls):
        return list(cls.VIBRATION_DATABASE.keys())

    @classmethod
    def get_vibrations(cls, molecule_key, adsorption_site='top', orientation='upright'):
        if molecule_key not in cls.VIBRATION_DATABASE:
            raise ValueError(f"Unknown molecule: {molecule_key}. Available: {cls.available_molecules()}")

        mol = cls.VIBRATION_DATABASE[molecule_key]
        site_factor = {
            'top': 1.0,
            'bridge': 0.8,
            'hollow': 0.6,
            'tilted': 0.9
        }.get(adsorption_site, 1.0)

        orientation_factor = {
            'upright': 1.0,
            'flat': 0.7,
            'tilted': 0.85
        }.get(orientation, 1.0)

        adjusted_modes = []
        for mode in mol['modes']:
            sym_factor = 1.0
            if mode['symmetry'] == 'stretch' and orientation == 'flat':
                sym_factor = 0.5
            if mode['symmetry'] == 'frustration' and adsorption_site == 'hollow':
                sym_factor = 1.5

            adjusted_intensity = mode['intensity'] * site_factor * orientation_factor * sym_factor
            adjusted_modes.append({
                'energy': mode['energy'],
                'intensity': adjusted_intensity,
                'symmetry': mode['symmetry'],
                'label': mode['label']
            })

        return {
            'key': molecule_key,
            'name': mol['name'],
            'modes': adjusted_modes,
            'adsorption_site': adsorption_site,
            'orientation': orientation
        }


class IETSSimulator:
    """
    Inelastic Electron Tunneling Spectroscopy (IETS) simulator.

    Physical model:
    - Elastic channel: always open, contributes constant background
    - Inelastic channel: opens when eV ≥ ℏω, gives step in dI/dV
    - Total conductance: dI/dV = σ_elastic + Σ σ_inelastic * f(eV - ℏω)

    where f is the Fermi broadening function (thermal smearing).
    """

    def __init__(self, kT=0.025, modulation_amplitude=0.01):
        self.kT = kT
        self.mod_amp = modulation_amplitude

    def _fermi_broadened_step(self, V, V0, intensity):
        exp_arg = (V - V0) / self.kT
        exp_arg = np.clip(exp_arg, -100, 100)
        return intensity / (1.0 + np.exp(-exp_arg))

    def calculate_dIdV(self, voltages, molecule_info, elastic_background=1.0, inelastic_coupling=0.05):
        dIdV = np.ones_like(voltages) * elastic_background

        for mode in molecule_info['modes']:
            V_mode = mode['energy'] / 1000.0
            inelastic_step = inelastic_coupling * mode['intensity']
            dIdV += self._fermi_broadened_step(voltages, V_mode, inelastic_step)
            dIdV += self._fermi_broadened_step(-voltages, V_mode, inelastic_step)

        return dIdV

    def calculate_d2IdV2(self, voltages, dIdV=None, molecule_info=None,
                         elastic_background=1.0, inelastic_coupling=0.05):
        if dIdV is None:
            dIdV = self.calculate_dIdV(voltages, molecule_info, elastic_background, inelastic_coupling)

        d2IdV2 = np.gradient(dIdV, voltages)
        return d2IdV2

    def add_noise(self, spectrum, noise_level=0.002):
        return spectrum + noise_level * np.random.randn(len(spectrum))

    def simulate_experimental(self, voltages, molecule_info, elastic_background=1.0,
                              inelastic_coupling=0.05, noise_level=0.003):
        dIdV_clean = self.calculate_dIdV(voltages, molecule_info, elastic_background, inelastic_coupling)
        dIdV_noisy = self.add_noise(dIdV_clean, noise_level)
        d2IdV2_clean = self.calculate_d2IdV2(voltages, dIdV_clean)
        d2IdV2_noisy = self.calculate_d2IdV2(voltages, dIdV_noisy)
        return dIdV_clean, dIdV_noisy, d2IdV2_clean, d2IdV2_noisy

    def compare_molecules(self, voltages, molecule_keys, **kwargs):
        results = {}
        for key in molecule_keys:
            mol_info = MolecularVibrations.get_vibrations(key)
            dIdV_clean, dIdV_noisy, d2_clean, d2_noisy = self.simulate_experimental(
                voltages, mol_info, **kwargs)
            results[key] = {
                'mol_info': mol_info,
                'dIdV_clean': dIdV_clean,
                'dIdV_noisy': dIdV_noisy,
                'd2IdV2_clean': d2_clean,
                'd2IdV2_noisy': d2_noisy
            }
        return results

    def compare_adsorption_configs(self, voltages, molecule_key, sites, orientations, **kwargs):
        results = {}
        for site in sites:
            for orient in orientations:
                mol_info = MolecularVibrations.get_vibrations(molecule_key, site, orient)
                dIdV_clean, dIdV_noisy, d2_clean, d2_noisy = self.simulate_experimental(
                    voltages, mol_info, **kwargs)
                config_key = f"{site}_{orient}"
                results[config_key] = {
                    'mol_info': mol_info,
                    'dIdV_clean': dIdV_clean,
                    'dIdV_noisy': dIdV_noisy,
                    'd2IdV2_clean': d2_clean,
                    'd2IdV2_noisy': d2_noisy
                }
        return results

    def identify_molecule(self, voltages, experimental_d2IdV2, molecule_keys=None,
                          noise_floor=0.001):
        if molecule_keys is None:
            molecule_keys = MolecularVibrations.available_molecules()

        best_score = -np.inf
        best_key = None
        scores = {}

        for key in molecule_keys:
            mol_info = MolecularVibrations.get_vibrations(key)
            d2_pred = self.calculate_d2IdV2(voltages, molecule_info=mol_info)

            exp_norm = experimental_d2IdV2 / (np.max(np.abs(experimental_d2IdV2)) + 1e-10)
            pred_norm = d2_pred / (np.max(np.abs(d2_pred)) + 1e-10)

            correlation = np.corrcoef(exp_norm, pred_norm)[0, 1] if np.std(exp_norm) > 1e-5 else -1
            scores[key] = correlation

            if correlation > best_score:
                best_score = correlation
                best_key = key

        return best_key, best_score, scores


class STMSimulator:
    def __init__(self, lattice_constant=2.8, grid_size=100, scan_height=1.5, 
                 work_function=4.0, hbar=1.0545718e-34, m_e=9.10938356e-31, eV_to_J=1.602176634e-19):
        self.lattice_constant = lattice_constant
        self.grid_size = grid_size
        self.scan_height = scan_height
        self.work_function = work_function
        self.hbar = hbar
        self.m_e = m_e
        self.eV_to_J = eV_to_J
        self.atoms = []
        
        kappa = np.sqrt(2 * m_e * work_function * eV_to_J) / hbar
        self.kappa = kappa * 1e-10
        
    def add_atom(self, x, y, z, species='Si', orbital='s', weight=1.0):
        self.atoms.append({
            'x': x, 'y': y, 'z': z,
            'species': species,
            'orbital': orbital,
            'weight': weight
        })
    
    def generate_square_lattice(self, nx, ny, surface='100'):
        a = self.lattice_constant
        for i in range(nx):
            for j in range(ny):
                x = i * a
                y = j * a
                z = 0.0
                self.add_atom(x, y, z)
    
    def generate_hexagonal_lattice(self, nx, ny):
        a = self.lattice_constant
        for i in range(nx):
            for j in range(ny):
                x = i * a
                y = j * a * np.sqrt(3) / 2
                if j % 2 == 1:
                    x += a / 2
                self.add_atom(x, y, 0.0)
    
    def generate_honeycomb_lattice(self, nx, ny):
        a = self.lattice_constant
        for i in range(nx):
            for j in range(ny):
                x1 = i * a * np.sqrt(3)
                y1 = j * a * 1.5
                self.add_atom(x1, y1, 0.0)
                x2 = x1 + a * np.sqrt(3) / 2
                y2 = y1 + a * 0.5
                self.add_atom(x2, y2, 0.0)
    
    def _s_orbital_amplitude(self, r, kappa):
        return np.exp(-kappa * r) / r
    
    def _pz_orbital_amplitude(self, r, dz, kappa):
        return (dz / r) * np.exp(-kappa * r) / r**2
    
    def _px_orbital_amplitude(self, r, dx, kappa):
        return (dx / r) * np.exp(-kappa * r) / r**2
    
    def _py_orbital_amplitude(self, r, dy, kappa):
        return (dy / r) * np.exp(-kappa * r) / r**2
    
    def _dx2y2_orbital_amplitude(self, r, dx, dy, kappa):
        return ((dx**2 - dy**2) / r**2) * np.exp(-kappa * r) / r**2
    
    def _dxy_orbital_amplitude(self, r, dx, dy, kappa):
        return ((2 * dx * dy) / r**2) * np.exp(-kappa * r) / r**2
    
    def _dz2_orbital_amplitude(self, r, dz, kappa):
        return ((3 * dz**2 - r**2) / r**2) * np.exp(-kappa * r) / r**2
    
    def ldos_contribution(self, X, Y, z, atom, bias_voltage):
        dx = X - atom['x']
        dy = Y - atom['y']
        dz = z - atom['z']
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        
        r[r < 0.1] = 0.1
        
        phi_eff = self.work_function - 0.5 * bias_voltage
        phi_eff = max(phi_eff, 0.1)
        kappa_eff = self.kappa * np.sqrt(phi_eff / self.work_function)
        
        orbital = atom['orbital']
        if orbital == 's':
            amplitude = self._s_orbital_amplitude(r, kappa_eff)
        elif orbital == 'pz':
            amplitude = self._pz_orbital_amplitude(r, dz, kappa_eff)
        elif orbital == 'px':
            amplitude = self._px_orbital_amplitude(r, dx, kappa_eff)
        elif orbital == 'py':
            amplitude = self._py_orbital_amplitude(r, dy, kappa_eff)
        elif orbital == 'dx2y2':
            amplitude = self._dx2y2_orbital_amplitude(r, dx, dy, kappa_eff)
        elif orbital == 'dxy':
            amplitude = self._dxy_orbital_amplitude(r, dx, dy, kappa_eff)
        elif orbital == 'dz2':
            amplitude = self._dz2_orbital_amplitude(r, dz, kappa_eff)
        else:
            amplitude = self._s_orbital_amplitude(r, kappa_eff)
        
        kT = 0.025
        exp_factor = np.exp(bias_voltage / kT)
        fermi_factor = (1.0 / kT) * exp_factor / (1.0 + exp_factor)**2
        
        return atom['weight'] * np.abs(amplitude)**2 * fermi_factor
    
    def calculate_ldos(self, bias_voltage, scan_height=None):
        if scan_height is None:
            scan_height = self.scan_height
        
        if not self.atoms:
            raise ValueError("No atoms in the system. Add atoms first.")
        
        xs = np.array([atom['x'] for atom in self.atoms])
        ys = np.array([atom['y'] for atom in self.atoms])
        
        margin = self.lattice_constant * 2
        x_min, x_max = xs.min() - margin, xs.max() + margin
        y_min, y_max = ys.min() - margin, ys.max() + margin
        
        x = np.linspace(x_min, x_max, self.grid_size)
        y = np.linspace(y_min, y_max, self.grid_size)
        X, Y = np.meshgrid(x, y)
        
        ldos = np.zeros_like(X)
        
        for atom in self.atoms:
            ldos += self.ldos_contribution(X, Y, scan_height, atom, bias_voltage)
        
        return X, Y, ldos
    
    def constant_current_mode(self, current_setpoint, bias_voltage, z_range=2.0, max_iter=30, tol=1e-5):
        xs = np.array([atom['x'] for atom in self.atoms])
        ys = np.array([atom['y'] for atom in self.atoms])
        
        margin = self.lattice_constant * 2
        x_min, x_max = xs.min() - margin, xs.max() + margin
        y_min, y_max = ys.min() - margin, ys.max() + margin
        
        x = np.linspace(x_min, x_max, self.grid_size)
        y = np.linspace(y_min, y_max, self.grid_size)
        X, Y = np.meshgrid(x, y)
        
        z_topography = np.zeros_like(X)
        z_low = self.scan_height - z_range / 2
        z_high = self.scan_height + z_range / 2
        
        def calculate_total_ldos(z):
            total = np.zeros_like(X)
            for atom in self.atoms:
                total += self.ldos_contribution(X, Y, z, atom, bias_voltage)
            return total
        
        z_mid = (z_low + z_high) / 2
        for _ in range(max_iter):
            ldos_mid = calculate_total_ldos(z_mid)
            diff = ldos_mid - current_setpoint
            
            mask_near_zero = np.abs(diff) < tol
            
            ldos_low = calculate_total_ldos(z_low)
            ldos_high = calculate_total_ldos(z_high)
            
            mask_low = (ldos_low - current_setpoint) * diff < 0
            mask_high = ~mask_low & ~mask_near_zero
            
            z_high = np.where(mask_low, z_mid, z_high)
            z_low = np.where(mask_high, z_mid, z_low)
            
            z_mid_new = (z_low + z_high) / 2
            
            if np.max(np.abs(z_mid_new - z_mid)) < tol:
                break
            z_mid = z_mid_new
        
        z_topography = z_mid
        
        return X, Y, z_topography
    
    def simulate_dIdV(self, bias_min, bias_max, num_points=20, scan_height=None):
        if scan_height is None:
            scan_height = self.scan_height
        
        biases = np.linspace(bias_min, bias_max, num_points)
        dIdV_maps = []
        
        for bias in biases:
            _, _, ldos = self.calculate_ldos(bias, scan_height)
            dIdV_maps.append(ldos)
        
        return biases, np.array(dIdV_maps)
    
    def calculate_tunneling_current(self, bias_voltage, scan_height=None, num_energies=100):
        if scan_height is None:
            scan_height = self.scan_height
        
        if not self.atoms:
            raise ValueError("No atoms in the system. Add atoms first.")
        
        xs = np.array([atom['x'] for atom in self.atoms])
        ys = np.array([atom['y'] for atom in self.atoms])
        
        margin = self.lattice_constant * 2
        x_min, x_max = xs.min() - margin, xs.max() + margin
        y_min, y_max = ys.min() - margin, ys.max() + margin
        
        x = np.linspace(x_min, x_max, self.grid_size)
        y = np.linspace(y_min, y_max, self.grid_size)
        X, Y = np.meshgrid(x, y)
        
        kT = 0.025
        
        if bias_voltage >= 0:
            E_min = 0.0
            E_max = bias_voltage
        else:
            E_min = bias_voltage
            E_max = 0.0
        
        energies = np.linspace(E_min, E_max, num_energies)
        dE = energies[1] - energies[0] if num_energies > 1 else 0.001
        
        current = np.zeros_like(X)
        
        for E in energies:
            fermi_diff = 1.0 / (1.0 + np.exp(E / kT)) - 1.0 / (1.0 + np.exp((E - bias_voltage) / kT))
            
            ldos_E = np.zeros_like(X)
            for atom in self.atoms:
                ldos_E += self.ldos_contribution(X, Y, scan_height, atom, E)
            
            current += ldos_E * fermi_diff * dE
        
        return X, Y, current
    
    def calculate_sts_spectrum(self, x_pos, y_pos, bias_min, bias_max, num_points=50, scan_height=None):
        if scan_height is None:
            scan_height = self.scan_height
        
        if not self.atoms:
            raise ValueError("No atoms in the system. Add atoms first.")
        
        biases = np.linspace(bias_min, bias_max, num_points)
        dIdV_spectrum = np.zeros(num_points)
        
        X_pt = np.array([[x_pos]])
        Y_pt = np.array([[y_pos]])
        
        for i, bias in enumerate(biases):
            ldos = np.zeros((1, 1))
            for atom in self.atoms:
                ldos += self.ldos_contribution(X_pt, Y_pt, scan_height, atom, bias)
            dIdV_spectrum[i] = ldos[0, 0]
        
        return biases, dIdV_spectrum
    
    def extract_line_profile(self, X, Y, data, x_start, y_start, x_end, y_end, num_points=100):
        t = np.linspace(0, 1, num_points)
        x_line = x_start + t * (x_end - x_start)
        y_line = y_start + t * (y_end - y_start)
        
        x_flat = X.ravel()
        y_flat = Y.ravel()
        data_flat = data.ravel()
        
        distances = np.sqrt((x_flat[:, None] - x_line[None, :])**2 + 
                           (y_flat[:, None] - y_line[None, :])**2)
        
        nearest_indices = np.argmin(distances, axis=0)
        profile = data_flat[nearest_indices]
        distance_along_line = t * np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
        
        return distance_along_line, profile
    
    def add_noise(self, data, noise_level=0.02):
        max_val = np.max(np.abs(data))
        noise = noise_level * max_val * np.random.randn(*data.shape)
        return data + noise
    
    def fft_analysis(self, data):
        fft_data = np.fft.fft2(data)
        fft_shifted = np.fft.fftshift(fft_data)
        magnitude = np.abs(fft_shifted)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        ny, nx = data.shape
        freqs_x = np.fft.fftshift(np.fft.fftfreq(nx))
        freqs_y = np.fft.fftshift(np.fft.fftfreq(ny))
        FX, FY = np.meshgrid(freqs_x, freqs_y)
        return FX, FY, magnitude_db

    def _atomic_ldos_derivatives(self, X, Y, z, atom, bias_voltage):
        dx = X - atom['x']
        dy = Y - atom['y']
        dz = z - atom['z']
        r = np.sqrt(dx**2 + dy**2 + dz**2)

        r_safe = np.where(r < 0.1, 0.1, r)
        inv_r = 1.0 / r_safe

        phi_eff = self.work_function - 0.5 * bias_voltage
        phi_eff = max(phi_eff, 0.1)
        kappa_eff = self.kappa * np.sqrt(phi_eff / self.work_function)

        kT = 0.025
        exp_factor = np.exp(bias_voltage / kT)
        fermi_factor = (1.0 / kT) * exp_factor / (1.0 + exp_factor)**2

        g = np.exp(-2.0 * kappa_eff * r_safe) * inv_r**2
        rho = atom['weight'] * g * fermi_factor

        gp_over_g = -(2.0 * kappa_eff + 2.0 * inv_r)
        gpp_over_g = (2.0 * kappa_eff + 2.0 * inv_r)**2 + 2.0 * inv_r**2

        drho_dx = rho * gp_over_g * dx * inv_r
        drho_dy = rho * gp_over_g * dy * inv_r
        drho_dz = rho * gp_over_g * dz * inv_r

        dx_r = dx * inv_r
        dy_r = dy * inv_r
        dz_r = dz * inv_r

        d2rho_dx2 = rho * (gpp_over_g * dx_r**2 + gp_over_g * (inv_r - dx_r**2 * inv_r))
        d2rho_dy2 = rho * (gpp_over_g * dy_r**2 + gp_over_g * (inv_r - dy_r**2 * inv_r))
        d2rho_dz2 = rho * (gpp_over_g * dz_r**2 + gp_over_g * (inv_r - dz_r**2 * inv_r))
        d2rho_dxdy = rho * (gpp_over_g * dx_r * dy_r + gp_over_g * (-dx_r * dy_r * inv_r))
        d2rho_dxdz = rho * (gpp_over_g * dx_r * dz_r + gp_over_g * (-dx_r * dz_r * inv_r))
        d2rho_dydz = rho * (gpp_over_g * dy_r * dz_r + gp_over_g * (-dy_r * dz_r * inv_r))

        return (rho, drho_dx, drho_dy, drho_dz,
                d2rho_dx2, d2rho_dy2, d2rho_dz2,
                d2rho_dxdy, d2rho_dxdz, d2rho_dydz)

    def calculate_ldos_with_tip(self, bias_voltage, tip_model=None, scan_height=None):
        if scan_height is None:
            scan_height = self.scan_height

        if tip_model is None:
            return self.calculate_ldos(bias_voltage, scan_height)

        if not self.atoms:
            raise ValueError("No atoms in the system. Add atoms first.")

        xs = np.array([atom['x'] for atom in self.atoms])
        ys = np.array([atom['y'] for atom in self.atoms])

        margin = self.lattice_constant * 2
        x_min, x_max = xs.min() - margin, xs.max() + margin
        y_min, y_max = ys.min() - margin, ys.max() + margin

        x = np.linspace(x_min, x_max, self.grid_size)
        y = np.linspace(y_min, y_max, self.grid_size)
        X, Y = np.meshgrid(x, y)

        rho = np.zeros_like(X)
        drho_dx = np.zeros_like(X)
        drho_dy = np.zeros_like(X)
        drho_dz = np.zeros_like(X)
        d2rho_dx2 = np.zeros_like(X)
        d2rho_dy2 = np.zeros_like(X)
        d2rho_dz2 = np.zeros_like(X)
        d2rho_dxdy = np.zeros_like(X)
        d2rho_dxdz = np.zeros_like(X)
        d2rho_dydz = np.zeros_like(X)

        for atom in self.atoms:
            (rh, dxp, dyp, dzp,
             d2x2, d2y2, d2z2,
             d2xy, d2xz, d2yz) = self._atomic_ldos_derivatives(
                X, Y, scan_height, atom, bias_voltage)
            rho += rh
            drho_dx += dxp
            drho_dy += dyp
            drho_dz += dzp
            d2rho_dx2 += d2x2
            d2rho_dy2 += d2y2
            d2rho_dz2 += d2z2
            d2rho_dxdy += d2xy
            d2rho_dxdz += d2xz
            d2rho_dydz += d2yz

        coeffs = tip_model.normalized()

        tip_response = np.zeros_like(X)

        if 's' in coeffs:
            tip_response += (coeffs['s'] * rho) ** 2
        if 'px' in coeffs:
            tip_response += (coeffs['px'] * drho_dx) ** 2
        if 'py' in coeffs:
            tip_response += (coeffs['py'] * drho_dy) ** 2
        if 'pz' in coeffs:
            tip_response += (coeffs['pz'] * drho_dz) ** 2
        if 'dx2y2' in coeffs:
            tip_response += (coeffs['dx2y2'] * (d2rho_dx2 - d2rho_dy2)) ** 2
        if 'dxy' in coeffs:
            tip_response += (coeffs['dxy'] * d2rho_dxdy) ** 2
        if 'dz2' in coeffs:
            tip_response += (coeffs['dz2'] * (3.0 * d2rho_dz2 - d2rho_dx2 - d2rho_dy2)) ** 2
        if 'dxz' in coeffs:
            tip_response += (coeffs['dxz'] * d2rho_dxdz) ** 2
        if 'dyz' in coeffs:
            tip_response += (coeffs['dyz'] * d2rho_dydz) ** 2

        image = np.sqrt(tip_response + 1e-20)

        return X, Y, image


def plot_stm_image(X, Y, data, title="STM Image", cmap='viridis', 
                   xlabel='X (Å)', ylabel='Y (Å)', clabel='LDOS (a.u.)'):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.pcolormesh(X, Y, data, cmap=cmap, shading='auto')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label=clabel)
    plt.tight_layout()
    return fig, ax


def plot_3d_stm(X, Y, data, title="STM 3D View", cmap=cm.viridis):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, data, cmap=cmap,
                           linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel('X (Å)', fontsize=10)
    ax.set_ylabel('Y (Å)', fontsize=10)
    ax.set_zlabel('Z (Å)', fontsize=10)
    ax.set_title(title, fontsize=14)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Height (Å)')
    plt.tight_layout()
    return fig, ax


def plot_dIdV_spectrum(biases, dIdV_maps, x_idx, y_idx, title="dI/dV Spectrum"):
    fig, ax = plt.subplots(figsize=(8, 6))
    spectrum = dIdV_maps[:, y_idx, x_idx]
    ax.plot(biases, spectrum, 'b-o', linewidth=2, markersize=4)
    ax.set_xlabel('Bias Voltage (V)', fontsize=12)
    ax.set_ylabel('dI/dV (a.u.)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_line_profile(distance, profile, title="STM Line Profile", 
                      xlabel='Distance (Å)', ylabel='LDOS (a.u.)'):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(distance, profile, 'b-', linewidth=2)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_tip_comparison(X, Y, images, titles, ncols=3, cmap='viridis',
                        super_title="Tip Shape Effect on STM Images"):
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes.reshape(-1, 1)
    for idx, (img, title) in enumerate(zip(images, titles)):
        ax = axes[idx // ncols, idx % ncols]
        vmax = np.max(img) if np.max(img) > 0 else 1.0
        im = ax.pcolormesh(X, Y, img, cmap=cmap, shading='auto', vmin=0, vmax=vmax)
        ax.set_xlabel('X (Å)', fontsize=10)
        ax.set_ylabel('Y (Å)', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='a.u.', shrink=0.8)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis('off')
    fig.suptitle(super_title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig, axes


def plot_tip_derivative_components(X, Y, derivatives, titles, super_title="Derivative Components"):
    ncols = 3
    n = len(derivatives)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes.reshape(-1, 1)
    for idx, (deriv, title) in enumerate(zip(derivatives, titles)):
        ax = axes[idx // ncols, idx % ncols]
        vmax = np.max(np.abs(deriv))
        if vmax == 0:
            vmax = 1.0
        im = ax.pcolormesh(X, Y, deriv, cmap='RdBu_r', shading='auto',
                           vmin=-vmax, vmax=vmax)
        ax.set_xlabel('X (Å)', fontsize=10)
        ax.set_ylabel('Y (Å)', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, shrink=0.8)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis('off')
    fig.suptitle(super_title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig, axes


def plot_iets_spectrum(voltages, dIdV_clean, dIdV_noisy, d2_clean, d2_noisy,
                       molecule_info, title="IETS Spectrum"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    mol_name = molecule_info.get('name', 'Unknown')
    site = molecule_info.get('adsorption_site', 'top')
    orient = molecule_info.get('orientation', 'upright')

    ax1.plot(voltages * 1000, dIdV_clean, 'r-', linewidth=2, label='Clean')
    ax1.plot(voltages * 1000, dIdV_noisy, 'b.', markersize=2, alpha=0.6, label='Noisy')
    ax1.set_xlabel('Bias Voltage (mV)', fontsize=12)
    ax1.set_ylabel('dI/dV (a.u.)', fontsize=12)
    ax1.set_title(f'dI/dV Spectrum — {mol_name} ({site}, {orient})', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(voltages * 1000, d2_clean * 1000, 'r-', linewidth=2, label='Clean')
    ax2.plot(voltages * 1000, d2_noisy * 1000, 'b.', markersize=2, alpha=0.6, label='Noisy')
    ax2.set_xlabel('Bias Voltage (mV)', fontsize=12)
    ax2.set_ylabel('d²I/dV² (a.u. ×10⁻³)', fontsize=12)
    ax2.set_title(f'd²I/dV² Spectrum — {mol_name} ({site}, {orient})', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    for mode in molecule_info['modes']:
        ax2.axvline(x=mode['energy'], color='gray', linestyle='--', alpha=0.5)
        ax2.annotate(mode['label'], xy=(mode['energy'], 0), fontsize=7,
                    rotation=90, va='bottom', alpha=0.7)

    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_iets_molecule_comparison(voltages, results, ncols=3, super_title="IETS Molecule Identification"):
    keys = list(results.keys())
    n = len(keys)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes.reshape(-1, 1)

    for idx, key in enumerate(keys):
        ax = axes[idx // ncols, idx % ncols]
        res = results[key]
        mol = res['mol_info']

        ax.plot(voltages * 1000, res['d2IdV2_noisy'] * 1000, 'b.', markersize=2, alpha=0.5)
        ax.plot(voltages * 1000, res['d2IdV2_clean'] * 1000, 'r-', linewidth=1.5)

        for mode in mol['modes']:
            ax.axvline(x=mode['energy'], color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
            ax.annotate(f"{mode['label']} ({mode['energy']}meV)",
                       xy=(mode['energy'], ax.get_ylim()[1] * 0.9),
                       fontsize=6, rotation=90, va='top', ha='right', alpha=0.7)

        ax.set_xlabel('Bias (mV)', fontsize=9)
        ax.set_ylabel('d²I/dV²', fontsize=9)
        ax.set_title(f"{mol['name']} ({key})", fontsize=10)
        ax.grid(True, alpha=0.2)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis('off')

    fig.suptitle(super_title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig, axes


def plot_iets_adsorption_comparison(voltages, results, title="Adsorption Configuration Effect"):
    keys = list(results.keys())
    n = len(keys)
    colors = plt.cm.viridis(np.linspace(0, 1, n))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for idx, key in enumerate(keys):
        res = results[key]
        ax1.plot(voltages * 1000, res['dIdV_clean'], color=colors[idx],
                linewidth=2, label=key)
        ax2.plot(voltages * 1000, res['d2IdV2_clean'] * 1000, color=colors[idx],
                linewidth=2, label=key)

    mol_name = results[keys[0]]['mol_info']['name']
    ax1.set_xlabel('Bias (mV)', fontsize=12)
    ax1.set_ylabel('dI/dV (a.u.)', fontsize=12)
    ax1.set_title(f'dI/dV — {mol_name}', fontsize=13)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Bias (mV)', fontsize=12)
    ax2.set_ylabel('d²I/dV² (a.u. ×10⁻³)', fontsize=12)
    ax2.set_title(f'd²I/dV² — {mol_name}', fontsize=13)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, (ax1, ax2)


def main():
    print("STM Simulation based on Tersoff-Hamann Approximation")
    print("=" * 60)
    
    print("\n1. Square Lattice Simulation...")
    sim = STMSimulator(lattice_constant=3.0, grid_size=80, scan_height=1.5)
    sim.generate_square_lattice(5, 5)
    print(f"   Total atoms: {len(sim.atoms)}")
    
    print("   Calculating LDOS at +1.5 V...")
    X, Y, ldos_pos = sim.calculate_ldos(bias_voltage=1.5)
    plot_stm_image(X, Y, ldos_pos, title="STM LDOS (Bias = +1.5 V) - Square Lattice")
    plt.savefig('stm_ldos_square_positive.png', dpi=150)
    
    print("   Calculating LDOS at -1.0 V...")
    X, Y, ldos_neg = sim.calculate_ldos(bias_voltage=-1.0)
    plot_stm_image(X, Y, ldos_neg, title="STM LDOS (Bias = -1.0 V) - Square Lattice")
    plt.savefig('stm_ldos_square_negative.png', dpi=150)
    
    print("   Calculating constant current topography...")
    current_setpoint = 0.05
    X, Y, topography = sim.constant_current_mode(
        current_setpoint=current_setpoint,
        bias_voltage=1.0,
        z_range=1.5
    )
    plot_stm_image(X, Y, topography, title="STM Constant Current Topography - Square Lattice", 
                   cmap='terrain', clabel='Height (Å)')
    plt.savefig('stm_topography_square.png', dpi=150)
    
    plot_3d_stm(X, Y, topography, title="STM Topography 3D View - Square Lattice")
    plt.savefig('stm_3d_square.png', dpi=150)
    
    print("   Extracting line profile...")
    x_min, x_max = X.min(), X.max()
    y_mid = (Y.min() + Y.max()) / 2
    dist, prof = sim.extract_line_profile(X, Y, ldos_pos, x_min, y_mid, x_max, y_mid)
    plot_line_profile(dist, prof, title="LDOS Line Profile - Square Lattice")
    plt.savefig('stm_line_profile_square.png', dpi=150)
    
    print("\n2. Hexagonal Lattice Simulation...")
    sim2 = STMSimulator(lattice_constant=2.5, grid_size=80, scan_height=1.2)
    sim2.generate_hexagonal_lattice(6, 5)
    print(f"   Total atoms: {len(sim2.atoms)}")
    
    X2, Y2, ldos_hex = sim2.calculate_ldos(bias_voltage=1.0)
    plot_stm_image(X2, Y2, ldos_hex, title="STM LDOS (Bias = +1.0 V) - Hexagonal Lattice")
    plt.savefig('stm_ldos_hexagonal.png', dpi=150)
    
    print("\n3. Honeycomb Lattice (Graphene-like) Simulation...")
    sim3 = STMSimulator(lattice_constant=1.42, grid_size=80, scan_height=1.0)
    sim3.generate_honeycomb_lattice(5, 6)
    print(f"   Total atoms: {len(sim3.atoms)}")
    
    X3, Y3, ldos_honey = sim3.calculate_ldos(bias_voltage=0.5)
    plot_stm_image(X3, Y3, ldos_honey, title="STM LDOS (Bias = +0.5 V) - Honeycomb Lattice")
    plt.savefig('stm_ldos_honeycomb.png', dpi=150)
    
    print("   Adding noise to simulate experimental image...")
    ldos_honey_noisy = sim3.add_noise(ldos_honey, noise_level=0.03)
    plot_stm_image(X3, Y3, ldos_honey_noisy, title="STM LDOS with Noise - Honeycomb Lattice")
    plt.savefig('stm_ldos_honeycomb_noisy.png', dpi=150)
    
    print("\n4. Mixed Orbitals Simulation...")
    sim4 = STMSimulator(lattice_constant=3.0, grid_size=80, scan_height=1.5)
    sim4.add_atom(0, 0, 0, orbital='s', weight=1.0)
    sim4.add_atom(3, 0, 0, orbital='pz', weight=1.5)
    sim4.add_atom(0, 3, 0, orbital='dx2y2', weight=1.2)
    sim4.add_atom(3, 3, 0, orbital='dxy', weight=1.2)
    sim4.add_atom(1.5, 1.5, 0, orbital='dz2', weight=1.3)
    
    X4, Y4, ldos_mixed = sim4.calculate_ldos(bias_voltage=1.0)
    plot_stm_image(X4, Y4, ldos_mixed, title="STM LDOS - Mixed Orbitals")
    plt.savefig('stm_ldos_mixed_orbitals.png', dpi=150)
    
    print("\n5. dI/dV Spectroscopy Simulation...")
    biases, dIdV_maps = sim.simulate_dIdV(bias_min=-2.0, bias_max=2.0, num_points=21)
    mid_idx = sim.grid_size // 2
    plot_dIdV_spectrum(biases, dIdV_maps, mid_idx, mid_idx, title="dI/dV Spectrum at Center")
    plt.savefig('stm_dIdV_spectrum.png', dpi=150)
    
    print("\n6. STS Point Spectrum...")
    center_x = (x_min + x_max) / 2
    center_y = (Y.min() + Y.max()) / 2
    sts_biases, sts_spectrum = sim.calculate_sts_spectrum(
        center_x, center_y, bias_min=-2.0, bias_max=2.0, num_points=41
    )
    plot_line_profile(sts_biases, sts_spectrum, 
                      title="STS Point Spectrum at Center",
                      xlabel='Bias Voltage (V)', ylabel='dI/dV (a.u.)')
    plt.savefig('stm_sts_point.png', dpi=150)
    
    print("\n7. Bardeen Tunneling Current...")
    X_cur, Y_cur, current = sim.calculate_tunneling_current(bias_voltage=1.5)
    plot_stm_image(X_cur, Y_cur, current, title="Bardeen Tunneling Current (V=1.5V)",
                   cmap='hot', clabel='Current (a.u.)')
    plt.savefig('stm_bardeen_current.png', dpi=150)
    
    print("\n8. FFT Analysis...")
    FX, FY, fft_mag = sim.fft_analysis(ldos_pos)
    plot_stm_image(FX, FY, fft_mag, title="FFT of STM Image - Square Lattice",
                   cmap='jet', xlabel='f_x (1/Å)', ylabel='f_y (1/Å)', clabel='Magnitude (dB)')
    plt.savefig('stm_fft_square.png', dpi=150)
    
    print("\n9. Tip Shape Effect - s-wave vs p-wave vs d-wave...")
    sim_tip = STMSimulator(lattice_constant=3.0, grid_size=100, scan_height=1.8)
    sim_tip.generate_square_lattice(4, 4)
    print(f"   Total atoms: {len(sim_tip.atoms)}")
    
    tip_s = TipModel(s=1.0)
    tip_pz = TipModel(pz=1.0)
    tip_px = TipModel(px=1.0)
    tip_dz2 = TipModel(dz2=1.0)
    tip_dx2y2 = TipModel(dx2y2=1.0)
    tip_dxy = TipModel(dxy=1.0)
    
    print(f"   Tip models: {tip_s}, {tip_pz}, {tip_px}, {tip_dz2}, {tip_dx2y2}, {tip_dxy}")
    
    Xt, Yt, img_s = sim_tip.calculate_ldos_with_tip(1.0, tip_s)
    _, _, img_pz = sim_tip.calculate_ldos_with_tip(1.0, tip_pz)
    _, _, img_px = sim_tip.calculate_ldos_with_tip(1.0, tip_px)
    _, _, img_dz2 = sim_tip.calculate_ldos_with_tip(1.0, tip_dz2)
    _, _, img_dx2y2 = sim_tip.calculate_ldos_with_tip(1.0, tip_dx2y2)
    _, _, img_dxy = sim_tip.calculate_ldos_with_tip(1.0, tip_dxy)
    
    plot_tip_comparison(Xt, Yt,
        [img_s, img_pz, img_px, img_dz2, img_dx2y2, img_dxy],
        ["s-wave tip", "p_z tip", "p_x tip", "d_{z²} tip", "d_{x²-y²} tip", "d_{xy} tip"],
        ncols=3, super_title="STM Image Distortion by Tip Multipole Components")
    plt.savefig('stm_tip_comparison.png', dpi=150, bbox_inches='tight')
    
    print("\n10. Asymmetric Tip (s + p_x + d_xy mixture)...")
    tip_mixed1 = TipModel(s=0.7, px=0.5, dxy=0.3)
    tip_mixed2 = TipModel(s=0.5, pz=0.6, dxz=0.4)
    
    print(f"   Tip 1: {tip_mixed1}")
    print(f"   Tip 2: {tip_mixed2}")
    
    _, _, img_mix1 = sim_tip.calculate_ldos_with_tip(1.0, tip_mixed1)
    _, _, img_mix2 = sim_tip.calculate_ldos_with_tip(1.0, tip_mixed2)
    
    plot_tip_comparison(Xt, Yt,
        [img_s, img_mix1, img_mix2],
        ["s-wave (reference)", f"mixed tip: s + p_x + d_xy", f"mixed tip: s + p_z + d_xz"],
        ncols=3, super_title="Asymmetric Tip-Induced Image Distortion")
    plt.savefig('stm_tip_asymmetric.png', dpi=150, bbox_inches='tight')
    
    print("\n11. Line Profile Comparison (s vs p_x vs d_xy)...")
    x_tmin, x_tmax = Xt.min(), Xt.max()
    y_tmid = (Yt.min() + Yt.max()) / 2
    
    d_s, p_s = sim_tip.extract_line_profile(Xt, Yt, img_s, x_tmin, y_tmid, x_tmax, y_tmid)
    _, p_px = sim_tip.extract_line_profile(Xt, Yt, img_px, x_tmin, y_tmid, x_tmax, y_tmid)
    _, p_dxy = sim_tip.extract_line_profile(Xt, Yt, img_dxy, x_tmin, y_tmid, x_tmax, y_tmid)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(d_s, p_s, 'b-', linewidth=2, label='s-wave tip')
    ax.plot(d_s, p_px, 'r--', linewidth=2, label='p_x tip')
    ax.plot(d_s, p_dxy, 'g-.', linewidth=2, label='d_{xy} tip')
    ax.set_xlabel('Distance (Å)', fontsize=12)
    ax.set_ylabel('Signal (a.u.)', fontsize=12)
    ax.set_title('Line Profile Comparison — Different Tip States', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('stm_tip_line_profiles.png', dpi=150)
    
    print("\n12. IETS — CO on Cu(111) top site...")
    iets = IETSSimulator(kT=0.003, modulation_amplitude=0.002)
    voltages = np.linspace(-0.05, 0.4, 1000)
    
    mol_co = MolecularVibrations.get_vibrations('CO', adsorption_site='top', orientation='upright')
    print(f"   Molecule: {mol_co['name']} ({mol_co['adsorption_site']}, {mol_co['orientation']})")
    for m in mol_co['modes']:
        print(f"     {m['label']}: {m['energy']} meV, intensity={m['intensity']:.2f}")
    
    dIdV_c, dIdV_n, d2_c, d2_n = iets.simulate_experimental(
        voltages, mol_co, elastic_background=1.0, inelastic_coupling=0.08, noise_level=0.001)
    plot_iets_spectrum(voltages, dIdV_c, dIdV_n, d2_c, d2_n, mol_co,
                      title="IETS Spectrum — CO on Cu(111)")
    plt.savefig('iets_co_spectrum.png', dpi=150)
    
    print("\n13. IETS — Molecule Identification Comparison...")
    available = MolecularVibrations.available_molecules()
    print(f"   Available molecules: {available}")
    
    mol_results = iets.compare_molecules(voltages, ['CO', 'C2H2', 'H2O', 'C6H6'],
                                         elastic_background=1.0, inelastic_coupling=0.06,
                                         noise_level=0.001)
    plot_iets_molecule_comparison(voltages, mol_results, ncols=2,
                                  super_title="IETS Molecule Identification — d²I/dV² Spectra")
    plt.savefig('iets_molecule_comparison.png', dpi=150, bbox_inches='tight')
    
    print("\n14. IETS — Adsorption Configuration Effect (CO)...")
    config_results = iets.compare_adsorption_configs(
        voltages, 'CO',
        sites=['top', 'bridge', 'hollow'],
        orientations=['upright'],
        elastic_background=1.0, inelastic_coupling=0.08, noise_level=0.001)
    plot_iets_adsorption_comparison(voltages, config_results,
                                    title="IETS — CO Adsorption Configuration Dependence")
    plt.savefig('iets_adsorption_config.png', dpi=150)
    
    print("\n15. IETS — Molecule Identification via Correlation...")
    target = 'C2H2'
    target_info = MolecularVibrations.get_vibrations(target)
    _, _, _, d2_target_noisy = iets.simulate_experimental(
        voltages, target_info, elastic_background=1.0, inelastic_coupling=0.07, noise_level=0.002)
    
    best_key, best_score, scores = iets.identify_molecule(voltages, d2_target_noisy)
    
    print(f"   Target molecule: {target}")
    print(f"   Predicted molecule: {best_key} (correlation={best_score:.4f})")
    print(f"   All scores:")
    for k, v in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"     {k}: {v:.4f}")
    
    print(f"\n   Identification {'SUCCESS' if best_key == target else 'FAILED'}!")
    
    print("\nAll plots saved successfully!")
    print("- stm_ldos_square_positive.png")
    print("- stm_ldos_square_negative.png")
    print("- stm_topography_square.png")
    print("- stm_3d_square.png")
    print("- stm_line_profile_square.png")
    print("- stm_ldos_hexagonal.png")
    print("- stm_ldos_honeycomb.png")
    print("- stm_ldos_honeycomb_noisy.png")
    print("- stm_ldos_mixed_orbitals.png")
    print("- stm_dIdV_spectrum.png")
    print("- stm_sts_point.png")
    print("- stm_bardeen_current.png")
    print("- stm_fft_square.png")
    print("- stm_tip_comparison.png")
    print("- stm_tip_asymmetric.png")
    print("- stm_tip_line_profiles.png")
    print("- iets_co_spectrum.png")
    print("- iets_molecule_comparison.png")
    print("- iets_adsorption_config.png")
    
    plt.show()


if __name__ == "__main__":
    main()
