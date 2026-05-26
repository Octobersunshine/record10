import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
from scipy import stats
import emcee
import corner
import warnings
warnings.filterwarnings('ignore')

from baryon_models import BaryonicModels, MONDModel, generate_sample_rotation_curve


class ComprehensiveRotationCurveFitter:
    def __init__(self):
        self.G = 4.30091e-6
        self.baryon_model = BaryonicModels()
        self.mond_model = MONDModel()
        
        self.r = None
        self.v = None
        self.v_err = None
        
        self.baryon_param_names = ['M_disk', 'R_disk', 'M_bulge', 'r_bulge', 
                                   'M_hi', 'R_hi', 'M_h2', 'R_h2']
        self.dm_param_names = ['rho_s', 'r_s']
        
        self.baryon_p0 = [0.05, 3.0, 0.01, 0.5, 0.015, 8.0, 0.005, 4.0]
        self.baryon_bounds = ([1e-5, 0.5, 1e-5, 0.1, 1e-5, 1.0, 1e-5, 0.5],
                              [2.0, 10.0, 0.5, 5.0, 0.5, 25.0, 0.2, 15.0])
        
        self.dm_p0 = [0.05, 15.0]
        self.dm_bounds = ([1e-5, 1.0], [2.0, 100.0])
        
        self.ml_baryon_params = None
        self.ml_dm_params = None
        self.ml_mond_params = None
        
        self.baryon_sampler = None
        self.dm_sampler = None
        self.mond_sampler = None
        
        self.baryon_flat_samples = None
        self.dm_flat_samples = None
        self.mond_flat_samples = None

    def load_data(self, r, v, v_err=None):
        self.r = np.array(r)
        self.v = np.array(v)
        if v_err is None:
            self.v_err = np.full_like(self.v, 5.0)
        else:
            self.v_err = np.array(v_err)
        return self

    def load_from_file(self, filename, delimiter=',', skiprows=0):
        data = np.loadtxt(filename, delimiter=delimiter, skiprows=skiprows)
        if data.shape[1] == 2:
            self.r, self.v = data[:, 0], data[:, 1]
            self.v_err = np.full_like(self.v, 5.0)
        elif data.shape[1] >= 3:
            self.r, self.v, self.v_err = data[:, 0], data[:, 1], data[:, 2]
        return self

    def nfw_v_circular(self, r, rho_s, r_s):
        x = r / r_s
        rho_s_converted = rho_s * 1e9
        M_enclosed = 4 * np.pi * rho_s_converted * r_s**3 * (np.log(1 + x) - x / (1 + x))
        return np.sqrt(self.G * M_enclosed / r)

    def total_v_dm(self, r, all_params):
        baryon_params = all_params[:8]
        dm_params = all_params[8:]
        
        v_baryon, baryon_components = self.baryon_model.total_baryonic_v(r, baryon_params)
        v_dm = self.nfw_v_circular(r, *dm_params)
        
        v_total = np.sqrt(v_baryon**2 + v_dm**2)
        return v_total, baryon_components, v_dm, v_baryon

    def fit_maximum_likelihood(self, model='dm'):
        if model == 'dm':
            def neg_log_prob(params):
                v_pred, _, _, _ = self.total_v_dm(self.r, params)
                if not np.all(np.isfinite(v_pred)):
                    return 1e10
                return 0.5 * np.sum(((self.v - v_pred) / self.v_err)**2)
            
            p0 = self.baryon_p0 + self.dm_p0
            bounds = [list(b) for b in zip(self.baryon_bounds[0] + self.dm_bounds[0],
                                           self.baryon_bounds[1] + self.dm_bounds[1])]
            
            result = minimize(neg_log_prob, p0, method='Nelder-Mead',
                             options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 20000})
            
            if result.success:
                self.ml_baryon_params = result.x[:8]
                self.ml_dm_params = result.x[8:]
                return result.x
            else:
                return p0
        
        elif model == 'mond':
            def neg_log_prob(params):
                v_pred, _, _ = self.mond_model.mond_v_circular(self.r, params, self.baryon_model)
                if not np.all(np.isfinite(v_pred)):
                    return 1e10
                return 0.5 * np.sum(((self.v - v_pred) / self.v_err)**2)
            
            p0 = self.baryon_p0
            bounds = [list(b) for b in zip(self.baryon_bounds[0], self.baryon_bounds[1])]
            
            result = minimize(neg_log_prob, p0, method='Nelder-Mead',
                             options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 20000})
            
            if result.success:
                self.ml_mond_params = result.x
                return result.x
            else:
                return p0

    def log_prob_dm(self, params):
        baryon_params = params[:8]
        dm_params = params[8:]
        
        for i in range(8):
            if not (self.baryon_bounds[0][i] <= baryon_params[i] <= self.baryon_bounds[1][i]):
                return -np.inf
        
        for i in range(2):
            if not (self.dm_bounds[0][i] <= dm_params[i] <= self.dm_bounds[1][i]):
                return -np.inf
        
        v_pred, _, _, _ = self.total_v_dm(self.r, params)
        if not np.all(np.isfinite(v_pred)):
            return -np.inf
        
        chi2 = np.sum(((self.v - v_pred) / self.v_err)**2)
        return -0.5 * chi2

    def log_prob_mond(self, params):
        for i in range(8):
            if not (self.baryon_bounds[0][i] <= params[i] <= self.baryon_bounds[1][i]):
                return -np.inf
        
        v_pred, _, _ = self.mond_model.mond_v_circular(self.r, params, self.baryon_model)
        if not np.all(np.isfinite(v_pred)):
            return -np.inf
        
        chi2 = np.sum(((self.v - v_pred) / self.v_err)**2)
        return -0.5 * chi2

    def run_mcmc(self, model='dm', n_walkers=32, n_steps=5000, n_burn=1000):
        if self.r is None:
            raise ValueError("No data loaded!")
        
        if model == 'dm':
            ndim = 10
            if self.ml_baryon_params is not None and self.ml_dm_params is not None:
                p0 = np.concatenate([self.ml_baryon_params, self.ml_dm_params])
            else:
                p0 = np.array(self.baryon_p0 + self.dm_p0)
            
            log_prob = self.log_prob_dm
            
        elif model == 'mond':
            ndim = 8
            if self.ml_mond_params is not None:
                p0 = self.ml_mond_params
            else:
                p0 = np.array(self.baryon_p0)
            
            log_prob = self.log_prob_mond
        
        initial_positions = []
        for _ in range(n_walkers):
            pos = np.array([p0[j] * (0.8 + 0.4 * np.random.rand()) for j in range(ndim)])
            for j in range(ndim):
                if model == 'dm':
                    low = self.baryon_bounds[0][j] if j < 8 else self.dm_bounds[0][j-8]
                    high = self.baryon_bounds[1][j] if j < 8 else self.dm_bounds[1][j-8]
                else:
                    low = self.baryon_bounds[0][j]
                    high = self.baryon_bounds[1][j]
                pos[j] = np.clip(pos[j], low * 1.1, high * 0.9)
            initial_positions.append(pos)
        initial_positions = np.array(initial_positions)
        
        sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob)
        
        print(f"Running MCMC for {model.upper()} model with {ndim} parameters...")
        print(f"  Walkers: {n_walkers}, Steps: {n_steps}, Burn-in: {n_burn}")
        
        sampler.run_mcmc(initial_positions, n_steps, progress=False)
        
        flat_samples = sampler.get_chain(discard=n_burn, thin=15, flat=True)
        
        if model == 'dm':
            self.dm_sampler = sampler
            self.dm_flat_samples = flat_samples
        else:
            self.mond_sampler = sampler
            self.mond_flat_samples = flat_samples
        
        return flat_samples, np.median(flat_samples, axis=0), np.std(flat_samples, axis=0)

    def compute_goodness_of_fit(self, model='dm', params=None):
        if model == 'dm':
            if params is None:
                if self.ml_baryon_params is not None and self.ml_dm_params is not None:
                    params = np.concatenate([self.ml_baryon_params, self.ml_dm_params])
                else:
                    params = np.median(self.dm_flat_samples, axis=0)
            v_pred, _, _, _ = self.total_v_dm(self.r, params)
            dof = len(self.r) - 10
        elif model == 'mond':
            if params is None:
                if self.ml_mond_params is not None:
                    params = self.ml_mond_params
                else:
                    params = np.median(self.mond_flat_samples, axis=0)
            v_pred, _, _ = self.mond_model.mond_v_circular(self.r, params, self.baryon_model)
            dof = len(self.r) - 8
        
        chi2 = np.sum(((self.v - v_pred) / self.v_err)**2)
        reduced_chi2 = chi2 / dof
        p_value = 1 - stats.chi2.cdf(chi2, dof)
        aic = chi2 + 2 * (dof + 1)
        bic = chi2 + np.log(len(self.r)) * (dof + 1)
        
        return {'chi2': chi2, 'reduced_chi2': reduced_chi2, 
                'p_value': p_value, 'dof': dof, 'aic': aic, 'bic': bic}

    def plot_decomposition(self, model='dm', save_path='decomposition.png'):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        r_fine = np.linspace(min(self.r), max(self.r), 1000)
        
        ax.errorbar(self.r, self.v, yerr=self.v_err, fmt='o', color='black', 
                     label='Observations', capsize=5, markersize=6, zorder=5)
        
        if model == 'dm':
            if self.ml_baryon_params is not None and self.ml_dm_params is not None:
                params = np.concatenate([self.ml_baryon_params, self.ml_dm_params])
            else:
                params = np.median(self.dm_flat_samples, axis=0)
            
            v_total, baryon_components, v_dm, v_baryon = self.total_v_dm(r_fine, params)
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            labels = ['Stellar Disk', 'Bulge', 'HI Gas', 'H₂ Gas']
            keys = ['disk', 'bulge', 'hi', 'h2']
            
            for i, key in enumerate(keys):
                _, comp, _, _ = self.total_v_dm(r_fine, params)
                ax.fill_between(r_fine, 0, comp[key], alpha=0.3, color=colors[i], label=labels[i])
            
            ax.fill_between(r_fine, v_baryon, np.sqrt(v_baryon**2 + v_dm**2), 
                           alpha=0.3, color='#9467bd', label='Dark Matter')
            
            ax.plot(r_fine, v_total, 'r-', linewidth=2.5, label='Total')
            ax.plot(r_fine, v_baryon, 'k--', linewidth=1.5, label='Baryons Only')
            ax.plot(r_fine, v_dm, 'm-', linewidth=1.5, alpha=0.7, label='Dark Matter Only')
            
        elif model == 'mond':
            if self.ml_mond_params is not None:
                params = self.ml_mond_params
            else:
                params = np.median(self.mond_flat_samples, axis=0)
            
            v_mond, v_baryon, baryon_components = self.mond_model.mond_v_circular(
                r_fine, params, self.baryon_model)
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            labels = ['Stellar Disk', 'Bulge', 'HI Gas', 'H₂ Gas']
            keys = ['disk', 'bulge', 'hi', 'h2']
            
            for i, key in enumerate(keys):
                _, _, comp = self.mond_model.mond_v_circular(r_fine, params, self.baryon_model)
                ax.fill_between(r_fine, 0, comp[key], alpha=0.3, color=colors[i], label=labels[i])
            
            ax.plot(r_fine, v_mond, 'r-', linewidth=2.5, label='MOND Prediction')
            ax.plot(r_fine, v_baryon, 'k--', linewidth=1.5, label='Newtonian Baryons')
        
        ax.set_xlabel('Radius r (kpc)', fontsize=13)
        ax.set_ylabel('Rotation Velocity v (km/s)', fontsize=13)
        ax.set_title(f'Rotation Curve Decomposition - {model.upper()}', 
                    fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, loc='lower right', ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Decomposition plot saved: {save_path}")

    def compare_models(self, save_path='model_comparison.png'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        r_fine = np.linspace(min(self.r), max(self.r), 1000)
        
        ax1.errorbar(self.r, self.v, yerr=self.v_err, fmt='o', color='black', 
                     label='Observations', capsize=5, markersize=6, zorder=5)
        
        if self.ml_baryon_params is not None and self.ml_dm_params is not None:
            params_dm = np.concatenate([self.ml_baryon_params, self.ml_dm_params])
            v_dm_total, _, _, v_baryon_dm = self.total_v_dm(r_fine, params_dm)
            ax1.plot(r_fine, v_dm_total, 'b-', linewidth=2.5, label='Dark Matter (NFW)')
        
        if self.ml_mond_params is not None:
            v_mond_total, _, _ = self.mond_model.mond_v_circular(
                r_fine, self.ml_mond_params, self.baryon_model)
            ax1.plot(r_fine, v_mond_total, 'r-', linewidth=2.5, label='MOND')
        
        ax1.set_xlabel('Radius r (kpc)', fontsize=13)
        ax1.set_ylabel('Rotation Velocity v (km/s)', fontsize=13)
        ax1.set_title('Model Comparison: Rotation Curves', fontsize=15, fontweight='bold')
        ax1.legend(fontsize=11, loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        
        gof_dm = self.compute_goodness_of_fit('dm')
        gof_mond = self.compute_goodness_of_fit('mond')
        
        models = ['Dark Matter', 'MOND']
        metrics = ['χ²_red', 'AIC', 'BIC']
        values_dm = [gof_dm['reduced_chi2'], gof_dm['aic'], gof_dm['bic']]
        values_mond = [gof_mond['reduced_chi2'], gof_mond['aic'], gof_mond['bic']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, values_dm, width, label='Dark Matter', color='#1f77b4', alpha=0.7)
        bars2 = ax2.bar(x + width/2, values_mond, width, label='MOND', color='#ff7f0e', alpha=0.7)
        
        ax2.set_xlabel('Metric', fontsize=13)
        ax2.set_ylabel('Value', fontsize=13)
        ax2.set_title('Goodness of Fit Comparison', fontsize=15, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar in bars1:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Model comparison plot saved: {save_path}")

    def print_results(self):
        print("\n" + "=" * 80)
        print("COMPREHENSIVE ROTATION CURVE ANALYSIS RESULTS")
        print("=" * 80)
        
        if self.ml_baryon_params is not None:
            print("\n[Baryonic Parameters]")
            names = self.baryon_param_names
            units = ['10^10 M☉', 'kpc', '10^10 M☉', 'kpc', 
                    '10^10 M☉', 'kpc', '10^10 M☉', 'kpc']
            for i, (name, val, unit) in enumerate(zip(names, self.ml_baryon_params, units)):
                print(f"  {name:>8} = {val:8.4f} {unit}")
            
            M_disk, R_disk, M_bulge, r_bulge, M_hi, R_hi, M_h2, R_h2 = self.ml_baryon_params
            total_baryonic = M_disk + M_bulge + M_hi + M_h2
            print(f"\n  Total baryonic mass: {total_baryonic:.2f} × 10^10 M☉")
            print(f"  Stellar fraction: {(M_disk + M_bulge)/total_baryonic:.1%}")
            print(f"  Gas fraction: {(M_hi + M_h2)/total_baryonic:.1%}")
        
        if self.ml_dm_params is not None:
            print(f"\n[Dark Matter Parameters (NFW)]")
            print(f"  rho_s = {self.ml_dm_params[0]:.4f} × 10⁻⁹ M☉/pc³")
            print(f"  r_s   = {self.ml_dm_params[1]:.4f} kpc")
            
            gof = self.compute_goodness_of_fit('dm')
            print(f"\n  Goodness of Fit:")
            print(f"    χ² = {gof['chi2']:.2f}, χ²_red = {gof['reduced_chi2']:.2f}")
            print(f"    AIC = {gof['aic']:.2f}, BIC = {gof['bic']:.2f}")
            print(f"    p-value = {gof['p_value']:.4f}")
        
        if self.ml_mond_params is not None:
            print(f"\n[MOND Parameters]")
            names = self.baryon_param_names
            units = ['10^10 M☉', 'kpc', '10^10 M☉', 'kpc', 
                    '10^10 M☉', 'kpc', '10^10 M☉', 'kpc']
            for i, (name, val, unit) in enumerate(zip(names, self.ml_mond_params, units)):
                print(f"  {name:>8} = {val:8.4f} {unit}")
            
            gof = self.compute_goodness_of_fit('mond')
            print(f"\n  Goodness of Fit:")
            print(f"    χ² = {gof['chi2']:.2f}, χ²_red = {gof['reduced_chi2']:.2f}")
            print(f"    AIC = {gof['aic']:.2f}, BIC = {gof['bic']:.2f}")
            print(f"    p-value = {gof['p_value']:.4f}")
        
        if self.ml_dm_params is not None and self.ml_mond_params is not None:
            print(f"\n[Model Comparison]")
            gof_dm = self.compute_goodness_of_fit('dm')
            gof_mond = self.compute_goodness_of_fit('mond')
            
            if gof_dm['reduced_chi2'] < gof_mond['reduced_chi2']:
                print(f"  Preferred model: Dark Matter (lower χ²_red)")
                delta_aic = gof_dm['aic'] - gof_mond['aic']
                if abs(delta_aic) > 10:
                    print(f"  ΔAIC = {delta_aic:.1f}: Strong evidence for {'DM' if delta_aic < 0 else 'MOND'}")
                elif abs(delta_aic) > 2:
                    print(f"  ΔAIC = {delta_aic:.1f}: Weak evidence for {'DM' if delta_aic < 0 else 'MOND'}")
                else:
                    print(f"  ΔAIC = {delta_aic:.1f}: Models are essentially equivalent")
        
        print("\n" + "=" * 80)
