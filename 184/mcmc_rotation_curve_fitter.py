import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
import emcee
import corner
import warnings
warnings.filterwarnings('ignore')


class MCMCRotationCurveFitter:
    def __init__(self):
        self.G = 4.30091e-6
        self.r = None
        self.v = None
        self.v_err = None
        self.ml_params = None
        self.sampler = None
        self.samples = None
        self.flat_samples = None
        self.best_fit_params = None
        self.param_uncertainties = None
        
    def nfw_v_circular(self, r, rho_s, r_s):
        x = r / r_s
        rho_s_converted = rho_s * 1e9
        M_enclosed = 4 * np.pi * rho_s_converted * r_s**3 * (np.log(1 + x) - x / (1 + x))
        return np.sqrt(self.G * M_enclosed / r)
    
    def burkert_v_circular(self, r, rho_0, r_0):
        x = r / r_0
        rho_0_converted = rho_0 * 1e9
        term1 = np.log(1 + x)
        term2 = -0.5 * np.log(1 + x**2)
        term3 = -np.arctan(x)
        M_enclosed = 4 * np.pi * rho_0_converted * r_0**3 * (term1 + term2 + term3)
        M_enclosed = np.maximum(M_enclosed, 1e-10)
        return np.sqrt(self.G * M_enclosed / r)
    
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
    
    def generate_sample_data(self, r_min=1, r_max=30, n_points=20, noise=5, 
                            model='nfw', true_params=(5.0, 15.0)):
        self.r = np.linspace(r_min, r_max, n_points)
        if model == 'nfw':
            self.v = self.nfw_v_circular(self.r, *true_params)
        else:
            self.v = self.burkert_v_circular(self.r, *true_params)
        self.v += np.random.normal(0, noise, n_points)
        self.v_err = np.full_like(self.v, noise)
        return self, true_params
    
    def log_likelihood(self, params, model='nfw'):
        if model == 'nfw':
            rho_s, r_s = params
            if rho_s <= 0 or r_s <= 0:
                return -np.inf
            v_pred = self.nfw_v_circular(self.r, rho_s, r_s)
        else:
            rho_0, r_0 = params
            if rho_0 <= 0 or r_0 <= 0:
                return -np.inf
            v_pred = self.burkert_v_circular(self.r, rho_0, r_0)
        
        if not np.all(np.isfinite(v_pred)):
            return -np.inf
        
        chi2 = np.sum(((self.v - v_pred) / self.v_err)**2)
        return -0.5 * chi2
    
    def log_prior(self, params, model='nfw'):
        if model == 'nfw':
            rho_s, r_s = params
            if 0.001 <= rho_s <= 100 and 1.0 <= r_s <= 100:
                return 0.0
        else:
            rho_0, r_0 = params
            if 0.001 <= rho_0 <= 100 and 1.0 <= r_0 <= 100:
                return 0.0
        return -np.inf
    
    def log_probability(self, params, model='nfw'):
        lp = self.log_prior(params, model)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(params, model)
    
    def find_maximum_likelihood(self, model='nfw', p0=(5.0, 15.0)):
        def neg_log_prob(params):
            return -self.log_probability(params, model)
        
        result = minimize(neg_log_prob, p0, method='Nelder-Mead',
                         options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 10000})
        
        if result.success:
            self.ml_params = result.x
            return result.x
        else:
            return p0
    
    def run_mcmc(self, model='nfw', n_walkers=32, n_steps=5000, n_burn=1000, p0=None):
        if self.r is None:
            raise ValueError("No data loaded!")
        
        if p0 is None:
            if self.ml_params is not None:
                p0 = self.ml_params
            else:
                p0 = self.find_maximum_likelihood(model)
        
        ndim = 2
        initial_positions = [p0 + 1e-4 * np.random.randn(ndim) for _ in range(n_walkers)]
        initial_positions = np.array([[max(p[0], 0.01), max(p[1], 1.0)] for p in initial_positions])
        
        self.sampler = emcee.EnsembleSampler(n_walkers, ndim, self.log_probability, 
                                             args=[model])
        
        print(f"Running MCMC for {model.upper()} model...")
        print(f"  Walkers: {n_walkers}, Steps: {n_steps}, Burn-in: {n_burn}")
        
        self.sampler.run_mcmc(initial_positions, n_steps, progress=True)
        
        self.samples = self.sampler.get_chain()
        self.flat_samples = self.sampler.get_chain(discard=n_burn, thin=15, flat=True)
        
        self.best_fit_params = np.median(self.flat_samples, axis=0)
        self.param_uncertainties = np.std(self.flat_samples, axis=0)
        
        return self.flat_samples, self.best_fit_params, self.param_uncertainties
    
    def get_parameter_uncertainties(self, ci=68):
        lower = np.percentile(self.flat_samples, (100 - ci) / 2, axis=0)
        upper = np.percentile(self.flat_samples, 100 - (100 - ci) / 2, axis=0)
        return lower, upper
    
    def get_correlation_matrix(self):
        return np.corrcoef(self.flat_samples.T)
    
    def compute_goodness_of_fit(self, model='nfw'):
        if model == 'nfw':
            v_pred = self.nfw_v_circular(self.r, *self.best_fit_params)
        else:
            v_pred = self.burkert_v_circular(self.r, *self.best_fit_params)
        
        chi2 = np.sum(((self.v - v_pred) / self.v_err)**2)
        dof = len(self.r) - 2
        reduced_chi2 = chi2 / dof
        p_value = 1 - stats.chi2.cdf(chi2, dof)
        
        return {'chi2': chi2, 'reduced_chi2': reduced_chi2, 
                'p_value': p_value, 'dof': dof}
    
    def check_convergence(self):
        try:
            tau = self.sampler.get_autocorr_time()
            print(f"  Autocorrelation times: {tau}")
            return tau
        except:
            print("  Warning: Could not compute autocorrelation time")
            return None
    
    def plot_trace(self, model='nfw', save_path='trace_plot.png'):
        fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
        
        if model == 'nfw':
            labels = [r'$\rho_s$ [$\times 10^{-9}$ M$_\odot$/pc$^3$]', r'$r_s$ [kpc]']
        else:
            labels = [r'$\rho_0$ [$\times 10^{-9}$ M$_\odot$/pc$^3$]', r'$r_0$ [kpc]']
        
        for i in range(2):
            ax = axes[i]
            ax.plot(self.samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(self.samples))
            ax.set_ylabel(labels[i], fontsize=12)
            ax.yaxis.set_label_coords(-0.1, 0.5)
        
        axes[-1].set_xlabel("Step number", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Trace plot saved: {save_path}")
    
    def plot_corner(self, model='nfw', save_path='corner_plot.png'):
        if model == 'nfw':
            labels = [r'$\rho_s$ [$\times 10^{-9}$ M$_\odot$/pc$^3$]', r'$r_s$ [kpc]']
            truths = self.ml_params if self.ml_params is not None else None
        else:
            labels = [r'$\rho_0$ [$\times 10^{-9}$ M$_\odot$/pc$^3$]', r'$r_0$ [kpc]']
            truths = self.ml_params if self.ml_params is not None else None
        
        fig = corner.corner(self.flat_samples, labels=labels, 
                           truths=truths,
                           quantiles=[0.16, 0.5, 0.84],
                           show_titles=True,
                           title_kwargs={"fontsize": 12},
                           color='steelblue',
                           truth_color='red')
        
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Corner plot saved: {save_path}")
    
    def plot_fit_with_uncertainty(self, model='nfw', save_path='fit_with_uncertainty.png'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        r_fine = np.linspace(min(self.r), max(self.r), 1000)
        
        ax1.errorbar(self.r, self.v, yerr=self.v_err, fmt='o', color='black', 
                     label='Observations', capsize=5, markersize=6, zorder=5)
        
        n_samples = min(100, len(self.flat_samples))
        indices = np.random.choice(len(self.flat_samples), n_samples, replace=False)
        
        for idx in indices:
            params = self.flat_samples[idx]
            if model == 'nfw':
                v_pred = self.nfw_v_circular(r_fine, *params)
            else:
                v_pred = self.burkert_v_circular(r_fine, *params)
            ax1.plot(r_fine, v_pred, color='steelblue', alpha=0.05)
        
        if model == 'nfw':
            v_best = self.nfw_v_circular(r_fine, *self.best_fit_params)
        else:
            v_best = self.burkert_v_circular(r_fine, *self.best_fit_params)
        
        ax1.plot(r_fine, v_best, color='red', linewidth=2.5, 
                label=f'Best fit ({model.upper()})')
        
        v_low = np.percentile([[self.nfw_v_circular(r, *p) if model == 'nfw' 
                                else self.burkert_v_circular(r, *p) 
                                for p in self.flat_samples] for r in r_fine], 16, axis=1)
        v_high = np.percentile([[self.nfw_v_circular(r, *p) if model == 'nfw' 
                                 else self.burkert_v_circular(r, *p) 
                                 for p in self.flat_samples] for r in r_fine], 84, axis=1)
        
        ax1.fill_between(r_fine, v_low, v_high, color='steelblue', alpha=0.2, 
                        label='68% credible region')
        
        ax1.set_xlabel('Radius r (kpc)', fontsize=13)
        ax1.set_ylabel('Rotation Velocity v (km/s)', fontsize=13)
        ax1.set_title(f'Rotation Curve: {model.upper()} Model', fontsize=15, fontweight='bold')
        ax1.legend(fontsize=11, loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        
        if model == 'nfw':
            v_pred = self.nfw_v_circular(self.r, *self.best_fit_params)
        else:
            v_pred = self.burkert_v_circular(self.r, *self.best_fit_params)
        
        residuals = (self.v - v_pred) / self.v_err
        ax2.scatter(self.r, residuals, color='black', s=50, zorder=5)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.3)
        ax2.axhline(y=-1, color='gray', linestyle='--', alpha=0.3)
        ax2.fill_between(self.r, -1, 1, color='gray', alpha=0.1)
        ax2.set_xlabel('Radius r (kpc)', fontsize=13)
        ax2.set_ylabel('Normalized Residual (v - v_pred)/σ', fontsize=13)
        ax2.set_title('Fit Residuals', fontsize=15, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Fit plot saved: {save_path}")
    
    def print_results(self, model='nfw'):
        print("\n" + "=" * 70)
        print(f"MCMC FITTING RESULTS - {model.upper()} MODEL")
        print("=" * 70)
        
        if model == 'nfw':
            p1_label, p2_label = 'rho_s', 'r_s'
            p1_unit = 'x 10^-9 M_sun/pc^3'
            p2_unit = 'kpc'
        else:
            p1_label, p2_label = 'rho_0', 'r_0'
            p1_unit = 'x 10^-9 M_sun/pc^3'
            p2_unit = 'kpc'
        
        print(f"\nMaximum Likelihood Estimate:")
        if self.ml_params is not None:
            print(f"  {p1_label} = {self.ml_params[0]:.4f} {p1_unit}")
            print(f"  {p2_label} = {self.ml_params[1]:.4f} {p2_unit}")
        
        print(f"\nBayesian Posterior Summary:")
        print(f"  {p1_label} = {self.best_fit_params[0]:.4f} ± {self.param_uncertainties[0]:.4f} {p1_unit}")
        print(f"  {p2_label} = {self.best_fit_params[1]:.4f} ± {self.param_uncertainties[1]:.4f} {p2_unit}")
        
        lower, upper = self.get_parameter_uncertainties()
        print(f"\n  68% Credible Intervals:")
        print(f"  {p1_label}: [{lower[0]:.4f}, {upper[0]:.4f}]")
        print(f"  {p2_label}: [{lower[1]:.4f}, {upper[1]:.4f}]")
        
        lower95, upper95 = self.get_parameter_uncertainties(ci=95)
        print(f"\n  95% Credible Intervals:")
        print(f"  {p1_label}: [{lower95[0]:.4f}, {upper95[0]:.4f}]")
        print(f"  {p2_label}: [{lower95[1]:.4f}, {upper95[1]:.4f}]")
        
        corr = self.get_correlation_matrix()
        print(f"\nParameter Correlation Matrix:")
        print(f"  [{corr[0,0]:.3f}  {corr[0,1]:.3f}]")
        print(f"  [{corr[1,0]:.3f}  {corr[1,1]:.3f}]")
        print(f"\n  Correlation coefficient: {corr[0,1]:.4f}")
        
        if corr[0,1] > 0.5:
            print(f"  Warning: Strong positive correlation detected!")
        elif corr[0,1] < -0.5:
            print(f"  Warning: Strong negative correlation detected!")
        
        gof = self.compute_goodness_of_fit(model)
        print(f"\nGoodness of Fit (at median):")
        print(f"  chi^2 = {gof['chi2']:.2f}")
        print(f"  reduced chi^2 = {gof['reduced_chi2']:.2f}")
        print(f"  p-value = {gof['p_value']:.4f}")
        
        self.check_convergence()
        
        print("\n" + "=" * 70)
    
    def full_analysis(self, model='nfw', n_walkers=32, n_steps=5000, n_burn=1000):
        print(f"\n{'='*70}")
        print(f"FULL BAYESIAN ANALYSIS - {model.upper()} MODEL")
        print(f"{'='*70}")
        
        self.find_maximum_likelihood(model)
        print(f"\nMaximum likelihood found: {self.ml_params}")
        
        self.run_mcmc(model, n_walkers, n_steps, n_burn)
        
        self.print_results(model)
        
        prefix = model.lower()
        self.plot_trace(model, f'{prefix}_trace.png')
        self.plot_corner(model, f'{prefix}_corner.png')
        self.plot_fit_with_uncertainty(model, f'{prefix}_fit_uncertainty.png')
        
        return self.best_fit_params, self.param_uncertainties


def main():
    np.random.seed(42)
    
    print("=" * 70)
    print("GALAXY ROTATION CURVE MCMC FITTING DEMONSTRATION")
    print("=" * 70)
    
    fitter = MCMCRotationCurveFitter()
    fitter.generate_sample_data(r_min=2, r_max=35, n_points=25, noise=8, 
                                model='nfw', true_params=(5.0, 15.0))
    
    print(f"\nGenerated data with true parameters:")
    print(f"  rho_s = 5.00 x 10^-9 M_sun/pc^3")
    print(f"  r_s = 15.00 kpc")
    
    fitter.full_analysis(model='nfw', n_walkers=32, n_steps=5000, n_burn=1000)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  nfw_trace.png - MCMC chain trace plots")
    print("  nfw_corner.png - Posterior distribution corner plot")
    print("  nfw_fit_uncertainty.png - Rotation curve with uncertainty bands")


if __name__ == "__main__":
    main()
