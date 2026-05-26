import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class GalaxyRotationCurveFitter:
    def __init__(self):
        self.G = 4.30091e-6
        self.r = None
        self.v = None
        self.v_err = None
        self.nfw_params = None
        self.nfw_errors = None
        self.burkert_params = None
        self.burkert_errors = None
        
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
    
    def fit_nfw(self, p0=(5.0, 15.0), bounds=([0.01, 1.0], [100.0, 100.0])):
        if self.r is None:
            raise ValueError("No data loaded!")
        
        def objective(r, rho_s, r_s):
            return self.nfw_v_circular(r, rho_s, r_s)
        
        popt, pcov = curve_fit(objective, self.r, self.v, p0=p0, 
                              sigma=self.v_err, bounds=bounds, absolute_sigma=True)
        self.nfw_params = popt
        self.nfw_errors = np.sqrt(np.diag(pcov))
        return popt, self.nfw_errors
    
    def fit_burkert(self, p0=(5.0, 15.0), bounds=([0.01, 1.0], [100.0, 100.0])):
        if self.r is None:
            raise ValueError("No data loaded!")
        
        def objective(r, rho_0, r_0):
            return self.burkert_v_circular(r, rho_0, r_0)
        
        popt, pcov = curve_fit(objective, self.r, self.v, p0=p0, 
                              sigma=self.v_err, bounds=bounds, absolute_sigma=True)
        self.burkert_params = popt
        self.burkert_errors = np.sqrt(np.diag(pcov))
        return popt, self.burkert_errors
    
    def goodness_of_fit(self, model='nfw'):
        if model == 'nfw':
            if self.nfw_params is None:
                raise ValueError("NFW model not fitted!")
            v_pred = self.nfw_v_circular(self.r, *self.nfw_params)
        else:
            if self.burkert_params is None:
                raise ValueError("Burkert model not fitted!")
            v_pred = self.burkert_v_circular(self.r, *self.burkert_params)
        
        chi2 = np.sum(((self.v - v_pred) / self.v_err)**2)
        dof = len(self.r) - 2
        reduced_chi2 = chi2 / dof
        p_value = 1 - stats.chi2.cdf(chi2, dof)
        
        return {'chi2': chi2, 'reduced_chi2': reduced_chi2, 
                'p_value': p_value, 'dof': dof}
    
    def fit_both(self):
        nfw_popt, nfw_perr = self.fit_nfw()
        burkert_popt, burkert_perr = self.fit_burkert()
        return nfw_popt, burkert_popt
    
    def print_results(self):
        print("=" * 65)
        print("GALAXY ROTATION CURVE FITTING RESULTS")
        print("=" * 65)
        
        if self.nfw_params is not None:
            gof_nfw = self.goodness_of_fit('nfw')
            print("\n[NFW Profile]")
            print(f"  rho_s = {self.nfw_params[0]:.3f} ± {self.nfw_errors[0]:.3f} × 10⁻⁹ M☉/pc³")
            print(f"  r_s   = {self.nfw_params[1]:.3f} ± {self.nfw_errors[1]:.3f} kpc")
            print(f"  χ²    = {gof_nfw['chi2']:.2f}")
            print(f"  χ²_red= {gof_nfw['reduced_chi2']:.2f}")
            print(f"  p-val = {gof_nfw['p_value']:.4f}")
        
        if self.burkert_params is not None:
            gof_burkert = self.goodness_of_fit('burkert')
            print("\n[Burkert Profile]")
            print(f"  rho_0 = {self.burkert_params[0]:.3f} ± {self.burkert_errors[0]:.3f} × 10⁻⁹ M☉/pc³")
            print(f"  r_0   = {self.burkert_params[1]:.3f} ± {self.burkert_errors[1]:.3f} kpc")
            print(f"  χ²    = {gof_burkert['chi2']:.2f}")
            print(f"  χ²_red= {gof_burkert['reduced_chi2']:.2f}")
            print(f"  p-val = {gof_burkert['p_value']:.4f}")
        
        if self.nfw_params is not None and self.burkert_params is not None:
            print("\n" + "-" * 65)
            gof_nfw = self.goodness_of_fit('nfw')
            gof_burkert = self.goodness_of_fit('burkert')
            if gof_nfw['reduced_chi2'] < gof_burkert['reduced_chi2']:
                print(f"Best model: NFW (χ²_red = {gof_nfw['reduced_chi2']:.2f})")
            else:
                print(f"Best model: Burkert (χ²_red = {gof_burkert['reduced_chi2']:.2f})")
        
        print("\n" + "=" * 65)
    
    def plot(self, save_path='rotation_curve_fit.png', show=True):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        r_fine = np.linspace(min(self.r), max(self.r), 1000)
        
        ax1.errorbar(self.r, self.v, yerr=self.v_err, fmt='o', color='black', 
                     label='Observations', capsize=5, markersize=6, zorder=5)
        
        if self.nfw_params is not None:
            v_nfw = self.nfw_v_circular(r_fine, *self.nfw_params)
            ax1.plot(r_fine, v_nfw, 'r-', linewidth=2.5, 
                     label=f'NFW: $\\rho_s$={self.nfw_params[0]:.1f}, $r_s$={self.nfw_params[1]:.1f}')
        
        if self.burkert_params is not None:
            v_burkert = self.burkert_v_circular(r_fine, *self.burkert_params)
            ax1.plot(r_fine, v_burkert, 'b--', linewidth=2.5, 
                     label=f'Burkert: $\\rho_0$={self.burkert_params[0]:.1f}, $r_0$={self.burkert_params[1]:.1f}')
        
        ax1.set_xlabel('Radius r (kpc)', fontsize=13)
        ax1.set_ylabel('Rotation Velocity v (km/s)', fontsize=13)
        ax1.set_title('Rotation Curve Fitting', fontsize=15, fontweight='bold')
        ax1.legend(fontsize=11, loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        
        if self.nfw_params is not None:
            v_pred_nfw = self.nfw_v_circular(self.r, *self.nfw_params)
            ax2.scatter(self.r, (self.v - v_pred_nfw) / self.v_err, 
                       color='red', label='NFW', alpha=0.7, s=50)
        
        if self.burkert_params is not None:
            v_pred_burkert = self.burkert_v_circular(self.r, *self.burkert_params)
            ax2.scatter(self.r, (self.v - v_pred_burkert) / self.v_err, 
                       color='blue', label='Burkert', alpha=0.7, s=50)
        
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.3)
        ax2.axhline(y=-1, color='gray', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Radius r (kpc)', fontsize=13)
        ax2.set_ylabel('Normalized Residual (v - v_pred)/σ', fontsize=13)
        ax2.set_title('Fit Residuals', fontsize=15, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        return save_path


def main():
    np.random.seed(42)
    
    print("Example 1: Fitting data generated from NFW profile")
    fitter = GalaxyRotationCurveFitter()
    fitter.generate_sample_data(r_min=2, r_max=35, n_points=25, noise=8, model='nfw')
    fitter.fit_both()
    fitter.print_results()
    fitter.plot('nfw_example.png', show=False)
    print(f"Plot saved: nfw_example.png\n")
    
    print("Example 2: Fitting data generated from Burkert profile")
    fitter2 = GalaxyRotationCurveFitter()
    fitter2.generate_sample_data(r_min=2, r_max=35, n_points=25, noise=8, 
                                 model='burkert', true_params=(3.0, 20.0))
    fitter2.fit_both()
    fitter2.print_results()
    fitter2.plot('burkert_example.png', show=False)
    print(f"Plot saved: burkert_example.png")
    
    print("\n" + "=" * 65)
    print("To use your own data:")
    print("  fitter = GalaxyRotationCurveFitter()")
    print("  fitter.load_data(r, v, v_err)  # or fitter.load_from_file('data.csv')")
    print("  fitter.fit_both()")
    print("  fitter.print_results()")
    print("  fitter.plot()")
    print("=" * 65)


if __name__ == "__main__":
    main()
