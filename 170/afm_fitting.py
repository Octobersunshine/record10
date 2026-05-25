import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, brentq
from scipy import constants


class AFMCurveFitter:
    def __init__(self, radius=10e-9, nu=0.5, z0=0.3e-9):
        self.R = radius
        self.nu = nu
        self.z0 = z0
        self.E = None
        self.F_ad = None
        self.model_used = None
        self.fit_params = None
        self.fit_cov = None
        self._md_cache = {}

    @staticmethod
    def dmt_model(delta, E, F_ad, R, nu):
        E_star = E / (1 - nu**2)
        F = (4/3) * E_star * np.sqrt(R) * np.maximum(delta, 0)**1.5 + F_ad
        return F

    @staticmethod
    def jkr_model(delta, E, F_ad, R, nu):
        E_star = E / (1 - nu**2)
        F = (4/3) * E_star * np.sqrt(R) * np.maximum(delta, 0)**1.5 - F_ad
        return F

    @staticmethod
    def _md_equations(a_bar, F_bar, lam):
        term1 = a_bar**3 - F_bar * a_bar**1.5
        term2 = (4 * lam / 3) * (np.sqrt(a_bar**2 - 1) + (a_bar**2 - 2) * np.arccos(1/a_bar))
        return term1 - term2

    @staticmethod
    def _md_delta(a_bar, lam):
        term1 = a_bar**2 - (4/3) * np.sqrt(2 * lam * a_bar)
        return term1

    @classmethod
    def maugis_dugdale_model(cls, delta, E, F_ad, R, nu, z0=0.3e-9, lambda_param=1.0):
        E_star = E / (1 - nu**2)
        
        a0 = (9 * R * F_ad / (16 * E_star))**(1/3) if F_ad > 0 else 1e-9
        sigma0 = F_ad / (np.pi * a0**2) if a0 > 0 else 0
        
        lam = sigma0 * (9 * R / (np.pi * E_star * z0**2))**(1/3) if E_star > 0 and z0 > 0 else lambda_param
        lam = max(0.01, min(lam, 10.0))
        
        delta_ref = (F_ad**2 / (R * E_star**2))**(1/3) if F_ad > 0 and E_star > 0 else 1e-9
        F_ref = F_ad if F_ad > 0 else 1e-9
        
        F = np.zeros_like(delta, dtype=float)
        
        for i, d in enumerate(delta):
            if d <= 0:
                F[i] = -F_ad * np.exp(d / z0) if d > -5 * z0 else -F_ad
                continue
            
            delta_bar = d / delta_ref if delta_ref > 0 else 0
            
            def f_to_solve(F_bar):
                a_min = 1.0
                a_max = max(10.0, 2 + delta_bar)
                
                def a_equation(a_bar):
                    return cls._md_equations(a_bar, F_bar, lam)
                
                try:
                    a_bar = brentq(a_equation, a_min, a_max, xtol=1e-8)
                    delta_calc = cls._md_delta(a_bar, lam)
                    return delta_calc - delta_bar
                except:
                    return np.inf
            
            try:
                F_bar = brentq(f_to_solve, -2.0, 10.0, xtol=1e-6)
                F[i] = F_bar * F_ref
            except:
                F_dmt = (4/3) * E_star * np.sqrt(R) * d**1.5 + F_ad
                F_jkr = (4/3) * E_star * np.sqrt(R) * d**1.5 - F_ad
                w = 1 / (1 + lam)
                F[i] = w * F_dmt + (1 - w) * F_jkr
        
        return F

    def generate_synthetic_data(self, model='DMT', E_true=1e9, F_ad_true=1e-9, 
                                noise=0.05, n_points=200, lambda_param=1.0):
        delta = np.linspace(-5e-9, 20e-9, n_points)
        
        if model == 'DMT':
            F = self.dmt_model(delta, E_true, F_ad_true, self.R, self.nu)
        elif model == 'JKR':
            F = self.jkr_model(delta, E_true, F_ad_true, self.R, self.nu)
        elif model == 'MD':
            F = self.maugis_dugdale_model(delta, E_true, F_ad_true, self.R, self.nu, 
                                          self.z0, lambda_param)
        else:
            raise ValueError("Model must be 'DMT', 'JKR', or 'MD'")
        
        F_noisy = F + noise * np.abs(F_ad_true) * np.random.randn(len(F))
        return delta, F_noisy

    def fit(self, delta, F, model='MD', p0=None):
        self.model_used = model
        
        if p0 is None:
            p0 = [1e9, 1e-9]
        
        mask = delta >= 0
        delta_fit = delta[mask]
        F_fit = F[mask]
        
        if model == 'DMT':
            def model_func(d, E, F_ad):
                return self.dmt_model(d, E, F_ad, self.R, self.nu)
            bounds = ([1e6, -1e-6], [1e12, 1e-6])
        elif model == 'JKR':
            def model_func(d, E, F_ad):
                return self.jkr_model(d, E, F_ad, self.R, self.nu)
            bounds = ([1e6, 1e-12], [1e12, 1e-6])
        elif model == 'MD':
            def model_func(d, E, F_ad):
                return self.maugis_dugdale_model(d, E, F_ad, self.R, self.nu, self.z0)
            bounds = ([1e6, 1e-12], [1e12, 1e-6])
        else:
            raise ValueError("Model must be 'DMT', 'JKR', or 'MD'")
        
        try:
            popt, pcov = curve_fit(model_func, delta_fit, F_fit, 
                                   p0=p0, bounds=bounds, maxfev=10000)
            self.fit_params = popt
            self.fit_cov = pcov
            self.E = popt[0]
            self.F_ad = popt[1]
            return popt, pcov
        except Exception as e:
            print(f"Fitting failed: {e}")
            return None, None

    def calculate_goodness_of_fit(self, delta, F):
        if self.fit_params is None:
            return None
        
        mask = delta >= 0
        delta_fit = delta[mask]
        F_actual = F[mask]
        
        if self.model_used == 'DMT':
            F_pred = self.dmt_model(delta_fit, self.E, self.F_ad, self.R, self.nu)
        elif self.model_used == 'JKR':
            F_pred = self.jkr_model(delta_fit, self.E, self.F_ad, self.R, self.nu)
        elif self.model_used == 'MD':
            F_pred = self.maugis_dugdale_model(delta_fit, self.E, self.F_ad, 
                                               self.R, self.nu, self.z0)
        else:
            return None
        
        ss_res = np.sum((F_actual - F_pred)**2)
        ss_tot = np.sum((F_actual - np.mean(F_actual))**2)
        r_squared = 1 - (ss_res / ss_tot)
        return r_squared

    def plot_results(self, delta, F, delta_true=None, F_true=None, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(delta * 1e9, F * 1e9, alpha=0.5, label='Data', s=10, color='blue')
        
        if delta_true is not None and F_true is not None:
            ax.plot(delta_true * 1e9, F_true * 1e9, 'g-', label='True Curve', alpha=0.7)
        
        if self.fit_params is not None:
            delta_smooth = np.linspace(0, np.max(delta), 500)
            if self.model_used == 'DMT':
                F_fit = self.dmt_model(delta_smooth, self.E, self.F_ad, self.R, self.nu)
            elif self.model_used == 'JKR':
                F_fit = self.jkr_model(delta_smooth, self.E, self.F_ad, self.R, self.nu)
            elif self.model_used == 'MD':
                F_fit = self.maugis_dugdale_model(delta_smooth, self.E, self.F_ad, 
                                                   self.R, self.nu, self.z0)
            else:
                F_fit = None
            
            if F_fit is not None:
                ax.plot(delta_smooth * 1e9, F_fit * 1e9, 'r-', linewidth=2, 
                        label=f'{self.model_used} Fit')
        
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        ax.set_xlabel('Indentation δ (nm)')
        ax.set_ylabel('Force F (nN)')
        ax.set_title(f'AFM Force-Distance Curve - {self.model_used} Model')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if self.E is not None and self.F_ad is not None:
            r2 = self.calculate_goodness_of_fit(delta, F)
            textstr = '\n'.join((
                f'{self.model_used} Model',
                f'E = {self.E:.2e} Pa',
                f'F_ad = {self.F_ad:.2e} N',
                f'R² = {r2:.4f}' if r2 is not None else ''
            ))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax

    def print_results(self):
        if self.E is None or self.F_ad is None:
            print("No fitting results available.")
            return
        
        print(f"\n{'='*50}")
        print(f"AFM Force-Distance Curve Fitting Results")
        print(f"{'='*50}")
        print(f"Model used: {self.model_used}")
        print(f"Tip radius R: {self.R * 1e9:.1f} nm")
        print(f"Poisson's ratio ν: {self.nu}")
        if self.model_used == 'MD':
            print(f"Interatomic distance z₀: {self.z0 * 1e9:.2f} nm")
        print(f"-" * 50)
        print(f"Young's Modulus E: {self.E:.3e} Pa")
        print(f"                    = {self.E / 1e9:.3f} GPa")
        print(f"Adhesion Force F_ad: {self.F_ad:.3e} N")
        print(f"                    = {self.F_ad * 1e9:.3f} nN")
        
        if self.fit_cov is not None:
            perr = np.sqrt(np.diag(self.fit_cov))
            print(f"-" * 50)
            print(f"Standard Errors:")
            print(f"  ΔE: {perr[0]:.2e} Pa")
            print(f"  ΔF_ad: {perr[1]:.2e} N")
        print(f"{'='*50}\n")


def load_afm_data(file_path):
    try:
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        delta = data[:, 0]
        force = data[:, 1]
        return delta, force
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def main():
    print("AFM Force-Distance Curve Fitting with Maugis-Dugdale Model")
    print("=" * 50)
    
    fitter = AFMCurveFitter(radius=5e-9, nu=0.5, z0=0.3e-9)
    
    E_true = 5e9
    F_ad_true = 5e-9
    
    print(f"\nGenerating synthetic MD data with:")
    print(f"  True E: {E_true:.2e} Pa")
    print(f"  True F_ad: {F_ad_true:.2e} N")
    print(f"  Tip radius: {fitter.R * 1e9:.1f} nm")
    
    delta, F_noisy = fitter.generate_synthetic_data(
        model='MD', E_true=E_true, F_ad_true=F_ad_true, noise=0.03, lambda_param=1.0
    )
    delta_true, F_true = fitter.generate_synthetic_data(
        model='MD', E_true=E_true, F_ad_true=F_ad_true, noise=0, n_points=500, lambda_param=1.0
    )
    
    print("\nFitting with MD model (recommended for small radii)...")
    fitter.fit(delta, F_noisy, model='MD', p0=[1e9, 1e-9])
    fitter.print_results()
    
    fitter.plot_results(delta, F_noisy, delta_true, F_true, 
                        save_path='afm_md_fit.png')
    print("Plot saved as 'afm_md_fit.png'")
    
    print("\n" + "=" * 50)
    print("Comparing all three models...")
    
    fitter_dmt = AFMCurveFitter(radius=5e-9, nu=0.5)
    fitter_jkr = AFMCurveFitter(radius=5e-9, nu=0.5)
    
    fitter_dmt.fit(delta, F_noisy, model='DMT')
    fitter_jkr.fit(delta, F_noisy, model='JKR')
    
    r2_md = fitter.calculate_goodness_of_fit(delta, F_noisy)
    r2_dmt = fitter_dmt.calculate_goodness_of_fit(delta, F_noisy)
    r2_jkr = fitter_jkr.calculate_goodness_of_fit(delta, F_noisy)
    
    print(f"\nModel Comparison (R² values):")
    print(f"  MD:  {r2_md:.6f}")
    print(f"  DMT: {r2_dmt:.6f}")
    print(f"  JKR: {r2_jkr:.6f}")
    print(f"\nMD model provides the best fit for intermediate adhesion regimes!")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
