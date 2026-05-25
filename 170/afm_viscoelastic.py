import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


class ViscoelasticAFM:
    def __init__(self, radius=10e-9, nu=0.5):
        self.R = radius
        self.nu = nu
        self.fit_params = None
        self.fit_cov = None
        self.relaxation_times = None
        self.relaxation_moduli = None
        self.model_used = None

    @staticmethod
    def hertz_contact_stiffness(delta, E, R, nu):
        E_star = E / (1 - nu**2)
        k = 2 * E_star * np.sqrt(R * delta)
        return k

    @staticmethod
    def sls_creep(t, E0, Einf, tau, delta0, R, nu):
        E_star_0 = E0 / (1 - nu**2)
        E_star_inf = Einf / (1 - nu**2)
        
        a0 = np.sqrt(R * delta0)
        k0 = 2 * E_star_0 * a0
        F0 = (4/3) * E_star_0 * np.sqrt(R) * delta0**1.5
        
        J_g = 1 / E_star_inf
        J_e = (1 / E_star_0) - J_g
        
        J_t = J_g + J_e * (1 - np.exp(-t / tau))
        
        delta_t = delta0 * (1 + (E_star_0 * J_e) * (1 - np.exp(-t / tau)))
        F_t = F0 * (1 - (J_e / (1 / E_star_0)) * (1 - np.exp(-t / tau)))
        
        return delta_t, F_t

    @staticmethod
    def sls_force_relaxation(t, F0, E0, Einf, tau):
        F_inf = F0 * (Einf / E0)
        F_t = F_inf + (F0 - F_inf) * np.exp(-t / tau)
        return F_t

    @staticmethod
    def generalized_maxwell_model(t, Ge, *params):
        n = len(params) // 2
        tau = np.array(params[:n])
        Gi = np.array(params[n:])
        
        G_t = Ge
        for i in range(n):
            G_t += Gi[i] * np.exp(-t / tau[i])
        
        return G_t

    @staticmethod
    def creep_compliance_sls(t, Jg, Je, tau):
        J_t = Jg + Je * (1 - np.exp(-t / tau))
        return J_t

    @staticmethod
    def creep_compliance_generalized(t, Jg, *params):
        n = len(params) // 2
        tau = np.array(params[:n])
        Ji = np.array(params[n:])
        
        J_t = Jg
        for i in range(n):
            J_t += Ji[i] * (1 - np.exp(-t / tau[i]))
        
        return J_t

    def generate_creep_data(self, t_total=10.0, n_points=500, 
                            E0=1e3, Einf=500, tau=1.0, 
                            delta0=100e-9, noise=0.02):
        t = np.linspace(0, t_total, n_points)
        
        delta_t, F_t = self.sls_creep(t, E0, Einf, tau, delta0, self.R, self.nu)
        
        noise_amp = noise * np.abs(delta_t).mean()
        delta_noisy = delta_t + noise_amp * np.random.randn(len(delta_t))
        
        return t, delta_noisy, delta_t

    def generate_relaxation_data(self, t_total=10.0, n_points=500,
                                 F0=1e-9, E0=1e3, Einf=500, tau=1.0,
                                 noise=0.02):
        t = np.linspace(0, t_total, n_points)
        
        F_t = self.sls_force_relaxation(t, F0, E0, Einf, tau)
        
        noise_amp = noise * np.abs(F_t).mean()
        F_noisy = F_t + noise_amp * np.random.randn(len(F_t))
        
        return t, F_noisy, F_t

    def fit_sls_creep(self, t, delta, p0=None):
        self.model_used = 'SLS'
        
        if p0 is None:
            p0 = [1e3, 500, 1.0, 100e-9]
        
        def model_func(t, E0, Einf, tau, delta0):
            delta_t, _ = self.sls_creep(t, E0, Einf, tau, delta0, self.R, self.nu)
            return delta_t
        
        bounds = (
            [1e1, 1e1, 1e-3, 1e-9],
            [1e6, 1e6, 1e3, 1e-5]
        )
        
        try:
            popt, pcov = curve_fit(model_func, t, delta, p0=p0, bounds=bounds, maxfev=10000)
            self.fit_params = popt
            self.fit_cov = pcov
            return popt, pcov
        except Exception as e:
            print(f"SLS creep fitting failed: {e}")
            return None, None

    def calculate_stress_relaxation_spectrum(self, t, G_t, n_modes=3, 
                                             tau_min=1e-3, tau_max=1e2):
        tau = np.logspace(np.log10(tau_min), np.log10(tau_max), n_modes)
        
        Ge_guess = G_t[-1]
        Gi_guess = np.ones(n_modes) * (G_t[0] - G_t[-1]) / n_modes
        p0 = [Ge_guess] + list(tau) + list(Gi_guess)
        
        def model_func(t, Ge, *params):
            return self.generalized_maxwell_model(t, Ge, *params)
        
        n_params = len(p0)
        bounds = (
            [0] + [tau_min]*n_modes + [0]*n_modes,
            [np.inf] + [tau_max]*n_modes + [np.inf]*n_modes
        )
        
        try:
            popt, pcov = curve_fit(model_func, t, G_t, p0=p0, bounds=bounds, maxfev=50000)
            
            Ge = popt[0]
            tau_fit = popt[1:1+n_modes]
            Gi_fit = popt[1+n_modes:]
            
            self.relaxation_times = tau_fit
            self.relaxation_moduli = Gi_fit
            
            return tau_fit, Gi_fit, Ge
        except Exception as e:
            print(f"Relaxation spectrum calculation failed: {e}")
            return None, None, None

    def extract_relaxation_spectrum_from_creep(self, t, delta, delta0, 
                                               n_modes=3, method='compliance'):
        Jg_guess = 1e-6
        tau_guess = np.logspace(-2, 1, n_modes)
        Ji_guess = np.ones(n_modes) * 1e-6
        
        p0 = [Jg_guess] + list(tau_guess) + list(Ji_guess)
        
        E_eff = (3 * delta0**0.5) / (4 * np.sqrt(self.R) * (1 - self.nu**2))
        J_t = E_eff * (delta / delta0**1.5)
        
        def model_func(t, Jg, *params):
            return self.creep_compliance_generalized(t, Jg, *params)
        
        n_params = len(p0)
        bounds = (
            [0] + [1e-4]*n_modes + [0]*n_modes,
            [np.inf] + [1e4]*n_modes + [np.inf]*n_modes
        )
        
        try:
            popt, pcov = curve_fit(model_func, t, J_t, p0=p0, bounds=bounds, maxfev=50000)
            
            Jg = popt[0]
            tau_fit = popt[1:1+n_modes]
            Ji_fit = popt[1+n_modes:]
            
            self.relaxation_times = tau_fit
            
            return tau_fit, Ji_fit, Jg
        except Exception as e:
            print(f"Spectrum extraction failed: {e}")
            return None, None, None

    def plot_creep_curve(self, t, delta, delta_true=None, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(t, delta * 1e9, alpha=0.5, label='Data', s=10, color='blue')
        
        if delta_true is not None:
            ax.plot(t, delta_true * 1e9, 'g-', label='True Curve', alpha=0.7)
        
        if self.fit_params is not None and self.model_used == 'SLS':
            E0, Einf, tau, delta0 = self.fit_params
            _, delta_fit = self.sls_creep(t, E0, Einf, tau, delta0, self.R, self.nu)
            ax.plot(t, delta_fit * 1e9, 'r-', linewidth=2, label='SLS Fit')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Indentation δ (nm)')
        ax.set_title('AFM Creep Curve - Viscoelastic Response')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        if self.fit_params is not None:
            E0, Einf, tau, delta0 = self.fit_params
            textstr = '\n'.join((
                'SLS Model Parameters',
                f'E₀ = {E0:.2e} Pa',
                f'E∞ = {Einf:.2e} Pa',
                f'τ = {tau:.3f} s',
                f'δ₀ = {delta0*1e9:.1f} nm'
            ))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax

    def plot_relaxation_spectrum(self, save_path=None):
        if self.relaxation_times is None:
            print("No relaxation spectrum available.")
            return None, None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.stem(self.relaxation_times, self.relaxation_moduli, 
                 basefmt='k-', linefmt='b-', markerfmt='bo')
        ax1.set_xscale('log')
        ax1.set_xlabel('Relaxation Time τ (s)')
        ax1.set_ylabel('Relaxation Modulus G_i (Pa)')
        ax1.set_title('Discrete Relaxation Spectrum')
        ax1.grid(True, alpha=0.3)
        
        tau_smooth = np.logspace(np.log10(self.relaxation_times.min()) - 1,
                                 np.log10(self.relaxation_times.max()) + 1, 200)
        H = np.zeros_like(tau_smooth)
        for i, tau in enumerate(self.relaxation_times):
            H += self.relaxation_moduli[i] * np.exp(-(np.log(tau_smooth) - np.log(tau))**2 / 0.1)
        
        ax2.plot(tau_smooth, H, 'r-', linewidth=2)
        ax2.set_xscale('log')
        ax2.set_xlabel('Relaxation Time τ (s)')
        ax2.set_ylabel('Relaxation Spectrum H(τ) (Pa)')
        ax2.set_title('Continuous Relaxation Spectrum')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, (ax1, ax2)

    def print_viscoelastic_results(self):
        if self.fit_params is None and self.relaxation_times is None:
            print("No viscoelastic fitting results available.")
            return
        
        print(f"\n{'='*60}")
        print(f"AFM Viscoelastic Analysis Results")
        print(f"{'='*60}")
        print(f"Tip radius R: {self.R * 1e9:.1f} nm")
        print(f"Poisson's ratio ν: {self.nu}")
        
        if self.fit_params is not None and self.model_used == 'SLS':
            E0, Einf, tau, delta0 = self.fit_params
            print(f"-" * 60)
            print(f"SLS (Standard Linear Solid) Model:")
            print(f"  Instantaneous Modulus E₀: {E0:.3e} Pa = {E0/1e3:.2f} kPa")
            print(f"  Equilibrium Modulus E∞: {Einf:.3e} Pa = {Einf/1e3:.2f} kPa")
            print(f"  Relaxation Time τ: {tau:.4f} s")
            print(f"  Initial Indentation δ₀: {delta0*1e9:.2f} nm")
            print(f"  Elastic Ratio E∞/E₀: {Einf/E0:.2%}")
        
        if self.relaxation_times is not None:
            print(f"-" * 60)
            print(f"Relaxation Time Spectrum ({len(self.relaxation_times)} modes):")
            for i, (tau, G) in enumerate(zip(self.relaxation_times, self.relaxation_moduli)):
                print(f"  Mode {i+1}: τ = {tau:.4f} s, G_i = {G:.2e} Pa")
        
        print(f"{'='*60}\n")


def load_creep_data(file_path):
    try:
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        t = data[:, 0]
        delta = data[:, 1]
        return t, delta
    except Exception as e:
        print(f"Error loading creep data: {e}")
        return None, None


def main():
    print("AFM Viscoelastic (Creep) Analysis for Biomaterials")
    print("=" * 60)
    
    ve_afm = ViscoelasticAFM(radius=20e-9, nu=0.5)
    
    print("\n1. Generating synthetic creep data...")
    E0_true = 2000
    Einf_true = 800
    tau_true = 2.0
    delta0_true = 50e-9
    
    print(f"   True parameters:")
    print(f"     E₀ = {E0_true} Pa")
    print(f"     E∞ = {Einf_true} Pa")
    print(f"     τ = {tau_true} s")
    print(f"     δ₀ = {delta0_true*1e9} nm")
    
    t, delta_noisy, delta_true = ve_afm.generate_creep_data(
        t_total=20.0, n_points=400,
        E0=E0_true, Einf=Einf_true, tau=tau_true, delta0=delta0_true,
        noise=0.02
    )
    
    print("\n2. Fitting SLS model to creep data...")
    ve_afm.fit_sls_creep(t, delta_noisy, p0=[1500, 500, 1.0, 40e-9])
    ve_afm.print_viscoelastic_results()
    
    ve_afm.plot_creep_curve(t, delta_noisy, delta_true, save_path='creep_sls_fit.png')
    print("   Creep curve plot saved as 'creep_sls_fit.png'")
    
    print("\n3. Extracting relaxation time spectrum (3 modes)...")
    tau_spec, G_spec, Ge = ve_afm.calculate_stress_relaxation_spectrum(
        t, np.linspace(2000, 800, len(t)), n_modes=3
    )
    
    if tau_spec is not None:
        ve_afm.plot_relaxation_spectrum(save_path='relaxation_spectrum.png')
        print("   Relaxation spectrum plot saved as 'relaxation_spectrum.png'")
        ve_afm.print_viscoelastic_results()
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
