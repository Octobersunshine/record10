import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class TokamakDivertorSheath:
    def __init__(self, n0=5e19, Te=20.0, Ti=10.0, mi=3.34e-27, 
                 B_field=2.0, alpha_angle=1.5, 
                 see_model='furman', wall_material='W'):
        self.n0 = n0
        self.Te = Te
        self.Ti = Ti
        self.mi = mi
        self.B_field = B_field
        self.alpha_angle = alpha_angle * np.pi / 180.0
        
        self.wall_material = wall_material
        self.see_model = see_model
        
        self.e = 1.602e-19
        self.eps0 = 8.854e-12
        self.me = 9.109e-31
        
        self.TeV = Te * self.e
        self.TiV = Ti * self.e
        
        self.cs = np.sqrt(self.TeV / mi)
        self.lambda_D = np.sqrt(self.eps0 * self.TeV / (self.n0 * self.e**2))
        
        self.vte = np.sqrt(self.TeV / self.me)
        self.vti = np.sqrt(self.TiV / mi)
        
        self.rho_i = mi * self.vti / (self.e * B_field)
        self.omega_i = self.e * B_field / mi
        
        self.sin_alpha = np.sin(self.alpha_angle)
        self.cos_alpha = np.cos(self.alpha_angle)
        self.tan_alpha = np.tan(self.alpha_angle)
        
        self._init_see_parameters()
        
        self.x_mps = None
        self.phi_mps = None
        self.ne_mps = None
        self.ni_mps = None
        self.vpar_mps = None
        self.vperp_mps = None
        self.M_mps = None
        
        self.x_sheath = None
        self.phi_sheath = None
        self.ne_sheath = None
        self.ni_sheath = None
        self.vi_sheath = None
        self.E_sheath = None
        self.M_sheath = None
        self.n_se = None
        
        self.x_combined = None
        self.phi_combined = None
        self.ne_combined = None
        self.ni_combined = None
        self.vi_combined = None
        self.E_combined = None
        self.M_combined = None
        
        self.heat_flux_ion = None
        self.heat_flux_electron = None
        self.heat_flux_radiation = None
        self.heat_flux_total = None
        
        self.x_match = 0.0
        self.phi_bohm = -0.5 * self.Te
    
    def _init_see_parameters(self):
        materials = {
            'W': {'Emax': 250, 'delta_max': 1.2, 'E1': 10, 'delta1': 0.5, 'R': 0.05},
            'C': {'Emax': 300, 'delta_max': 1.0, 'E1': 15, 'delta1': 0.4, 'R': 0.08},
            'Be': {'Emax': 200, 'delta_max': 0.8, 'E1': 8, 'delta1': 0.35, 'R': 0.03},
            'Mo': {'Emax': 220, 'delta_max': 1.1, 'E1': 12, 'delta1': 0.45, 'R': 0.06}
        }
        
        if self.wall_material in materials:
            params = materials[self.wall_material]
            self.Emax = params['Emax']
            self.delta_max = params['delta_max']
            self.E1 = params['E1']
            self.delta1 = params['delta1']
            self.R_backscatter = params['R']
        else:
            self.Emax = 250
            self.delta_max = 1.2
            self.E1 = 10
            self.delta1 = 0.5
            self.R_backscatter = 0.05
    
    def see_yield(self, E_eV):
        if self.see_model == 'furman':
            return self._see_furman(E_eV)
        elif self.see_model == 'vdovicheva':
            return self._see_vdovicheva(E_eV)
        else:
            return self._see_simple(E_eV)
    
    def _see_simple(self, E_eV):
        if E_eV <= 0:
            return 0.0
        
        x = E_eV / self.Emax
        delta = self.delta_max * 1.56 * np.sqrt(x) / (1.0 + 0.56 * x + 0.12 * x**2 + 0.01 * x**3)
        
        return np.minimum(delta, self.delta_max * (1 - self.R_backscatter))
    
    def _see_furman(self, E_eV):
        if E_eV <= 0:
            return 0.0
        
        E_norm = E_eV / self.Emax
        
        delta_true = self.delta_max * 1.4 * np.sqrt(E_norm) / (1.0 + 0.4 * E_norm + 0.1 * E_norm**2)
        
        delta_backscatter = self.R_backscatter * np.exp(-(E_eV - 100)**2 / (2 * 200**2)) if E_eV > 50 else 0
        
        return delta_true + delta_backscatter
    
    def _see_vdovicheva(self, E_eV):
        if E_eV <= 0:
            return 0.0
        
        s = np.log(self.Emax / self.E1) / np.log(self.delta_max / self.delta1)
        
        if E_eV <= self.Emax:
            delta = self.delta_max * (E_eV / self.Emax)**s
        else:
            delta = self.delta_max * (E_eV / self.Emax)**(-0.35)
        
        return delta
    
    def see_energy_spectrum(self, E_incident, num_points=100):
        E_se = np.linspace(0, E_incident, num_points)
        spec = np.zeros_like(E_se)
        
        true_mask = E_se < 0.1 * E_incident
        spec[true_mask] = 2.0 * E_se[true_mask] / (0.1 * E_incident)**2
        
        back_mask = E_se >= 0.1 * E_incident
        spec[back_mask] = 0.1 / E_incident
        
        return E_se, spec
    
    def _mps_ode(self, x, y):
        phi, vpar, vperp = y
        
        B_par = self.B_field * self.cos_alpha
        B_perp = self.B_field * self.sin_alpha
        
        ne = self.n0 * np.exp(phi / self.Te)
        
        energy_par = 0.5 * self.mi * vpar**2
        energy_perp = 0.5 * self.mi * vperp**2
        total_energy = energy_par + energy_perp + self.e * phi
        
        ni = self.n0 * (1.0 - self.e * phi / (total_energy / self.n0 + self.e * phi))
        ni = np.maximum(ni, 1e10)
        
        dvpar_dx = (self.e / self.mi) * (phi / max(abs(phi), 1e-10)) * (vperp / max(vpar, 1.0))
        
        dphi_dx = -(self.e / self.eps0) * (ni - ne) * self.lambda_D**2 / self.Te
        
        dvperp_dx = -dvpar_dx * (vpar / max(vperp, 1.0))
        
        return [dphi_dx, dvpar_dx, dvperp_dx]
    
    def solve_magnetic_presheath(self, n_points=300):
        L_mps = 5.0 * self.rho_i * self.sin_alpha
        
        x_span = [0, L_mps]
        x_eval = np.linspace(0, L_mps, n_points)
        
        phi0 = 0.0
        vpar0 = 0.1 * self.cs
        vperp0 = self.vti * 0.5
        
        y0 = [phi0, vpar0, vperp0]
        
        sol = solve_ivp(
            self._mps_ode,
            x_span,
            y0,
            t_eval=x_eval,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        x_vals = sol.t
        phi_vals = sol.y[0]
        vpar_vals = sol.y[1]
        vperp_vals = sol.y[2]
        
        M_vals = np.sqrt(vpar_vals**2 + vperp_vals**2) / self.cs
        ne_vals = self.n0 * np.exp(phi_vals / self.Te)
        ni_vals = self.n0 / np.sqrt(1.0 - 2.0 * phi_vals / self.Te)
        ni_vals = np.maximum(ni_vals, 1e10)
        
        self.x_mps = x_vals
        self.phi_mps = phi_vals
        self.ne_mps = ne_vals
        self.ni_mps = ni_vals
        self.vpar_mps = vpar_vals
        self.vperp_mps = vperp_vals
        self.M_mps = M_vals
        
        self.x_match = x_vals[-1]
        
        return {
            'x': x_vals,
            'phi': phi_vals,
            'ne': ne_vals,
            'ni': ni_vals,
            'vpar': vpar_vals,
            'vperp': vperp_vals,
            'M': M_vals
        }
    
    def _sheath_ode_see(self, x, y):
        phi, dphi_dx = y
        
        M_sq = 1.0 - 2.0 * phi / self.Te
        M_sq = np.maximum(M_sq, 1e-10)
        
        ne_primary = self.n0 * np.exp(phi / self.Te)
        
        if phi < 0:
            E_impact = -phi
            delta = self.see_yield(E_impact)
            
            if delta > 0 and delta < 1:
                n_se = ne_primary * delta / (1.0 - delta)
            else:
                n_se = ne_primary * 0.5
        else:
            n_se = 0
        
        ne_total = ne_primary + n_se
        
        ni = self.n0 / np.sqrt(M_sq)
        
        d2phi_dx2 = (self.e / self.eps0) * (ne_total - ni)
        
        return [dphi_dx, d2phi_dx2]
    
    def solve_sheath_see(self, phi_wall=-100.0, n_points=500, dphi_guess=None):
        phi_bohm = self.phi_bohm
        
        if dphi_guess is None:
            dphi_guess = -0.5 * self.Te / self.lambda_D
        
        L_sheath_guess = 20.0 * self.lambda_D
        
        def residual(dphi_guess_val):
            x_span = [0, L_sheath_guess]
            x_eval = np.linspace(0, L_sheath_guess, n_points)
            
            y0 = [phi_bohm, dphi_guess_val]
            
            sol = solve_ivp(
                self._sheath_ode_see,
                x_span,
                y0,
                t_eval=x_eval,
                method='RK45',
                rtol=1e-8,
                atol=1e-10
            )
            
            phi_wall_calc = sol.y[0, -1]
            
            return phi_wall_calc - phi_wall
        
        dphi_low = -20.0 * self.Te / self.lambda_D
        dphi_high = -0.01 * self.Te / self.lambda_D
        
        max_iter = 100
        tol = 1e-6
        
        for i in range(max_iter):
            dphi_mid = (dphi_low + dphi_high) / 2.0
            
            res_low = residual(dphi_low)
            res_mid = residual(dphi_mid)
            
            if abs(res_mid) < tol:
                break
            
            if res_low * res_mid < 0:
                dphi_high = dphi_mid
            else:
                dphi_low = dphi_mid
        
        x_span = [0, L_sheath_guess]
        x_eval = np.linspace(0, L_sheath_guess, n_points)
        
        y0 = [phi_bohm, dphi_mid]
        
        sol = solve_ivp(
            self._sheath_ode_see,
            x_span,
            y0,
            t_eval=x_eval,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        x_vals = sol.t + self.x_match
        phi_vals = sol.y[0]
        dphi_vals = sol.y[1]
        
        M_sq = 1.0 - 2.0 * phi_vals / self.Te
        M_sq = np.maximum(M_sq, 1e-10)
        M_vals = np.sqrt(M_sq)
        
        ne_primary = self.n0 * np.exp(phi_vals / self.Te)
        
        n_se_vals = np.zeros_like(phi_vals)
        for i, p in enumerate(phi_vals):
            if p < 0:
                E_impact = -p
                delta = self.see_yield(E_impact)
                if delta > 0 and delta < 1:
                    n_se_vals[i] = ne_primary[i] * delta / (1.0 - delta)
        
        ne_vals = ne_primary + n_se_vals
        ni_vals = self.n0 / M_vals
        vi_vals = M_vals * self.cs
        E_vals = -dphi_vals
        
        idx = np.where(phi_vals >= phi_wall)[0]
        
        self.x_sheath = x_vals[idx]
        self.phi_sheath = phi_vals[idx]
        self.ne_sheath = ne_vals[idx]
        self.ni_sheath = ni_vals[idx]
        self.vi_sheath = vi_vals[idx]
        self.E_sheath = E_vals[idx]
        self.M_sheath = M_vals[idx]
        self.n_se = n_se_vals[idx]
        
        return {
            'x': self.x_sheath,
            'phi': self.phi_sheath,
            'ne': self.ne_sheath,
            'ni': self.ni_sheath,
            'vi': self.vi_sheath,
            'E': self.E_sheath,
            'M': self.M_sheath,
            'n_se': self.n_se
        }
    
    def simulate_divertor_sheath(self, phi_wall=-100.0, n_mps=300, n_sheath=600):
        print("=" * 70)
        print("TOKAMAK DIVERTOR SHEATH SIMULATION")
        print("=" * 70)
        
        print(f"\nPlasma Parameters:")
        print(f"  Plasma density n0: {self.n0:.2e} m^-3")
        print(f"  Electron temperature Te: {self.Te} eV")
        print(f"  Ion temperature Ti: {self.Ti} eV")
        print(f"  Magnetic field B: {self.B_field} T")
        print(f"  Field angle α: {self.alpha_angle*180/np.pi:.1f}°")
        print(f"  Wall material: {self.wall_material}")
        
        print(f"\nDerived Parameters:")
        print(f"  Ion sound speed cs: {self.cs:.2e} m/s")
        print(f"  Debye length λ_D: {self.lambda_D:.2e} m")
        print(f"  Ion gyroradius ρ_i: {self.rho_i:.2e} m")
        print(f"  Ion gyro-frequency ω_i: {self.omega_i:.2e} rad/s")
        
        print("\nSolving Magnetic Presheath (MPS)...")
        mps_result = self.solve_magnetic_presheath(n_points=n_mps)
        print(f"  MPS length: {self.x_match:.2e} m ({self.x_match/self.rho_i:.2f} ρ_i)")
        print(f"  Potential drop across MPS: {self.phi_mps[0] - self.phi_mps[-1]:.2f} V")
        
        print("\nSolving Sheath with Secondary Electron Emission...")
        sheath_result = self.solve_sheath_see(phi_wall=phi_wall, n_points=n_sheath)
        print(f"  Sheath length: {self.x_sheath[-1] - self.x_match:.2e} m")
        print(f"  SEE yield at wall: {self.see_yield(abs(phi_wall)):.3f}")
        
        print("\nCombining solutions...")
        self._combine_solutions()
        
        print("\nCalculating heat fluxes...")
        self.calculate_heat_fluxes()
        
        print("\nSimulation complete!")
        print("=" * 70)
        
        return self.get_combined_results()
    
    def _combine_solutions(self):
        x_pre = self.x_mps[:-1]
        phi_pre = self.phi_mps[:-1]
        ne_pre = self.ne_mps[:-1]
        ni_pre = self.ni_mps[:-1]
        vi_pre = np.sqrt(self.vpar_mps[:-1]**2 + self.vperp_mps[:-1]**2)
        M_pre = self.M_mps[:-1]
        E_pre = -np.gradient(phi_pre, x_pre)
        
        x_she = self.x_sheath
        phi_she = self.phi_sheath
        ne_she = self.ne_sheath
        ni_she = self.ni_sheath
        vi_she = self.vi_sheath
        E_she = self.E_sheath
        M_she = self.M_sheath
        
        self.x_combined = np.concatenate([x_pre, x_she])
        self.phi_combined = np.concatenate([phi_pre, phi_she])
        self.ne_combined = np.concatenate([ne_pre, ne_she])
        self.ni_combined = np.concatenate([ni_pre, ni_she])
        self.vi_combined = np.concatenate([vi_pre, vi_she])
        self.E_combined = np.concatenate([E_pre, E_she])
        self.M_combined = np.concatenate([M_pre, M_she])
    
    def get_combined_results(self):
        return {
            'x': self.x_combined,
            'phi': self.phi_combined,
            'ne': self.ne_combined,
            'ni': self.ni_combined,
            'vi': self.vi_combined,
            'E': self.E_combined,
            'M': self.M_combined,
            'x_match': self.x_match,
            'heat_flux_ion': self.heat_flux_ion,
            'heat_flux_electron': self.heat_flux_electron,
            'heat_flux_total': self.heat_flux_total
        }
    
    def calculate_heat_fluxes(self):
        if self.x_sheath is None:
            return None
        
        phi_wall = self.phi_sheath[-1]
        V_sheath = abs(phi_wall)
        
        gamma_i = 0.5 * self.mi * self.ni_sheath[-1] * self.vi_sheath[-1]**3
        E_i_surface = 0.5 * self.mi * self.vi_sheath[-1]**2 + self.e * V_sheath
        q_i = self.ni_sheath[-1] * self.vi_sheath[-1] * (2 * self.TeV + E_i_surface)
        q_i = q_i / 1e6
        
        E_e_impact = V_sheath
        delta = self.see_yield(E_e_impact)
        
        n_e_wall = self.ne_sheath[-1]
        
        if self.sin_alpha > 0.01:
            n_e_incident = self.n0 * (self.vte / (2 * np.sqrt(np.pi))) * np.exp(-V_sheath / self.Te)
        else:
            n_e_incident = self.n0 * (self.vte / (2 * np.sqrt(np.pi))) * np.exp(-V_sheath / self.Te)
        
        q_e_primary = n_e_incident * (2 * self.TeV + self.e * V_sheath)
        
        E_se_avg = 2.0
        q_se_cooling = n_e_incident * delta * self.e * E_se_avg
        
        q_e = (q_e_primary - q_se_cooling) / 1e6
        
        q_rad = 0.1 * (q_i + q_e)
        
        q_total = q_i + q_e + q_rad
        
        self.heat_flux_ion = q_i
        self.heat_flux_electron = q_e
        self.heat_flux_radiation = q_rad
        self.heat_flux_total = q_total
        
        return {
            'ion': q_i,
            'electron': q_e,
            'radiation': q_rad,
            'total': q_total,
            'see_yield': delta
        }
    
    def plot_divertor_results(self, save_path=None):
        if self.x_combined is None:
            raise ValueError("Simulation not run yet. Call simulate_divertor_sheath() first.")
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.x_combined * 1e3, self.phi_combined / self.Te, 'b-', linewidth=2)
        ax1.axvline(self.x_match * 1e3, color='k', linestyle=':', linewidth=2, label='MPS-Sheath Boundary')
        ax1.set_xlabel('Position x (mm)')
        ax1.set_ylabel('Potential φ / Te')
        ax1.set_title('Electric Potential')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.x_combined * 1e3, self.ne_combined / self.n0, 'r--', linewidth=2, label='Total Electrons')
        ax2.plot(self.x_combined * 1e3, self.ni_combined / self.n0, 'b-', linewidth=2, label='Ions')
        if self.n_se is not None:
            x_she_mm = self.x_sheath * 1e3
            ax2.plot(x_she_mm, self.n_se / self.n0, 'g-.', linewidth=2, label='Secondary Electrons')
        ax2.axvline(self.x_match * 1e3, color='k', linestyle=':', linewidth=2)
        ax2.set_xlabel('Position x (mm)')
        ax2.set_ylabel('Density n / n0')
        ax2.set_title('Density Profiles')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(self.x_combined * 1e3, self.M_combined, 'g-', linewidth=2)
        ax3.axvline(self.x_match * 1e3, color='k', linestyle=':', linewidth=2)
        ax3.axhline(1.0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax3.set_xlabel('Position x (mm)')
        ax3.set_ylabel('Mach Number M')
        ax3.set_title('Ion Mach Number')
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(self.x_combined * 1e3, self.E_combined / 1e6, 'm-', linewidth=2)
        ax4.axvline(self.x_match * 1e3, color='k', linestyle=':', linewidth=2)
        ax4.set_xlabel('Position x (mm)')
        ax4.set_ylabel('Electric Field E (MV/m)')
        ax4.set_title('Electric Field')
        ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(gs[1, 1])
        charge_density = self.e * (self.ni_combined - self.ne_combined)
        ax5.plot(self.x_combined * 1e3, charge_density, 'c-', linewidth=2)
        ax5.axvline(self.x_match * 1e3, color='k', linestyle=':', linewidth=2)
        ax5.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax5.set_xlabel('Position x (mm)')
        ax5.set_ylabel('Charge Density ρ (C/m³)')
        ax5.set_title('Space Charge Density')
        ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(gs[1, 2])
        E_range = np.linspace(0, 200, 100)
        delta_vals = [self.see_yield(E) for E in E_range]
        ax6.plot(E_range, delta_vals, 'r-', linewidth=2)
        ax6.axhline(1.0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='δ=1')
        ax6.set_xlabel('Electron Energy (eV)')
        ax6.set_ylabel('SEE Yield δ')
        ax6.set_title(f'SEE Yield ({self.wall_material})')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        ax7 = fig.add_subplot(gs[2, 0:2])
        if self.heat_flux_ion is not None:
            fluxes = [self.heat_flux_ion, self.heat_flux_electron, self.heat_flux_radiation]
            labels = ['Ion', 'Electron', 'Radiation']
            colors = ['#ff7f0e', '#1f77b4', '#2ca02c']
            bars = ax7.bar(labels, fluxes, color=colors, edgecolor='black', linewidth=1.5)
            
            for bar, flux in zip(bars, fluxes):
                height = bar.get_height()
                ax7.text(bar.get_x() + bar.get_width()/2., height,
                        f'{flux:.2f} MW/m²',
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            ax7.set_ylabel('Heat Flux (MW/m²)')
            ax7.set_title(f'Wall Heat Flux Components (Total: {self.heat_flux_total:.2f} MW/m²)')
            ax7.grid(True, alpha=0.3, axis='y')
        
        ax8 = fig.add_subplot(gs[2, 2])
        angles = np.linspace(0.5, 5.0, 20)
        q_total_vs_angle = []
        for a in angles:
            temp_sim = TokamakDivertorSheath(
                n0=self.n0, Te=self.Te, Ti=self.Ti, mi=self.mi,
                B_field=self.B_field, alpha_angle=a,
                wall_material=self.wall_material
            )
            temp_sim.simulate_divertor_sheath(phi_wall=-100, n_mps=100, n_sheath=200)
            q_total_vs_angle.append(temp_sim.heat_flux_total)
        
        ax8.plot(angles, q_total_vs_angle, 'b-o', linewidth=2, markersize=6)
        ax8.set_xlabel('Magnetic Field Angle (degrees)')
        ax8.set_ylabel('Total Heat Flux (MW/m²)')
        ax8.set_title('Heat Flux vs Field Angle')
        ax8.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def print_divertor_summary(self):
        print("\n" + "=" * 70)
        print("DIVERTOR SHEATH SIMULATION SUMMARY")
        print("=" * 70)
        
        print(f"\nPlasma Parameters:")
        print(f"  Plasma density n0: {self.n0:.2e} m^-3")
        print(f"  Electron temperature Te: {self.Te} eV")
        print(f"  Ion temperature Ti: {self.Ti} eV")
        print(f"  Magnetic field B: {self.B_field} T")
        print(f"  Field incidence angle: {self.alpha_angle*180/np.pi:.2f}°")
        print(f"  Wall material: {self.wall_material}")
        
        print(f"\nKey Scales:")
        print(f"  Ion sound speed cs: {self.cs:.2e} m/s")
        print(f"  Debye length λ_D: {self.lambda_D:.2e} m")
        print(f"  Ion gyroradius ρ_i: {self.rho_i:.2e} m")
        print(f"  ρ_i / λ_D: {self.rho_i / self.lambda_D:.1f}")
        
        if self.x_mps is not None:
            print(f"\nMagnetic Presheath (MPS):")
            print(f"  Length: {self.x_match:.2e} m ({self.x_match/self.rho_i:.2f} ρ_i)")
            print(f"  Potential drop: {self.phi_mps[0] - self.phi_mps[-1]:.2f} V")
        
        if self.x_sheath is not None:
            print(f"\nDebye Sheath:")
            print(f"  Length: {self.x_sheath[-1] - self.x_match:.2e} m")
            print(f"  Wall potential: {self.phi_sheath[-1]:.2f} V")
            print(f"  Ion density at wall: {self.ni_sheath[-1]:.2e} m^-3")
            print(f"  Electron density at wall: {self.ne_sheath[-1]:.2e} m^-3")
            print(f"  SEE yield at wall: {self.see_yield(abs(self.phi_sheath[-1])):.3f}")
            print(f"  Max electric field: {np.max(np.abs(self.E_sheath)):.2e} V/m")
        
        if self.heat_flux_total is not None:
            print(f"\nWall Heat Fluxes:")
            print(f"  Ion heat flux: {self.heat_flux_ion:.3f} MW/m²")
            print(f"  Electron heat flux: {self.heat_flux_electron:.3f} MW/m²")
            print(f"  Radiation heat flux: {self.heat_flux_radiation:.3f} MW/m²")
            print(f"  Total heat flux: {self.heat_flux_total:.3f} MW/m²")
            print(f"  Ion/electron ratio: {self.heat_flux_ion/self.heat_flux_electron:.2f}")
        
        print("\n" + "=" * 70)


def run_divertor_example():
    simulator = TokamakDivertorSheath(
        n0=5e19,
        Te=20.0,
        Ti=10.0,
        mi=3.34e-27,
        B_field=2.5,
        alpha_angle=1.5,
        wall_material='W',
        see_model='furman'
    )
    
    results = simulator.simulate_divertor_sheath(
        phi_wall=-80.0,
        n_mps=300,
        n_sheath=600
    )
    
    simulator.print_divertor_summary()
    
    print("\nGenerating comprehensive plots...")
    simulator.plot_divertor_results(save_path='divertor_sheath_analysis.png')
    
    return simulator, results


if __name__ == "__main__":
    simulator, results = run_divertor_example()
