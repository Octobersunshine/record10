import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class EnergyBalanceModel1D:
    def __init__(self, n_lat=36, dt=86400.0, n_years=100):
        self.n_lat = n_lat
        self.dt = dt
        self.n_years = n_years
        
        self.lat = np.linspace(-87.5, 87.5, n_lat)
        self.phi = np.deg2rad(self.lat)
        
        self.R = 6371000.0
        self.C = 2.08e8
        self.D = 0.65
        
        self.Q0 = 1368.0
        self.s2 = -0.48
        
        self.sigma = 5.67e-8
        self.epsilon = 0.61
        
        self.A = 203.3
        self.B = 2.09
        
        self.tau_lw = 0.61
        self.use_physical_olr = False
        
        self.a0 = 0.5
        self.a2 = 0.2
        self.Tf = -10.0
        
        self.enable_cloud_feedback = True
        
        self.c_base = 0.65
        self.c_lat = -0.3 * 0.5 * (3 * np.sin(self.phi)**2 - 1)
        self.c_T = -0.015
        self.c_sw_eff = 0.22
        self.c_lw_eff = 0.15
        
        self.co2_ppm = 415.0
        self.co2_forcing_scale = 5.35
        self.forcing_co2 = 0.0
        
        self.n_steps_per_year = int(365 * 86400 / dt)
        self.n_steps = n_years * self.n_steps_per_year
        
        self.time = np.arange(self.n_steps) * dt / (86400 * 365)
        
        self.T = np.zeros((self.n_steps, n_lat))
        self.T[0, :] = 288.0 - 30.0 * np.sin(self.phi)**2
        
        self.cloud_cover = np.zeros((self.n_steps, n_lat))
        self.cloud_sw_forcing = np.zeros((self.n_steps, n_lat))
        self.cloud_lw_forcing = np.zeros((self.n_steps, n_lat))
        
    def s(self, phi):
        return 1 + self.s2 * 0.5 * (3 * np.sin(phi)**2 - 1)
    
    def albedo(self, T):
        return 0.31 + 0.25 * 0.5 * (3 * np.sin(self.phi)**2 - 1)
    
    def compute_cloud_cover(self, T, step):
        T_anomaly = T - 288.0
        
        c = self.c_base + self.c_lat + self.c_T * T_anomaly
        
        c = np.clip(c, 0.2, 0.9)
        
        return c
    
    def compute_cloud_forcing(self, T, c, step):
        T_ref = 288.0
        c_ref = 0.65
        
        s_dist = self.s(self.phi)
        sw_incident = (self.Q0 / 4) * s_dist
        
        c_anomaly = c - c_ref
        
        sw_forcing = -self.c_sw_eff * sw_incident * c_anomaly
        
        sigma_T4 = self.sigma * T**4
        lw_forcing = self.c_lw_eff * sigma_T4 * c_anomaly
        
        return sw_forcing, lw_forcing
    
    def compute_co2_forcing(self, co2_ppm=None):
        if co2_ppm is None:
            co2_ppm = self.co2_ppm
        
        co2_ref = 284.0
        forcing = self.co2_forcing_scale * np.log(co2_ppm / co2_ref)
        
        return forcing
    
    def shortwave_absorption(self, T, step):
        s_dist = self.s(self.phi)
        alpha = self.albedo(T)
        
        SW = (self.Q0 / 4) * (1 - alpha) * s_dist
        
        if self.enable_cloud_feedback and step is not None:
            c = self.compute_cloud_cover(T, step)
            sw_forcing, lw_forcing = self.compute_cloud_forcing(T, c, step)
            SW = SW + sw_forcing
        
        SW = SW + self.forcing_co2
        
        return SW
    
    def longwave_emission(self, T):
        return self.A + self.B * (T - 273.15)
    
    def outgoing_longwave_radiation(self, T):
        return self.tau_lw * self.sigma * T**4
    
    def greenhouse_effect(self, T):
        sigma_T4 = self.sigma * T**4
        eps_eff = self.epsilon * (1 + 0.1 * np.sin(self.phi)**2)
        return eps_eff * sigma_T4
    
    def compute_olr_physical(self, T):
        T_ref = 288.0
        epsilon_eff = self.epsilon
        
        alpha_lapse = 0.0065
        z_eff = 5500.0
        T_atm = T - alpha_lapse * z_eff
        
        eps_surface = 0.96
        surface_emission = eps_surface * self.sigma * T**4
        
        tau_co2 = 0.56
        tau_h2o = 0.75
        tau_atm = tau_co2 * tau_h2o
        
        transmitted = tau_atm * surface_emission
        atm_emission = (1 - tau_atm**0.5) * self.sigma * T_atm**4
        
        return transmitted + atm_emission
    
    def compute_olr_calibrated(self, T):
        T_celsius = T - 273.15
        
        A_calib = 203.3
        B_calib = 2.09
        
        return A_calib + B_calib * T_celsius
    
    def compute_olr(self, T):
        if self.use_physical_olr:
            return self.compute_olr_physical(T)
        else:
            return self.compute_olr_calibrated(T)
    
    def compute_radiative_forcing(self, T, co2_factor=1.0):
        tau_co2_base = 0.56
        tau_co2_new = tau_co2_base * co2_factor
        
        olr_base = self.compute_olr(T)
        
        old_tau = self.tau_lw
        self.tau_lw = tau_co2_new
        olr_new = self.compute_olr_physical(T)
        self.tau_lw = old_tau
        
        return olr_base - olr_new
    
    def diffusion(self, T):
        phi = self.phi
        dphi = phi[1] - phi[0]
        
        dT_dphi = np.gradient(T, dphi)
        flux = -self.D * self.C * (1 - np.sin(phi)**2) * dT_dphi
        
        dflux_dphi = np.gradient(flux, dphi)
        div_F = -dflux_dphi / (self.R**2 * np.cos(phi))
        
        return div_F
    
    def step_forward(self, T_current, step):
        SW = self.shortwave_absorption(T_current, step)
        LW = self.compute_olr(T_current)
        diffusion = self.diffusion(T_current)
        
        if self.enable_cloud_feedback:
            c = self.compute_cloud_cover(T_current, step)
            sw_forcing, lw_forcing = self.compute_cloud_forcing(T_current, c, step)
            LW = LW + lw_forcing
        
        dT_dt = (SW - LW + diffusion) / self.C
        return T_current + self.dt * dT_dt
    
    def verify_energy_balance(self, T):
        SW = self.shortwave_absorption(T, 0)
        LW = self.compute_olr(T)
        diffusion = self.diffusion(T)
        
        net = SW - LW + diffusion
        
        weights = np.cos(self.phi)
        global_net = np.average(net, weights=weights)
        
        return {
            'shortwave': np.average(SW, weights=weights),
            'longwave': np.average(LW, weights=weights),
            'diffusion': np.average(diffusion, weights=weights),
            'net': global_net,
            'T_global': np.average(T, weights=weights) - 273.15
        }
    
    def run(self, co2_scenarios=None):
        print("Running 1D Energy Balance Model...")
        for i in range(self.n_steps - 1):
            if co2_scenarios is not None:
                year = i / self.n_steps_per_year
                self.co2_ppm = self.get_rcmip_co2(year, co2_scenarios)
                self.forcing_co2 = self.compute_co2_forcing()
            
            self.T[i+1, :] = self.step_forward(self.T[i, :], i)
            
            if self.enable_cloud_feedback:
                c = self.compute_cloud_cover(self.T[i+1, :], i+1)
                swf, lwf = self.compute_cloud_forcing(self.T[i+1, :], c, i+1)
                self.cloud_cover[i+1, :] = c
                self.cloud_sw_forcing[i+1, :] = swf
                self.cloud_lw_forcing[i+1, :] = lwf
            
            if (i + 1) % self.n_steps_per_year == 0:
                year = (i + 1) // self.n_steps_per_year
                if year <= self.n_years:
                    print(f"Year {year}/{self.n_years} completed")
        
        print("Model run complete!")
        return self.T
    
    def get_rcmip_co2(self, year, scenario='ssp585'):
        co2_2020 = 415.0
        
        if scenario == 'ssp119':
            growth_rate = 0.1
            peak_year = 2050
        elif scenario == 'ssp126':
            growth_rate = 0.3
            peak_year = 2070
        elif scenario == 'ssp245':
            growth_rate = 0.8
            peak_year = 2100
        elif scenario == 'ssp370':
            growth_rate = 1.5
            peak_year = 2150
        elif scenario == 'ssp585':
            growth_rate = 2.5
            peak_year = 2200
        else:
            growth_rate = 1.0
            peak_year = 2100
        
        if year < peak_year - 2020:
            co2 = co2_2020 + growth_rate * year
        else:
            co2 = co2_2020 + growth_rate * (peak_year - 2020)
        
        return co2
    
    def run_co2_doubling_experiment(self, feedback_strength=None):
        print("\n" + "="*60)
        print("CO₂ 倍增敏感性实验 (RCMIP 场景)")
        print("="*60)
        
        results = {}
        
        print("\n[1/3] 运行控制实验 (284 ppm CO₂)...")
        model_ctrl = EnergyBalanceModel1D(n_lat=self.n_lat, dt=self.dt, n_years=self.n_years)
        model_ctrl.enable_cloud_feedback = False
        model_ctrl.co2_ppm = 284.0
        model_ctrl.forcing_co2 = model_ctrl.compute_co2_forcing()
        T_ctrl = model_ctrl.run()
        
        weights = np.cos(self.phi)
        T_global_ctrl = np.average(T_ctrl[-1, :], weights=weights)
        
        results['control'] = {
            'co2': 284.0,
            'T_global': T_global_ctrl - 273.15,
            'forcing': model_ctrl.forcing_co2,
            'model': model_ctrl
        }
        
        print(f"控制实验全球平均温度: {T_global_ctrl - 273.15:.2f}°C")
        
        print(f"\n[2/3] 运行 CO₂ 加倍实验 (568 ppm) - 无云反馈...")
        model_no_cloud = EnergyBalanceModel1D(n_lat=self.n_lat, dt=self.dt, n_years=self.n_years)
        model_no_cloud.enable_cloud_feedback = False
        model_no_cloud.co2_ppm = 568.0
        model_no_cloud.forcing_co2 = model_no_cloud.compute_co2_forcing()
        forcing = model_no_cloud.forcing_co2
        print(f"CO₂ 加倍辐射强迫: {forcing:.2f} W/m²")
        
        T_no_cloud = model_no_cloud.run()
        T_global_no_cloud = np.average(T_no_cloud[-1, :], weights=weights)
        T_response_no_cloud = T_global_no_cloud - T_global_ctrl
        
        results['no_cloud_feedback'] = {
            'co2': 568.0,
            'T_global': T_global_no_cloud - 273.15,
            'T_response': T_response_no_cloud,
            'forcing': forcing,
            'model': model_no_cloud
        }
        
        print(f"无云反馈时变暖: {T_response_no_cloud:.2f}°C")
        
        print(f"\n[3/3] 运行 CO₂ 加倍实验 (568 ppm) - 有云反馈...")
        model_with_cloud = EnergyBalanceModel1D(n_lat=self.n_lat, dt=self.dt, n_years=self.n_years)
        model_with_cloud.enable_cloud_feedback = True
        model_with_cloud.co2_ppm = 568.0
        model_with_cloud.forcing_co2 = model_with_cloud.compute_co2_forcing()
        
        if feedback_strength is not None:
            model_with_cloud.c_T = feedback_strength
        
        T_with_cloud = model_with_cloud.run()
        T_global_with_cloud = np.average(T_with_cloud[-1, :], weights=weights)
        T_response_with_cloud = T_global_with_cloud - T_global_ctrl
        
        results['with_cloud_feedback'] = {
            'co2': 568.0,
            'T_global': T_global_with_cloud - 273.15,
            'T_response': T_response_with_cloud,
            'forcing': forcing,
            'cloud_feedback_strength': model_with_cloud.c_T,
            'model': model_with_cloud
        }
        
        print(f"有云反馈时变暖: {T_response_with_cloud:.2f}°C")
        
        print("\n" + "="*60)
        print("实验结果总结")
        print("="*60)
        print(f"控制实验 (284 ppm):   {results['control']['T_global']:.2f}°C")
        print(f"CO₂加倍 (无云反馈):   {results['no_cloud_feedback']['T_global']:.2f}°C")
        print(f"CO₂加倍 (有云反馈):   {results['with_cloud_feedback']['T_global']:.2f}°C")
        print(f"\n气候敏感性 (无云反馈): {T_response_no_cloud:.2f}°C / 2×CO₂")
        print(f"气候敏感性 (有云反馈): {T_response_with_cloud:.2f}°C / 2×CO₂")
        print(f"云反馈贡献:            {T_response_with_cloud - T_response_no_cloud:.2f}°C")
        
        lambda_no_cloud = forcing / T_response_no_cloud
        lambda_with_cloud = forcing / T_response_with_cloud
        print(f"\n反馈参数 (无云反馈):   {lambda_no_cloud:.2f} W/m²/K")
        print(f"反馈参数 (有云反馈):   {lambda_with_cloud:.2f} W/m²/K")
        print(f"云反馈参数:            {lambda_with_cloud - lambda_no_cloud:.2f} W/m²/K")
        
        print("="*60)
        
        return results
    
    def plot_equilibrium(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.lat, self.T[-1, :] - 273.15, 'b-', linewidth=2)
        plt.xlabel('Latitude (degrees)')
        plt.ylabel('Temperature (°C)')
        plt.title('Equilibrium Temperature Distribution')
        plt.grid(True)
        plt.xlim(-90, 90)
        plt.show()
    
    def plot_time_evolution(self):
        plt.figure(figsize=(12, 6))
        im = plt.pcolormesh(self.lat, self.time, self.T - 273.15, 
                           cmap='RdBu_r', shading='auto')
        plt.colorbar(im, label='Temperature (°C)')
        plt.xlabel('Latitude (degrees)')
        plt.ylabel('Time (years)')
        plt.title('Temperature Evolution')
        plt.show()
    
    def plot_cloud_feedback(self):
        if not self.enable_cloud_feedback:
            print("云反馈未启用！")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0,0].plot(self.lat, self.cloud_cover[-1, :] * 100, 'b-', linewidth=2)
        axes[0,0].set_xlabel('Latitude (degrees)')
        axes[0,0].set_ylabel('Cloud Cover (%)')
        axes[0,0].set_title('Equilibrium Cloud Cover')
        axes[0,0].grid(True)
        axes[0,0].set_xlim(-90, 90)
        
        axes[0,1].plot(self.lat, self.cloud_sw_forcing[-1, :], 'r-', linewidth=2, label='SW')
        axes[0,1].plot(self.lat, self.cloud_lw_forcing[-1, :], 'b-', linewidth=2, label='LW')
        axes[0,1].plot(self.lat, self.cloud_sw_forcing[-1, :] + self.cloud_lw_forcing[-1, :], 
                       'k--', linewidth=2, label='Net')
        axes[0,1].set_xlabel('Latitude (degrees)')
        axes[0,1].set_ylabel('Cloud Forcing (W/m²)')
        axes[0,1].set_title('Cloud Radiative Forcing')
        axes[0,1].legend()
        axes[0,1].grid(True)
        axes[0,1].set_xlim(-90, 90)
        
        weights = np.cos(self.phi)
        T_global = np.average(self.T, weights=weights, axis=1) - 273.15
        c_global = np.average(self.cloud_cover, weights=weights, axis=1) * 100
        
        axes[1,0].plot(self.time, T_global, 'r-', linewidth=2)
        axes[1,0].set_xlabel('Time (years)')
        axes[1,0].set_ylabel('Global Mean Temperature (°C)')
        axes[1,0].set_title('Temperature Evolution')
        axes[1,0].grid(True)
        
        axes[1,1].plot(self.time, c_global, 'b-', linewidth=2)
        axes[1,1].set_xlabel('Time (years)')
        axes[1,1].set_ylabel('Global Mean Cloud Cover (%)')
        axes[1,1].set_title('Cloud Cover Evolution')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_sensitivity_experiment(results):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        lat = results['control']['model'].lat
        weights = np.cos(results['control']['model'].phi)
        
        T_ctrl = results['control']['model'].T[-1, :] - 273.15
        T_no_cloud = results['no_cloud_feedback']['model'].T[-1, :] - 273.15
        T_with_cloud = results['with_cloud_feedback']['model'].T[-1, :] - 273.15
        
        axes[0,0].plot(lat, T_ctrl, 'k-', linewidth=2, label='Control')
        axes[0,0].plot(lat, T_no_cloud, 'b--', linewidth=2, label='2×CO₂ (no cloud)')
        axes[0,0].plot(lat, T_with_cloud, 'r-', linewidth=2, label='2×CO₂ (with cloud)')
        axes[0,0].set_xlabel('Latitude (degrees)')
        axes[0,0].set_ylabel('Temperature (°C)')
        axes[0,0].set_title('Temperature Distribution')
        axes[0,0].legend()
        axes[0,0].grid(True)
        axes[0,0].set_xlim(-90, 90)
        
        dT_no_cloud = T_no_cloud - T_ctrl
        dT_with_cloud = T_with_cloud - T_ctrl
        axes[0,1].plot(lat, dT_no_cloud, 'b--', linewidth=2, label='No cloud feedback')
        axes[0,1].plot(lat, dT_with_cloud, 'r-', linewidth=2, label='With cloud feedback')
        axes[0,1].set_xlabel('Latitude (degrees)')
        axes[0,1].set_ylabel('ΔTemperature (°C)')
        axes[0,1].set_title('Warming Response to CO₂ Doubling')
        axes[0,1].legend()
        axes[0,1].grid(True)
        axes[0,1].set_xlim(-90, 90)
        
        if results['with_cloud_feedback']['model'].enable_cloud_feedback:
            cloud_cover = results['with_cloud_feedback']['model'].cloud_cover[-1, :] * 100
            axes[1,0].plot(lat, cloud_cover, 'b-', linewidth=2)
            axes[1,0].set_xlabel('Latitude (degrees)')
            axes[1,0].set_ylabel('Cloud Cover (%)')
            axes[1,0].set_title('Cloud Cover (With Feedback)')
            axes[1,0].grid(True)
            axes[1,0].set_xlim(-90, 90)
        
        scenarios = ['Control', '2×CO₂\n(No Cloud)', '2×CO₂\n(With Cloud)']
        temps = [results['control']['T_global'],
                 results['no_cloud_feedback']['T_global'],
                 results['with_cloud_feedback']['T_global']]
        colors = ['gray', 'blue', 'red']
        
        axes[1,1].bar(scenarios, temps, color=colors, alpha=0.7)
        axes[1,1].set_ylabel('Global Mean Temperature (°C)')
        axes[1,1].set_title('Climate Sensitivity Comparison')
        axes[1,1].grid(True, axis='y')
        
        for i, v in enumerate(temps):
            axes[1,1].text(i, v + 0.1, f'{v:.1f}°C', ha='center')
        
        plt.tight_layout()
        plt.show()
    
    def animate_evolution(self, interval=50):
        fig, ax = plt.subplots(figsize=(10, 6))
        line, = ax.plot(self.lat, self.T[0, :] - 273.15, 'b-', linewidth=2)
        
        ax.set_xlabel('Latitude (degrees)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Temperature Evolution')
        ax.grid(True)
        ax.set_xlim(-90, 90)
        ax.set_ylim(-40, 40)
        
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        
        def animate(i):
            line.set_ydata(self.T[i, :] - 273.15)
            year = i * self.dt / (86400 * 365)
            time_text.set_text(f'Time: {year:.1f} years')
            return line, time_text
        
        anim = FuncAnimation(fig, animate, frames=range(0, self.n_steps, 10),
                            interval=interval, blit=True)
        plt.show()
        return anim


if __name__ == "__main__":
    print("=" * 60)
    print("一维能量平衡模型 - 云反馈演示")
    print("=" * 60)
    
    model = EnergyBalanceModel1D(n_lat=36, dt=86400.0 * 5, n_years=80)
    model.enable_cloud_feedback = True
    model.co2_ppm = 568.0
    model.forcing_co2 = model.compute_co2_forcing()
    
    print(f"\nCO₂ 浓度: {model.co2_ppm} ppm")
    print(f"CO₂ 辐射强迫: {model.forcing_co2:.2f} W/m²")
    print(f"云反馈启用: {model.enable_cloud_feedback}")
    
    model.run()
    model.plot_equilibrium()
    model.plot_cloud_feedback()
    model.plot_time_evolution()
    
    print("\n" + "=" * 60)
    print("运行 CO₂ 倍增敏感性实验...")
    print("=" * 60)
    
    results = model.run_co2_doubling_experiment()
    EnergyBalanceModel1D.plot_sensitivity_experiment(results)
