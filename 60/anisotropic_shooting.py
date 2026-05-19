import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar


class VTIModel:
    """
    垂直横向各向同性(VTI)介质模型 - 用于描述垂直接触的裂缝介质
    
    Thomsen参数:
    - vp0: 垂直方向P波速度
    - vs0: 垂直方向S波速度
    - epsilon: P波各向异性参数
    - delta: 近轴各向异性参数
    - gamma: S波各向异性参数
    - rho: 密度
    """
    def __init__(self, vp0=3000, vs0=1500, epsilon=0.2, delta=0.1, gamma=0.15, rho=2500):
        self.vp0 = vp0
        self.vs0 = vs0
        self.epsilon = epsilon
        self.delta = delta
        self.gamma = gamma
        self.rho = rho
        self._compute_stiffness()
        
    def _compute_stiffness(self):
        """计算弹性刚度张量 (Voigt notation, 6x6)"""
        vp0, vs0 = self.vp0, self.vs0
        epsilon, delta, gamma = self.epsilon, self.delta, self.gamma
        rho = self.rho
        
        c33 = rho * vp0**2
        c44 = rho * vs0**2
        c11 = c33 * (1 + 2 * epsilon)
        c66 = c44 * (1 + 2 * gamma)
        
        c13 = np.sqrt((c33 - c44) * (c33 * (1 + 2 * delta) - c44)) - c44
        
        self.c = np.zeros((6, 6))
        self.c[0, 0] = c11
        self.c[0, 1] = c11 - 2 * c66
        self.c[0, 2] = c13
        self.c[1, 1] = c11
        self.c[1, 2] = c13
        self.c[2, 2] = c33
        self.c[3, 3] = c44
        self.c[4, 4] = c44
        self.c[5, 5] = c66
        
        self.c11 = c11
        self.c33 = c33
        self.c13 = c13
        self.c44 = c44
        self.c66 = c66
        
    def christoffel_phase_velocity(self, theta, wave_type='qP'):
        """
        使用Christoffel方程计算相速度
        
        参数:
            theta: 相角 (相对于对称轴的角度, 弧度)
            wave_type: 'qP' (准纵波), 'qSV' (准横波SV), 'qSH' (准横波SH)
        
        返回:
            v_phase: 相速度
        """
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin2 = sin_theta**2
        cos2 = cos_theta**2
        rho = self.rho
        
        if wave_type == 'qP':
            A = (self.c11 * sin2 + self.c33 * cos2) / rho
            B = (self.c13 + self.c44)**2 * sin2 * cos2 / (rho**2)
            C = (self.c11 - self.c44) * (self.c33 - self.c44) * sin2 * cos2 / (rho**2)
            v_sq = 0.5 * (A + np.sqrt(A**2 - 4 * (C - B)))
            return np.sqrt(v_sq)
            
        elif wave_type == 'qSV':
            A = (self.c11 * sin2 + self.c33 * cos2) / rho
            B = (self.c13 + self.c44)**2 * sin2 * cos2 / (rho**2)
            C = (self.c11 - self.c44) * (self.c33 - self.c44) * sin2 * cos2 / (rho**2)
            v_sq = 0.5 * (A - np.sqrt(A**2 - 4 * (C - B)))
            return np.sqrt(np.maximum(v_sq, 0))
            
        elif wave_type == 'qSH':
            v_sq = (self.c66 * sin2 + self.c44 * cos2) / rho
            return np.sqrt(v_sq)
            
        else:
            raise ValueError("wave_type must be 'qP', 'qSV', or 'qSH'")
    
    def group_velocity(self, theta, wave_type='qP'):
        """
        计算群速度和群角
        
        参数:
            theta: 相角 (弧度)
            wave_type: 'qP', 'qSV', 'qSH'
        
        返回:
            v_group: 群速度大小
            phi: 群角 (相对于对称轴, 弧度)
        """
        dtheta = 1e-6
        v1 = self.christoffel_phase_velocity(theta, wave_type)
        v2 = self.christoffel_phase_velocity(theta + dtheta, wave_type)
        dv_dtheta = (v2 - v1) / dtheta
        
        p = v1 * np.sin(theta)
        q = v1 * np.cos(theta)
        
        dp_dtheta = dv_dtheta * np.sin(theta) + v1 * np.cos(theta)
        dq_dtheta = dv_dtheta * np.cos(theta) - v1 * np.sin(theta)
        
        vgx = p * dp_dtheta + q * dq_dtheta
        vgz = p * dq_dtheta - q * dp_dtheta
        
        v_group = np.sqrt(vgx**2 + vgz**2) / v1
        phi = np.arctan2(vgx, vgz)
        
        return v_group, phi
    
    def phase_angle_from_group_angle(self, phi, wave_type='qP', tol=1e-8, max_iter=100):
        """从群角反求相角 (用于射线追踪)"""
        theta = phi
        for _ in range(max_iter):
            vg, phi_calc = self.group_velocity(theta, wave_type)
            dphi_dtheta = (self.group_velocity(theta + 1e-6, wave_type)[1] - phi_calc) / 1e-6
            dphi = phi - phi_calc
            theta += dphi / (dphi_dtheta + 1e-10)
            if abs(dphi) < tol:
                break
        return theta
    
    def velocity_at_angle(self, angle, wave_type='qP', angle_type='group'):
        """获取指定角度的速度"""
        if angle_type == 'group':
            theta = self.phase_angle_from_group_angle(angle, wave_type)
            vg, _ = self.group_velocity(theta, wave_type)
            return vg
        else:
            return self.christoffel_phase_velocity(angle, wave_type)
    
    def get_slowness_vector(self, phi, wave_type='qP'):
        """从群角获取慢度矢量"""
        theta = self.phase_angle_from_group_angle(phi, wave_type)
        vp = self.christoffel_phase_velocity(theta, wave_type)
        px = np.sin(theta) / vp
        pz = np.cos(theta) / vp
        return px, pz, theta
    
    def group_velocity_components(self, px, pz, wave_type='qP'):
        """从慢度矢量计算群速度分量"""
        theta = np.arctan2(px, pz)
        vg, phi = self.group_velocity(theta, wave_type)
        vgx = vg * np.sin(phi)
        vgz = vg * np.cos(phi)
        return vgx, vgz


class AnisotropicShooting:
    """各向异性介质打靶法射线追踪"""
    
    def __init__(self, anisotropic_model, wave_type='qP', debug=False):
        self.model = anisotropic_model
        self.wave_type = wave_type
        self.debug = debug
        
    def _ray_ode(self, s, state):
        """
        各向异性介质射线常微分方程
        
        state = [x, z, px, pz]
        其中 (px, pz) 是慢度矢量
        """
        x, z, px, pz = state
        
        theta = np.arctan2(px, pz)
        vg, phi = self.model.group_velocity(theta, self.wave_type)
        
        dx_ds = vg * np.sin(phi)
        dz_ds = vg * np.cos(phi)
        
        dpx_ds = 0.0
        dpz_ds = 0.0
        
        return [dx_ds, dz_ds, dpx_ds, dpz_ds]
    
    def shoot_ray(self, x0, z0, takeoff_angle, max_s=20000):
        """
        发射一条射线
        
        参数:
            x0, z0: 震源位置
            takeoff_angle: 出射群角 (度, 相对于垂直轴)
            max_s: 最大射线路径长度
        """
        phi0 = np.radians(takeoff_angle)
        px, pz, theta = self.model.get_slowness_vector(phi0, self.wave_type)
        
        if self.debug:
            vp = self.model.christoffel_phase_velocity(theta, self.wave_type)
            vg, _ = self.model.group_velocity(theta, self.wave_type)
            print(f"出射角: {takeoff_angle:.2f}°")
            print(f"  相角: {np.degrees(theta):.2f}°")
            print(f"  相速度: {vp:.2f} m/s")
            print(f"  群速度: {vg:.2f} m/s")
        
        initial_state = [x0, z0, px, pz]
        s_span = [0, max_s]
        
        try:
            sol = solve_ivp(
                self._ray_ode,
                s_span,
                initial_state,
                method='RK45',
                max_step=10.0
            )
            
            x = sol.y[0]
            z = sol.y[1]
            travel_time = self._compute_travel_time(x, z)
            
            return x, z, travel_time
            
        except Exception as e:
            if self.debug:
                print(f"  射线追踪失败: {e}")
            return np.array([x0]), np.array([z0]), 0.0
    
    def _compute_travel_time(self, x, z):
        """计算走时"""
        if len(x) < 2:
            return 0.0
        
        v = []
        for i in range(len(x)):
            dx = x[i] - (x[i-1] if i > 0 else x[0])
            dz = z[i] - (z[i-1] if i > 0 else z[0])
            phi = np.arctan2(abs(dx), abs(dz)) if (abs(dx) + abs(dz) > 1e-10) else 0
            v.append(self.model.velocity_at_angle(phi, self.wave_type, 'group'))
        
        v = np.array(v)
        ds = np.sqrt(np.diff(x)**2 + np.diff(z)**2)
        travel_time = np.sum(ds / v[:-1])
        return travel_time
    
    def _objective_function(self, angle, x0, z0, xr, zr):
        """目标函数：射线到达点与接收点的距离"""
        x, z, _ = self.shoot_ray(x0, z0, angle)
        if len(x) < 2:
            return 1e10
        
        idx = np.argmin(np.abs(z - zr))
        z_error = abs(z[idx] - zr)
        
        if z_error > 200:
            return 1e10 + z_error * 100
        
        return abs(x[idx] - xr)
    
    def find_ray(self, x0, z0, xr, zr, angle_range=(0.1, 89)):
        """寻找到达目标接收点的射线"""
        angles_to_test = np.linspace(angle_range[0], angle_range[1], 60)
        best_angle = None
        best_dist = float('inf')
        best_x, best_z, best_tt = None, None, None
        
        if self.debug:
            print(f"\n搜索射线: 目标 ({xr}, {zr})")
        
        for angle in angles_to_test:
            try:
                x, z, tt = self.shoot_ray(x0, z0, angle)
                if len(x) < 2:
                    continue
                
                idx = np.argmin(np.sqrt((x - xr)**2 + (z - zr)**2))
                dist = np.sqrt((x[idx] - xr)**2 + (z[idx] - zr)**2)
                
                if dist < best_dist:
                    best_dist = dist
                    best_angle = angle
                    best_x, best_z, best_tt = x, z, tt
            except:
                continue
        
        if best_angle is None:
            return None, None, None, None
        
        try:
            def objective(angle):
                return self._objective_function(angle, x0, z0, xr, zr)
            
            search_range = 8.0
            res = minimize_scalar(
                objective,
                bracket=(max(angle_range[0], best_angle - search_range),
                         best_angle,
                         min(angle_range[1], best_angle + search_range)),
                method='brent'
            )
            
            optimal_angle = res.x
            x, z, tt = self.shoot_ray(x0, z0, optimal_angle)
            final_dist = self._objective_function(optimal_angle, x0, z0, xr, zr)
            
            if final_dist < best_dist:
                if self.debug:
                    print(f"  优化成功: 初始角度 {best_angle:.2f}° -> 最优角度 {optimal_angle:.2f}°")
                    print(f"  距离误差: {best_dist:.2f}m -> {final_dist:.2f}m")
                return x, z, tt, optimal_angle
            else:
                return best_x, best_z, best_tt, best_angle
                
        except Exception as e:
            if self.debug:
                print(f"  优化失败: {e}, 使用初始角度 {best_angle:.2f}°")
            return best_x, best_z, best_tt, best_angle
    
    def compute_rays(self, source, receivers, angle_range=(0.1, 89)):
        """计算到所有接收点的射线"""
        rays = []
        for receiver in receivers:
            x, z, tt, angle = self.find_ray(
                source[0], source[1],
                receiver[0], receiver[1],
                angle_range
            )
            if x is not None and len(x) > 1:
                rays.append({
                    'receiver': receiver,
                    'x': x,
                    'z': z,
                    'travel_time': tt,
                    'takeoff_angle': angle
                })
        return rays


def plot_anisotropic_comparison(vti_model, wave_types=['qP', 'qSV', 'qSH']):
    """绘制各向同性与各向异性的速度对比"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    thetas = np.linspace(0, np.pi/2, 100)
    angles_deg = np.degrees(thetas)
    
    for i, wave_type in enumerate(wave_types):
        v_phase = [vti_model.christoffel_phase_velocity(theta, wave_type) for theta in thetas]
        
        v_group = []
        phis = []
        for theta in thetas:
            vg, phi = vti_model.group_velocity(theta, wave_type)
            v_group.append(vg)
            phis.append(np.degrees(phi))
        
        axes[i].plot(angles_deg, v_phase, 'b-', linewidth=2, label='相速度')
        axes[i].plot(phis, v_group, 'r--', linewidth=2, label='群速度')
        axes[i].set_xlabel('角度 (度)')
        axes[i].set_ylabel('速度 (m/s)')
        axes[i].set_title(f'{wave_type} 波 - 速度随角度变化')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('anisotropic_velocity_comparison.png', dpi=150, bbox_inches='tight')
    print("各向异性速度对比图已保存为 anisotropic_velocity_comparison.png")
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    for wave_type in wave_types:
        thetas = np.linspace(0, 2*np.pi, 200)
        vx, vz = [], []
        for theta in thetas:
            vg, phi = vti_model.group_velocity(theta, wave_type)
            vx.append(vg * np.sin(phi))
            vz.append(vg * np.cos(phi))
        ax.plot(vx, vz, '-', linewidth=2, label=f'{wave_type} 群速度面')
    
    ax.set_xlabel('X (m/s)')
    ax.set_ylabel('Z (m/s)')
    ax.set_title('群速度面 (波前)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.savefig('group_velocity_surface.png', dpi=150, bbox_inches='tight')
    print("群速度面图已保存为 group_velocity_surface.png")
    plt.close()


def plot_anisotropic_rays(rays, source, model, wave_type, title=None):
    """绘制各向异性介质中的射线路径"""
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(rays))) if rays else []
    
    for i, ray in enumerate(rays):
        plt.plot(ray['x'], ray['z'], color=colors[i], linewidth=2,
                 label=f"Receiver {i+1}: {ray['travel_time']:.3f}s, {ray['takeoff_angle']:.1f}°")
    
    plt.scatter(source[0], source[1], c='red', s=300, marker='*', 
                edgecolors='black', label='Source', zorder=10)
    
    for i, ray in enumerate(rays):
        plt.scatter(ray['receiver'][0], ray['receiver'][1], c=colors[i], 
                    s=150, marker='v', edgecolors='black', zorder=10)
    
    plt.xlabel('X (m)')
    plt.ylabel('Depth Z (m)')
    plt_title = f'{wave_type} 波射线追踪 (VTI裂缝介质)'
    if title:
        plt_title += ' - ' + title
    plt.title(plt_title)
    plt.legend(fontsize=8, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()
    
    info_text = f"VTI参数:\nVp0={model.vp0}m/s, Vs0={model.vs0}m/s\n"
    info_text += f"ε={model.epsilon}, δ={model.delta}, γ={model.gamma}"
    plt.text(0.02, 0.02, info_text, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8), fontsize=9)
    
    plt.tight_layout()
    filename = f"anisotropic_rays_{wave_type}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"各向异性射线路径图已保存为 {filename}")
    plt.close()


def main():
    print("=" * 70)
    print("各向异性介质射线追踪 - VTI裂缝介质")
    print("使用Christoffel方程求解群速度方向")
    print("=" * 70)
    
    print("\n创建VTI介质模型...")
    print("-" * 70)
    vti = VTIModel(
        vp0=3000,      # 垂直P波速度
        vs0=1500,      # 垂直S波速度
        epsilon=0.25,  # Thomsen ε
        delta=0.15,    # Thomsen δ
        gamma=0.2,     # Thomsen γ
        rho=2500       # 密度
    )
    
    print(f"VTI模型参数:")
    print(f"  垂直P波速度 vp0 = {vti.vp0} m/s")
    print(f"  垂直S波速度 vs0 = {vti.vs0} m/s")
    print(f"  Thomsen ε = {vti.epsilon} (P波各向异性)")
    print(f"  Thomsen δ = {vti.delta} (近轴各向异性)")
    print(f"  Thomsen γ = {vti.gamma} (S波各向异性)")
    
    print("\n绘制速度对比图...")
    plot_anisotropic_comparison(vti)
    
    print("\n" + "=" * 70)
    print("演示不同角度的群速度与相速度")
    print("-" * 70)
    
    test_angles = [0, 15, 30, 45, 60, 75, 90]
    for wave_type in ['qP', 'qSV', 'qSH']:
        print(f"\n{wave_type} 波:")
        print(f"{'角度(°)':>10} {'相速度(m/s)':>15} {'群速度(m/s)':>15} {'群角(°)':>10}")
        print("-" * 55)
        for angle in test_angles:
            theta = np.radians(angle)
            vp = vti.christoffel_phase_velocity(theta, wave_type)
            vg, phi = vti.group_velocity(theta, wave_type)
            print(f"{angle:10.1f} {vp:15.2f} {vg:15.2f} {np.degrees(phi):10.2f}")
    
    print("\n" + "=" * 70)
    print("各向异性射线追踪演示 - qP波")
    print("-" * 70)
    
    source = (0, 0)
    receivers = [(2000, 1000), (4000, 2000), (6000, 3000)]
    
    print(f"\n震源位置: {source}")
    print(f"接收点位置: {receivers}")
    
    for wave_type in ['qP', 'qSV']:
        print(f"\n{'='*70}")
        print(f"追踪 {wave_type} 波...")
        print("-" * 70)
        
        shooter = AnisotropicShooting(vti, wave_type=wave_type, debug=True)
        rays = shooter.compute_rays(source, receivers)
        
        print(f"\n成功计算 {len(rays)} 条射线:")
        for i, ray in enumerate(rays):
            print(f"\n接收点 {i+1}: (x={ray['receiver'][0]}m, z={ray['receiver'][1]}m)")
            print(f"  最优出射群角: {ray['takeoff_angle']:.2f}°")
            print(f"  走时: {ray['travel_time']:.4f} s")
        
        plot_anisotropic_rays(rays, source, vti, wave_type)
    
    print("\n" + "=" * 70)
    print("各向同性 vs 各向异性走时对比")
    print("-" * 70)
    
    iso_vp = vti.vp0
    for receiver in receivers:
        dist = np.sqrt(receiver[0]**2 + receiver[1]**2)
        iso_tt = dist / iso_vp
        
        shooter_qP = AnisotropicShooting(vti, wave_type='qP')
        x, z, ani_tt, angle = shooter_qP.find_ray(0, 0, receiver[0], receiver[1])
        
        print(f"\n接收点: {receiver}")
        print(f"  各向同性走时: {iso_tt:.4f} s (vp={iso_vp}m/s)")
        print(f"  各向异性走时: {ani_tt:.4f} s")
        print(f"  走时差: {(ani_tt - iso_tt)*1000:.2f} ms")
        print(f"  相对差异: {((ani_tt - iso_tt)/iso_tt*100):.2f}%")
    
    print("\n" + "=" * 70)
    print("计算完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
