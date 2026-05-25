import numpy as np
import matplotlib.pyplot as plt

class CompositeLaminate:
    def __init__(self, ply_properties, layup, ply_thickness):
        """
        初始化复合材料层板
        
        参数:
            ply_properties: 单层板材料属性字典
                - E1, E2: 纵向和横向弹性模量
                - G12, G13, G23: 剪切模量
                - nu12: 泊松比
            layup: 铺层角度列表, 如 [0, 45, -45, 90]
            ply_thickness: 单层板厚度
        """
        self.ply_props = ply_properties
        self.layup = layup
        self.t_ply = ply_thickness
        self.n_plies = len(layup)
        self.h_total = self.n_plies * ply_thickness
        
        self._calculate_ABD_matrix()
        self._calculate_shear_stiffness()
        
    def _get_stiffness_matrix(self, theta):
        """计算任意角度下的平面应力刚度矩阵 Q"""
        E1 = self.ply_props['E1']
        E2 = self.ply_props['E2']
        G12 = self.ply_props['G12']
        nu12 = self.ply_props['nu12']
        nu21 = nu12 * E2 / E1
        
        Q11 = E1 / (1 - nu12 * nu21)
        Q22 = E2 / (1 - nu12 * nu21)
        Q12 = nu12 * E2 / (1 - nu12 * nu21)
        Q66 = G12
        
        theta_rad = np.radians(theta)
        c = np.cos(theta_rad)
        s = np.sin(theta_rad)
        
        Q_bar = np.zeros((3, 3))
        Q_bar[0, 0] = Q11 * c**4 + 2 * (Q12 + 2 * Q66) * s**2 * c**2 + Q22 * s**4
        Q_bar[0, 1] = (Q11 + Q22 - 4 * Q66) * s**2 * c**2 + Q12 * (s**4 + c**4)
        Q_bar[0, 2] = (Q11 - Q12 - 2 * Q66) * s * c**3 + (Q12 - Q22 + 2 * Q66) * s**3 * c
        Q_bar[1, 0] = Q_bar[0, 1]
        Q_bar[1, 1] = Q11 * s**4 + 2 * (Q12 + 2 * Q66) * s**2 * c**2 + Q22 * c**4
        Q_bar[1, 2] = (Q11 - Q12 - 2 * Q66) * s**3 * c + (Q12 - Q22 + 2 * Q66) * s * c**3
        Q_bar[2, 0] = Q_bar[0, 2]
        Q_bar[2, 1] = Q_bar[1, 2]
        Q_bar[2, 2] = (Q11 + Q22 - 2 * Q12 - 2 * Q66) * s**2 * c**2 + Q66 * (s**4 + c**4)
        
        return Q_bar
    
    def _calculate_ABD_matrix(self):
        """计算层板的ABD矩阵"""
        self.A = np.zeros((3, 3))
        self.B = np.zeros((3, 3))
        self.D = np.zeros((3, 3))
        
        z = np.zeros(self.n_plies + 1)
        for i in range(self.n_plies + 1):
            z[i] = -self.h_total / 2 + i * self.t_ply
        
        for k in range(self.n_plies):
            theta = self.layup[k]
            Q_bar = self._get_stiffness_matrix(theta)
            
            z_k = z[k]
            z_k1 = z[k + 1]
            
            self.A += Q_bar * (z_k1 - z_k)
            self.B += 0.5 * Q_bar * (z_k1**2 - z_k**2)
            self.D += (1 / 3) * Q_bar * (z_k1**3 - z_k**3)
        
        self.ABD = np.block([[self.A, self.B], [self.B, self.D]])
        
    def _calculate_shear_stiffness(self):
        """计算剪切刚度矩阵 (用于一阶剪切变形理论)"""
        G13 = self.ply_props['G13']
        G23 = self.ply_props['G23']
        
        z = np.zeros(self.n_plies + 1)
        for i in range(self.n_plies + 1):
            z[i] = -self.h_total / 2 + i * self.t_ply
        
        self.A44 = 0
        self.A55 = 0
        
        for k in range(self.n_plies):
            theta = np.radians(self.layup[k])
            c = np.cos(theta)
            s = np.sin(theta)
            
            C44 = G23 * c**2 + G13 * s**2
            C55 = G13 * c**2 + G23 * s**2
            
            z_k = z[k]
            z_k1 = z[k + 1]
            
            self.A44 += C44 * (z_k1 - z_k)
            self.A55 += C55 * (z_k1 - z_k)
        
        self.k_shear = 5/6
        
    def apply_bending_load(self, Mx=0, My=0, Mxy=0):
        """
        应用弯曲载荷 (经典层板理论)
        
        参数:
            Mx, My, Mxy: 单位长度弯矩 (N)
        """
        force_vector = np.zeros(6)
        force_vector[3:] = [Mx, My, Mxy]
        
        result = np.linalg.solve(self.ABD, force_vector)
        self.eps0 = result[:3]
        self.kappa = result[3:]
        
        self._calculate_stresses()
        
    def apply_bending_load_fsdm(self, Mx=0, My=0, Mxy=0, Qx=0, Qy=0):
        """
        应用弯曲载荷 (一阶剪切变形理论)
        
        参数:
            Mx, My, Mxy: 单位长度弯矩 (N)
            Qx, Qy: 单位长度剪力 (N/m)
        """
        self.Qx = Qx
        self.Qy = Qy
        
        self.apply_bending_load(Mx, My, Mxy)
        
    def _calculate_stresses(self):
        """计算各层的面内应力"""
        z = np.zeros(self.n_plies + 1)
        for i in range(self.n_plies + 1):
            z[i] = -self.h_total / 2 + i * self.t_ply
        
        self.layer_stresses = []
        self.z_positions = z
        
        for k in range(self.n_plies):
            theta = self.layup[k]
            Q_bar = self._get_stiffness_matrix(theta)
            
            z_bottom = z[k]
            z_top = z[k + 1]
            
            eps_bottom = self.eps0 + z_bottom * self.kappa
            eps_top = self.eps0 + z_top * self.kappa
            
            sigma_bottom = Q_bar @ eps_bottom
            sigma_top = Q_bar @ eps_top
            
            self.layer_stresses.append({
                'theta': theta,
                'z_bottom': z_bottom,
                'z_top': z_top,
                'sigma_bottom': sigma_bottom,
                'sigma_top': sigma_top,
                'Q_bar': Q_bar
            })
    
    def calculate_interlaminar_shear_stress(self):
        """
        计算层间剪应力 (基于面内应力积分)
        
        返回:
            z_coords: z坐标列表
            tau_xz: xz方向层间剪应力
            tau_yz: yz方向层间剪应力
        """
        z = self.z_positions
        n_points = 100
        z_fine = np.linspace(z[0], z[-1], n_points)
        
        tau_xz = np.zeros_like(z_fine)
        tau_yz = np.zeros_like(z_fine)
        
        for i, z_i in enumerate(z_fine):
            integral_x = 0
            integral_y = 0
            
            for k in range(self.n_plies):
                layer = self.layer_stresses[k]
                z_b = layer['z_bottom']
                z_t = layer['z_top']
                
                if z_i <= z_b:
                    continue
                
                z_upper = min(z_i, z_t)
                
                dz = z_upper - z_b
                if dz <= 0:
                    continue
                
                dsigma_x_dx = 0
                dsigma_y_dy = 0
                dsigma_x_dy = 0
                dsigma_y_dx = 0
                
                if abs(z_t - z_b) > 1e-10:
                    sigma_x_grad = (layer['sigma_top'][0] - layer['sigma_bottom'][0]) / (z_t - z_b)
                    sigma_y_grad = (layer['sigma_top'][1] - layer['sigma_bottom'][1]) / (z_t - z_b)
                    
                    avg_sigma_x = (layer['sigma_bottom'][0] + layer['sigma_top'][0]) / 2
                    avg_sigma_y = (layer['sigma_bottom'][1] + layer['sigma_top'][1]) / 2
                    
                    dsigma_x_dx = sigma_x_grad * self.kappa[0]
                    dsigma_y_dy = sigma_y_grad * self.kappa[1]
                    dsigma_x_dy = sigma_x_grad * self.kappa[2]
                    dsigma_y_dx = sigma_y_grad * self.kappa[2]
                
                integral_x += (dsigma_x_dx + dsigma_x_dy) * dz
                integral_y += (dsigma_y_dy + dsigma_y_dx) * dz
            
            tau_xz[i] = -integral_x
            tau_yz[i] = -integral_y
        
        tau_xz = self._normalize_shear_stress(z_fine, tau_xz)
        tau_yz = self._normalize_shear_stress(z_fine, tau_yz)
        
        return z_fine, tau_xz, tau_yz
    
    def _normalize_shear_stress(self, z, tau):
        """归一化剪应力分布 (抛物线分布近似)"""
        max_tau = np.max(np.abs(tau))
        if max_tau > 1e-10:
            theoretical = 1.5 * np.max(tau) * (1 - (2 * z / self.h_total)**2)
            scale = np.max(np.abs(theoretical)) / max_tau
            tau = tau * scale
        return tau
    
    def calculate_peeling_stress(self, a=1.0, b=1.0):
        """
        计算层间剥离应力 (基于简单梁理论的近似)
        
        参数:
            a, b: 层板尺寸 (m)
        
        返回:
            z_coords: z坐标列表
            sigma_z: 剥离应力
        """
        z = self.z_positions
        n_points = 100
        z_fine = np.linspace(z[0], z[-1], n_points)
        
        sigma_z = np.zeros_like(z_fine)
        
        sigma_x_top = self.layer_stresses[-1]['sigma_top'][0]
        sigma_x_bot = self.layer_stresses[0]['sigma_bottom'][0]
        
        for i, z_i in enumerate(z_fine):
            zi_norm = 2 * z_i / self.h_total
            sigma_z[i] = 0.5 * (sigma_x_top + sigma_x_bot) * (1 - zi_norm**2) * 0.1
        
        return z_fine, sigma_z
    
    def calculate_fsdm_shear_stress(self, Qx, Qy):
        """
        用一阶剪切变形理论计算层间剪应力
        
        参数:
            Qx, Qy: 单位长度剪力 (N/m)
        
        返回:
            z_coords: z坐标列表
            tau_xz: xz方向剪应力
            tau_yz: yz方向剪应力
        """
        n_points = 100
        z_fine = np.linspace(-self.h_total/2, self.h_total/2, n_points)
        
        tau_xz = np.zeros_like(z_fine)
        tau_yz = np.zeros_like(z_fine)
        
        gamma_xz_avg = Qx / (self.k_shear * self.A55) if self.A55 > 0 else 0
        gamma_yz_avg = Qy / (self.k_shear * self.A44) if self.A44 > 0 else 0
        
        for i, z_i in enumerate(z_fine):
            zi_norm = 2 * z_i / self.h_total
            
            G13 = self.ply_props['G13']
            G23 = self.ply_props['G23']
            
            tau_xz[i] = G13 * gamma_xz_avg * (1 - zi_norm**2) * 1.5
            tau_yz[i] = G23 * gamma_yz_avg * (1 - zi_norm**2) * 1.5
        
        return z_fine, tau_xz, tau_yz
    
    def get_stress_at_z(self, z_coord):
        """获取指定z坐标处的面内应力"""
        for layer in self.layer_stresses:
            if layer['z_bottom'] <= z_coord <= layer['z_top']:
                if abs(layer['z_top'] - layer['z_bottom']) > 1e-10:
                    ratio = (z_coord - layer['z_bottom']) / (layer['z_top'] - layer['z_bottom'])
                    sigma = layer['sigma_bottom'] + ratio * (layer['sigma_top'] - layer['sigma_bottom'])
                else:
                    sigma = layer['sigma_bottom']
                return sigma
        return None
    
    def print_summary(self):
        """打印层板计算摘要"""
        print("=" * 60)
        print("复合材料层板分析摘要")
        print("=" * 60)
        print(f"铺层角度: {self.layup}")
        print(f"总层数: {self.n_plies}")
        print(f"单层厚度: {self.t_ply * 1000:.2f} mm")
        print(f"总厚度: {self.h_total * 1000:.2f} mm")
        print("\nA矩阵 (面内刚度):")
        print(self.A)
        print("\nB矩阵 (耦合刚度):")
        print(self.B)
        print("\nD矩阵 (弯曲刚度):")
        print(self.D)
        print("\n中面应变:")
        print(f"  εx0 = {self.eps0[0]:.6e}")
        print(f"  εy0 = {self.eps0[1]:.6e}")
        print(f"  γxy0 = {self.eps0[2]:.6e}")
        print("\n曲率:")
        print(f"  κx = {self.kappa[0]:.6e}")
        print(f"  κy = {self.kappa[1]:.6e}")
        print(f"  κxy = {self.kappa[2]:.6e}")
        print("=" * 60)


def plot_stress_distribution(laminate, title_prefix=""):
    """绘制应力分布图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{title_prefix}层间应力分布', fontsize=14)
    
    z_clt, tau_xz_clt, tau_yz_clt = laminate.calculate_interlaminar_shear_stress()
    z_peel, sigma_z = laminate.calculate_peeling_stress()
    
    Qx = 1000
    Qy = 500
    z_fsdm, tau_xz_fsdm, tau_yz_fsdm = laminate.calculate_fsdm_shear_stress(Qx, Qy)
    
    ax = axes[0, 0]
    ax.plot(tau_xz_clt, z_clt, 'b-', linewidth=2, label='CLT')
    ax.plot(tau_xz_fsdm, z_fsdm, 'r--', linewidth=2, label='FSDT')
    ax.set_xlabel('τ_xz (Pa)')
    ax.set_ylabel('z (m)')
    ax.set_title('XZ方向层间剪应力')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(tau_yz_clt, z_clt, 'b-', linewidth=2, label='CLT')
    ax.plot(tau_yz_fsdm, z_fsdm, 'r--', linewidth=2, label='FSDT')
    ax.set_xlabel('τ_yz (Pa)')
    ax.set_ylabel('z (m)')
    ax.set_title('YZ方向层间剪应力')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(sigma_z, z_peel, 'g-', linewidth=2)
    ax.set_xlabel('σ_z (Pa)')
    ax.set_ylabel('z (m)')
    ax.set_title('层间剥离应力')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    layer_stresses_x = []
    layer_stresses_y = []
    z_pos = []
    
    for layer in laminate.layer_stresses:
        z_pos.append(layer['z_bottom'])
        layer_stresses_x.append(layer['sigma_bottom'][0])
        layer_stresses_y.append(layer['sigma_bottom'][1])
        z_pos.append(layer['z_top'])
        layer_stresses_x.append(layer['sigma_top'][0])
        layer_stresses_y.append(layer['sigma_top'][1])
    
    ax.plot(layer_stresses_x, z_pos, 'b-', linewidth=2, label='σ_x')
    ax.plot(layer_stresses_y, z_pos, 'r-', linewidth=2, label='σ_y')
    ax.set_xlabel('面内应力 (Pa)')
    ax.set_ylabel('z (m)')
    ax.set_title('面内应力分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    ply_properties = {
        'E1': 138e9,
        'E2': 8.96e9,
        'G12': 7.1e9,
        'G13': 7.1e9,
        'G23': 3.9e9,
        'nu12': 0.3
    }
    
    layup = [0, 45, -45, 90, 90, -45, 45, 0]
    ply_thickness = 0.125e-3
    
    laminate = CompositeLaminate(ply_properties, layup, ply_thickness)
    
    Mx = 100.0
    My = 50.0
    Mxy = 0.0
    
    laminate.apply_bending_load(Mx=Mx, My=My, Mxy=Mxy)
    laminate.print_summary()
    
    print("\n各层面内应力:")
    for k, layer in enumerate(laminate.layer_stresses):
        print(f"\n第{k+1}层 (θ={layer['theta']}°):")
        print(f"  底面: σx={layer['sigma_bottom'][0]:.3e} Pa, σy={layer['sigma_bottom'][1]:.3e} Pa, τxy={layer['sigma_bottom'][2]:.3e} Pa")
        print(f"  顶面: σx={layer['sigma_top'][0]:.3e} Pa, σy={layer['sigma_top'][1]:.3e} Pa, τxy={layer['sigma_top'][2]:.3e} Pa")
    
    z_clt, tau_xz_clt, tau_yz_clt = laminate.calculate_interlaminar_shear_stress()
    print(f"\n最大层间剪应力 τ_xz (CLT): {np.max(np.abs(tau_xz_clt)):.3e} Pa")
    print(f"最大层间剪应力 τ_yz (CLT): {np.max(np.abs(tau_yz_clt)):.3e} Pa")
    
    z_peel, sigma_z = laminate.calculate_peeling_stress()
    print(f"最大剥离应力 σ_z: {np.max(np.abs(sigma_z)):.3e} Pa")
    
    Qx = 1000
    Qy = 500
    z_fsdm, tau_xz_fsdm, tau_yz_fsdm = laminate.calculate_fsdm_shear_stress(Qx, Qy)
    print(f"\n最大层间剪应力 τ_xz (FSDT): {np.max(np.abs(tau_xz_fsdm)):.3e} Pa")
    print(f"最大层间剪应力 τ_yz (FSDT): {np.max(np.abs(tau_yz_fsdm)):.3e} Pa")
    
    fig = plot_stress_distribution(laminate, title_prefix="[0/45/-45/90]s")
    fig.savefig('composite_stresses.png', dpi=300, bbox_inches='tight')
    print("\n应力分布图已保存为 'composite_stresses.png'")
    plt.show()


if __name__ == "__main__":
    main()
