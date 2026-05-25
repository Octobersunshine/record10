import numpy as np
from scipy.integrate import quad
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
rcParams['axes.unicode_minus'] = False


class CompositeLaminate3D:
    def __init__(self, ply_properties, layup, ply_thickness, plate_width=None):
        self.ply_props = ply_properties
        self.layup = layup
        self.t_ply = ply_thickness
        self.n_plies = len(layup)
        self.h_total = self.n_plies * ply_thickness
        
        if plate_width is None:
            self.b = 20 * self.h_total
        else:
            self.b = plate_width
        
        self._calculate_3d_stiffness()
        self._calculate_ABD_matrix()
        self._calculate_shear_stiffness()
        
    def _get_stiffness_matrix(self, theta):
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
    
    def _calculate_3d_stiffness(self):
        E1 = self.ply_props['E1']
        E2 = self.ply_props['E2']
        E3 = self.ply_props.get('E3', E2)
        nu12 = self.ply_props['nu12']
        nu13 = self.ply_props.get('nu13', nu12)
        nu23 = self.ply_props.get('nu23', 0.3)
        G12 = self.ply_props['G12']
        G13 = self.ply_props['G13']
        G23 = self.ply_props['G23']
        
        delta = (1 - nu12**2 * E2/E1 - nu23**2 * E3/E2 - nu13**2 * E3/E1 
                - 2*nu12*nu23*nu13*E3/E1) / (E1*E2*E3)
        
        self.C = np.zeros((6, 6))
        self.C[0, 0] = (1 - nu23**2) / (E2*E3*delta)
        self.C[1, 1] = (1 - nu13**2 * E3/E1) / (E1*E3*delta)
        self.C[2, 2] = (1 - nu12**2 * E2/E1) / (E1*E2*delta)
        self.C[0, 1] = (nu12 + nu13*nu23 * E3/E1) / (E1*E3*delta)
        self.C[0, 2] = (nu13 + nu12*nu23) / (E1*E2*delta)
        self.C[1, 2] = (nu23 + nu12*nu13 * E2/E1) / (E1*E2*delta)
        self.C[1, 0] = self.C[0, 1]
        self.C[2, 0] = self.C[0, 2]
        self.C[2, 1] = self.C[1, 2]
        self.C[3, 3] = G23
        self.C[4, 4] = G13
        self.C[5, 5] = G12
        
    def _calculate_ABD_matrix(self):
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
        G13 = self.ply_props['G13']
        G23 = self.ply_props['G23']
        
        self.A44 = 0
        self.A55 = 0
        
        for k in range(self.n_plies):
            theta = np.radians(self.layup[k])
            c = np.cos(theta)
            s = np.sin(theta)
            C44 = G23 * c**2 + G13 * s**2
            C55 = G13 * c**2 + G23 * s**2
            self.A44 += C44 * self.t_ply
            self.A55 += C55 * self.t_ply
        
        self.k_shear = 5/6
        
    def _calculate_fsdm_shear(self, Qx, Qy):
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
        
    def apply_bending_load(self, Mx=0, My=0, Mxy=0):
        force_vector = np.zeros(6)
        force_vector[3:] = [Mx, My, Mxy]
        
        result = np.linalg.solve(self.ABD, force_vector)
        self.eps0 = result[:3]
        self.kappa = result[3:]
        self._calculate_stresses()
        
    def _calculate_stresses(self):
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
    
    def calculate_interlaminar_shear_exact(self, y_points=None):
        if y_points is None:
            y_points = np.linspace(-self.b/2, self.b/2, 100)
        
        n_points_z = 50
        z_fine = np.linspace(-self.h_total/2, self.h_total/2, n_points_z)
        
        tau_xz = np.zeros((len(y_points), n_points_z))
        tau_yz = np.zeros((len(y_points), n_points_z))
        sigma_z = np.zeros((len(y_points), n_points_z))
        
        for i, y in enumerate(y_points):
            tau_xz[i, :], tau_yz[i, :], sigma_z[i, :] = self._pagano_solution(y, z_fine)
        
        return y_points, z_fine, tau_xz, tau_yz, sigma_z
    
    def _pagano_solution(self, y, z_values):
        Mx = self._get_bending_moment()
        My = self._get_bending_moment_y()
        
        b = self.b
        h = self.h_total
        
        sigma_x_avg = 0
        for layer in self.layer_stresses:
            sigma_x_avg += (layer['sigma_bottom'][0] + layer['sigma_top'][0]) / 2
        sigma_x_avg /= self.n_plies
        
        beta = np.pi / b
        
        tau_xz = np.zeros_like(z_values)
        tau_yz = np.zeros_like(z_values)
        sigma_z_arr = np.zeros_like(z_values)
        
        for j, z in enumerate(z_values):
            zi_norm = 2 * z / h
            
            tau_xz[j] = self._calculate_tau_xz_pagano(z, y, beta, sigma_x_avg)
            tau_yz[j] = self._calculate_tau_yz_pagano(z, y, beta)
            sigma_z_arr[j] = self._calculate_sigma_z_pagano(z, y, beta, sigma_x_avg)
        
        return tau_xz, tau_yz, sigma_z_arr
    
    def _calculate_tau_xz_pagano(self, z, y, beta, sigma_x_avg):
        h = self.h_total
        b = self.b
        
        cos_y = np.cos(beta * y)
        
        I = self.h_total**3 / 12
        Q = sigma_x_avg * h / 2
        
        tau_basic = Q / (2 * I) * (h**2 / 4 - z**2)
        
        boundary_layer = self._boundary_layer_shear(z, y, b)
        
        return tau_basic * cos_y + boundary_layer[0]
    
    def _calculate_tau_yz_pagano(self, z, y, beta):
        h = self.h_total
        b = self.b
        
        sin_y = np.sin(beta * y)
        
        sigma_y_avg = 0
        for layer in self.layer_stresses:
            sigma_y_avg += (layer['sigma_bottom'][1] + layer['sigma_top'][1]) / 2
        sigma_y_avg /= self.n_plies
        
        I = self.h_total**3 / 12
        Q = sigma_y_avg * h / 2
        
        tau_basic = Q / (2 * I) * (h**2 / 4 - z**2)
        
        boundary_layer = self._boundary_layer_shear(z, y, b)
        
        return tau_basic * sin_y + boundary_layer[1]
    
    def _calculate_sigma_z_pagano(self, z, y, beta, sigma_x_avg):
        h = self.h_total
        b = self.b
        
        cos_y = np.cos(beta * y)
        
        sigma_z_basic = sigma_x_avg * 0.05 * (1 - (2*z/h)**2)
        
        boundary_layer = self._boundary_layer_normal(z, y, b)
        
        return sigma_z_basic * cos_y + boundary_layer
    
    def _boundary_layer_shear(self, z, y, b):
        h = self.h_total
        lambda_bl = 3.0 / h
        
        y_dist = np.min([abs(y - b/2), abs(y + b/2), abs(y)])
        
        decay = np.exp(-lambda_bl * y_dist)
        
        sigma_x_top = self.layer_stresses[-1]['sigma_top'][0]
        sigma_x_bot = self.layer_stresses[0]['sigma_bottom'][0]
        
        delta_sigma = (sigma_x_top - sigma_x_bot) / 2
        
        tau_xz_bl = 0.3 * delta_sigma * decay * (1 - (2*z/h)**2)
        
        sigma_y_top = self.layer_stresses[-1]['sigma_top'][1]
        sigma_y_bot = self.layer_stresses[0]['sigma_bottom'][1]
        delta_sigma_y = (sigma_y_top - sigma_y_bot) / 2
        
        tau_yz_bl = 0.3 * delta_sigma_y * decay * (1 - (2*z/h)**2)
        
        return tau_xz_bl, tau_yz_bl
    
    def _boundary_layer_normal(self, z, y, b):
        h = self.h_total
        lambda_bl = 3.0 / h
        
        y_dist = np.min([abs(y - b/2), abs(y + b/2), abs(y)])
        
        decay = np.exp(-lambda_bl * y_dist)
        
        sigma_x_top = self.layer_stresses[-1]['sigma_top'][0]
        sigma_x_bot = self.layer_stresses[0]['sigma_bottom'][0]
        
        delta_sigma = (sigma_x_top - sigma_x_bot) / 2
        
        sigma_z_bl = 0.15 * delta_sigma * decay * (1 - (2*z/h)**2)
        
        return sigma_z_bl
    
    def calculate_interlaminar_stress_integral(self, y=None):
        n_points_z = 100
        z_fine = np.linspace(-self.h_total/2, self.h_total/2, n_points_z)
        
        tau_xz = np.zeros_like(z_fine)
        tau_yz = np.zeros_like(z_fine)
        sigma_z = np.zeros_like(z_fine)
        
        for i, z_i in enumerate(z_fine):
            result_x, result_y, result_z = self._integrate_stresses(z_i)
            tau_xz[i] = result_x
            tau_yz[i] = result_y
            sigma_z[i] = result_z
        
        return z_fine, tau_xz, tau_yz, sigma_z
    
    def _integrate_stresses(self, z_target):
        def integrand_x(z):
            stress_info = self._get_stress_gradient(z)
            return stress_info[0]
        
        def integrand_y(z):
            stress_info = self._get_stress_gradient(z)
            return stress_info[1]
        
        def integrand_z(z):
            stress_info = self._get_stress_gradient(z)
            return stress_info[2]
        
        z_bottom = -self.h_total / 2
        
        tau_xz, _ = quad(integrand_x, z_bottom, z_target, limit=100)
        tau_yz, _ = quad(integrand_y, z_bottom, z_target, limit=100)
        sigma_z, _ = quad(integrand_z, z_bottom, z_target, limit=100)
        
        return -tau_xz, -tau_yz, -sigma_z
    
    def _get_stress_gradient(self, z):
        for layer in self.layer_stresses:
            if layer['z_bottom'] <= z <= layer['z_top']:
                z_b = layer['z_bottom']
                z_t = layer['z_top']
                
                if abs(z_t - z_b) > 1e-15:
                    ratio = (z - z_b) / (z_t - z_b)
                    sigma_x = layer['sigma_bottom'][0] + ratio * (layer['sigma_top'][0] - layer['sigma_bottom'][0])
                    sigma_y = layer['sigma_bottom'][1] + ratio * (layer['sigma_top'][1] - layer['sigma_bottom'][1])
                    tau_xy = layer['sigma_bottom'][2] + ratio * (layer['sigma_top'][2] - layer['sigma_bottom'][2])
                    
                    dsigma_x_dx = (layer['sigma_top'][0] - layer['sigma_bottom'][0]) / (z_t - z_b) * self.kappa[0]
                    dsigma_y_dy = (layer['sigma_top'][1] - layer['sigma_bottom'][1]) / (z_t - z_b) * self.kappa[1]
                    dtau_xy_dx = (layer['sigma_top'][2] - layer['sigma_bottom'][2]) / (z_t - z_b) * self.kappa[0]
                    dtau_xy_dy = (layer['sigma_top'][2] - layer['sigma_bottom'][2]) / (z_t - z_b) * self.kappa[2]
                    
                    grad_x = dsigma_x_dx + dtau_xy_dy
                    grad_y = dsigma_y_dy + dtau_xy_dx
                    grad_z = self._calculate_normal_gradient(sigma_x, sigma_y, z)
                    
                    return grad_x, grad_y, grad_z
                else:
                    return 0, 0, 0
        return 0, 0, 0
    
    def _calculate_normal_gradient(self, sigma_x, sigma_y, z):
        h = self.h_total
        zi_norm = 2 * z / h
        return 0.1 * (sigma_x + sigma_y) * zi_norm / h * np.exp(-abs(zi_norm) * 2)
    
    def calculate_3d_pagano_full(self, n_terms=10):
        h = self.h_total
        b = self.b
        
        Mx = 6 * self.D[0, 0] * self.kappa[0] + 6 * self.D[0, 1] * self.kappa[1]
        My = 6 * self.D[1, 1] * self.kappa[1] + 6 * self.D[0, 1] * self.kappa[0]
        
        z_mid = 0
        Q_bar_mid = self._get_stiffness_matrix_at_z(z_mid)
        
        C11 = Q_bar_mid[0, 0]
        C22 = Q_bar_mid[1, 1]
        C12 = Q_bar_mid[0, 1]
        C66 = Q_bar_mid[2, 2]
        C55 = self.ply_props['G13']
        C44 = self.ply_props['G23']
        
        n_y = 50
        n_z = 50
        y_arr = np.linspace(-b/2, b/2, n_y)
        z_arr = np.linspace(-h/2, h/2, n_z)
        
        tau_xz = np.zeros((n_y, n_z))
        tau_yz = np.zeros((n_y, n_z))
        sigma_z = np.zeros((n_y, n_z))
        
        for m in range(1, n_terms + 1):
            alpha_m = m * np.pi / b
            
            X_m = np.sin(alpha_m * b/2) / (alpha_m * b/2) if m > 0 else 1
            
            for i, y in enumerate(y_arr):
                sin_term = np.sin(alpha_m * y)
                cos_term = np.cos(alpha_m * y)
                
                for j, z in enumerate(z_arr):
                    zi_norm = 2 * z / h
                    
                    f_z = (1 - zi_norm**2)
                    f_z_deriv = -4 * zi_norm / h
                    
                    A_m = Mx * X_m / (C11 * h**2 / 12)
                    B_m = My * X_m / (C22 * h**2 / 12)
                    
                    tau_xz[i, j] += A_m * C55 * alpha_m * f_z * sin_term / m
                    tau_yz[i, j] += B_m * C44 * alpha_m * f_z * cos_term / m
                    sigma_z[i, j] += 0.1 * (A_m + B_m) * f_z * cos_term / m
        
        return y_arr, z_arr, tau_xz, tau_yz, sigma_z
    
    def _get_stiffness_matrix_at_z(self, z):
        for layer in self.layer_stresses:
            if layer['z_bottom'] <= z <= layer['z_top']:
                return layer['Q_bar']
        return self.layer_stresses[0]['Q_bar']
    
    def get_peak_stresses(self, method='pagano'):
        if method == 'pagano':
            y_arr, z_arr, tau_xz, tau_yz, sigma_z = self.calculate_3d_pagano_full()
        elif method == 'integral':
            z_fine, tau_xz_1d, tau_yz_1d, sigma_z_1d = self.calculate_interlaminar_stress_integral()
            return {
                'tau_xz_max': np.max(np.abs(tau_xz_1d)),
                'tau_yz_max': np.max(np.abs(tau_yz_1d)),
                'sigma_z_max': np.max(np.abs(sigma_z_1d)),
                'z_at_max_tau_xz': z_fine[np.argmax(np.abs(tau_xz_1d))],
                'z_at_max_tau_yz': z_fine[np.argmax(np.abs(tau_yz_1d))],
                'z_at_max_sigma_z': z_fine[np.argmax(np.abs(sigma_z_1d))]
            }
        else:
            y_arr, z_arr, tau_xz, tau_yz, sigma_z = self.calculate_interlaminar_shear_exact()
        
        return {
            'tau_xz_max': np.max(np.abs(tau_xz)),
            'tau_yz_max': np.max(np.abs(tau_yz)),
            'sigma_z_max': np.max(np.abs(sigma_z)),
            'y_at_max_tau_xz': y_arr[np.unravel_index(np.argmax(np.abs(tau_xz)), tau_xz.shape)[0]],
            'z_at_max_tau_xz': z_arr[np.unravel_index(np.argmax(np.abs(tau_xz)), tau_xz.shape)[1]]
        }
    
    def _get_bending_moment(self):
        return 6 * self.D[0, 0] * self.kappa[0] + 6 * self.D[0, 1] * self.kappa[1]
    
    def _get_bending_moment_y(self):
        return 6 * self.D[1, 1] * self.kappa[1] + 6 * self.D[0, 1] * self.kappa[0]
    
    def print_summary(self):
        print("=" * 70)
        print("复合材料层板三维应力分析 (Pagano方法 + 边界层模型)")
        print("=" * 70)
        print(f"铺层角度: {self.layup}")
        print(f"总层数: {self.n_plies}")
        print(f"单层厚度: {self.t_ply * 1000:.3f} mm")
        print(f"总厚度: {self.h_total * 1000:.3f} mm")
        print(f"层板宽度: {self.b * 1000:.2f} mm")
        
        print("\nA矩阵 (面内刚度):")
        self._print_matrix(self.A)
        
        print("\nB矩阵 (耦合刚度):")
        self._print_matrix(self.B)
        
        print("\nD矩阵 (弯曲刚度):")
        self._print_matrix(self.D)
        
        print("\n中面应变:")
        print(f"  εx0 = {self.eps0[0]:.6e}")
        print(f"  εy0 = {self.eps0[1]:.6e}")
        print(f"  γxy0 = {self.eps0[2]:.6e}")
        
        print("\n曲率:")
        print(f"  κx = {self.kappa[0]:.6e}")
        print(f"  κy = {self.kappa[1]:.6e}")
        print(f"  κxy = {self.kappa[2]:.6e}")
        
        print("\n" + "=" * 70)
    
    def _print_matrix(self, matrix):
        for i in range(3):
            print(f"  [{matrix[i, 0]:.6e}, {matrix[i, 1]:.6e}, {matrix[i, 2]:.6e}]")


def plot_3d_stress_distribution(laminate, title_prefix=""):
    fig = plt.figure(figsize=(15, 12))
    
    y_arr, z_arr, tau_xz, tau_yz, sigma_z = laminate.calculate_3d_pagano_full()
    
    Y, Z = np.meshgrid(y_arr, z_arr)
    
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(Y * 1000, Z * 1000, tau_xz.T / 1e6, 
                             cmap='RdBu_r', edgecolor='none', alpha=0.9)
    ax1.set_xlabel('y (mm)')
    ax1.set_ylabel('z (mm)')
    ax1.set_zlabel('τ_xz (MPa)')
    ax1.set_title('XZ方向层间剪应力分布')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(Y * 1000, Z * 1000, tau_yz.T / 1e6, 
                             cmap='RdBu_r', edgecolor='none', alpha=0.9)
    ax2.set_xlabel('y (mm)')
    ax2.set_ylabel('z (mm)')
    ax2.set_zlabel('τ_yz (MPa)')
    ax2.set_title('YZ方向层间剪应力分布')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
    
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    surf3 = ax3.plot_surface(Y * 1000, Z * 1000, sigma_z.T / 1e6, 
                             cmap='RdYlGn', edgecolor='none', alpha=0.9)
    ax3.set_xlabel('y (mm)')
    ax3.set_ylabel('z (mm)')
    ax3.set_zlabel('σ_z (MPa)')
    ax3.set_title('层间剥离应力分布')
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
    
    ax4 = fig.add_subplot(2, 2, 4)
    
    y_mid = len(y_arr) // 2
    ax4.plot(z_arr * 1000, tau_xz[y_mid, :] / 1e6, 'b-', linewidth=2, label='τ_xz (中面)')
    ax4.plot(z_arr * 1000, tau_yz[y_mid, :] / 1e6, 'r-', linewidth=2, label='τ_yz (中面)')
    ax4.plot(z_arr * 1000, sigma_z[y_mid, :] / 1e6, 'g-', linewidth=2, label='σ_z (中面)')
    
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.set_xlabel('z (mm)')
    ax4.set_ylabel('应力 (MPa)')
    ax4.set_title('中面处应力沿厚度分布')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'{title_prefix}三维弹性解 (Pagano方法)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_comparison(laminate, title_prefix=""):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    y_arr, z_arr, tau_xz_3d, tau_yz_3d, sigma_z_3d = laminate.calculate_3d_pagano_full()
    
    z_int, tau_xz_int, tau_yz_int, sigma_z_int = laminate.calculate_interlaminar_stress_integral()
    
    Qx = 1000
    Qy = 500
    z_fsdm, tau_xz_fsdm, tau_yz_fsdm = laminate._calculate_fsdm_shear(Qx, Qy)
    
    y_mid_idx = len(y_arr) // 2
    
    ax = axes[0, 0]
    ax.plot(z_arr * 1000, tau_xz_3d[y_mid_idx, :] / 1e6, 'b-', linewidth=2.5, label='Pagano 3D')
    ax.plot(z_int * 1000, tau_xz_int / 1e6, 'r--', linewidth=2, label='积分方法')
    ax.plot(z_fsdm * 1000, tau_xz_fsdm / 1e6, 'g-.', linewidth=2, label='FSDT')
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('τ_xz (MPa)')
    ax.set_title('XZ方向层间剪应力 (中面)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(z_arr * 1000, tau_yz_3d[y_mid_idx, :] / 1e6, 'b-', linewidth=2.5, label='Pagano 3D')
    ax.plot(z_int * 1000, tau_yz_int / 1e6, 'r--', linewidth=2, label='积分方法')
    ax.plot(z_fsdm * 1000, tau_yz_fsdm / 1e6, 'g-.', linewidth=2, label='FSDT')
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('τ_yz (MPa)')
    ax.set_title('YZ方向层间剪应力 (中面)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    ax.plot(z_arr * 1000, sigma_z_3d[y_mid_idx, :] / 1e6, 'b-', linewidth=2.5, label='Pagano 3D')
    ax.plot(z_int * 1000, sigma_z_int / 1e6, 'r--', linewidth=2, label='积分方法')
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('σ_z (MPa)')
    ax.set_title('层间剥离应力 (中面)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    z_mid_idx = len(z_arr) // 2
    ax.plot(y_arr * 1000, tau_xz_3d[:, z_mid_idx] / 1e6, 'b-', linewidth=2.5)
    ax.axvline(x=self.b/2 * 1000, color='r', linestyle='--', alpha=0.7, label='自由边界')
    ax.axvline(x=-self.b/2 * 1000, color='r', linestyle='--', alpha=0.7)
    ax.set_xlabel('y (mm)')
    ax.set_ylabel('τ_xz (MPa)')
    ax.set_title('XZ方向剪应力沿宽度分布 (中面)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(y_arr * 1000, tau_yz_3d[:, z_mid_idx] / 1e6, 'b-', linewidth=2.5)
    ax.axvline(x=self.b/2 * 1000, color='r', linestyle='--', alpha=0.7, label='自由边界')
    ax.axvline(x=-self.b/2 * 1000, color='r', linestyle='--', alpha=0.7)
    ax.set_xlabel('y (mm)')
    ax.set_ylabel('τ_yz (MPa)')
    ax.set_title('YZ方向剪应力沿宽度分布 (中面)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 2]
    ax.plot(y_arr * 1000, sigma_z_3d[:, z_mid_idx] / 1e6, 'b-', linewidth=2.5)
    ax.axvline(x=self.b/2 * 1000, color='r', linestyle='--', alpha=0.7, label='自由边界')
    ax.axvline(x=-self.b/2 * 1000, color='r', linestyle='--', alpha=0.7)
    ax.set_xlabel('y (mm)')
    ax.set_ylabel('σ_z (MPa)')
    ax.set_title('剥离应力沿宽度分布 (中面)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{title_prefix}不同方法对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def main():
    ply_properties = {
        'E1': 138e9,
        'E2': 8.96e9,
        'E3': 8.96e9,
        'G12': 7.1e9,
        'G13': 7.1e9,
        'G23': 3.9e9,
        'nu12': 0.3,
        'nu13': 0.3,
        'nu23': 0.3
    }
    
    layup = [0, 45, -45, 90, 90, -45, 45, 0]
    ply_thickness = 0.125e-3
    
    laminate = CompositeLaminate3D(ply_properties, layup, ply_thickness, plate_width=0.01)
    
    Mx = 100.0
    My = 50.0
    Mxy = 0.0
    
    laminate.apply_bending_load(Mx=Mx, My=My, Mxy=Mxy)
    laminate.print_summary()
    
    print("\n各层面内应力:")
    print("=" * 70)
    for k, layer in enumerate(laminate.layer_stresses):
        print(f"\n第{k+1}层 (θ={layer['theta']}°):")
        print(f"  底面 (z={layer['z_bottom']*1000:.3f}mm):")
        print(f"    σx={layer['sigma_bottom'][0]/1e6:.3f} MPa, σy={layer['sigma_bottom'][1]/1e6:.3f} MPa, τxy={layer['sigma_bottom'][2]/1e6:.3f} MPa")
        print(f"  顶面 (z={layer['z_top']*1000:.3f}mm):")
        print(f"    σx={layer['sigma_top'][0]/1e6:.3f} MPa, σy={layer['sigma_top'][1]/1e6:.3f} MPa, τxy={layer['sigma_top'][2]/1e6:.3f} MPa")
    
    print("\n" + "=" * 70)
    print("层间应力峰值分析:")
    print("=" * 70)
    
    print("\n方法1: 三维弹性解 (Pagano方法 + 边界层修正)")
    y_arr, z_arr, tau_xz_3d, tau_yz_3d, sigma_z_3d = laminate.calculate_3d_pagano_full()
    print(f"  最大 τ_xz: {np.max(np.abs(tau_xz_3d))/1e6:.3f} MPa")
    print(f"  最大 τ_yz: {np.max(np.abs(tau_yz_3d))/1e6:.3f} MPa")
    print(f"  最大 σ_z:  {np.max(np.abs(sigma_z_3d))/1e6:.3f} MPa")
    
    print("\n方法2: 精确积分方法")
    z_int, tau_xz_int, tau_yz_int, sigma_z_int = laminate.calculate_interlaminar_stress_integral()
    print(f"  最大 τ_xz: {np.max(np.abs(tau_xz_int))/1e6:.3f} MPa")
    print(f"  最大 τ_yz: {np.max(np.abs(tau_yz_int))/1e6:.3f} MPa")
    print(f"  最大 σ_z:  {np.max(np.abs(sigma_z_int))/1e6:.3f} MPa")
    
    print("\n方法3: 一阶剪切变形理论 (FSDT)")
    Qx = 1000
    Qy = 500
    z_fsdm, tau_xz_fsdm, tau_yz_fsdm = laminate._calculate_fsdm_shear(Qx, Qy)
    print(f"  最大 τ_xz: {np.max(np.abs(tau_xz_fsdm))/1e6:.3f} MPa")
    print(f"  最大 τ_yz: {np.max(np.abs(tau_yz_fsdm))/1e6:.3f} MPa")
    
    print("\n" + "=" * 70)
    print("边界效应分析:")
    print("=" * 70)
    
    z_mid_idx = len(z_arr) // 2
    tau_xz_center = tau_xz_3d[len(y_arr)//2, z_mid_idx]
    tau_xz_edge = tau_xz_3d[0, z_mid_idx]
    
    print(f"  中面处 τ_xz: {tau_xz_center/1e6:.3f} MPa")
    print(f"  边界处 τ_xz: {tau_xz_edge/1e6:.3f} MPa")
    print(f"  边界放大系数: {np.abs(tau_xz_edge/tau_xz_center):.2f}")
    
    sigma_z_center = sigma_z_3d[len(y_arr)//2, z_mid_idx]
    sigma_z_edge = sigma_z_3d[0, z_mid_idx]
    print(f"  中面处 σ_z: {sigma_z_center/1e6:.3f} MPa")
    print(f"  边界处 σ_z: {sigma_z_edge/1e6:.3f} MPa")
    print(f"  边界放大系数: {np.abs(sigma_z_edge/sigma_z_center):.2f}")
    
    fig1 = plot_3d_stress_distribution(laminate, title_prefix="[0/45/-45/90]s ")
    fig1.savefig('composite_stresses_3d.png', dpi=300, bbox_inches='tight')
    print("\n三维应力分布图已保存为 'composite_stresses_3d.png'")
    
    fig2 = plot_comparison(laminate, title_prefix="[0/45/-45/90]s ")
    fig2.savefig('composite_stresses_comparison.png', dpi=300, bbox_inches='tight')
    print("应力对比图已保存为 'composite_stresses_comparison.png'")
    
    plt.show()


if __name__ == "__main__":
    main()
