import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


class DarcyFlowSolver:
    def __init__(self, nx, ny, dx, dy):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.n_nodes = nx * ny
        
        self.K = np.ones((ny, nx))
        self.h = np.zeros((ny, nx))
        
        self.bc_dirichlet = {}
        self.bc_neumann = {}
        
        self.flux_bc_left = {}
        self.flux_bc_right = {}
        self.flux_bc_bottom = {}
        self.flux_bc_top = {}
    
    def set_hydraulic_conductivity(self, K_field):
        if K_field.shape != (self.ny, self.nx):
            raise ValueError(f"K场维度不匹配，需要({self.ny}, {self.nx})，实际为{K_field.shape}")
        self.K = K_field.copy()
    
    def set_dirichlet_bc(self, i, j, value):
        idx = j * self.nx + i
        self.bc_dirichlet[idx] = value
    
    def set_neumann_bc(self, i, j, value):
        idx = j * self.nx + i
        self.bc_neumann[idx] = value
    
    def set_flux_bc_left(self, j, flux):
        self.flux_bc_left[j] = flux
    
    def set_flux_bc_right(self, j, flux):
        self.flux_bc_right[j] = flux
    
    def set_flux_bc_bottom(self, i, flux):
        self.flux_bc_bottom[i] = flux
    
    def set_flux_bc_top(self, i, flux):
        self.flux_bc_top[i] = flux
    
    def _node_idx(self, i, j):
        return j * self.nx + i
    
    @staticmethod
    def harmonic_mean(k1, k2):
        if k1 <= 0 or k2 <= 0:
            return 0.0
        return 2.0 * k1 * k2 / (k1 + k2)
    
    @staticmethod
    def arithmetic_mean(k1, k2):
        return 0.5 * (k1 + k2)
    
    def assemble_matrix(self):
        A = lil_matrix((self.n_nodes, self.n_nodes))
        b = np.zeros(self.n_nodes)
        
        for j in range(self.ny):
            for i in range(self.nx):
                idx = self._node_idx(i, j)
                
                if idx in self.bc_dirichlet:
                    A[idx, idx] = 1.0
                    b[idx] = self.bc_dirichlet[idx]
                    continue
                
                coeff = 0.0
                
                if i > 0:
                    k_left = self.harmonic_mean(self.K[j, i], self.K[j, i-1])
                    val = k_left / (self.dx ** 2)
                    A[idx, self._node_idx(i-1, j)] = -val
                    coeff += val
                else:
                    if j in self.flux_bc_left:
                        b[idx] += self.flux_bc_left[j] / self.dx
                    elif idx in self.bc_neumann:
                        b[idx] += self.bc_neumann[idx] / self.dx
                
                if i < self.nx - 1:
                    k_right = self.harmonic_mean(self.K[j, i], self.K[j, i+1])
                    val = k_right / (self.dx ** 2)
                    A[idx, self._node_idx(i+1, j)] = -val
                    coeff += val
                else:
                    if j in self.flux_bc_right:
                        b[idx] -= self.flux_bc_right[j] / self.dx
                    elif idx in self.bc_neumann:
                        b[idx] -= self.bc_neumann[idx] / self.dx
                
                if j > 0:
                    k_bottom = self.harmonic_mean(self.K[j, i], self.K[j-1, i])
                    val = k_bottom / (self.dy ** 2)
                    A[idx, self._node_idx(i, j-1)] = -val
                    coeff += val
                else:
                    if i in self.flux_bc_bottom:
                        b[idx] += self.flux_bc_bottom[i] / self.dy
                    elif idx in self.bc_neumann:
                        b[idx] += self.bc_neumann[idx] / self.dy
                
                if j < self.ny - 1:
                    k_top = self.harmonic_mean(self.K[j, i], self.K[j+1, i])
                    val = k_top / (self.dy ** 2)
                    A[idx, self._node_idx(i, j+1)] = -val
                    coeff += val
                else:
                    if i in self.flux_bc_top:
                        b[idx] -= self.flux_bc_top[i] / self.dy
                    elif idx in self.bc_neumann:
                        b[idx] -= self.bc_neumann[idx] / self.dy
                
                A[idx, idx] = coeff
        
        return csr_matrix(A), b
    
    def solve(self):
        A, b = self.assemble_matrix()
        h_flat = spsolve(A, b)
        self.h = h_flat.reshape((self.ny, self.nx))
        return self.h
    
    def compute_velocity(self):
        u_face, v_face = self._compute_face_velocities()
        
        u_center = np.zeros((self.ny, self.nx))
        v_center = np.zeros((self.ny, self.nx))
        
        for j in range(self.ny):
            for i in range(self.nx):
                if i == 0:
                    u_center[j, i] = u_face[j, i]
                elif i == self.nx - 1:
                    u_center[j, i] = u_face[j, i-1]
                else:
                    u_center[j, i] = 0.5 * (u_face[j, i-1] + u_face[j, i])
                
                if j == 0:
                    v_center[j, i] = v_face[j, i]
                elif j == self.ny - 1:
                    v_center[j, i] = v_face[j-1, i]
                else:
                    v_center[j, i] = 0.5 * (v_face[j-1, i] + v_face[j, i])
        
        return u_center, v_center
    
    def _compute_face_velocities(self):
        u_face = np.zeros((self.ny, self.nx))
        v_face = np.zeros((self.ny, self.nx))
        
        for j in range(self.ny):
            for i in range(self.nx - 1):
                k_interface = self.harmonic_mean(self.K[j, i], self.K[j, i+1])
                u_face[j, i] = -k_interface * (self.h[j, i+1] - self.h[j, i]) / self.dx
        
        for j in range(self.ny - 1):
            for i in range(self.nx):
                k_interface = self.harmonic_mean(self.K[j, i], self.K[j+1, i])
                v_face[j, i] = -k_interface * (self.h[j+1, i] - self.h[j, i]) / self.dy
        
        return u_face, v_face
    
    def check_flux_conservation(self):
        u_face, v_face = self._compute_face_velocities()
        
        print("\n" + "="*60)
        print("通量守恒检查")
        print("="*60)
        
        left_flux = 0.0
        right_flux = 0.0
        bottom_flux = 0.0
        top_flux = 0.0
        
        for j in range(self.ny):
            left_flux += u_face[j, 0] * self.dy
        
        for j in range(self.ny):
            right_flux += u_face[j, -2] * self.dy
        
        for i in range(self.nx):
            bottom_flux += v_face[0, i] * self.dx
        
        for i in range(self.nx):
            top_flux += v_face[-2, i] * self.dx
        
        total_inflow = 0.0
        total_outflow = 0.0
        
        if left_flux > 0:
            total_inflow += left_flux
        else:
            total_outflow += -left_flux
        
        if right_flux > 0:
            total_inflow += right_flux
        else:
            total_outflow += -right_flux
        
        if bottom_flux > 0:
            total_inflow += bottom_flux
        else:
            total_outflow += -bottom_flux
        
        if top_flux > 0:
            total_inflow += top_flux
        else:
            total_outflow += -top_flux
        
        print(f"左边界通量: {left_flux:.10e}")
        print(f"右边界通量: {right_flux:.10e}")
        print(f"底边界通量: {bottom_flux:.10e}")
        print(f"顶边界通量: {top_flux:.10e}")
        print(f"总流入: {total_inflow:.10e}")
        print(f"总流出: {total_outflow:.10e}")
        print(f"质量守恒误差: {abs(total_inflow - total_outflow):.10e}")
        
        if max(abs(total_inflow), abs(total_outflow)) > 1e-20:
            print(f"相对误差: {abs(total_inflow - total_outflow) / max(abs(total_inflow), abs(total_outflow)):.10e}")
        
        return total_inflow, total_outflow
    
    def check_interface_flux_continuity(self):
        print("\n" + "="*60)
        print("界面通量连续性检查")
        print("="*60)
        
        u_face, v_face = self._compute_face_velocities()
        
        max_u_jump = 0.0
        max_v_jump = 0.0
        
        for j in range(self.ny):
            for i in range(1, self.nx - 1):
                if abs(self.K[j, i] - self.K[j, i-1]) > 1e-15:
                    k_left = self.harmonic_mean(self.K[j, i-1], self.K[j, i])
                    flux_left = -k_left * (self.h[j, i] - self.h[j, i-1]) / self.dx
                    
                    k_right = self.harmonic_mean(self.K[j, i], self.K[j, i+1])
                    flux_right = -k_right * (self.h[j, i+1] - self.h[j, i]) / self.dx
                    
                    jump = abs(flux_left - flux_right)
                    if jump > max_u_jump:
                        max_u_jump = jump
        
        for j in range(1, self.ny - 1):
            for i in range(self.nx):
                if abs(self.K[j, i] - self.K[j-1, i]) > 1e-15:
                    k_bottom = self.harmonic_mean(self.K[j-1, i], self.K[j, i])
                    flux_bottom = -k_bottom * (self.h[j, i] - self.h[j-1, i]) / self.dy
                    
                    k_top = self.harmonic_mean(self.K[j, i], self.K[j+1, i])
                    flux_top = -k_top * (self.h[j+1, i] - self.h[j, i]) / self.dy
                    
                    jump = abs(flux_bottom - flux_top)
                    if jump > max_v_jump:
                        max_v_jump = jump
        
        print(f"x方向界面通量最大跳变: {max_u_jump:.10e}")
        print(f"y方向界面通量最大跳变: {max_v_jump:.10e}")
        
        if max_u_jump < 1e-10 and max_v_jump < 1e-10:
            print("✓ 界面通量连续性良好（调和平均保证了法向通量连续")
        else:
            print("⚠ 界面通量存在不连续，可能需要检查")
        
        return max_u_jump, max_v_jump
    
    def plot_results(self, u=None, v=None):
        if u is None or v is None:
            u, v = self.compute_velocity()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        x = np.linspace(0, (self.nx - 1) * self.dx, self.nx)
        y = np.linspace(0, (self.ny - 1) * self.dy, self.ny)
        X, Y = np.meshgrid(x, y)
        
        im1 = axes[0, 0].contourf(X, Y, self.h, levels=25, cmap='viridis')
        axes[0, 0].set_title('水头分布 h(x,y)')
        axes[0, 0].set_xlabel('x (m)')
        axes[0, 0].set_ylabel('y (m)')
        plt.colorbar(im1, ax=axes[0, 0], label='水头 (m)')
        
        im2 = axes[0, 1].contourf(X, Y, np.log10(self.K), levels=25, cmap='jet')
        axes[0, 1].set_title('渗透系数场 log10(K)')
        axes[0, 1].set_xlabel('x (m)')
        axes[0, 1].set_ylabel('y (m)')
        plt.colorbar(im2, ax=axes[0, 1], label='log10(K)')
        
        speed = np.sqrt(u ** 2 + v ** 2)
        im3 = axes[0, 2].contourf(X, Y, np.log10(speed + 1e-20), levels=25, cmap='hot')
        axes[0, 2].set_title('流速大小 log10(|q|)')
        axes[0, 2].set_xlabel('x (m)')
        axes[0, 2].set_ylabel('y (m)')
        plt.colorbar(im3, ax=axes[0, 2], label='log10(速度)')
        
        axes[1, 0].streamplot(X, Y, u, v, density=0.8, cmap='autumn', linewidth=0.5)
        axes[1, 0].set_title('流线图')
        axes[1, 0].set_xlabel('x (m)')
        axes[1, 0].set_ylabel('y (m)')
        
        mid_j = self.ny // 2
        axes[1, 1].plot(x, self.h[mid_j, :], 'b-', linewidth=2)
        axes[1, 1].set_title(f'中线水头分布 (y={mid_j*self.dy:.1f}m)')
        axes[1, 1].set_xlabel('x (m)')
        axes[1, 1].set_ylabel('水头 (m)')
        axes[1, 1].grid(True, alpha=0.3)
        
        u_face, _ = self._compute_face_velocities()
        mid_j = self.ny // 2
        axes[1, 2].semilogy(x[:-1], np.abs(u_face[mid_j, :-1]) + 1e-20, 'r-', linewidth=2)
        axes[1, 2].set_title(f'界面法向通量 (y={mid_j*self.dy:.1f}m)')
        axes[1, 2].set_xlabel('x (m)')
        axes[1, 2].set_ylabel('|q| (m/s)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('darcy_flow_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n水头范围: {self.h.min():.4f} m ~ {self.h.max():.4f} m")
        print(f"最大流速: {speed.max():.6e} m/s")
        
        return u, v


def example_layered():
    print("="*60)
    print("示例1: 层状介质 - 验证调和平均保证通量连续")
    print("="*60)
    
    nx, ny = 60, 40
    dx, dy = 1.0, 1.0
    
    solver = DarcyFlowSolver(nx, ny, dx, dy)
    
    K = np.ones((ny, nx)) * 1e-5
    
    K[0:ny//3, :] = 1e-4
    K[ny//3:2*ny//3, :] = 1e-6
    K[2*ny//3:, :] = 1e-5
    
    solver.set_hydraulic_conductivity(K)
    
    for j in range(ny):
        solver.set_dirichlet_bc(0, j, 10.0)
        solver.set_dirichlet_bc(nx-1, j, 5.0)
    
    for i in range(nx):
        solver.set_neumann_bc(i, 0, 0.0)
        solver.set_neumann_bc(i, ny-1, 0.0)
    
    h = solver.solve()
    u, v = solver.plot_results()
    
    solver.check_flux_conservation()
    solver.check_interface_flux_continuity()
    
    return solver


def example_lens():
    print("\n" + "="*60)
    print("示例2: 含低渗透透镜体的非均质介质")
    print("="*60)
    
    nx, ny = 60, 60
    dx, dy = 1.0, 1.0
    
    solver = DarcyFlowSolver(nx, ny, dx, dy)
    
    K = np.ones((ny, nx)) * 1e-5
    
    cx, cy, r = 30, 30, 10
    y_grid, x_grid = np.ogrid[:ny, :nx]
    dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
    K[dist <= r] = 1e-8
    
    solver.set_hydraulic_conductivity(K)
    
    for j in range(ny):
        solver.set_dirichlet_bc(0, j, 10.0)
        solver.set_dirichlet_bc(nx-1, j, 5.0)
    
    for i in range(nx):
        solver.set_neumann_bc(i, 0, 0.0)
        solver.set_neumann_bc(i, ny-1, 0.0)
    
    h = solver.solve()
    u, v = solver.plot_results()
    
    solver.check_flux_conservation()
    solver.check_interface_flux_continuity()
    
    return solver


def example_well():
    print("\n" + "="*60)
    print("示例3: 抽水井问题")
    print("="*60)
    
    nx, ny = 50, 50
    dx, dy = 2.0, 2.0
    
    solver = DarcyFlowSolver(nx, ny, dx, dy)
    
    K = np.ones((ny, nx)) * 5e-6
    solver.set_hydraulic_conductivity(K)
    
    for j in range(ny):
        solver.set_dirichlet_bc(0, j, 20.0)
        solver.set_dirichlet_bc(nx-1, j, 18.0)
    
    for i in range(nx):
        solver.set_neumann_bc(i, 0, 0.0)
        solver.set_neumann_bc(i, ny-1, 0.0)
    
    well_i, well_j = 25, 25
    Q = 0.001
    solver.set_neumann_bc(well_i, well_j, Q/(dx*dy))
    
    h = solver.solve()
    u, v = solver.plot_results()
    
    drawdown = 20.0 - h[well_j, well_i]
    print(f"\n井中心降深: {drawdown:.4f} m")
    
    solver.check_flux_conservation()
    
    return solver


def example_compare_methods():
    print("\n" + "="*60)
    print("示例4: 调和平均 vs 算术平均对比")
    print("="*60)
    
    nx, ny = 40, 20
    dx, dy = 1.0, 1.0
    
    solver = DarcyFlowSolver(nx, ny, dx, dy)
    
    K = np.ones((ny, nx)) * 1e-5
    K[:, nx//2:] = 1e-7
    
    solver.set_hydraulic_conductivity(K)
    
    for j in range(ny):
        solver.set_dirichlet_bc(0, j, 10.0)
        solver.set_dirichlet_bc(nx-1, j, 5.0)
    
    for i in range(nx):
        solver.set_neumann_bc(i, 0, 0.0)
        solver.set_neumann_bc(i, ny-1, 0.0)
    
    h = solver.solve()
    u, v = solver.compute_velocity()
    
    u_face, v_face = solver._compute_face_velocities()
    
    print("\n界面处通量分析 (x=20m):")
    print("-" * 50)
    
    interface_idx = nx // 2
    for j in range(0, ny, 4):
        k_left = solver.K[j, interface_idx-1]
        k_right = solver.K[j, interface_idx]
        
        k_harmonic = DarcyFlowSolver.harmonic_mean(k_left, k_right)
        k_arithmetic = DarcyFlowSolver.arithmetic_mean(k_left, k_right)
        
        flux_harmonic = u_face[j, interface_idx-1]
        
        print(f"y={j*dy:3.0f}m: K左={k_left:.1e}, K右={k_right:.1e}")
        print(f"       调和平均K={k_harmonic:.2e}, 算术平均K={k_arithmetic:.2e}")
        print(f"       界面通量={flux_harmonic:.6e} m/s")
        
        if abs(flux_harmonic) > 0:
            print(f"       通量误差(调和)={flux_harmonic:.6e}")
    
    print("\n通量守恒检查:")
    solver.check_flux_conservation()
    solver.check_interface_flux_continuity()
    
    return solver


if __name__ == "__main__":
    print("达西方程有限差分求解器")
    print("控制方程: ∇·(K∇h) = 0 (稳态无汇源)")
    print("达西定律: q = -K∇h")
    print("界面处理: 调和平均保证法向通量连续")
    
    solver1 = example_layered()
    solver2 = example_lens()
    solver3 = example_well()
    solver4 = example_compare_methods()
    
    print("\n" + "="*60)
    print("所有计算完成！")
    print("="*60)
    print("\n关键改进:")
    print("1. 使用调和平均处理异质界面")
    print("2. 界面处法向通量连续")
    print("3. 添加通量守恒验证")
    print("4. 流线图和界面通量分析")
    print("结果已保存为 darcy_flow_results.png")
