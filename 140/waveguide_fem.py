import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from typing import List, Tuple, Dict, Optional


class Material:
    def __init__(self, epsilon_r, mu_r=1.0, loss_tan_delta=0.0, sigma=0.0):
        self.epsilon_r = np.asarray(epsilon_r, dtype=complex)
        self.mu_r = np.asarray(mu_r, dtype=complex)
        self.loss_tan_delta = loss_tan_delta
        self.sigma = sigma
        
        if self.epsilon_r.ndim == 0:
            self.epsilon_r = self.epsilon_r * np.eye(3)
        elif self.epsilon_r.shape == (3,):
            self.epsilon_r = np.diag(self.epsilon_r)
        
        if self.mu_r.ndim == 0:
            self.mu_r = self.mu_r * np.eye(3)
        elif self.mu_r.shape == (3,):
            self.mu_r = np.diag(self.mu_r)
        
        self.epsilon0 = 8.854e-12
        self.mu0 = 4 * np.pi * 1e-7
    
    def get_epsilon(self, omega):
        eps = self.epsilon_r * self.epsilon0
        eps[2, 2] *= (1 + 1j * self.loss_tan_delta)
        eps[2, 2] += 1j * self.sigma / omega if omega > 0 else 0
        return eps
    
    def get_mu(self):
        return self.mu_r * self.mu0


class TriangularMesh:
    def __init__(self, nodes, elements, element_materials=None):
        self.nodes = np.asarray(nodes)
        self.elements = np.asarray(elements, dtype=int)
        self.element_materials = element_materials or [0] * len(elements)
        
        self._build_edges()
        self._build_boundary_edges()
    
    def _build_edges(self):
        edge_set = {}
        self.edges = []
        self.element_edges = []
        
        for elem_idx, elem in enumerate(self.elements):
            elem_edges = []
            for i in range(3):
                n1 = elem[i]
                n2 = elem[(i + 1) % 3]
                edge_key = tuple(sorted((n1, n2)))
                
                if edge_key not in edge_set:
                    edge_set[edge_key] = len(self.edges)
                    self.edges.append(edge_key)
                
                elem_edges.append((edge_set[edge_key], 1 if n1 < n2 else -1))
            
            self.element_edges.append(elem_edges)
        
        self.edges = np.array(self.edges)
        self.element_edges = np.array(self.element_edges)
    
    def _build_boundary_edges(self):
        edge_counts = {}
        for elem in self.elements:
            for i in range(3):
                n1 = elem[i]
                n2 = elem[(i + 1) % 3]
                edge_key = tuple(sorted((n1, n2)))
                edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1
        
        self.boundary_edges = set()
        for edge_key, count in edge_counts.items():
            if count == 1:
                self.boundary_edges.add(edge_key)
        
        self.boundary_edge_indices = []
        for i, edge in enumerate(self.edges):
            if tuple(edge) in self.boundary_edges:
                self.boundary_edge_indices.append(i)
        
        self.boundary_edge_indices = np.array(self.boundary_edge_indices)
    
    def is_edge_on_boundary(self, edge_idx):
        return tuple(self.edges[edge_idx]) in self.boundary_edges
    
    def get_element_centroid(self, elem_idx):
        elem = self.elements[elem_idx]
        return np.mean(self.nodes[elem], axis=0)
    
    def plot(self, ax=None, show_nodes=False, show_edge_indices=False):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        for i, elem in enumerate(self.elements):
            points = self.nodes[elem]
            poly = Polygon(points, closed=True, fill=False, edgecolor='black', linewidth=0.5)
            ax.add_patch(poly)
        
        if show_nodes:
            ax.plot(self.nodes[:, 0], self.nodes[:, 1], 'ro', markersize=3)
            for i, node in enumerate(self.nodes):
                ax.annotate(str(i), node, fontsize=8)
        
        if show_edge_indices:
            for i, edge in enumerate(self.edges):
                midpoint = np.mean(self.nodes[edge], axis=0)
                color = 'red' if self.is_edge_on_boundary(i) else 'blue'
                ax.annotate(str(i), midpoint, fontsize=7, color=color)
        
        ax.set_aspect('equal')
        ax.autoscale()
        return ax


def generate_rectangular_mesh(a, b, nx, ny, material_id=0):
    x = np.linspace(0, a, nx)
    y = np.linspace(0, b, ny)
    
    nodes = []
    for j in range(ny):
        for i in range(nx):
            nodes.append([x[i], y[j]])
    nodes = np.array(nodes)
    
    elements = []
    element_materials = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n00 = j * nx + i
            n10 = j * nx + i + 1
            n01 = (j + 1) * nx + i
            n11 = (j + 1) * nx + i + 1
            
            elements.append([n00, n10, n11])
            element_materials.append(material_id)
            elements.append([n00, n11, n01])
            element_materials.append(material_id)
    
    return TriangularMesh(nodes, elements, element_materials)


def generate_ridged_waveguide_mesh(a, b, ridge_width, ridge_height, 
                                    nx, ny, ridge_nx=None, ridge_ny=None,
                                    substrate_material=0, ridge_material=1):
    ridge_nx = ridge_nx or nx
    ridge_ny = ridge_ny or ny
    
    x1 = (a - ridge_width) / 2
    x2 = x1 + ridge_width
    y1 = ridge_height
    
    nodes = []
    node_id_map = {}
    
    x_coords = np.concatenate([
        np.linspace(0, x1, nx // 2)[:-1],
        np.linspace(x1, x2, ridge_nx),
        np.linspace(x2, a, nx // 2)[1:]
    ])
    
    y_lower = np.linspace(0, y1, ridge_ny)
    y_upper = np.linspace(y1, b, ny)
    y_coords = np.concatenate([y_lower[:-1], y_upper])
    
    for j, y in enumerate(y_coords):
        for i, x in enumerate(x_coords):
            node_id_map[(i, j)] = len(nodes)
            nodes.append([x, y])
    
    nodes = np.array(nodes)
    elements = []
    element_materials = []
    
    for j in range(len(y_coords) - 1):
        for i in range(len(x_coords) - 1):
            n00 = node_id_map[(i, j)]
            n10 = node_id_map[(i + 1, j)]
            n01 = node_id_map[(i, j + 1)]
            n11 = node_id_map[(i + 1, j + 1)]
            
            y_mid = (y_coords[j] + y_coords[j + 1]) / 2
            x_mid = (x_coords[i] + x_coords[i + 1]) / 2
            
            if y_mid < y1 and x1 < x_mid < x2:
                mat = ridge_material
            else:
                mat = substrate_material
            
            elements.append([n00, n10, n11])
            element_materials.append(mat)
            elements.append([n00, n11, n01])
            element_materials.append(mat)
    
    return TriangularMesh(nodes, elements, element_materials)


class WaveguideFEM:
    def __init__(self, mesh: TriangularMesh, materials: List[Material]):
        self.mesh = mesh
        self.materials = materials
        self.omega = None
        self.A = None
        self.B = None
        self.free_edge_indices = None
    
    def _get_free_edges(self):
        if self.free_edge_indices is None:
            free = []
            for i in range(len(self.mesh.edges)):
                if not self.mesh.is_edge_on_boundary(i):
                    free.append(i)
            self.free_edge_indices = np.array(free)
        return self.free_edge_indices
    
    def _compute_element_matrices(self, elem_idx, omega):
        elem = self.mesh.elements[elem_idx]
        elem_edges = self.mesh.element_edges[elem_idx]
        nodes = self.mesh.nodes[elem]
        
        mat_id = self.mesh.element_materials[elem_idx]
        material = self.materials[mat_id]
        
        eps = material.get_epsilon(omega)
        mu = material.get_mu()
        
        eps_xy = eps[:2, :2]
        eps_zz = eps[2, 2]
        mu_xy_inv = np.linalg.inv(mu[:2, :2])
        mu_zz_inv = 1.0 / mu[2, 2]
        
        x = nodes[:, 0]
        y = nodes[:, 1]
        
        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        
        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]]) / (2 * area)
        c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]]) / (2 * area)
        
        Ne = np.zeros((3, 2, 2))
        for i in range(3):
            edge_idx, sign = elem_edges[i]
            n1, n2 = self.mesh.edges[edge_idx]
            
            p1 = np.where(elem == n1)[0][0]
            p2 = np.where(elem == n2)[0][0]
            
            length = np.linalg.norm(nodes[p2] - nodes[p1])
            
            Ne[i] = sign * length * np.array([[b[i], 0], [0, c[i]]])
        
        A_tt = np.zeros((3, 3), dtype=complex)
        A_tz = np.zeros((3, 3), dtype=complex)
        A_zt = np.zeros((3, 3), dtype=complex)
        A_zz = np.zeros((3, 3), dtype=complex)
        B_tt = np.zeros((3, 3), dtype=complex)
        B_zz = np.zeros((3, 3), dtype=complex)
        
        for i in range(3):
            for j in range(3):
                curl_i = np.array([-c[i], b[i]])
                curl_j = np.array([-c[j], b[j]])
                
                A_tt[i, j] = area * (curl_i @ mu_zz_inv @ curl_j)
                B_tt[i, j] = area * (Ne[i].flatten() @ eps_xy @ Ne[j].flatten()) * (1/12)
        
        const = area / 60
        M = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
        for i in range(3):
            for j in range(3):
                A_zz[i, j] = area * (b[i] * mu_xy_inv[0, 0] * b[j] + 
                                    b[i] * mu_xy_inv[0, 1] * c[j] + 
                                    c[i] * mu_xy_inv[1, 0] * b[j] + 
                                    c[i] * mu_xy_inv[1, 1] * c[j]) / 12
                B_zz[i, j] = eps_zz * const * M[i, j]
        
        return A_tt, A_tz, A_zt, A_zz, B_tt, B_zz, area
    
    def assemble_matrices(self, omega):
        self.omega = omega
        n_edges = len(self.mesh.edges)
        n_nodes = len(self.mesh.nodes)
        
        free_edges = self._get_free_edges()
        n_free_edges = len(free_edges)
        
        edge_to_free = {idx: i for i, idx in enumerate(free_edges)}
        
        n_total = n_free_edges + n_nodes
        
        A = lil_matrix((n_total, n_total), dtype=complex)
        B = lil_matrix((n_total, n_total), dtype=complex)
        
        for elem_idx in range(len(self.mesh.elements)):
            A_tt, A_tz, A_zt, A_zz, B_tt, B_zz, area = \
                self._compute_element_matrices(elem_idx, omega)
            
            elem_edges = self.mesh.element_edges[elem_idx]
            elem_nodes = self.mesh.elements[elem_idx]
            
            for i in range(3):
                global_edge_i, sign_i = elem_edges[i]
                if global_edge_i in edge_to_free:
                    free_i = edge_to_free[global_edge_i]
                    
                    for j in range(3):
                        global_edge_j, sign_j = elem_edges[j]
                        if global_edge_j in edge_to_free:
                            free_j = edge_to_free[global_edge_j]
                            A[free_i, free_j] += sign_i * sign_j * A_tt[i, j]
                            B[free_i, free_j] += sign_i * sign_j * B_tt[i, j]
                    
                    for j in range(3):
                        node_j = elem_nodes[j]
                        dof_j = n_free_edges + node_j
                        A[free_i, dof_j] += sign_i * A_tz[i, j]
                        A[dof_j, free_i] += sign_j * A_zt[j, i]
            
            for i in range(3):
                node_i = elem_nodes[i]
                dof_i = n_free_edges + node_i
                
                for j in range(3):
                    node_j = elem_nodes[j]
                    dof_j = n_free_edges + node_j
                    A[dof_i, dof_j] += A_zz[i, j]
                    B[dof_i, dof_j] += B_zz[i, j]
        
        self.A = csr_matrix(A)
        self.B = csr_matrix(B)
        self.n_free_edges = n_free_edges
        
        return self.A, self.B
    
    def solve_eigenvalue(self, omega, k0_guess=None, n_modes=5):
        A, B = self.assemble_matrices(omega)
        
        if k0_guess is None:
            k0_guess = omega * np.sqrt(8.854e-12 * 4 * np.pi * 1e-7)
        
        sigma = k0_guess ** 2
        
        try:
            eigenvalues, eigenvectors = eigs(A, k=n_modes, M=B, sigma=sigma, which='LM')
        except Exception as e:
            print(f"Eigenvalue solver warning: {e}")
            eigenvalues, eigenvectors = eigs(A, k=n_modes, M=B, sigma=sigma, which='SM')
        
        kc_squared = eigenvalues
        propagation_constants = np.sqrt(omega**2 * 8.854e-12 * 4 * np.pi * 1e-7 - kc_squared)
        
        sorted_idx = np.argsort(-np.real(propagation_constants))
        propagation_constants = propagation_constants[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        kc_squared = kc_squared[sorted_idx]
        
        return propagation_constants, kc_squared, eigenvectors
    
    def get_mode_fields(self, eigenvector, x, y, z=0):
        n_free_edges = self.n_free_edges
        
        et_vec = eigenvector[:n_free_edges]
        ez_vec = eigenvector[n_free_edges:]
        
        free_edges = self._get_free_edges()
        
        Ex = np.zeros_like(x, dtype=complex)
        Ey = np.zeros_like(x, dtype=complex)
        Ez = np.zeros_like(x, dtype=complex)
        
        for elem_idx in range(len(self.mesh.elements)):
            elem = self.mesh.elements[elem_idx]
            elem_edges = self.mesh.element_edges[elem_idx]
            nodes = self.mesh.nodes[elem]
            
            x_nodes = nodes[:, 0]
            y_nodes = nodes[:, 1]
            
            area = 0.5 * abs((x_nodes[1] - x_nodes[0]) * (y_nodes[2] - y_nodes[0]) - 
                           (x_nodes[2] - x_nodes[0]) * (y_nodes[1] - y_nodes[0]))
            
            b = np.array([y_nodes[1] - y_nodes[2], 
                          y_nodes[2] - y_nodes[0], 
                          y_nodes[0] - y_nodes[1]]) / (2 * area)
            c = np.array([x_nodes[2] - x_nodes[1], 
                          x_nodes[0] - x_nodes[2], 
                          x_nodes[1] - x_nodes[0]]) / (2 * area)
            
            xi, yi = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
            xi = xi.flatten()
            yi = yi.flatten()
            mask = xi + yi <= 1
            
            for k in range(len(x)):
                px, py = x[k], y[k]
                
                L1 = ((y_nodes[1] - y_nodes[2]) * (px - x_nodes[2]) + 
                      (x_nodes[2] - x_nodes[1]) * (py - y_nodes[2])) / (2 * area)
                L2 = ((y_nodes[2] - y_nodes[0]) * (px - x_nodes[2]) + 
                      (x_nodes[0] - x_nodes[2]) * (py - y_nodes[2])) / (2 * area)
                L3 = 1 - L1 - L2
                
                if 0 <= L1 <= 1 and 0 <= L2 <= 1 and 0 <= L3 <= 1:
                    L = np.array([L1, L2, L3])
                    
                    for i in range(3):
                        global_edge, sign = elem_edges[i]
                        edge_pos = np.where(free_edges == global_edge)[0]
                        if len(edge_pos) > 0:
                            edge_idx = edge_pos[0]
                            Ne_x = sign * b[i]
                            Ne_y = sign * c[i]
                            
                            Ex[k] += et_vec[edge_idx] * Ne_x
                            Ey[k] += et_vec[edge_idx] * Ne_y
                    
                    for i in range(3):
                        Ez[k] += ez_vec[elem[i]] * L[i]
        
        return {'Ex': Ex, 'Ey': Ey, 'Ez': Ez}
    
    def plot_mesh(self, ax=None):
        return self.mesh.plot(ax=ax, show_nodes=False, show_edge_indices=False)
    
    def plot_mode(self, eigenvector, field='E', resolution=50, title=None):
        x_min, y_min = np.min(self.mesh.nodes, axis=0)
        x_max, y_max = np.max(self.mesh.nodes, axis=0)
        
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)
        
        x_flat = X.flatten()
        y_flat = Y.flatten()
        
        fields = self.get_mode_fields(eigenvector, x_flat, y_flat)
        
        if field == 'E':
            E_mag = np.sqrt(np.abs(fields['Ex'])**2 + 
                           np.abs(fields['Ey'])**2 + 
                           np.abs(fields['Ez'])**2)
            data = E_mag.reshape(X.shape)
            if title is None:
                title = 'Electric Field Magnitude'
        elif field == 'Ez':
            data = np.real(fields['Ez']).reshape(X.shape)
            if title is None:
                title = 'Ez Field (real part)'
        elif field == 'Et':
            Et_mag = np.sqrt(np.abs(fields['Ex'])**2 + np.abs(fields['Ey'])**2)
            data = Et_mag.reshape(X.shape)
            if title is None:
                title = 'Transverse Electric Field Magnitude'
        else:
            raise ValueError(f"Unknown field type: {field}")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(X, Y, data, 50, cmap='viridis')
        self.plot_mesh(ax=ax)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(title)
        plt.colorbar(contour, ax=ax)
        ax.set_aspect('equal')
        
        return fig, ax


def demo_rectangular_waveguide_fem():
    print("=" * 60)
    print("FEM Analysis of Rectangular Waveguide")
    print("=" * 60)
    
    a = 22.86e-3
    b = 10.16e-3
    frequency = 10e9
    omega = 2 * np.pi * frequency
    
    print(f"\nWaveguide: {a*1000:.2f} mm x {b*1000:.2f} mm")
    print(f"Frequency: {frequency/1e9:.2f} GHz")
    
    print("\nGenerating mesh...")
    mesh = generate_rectangular_mesh(a, b, 20, 10)
    print(f"  Nodes: {len(mesh.nodes)}")
    print(f"  Elements: {len(mesh.elements)}")
    print(f"  Edges: {len(mesh.edges)}")
    print(f"  Boundary edges: {len(mesh.boundary_edge_indices)}")
    
    materials = [Material(epsilon_r=1.0)]
    
    print("\nAssembling FEM matrices...")
    solver = WaveguideFEM(mesh, materials)
    
    print("Solving eigenvalue problem...")
    betas, kc_squared, eigenvectors = solver.solve_eigenvalue(omega, n_modes=5)
    
    print("\nComputed propagation constants:")
    for i, beta in enumerate(betas):
        if np.abs(np.imag(beta)) < 1e-10:
            lambda_g = 2 * np.pi / np.real(beta) if np.real(beta) > 0 else float('inf')
            kc = np.sqrt(np.abs(np.real(kc_squared[i])))
            lambda_c = 2 * np.pi / kc if kc > 0 else float('inf')
            print(f"  Mode {i+1}: β = {np.real(beta):.4f} rad/m, "
                  f"λg = {lambda_g*1000:.2f} mm, λc = {lambda_c*1000:.2f} mm")
        else:
            print(f"  Mode {i+1}: β = {beta:.4e} (complex/evanescent)")
    
    print("\nGenerating mode plots...")
    fig, ax = solver.plot_mode(eigenvectors[:, 0], field='E', title='Mode 1 - E Field')
    fig.savefig('fem_mode1.png', dpi=150, bbox_inches='tight')
    
    fig2, ax2 = solver.plot_mode(eigenvectors[:, 0], field='Ez', title='Mode 1 - Ez Field')
    fig2.savefig('fem_mode1_ez.png', dpi=150, bbox_inches='tight')
    
    print("Plots saved.")
    plt.close('all')


def demo_ridged_waveguide():
    print("\n" + "=" * 60)
    print("FEM Analysis of Ridged Waveguide")
    print("=" * 60)
    
    a = 22.86e-3
    b = 10.16e-3
    ridge_width = 5e-3
    ridge_height = 4e-3
    frequency = 8e9
    omega = 2 * np.pi * frequency
    
    print(f"\nWaveguide: {a*1000:.2f} mm x {b*1000:.2f} mm")
    print(f"Ridge: {ridge_width*1000:.2f} mm x {ridge_height*1000:.2f} mm")
    print(f"Frequency: {frequency/1e9:.2f} GHz")
    
    print("\nGenerating ridged waveguide mesh...")
    mesh = generate_ridged_waveguide_mesh(a, b, ridge_width, ridge_height, 15, 12, 20, 10)
    print(f"  Nodes: {len(mesh.nodes)}")
    print(f"  Elements: {len(mesh.elements)}")
    
    materials = [Material(epsilon_r=1.0), Material(epsilon_r=9.0)]
    
    fig_mesh, ax_mesh = plt.subplots(figsize=(8, 6))
    mesh.plot(ax=ax_mesh)
    ax_mesh.set_title('Ridged Waveguide Mesh')
    fig_mesh.savefig('ridged_mesh.png', dpi=150, bbox_inches='tight')
    
    print("\nAssembling FEM matrices...")
    solver = WaveguideFEM(mesh, materials)
    
    print("Solving eigenvalue problem...")
    betas, kc_squared, eigenvectors = solver.solve_eigenvalue(omega, n_modes=4)
    
    print("\nComputed propagation constants:")
    for i, beta in enumerate(betas):
        if np.abs(np.imag(beta)) < 1e-10 and np.real(beta) > 0:
            lambda_g = 2 * np.pi / np.real(beta)
            print(f"  Mode {i+1}: β = {np.real(beta):.4f} rad/m, λg = {lambda_g*1000:.2f} mm")
        else:
            print(f"  Mode {i+1}: β = {beta:.4e}")
    
    print("\nGenerating mode plots...")
    fig, ax = solver.plot_mode(eigenvectors[:, 0], field='E', title='Ridged Mode 1 - E Field')
    fig.savefig('ridged_mode1.png', dpi=150, bbox_inches='tight')
    
    print("Ridged waveguide analysis complete!")
    plt.close('all')


def demo_anisotropic_medium():
    print("\n" + "=" * 60)
    print("FEM Analysis with Anisotropic Medium")
    print("=" * 60)
    
    a = 20e-3
    b = 10e-3
    frequency = 10e9
    omega = 2 * np.pi * frequency
    
    print(f"\nWaveguide: {a*1000:.2f} mm x {b*1000:.2f} mm")
    print(f"Frequency: {frequency/1e9:.2f} GHz")
    
    mesh = generate_rectangular_mesh(a, b, 15, 10)
    
    eps_tensor = [2.0, 3.0, 4.0]
    print(f"\nAnisotropic ε_r tensor: diag({eps_tensor})")
    materials = [Material(epsilon_r=eps_tensor)]
    
    solver = WaveguideFEM(mesh, materials)
    
    print("Solving eigenvalue problem...")
    betas, kc_squared, eigenvectors = solver.solve_eigenvalue(omega, n_modes=4)
    
    print("\nComputed propagation constants (anisotropic):")
    for i, beta in enumerate(betas):
        if np.real(beta) > 0:
            print(f"  Mode {i+1}: β = {np.real(beta):.4f} rad/m")
    
    print("\nAnisotropic analysis complete!")
    plt.close('all')


if __name__ == "__main__":
    demo_rectangular_waveguide_fem()
    demo_ridged_waveguide()
    demo_anisotropic_medium()
