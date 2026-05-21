import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.interpolate import griddata


class EITMesh:
    def __init__(self, n_radius=8, n_angles=16, r=1.0):
        self.n_radius = n_radius
        self.n_angles = n_angles
        self.r = r
        self.nodes = None
        self.elements = None
        self.electrodes = None
        self._generate_mesh()

    def _generate_mesh(self):
        nodes = []
        node_id = 0
        node_map = {}

        for i in range(self.n_radius + 1):
            radius = (i / self.n_radius) * self.r
            if i == 0:
                nodes.append([0.0, 0.0])
                node_map[(i, 0)] = node_id
                node_id += 1
            else:
                angles = np.linspace(0, 2 * np.pi, self.n_angles, endpoint=False)
                for j, theta in enumerate(angles):
                    x = radius * np.cos(theta)
                    y = radius * np.sin(theta)
                    nodes.append([x, y])
                    node_map[(i, j)] = node_id
                    node_id += 1

        self.nodes = np.array(nodes)
        self.n_nodes = len(nodes)

        elements = []
        for i in range(self.n_radius):
            if i == 0:
                for j in range(self.n_angles):
                    j_next = (j + 1) % self.n_angles
                    tri = [node_map[(0, 0)], node_map[(1, j)], node_map[(1, j_next)]]
                    elements.append(tri)
            else:
                for j in range(self.n_angles):
                    j_next = (j + 1) % self.n_angles
                    tri1 = [node_map[(i, j)], node_map[(i + 1, j)], node_map[(i + 1, j_next)]]
                    tri2 = [node_map[(i, j)], node_map[(i + 1, j_next)], node_map[(i, j_next)]]
                    elements.append(tri1)
                    elements.append(tri2)

        self.elements = np.array(elements, dtype=int)
        self.n_elements = len(elements)

        self.electrodes = [node_map[(self.n_radius, j)] for j in range(self.n_angles)]
        self.n_electrodes = len(self.electrodes)

    def get_boundary_nodes(self):
        return self.electrodes

    def element_area(self, e_idx):
        nodes = self.nodes[self.elements[e_idx]]
        x1, y1 = nodes[0]
        x2, y2 = nodes[1]
        x3, y3 = nodes[2]
        return 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    def element_gradient(self, e_idx):
        nodes = self.nodes[self.elements[e_idx]]
        x1, y1 = nodes[0]
        x2, y2 = nodes[1]
        x3, y3 = nodes[2]
        area = self.element_area(e_idx)
        B = np.array([
            [y2 - y3, y3 - y1, y1 - y2],
            [x3 - x2, x1 - x3, x2 - x1]
        ]) / (2 * area)
        return B


class EITForward:
    def __init__(self, mesh):
        self.mesh = mesh
        self.n_nodes = mesh.n_nodes
        self.n_elements = mesh.n_elements
        self.n_electrodes = mesh.n_electrodes

    def assemble_stiffness_matrix(self, sigma):
        K = lil_matrix((self.n_nodes, self.n_nodes))
        
        for e_idx in range(self.n_elements):
            elem_nodes = self.mesh.elements[e_idx]
            area = self.mesh.element_area(e_idx)
            B = self.mesh.element_gradient(e_idx)
            
            sigma_e = sigma[e_idx] if sigma.ndim == 1 else sigma
            Ke = sigma_e * area * (B.T @ B)
            
            for i in range(3):
                for j in range(3):
                    K[elem_nodes[i], elem_nodes[j]] += Ke[i, j]
        
        return csr_matrix(K)

    def solve(self, sigma, current_pattern):
        K = self.assemble_stiffness_matrix(sigma)
        
        I = np.zeros(self.n_nodes)
        for e_idx, curr in enumerate(current_pattern):
            I[self.mesh.electrodes[e_idx]] = curr
        
        ground_node = self.mesh.electrodes[0]
        K[ground_node, :] = 0
        K[:, ground_node] = 0
        K[ground_node, ground_node] = 1
        I[ground_node] = 0
        
        V = spsolve(K, I)
        return V

    def get_boundary_voltages(self, V):
        return V[self.mesh.electrodes]

    def simulate_measurements(self, sigma):
        n_elec = self.n_electrodes
        measurements = []
        
        for i in range(n_elec):
            current = np.zeros(n_elec)
            current[i] = 1.0
            current[(i + 1) % n_elec] = -1.0
            
            V = self.solve(sigma, current)
            v_boundary = self.get_boundary_voltages(V)
            
            for j in range(n_elec):
                if j != i and j != (i + 1) % n_elec:
                    measurements.append(v_boundary[j])
        
        return np.array(measurements)


class EITInverse:
    def __init__(self, forward_solver, max_iter=20, reg_param=1e-3, tol=1e-4):
        self.forward = forward_solver
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.tol = tol
        self.mesh = forward_solver.mesh

    def compute_jacobian(self, sigma):
        n_elec = self.mesh.n_electrodes
        n_elem = self.mesh.n_elements
        
        jacobian_rows = []
        
        for i in range(n_elec):
            current = np.zeros(n_elec)
            current[i] = 1.0
            current[(i + 1) % n_elec] = -1.0
            
            V = self.forward.solve(sigma, current)
            
            for e_idx in range(n_elem):
                elem_nodes = self.mesh.elements[e_idx]
                area = self.mesh.element_area(e_idx)
                B = self.mesh.element_gradient(e_idx)
                
                u_e = V[elem_nodes]
                grad_u = B @ u_e
                
                dsigma = -area * np.sum(grad_u**2)
                
                for j in range(n_elec):
                    if j != i and j != (i + 1) % n_elec:
                        row = np.zeros(n_elem)
                        row[e_idx] = dsigma
                        jacobian_rows.append(row)
        
        return np.array(jacobian_rows)

    def reconstruct(self, measured_voltages, sigma0=None):
        if sigma0 is None:
            sigma0 = np.ones(self.mesh.n_elements)
        
        sigma = sigma0.copy()
        
        for iteration in range(self.max_iter):
            predicted = self.forward.simulate_measurements(sigma)
            residual = measured_voltages - predicted
            
            J = self.compute_jacobian(sigma)
            
            JTJ = J.T @ J
            regularization = self.reg_param * np.eye(self.mesh.n_elements)
            delta = np.linalg.solve(JTJ + regularization, J.T @ residual)
            
            sigma += delta
            
            sigma = np.maximum(sigma, 0.1)
            sigma = np.minimum(sigma, 10.0)
            
            error = np.linalg.norm(residual) / np.linalg.norm(measured_voltages)
            print(f"Iteration {iteration + 1}: Relative error = {error:.6f}")
            
            if error < self.tol:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        return sigma


def visualize_conductivity(mesh, sigma, title="Conductivity Distribution"):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]
    
    elem_centers = np.zeros((mesh.n_elements, 2))
    for e_idx in range(mesh.n_elements):
        elem_nodes = mesh.nodes[mesh.elements[e_idx]]
        elem_centers[e_idx] = np.mean(elem_nodes, axis=0)
    
    xi = np.linspace(-1.1, 1.1, 100)
    yi = np.linspace(-1.1, 1.1, 100)
    XI, YI = np.meshgrid(xi, yi)
    
    zi = griddata(elem_centers, sigma, (XI, YI), method='cubic')
    
    mask = XI**2 + YI**2 > 1.0
    zi[mask] = np.nan
    
    im = ax.pcolormesh(XI, YI, zi, cmap='viridis', shading='auto')
    
    circle = Circle((0, 0), 1.0, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)
    
    elec_x = mesh.nodes[mesh.electrodes, 0]
    elec_y = mesh.nodes[mesh.electrodes, 1]
    ax.scatter(elec_x, elec_y, c='red', s=100, marker='o', edgecolors='white', zorder=5)
    
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    plt.colorbar(im, ax=ax, label='Conductivity (S/m)')
    plt.tight_layout()
    return fig, ax
