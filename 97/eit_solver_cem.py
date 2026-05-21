import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, bmat
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.interpolate import griddata


class EITMesh:
    def __init__(self, n_radius=8, n_angles=16, r=1.0, electrode_angle_width=0.3):
        self.n_radius = n_radius
        self.n_angles = n_angles
        self.r = r
        self.electrode_angle_width = electrode_angle_width
        self.nodes = None
        self.elements = None
        self.electrodes = None
        self.electrode_nodes = None
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

        self._assign_electrodes(node_map)

    def _assign_electrodes(self, node_map):
        self.electrodes = []
        self.electrode_nodes = []
        
        half_width = self.electrode_angle_width / 2
        
        for e_idx in range(self.n_angles):
            center_angle = 2 * np.pi * e_idx / self.n_angles
            
            nodes_on_electrode = []
            for j in range(self.n_angles):
                node_angle = 2 * np.pi * j / self.n_angles
                
                angle_diff = abs(node_angle - center_angle)
                angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                
                if angle_diff <= half_width:
                    node_id = node_map[(self.n_radius, j)]
                    nodes_on_electrode.append(node_id)
            
            self.electrode_nodes.append(nodes_on_electrode)
            if nodes_on_electrode:
                self.electrodes.append(nodes_on_electrode[0])
            else:
                self.electrodes.append(node_map[(self.n_radius, e_idx)])
        
        self.n_electrodes = len(self.electrodes)

    def get_boundary_nodes(self):
        boundary = []
        for j in range(self.n_angles):
            boundary.append(self.nodes.shape[0] - self.n_angles + j)
        return boundary

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

    def electrode_length(self, e_idx):
        nodes = self.electrode_nodes[e_idx]
        if len(nodes) < 2:
            return 0.1 * self.r
        
        angles = []
        for node_id in nodes:
            x, y = self.nodes[node_id]
            angles.append(np.arctan2(y, x))
        
        angles = np.array(angles)
        angle_span = np.max(angles) - np.min(angles)
        if angle_span > np.pi:
            angle_span = 2 * np.pi - angle_span
        
        return self.r * angle_span


class EITForwardCEM:
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

    def assemble_cem_system(self, sigma, contact_impedance):
        n = self.n_nodes
        L = self.n_electrodes
        
        K = self.assemble_stiffness_matrix(sigma)
        
        C = lil_matrix((L, n))
        D = lil_matrix((L, L))
        
        for l in range(L):
            elec_nodes = self.mesh.electrode_nodes[l]
            z_l = contact_impedance[l]
            len_l = self.mesh.electrode_length(l)
            
            for node in elec_nodes:
                share = 1.0 / len(elec_nodes) if len(elec_nodes) > 0 else 1.0
                C[l, node] = -share
            
            D[l, l] = z_l / len_l if len_l > 0 else z_l
        
        P = lil_matrix((1, L))
        P[0, :] = 1.0
        
        sys_top = bmat([[K, C.T], [C, D]], format='csr')
        sys_bottom = csr_matrix((1, n + L))
        sys_bottom[0, n:n+L] = 1.0
        
        sys = bmat([[sys_top], [sys_bottom]], format='csr')
        
        return sys

    def solve_cem(self, sigma, contact_impedance, current_pattern):
        n = self.n_nodes
        L = self.n_electrodes
        
        sys = self.assemble_cem_system(sigma, contact_impedance)
        
        rhs = np.zeros(n + L + 1)
        rhs[n:n+L] = current_pattern
        
        x = spsolve(sys, rhs)
        
        u = x[:n]
        v_electrode = x[n:n+L]
        
        return u, v_electrode

    def get_boundary_voltages(self, u):
        boundary_voltages = []
        for l in range(self.n_electrodes):
            elec_nodes = self.mesh.electrode_nodes[l]
            if len(elec_nodes) > 0:
                boundary_voltages.append(np.mean(u[elec_nodes]))
            else:
                boundary_voltages.append(u[self.mesh.electrodes[l]])
        return np.array(boundary_voltages)

    def simulate_measurements(self, sigma, contact_impedance=None):
        if contact_impedance is None:
            contact_impedance = np.ones(self.n_electrodes) * 0.1
        
        n_elec = self.n_electrodes
        measurements = []
        
        for i in range(n_elec):
            current = np.zeros(n_elec)
            current[i] = 1.0
            current[(i + 1) % n_elec] = -1.0
            
            u, v_electrode = self.solve_cem(sigma, contact_impedance, current)
            
            for j in range(n_elec):
                if j != i and j != (i + 1) % n_elec:
                    measurements.append(v_electrode[j])
        
        return np.array(measurements)

    def simulate_measurements_simple(self, sigma):
        measurements = []
        n_elec = self.n_electrodes
        
        for i in range(n_elec):
            current = np.zeros(n_elec)
            current[i] = 1.0
            current[(i + 1) % n_elec] = -1.0
            
            K = self.assemble_stiffness_matrix(sigma)
            I = np.zeros(self.n_nodes)
            for e_idx, curr in enumerate(current):
                for node in self.mesh.electrode_nodes[e_idx]:
                    I[node] = curr / len(self.mesh.electrode_nodes[e_idx])
            
            ground_node = self.mesh.electrodes[0]
            K[ground_node, :] = 0
            K[:, ground_node] = 0
            K[ground_node, ground_node] = 1
            I[ground_node] = 0
            
            u = spsolve(K, I)
            
            for j in range(n_elec):
                if j != i and j != (i + 1) % n_elec:
                    elec_nodes = self.mesh.electrode_nodes[j]
                    measurements.append(np.mean(u[elec_nodes]))
        
        return np.array(measurements)


class EITInverseCEM:
    def __init__(self, forward_solver, max_iter=20, reg_sigma=1e-3, reg_z=1e-2, tol=1e-4):
        self.forward = forward_solver
        self.max_iter = max_iter
        self.reg_sigma = reg_sigma
        self.reg_z = reg_z
        self.tol = tol
        self.mesh = forward_solver.mesh

    def compute_jacobian_cem(self, sigma, contact_impedance):
        n_elec = self.mesh.n_electrodes
        n_elem = self.mesh.n_elements
        n_meas_per_pattern = n_elec - 2
        
        n_meas_total = n_elec * n_meas_per_pattern
        
        J_sigma = np.zeros((n_meas_total, n_elem))
        J_z = np.zeros((n_meas_total, n_elec))
        
        meas_idx = 0
        
        for i in range(n_elec):
            current = np.zeros(n_elec)
            current[i] = 1.0
            current[(i + 1) % n_elec] = -1.0
            
            u, v_electrode = self.forward.solve_cem(sigma, contact_impedance, current)
            
            for e_idx in range(n_elem):
                elem_nodes = self.mesh.elements[e_idx]
                area = self.mesh.element_area(e_idx)
                B = self.mesh.element_gradient(e_idx)
                
                u_e = u[elem_nodes]
                grad_u = B @ u_e
                
                dsigma = -area * np.sum(grad_u**2)
                
                pattern_meas_idx = 0
                for j in range(n_elec):
                    if j != i and j != (i + 1) % n_elec:
                        global_idx = i * n_meas_per_pattern + pattern_meas_idx
                        if global_idx < n_meas_total:
                            J_sigma[global_idx, e_idx] = dsigma
                        pattern_meas_idx += 1
            
            h = 1e-6
            for z_idx in range(n_elec):
                z_pert = contact_impedance.copy()
                z_pert[z_idx] += h
                
                _, v_pert = self.forward.solve_cem(sigma, z_pert, current)
                
                dv_dz = (v_pert - v_electrode) / h
                
                pattern_meas_idx = 0
                for j in range(n_elec):
                    if j != i and j != (i + 1) % n_elec:
                        global_idx = i * n_meas_per_pattern + pattern_meas_idx
                        if global_idx < n_meas_total:
                            J_z[global_idx, z_idx] = dv_dz[j]
                        pattern_meas_idx += 1
        
        return J_sigma, J_z

    def reconstruct_joint(self, measured_voltages, sigma0=None, z0=None):
        if sigma0 is None:
            sigma0 = np.ones(self.mesh.n_elements)
        if z0 is None:
            z0 = np.ones(self.mesh.n_electrodes) * 0.1
        
        sigma = sigma0.copy()
        z = z0.copy()
        
        n_elem = self.mesh.n_elements
        n_elec = self.mesh.n_electrodes
        
        for iteration in range(self.max_iter):
            predicted = self.forward.simulate_measurements(sigma, z)
            residual = measured_voltages - predicted
            
            J_sigma, J_z = self.compute_jacobian_cem(sigma, z)
            J = np.hstack([J_sigma, J_z])
            
            JTJ = J.T @ J
            reg_block = np.diag([self.reg_sigma] * n_elem + [self.reg_z] * n_elec)
            
            delta = np.linalg.solve(JTJ + reg_block, J.T @ residual)
            
            sigma += delta[:n_elem]
            z += delta[n_elem:]
            
            sigma = np.maximum(sigma, 0.1)
            sigma = np.minimum(sigma, 10.0)
            z = np.maximum(z, 0.001)
            z = np.minimum(z, 1.0)
            
            error = np.linalg.norm(residual) / np.linalg.norm(measured_voltages)
            print(f"Iteration {iteration + 1}: Relative error = {error:.6f}, Mean z = {np.mean(z):.4f}")
            
            if error < self.tol:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        return sigma, z

    def reconstruct_with_known_z(self, measured_voltages, contact_impedance, sigma0=None):
        if sigma0 is None:
            sigma0 = np.ones(self.mesh.n_elements)
        
        sigma = sigma0.copy()
        n_elem = self.mesh.n_elements
        
        for iteration in range(self.max_iter):
            predicted = self.forward.simulate_measurements(sigma, contact_impedance)
            residual = measured_voltages - predicted
            
            J_sigma, _ = self.compute_jacobian_cem(sigma, contact_impedance)
            
            JTJ = J_sigma.T @ J_sigma
            regularization = self.reg_sigma * np.eye(n_elem)
            delta = np.linalg.solve(JTJ + regularization, J_sigma.T @ residual)
            
            sigma += delta
            
            sigma = np.maximum(sigma, 0.1)
            sigma = np.minimum(sigma, 10.0)
            
            error = np.linalg.norm(residual) / np.linalg.norm(measured_voltages)
            print(f"Iteration {iteration + 1}: Relative error = {error:.6f}")
            
            if error < self.tol:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        return sigma


def visualize_conductivity(mesh, sigma, title="Conductivity Distribution", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
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
    
    return im, ax
