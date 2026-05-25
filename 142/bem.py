import numpy as np
from typing import Optional, Tuple, Callable
from scipy.sparse.linalg import gmres, LinearOperator
from scipy.linalg import lu_factor, lu_solve

from .mesh import Mesh
from .kernels import HelmholtzKernel
from .fmm import FMMMatrixFree
from .broadband_fmm import BroadbandFMMMatrixFree


class BEMSolver:
    def __init__(self, mesh: Mesh, k: float, c: float = 343.0, rho: float = 1.21):
        self.mesh = mesh
        self.k = k
        self.c = c
        self.rho = rho
        self.omega = k * c
        self.kernel = HelmholtzKernel(k, c, rho)
        self._H = None
        self._G = None
        self.use_fmm = False
        self.fmm = None

    def enable_fmm(self, p: int = 8, max_points_per_leaf: int = 25):
        self.use_fmm = True
        self.fmm_type = 'standard'
        self.fmm = FMMMatrixFree(self.k, p, max_points_per_leaf)
        self.fmm.setup(self.mesh.centroids, self.mesh.normals)

    def enable_broadband_fmm(self, tol: float = 1e-6,
                             max_points_per_leaf: int = 25,
                             kr_switch: float = 8.0):
        self.use_fmm = True
        self.fmm_type = 'broadband'
        self.fmm = BroadbandFMMMatrixFree(self.k, tol, max_points_per_leaf, kr_switch)
        self.fmm.setup(self.mesh.centroids, self.mesh.normals)

    def get_expansion_info(self) -> dict:
        if not hasattr(self, 'fmm_type') or self.fmm_type != 'broadband':
            return {}
        info = {}
        if hasattr(self.fmm.fmm, 'octree') and self.fmm.fmm.octree:
            for level, nodes in self.fmm.fmm.octree.nodes_by_level.items():
                if nodes:
                    config = nodes[0].expansion_config
                    if config:
                        kr = self.k * nodes[0].size
                        info[level] = {
                            'kr': kr,
                            'box_size': nodes[0].size,
                            'expansion_type': config.expansion_type.value,
                            'p': config.p,
                            'n_theta': config.n_theta,
                            'n_phi': config.n_phi
                        }
        return info

    def assemble_matrices(self, use_singular_correction: bool = True):
        if self.use_fmm:
            return

        N = self.mesh.num_elements
        centroids = self.mesh.centroids
        normals = self.mesh.normals
        areas = self.mesh.areas

        self._H = np.zeros((N, N), dtype=complex)
        self._G = np.zeros((N, N), dtype=complex)

        for i in range(N):
            xi = centroids[i]
            ni = normals[i]
            for j in range(N):
                xj = centroids[j]
                r = xi - xj
                r_norm = np.linalg.norm(r)

                if i == j:
                    if use_singular_correction:
                        self._H[i, j] = 0.5
                        self._G[i, j] = self._singular_integration(i)
                    else:
                        self._H[i, j] = 0.5
                        self._G[i, j] = 0.0
                else:
                    self._H[i, j] = self.kernel.double_layer(r, ni) * areas[j]
                    self._G[i, j] = self.kernel.single_layer(r) * areas[j]

    def _singular_integration(self, element_idx: int) -> complex:
        elem = self.mesh.get_element(element_idx)
        verts = elem.vertices
        centroid = elem.centroid
        area = elem.area

        if len(verts) == 3:
            return self._triangular_singular_integration(verts, centroid)
        elif len(verts) == 4:
            return self._quadrilateral_singular_integration(verts, centroid)
        return 0.0

    def _triangular_singular_integration(self, verts: np.ndarray,
                                          centroid: np.ndarray) -> complex:
        n_gauss = 7
        gauss_pts = np.array([
            [1/3, 1/3, 1/3],
            [0.797426985353087, 0.101286507323456, 0.101286507323456],
            [0.101286507323456, 0.797426985353087, 0.101286507323456],
            [0.101286507323456, 0.101286507323456, 0.797426985353087],
            [0.059715871789770, 0.470142064105115, 0.470142064105115],
            [0.470142064105115, 0.059715871789770, 0.470142064105115],
            [0.470142064105115, 0.470142064105115, 0.059715871789770],
        ])
        gauss_wts = np.array([
            0.225000000000000,
            0.125939180544827,
            0.125939180544827,
            0.125939180544827,
            0.132394152788506,
            0.132394152788506,
            0.132394152788506,
        ])

        v1, v2, v3 = verts
        result = 0j

        for pt, wt in zip(gauss_pts, gauss_wts):
            x = pt[0] * v1 + pt[1] * v2 + pt[2] * v3
            r = centroid - x
            r_norm = np.linalg.norm(r)
            if r_norm > 1e-12:
                G = np.exp(-1j * self.k * r_norm) / (4 * np.pi * r_norm)
                area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
                result += G * wt * area

        return result

    def _quadrilateral_singular_integration(self, verts: np.ndarray,
                                             centroid: np.ndarray) -> complex:
        v1, v2, v3, v4 = verts
        tri1 = [v1, v2, v3]
        tri2 = [v1, v3, v4]

        return (self._triangular_singular_integration(np.array(tri1), centroid) +
                self._triangular_singular_integration(np.array(tri2), centroid))

    def solve_dirichlet(self, p_incident: np.ndarray,
                        tol: float = 1e-6,
                        maxiter: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        N = self.mesh.num_elements
        p_boundary = p_incident.copy()

        if self.use_fmm:
            def matvec(x):
                return 0.5 * x + self.fmm.matvec_double_layer(x) - 1j * self.k * self.fmm.matvec_single_layer(x)

            A = LinearOperator((N, N), matvec=matvec, dtype=complex)
            b = 1j * self.k * p_boundary
            v, info = gmres(A, b, tol=tol, maxiter=maxiter)
        else:
            if self._H is None or self._G is None:
                self.assemble_matrices()

            A = self._H - 1j * self.k * self._G
            b = 1j * self.k * p_boundary
            lu, piv = lu_factor(A)
            v = lu_solve((lu, piv), b)

        p_total = p_boundary
        return p_total, v

    def solve_neumann(self, v_incident: np.ndarray,
                      tol: float = 1e-6,
                      maxiter: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        N = self.mesh.num_elements
        v_boundary = v_incident.copy()

        if self.use_fmm:
            def matvec(x):
                return self.fmm.matvec_single_layer(x)

            A = LinearOperator((N, N), matvec=matvec, dtype=complex)
            b = -0.5 * v_boundary - self.fmm.matvec_double_layer(v_boundary)
            p, info = gmres(A, b, tol=tol, maxiter=maxiter)
        else:
            if self._H is None or self._G is None:
                self.assemble_matrices()

            A = self._G
            b = -0.5 * v_boundary - self._H @ v_boundary
            lu, piv = lu_factor(A)
            p = lu_solve((lu, piv), b)

        return p, v_boundary

    def solve_combined(self, p_incident: np.ndarray,
                       alpha: complex = 1j,
                       tol: float = 1e-6,
                       maxiter: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        N = self.mesh.num_elements

        if self.use_fmm:
            def matvec(x):
                sl = self.fmm.matvec_single_layer(x)
                dl = self.fmm.matvec_double_layer(x)
                return dl - 1j * self.k * sl + alpha * sl - alpha / (1j * self.k) * dl

            A = LinearOperator((N, N), matvec=matvec, dtype=complex)
            b = alpha * p_incident + 1j * self.k * p_incident
            p, info = gmres(A, b, tol=tol, maxiter=maxiter)
            v = (p - p_incident) * 1j * self.k
        else:
            if self._H is None or self._G is None:
                self.assemble_matrices()

            A = self._H - 1j * self.k * self._G + alpha * self._G - alpha / (1j * self.k) * self._H
            b = alpha * p_incident + 1j * self.k * p_incident
            lu, piv = lu_factor(A)
            p = lu_solve((lu, piv), b)
            v = (p - p_incident) * 1j * self.k

        return p, v

    def evaluate_field(self, field_points: np.ndarray,
                       p_surface: np.ndarray,
                       v_surface: np.ndarray) -> np.ndarray:
        N_fp = len(field_points)
        N_el = self.mesh.num_elements
        centroids = self.mesh.centroids
        normals = self.mesh.normals
        areas = self.mesh.areas

        p_field = np.zeros(N_fp, dtype=complex)

        for i, fp in enumerate(field_points):
            for j in range(N_el):
                r = fp - centroids[j]
                r_norm = np.linalg.norm(r)
                if r_norm > 1e-12:
                    G = self.kernel.single_layer(r)
                    dGdn = self.kernel.double_layer(r, normals[j])
                    p_field[i] += (G * 1j * self.k * v_surface[j] - dGdn * p_surface[j]) * areas[j]

        return p_field


class AcousticScattering:
    def __init__(self, mesh: Mesh, k: float, c: float = 343.0, rho: float = 1.21):
        self.solver = BEMSolver(mesh, k, c, rho)
        self.mesh = mesh
        self.k = k

    def set_incident_plane_wave(self, direction: np.ndarray, amplitude: complex = 1.0):
        direction = direction / np.linalg.norm(direction)
        centroids = self.mesh.centroids
        p_incident = amplitude * np.exp(-1j * self.k * np.dot(centroids, direction))
        return p_incident

    def set_incident_point_source(self, source_pos: np.ndarray, amplitude: complex = 1.0):
        centroids = self.mesh.centroids
        r = centroids - source_pos
        r_norm = np.linalg.norm(r, axis=1)
        p_incident = amplitude * np.exp(-1j * self.k * r_norm) / (4 * np.pi * r_norm)
        return p_incident

    def solve(self, p_incident: np.ndarray,
              method: str = 'dirichlet',
              use_fmm: bool = False,
              use_broadband_fmm: bool = False,
              **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if use_broadband_fmm:
            self.solver.enable_broadband_fmm(**kwargs)
        elif use_fmm:
            self.solver.enable_fmm(**kwargs)

        if method == 'dirichlet':
            p_surface, v_surface = self.solver.solve_dirichlet(p_incident)
        elif method == 'neumann':
            v_incident = np.gradient(p_incident) / (1j * self.k * self.solver.rho * self.solver.omega)
            p_surface, v_surface = self.solver.solve_neumann(v_incident)
        elif method == 'combined':
            p_surface, v_surface = self.solver.solve_combined(p_incident, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        return p_surface, v_surface

    def get_expansion_info(self) -> dict:
        return self.solver.get_expansion_info()

    def compute_scattered_field(self, field_points: np.ndarray,
                                p_surface: np.ndarray,
                                v_surface: np.ndarray) -> np.ndarray:
        return self.solver.evaluate_field(field_points, p_surface, v_surface)

    def compute_target_strength(self, field_points: np.ndarray,
                                p_surface: np.ndarray,
                                v_surface: np.ndarray,
                                reference_distance: float = 1.0) -> np.ndarray:
        p_scattered = self.compute_scattered_field(field_points, p_surface, v_surface)
        r = np.linalg.norm(field_points, axis=1)
        ts = 20 * np.log10(np.abs(p_scattered) * r / reference_distance)
        return ts
