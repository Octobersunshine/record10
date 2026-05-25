import numpy as np
from typing import Callable, List, Dict, Optional, Tuple
from dataclasses import dataclass
from scipy.sparse.linalg import gmres, LinearOperator

from .mesh import Mesh
from .bem import BEMSolver
from .kernels import HelmholtzKernel


@dataclass
class ObjectiveFunction:
    name: str
    compute: Callable
    compute_derivative: Optional[Callable] = None


class ObjectiveFunctions:
    @staticmethod
    def sound_pressure_at_point(field_point: np.ndarray) -> ObjectiveFunction:
        def compute(p_surface: np.ndarray, v_surface: np.ndarray,
                    solver: BEMSolver) -> complex:
            p = solver.evaluate_field(np.array([field_point]), p_surface, v_surface)
            return p[0]

        def compute_derivative(p_surface: np.ndarray, v_surface: np.ndarray,
                               solver: BEMSolver) -> Tuple[np.ndarray, np.ndarray]:
            N = len(p_surface)
            dp_dp = np.zeros(N, dtype=complex)
            dp_dv = np.zeros(N, dtype=complex)

            centroids = solver.mesh.centroids
            normals = solver.mesh.normals
            areas = solver.mesh.areas
            k = solver.k

            for j in range(N):
                r = field_point - centroids[j]
                r_norm = np.linalg.norm(r)
                if r_norm > 1e-12:
                    G = np.exp(-1j * k * r_norm) / (4 * np.pi * r_norm)
                    grad_G = (-1j * k * r_norm - 1) * r * np.exp(-1j * k * r_norm) / (4 * np.pi * r_norm**3)
                    dGdn = np.sum(grad_G * normals[j])

                    dp_dp[j] = -dGdn * areas[j]
                    dp_dv[j] = 1j * k * G * areas[j]

            return dp_dp, dp_dv

        return ObjectiveFunction(
            name=f"p_at_{field_point}",
            compute=compute,
            compute_derivative=compute_derivative
        )

    @staticmethod
    def sound_pressure_level_at_point(field_point: np.ndarray,
                                       p_ref: float = 1e-6) -> ObjectiveFunction:
        def compute(p_surface: np.ndarray, v_surface: np.ndarray,
                    solver: BEMSolver) -> float:
            p = solver.evaluate_field(np.array([field_point]), p_surface, v_surface)
            return 20 * np.log10(np.abs(p[0]) / p_ref)

        return ObjectiveFunction(
            name=f"SPL_at_{field_point}",
            compute=compute
        )

    @staticmethod
    def target_strength(observation_direction: np.ndarray,
                        reference_distance: float = 1.0,
                        r_observation: float = 100.0) -> ObjectiveFunction:
        observation_point = observation_direction / np.linalg.norm(observation_direction) * r_observation

        def compute(p_surface: np.ndarray, v_surface: np.ndarray,
                    solver: BEMSolver) -> float:
            p = solver.evaluate_field(np.array([observation_point]), p_surface, v_surface)
            ts = 20 * np.log10(np.abs(p[0]) * r_observation / reference_distance)
            return ts

        def compute_derivative(p_surface: np.ndarray, v_surface: np.ndarray,
                               solver: BEMSolver) -> Tuple[np.ndarray, np.ndarray]:
            N = len(p_surface)
            dts_dp = np.zeros(N, dtype=complex)
            dts_dv = np.zeros(N, dtype=complex)

            centroids = solver.mesh.centroids
            normals = solver.mesh.normals
            areas = solver.mesh.areas
            k = solver.k

            p = solver.evaluate_field(np.array([observation_point]), p_surface, v_surface)
            p_scat = p[0]

            for j in range(N):
                r = observation_point - centroids[j]
                r_norm = np.linalg.norm(r)
                if r_norm > 1e-12:
                    G = np.exp(-1j * k * r_norm) / (4 * np.pi * r_norm)
                    grad_G = (-1j * k * r_norm - 1) * r * np.exp(-1j * k * r_norm) / (4 * np.pi * r_norm**3)
                    dGdn = np.sum(grad_G * normals[j])

                    dp_dp = -dGdn * areas[j]
                    dp_dv = 1j * k * G * areas[j]

                    dts_dp[j] = (20 / np.log(10)) * np.real(np.conj(p_scat) * dp_dp) / (np.abs(p_scat)**2)
                    dts_dv[j] = (20 / np.log(10)) * np.real(np.conj(p_scat) * dp_dv) / (np.abs(p_scat)**2)

            return dts_dp, dts_dv

        return ObjectiveFunction(
            name=f"TS_dir_{observation_direction}",
            compute=compute,
            compute_derivative=compute_derivative
        )

    @staticmethod
    def radiated_power() -> ObjectiveFunction:
        def compute(p_surface: np.ndarray, v_surface: np.ndarray,
                    solver: BEMSolver) -> float:
            areas = solver.mesh.areas
            intensity = 0.5 * np.real(p_surface * np.conj(v_surface))
            power = np.sum(intensity * areas)
            return power

        return ObjectiveFunction(
            name="radiated_power",
            compute=compute
        )

    @staticmethod
    def surface_pressure_norm() -> ObjectiveFunction:
        def compute(p_surface: np.ndarray, v_surface: np.ndarray,
                    solver: BEMSolver) -> float:
            areas = solver.mesh.areas
            return np.sum(np.abs(p_surface)**2 * areas)

        return ObjectiveFunction(
            name="surface_pressure_norm",
            compute=compute
        )


class AdjointSolver:
    def __init__(self, bem_solver: BEMSolver):
        self.solver = bem_solver
        self.k = bem_solver.k
        self.mesh = bem_solver.mesh
        self.kernel = bem_solver.kernel

    def solve_adjoint_dirichlet(self, dp_dJ: np.ndarray, dv_dJ: np.ndarray,
                                 tol: float = 1e-6, maxiter: int = 1000) -> np.ndarray:
        N = self.mesh.num_elements

        if self.solver.use_fmm:
            def matvec_adjoint(x):
                sl = self.solver.fmm.matvec_single_layer(x)
                dl = self.solver.fmm.matvec_double_layer(x)
                return 0.5 * x + dl.conj() + 1j * self.k * sl.conj()

            A_adj = LinearOperator((N, N), matvec=matvec_adjoint, dtype=complex)
            b = dv_dJ.conj()
            lambda_adj, info = gmres(A_adj, b, tol=tol, maxiter=maxiter)
        else:
            if self.solver._H is None or self.solver._G is None:
                self.solver.assemble_matrices()

            A_adj = self.solver._H.conj().T + 1j * self.k * self.solver._G.conj().T
            b = dv_dJ.conj()
            lambda_adj = np.linalg.solve(A_adj, b)

        return lambda_adj

    def solve_adjoint_combined(self, dp_dJ: np.ndarray, dv_dJ: np.ndarray,
                                alpha: complex = 1j,
                                tol: float = 1e-6, maxiter: int = 1000) -> np.ndarray:
        N = self.mesh.num_elements

        if self.solver.use_fmm:
            def matvec_adjoint(x):
                sl = self.solver.fmm.matvec_single_layer(x)
                dl = self.solver.fmm.matvec_double_layer(x)
                return (dl.conj() + 1j * self.k * sl.conj() +
                        alpha.conj() * sl.conj() - alpha.conj() / (-1j * self.k) * dl.conj())

            A_adj = LinearOperator((N, N), matvec=matvec_adjoint, dtype=complex)
            b = (dp_dJ + alpha.conj() / (-1j * self.k) * dv_dJ).conj()
            lambda_adj, info = gmres(A_adj, b, tol=tol, maxiter=maxiter)
        else:
            if self.solver._H is None or self.solver._G is None:
                self.solver.assemble_matrices()

            A_adj = (self.solver._H.conj().T + 1j * self.k * self.solver._G.conj().T +
                     alpha.conj() * self.solver._G.conj().T -
                     alpha.conj() / (-1j * self.k) * self.solver._H.conj().T)
            b = (dp_dJ + alpha.conj() / (-1j * self.k) * dv_dJ).conj()
            lambda_adj = np.linalg.solve(A_adj, b)

        return lambda_adj


class ShapeParameterization:
    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.design_variables: Dict[str, Callable] = {}
        self.num_dv = 0

    def add_sphere_radius(self, center: np.ndarray, name: str = "radius"):
        def update_mesh(radius: float) -> Mesh:
            vertices = self.mesh.vertices.copy()
            r_vec = vertices - center
            r_norm = np.linalg.norm(r_vec, axis=1, keepdims=True)
            r_norm = np.maximum(r_norm, 1e-12)
            scale = radius / r_norm
            vertices = center + r_vec * scale
            from .mesh import Mesh
            return Mesh(vertices, self.mesh.faces)

        def compute_derivative() -> np.ndarray:
            vertices = self.mesh.vertices
            r_vec = vertices - center
            r_norm = np.linalg.norm(r_vec, axis=1, keepdims=True)
            r_norm = np.maximum(r_norm, 1e-12)
            return r_vec / r_norm

        self.design_variables[name] = {
            'update': update_mesh,
            'derivative': compute_derivative,
            'type': 'global'
        }
        self.num_dv += 1

    def add_surface_displacement(self, element_indices: np.ndarray,
                                  direction: np.ndarray,
                                  name: str = "displacement"):
        direction = direction / np.linalg.norm(direction)

        def update_mesh(displacement: float) -> Mesh:
            vertices = self.mesh.vertices.copy()
            for idx in element_indices:
                elem = self.mesh.get_element(idx)
                for v_idx in range(len(elem.vertices)):
                    vertices[v_idx] += displacement * direction
            from .mesh import Mesh
            return Mesh(vertices, self.mesh.faces)

        def compute_derivative() -> np.ndarray:
            deriv = np.zeros((self.mesh.num_elements, 3))
            deriv[element_indices] = direction
            return deriv

        self.design_variables[name] = {
            'update': update_mesh,
            'derivative': compute_derivative,
            'type': 'local'
        }
        self.num_dv += 1

    def add_morphing(self, control_points: np.ndarray, name: str = "morph"):
        self.design_variables[name] = {
            'control_points': control_points,
            'type': 'morphing'
        }
        self.num_dv += len(control_points)


class SensitivityAnalyzer:
    def __init__(self, mesh: Mesh, k: float, c: float = 343.0, rho: float = 1.21):
        self.mesh = mesh
        self.k = k
        self.c = c
        self.rho = rho
        self.solver = BEMSolver(mesh, k, c, rho)
        self.adjoint_solver = AdjointSolver(self.solver)
        self.param = ShapeParameterization(mesh)

    def solve_primal(self, p_incident: np.ndarray, method: str = 'dirichlet',
                      use_fmm: bool = False, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if use_fmm:
            self.solver.enable_fmm(**kwargs)

        if method == 'dirichlet':
            p_surface, v_surface = self.solver.solve_dirichlet(p_incident)
        elif method == 'combined':
            p_surface, v_surface = self.solver.solve_combined(p_incident, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        self.p_surface = p_surface
        self.v_surface = v_surface
        return p_surface, v_surface

    def compute_sensitivity(self, objective: ObjectiveFunction,
                            dv_name: str,
                            method: str = 'dirichlet',
                            use_adjoint: bool = True) -> float:
        if not hasattr(self, 'p_surface'):
            raise RuntimeError("Must call solve_primal first!")

        if objective.compute_derivative is None:
            return self._compute_sensitivity_fd(objective, dv_name)

        dp_dJ, dv_dJ = objective.compute_derivative(
            self.p_surface, self.v_surface, self.solver
        )

        if use_adjoint:
            if method == 'dirichlet':
                lambda_adj = self.adjoint_solver.solve_adjoint_dirichlet(dp_dJ, dv_dJ)
            elif method == 'combined':
                lambda_adj = self.adjoint_solver.solve_adjoint_combined(dp_dJ, dv_dJ)
            else:
                raise ValueError(f"Unknown method: {method}")

            dJ_dx_shape = self._compute_shape_derivative(lambda_adj, dv_name)
            dJ_dx = np.real(np.sum(dp_dJ * self.p_surface) +
                             np.sum(dv_dJ * self.v_surface) +
                             np.sum(lambda_adj * dJ_dx_shape))
        else:
            dJ_dx = self._compute_sensitivity_fd(objective, dv_name)

        return dJ_dx

    def _compute_shape_derivative(self, lambda_adj: np.ndarray, dv_name: str) -> np.ndarray:
        N = self.mesh.num_elements
        centroids = self.mesh.centroids
        normals = self.mesh.normals
        areas = self.mesh.areas

        dX = self.param.design_variables[dv_name]['derivative']()

        dA = np.zeros(N)
        for i in range(N):
            elem = self.mesh.get_element(i)
            if len(elem.vertices) == 3:
                dA[i] = self._compute_area_derivative_tri(elem, dX)
            elif len(elem.vertices) == 4:
                dA[i] = self._compute_area_derivative_quad(elem, dX)

        dJ_dx = np.zeros(N, dtype=complex)
        k = self.k

        for i in range(N):
            for j in range(N):
                if i != j:
                    r = centroids[i] - centroids[j]
                    r_norm = np.linalg.norm(r)
                    if r_norm > 1e-12:
                        G = np.exp(-1j * k * r_norm) / (4 * np.pi * r_norm)
                        grad_G = (-1j * k * r_norm - 1) * r * np.exp(-1j * k * r_norm) / (4 * np.pi * r_norm**3)
                        dGdn = np.sum(grad_G * normals[i])

                        dJ_dx[i] += lambda_adj[j] * (
                            -dGdn * areas[j] + 1j * k * G * dA[j]
                        )

        return dJ_dx

    def _compute_area_derivative_tri(self, elem, dX: np.ndarray) -> float:
        v0, v1, v2 = elem.vertices
        e1 = v1 - v0
        e2 = v2 - v0
        n = np.cross(e1, e2)
        area = 0.5 * np.linalg.norm(n)

        if area < 1e-12:
            return 0.0

        n_unit = n / np.linalg.norm(n)

        dX0 = dX[elem.index] if len(dX.shape) == 2 else dX
        dX1 = dX[elem.index]
        dX2 = dX[elem.index]

        de1 = dX1 - dX0
        de2 = dX2 - dX0

        dn = np.cross(de1, e2) + np.cross(e1, de2)
        darea = 0.5 * np.sum(n_unit * dn)

        return darea

    def _compute_area_derivative_quad(self, elem, dX: np.ndarray) -> float:
        return 0.0

    def _compute_sensitivity_fd(self, objective: ObjectiveFunction,
                                  dv_name: str, h: float = 1e-6) -> float:
        dv_info = self.param.design_variables[dv_name]

        J0 = objective.compute(self.p_surface, self.v_surface, self.solver)

        original_mesh = self.mesh
        perturbed_mesh = dv_info['update'](h)

        from .bem import BEMSolver
        perturbed_solver = BEMSolver(perturbed_mesh, self.k, self.c, self.rho)

        direction = np.array([1.0, 0.0, 0.0])
        p_incident = np.exp(-1j * self.k * np.dot(perturbed_mesh.centroids, direction))
        p_pert, v_pert = perturbed_solver.solve_dirichlet(p_incident)

        Jh = objective.compute(p_pert, v_pert, perturbed_solver)

        dJ_dx = (Jh - J0) / h

        self.mesh = original_mesh
        self.solver.mesh = original_mesh

        return np.real(dJ_dx)

    def optimize(self, objective: ObjectiveFunction,
                 dv_name: str,
                 n_iterations: int = 10,
                 step_size: float = 0.1,
                 method: str = 'gradient_descent') -> Dict:
        history = {
            'objective': [],
            'design_variable': [],
            'sensitivity': []
        }

        for iter in range(n_iterations):
            sensitivity = self.compute_sensitivity(objective, dv_name)

            J = objective.compute(self.p_surface, self.v_surface, self.solver)
            history['objective'].append(J)
            history['sensitivity'].append(sensitivity)

            print(f"Iteration {iter}: J = {J:.4e}, dJ/dx = {sensitivity:.4e}")

            if method == 'gradient_descent':
                dv_info = self.param.design_variables[dv_name]
                new_mesh = dv_info['update'](-step_size * sensitivity)
                self.mesh = new_mesh
                self.solver.mesh = new_mesh

                direction = np.array([1.0, 0.0, 0.0])
                p_incident = np.exp(-1j * self.k * np.dot(new_mesh.centroids, direction))
                self.solve_primal(p_incident)

        return history


class AdjointFMM:
    def __init__(self, fmm_solver):
        self.fmm = fmm_solver
        self.k = fmm_solver.k

    def matvec_single_layer_adjoint(self, x: np.ndarray) -> np.ndarray:
        return self.fmm.matvec_single_layer(x.conj()).conj()

    def matvec_double_layer_adjoint(self, x: np.ndarray) -> np.ndarray:
        return self.fmm.matvec_double_layer(x.conj()).conj()

    def matvec_combined_adjoint(self, x: np.ndarray, alpha: complex = 1j) -> np.ndarray:
        sl = self.matvec_single_layer_adjoint(x)
        dl = self.matvec_double_layer_adjoint(x)
        return dl + 1j * self.k * sl + alpha.conj() * sl - alpha.conj() / (-1j * self.k) * dl
