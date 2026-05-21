import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class Material:
    sigma_t: float
    sigma_s: float
    nu_sigma_f: Optional[float] = None


@dataclass
class Mesh:
    x_min: float
    x_max: float
    n_cells: int

    def __post_init__(self):
        self.dx = (self.x_max - self.x_min) / self.n_cells
        self.x_edges = np.linspace(self.x_min, self.x_max, self.n_cells + 1)
        self.x_centers = (self.x_edges[:-1] + self.x_edges[1:]) / 2


class SNQuadrature:
    def __init__(self, N: int):
        self.N = N
        self.mus, self.weights = self._generate_quadrature(N)

    def _generate_quadrature(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        if N == 2:
            mus = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
            weights = np.array([1.0, 1.0])
        elif N == 4:
            mus = np.array([-0.861136, -0.339981, 0.339981, 0.861136])
            weights = np.array([0.347855, 0.652145, 0.652145, 0.347855])
        elif N == 8:
            mus = np.array([-0.960290, -0.796666, -0.525532, -0.183435,
                            0.183435, 0.525532, 0.796666, 0.960290])
            weights = np.array([0.101229, 0.222381, 0.313707, 0.362684,
                                0.362684, 0.313707, 0.222381, 0.101229])
        elif N == 16:
            mus = np.array([-0.989401, -0.944575, -0.865631, -0.755404,
                            -0.617876, -0.458017, -0.281604, -0.0950125,
                            0.0950125, 0.281604, 0.458017, 0.617876,
                            0.755404, 0.865631, 0.944575, 0.989401])
            weights = np.array([0.0271525, 0.0622535, 0.0951585, 0.124629,
                                0.149596, 0.169157, 0.182603, 0.189451,
                                0.189451, 0.182603, 0.169157, 0.149596,
                                0.124629, 0.0951585, 0.0622535, 0.0271525])
        else:
            raise ValueError(f"Unsupported SN order: {N}. Use 2, 4, 8, or 16.")

        return mus, weights


class SNSolver:
    def __init__(self, mesh: Mesh, quadrature: SNQuadrature,
                 materials: list, bc_left: str = 'vacuum', bc_right: str = 'vacuum',
                 fix_negative_flux: bool = True, negative_flux_tol: float = 1e-10,
                 spatial_method: str = 'wdd', alpha_wdd: Optional[float] = None,
                 negative_flux_fix_method: str = 'conservative'):
        self.mesh = mesh
        self.quad = quadrature
        self.materials = materials
        self.bc_left = bc_left
        self.bc_right = bc_right
        self.fix_negative_flux = fix_negative_flux
        self.negative_flux_tol = negative_flux_tol
        self.negative_flux_count = 0
        self.max_negative_flux = 0.0
        self.spatial_method = spatial_method.lower()
        self.alpha_wdd = alpha_wdd
        self.negative_flux_fix_method = negative_flux_fix_method.lower()

        valid_methods = ['dd', 'wdd', 'step', 'sc', 'edd']
        if self.spatial_method not in valid_methods:
            raise ValueError(f"Unknown spatial method: {spatial_method}. Use one of {valid_methods}.")
        
        valid_fix_methods = ['simple', 'conservative', 'redistribute', 'scale']
        if self.negative_flux_fix_method not in valid_fix_methods:
            raise ValueError(f"Unknown fix method: {negative_flux_fix_method}. Use one of {valid_fix_methods}.")

        self.n_cells = mesh.n_cells
        self.n_angles = len(quadrature.mus)

        self.psi = np.zeros((self.n_angles, self.n_cells + 1))
        self.phi = np.zeros(self.n_cells)

        self.psi_center = np.zeros((self.n_angles, self.n_cells))

    def _get_sigma_t(self, i: int) -> float:
        return self.materials[i].sigma_t

    def _get_sigma_s(self, i: int) -> float:
        return self.materials[i].sigma_s

    def _compute_scattering_source(self) -> np.ndarray:
        Q = np.zeros((self.n_angles, self.n_cells))
        for i in range(self.n_cells):
            sigma_s = self._get_sigma_s(i)
            Q[:, i] = sigma_s * self.phi[i]
        return Q

    def _compute_scalar_flux(self):
        if self.spatial_method in ['dd', 'step']:
            self._compute_scalar_flux_edge_average()
        elif self.spatial_method == 'wdd':
            self._compute_scalar_flux_wdd()
        elif self.spatial_method == 'sc':
            self._compute_scalar_flux_sc()

    def _compute_scalar_flux_edge_average(self):
        for i in range(self.n_cells):
            psi_edge_avg = 0.5 * (self.psi[:, i] + self.psi[:, i+1])
            self.phi[i] = 0.5 * np.sum(self.quad.weights * psi_edge_avg)

    def _compute_scalar_flux_wdd(self):
        dx = self.mesh.dx
        for i in range(self.n_cells):
            sigma_t = self._get_sigma_t(i)
            psi_center = np.zeros(self.n_angles)

            for n in range(self.n_angles):
                mu = self.quad.mus[n]
                abs_mu = abs(mu)
                tau = sigma_t * dx / abs_mu

                if self.alpha_wdd is not None:
                    alpha = self.alpha_wdd
                else:
                    alpha = max(0.5, min(1.0, 1.0 / tau))
                beta = 1.0 - alpha

                if mu > 0:
                    psi_center[n] = alpha * self.psi[n, i] + beta * self.psi[n, i+1]
                else:
                    psi_center[n] = alpha * self.psi[n, i+1] + beta * self.psi[n, i]

            self.psi_center[:, i] = psi_center
            self.phi[i] = 0.5 * np.sum(self.quad.weights * psi_center)

    def _compute_scalar_flux_sc(self):
        self.phi[:] = 0.0
        for i in range(self.n_cells):
            self.phi[i] = 0.5 * np.sum(self.quad.weights * self.psi_center[:, i])

    def _sweep(self, Q: np.ndarray):
        dx = self.mesh.dx

        if self.spatial_method == 'dd':
            self._sweep_dd(Q, dx)
        elif self.spatial_method == 'wdd':
            self._sweep_wdd(Q, dx)
        elif self.spatial_method == 'step':
            self._sweep_step(Q, dx)
        elif self.spatial_method == 'sc':
            self._sweep_sc(Q, dx)
        elif self.spatial_method == 'edd':
            self._sweep_edd(Q, dx)

    def _sweep_dd(self, Q: np.ndarray, dx: float):
        for n in range(self.n_angles):
            mu = self.quad.mus[n]

            if mu > 0:
                if self.bc_left == 'vacuum':
                    self.psi[n, 0] = 0.0
                elif self.bc_left == 'reflective':
                    self.psi[n, 0] = self.psi[self.n_angles - 1 - n, 0]

                for i in range(self.n_cells):
                    sigma_t = self._get_sigma_t(i)
                    coeff = sigma_t + 2 * abs(mu) / dx
                    rhs = Q[n, i] + 2 * mu / dx * self.psi[n, i]
                    self.psi[n, i+1] = rhs / coeff

            else:
                if self.bc_right == 'vacuum':
                    self.psi[n, -1] = 0.0
                elif self.bc_right == 'reflective':
                    self.psi[n, -1] = self.psi[self.n_angles - 1 - n, -1]

                for i in range(self.n_cells - 1, -1, -1):
                    sigma_t = self._get_sigma_t(i)
                    coeff = sigma_t + 2 * abs(mu) / dx
                    rhs = Q[n, i] - 2 * mu / dx * self.psi[n, i+1]
                    self.psi[n, i] = rhs / coeff

    def _sweep_wdd(self, Q: np.ndarray, dx: float):
        for n in range(self.n_angles):
            mu = self.quad.mus[n]
            abs_mu = abs(mu)

            if mu > 0:
                if self.bc_left == 'vacuum':
                    self.psi[n, 0] = 0.0
                elif self.bc_left == 'reflective':
                    self.psi[n, 0] = self.psi[self.n_angles - 1 - n, 0]

                for i in range(self.n_cells):
                    sigma_t = self._get_sigma_t(i)
                    tau = sigma_t * dx / abs_mu

                    if self.alpha_wdd is not None:
                        alpha = self.alpha_wdd
                    else:
                        alpha = max(0.5, min(1.0, 1.0 / tau))
                    beta = 1.0 - alpha

                    coeff = sigma_t * beta + mu / dx
                    rhs = Q[n, i] + mu / dx * self.psi[n, i] - sigma_t * alpha * self.psi[n, i]
                    self.psi[n, i+1] = rhs / coeff

            else:
                if self.bc_right == 'vacuum':
                    self.psi[n, -1] = 0.0
                elif self.bc_right == 'reflective':
                    self.psi[n, -1] = self.psi[self.n_angles - 1 - n, -1]

                for i in range(self.n_cells - 1, -1, -1):
                    sigma_t = self._get_sigma_t(i)
                    tau = sigma_t * dx / abs_mu

                    if self.alpha_wdd is not None:
                        alpha = self.alpha_wdd
                    else:
                        alpha = max(0.5, min(1.0, 1.0 / tau))
                    beta = 1.0 - alpha

                    coeff = sigma_t * beta - mu / dx
                    rhs = Q[n, i] - mu / dx * self.psi[n, i+1] - sigma_t * alpha * self.psi[n, i+1]
                    self.psi[n, i] = rhs / coeff

    def _sweep_step(self, Q: np.ndarray, dx: float):
        for n in range(self.n_angles):
            mu = self.quad.mus[n]
            abs_mu = abs(mu)

            if mu > 0:
                if self.bc_left == 'vacuum':
                    self.psi[n, 0] = 0.0
                elif self.bc_left == 'reflective':
                    self.psi[n, 0] = self.psi[self.n_angles - 1 - n, 0]

                for i in range(self.n_cells):
                    sigma_t = self._get_sigma_t(i)
                    tau = sigma_t * dx / abs_mu
                    exp_tau = np.exp(-tau)

                    psi_in = self.psi[n, i]
                    Q_avg = Q[n, i] / sigma_t if sigma_t > 0 else 0.0

                    self.psi[n, i+1] = psi_in * exp_tau + Q_avg * (1 - exp_tau)

            else:
                if self.bc_right == 'vacuum':
                    self.psi[n, -1] = 0.0
                elif self.bc_right == 'reflective':
                    self.psi[n, -1] = self.psi[self.n_angles - 1 - n, -1]

                for i in range(self.n_cells - 1, -1, -1):
                    sigma_t = self._get_sigma_t(i)
                    tau = sigma_t * dx / abs_mu
                    exp_tau = np.exp(-tau)

                    psi_in = self.psi[n, i+1]
                    Q_avg = Q[n, i] / sigma_t if sigma_t > 0 else 0.0

                    self.psi[n, i] = psi_in * exp_tau + Q_avg * (1 - exp_tau)

    def _sweep_sc(self, Q: np.ndarray, dx: float):
        for n in range(self.n_angles):
            mu = self.quad.mus[n]
            abs_mu = abs(mu)

            if mu > 0:
                if self.bc_left == 'vacuum':
                    self.psi[n, 0] = 0.0
                elif self.bc_left == 'reflective':
                    self.psi[n, 0] = self.psi[self.n_angles - 1 - n, 0]

                for i in range(self.n_cells):
                    sigma_t = self._get_sigma_t(i)
                    tau = sigma_t * dx / abs_mu

                    if tau < 1e-10:
                        self.psi[n, i+1] = self.psi[n, i]
                        self.psi_center[n, i] = self.psi[n, i]
                        continue

                    exp_tau = np.exp(-tau)
                    psi_in = self.psi[n, i]
                    Q_src = Q[n, i]

                    self.psi[n, i+1] = psi_in * exp_tau + Q_src / sigma_t * (1 - exp_tau)

                    psi_avg = (psi_in - self.psi[n, i+1]) / tau + Q_src / sigma_t * (1 - (1 - exp_tau) / tau)
                    self.psi_center[n, i] = psi_avg

            else:
                if self.bc_right == 'vacuum':
                    self.psi[n, -1] = 0.0
                elif self.bc_right == 'reflective':
                    self.psi[n, -1] = self.psi[self.n_angles - 1 - n, -1]

                for i in range(self.n_cells - 1, -1, -1):
                    sigma_t = self._get_sigma_t(i)
                    tau = sigma_t * dx / abs_mu

                    if tau < 1e-10:
                        self.psi[n, i] = self.psi[n, i+1]
                        self.psi_center[n, i] = self.psi[n, i+1]
                        continue

                    exp_tau = np.exp(-tau)
                    psi_in = self.psi[n, i+1]
                    Q_src = Q[n, i]

                    self.psi[n, i] = psi_in * exp_tau + Q_src / sigma_t * (1 - exp_tau)

                    psi_avg = (psi_in - self.psi[n, i]) / tau + Q_src / sigma_t * (1 - (1 - exp_tau) / tau)
                    self.psi_center[n, i] = psi_avg

    def _sweep_edd(self, Q: np.ndarray, dx: float):
        for n in range(self.n_angles):
            mu = self.quad.mus[n]
            abs_mu = abs(mu)

            if mu > 0:
                if self.bc_left == 'vacuum':
                    self.psi[n, 0] = 0.0
                elif self.bc_left == 'reflective':
                    self.psi[n, 0] = self.psi[self.n_angles - 1 - n, 0]

                for i in range(self.n_cells):
                    sigma_t = self._get_sigma_t(i)
                    tau = sigma_t * dx / abs_mu

                    if tau < 1e-10:
                        self.psi[n, i+1] = self.psi[n, i]
                        self.psi_center[n, i] = self.psi[n, i]
                        continue

                    exp_tau = np.exp(-tau)
                    psi_in = self.psi[n, i]
                    Q_src = Q[n, i]

                    psi_out = psi_in * exp_tau + Q_src / sigma_t * (1 - exp_tau)
                    self.psi[n, i+1] = psi_out

                    denom = 1.0 - exp_tau
                    if denom > 1e-10:
                        alpha = (1 - exp_tau - tau * exp_tau) / (tau * denom)
                    else:
                        alpha = 0.5
                    self.psi_center[n, i] = alpha * psi_in + (1 - alpha) * psi_out

            else:
                if self.bc_right == 'vacuum':
                    self.psi[n, -1] = 0.0
                elif self.bc_right == 'reflective':
                    self.psi[n, -1] = self.psi[self.n_angles - 1 - n, -1]

                for i in range(self.n_cells - 1, -1, -1):
                    sigma_t = self._get_sigma_t(i)
                    tau = sigma_t * dx / abs_mu

                    if tau < 1e-10:
                        self.psi[n, i] = self.psi[n, i+1]
                        self.psi_center[n, i] = self.psi[n, i+1]
                        continue

                    exp_tau = np.exp(-tau)
                    psi_in = self.psi[n, i+1]
                    Q_src = Q[n, i]

                    psi_out = psi_in * exp_tau + Q_src / sigma_t * (1 - exp_tau)
                    self.psi[n, i] = psi_out

                    denom = 1.0 - exp_tau
                    if denom > 1e-10:
                        alpha = (1 - exp_tau - tau * exp_tau) / (tau * denom)
                    else:
                        alpha = 0.5
                    self.psi_center[n, i] = alpha * psi_in + (1 - alpha) * psi_out

    def _fix_negative_flux(self):
        if not self.fix_negative_flux:
            return

        if self.negative_flux_fix_method == 'simple':
            self._fix_negative_flux_simple()
        elif self.negative_flux_fix_method == 'conservative':
            self._fix_negative_flux_conservative()
        elif self.negative_flux_fix_method == 'redistribute':
            self._fix_negative_flux_redistribute()
        elif self.negative_flux_fix_method == 'scale':
            self._fix_negative_flux_scale()

    def _fix_negative_flux_simple(self):
        for n in range(self.n_angles):
            for i in range(self.n_cells + 1):
                if self.psi[n, i] < self.negative_flux_tol:
                    self.negative_flux_count += 1
                    self.max_negative_flux = min(self.max_negative_flux, self.psi[n, i])
                    self.psi[n, i] = self.negative_flux_tol

        for n in range(self.n_angles):
            for i in range(self.n_cells):
                if self.psi_center[n, i] < self.negative_flux_tol:
                    self.psi_center[n, i] = self.negative_flux_tol

    def _fix_negative_flux_conservative(self):
        for i in range(self.n_cells + 1):
            neg_sum = 0.0
            pos_sum = 0.0

            for n in range(self.n_angles):
                if self.psi[n, i] < self.negative_flux_tol:
                    self.negative_flux_count += 1
                    self.max_negative_flux = min(self.max_negative_flux, self.psi[n, i])
                    neg_sum += self.negative_flux_tol - self.psi[n, i]
                    self.psi[n, i] = self.negative_flux_tol
                else:
                    pos_sum += self.psi[n, i] * self.quad.weights[n]

            if neg_sum > 0 and pos_sum > 0:
                scale_factor = 1.0 - neg_sum * 2.0 / pos_sum
                if scale_factor > 0:
                    for n in range(self.n_angles):
                        if self.psi[n, i] > self.negative_flux_tol:
                            self.psi[n, i] *= scale_factor

        for i in range(self.n_cells):
            for n in range(self.n_angles):
                if self.psi_center[n, i] < self.negative_flux_tol:
                    self.psi_center[n, i] = self.negative_flux_tol

    def _fix_negative_flux_redistribute(self):
        for i in range(self.n_cells + 1):
            neg_amount = 0.0
            pos_weights = 0.0

            for n in range(self.n_angles):
                if self.psi[n, i] < self.negative_flux_tol:
                    self.negative_flux_count += 1
                    self.max_negative_flux = min(self.max_negative_flux, self.psi[n, i])
                    neg_amount += self.negative_flux_tol - self.psi[n, i]
                    self.psi[n, i] = self.negative_flux_tol
                else:
                    pos_weights += self.quad.weights[n]

            if neg_amount > 0 and pos_weights > 0:
                for n in range(self.n_angles):
                    if self.psi[n, i] > self.negative_flux_tol:
                        self.psi[n, i] -= neg_amount * self.quad.weights[n] / pos_weights
                        if self.psi[n, i] < self.negative_flux_tol:
                            self.psi[n, i] = self.negative_flux_tol

        for i in range(self.n_cells):
            for n in range(self.n_angles):
                if self.psi_center[n, i] < self.negative_flux_tol:
                    self.psi_center[n, i] = self.negative_flux_tol

    def _fix_negative_flux_scale(self):
        for i in range(self.n_cells + 1):
            has_negative = False
            total_weighted = 0.0

            for n in range(self.n_angles):
                if self.psi[n, i] < self.negative_flux_tol:
                    has_negative = True
                    self.negative_flux_count += 1
                    self.max_negative_flux = min(self.max_negative_flux, self.psi[n, i])
                total_weighted += self.quad.weights[n] * max(self.psi[n, i], self.negative_flux_tol)

            if has_negative and total_weighted > 0:
                current_sum = 0.5 * np.sum(self.quad.weights * self.psi[:, i])
                target_sum = max(current_sum, self.negative_flux_tol * self.n_angles)

                scale = target_sum * 2.0 / total_weighted if total_weighted > 0 else 1.0
                for n in range(self.n_angles):
                    self.psi[n, i] = scale * max(self.psi[n, i], self.negative_flux_tol)

        for i in range(self.n_cells):
            for n in range(self.n_angles):
                if self.psi_center[n, i] < self.negative_flux_tol:
                    self.psi_center[n, i] = self.negative_flux_tol

    def solve(self, source: np.ndarray, max_iter: int = 1000, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        self.phi[:] = 0.0
        self.psi[:] = 0.0
        self.psi_center[:] = 0.0

        for it in range(max_iter):
            phi_old = self.phi.copy()

            Q = self._compute_scattering_source()
            for n in range(self.n_angles):
                Q[n, :] += source

            self._sweep(Q)
            self._fix_negative_flux()
            self._compute_scalar_flux()

            res = np.max(np.abs(self.phi - phi_old)) / (np.max(np.abs(self.phi)) + 1e-10)

            if res < tol:
                break

        return self.phi, self.psi


def solve_sn_1d(x_min: float, x_max: float, n_cells: int, sn_order: int,
                sigma_t: float, sigma_s: float, source: np.ndarray,
                bc_left: str = 'vacuum', bc_right: str = 'vacuum',
                spatial_method: str = 'wdd', tol: float = 1e-6,
                fix_negative_flux: bool = True,
                negative_flux_fix_method: str = 'conservative',
                max_iter: int = 1000) -> Tuple[Mesh, np.ndarray, np.ndarray]:
    mesh = Mesh(x_min, x_max, n_cells)
    quad = SNQuadrature(sn_order)
    materials = [Material(sigma_t, sigma_s) for _ in range(n_cells)]

    solver = SNSolver(mesh, quad, materials, bc_left, bc_right,
                      fix_negative_flux=fix_negative_flux,
                      negative_flux_fix_method=negative_flux_fix_method,
                      spatial_method=spatial_method)
    phi, psi = solver.solve(source, max_iter=max_iter, tol=tol)

    return mesh, phi, psi
