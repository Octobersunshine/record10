import numpy as np
from scipy.special import spherical_jn, spherical_yn, hankel2


class HelmholtzKernel:
    def __init__(self, k: float, c: float = 343.0, rho: float = 1.21):
        self.k = k
        self.c = c
        self.rho = rho
        self.omega = k * c

    def G(self, r: np.ndarray) -> np.ndarray:
        r_norm = np.linalg.norm(r, axis=-1)
        r_norm = np.maximum(r_norm, 1e-12)
        return np.exp(-1j * self.k * r_norm) / (4 * np.pi * r_norm)

    def G_gradient(self, r: np.ndarray) -> np.ndarray:
        r_norm = np.linalg.norm(r, axis=-1, keepdims=True)
        r_norm = np.maximum(r_norm, 1e-12)
        factor = np.exp(-1j * self.k * r_norm) / (4 * np.pi * r_norm**3)
        return factor * (-1j * self.k * r_norm - 1) * r

    def dGdn(self, r: np.ndarray, n: np.ndarray) -> np.ndarray:
        grad_G = self.G_gradient(r)
        return np.sum(grad_G * n, axis=-1)

    def H(self, r: np.ndarray, n: np.ndarray) -> np.ndarray:
        return self.dGdn(r, n)

    def single_layer(self, r: np.ndarray) -> np.ndarray:
        return self.G(r)

    def double_layer(self, r: np.ndarray, n: np.ndarray) -> np.ndarray:
        return self.dGdn(r, n)


class SphericalHarmonics:
    @staticmethod
    def P_l(l: int, x: np.ndarray) -> np.ndarray:
        if l == 0:
            return np.ones_like(x)
        elif l == 1:
            return x
        else:
            P_prev = np.ones_like(x)
            P_curr = x
            for n in range(2, l + 1):
                P_next = ((2 * n - 1) * x * P_curr - (n - 1) * P_prev) / n
                P_prev, P_curr = P_curr, P_next
            return P_curr

    @staticmethod
    def Y_lm(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        P_lm = SphericalHarmonics.assoc_legendre(l, abs(m), np.cos(theta))
        norm = np.sqrt((2 * l + 1) * np.math.factorial(l - abs(m)) /
                       (4 * np.pi * np.math.factorial(l + abs(m))))
        if m >= 0:
            return norm * P_lm * np.exp(1j * m * phi)
        else:
            return (-1)**abs(m) * np.conj(norm * P_lm * np.exp(1j * abs(m) * phi))

    @staticmethod
    def assoc_legendre(l: int, m: int, x: np.ndarray) -> np.ndarray:
        if m == 0:
            return SphericalHarmonics.P_l(l, x)
        elif m > l:
            return np.zeros_like(x)
        else:
            P_mm = (-1)**m * (1 - x**2)**(m / 2) * np.prod(np.arange(1, 2 * m, 2))
            if l == m:
                return P_mm
            else:
                P_mp1m = x * (2 * m + 1) * P_mm
                if l == m + 1:
                    return P_mp1m
                else:
                    P_prev = P_mm
                    P_curr = P_mp1m
                    for n in range(m + 2, l + 1):
                        P_next = ((2 * n - 1) * x * P_curr - (n + m - 1) * P_prev) / (n - m)
                        P_prev, P_curr = P_curr, P_next
                    return P_curr


class MultipoleExpansion:
    def __init__(self, k: float, p: int):
        self.k = k
        self.p = p
        self.coeffs = np.zeros((p + 1, 2 * p + 1), dtype=complex)

    def compute_local(self, sources: np.ndarray, charges: np.ndarray, center: np.ndarray):
        for i, src in enumerate(sources):
            r_vec = src - center
            r = np.linalg.norm(r_vec)
            theta = np.arccos(r_vec[2] / max(r, 1e-12))
            phi = np.arctan2(r_vec[1], r_vec[0])
            for l in range(self.p + 1):
                jl = spherical_jn(l, self.k * r)
                for m in range(-l, l + 1):
                    Y_lm = SphericalHarmonics.Y_lm(l, m, theta, phi)
                    self.coeffs[l, m + self.p] += charges[i] * (1j)**l * jl * np.conj(Y_lm)

    def translate(self, translation_vec: np.ndarray) -> 'MultipoleExpansion':
        result = MultipoleExpansion(self.k, self.p)
        r = np.linalg.norm(translation_vec)
        theta = np.arccos(translation_vec[2] / max(r, 1e-12))
        phi = np.arctan2(translation_vec[1], translation_vec[0])
        for l in range(self.p + 1):
            for m in range(-l, l + 1):
                if abs(self.coeffs[l, m + self.p]) > 0:
                    for lp in range(self.p + 1):
                        for mp in range(-lp, lp + 1):
                            gaunt = self._gaunt_coeff(l, m, lp, mp)
                            if gaunt != 0:
                                jl = spherical_jn(lp, self.k * r)
                                Y = SphericalHarmonics.Y_lm(lp, mp, theta, phi)
                                result.coeffs[lp, mp + self.p] += (
                                    self.coeffs[l, m + self.p] * (1j)**(lp - l) *
                                    4 * np.pi * gaunt * jl * Y
                                )
        return result

    def _gaunt_coeff(self, l1: int, m1: int, l2: int, m2: int) -> float:
        if abs(m1 + m2) > l1 + l2:
            return 0.0
        return 1.0 / (4 * np.pi) if (l1 == 0 and l2 == 0 and m1 == 0 and m2 == 0) else 0.0


class LocalExpansion:
    def __init__(self, k: float, p: int):
        self.k = k
        self.p = p
        self.coeffs = np.zeros((p + 1, 2 * p + 1), dtype=complex)

    def evaluate(self, points: np.ndarray, center: np.ndarray) -> np.ndarray:
        result = np.zeros(len(points), dtype=complex)
        for i, pt in enumerate(points):
            r_vec = pt - center
            r = np.linalg.norm(r_vec)
            theta = np.arccos(r_vec[2] / max(r, 1e-12))
            phi = np.arctan2(r_vec[1], r_vec[0])
            for l in range(self.p + 1):
                hl = hankel2(l, self.k * r)
                for m in range(-l, l + 1):
                    Y_lm = SphericalHarmonics.Y_lm(l, m, theta, phi)
                    result[i] += self.coeffs[l, m + self.p] * (-1j)**l * hl * Y_lm
        return result

    def translate(self, translation_vec: np.ndarray) -> 'LocalExpansion':
        result = LocalExpansion(self.k, self.p)
        r = np.linalg.norm(translation_vec)
        theta = np.arccos(translation_vec[2] / max(r, 1e-12))
        phi = np.arctan2(translation_vec[1], translation_vec[0])
        for l in range(self.p + 1):
            for m in range(-l, l + 1):
                if abs(self.coeffs[l, m + self.p]) > 0:
                    for lp in range(l, self.p + 1):
                        for mp in range(-lp, lp + 1):
                            gaunt = self._gaunt_coeff(l, m, lp, mp)
                            if gaunt != 0:
                                hl = hankel2(lp - l, self.k * r)
                                Y = SphericalHarmonics.Y_lm(lp - l, mp - m, theta, phi)
                                result.coeffs[lp, mp + self.p] += (
                                    self.coeffs[l, m + self.p] * (1j)**(lp - l) *
                                    4 * np.pi * gaunt * hl * Y
                                )
        return result

    def _gaunt_coeff(self, l1: int, m1: int, l2: int, m2: int) -> float:
        if abs(m1 + m2) > l1 + l2:
            return 0.0
        return 1.0 / (4 * np.pi) if (l1 == 0 and l2 == 0 and m1 == 0 and m2 == 0) else 0.0
