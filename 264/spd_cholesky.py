import numpy as np
import warnings


def is_spd_eigenvalue(A):
    A = np.asarray(A, dtype=float)
    if not np.allclose(A, A.T):
        return False
    eigenvalues = np.linalg.eigvalsh(A)
    return np.all(eigenvalues > 0)


def is_spd_cholesky(A):
    A = np.asarray(A, dtype=float)
    if not np.allclose(A, A.T):
        return False
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def modified_cholesky(A, delta_init=1e-10, max_iter=100):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if not np.allclose(A, A.T):
        raise ValueError("Matrix is not symmetric")

    L = np.zeros_like(A)
    diag = np.diag(A).copy()
    min_diag = np.min(diag)

    if min_diag <= 0:
        delta = max(1e-10, -min_diag + 1e-10)
    else:
        delta = delta_init

    for iteration in range(max_iter):
        A_perturbed = A + delta * np.eye(n)
        try:
            L = np.linalg.cholesky(A_perturbed)
            if iteration > 0:
                warnings.warn(
                    f"Cholesky decomposition succeeded after adding perturbation "
                    f"delta={delta:.2e} (iteration {iteration})",
                    UserWarning
                )
            return L, delta
        except np.linalg.LinAlgError:
            delta *= 10

    raise RuntimeError(f"Modified Cholesky failed after {max_iter} iterations")


def cholesky_spd(A, allow_perturbation=True):
    A = np.asarray(A, dtype=float)
    if not np.allclose(A, A.T):
        raise ValueError("Matrix is not symmetric")

    try:
        L = np.linalg.cholesky(A)
        return L, 0.0
    except np.linalg.LinAlgError as e:
        if not allow_perturbation:
            raise ValueError("Matrix is not positive definite (Cholesky decomposition failed)") from e

        eigvals = np.linalg.eigvalsh(A)
        min_eig = np.min(eigvals)

        if min_eig < -1e-8:
            warnings.warn(
                f"Matrix is numerically indefinite (min eigenvalue={min_eig:.2e}). "
                f"Attempting modified Cholesky with diagonal perturbation.",
                UserWarning
            )
        else:
            warnings.warn(
                f"Cholesky failed due to numerical error (min eigenvalue={min_eig:.2e}). "
                f"Applying small diagonal perturbation.",
                UserWarning
            )

        return modified_cholesky(A)


if __name__ == "__main__":
    A_spd = np.array([
        [4, 2, 1],
        [2, 5, 3],
        [1, 3, 6],
    ], dtype=float)

    A_near_singular = np.array([
        [1, 2],
        [2, 4.0000000001],
    ], dtype=float)

    A_almost_spd = np.array([
        [2.0, 1.0],
        [1.0, 0.499999999],
    ], dtype=float)

    A_non_sym = np.array([
        [1, 0],
        [1, 1],
    ], dtype=float)

    print("=== 1. Well-conditioned SPD Matrix ===")
    print(f"Eigenvalue check: {is_spd_eigenvalue(A_spd)}")
    print(f"Cholesky check:   {is_spd_cholesky(A_spd)}")
    L, delta = cholesky_spd(A_spd)
    print(f"Delta used: {delta:.2e}")
    print(f"L =\n{L}")
    print(f"Reconstruction matches: {np.allclose(L @ L.T, A_spd)}")

    print("\n=== 2. Near-singular SPD Matrix (numerical challenge) ===")
    print(f"Min eigenvalue: {np.min(np.linalg.eigvalsh(A_near_singular)):.2e}")
    print(f"Eigenvalue check: {is_spd_eigenvalue(A_near_singular)}")
    print(f"Cholesky check:   {is_spd_cholesky(A_near_singular)}")
    L, delta = cholesky_spd(A_near_singular)
    print(f"Delta used: {delta:.2e}")
    print(f"L =\n{L}")
    print(f"Reconstruction error: {np.linalg.norm(L @ L.T - A_near_singular):.2e}")

    print("\n=== 3. Matrix with tiny negative eigenvalue (numerical error) ===")
    min_eig_actual = np.min(np.linalg.eigvalsh(A_almost_spd))
    print(f"Min eigenvalue: {min_eig_actual:.2e}")
    print(f"Eigenvalue check: {is_spd_eigenvalue(A_almost_spd)}")
    print(f"Cholesky check:   {is_spd_cholesky(A_almost_spd)}")
    L, delta = cholesky_spd(A_almost_spd)
    print(f"Delta used: {delta:.2e}")
    print(f"L shape: {L.shape}")
    print(f"Reconstruction error: {np.linalg.norm(L @ L.T - A_almost_spd):.2e}")

    print("\n=== 4. Non-Symmetric Matrix ===")
    print(f"Eigenvalue check: {is_spd_eigenvalue(A_non_sym)}")
    print(f"Cholesky check:   {is_spd_cholesky(A_non_sym)}")
    try:
        cholesky_spd(A_non_sym)
    except ValueError as e:
        print(f"Expected error: {e}")
