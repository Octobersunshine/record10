import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import minimize


def is_valid_density_matrix(rho, tol=1e-6):
    if not np.allclose(rho, rho.conj().T, atol=tol):
        return False
    eigvals = np.linalg.eigvalsh(rho)
    if np.any(eigvals < -tol):
        return False
    if not np.isclose(np.trace(rho), 1.0, atol=tol):
        return False
    return True


def get_pauli_matrices():
    sigma0 = np.array([[1, 0], [0, 1]], dtype=complex)
    sigma1 = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma3 = np.array([[1, 0], [0, -1]], dtype=complex)
    return [sigma0, sigma1, sigma2, sigma3]


def tensor_product(ops):
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def generate_measurement_bases(n_qubits):
    pauli = get_pauli_matrices()
    bases = []
    single_bases = [1, 2, 3]
    
    def generate(idx, current):
        if idx == n_qubits:
            bases.append(current.copy())
            return
        for b in single_bases:
            current.append(b)
            generate(idx + 1, current)
            current.pop()
    
    generate(0, [])
    return bases


def get_projector(basis_idx, outcome):
    pauli = get_pauli_matrices()
    sigma = pauli[basis_idx]
    eigvals, eigvecs = np.linalg.eigh(sigma)
    if outcome == 0:
        return np.outer(eigvecs[:, 0], eigvecs[:, 0].conj())
    else:
        return np.outer(eigvecs[:, 1], eigvecs[:, 1].conj())


def get_measurement_projector(basis):
    n_qubits = len(basis)
    projectors = []
    for outcome in range(2**n_qubits):
        outcome_bits = [(outcome >> i) & 1 for i in range(n_qubits)]
        single_projectors = [get_projector(basis[i], outcome_bits[i]) for i in range(n_qubits)]
        P = tensor_product(single_projectors)
        projectors.append(P)
    return projectors


def fidelity(rho, sigma):
    sqrt_rho = sqrtm(rho)
    product = sqrt_rho @ sigma @ sqrt_rho
    sqrt_product = sqrtm(product)
    return np.real(np.trace(sqrt_product))**2


def reconstruct_mle(measurement_data, n_qubits, max_iter=5000, tol=1e-10):
    dim = 2**n_qubits
    bases = generate_measurement_bases(n_qubits)
    
    measurement_projectors = {}
    for basis in bases:
        key = tuple(basis)
        measurement_projectors[key] = get_measurement_projector(basis)
    
    def nll(params):
        L = np.zeros((dim, dim), dtype=complex)
        idx = 0
        for i in range(dim):
            for j in range(i + 1):
                if i == j:
                    L[i, j] = params[idx]
                    idx += 1
                else:
                    L[i, j] = params[idx] + 1j * params[idx + 1]
                    idx += 2
        
        rho = L @ L.conj().T
        rho /= np.trace(rho)
        
        nll_val = 0.0
        for basis_idx, basis in enumerate(bases):
            key = tuple(basis)
            if key not in measurement_data:
                continue
            counts = measurement_data[key]
            projectors = measurement_projectors[key]
            for outcome, P in enumerate(projectors):
                prob = np.real(np.trace(rho @ P))
                prob = max(prob, 1e-15)
                nll_val -= counts[outcome] * np.log(prob)
        
        return nll_val
    
    def initial_params():
        rho0 = np.eye(dim, dtype=complex) / dim
        L = np.linalg.cholesky(rho0)
        params = []
        for i in range(dim):
            for j in range(i + 1):
                if i == j:
                    params.append(np.real(L[i, j]))
                else:
                    params.append(np.real(L[i, j]))
                    params.append(np.imag(L[i, j]))
        return np.array(params)
    
    x0 = initial_params()
    result = minimize(nll, x0, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10})
    
    L = np.zeros((dim, dim), dtype=complex)
    idx = 0
    for i in range(dim):
        for j in range(i + 1):
            if i == j:
                L[i, j] = result.x[idx]
                idx += 1
            else:
                L[i, j] = result.x[idx] + 1j * result.x[idx + 1]
                idx += 2
    
    rho = L @ L.conj().T
    rho /= np.trace(rho)
    
    rho = (rho + rho.conj().T) / 2
    
    return rho


def simulate_measurements(true_rho, n_qubits, shots_per_basis=1000):
    bases = generate_measurement_bases(n_qubits)
    measurement_data = {}
    
    for basis in bases:
        key = tuple(basis)
        projectors = get_measurement_projector(basis)
        probs = [np.real(np.trace(true_rho @ P)) for P in projectors]
        probs = np.array(probs)
        probs /= probs.sum()
        counts = np.random.multinomial(shots_per_basis, probs)
        measurement_data[key] = counts.tolist()
    
    return measurement_data


def run_example():
    np.random.seed(42)
    
    n_qubits = 1
    print(f"=== {n_qubits}-qubit Quantum State Tomography ===")
    
    psi = np.array([1, 1]) / np.sqrt(2)
    true_rho = np.outer(psi, psi.conj())
    print("\nTrue density matrix:")
    print(true_rho)
    
    measurement_data = simulate_measurements(true_rho, n_qubits, shots_per_basis=5000)
    print("\nMeasurement data counts:")
    for basis, counts in measurement_data.items():
        basis_names = ['X', 'Y', 'Z']
        basis_str = ','.join([basis_names[b-1] for b in basis])
        print(f"  Basis [{basis_str}]: {counts}")
    
    reconstructed_rho = reconstruct_mle(measurement_data, n_qubits)
    print("\nReconstructed density matrix:")
    print(reconstructed_rho)
    
    fid = fidelity(reconstructed_rho, true_rho)
    print(f"\nFidelity: {fid:.6f}")
    
    print(f"\nValid density matrix: {is_valid_density_matrix(reconstructed_rho)}")


def run_two_qubit_example():
    np.random.seed(42)
    
    n_qubits = 2
    print(f"\n\n=== {n_qubits}-qubit Quantum State Tomography ===")
    
    psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
    true_rho = np.outer(psi, psi.conj())
    print("\nTrue density matrix (Bell state):")
    print(true_rho)
    
    measurement_data = simulate_measurements(true_rho, n_qubits, shots_per_basis=2000)
    
    reconstructed_rho = reconstruct_mle(measurement_data, n_qubits)
    print("\nReconstructed density matrix:")
    print(reconstructed_rho)
    
    fid = fidelity(reconstructed_rho, true_rho)
    print(f"\nFidelity: {fid:.6f}")
    
    print(f"\nValid density matrix: {is_valid_density_matrix(reconstructed_rho)}")


def generate_random_bases(n_qubits, n_bases, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    bases = []
    while len(bases) < n_bases:
        basis = tuple(rng.randint(1, 4, size=n_qubits))
        if basis not in bases:
            bases.append(basis)
    return bases


def simulate_measurements_with_bases(true_rho, n_qubits, bases, shots_per_basis=1000, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    measurement_data = {}
    
    for basis in bases:
        key = tuple(basis)
        projectors = get_measurement_projector(basis)
        probs = [np.real(np.trace(true_rho @ P)) for P in projectors]
        probs = np.array(probs)
        probs = np.maximum(probs, 0)
        probs /= probs.sum()
        counts = rng.multinomial(shots_per_basis, probs)
        measurement_data[key] = counts.tolist()
    
    return measurement_data


def generate_low_rank_state(n_qubits, rank, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    dim = 2**n_qubits
    
    eigvals = np.zeros(dim)
    eigvals[:rank] = rng.rand(rank)
    eigvals = eigvals / eigvals.sum()
    
    random_matrix = rng.randn(dim, dim) + 1j * rng.randn(dim, dim)
    Q, _ = np.linalg.qr(random_matrix)
    
    rho = Q @ np.diag(eigvals) @ Q.conj().T
    rho = (rho + rho.conj().T) / 2
    
    return rho


def generate_pure_product_state(n_qubits, theta=None, phi=None, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    if theta is None:
        theta = rng.rand(n_qubits) * np.pi
    if phi is None:
        phi = rng.rand(n_qubits) * 2 * np.pi
    
    single_qubit_states = []
    for i in range(n_qubits):
        psi = np.array([np.cos(theta[i]/2), np.sin(theta[i]/2) * np.exp(1j * phi[i])])
        single_qubit_states.append(psi)
    
    psi = single_qubit_states[0]
    for i in range(1, n_qubits):
        psi = np.kron(psi, single_qubit_states[i])
    
    rho = np.outer(psi, psi.conj())
    return rho


def reconstruct_cst(measurement_data, n_qubits, lambda_reg=0.1, max_iter=1000, tol=1e-6):
    dim = 2**n_qubits
    bases = list(measurement_data.keys())
    
    measurement_projectors = {}
    for basis in bases:
        measurement_projectors[tuple(basis)] = get_measurement_projector(basis)
    
    def objective(params):
        X = np.zeros((dim, dim), dtype=complex)
        idx = 0
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    X[i, j] = params[idx]
                    idx += 1
                elif j > i:
                    X[i, j] = params[idx] + 1j * params[idx + 1]
                    idx += 2
                else:
                    X[i, j] = np.conj(X[j, i])
        
        rho = (X + X.conj().T) / 2
        
        nuclear_norm = np.sum(np.abs(np.linalg.svd(rho, compute_uv=False)))
        
        data_fit = 0.0
        for basis in bases:
            key = tuple(basis)
            counts = measurement_data[key]
            total_counts = sum(counts)
            projectors = measurement_projectors[key]
            for outcome, P in enumerate(projectors):
                prob = np.real(np.trace(rho @ P))
                prob = np.clip(prob, 1e-15, 1 - 1e-15)
                freq = counts[outcome] / total_counts
                data_fit += total_counts * (freq - prob)**2
        
        return data_fit + lambda_reg * nuclear_norm
    
    def constraint_trace(params):
        X = np.zeros((dim, dim), dtype=complex)
        idx = 0
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    X[i, j] = params[idx]
                    idx += 1
                elif j > i:
                    X[i, j] = params[idx] + 1j * params[idx + 1]
                    idx += 2
                else:
                    X[i, j] = np.conj(X[j, i])
        return np.real(np.trace(X)) - 1.0
    
    def initial_params():
        rho0 = np.eye(dim, dtype=complex) / dim
        params = []
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    params.append(np.real(rho0[i, j]))
                elif j > i:
                    params.append(np.real(rho0[i, j]))
                    params.append(np.imag(rho0[i, j]))
        return np.array(params)
    
    x0 = initial_params()
    
    constraints = [{'type': 'eq', 'fun': constraint_trace}]
    
    result = minimize(objective, x0, method='SLSQP', constraints=constraints,
                      options={'maxiter': max_iter, 'ftol': tol})
    
    X = np.zeros((dim, dim), dtype=complex)
    idx = 0
    for i in range(dim):
        for j in range(dim):
            if i == j:
                X[i, j] = result.x[idx]
                idx += 1
            elif j > i:
                X[i, j] = result.x[idx] + 1j * result.x[idx + 1]
                idx += 2
            else:
                X[i, j] = np.conj(X[j, i])
    
    rho = (X + X.conj().T) / 2
    
    eigvals = np.linalg.eigvalsh(rho)
    if np.min(eigvals) < 0:
        eigvals = np.maximum(eigvals, 0)
        eigvals = eigvals / eigvals.sum()
        _, eigvecs = np.linalg.eigh(rho)
        rho = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    
    rho = rho / np.trace(rho)
    
    return rho


def reconstruct_cst_fast(measurement_data, n_qubits, lambda_reg=0.01, max_iter=500, tol=1e-6):
    dim = 2**n_qubits
    bases = list(measurement_data.keys())
    
    measurement_projectors = {}
    for basis in bases:
        measurement_projectors[tuple(basis)] = get_measurement_projector(basis)
    
    rho = np.eye(dim, dtype=complex) / dim
    mu = 1.0
    
    for iteration in range(max_iter):
        rho_prev = rho.copy()
        
        grad = np.zeros_like(rho)
        for basis in bases:
            key = tuple(basis)
            counts = measurement_data[key]
            total_counts = sum(counts)
            projectors = measurement_projectors[key]
            for outcome, P in enumerate(projectors):
                prob = np.real(np.trace(rho @ P))
                prob = np.clip(prob, 1e-15, 1 - 1e-15)
                freq = counts[outcome] / total_counts
                grad += 2 * total_counts * (prob - freq) * P
        
        U, s, Vh = np.linalg.svd(rho - mu * grad, full_matrices=False)
        s = np.maximum(s - mu * lambda_reg, 0)
        rho = U @ np.diag(s) @ Vh
        
        rho = (rho + rho.conj().T) / 2
        rho = rho / np.trace(rho)
        
        eigvals = np.linalg.eigvalsh(rho)
        if np.min(eigvals) < 0:
            eigvals = np.maximum(eigvals, 0)
            eigvals = eigvals / eigvals.sum()
            _, eigvecs = np.linalg.eigh(rho)
            rho = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        
        delta = np.linalg.norm(rho - rho_prev)
        if delta < tol:
            break
    
    return rho


def run_cst_example():
    np.random.seed(42)
    
    n_qubits = 3
    rank = 2
    print(f"=== {n_qubits}-qubit Compressed Sensing Tomography (rank-{rank}) ===")
    
    true_rho = generate_low_rank_state(n_qubits, rank=rank, rng=np.random.RandomState(42))
    print(f"\nTrue rank: {np.linalg.matrix_rank(true_rho)}")
    print(f"True eigenvalues: {np.sort(np.linalg.eigvalsh(true_rho))[::-1][:5]}")
    
    n_bases_full = 3**n_qubits
    n_bases_cst = int(4 * rank * n_qubits)
    print(f"\nFull tomography bases: {n_bases_full}")
    print(f"Compressed sensing bases: {n_bases_cst}")
    print(f"Reduction factor: {n_bases_full / n_bases_cst:.1f}x")
    
    shots_per_basis = 2000
    
    rng = np.random.RandomState(42)
    random_bases = generate_random_bases(n_qubits, n_bases_cst, rng=rng)
    
    measurement_data_cst = simulate_measurements_with_bases(
        true_rho, n_qubits, random_bases, shots_per_basis=shots_per_basis, rng=rng
    )
    
    lambda_reg = 0.001
    reconstructed_cst = reconstruct_cst_fast(
        measurement_data_cst, n_qubits, lambda_reg=lambda_reg
    )
    
    fid_cst = fidelity(reconstructed_cst, true_rho)
    print(f"\nCST Fidelity: {fid_cst:.6f}")
    print(f"Reconstructed rank: {np.linalg.matrix_rank(reconstructed_cst, tol=1e-3)}")
    print(f"Valid density matrix: {is_valid_density_matrix(reconstructed_cst)}")
    
    all_bases = generate_measurement_bases(n_qubits)
    measurement_data_full = simulate_measurements_with_bases(
        true_rho, n_qubits, all_bases, shots_per_basis=shots_per_basis, rng=rng
    )
    
    reconstructed_full = reconstruct_mle(measurement_data_full, n_qubits)
    fid_full = fidelity(reconstructed_full, true_rho)
    print(f"\nFull MLE Fidelity: {fid_full:.6f}")
    print(f"Valid density matrix: {is_valid_density_matrix(reconstructed_full)}")
    
    print(f"\n=== Summary ===")
    print(f"CST uses {n_bases_cst} bases ({n_bases_full / n_bases_cst:.1f}x less)")
    print(f"CST fidelity: {fid_cst:.4f}")
    print(f"Full MLE fidelity: {fid_full:.4f}")
    print(f"Fidelity difference: {abs(fid_cst - fid_full):.6f}")


def run_high_dimensional_example():
    np.random.seed(123)
    
    n_qubits = 4
    rank = 2
    print(f"\n\n=== High-Dimensional CST: {n_qubits} qubits (rank-{rank}) ===")
    
    true_rho = generate_pure_product_state(n_qubits, rng=np.random.RandomState(123))
    print(f"True state rank: {np.linalg.matrix_rank(true_rho)}")
    
    n_bases_cst = int(6 * rank * n_qubits)
    print(f"Number of CST bases: {n_bases_cst}")
    print(f"Full tomography would need: {3**n_qubits} bases")
    
    shots_per_basis = 3000
    
    rng = np.random.RandomState(123)
    random_bases = generate_random_bases(n_qubits, n_bases_cst, rng=rng)
    
    measurement_data_cst = simulate_measurements_with_bases(
        true_rho, n_qubits, random_bases, shots_per_basis=shots_per_basis, rng=rng
    )
    
    reconstructed_cst = reconstruct_cst_fast(
        measurement_data_cst, n_qubits, lambda_reg=0.0005
    )
    
    fid_cst = fidelity(reconstructed_cst, true_rho)
    print(f"\nCST Fidelity: {fid_cst:.6f}")
    print(f"Valid density matrix: {is_valid_density_matrix(reconstructed_cst)}")


if __name__ == "__main__":
    run_example()
    run_two_qubit_example()
    run_cst_example()
    run_high_dimensional_example()
