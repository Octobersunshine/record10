import numpy as np
from quantum_state_tomography import (
    reconstruct_mle, reconstruct_cst, reconstruct_cst_fast,
    fidelity, simulate_measurements, simulate_measurements_with_bases,
    is_valid_density_matrix, generate_random_bases,
    generate_low_rank_state, generate_pure_product_state,
    generate_measurement_bases
)


def test_single_qubit_mle():
    print("=== Test 1: 单量子比特 MLE 重构 ===")
    np.random.seed(42)
    
    psi = np.array([1, 1]) / np.sqrt(2)
    true_rho = np.outer(psi, psi.conj())
    
    measurement_data = simulate_measurements(true_rho, n_qubits=1, shots_per_basis=10000)
    reconstructed_rho = reconstruct_mle(measurement_data, n_qubits=1)
    
    fid = fidelity(reconstructed_rho, true_rho)
    valid = is_valid_density_matrix(reconstructed_rho)
    
    print(f"  Fidelity: {fid:.6f}")
    print(f"  Valid: {valid}")
    print(f"  {'PASS' if fid > 0.95 and valid else 'FAIL'}")
    
    return fid > 0.95 and valid


def test_two_qubit_bell_mle():
    print("\n=== Test 2: 双量子比特 Bell 态 MLE 重构 ===")
    np.random.seed(42)
    
    psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
    true_rho = np.outer(psi, psi.conj())
    
    measurement_data = simulate_measurements(true_rho, n_qubits=2, shots_per_basis=5000)
    reconstructed_rho = reconstruct_mle(measurement_data, n_qubits=2)
    
    fid = fidelity(reconstructed_rho, true_rho)
    valid = is_valid_density_matrix(reconstructed_rho)
    
    print(f"  Fidelity: {fid:.6f}")
    print(f"  Valid: {valid}")
    print(f"  {'PASS' if fid > 0.9 and valid else 'FAIL'}")
    
    return fid > 0.9 and valid


def test_cst_low_rank():
    print("\n=== Test 3: 压缩感知低秩态重构 ===")
    np.random.seed(42)
    
    n_qubits = 3
    rank = 2
    
    true_rho = generate_low_rank_state(n_qubits, rank=rank, rng=np.random.RandomState(42))
    
    n_bases_cst = int(4 * rank * n_qubits)
    n_bases_full = 3**n_qubits
    
    print(f"  Qubits: {n_qubits}, Rank: {rank}")
    print(f"  CST bases: {n_bases_cst}, Full bases: {n_bases_full}")
    print(f"  Reduction: {n_bases_full / n_bases_cst:.1f}x")
    
    rng = np.random.RandomState(42)
    random_bases = generate_random_bases(n_qubits, n_bases_cst, rng=rng)
    
    measurement_data = simulate_measurements_with_bases(
        true_rho, n_qubits, random_bases, shots_per_basis=3000, rng=rng
    )
    
    reconstructed_cst = reconstruct_cst_fast(
        measurement_data, n_qubits, lambda_reg=0.001
    )
    
    fid = fidelity(reconstructed_cst, true_rho)
    valid = is_valid_density_matrix(reconstructed_cst)
    recon_rank = np.linalg.matrix_rank(reconstructed_cst, tol=1e-3)
    
    print(f"  CST Fidelity: {fid:.6f}")
    print(f"  Reconstructed rank: {recon_rank}")
    print(f"  Valid: {valid}")
    print(f"  {'PASS' if fid > 0.85 and valid else 'FAIL'}")
    
    return fid > 0.85 and valid


def test_cst_vs_mle_comparison():
    print("\n=== Test 4: CST vs MLE 对比 ===")
    np.random.seed(42)
    
    n_qubits = 3
    rank = 1
    
    true_rho = generate_pure_product_state(n_qubits, rng=np.random.RandomState(42))
    
    n_bases_cst = int(5 * rank * n_qubits)
    n_bases_full = 3**n_qubits
    
    print(f"  Qubits: {n_qubits}, True rank: {rank}")
    print(f"  CST bases: {n_bases_cst}, Full bases: {n_bases_full}")
    
    rng = np.random.RandomState(42)
    shots_per_basis = 2000
    
    random_bases = generate_random_bases(n_qubits, n_bases_cst, rng=rng)
    measurement_data_cst = simulate_measurements_with_bases(
        true_rho, n_qubits, random_bases, shots_per_basis=shots_per_basis, rng=rng
    )
    
    all_bases = generate_measurement_bases(n_qubits)
    measurement_data_full = simulate_measurements_with_bases(
        true_rho, n_qubits, all_bases, shots_per_basis=shots_per_basis, rng=rng
    )
    
    reconstructed_cst = reconstruct_cst_fast(
        measurement_data_cst, n_qubits, lambda_reg=0.0005
    )
    fid_cst = fidelity(reconstructed_cst, true_rho)
    
    reconstructed_mle = reconstruct_mle(measurement_data_full, n_qubits)
    fid_mle = fidelity(reconstructed_mle, true_rho)
    
    print(f"  CST Fidelity: {fid_cst:.6f}")
    print(f"  MLE Fidelity: {fid_mle:.6f}")
    print(f"  Fidelity difference: {abs(fid_cst - fid_mle):.6f}")
    print(f"  {'PASS' if fid_cst > 0.9 else 'FAIL'}")
    
    return fid_cst > 0.9


def test_high_dimensional_cst():
    print("\n=== Test 5: 高维量子态 CST (4 qubits) ===")
    np.random.seed(123)
    
    n_qubits = 4
    rank = 1
    
    true_rho = generate_pure_product_state(n_qubits, rng=np.random.RandomState(123))
    
    n_bases_cst = int(6 * rank * n_qubits)
    n_bases_full = 3**n_qubits
    
    print(f"  Qubits: {n_qubits}")
    print(f"  CST bases: {n_bases_cst}, Full bases would require: {n_bases_full}")
    print(f"  Measurement reduction: {n_bases_full / n_bases_cst:.1f}x")
    
    rng = np.random.RandomState(123)
    random_bases = generate_random_bases(n_qubits, n_bases_cst, rng=rng)
    
    measurement_data = simulate_measurements_with_bases(
        true_rho, n_qubits, random_bases, shots_per_basis=3000, rng=rng
    )
    
    reconstructed_cst = reconstruct_cst_fast(
        measurement_data, n_qubits, lambda_reg=0.0005
    )
    
    fid = fidelity(reconstructed_cst, true_rho)
    valid = is_valid_density_matrix(reconstructed_cst)
    
    print(f"  Fidelity: {fid:.6f}")
    print(f"  Valid: {valid}")
    print(f"  {'PASS' if fid > 0.85 and valid else 'FAIL'}")
    
    return fid > 0.85 and valid


def test_hermiticity_positivity():
    print("\n=== Test 6: 密度矩阵物理性质验证 ===")
    np.random.seed(42)
    
    n_qubits = 2
    
    psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
    true_rho = np.outer(psi, psi.conj())
    
    measurement_data = simulate_measurements(true_rho, n_qubits=2, shots_per_basis=2000)
    
    reconstructed_mle = reconstruct_mle(measurement_data, n_qubits=2)
    
    is_hermitian = np.allclose(reconstructed_mle, reconstructed_mle.conj().T)
    eigvals = np.linalg.eigvalsh(reconstructed_mle)
    is_positive = np.all(eigvals >= -1e-10)
    trace_one = np.isclose(np.trace(reconstructed_mle), 1.0)
    
    print(f"  Hermitian: {is_hermitian}")
    print(f"  Positive semi-definite: {is_positive}")
    print(f"  Trace = 1: {trace_one}")
    print(f"  Eigenvalues: {eigvals}")
    
    passed = is_hermitian and is_positive and trace_one
    print(f"  {'PASS' if passed else 'FAIL'}")
    
    return passed


if __name__ == "__main__":
    results = []
    
    results.append(("单量子比特 MLE", test_single_qubit_mle()))
    results.append(("双量子比特 MLE", test_two_qubit_bell_mle()))
    results.append(("CST 低秩重构", test_cst_low_rank()))
    results.append(("CST vs MLE 对比", test_cst_vs_mle_comparison()))
    results.append(("高维 CST", test_high_dimensional_cst()))
    results.append(("物理性质验证", test_hermiticity_positivity()))
    
    print("\n" + "="*50)
    print("FINAL TEST SUMMARY")
    print("="*50)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    print(f"\n  Overall: {'ALL PASS' if all_passed else 'SOME FAIL'}")
