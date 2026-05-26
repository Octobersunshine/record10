import numpy as np
from scipy.linalg import eig

m = 10.0
I = 2.0
wh = 2 * np.pi * 1.0
wt = 2 * np.pi * 8.0
rho = 1.225
B = 0.3
zeta_b = 0.0001
zeta_t = 0.0001

print("U (m/s) | k_t | Bend: damp | freq  | Tors: damp | freq  | A2*")
print("-" * 75)

for U in [2.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 150.0, 200.0]:
    k_t = wt * B / (2 * U)
    q = 0.5 * rho * B
    
    H1 = 0.25 * (1 - np.exp(-0.60 * k_t)) - 0.01 * k_t
    H3 = 0.50 * (1 - np.exp(-1.00 * k_t)) - 0.02 * k_t
    A1 = 2.00 * (1 - np.exp(-0.80 * k_t)) - 0.05 * k_t
    A3 = 1.80 * (1 - np.exp(-0.70 * k_t))
    H2 = 0.30 * (1 - np.exp(-k_t / 0.40))
    A2 = 0.35 * (1 - np.exp(-k_t / 0.30)) - 0.30 * np.exp(-k_t / 0.10)
    H4 = -0.10 * k_t * np.exp(-0.8 * k_t)
    A4 = -0.15 * k_t * np.exp(-0.6 * k_t)
    
    Ka = q * U**2 * np.array([[H1, H3 * B], [A1 * B, A3 * B**2]])
    Ca = q * U * np.array([[H2, H4 * B], [A2 * B, A4 * B**2]])
    
    Ks = np.diag([m * wh**2, I * wt**2])
    Ms = np.diag([m, I])
    Cs = np.diag([2 * zeta_b * m * wh, 2 * zeta_t * I * wt])
    
    K = Ks + Ka
    C = Cs + Ca
    
    n = 2
    A = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [-np.linalg.inv(Ms) @ K, -np.linalg.inv(Ms) @ C]
    ])
    
    evals, _ = eig(A)
    
    omegas = np.abs(np.imag(evals))
    idx0 = int(np.argmin(np.abs(omegas - wh)))
    idx1 = int(np.argmin(np.abs(omegas - wt)))
    
    d0 = evals[idx0].real
    f0 = abs(evals[idx0].imag) / (2 * np.pi)
    d1 = evals[idx1].real
    f1 = abs(evals[idx1].imag) / (2 * np.pi)
    
    print(f"{U:7.1f} | {k_t:.3f} | "
          f"{d0:+.4e} | {f0:.3f} | "
          f"{d1:+.4e} | {f1:.3f} | "
          f"{A2:+.3f}")