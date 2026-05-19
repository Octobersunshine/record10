import numpy as np


def solve_schrodinger_1d(x_min=-10, x_max=10, num_points=1000, num_eigenvalues=5, lam=0.0, method='direct'):
    x_full = np.linspace(x_min, x_max, num_points)
    dx = x_full[1] - x_full[0]
    
    x = x_full[1:-1]
    n_inner = len(x)
    
    V0 = 0.5 * x**2
    V1 = lam * x**4
    V = V0 + V1
    
    T = -0.5 * (np.diag(np.ones(n_inner-1), -1) - 2*np.diag(np.ones(n_inner), 0) + np.diag(np.ones(n_inner-1), 1)) / dx**2
    
    if method == 'direct':
        H = T + np.diag(V)
        eigenvalues, eigenvectors_inner = np.linalg.eigh(H)
    elif method == 'perturbation_1st':
        H0 = T + np.diag(V0)
        E0, psi0 = np.linalg.eigh(H0)
        E0 = E0[:num_eigenvalues]
        psi0 = psi0[:, :num_eigenvalues]
        
        eigenvalues = np.zeros(num_eigenvalues)
        for n in range(num_eigenvalues):
            E1_n = np.sum(psi0[:, n] * V1 * psi0[:, n]) * dx
            eigenvalues[n] = E0[n] + E1_n
        
        eigenvectors_inner = psi0
    elif method == 'perturbation_2nd':
        H0 = T + np.diag(V0)
        E0, psi0 = np.linalg.eigh(H0)
        n_total = min(num_eigenvalues + 20, len(E0))
        E0 = E0[:n_total]
        psi0 = psi0[:, :n_total]
        
        eigenvalues = np.zeros(num_eigenvalues)
        eigenvectors_inner = np.zeros((n_inner, num_eigenvalues))
        
        for n in range(num_eigenvalues):
            E1_n = np.sum(psi0[:, n] * V1 * psi0[:, n]) * dx
            
            E2_n = 0.0
            for k in range(n_total):
                if k != n:
                    V_nk = np.sum(psi0[:, n] * V1 * psi0[:, k]) * dx
                    E2_n += V_nk**2 / (E0[n] - E0[k])
            
            eigenvalues[n] = E0[n] + E1_n + E2_n
            
            psi_n = psi0[:, n].copy()
            for k in range(n_total):
                if k != n:
                    V_nk = np.sum(psi0[:, n] * V1 * psi0[:, k]) * dx
                    psi_n += V_nk / (E0[n] - E0[k]) * psi0[:, k]
            
            eigenvectors_inner[:, n] = psi_n
    elif method == 'iterative':
        H = T + np.diag(V)
        eigenvalues, eigenvectors_inner = np.linalg.eigh(H)
        
        for _ in range(3):
            V_eff = np.diag(V)
            for n in range(num_eigenvalues):
                psi_n = eigenvectors_inner[:, n]
                rho_n = psi_n**2
                V_eff += lam * np.diag(rho_n * x**4) * 0.1
            
            H_eff = T + V_eff
            eigenvalues, eigenvectors_inner = np.linalg.eigh(H_eff)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    eigenvalues = eigenvalues[:num_eigenvalues]
    eigenvectors_inner = eigenvectors_inner[:, :num_eigenvalues]
    
    eigenvectors = np.zeros((num_points, num_eigenvalues))
    eigenvectors[1:-1, :] = eigenvectors_inner
    
    for i in range(num_eigenvalues):
        norm = np.sqrt(np.sum(eigenvectors[:, i]**2) * dx)
        eigenvectors[:, i] /= norm
    
    return x_full, eigenvalues, eigenvectors


def analytical_perturbation_1st(n, lam):
    E0 = n + 0.5
    E1 = lam * (3/4) * (2*n**2 + 2*n + 1)
    return E0 + E1


if __name__ == "__main__":
    lam = 0.1
    num_eigenvalues = 5
    
    print("=" * 80)
    print(f"非简谐振子求解：V(x) = 0.5*x² + {lam}*x⁴")
    print("=" * 80)
    print()
    
    print("方法对比：")
    print(f"{'能级':<6} {'直接对角化':<14} {'一阶微扰(数值)':<14} {'二阶微扰(数值)':<14} {'一阶微扰(解析)':<14}")
    print("-" * 80)
    
    _, E_direct, _ = solve_schrodinger_1d(x_min=-10, x_max=10, num_points=2000, 
                                           num_eigenvalues=num_eigenvalues, lam=lam, method='direct')
    _, E_pert1, _ = solve_schrodinger_1d(x_min=-10, x_max=10, num_points=2000, 
                                          num_eigenvalues=num_eigenvalues, lam=lam, method='perturbation_1st')
    _, E_pert2, _ = solve_schrodinger_1d(x_min=-10, x_max=10, num_points=2000, 
                                          num_eigenvalues=num_eigenvalues, lam=lam, method='perturbation_2nd')
    
    for n in range(num_eigenvalues):
        E_pert1_analytical = analytical_perturbation_1st(n, lam)
        print(f"E_{n:<5} {E_direct[n]:<14.6f} {E_pert1[n]:<14.6f} {E_pert2[n]:<14.6f} {E_pert1_analytical:<14.6f}")
    
    print()
    print("误差分析（以直接对角化为参考）：")
    print(f"{'能级':<6} {'一阶微扰误差':<14} {'二阶微扰误差':<14}")
    print("-" * 50)
    for n in range(num_eigenvalues):
        err1 = abs(E_pert1[n] - E_direct[n])
        err2 = abs(E_pert2[n] - E_direct[n])
        print(f"E_{n:<5} {err1:<14.6f} {err2:<14.6f}")
    
    print()
    print("=" * 80)
    print("迭代对角化测试：")
    _, E_iter, _ = solve_schrodinger_1d(x_min=-10, x_max=10, num_points=2000, 
                                         num_eigenvalues=num_eigenvalues, lam=lam, method='iterative')
    for n in range(num_eigenvalues):
        print(f"  E_{n} = {E_iter[n]:.6f}, 与直接对角化误差: {abs(E_iter[n] - E_direct[n]):.6f}")
    print("=" * 80)
