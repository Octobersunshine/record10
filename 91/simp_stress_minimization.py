import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def simp_stress_optimization(nelx=60, nely=30, volfrac=0.4, penal=3.0, rmin=1.5, max_iter=100,
                               stress_constraint=None, objective='compliance',
                               p_norm=6.0, q_penal=0.5,
                               use_projection=True, beta=8.0, eta=0.5):
    """
    应力约束/应力最小化 SIMP 拓扑优化算法
    
    参数:
        nelx, nely: x,y方向单元数
        volfrac: 体积约束
        penal: SIMP惩罚因子
        rmin: 滤波器半径
        max_iter: 最大迭代次数
        stress_constraint: 应力约束值（None表示无约束，最小化应力）
        objective: 目标函数 'compliance'(柔顺度) 或 'stress'(应力)
        p_norm: P-norm参数，越大越接近最大应力
        q_penal: q-relaxation参数，处理低密度单元应力奇异
        use_projection: 是否使用Heaviside投影
        beta, eta: Heaviside投影参数
    """
    
    E0 = 1.0
    Emin = 1e-9
    nu = 0.3
    
    def lk():
        """平面应力单元刚度矩阵"""
        k = np.array([1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8,
                       -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8])
        Ke = E0/(1-nu**2) * np.array([
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
        return Ke
    
    Ke = lk()
    
    def stress_displacement_matrix():
        """应力-位移矩阵 B，将位移转换为应力"""
        B = np.zeros((3, 8))
        a = 1.0 / nelx
        b = 1.0 / nely
        
        B[0, 0] = -b/4
        B[0, 2] = b/4
        B[0, 4] = b/4
        B[0, 6] = -b/4
        
        B[1, 1] = -a/4
        B[1, 3] = -a/4
        B[1, 5] = a/4
        B[1, 7] = a/4
        
        B[2, 0] = -a/4
        B[2, 1] = -b/4
        B[2, 2] = -a/4
        B[2, 3] = b/4
        B[2, 4] = a/4
        B[2, 5] = b/4
        B[2, 6] = a/4
        B[2, 7] = -b/4
        
        return B
    
    B = stress_displacement_matrix()
    
    def constitutive_matrix(E):
        """本构矩阵 D，平面应力"""
        D = E/(1-nu**2) * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1-nu)/2]
        ])
        return D
    
    D0 = constitutive_matrix(E0)
    DB = np.dot(D0, B)
    
    def edof(elx, ely):
        n1 = (nely + 1) * elx + ely
        n2 = (nely + 1) * (elx + 1) + ely
        return np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3])
    
    def heaviside_projection(x, beta, eta):
        numerator = np.tanh(beta * eta) + np.tanh(beta * (x - eta))
        denominator = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
        return numerator / denominator
    
    def compute_von_mises(U, x_phys):
        """计算所有单元的von Mises应力"""
        sigma_vm = np.zeros((nely, nelx))
        
        for elx in range(nelx):
            for ely in range(nely):
                ed = edof(elx, ely)
                Ue = U[ed]
                
                # 使用q-relaxation: E = Emin + x^q * (E0 - Emin)
                E_e = Emin + x_phys[ely, elx]**q_penal * (E0 - Emin)
                
                stress = np.dot(DB * (E_e / E0), Ue)
                sigma_x, sigma_y, tau_xy = stress
                
                sigma_vm[ely, elx] = np.sqrt(sigma_x**2 - sigma_x*sigma_y + sigma_y**2 + 3*tau_xy**2)
        
        return sigma_vm
    
    def compute_p_norm_stress(sigma_vm):
        """计算P-norm聚集应力（近似最大应力）"""
        max_sigma = np.max(sigma_vm)
        if max_sigma < 1e-10:
            return 0.0, np.zeros_like(sigma_vm)
        
        sigma_p = (np.sum(sigma_vm**p_norm) / (nelx * nely))**(1/p_norm)
        weights = (sigma_vm**(p_norm - 1)) / (np.sum(sigma_vm**p_norm))
        return sigma_p, weights
    
    x = volfrac * np.ones((nely, nelx))
    x_phys = x.copy()
    loop = 0
    change = 1.0
    
    history = {
        'compliance': [],
        'stress_pnorm': [],
        'stress_max': [],
        'volume': []
    }
    
    plt.ion()
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    ax_top, ax_stress, ax_conv, ax_hist, ax_vol, ax_sigma = axes.flatten()
    
    print("=" * 80)
    print("SIMP 拓扑优化 - 应力约束/最小化")
    print("=" * 80)
    print(f"计算域: {nelx} x {nely} 单元")
    print(f"目标函数: {'最小化应力' if objective == 'stress' else '最小化柔顺度'}")
    print(f"体积约束: {volfrac}")
    if stress_constraint is not None:
        print(f"应力约束: {stress_constraint}")
    print(f"SIMP惩罚因子: {penal}")
    print(f"P-norm参数: {p_norm}")
    print(f"q-relaxation: {q_penal}")
    print("=" * 80)
    print()
    
    while change > 0.005 and loop < max_iter:
        loop += 1
        
        if use_projection:
            x_phys = heaviside_projection(x, beta, eta)
        else:
            x_phys = x.copy()
        
        ndof = 2 * (nelx + 1) * (nely + 1)
        iK = np.zeros(64 * nelx * nely, dtype=np.int32)
        jK = np.zeros(64 * nelx * nely, dtype=np.int32)
        sK = np.zeros(64 * nelx * nely)
        
        for elx in range(nelx):
            for ely in range(nely):
                ed = edof(elx, ely)
                start = 64 * (ely + nely * elx)
                E_e = Emin + x_phys[ely, elx]**penal * (E0 - Emin)
                for i in range(8):
                    for j in range(8):
                        iK[start + 8*i + j] = ed[i]
                        jK[start + 8*i + j] = ed[j]
                        sK[start + 8*i + j] = E_e / E0 * Ke[i, j]
        
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof))
        K = (K + K.T) / 2.0
        
        F = np.zeros(ndof)
        F[2 * (nely + 1) - 1] = -1.0
        
        fixed_dofs = np.union1d(np.arange(0, 2*(nely+1), 2), np.arange(1, 2*(nely+1), 2))
        free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)
        
        U = np.zeros(ndof)
        U[free_dofs] = spsolve(K[free_dofs, :][:, free_dofs], F[free_dofs])
        
        c = 0.0
        dc = np.zeros((nely, nelx))
        
        for elx in range(nelx):
            for ely in range(nely):
                ed = edof(elx, ely)
                Ue = U[ed]
                ce = np.dot(np.dot(Ue.T, Ke), Ue)
                E_e = Emin + x_phys[ely, elx]**penal * (E0 - Emin)
                c += E_e / E0 * ce
                dc[ely, elx] = - penal * x_phys[ely, elx]**(penal - 1) * ce
        
        sigma_vm = compute_von_mises(U, x_phys)
        sigma_p, sigma_weights = compute_p_norm_stress(sigma_vm)
        sigma_max = np.max(sigma_vm)
        
        dsigma = np.zeros((nely, nelx))
        for elx in range(nelx):
            for ely in range(nely):
                ed = edof(elx, ely)
                Ue = U[ed]
                stress = np.dot(DB, Ue)
                sigma_x, sigma_y, tau_xy = stress
                
                if sigma_vm[ely, elx] > 1e-10:
                    dsigma_dU = np.zeros(8)
                    for i in range(8):
                        dsigma_dU[i] = (
                            (2*sigma_x - sigma_y) * DB[0, i] + \
                            (-sigma_x + 2*sigma_y) * DB[1, i] + \
                            6 * tau_xy * DB[2, i]
                        ) / (2 * sigma_vm[ely, elx])
                    
                    E_e = Emin + x_phys[ely, elx]**q_penal * (E0 - Emin)
                    dsigma[ely, elx] = np.dot(dsigma_dU, Ue) * q_penal * x_phys[ely, elx]**(q_penal - 1) * (E0 - Emin) / E0
        
        dsigma_p = p_norm * (sigma_p**(1 - p_norm)) * (sigma_vm**(p_norm - 1)) * dsigma / (nelx * nely)
        
        if objective == 'stress':
            d_obj = dsigma_p.copy()
            obj_val = sigma_p
        else:
            d_obj = dc.copy()
            obj_val = c
        
        dv = np.ones((nely, nelx))
        
        dcn = np.zeros((nely, nelx))
        dvy = np.zeros((nely, nelx))
        for i in range(nelx):
            for j in range(nely):
                sum_ = 0.0
                sum_dc = 0.0
                sum_dv = 0.0
                for k in range(max(0, int(i - np.floor(rmin))), min(nelx, int(i + np.floor(rmin) + 1))):
                    for l in range(max(0, int(j - np.floor(rmin))), min(nely, int(j + np.floor(rmin) + 1))):
                        fac = rmin - np.sqrt((i - k)**2 + (j - l)**2)
                        if fac > 0:
                            sum_ += fac
                            sum_dc += fac * d_obj[l, k]
                            sum_dv += fac * dv[l, k]
                dcn[j, i] = sum_dc / sum_
                dvy[j, i] = sum_dv / sum_
        d_obj = dcn
        dv = dvy
        
        if use_projection:
            dproj = beta * (1 - np.tanh(beta * (x - eta))**2) / denominator
            d_obj = d_obj * dproj
        
        l1 = 0.0
        l2 = 1e9
        move = 0.15
        xnew = np.zeros_like(x)
        while (l2 - l1) / (l1 + l2) > 1e-7:
            lmid = 0.5 * (l1 + l2)
            for i in range(nelx):
                for j in range(nely):
                    if d_obj[j, i] < 0:
                        be = (-d_obj[j, i] / (dv[j, i] * lmid))**(1/(penal-1))
                    else:
                        be = 1e10
                    if be < max(0.0, x[j, i] - move):
                        xnew[j, i] = max(0.0, x[j, i] - move)
                    elif be > min(1.0, x[j, i] + move):
                        xnew[j, i] = min(1.0, x[j, i] + move)
                    else:
                        xnew[j, i] = be
            if np.sum(xnew) - volfrac * nelx * nely > 0:
                l1 = lmid
            else:
                l2 = lmid
        x = xnew.copy()
        
        change = np.max(np.abs(x - x_phys))
        
        history['compliance'].append(c)
        history['stress_pnorm'].append(sigma_p)
        history['stress_max'].append(sigma_max)
        history['volume'].append(np.sum(x)/(nelx*nely))
        
        if loop % 5 == 0 or loop == 1:
            gray_ratio = np.sum((x_phys > 0.05) & (x_phys < 0.95)) / (nelx * nely)
            print(f"Iter {loop:3d} | Comp={c:.4f} | σ_p={sigma_p:.4f} | σ_max={sigma_max:.4f} | Vol={history['volume'][-1]:.4f} | Δ={change:.4f}")
            
            ax_top.clear()
            im = ax_top.imshow(1 - x_phys, cmap='gray', origin='upper')
            ax_top.set_title(f'Topology (Iter {loop})')
            ax_top.set_xticks([])
            ax_top.set_yticks([])
            
            ax_stress.clear()
            im_stress = ax_stress.imshow(sigma_vm, cmap='jet', origin='upper')
            ax_stress.set_title(f'von Mises Stress (max={sigma_max:.3f})')
            ax_stress.set_xticks([])
            ax_stress.set_yticks([])
            plt.colorbar(im_stress, ax=ax_stress)
            
            ax_conv.clear()
            ax_conv.plot(history['compliance'], 'b-', label='Compliance')
            ax_conv.plot(history['stress_pnorm'], 'r-', label='P-norm Stress')
            ax_conv.set_xlabel('Iteration')
            ax_conv.set_ylabel('Value')
            ax_conv.set_title('Convergence History')
            ax_conv.legend()
            ax_conv.grid(True, alpha=0.3)
            
            ax_hist.clear()
            stress_flat = sigma_vm[x_phys > 0.1].flatten()
            if len(stress_flat) > 0:
                ax_hist.hist(stress_flat, bins=30, edgecolor='black', alpha=0.7)
            ax_hist.set_xlabel('von Mises Stress')
            ax_hist.set_ylabel('Count')
            ax_hist.set_title('Stress Distribution')
            ax_hist.grid(True, alpha=0.3)
            
            ax_vol.clear()
            ax_vol.plot(history['volume'], 'g-', linewidth=2)
            ax_vol.axhline(y=volfrac, color='k', linestyle='--', alpha=0.5)
            ax_vol.set_xlabel('Iteration')
            ax_vol.set_ylabel('Volume Fraction')
            ax_vol.set_title('Volume Evolution')
            ax_vol.grid(True, alpha=0.3)
            
            ax_sigma.clear()
            ax_sigma.plot(history['stress_max'], 'r-', label='Max σ')
            ax_sigma.plot(history['stress_pnorm'], 'b--', label='P-norm σ')
            if stress_constraint is not None:
                ax_sigma.axhline(y=stress_constraint, color='k', linestyle='--')
            ax_sigma.set_xlabel('Iteration')
            ax_sigma.set_ylabel('Stress')
            ax_sigma.set_title('Stress Evolution')
            ax_sigma.legend()
            ax_sigma.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.pause(0.05)
    
    plt.ioff()
    
    fig_final, axes_final = plt.subplots(2, 3, figsize=(16, 10))
    af1, af2, af3, af4, af5, af6 = axes_final.flatten()
    
    af1.imshow(1 - x_phys, cmap='gray', origin='upper')
    af1.set_title(f'Final Topology\nCompliance={c:.4f}')
    af1.set_xticks([])
    af1.set_yticks([])
    
    im_final = af2.imshow(sigma_vm, cmap='jet', origin='upper')
    af2.set_title(f'von Mises Stress\nMax={sigma_max:.4f}')
    af2.set_xticks([])
    af2.set_yticks([])
    plt.colorbar(im_final, ax=af2)
    
    af3.plot(history['compliance'], 'b-', linewidth=2)
    af3.set_xlabel('Iteration')
    af3.set_ylabel('Compliance')
    af3.set_title('Compliance History')
    af3.grid(True, alpha=0.3)
    
    af4.plot(history['stress_max'], 'r-', linewidth=2)
    af4.plot(history['stress_pnorm'], 'b--', linewidth=2)
    af4.set_xlabel('Iteration')
    af4.set_ylabel('Stress')
    af4.set_title('Stress History')
    af4.grid(True, alpha=0.3)
    af4.legend(['Max σ', 'P-norm σ'])
    
    af5.plot(history['volume'], 'g-', linewidth=2)
    af5.axhline(y=volfrac, color='k', linestyle='--')
    af5.set_xlabel('Iteration')
    af5.set_ylabel('Volume Fraction')
    af5.set_title('Volume Evolution')
    af5.grid(True, alpha=0.3)
    
    stress_flat_final = sigma_vm[x_phys > 0.1].flatten()
    if len(stress_flat_final) > 0:
        af6.hist(stress_flat_final, bins=30, edgecolor='black', alpha=0.7)
    af6.set_xlabel('von Mises Stress')
    af6.set_ylabel('Count')
    af6.set_title('Final Stress Distribution')
    af6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print()
    print("=" * 80)
    print("优化完成!")
    print(f"最终柔顺度: {c:.4f}")
    print(f"最终最大应力: {sigma_max:.4f}")
    print(f"最终P-norm应力: {sigma_p:.4f}")
    print(f"最终体积分数: {np.sum(x)/(nelx*nely):.4f}")
    print("=" * 80)
    
    return x_phys, sigma_vm, history


def example_stress_minimization():
    """应力最小化示例"""
    print("\n" + "=" * 80)
    print("示例1: 应力最小化 (Minimize Stress)")
    print("=" * 80 + "\n")
    
    x_stress, sigma_stress, hist_stress = simp_stress_optimization(
        nelx=50, nely=25,
        volfrac=0.4,
        penal=3.0,
        rmin=2.0,
        max_iter=80,
        objective='stress',
        p_norm=8.0,
        q_penal=0.5
    )
    
    print("\n" + "=" * 80)
    print("示例2: 柔顺度最小化 (对比)")
    print("=" * 80 + "\n")
    
    x_comp, sigma_comp, hist_comp = simp_stress_optimization(
        nelx=50, nely=25,
        volfrac=0.4,
        penal=3.0,
        rmin=2.0,
        max_iter=80,
        objective='compliance',
        p_norm=8.0,
        q_penal=0.5
    )
    
    print("\n" + "=" * 80)
    print("结果对比:")
    print("=" * 80)
    print(f"应力最小化 - 柔顺度: {hist_stress['compliance'][-1]:.4f}, 最大应力: {hist_stress['stress_max'][-1]:.4f}")
    print(f"柔顺度最小化 - 柔顺度: {hist_comp['compliance'][-1]:.4f}, 最大应力: {hist_comp['stress_max'][-1]:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    example_stress_minimization()
