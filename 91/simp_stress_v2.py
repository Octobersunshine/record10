import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def simp_stress_optimization(nelx=50, nely=25, volfrac=0.4, penal=3.0, rmin=2.0, max_iter=80,
                               objective='compliance', p_norm=8.0, q_penal=0.5):
    """
    应力约束/应力最小化 SIMP 拓扑优化算法（稳定版）
    
    核心技术:
    1. q-relaxation: 处理低密度单元应力奇异性
    2. P-norm聚集: 将最大应力平滑化，便于梯度计算
    3. von Mises等效应力: 综合考虑正应力和剪应力
    
    参数:
        objective: 'compliance'(最小柔顺度) 或 'stress'(最小应力)
        p_norm: P-norm参数，越大越接近最大应力
        q_penal: q-relaxation参数，通常0.3-0.7
    """
    
    E0 = 1.0
    Emin = 1e-9
    nu = 0.3
    
    def lk():
        """平面应力四节点单元刚度矩阵"""
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
    
    def get_B_matrix():
        """应变-位移矩阵 B (平面应力)"""
        hx = 1.0 / nelx
        hy = 1.0 / nely
        B = np.zeros((3, 8))
        
        B[0, 0] = -1.0/(2*hx); B[0, 2] = 1.0/(2*hx); B[0, 4] = 1.0/(2*hx); B[0, 6] = -1.0/(2*hx)
        B[1, 1] = -1.0/(2*hy); B[1, 3] = -1.0/(2*hy); B[1, 5] = 1.0/(2*hy); B[1, 7] = 1.0/(2*hy)
        B[2, 0] = -1.0/(2*hy); B[2, 1] = -1.0/(2*hx); B[2, 2] = -1.0/(2*hy); B[2, 3] = 1.0/(2*hx)
        B[2, 4] = 1.0/(2*hy); B[2, 5] = 1.0/(2*hx); B[2, 6] = 1.0/(2*hy); B[2, 7] = -1.0/(2*hx)
        
        return B
    
    B = get_B_matrix()
    
    def get_D_matrix(E):
        """本构矩阵 D"""
        D = E/(1-nu**2) * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1-nu)/2]
        ])
        return D
    
    D0 = get_D_matrix(E0)
    DB = np.dot(D0, B)
    
    def edof(elx, ely):
        """单元自由度编号"""
        n1 = (nely + 1) * elx + ely
        n2 = (nely + 1) * (elx + 1) + ely
        return np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3])
    
    def compute_von_mises(U, x):
        """计算所有单元的von Mises等效应力"""
        sigma_vm = np.zeros((nely, nelx))
        
        for elx in range(nelx):
            for ely in range(nely):
                ed = edof(elx, ely)
                Ue = U[ed]
                
                E_e = Emin + x[ely, elx]**q_penal * (E0 - Emin)
                
                stress = np.dot(DB * (E_e / E0), Ue)
                sigma_x, sigma_y, tau_xy = stress
                
                sigma_vm[ely, elx] = np.sqrt(sigma_x**2 - sigma_x*sigma_y + sigma_y**2 + 3*tau_xy**2)
        
        return sigma_vm
    
    def compute_p_norm(sigma_vm, p):
        """计算P-norm聚集应力"""
        sum_p = np.sum(sigma_vm**p)
        if sum_p < 1e-20:
            return 0.0, np.zeros_like(sigma_vm)
        
        sigma_p = (sum_p / (nelx * nely))**(1/p)
        weights = (sigma_vm**(p - 1)) / sum_p
        return sigma_p, weights
    
    x = volfrac * np.ones((nely, nelx))
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
    ax_top, ax_stress, ax_conv, ax_hist, ax_vol, ax_comp = axes.flatten()
    
    print("=" * 80)
    print("SIMP 拓扑优化 - 应力最小化")
    print("=" * 80)
    print(f"计算域: {nelx} x {nely} 单元")
    print(f"目标函数: {'最小化应力' if objective == 'stress' else '最小化柔顺度'}")
    print(f"体积约束: {volfrac}")
    print(f"SIMP惩罚: p={penal}")
    print(f"应力聚集: P-norm={p_norm}")
    print(f"应力正则化: q={q_penal}")
    print("=" * 80)
    print()
    
    while change > 0.005 and loop < max_iter:
        loop += 1
        x_old = x.copy()
        
        ndof = 2 * (nelx + 1) * (nely + 1)
        iK = np.zeros(64 * nelx * nely, dtype=np.int32)
        jK = np.zeros(64 * nelx * nely, dtype=np.int32)
        sK = np.zeros(64 * nelx * nely)
        
        for elx in range(nelx):
            for ely in range(nely):
                ed = edof(elx, ely)
                start = 64 * (ely + nely * elx)
                E_e = Emin + x[ely, elx]**penal * (E0 - Emin)
                Ke_scaled = Ke * (E_e / E0)
                for i in range(8):
                    for j in range(8):
                        iK[start + 8*i + j] = ed[i]
                        jK[start + 8*i + j] = ed[j]
                        sK[start + 8*i + j] = Ke_scaled[i, j]
        
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
                E_e = Emin + x[ely, elx]**penal * (E0 - Emin)
                c += (E_e / E0) * ce
                dc[ely, elx] = - penal * x[ely, elx]**(penal - 1) * ce
        
        sigma_vm = compute_von_mises(U, x)
        sigma_p, sigma_weights = compute_p_norm(sigma_vm, p_norm)
        sigma_max = np.max(sigma_vm)
        
        dsigma = np.zeros((nely, nelx))
        for elx in range(nelx):
            for ely in range(nely):
                if sigma_vm[ely, elx] > 1e-10 and x[ely, elx] > 1e-3:
                    ed = edof(elx, ely)
                    Ue = U[ed]
                    stress = np.dot(DB, Ue)
                    sigma_x, sigma_y, tau_xy = stress
                    
                    dsigma_vm_dstress = np.zeros(3)
                    dsigma_vm_dstress[0] = (2*sigma_x - sigma_y) / (2 * sigma_vm[ely, elx])
                    dsigma_vm_dstress[1] = (-sigma_x + 2*sigma_y) / (2 * sigma_vm[ely, elx])
                    dsigma_vm_dstress[2] = (3 * tau_xy) / sigma_vm[ely, elx]
                    
                    dsigma[ely, elx] = q_penal * x[ely, elx]**(q_penal - 1) * np.dot(np.dot(dsigma_vm_dstress, DB), Ue)
        
        dsigma_p = sigma_vm**(p_norm - 1) * dsigma * p_norm * (sigma_p**(1 - p_norm)) / (nelx * nely)
        
        if objective == 'stress':
            d_obj = dsigma_p.copy()
        else:
            d_obj = dc.copy()
        
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
                    if be < max(0.001, x[j, i] - move):
                        xnew[j, i] = max(0.001, x[j, i] - move)
                    elif be > min(1.0, x[j, i] + move):
                        xnew[j, i] = min(1.0, x[j, i] + move)
                    else:
                        xnew[j, i] = be
            if np.sum(xnew) - volfrac * nelx * nely > 0:
                l1 = lmid
            else:
                l2 = lmid
        x = xnew.copy()
        
        change = np.max(np.abs(x - x_old))
        
        volume = np.sum(x) / (nelx * nely)
        history['compliance'].append(c)
        history['stress_pnorm'].append(sigma_p)
        history['stress_max'].append(sigma_max)
        history['volume'].append(volume)
        
        if loop % 5 == 0 or loop == 1:
            print(f"Iter {loop:3d} | Comp={c:.4f} | σ_max={sigma_max:.4f} | σ_p={sigma_p:.4f} | Vol={volume:.4f} | Δ={change:.4f}")
            
            ax_top.clear()
            ax_top.imshow(1 - x, cmap='gray', origin='upper')
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
            ax_conv.plot(history['stress_max'], 'r-', linewidth=2, label='Max Stress')
            ax_conv.set_xlabel('Iteration')
            ax_conv.set_ylabel('Max von Mises Stress')
            ax_conv.set_title('Stress Minimization')
            ax_conv.legend()
            ax_conv.grid(True, alpha=0.3)
            
            ax_hist.clear()
            stress_flat = sigma_vm[x > 0.1].flatten()
            if len(stress_flat) > 0:
                ax_hist.hist(stress_flat, bins=30, edgecolor='black', alpha=0.7)
            ax_hist.set_xlabel('von Mises Stress')
            ax_hist.set_ylabel('Element Count')
            ax_hist.set_title('Stress Distribution')
            ax_hist.grid(True, alpha=0.3)
            
            ax_vol.clear()
            ax_vol.plot(history['volume'], 'g-', linewidth=2)
            ax_vol.axhline(y=volfrac, color='k', linestyle='--', alpha=0.5)
            ax_vol.set_xlabel('Iteration')
            ax_vol.set_ylabel('Volume Fraction')
            ax_vol.set_title('Volume Evolution')
            ax_vol.grid(True, alpha=0.3)
            
            ax_comp.clear()
            ax_comp.plot(history['compliance'], 'b-', linewidth=2)
            ax_comp.set_xlabel('Iteration')
            ax_comp.set_ylabel('Compliance')
            ax_comp.set_title('Compliance Evolution')
            ax_comp.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.pause(0.05)
    
    plt.ioff()
    
    fig_final, axes_final = plt.subplots(2, 3, figsize=(16, 10))
    af1, af2, af3, af4, af5, af6 = axes_final.flatten()
    
    af1.imshow(1 - x, cmap='gray', origin='upper')
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
    
    af4.plot(history['stress_max'], 'r-', linewidth=2, label='Max σ')
    af4.plot(history['stress_pnorm'], 'b--', linewidth=2, label='P-norm σ')
    af4.set_xlabel('Iteration')
    af4.set_ylabel('Stress')
    af4.set_title('Stress History')
    af4.legend()
    af4.grid(True, alpha=0.3)
    
    af5.plot(history['volume'], 'g-', linewidth=2)
    af5.axhline(y=volfrac, color='k', linestyle='--')
    af5.set_xlabel('Iteration')
    af5.set_ylabel('Volume Fraction')
    af5.set_title('Volume Evolution')
    af5.grid(True, alpha=0.3)
    
    stress_flat_final = sigma_vm[x > 0.1].flatten()
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
    print(f"最终体积分数: {volume:.4f}")
    print("=" * 80)
    
    return x, sigma_vm, history


def compare_stress_vs_compliance():
    """对比应力最小化 vs 柔顺度最小化"""
    
    print("\n" + "=" * 80)
    print("对比实验: 应力最小化 vs 柔顺度最小化")
    print("=" * 80 + "\n")
    
    print("示例1: 应力最小化")
    print("-" * 60)
    x_stress, sigma_stress, hist_stress = simp_stress_optimization(
        nelx=40, nely=20,
        volfrac=0.4,
        penal=3.0,
        rmin=2.0,
        max_iter=60,
        objective='stress',
        p_norm=8.0,
        q_penal=0.5
    )
    
    print("\n示例2: 柔顺度最小化")
    print("-" * 60)
    x_comp, sigma_comp, hist_comp = simp_stress_optimization(
        nelx=40, nely=20,
        volfrac=0.4,
        penal=3.0,
        rmin=2.0,
        max_iter=60,
        objective='compliance',
        p_norm=8.0,
        q_penal=0.5
    )
    
    print("\n" + "=" * 80)
    print("结果对比 Summary:")
    print("=" * 80)
    print(f"应力最小化 - 柔顺度: {hist_stress['compliance'][-1]:.4f}, 最大应力: {hist_stress['stress_max'][-1]:.4f}")
    print(f"柔顺度最小化 - 柔顺度: {hist_comp['compliance'][-1]:.4f}, 最大应力: {hist_comp['stress_max'][-1]:.4f}")
    print()
    stress_reduction = (hist_comp['stress_max'][-1] - hist_stress['stress_max'][-1]) / hist_comp['stress_max'][-1] * 100
    compliance_increase = (hist_stress['compliance'][-1] - hist_comp['compliance'][-1]) / hist_comp['compliance'][-1] * 100
    print(f"应力降低: {stress_reduction:.1f}%")
    print(f"柔顺度增加: {compliance_increase:.1f}%")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].imshow(1 - x_stress, cmap='gray', origin='upper')
    axes[0, 0].set_title(f'Stress Minimization\nMax Stress={hist_stress["stress_max"][-1]:.3f}')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    
    im1 = axes[0, 1].imshow(sigma_stress, cmap='jet', origin='upper')
    axes[0, 1].set_title('von Mises Stress')
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    plt.colorbar(im1, ax=axes[0, 1])
    
    axes[1, 0].imshow(1 - x_comp, cmap='gray', origin='upper')
    axes[1, 0].set_title(f'Compliance Minimization\nCompliance={hist_comp["compliance"][-1]:.3f}')
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    
    im2 = axes[1, 1].imshow(sigma_comp, cmap='jet', origin='upper')
    axes[1, 1].set_title('von Mises Stress')
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    plt.colorbar(im2, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_stress_vs_compliance()
