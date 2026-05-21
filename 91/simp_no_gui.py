import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def simp_topology_optimization(nelx=30, nely=15, volfrac=0.5, penal=3.0, rmin=1.5, max_iter=20):
    """
    SIMP 拓扑优化算法（无GUI版本）
    
    参数:
        nelx: x方向单元数
        nely: y方向单元数
        volfrac: 体积约束分数
        penal: SIMP惩罚因子
        rmin: 滤波器半径
        max_iter: 最大迭代次数
    
    返回:
        x: 最终密度分布
        c_history: 柔顺度历史
    """
    
    def lk():
        E = 1.0
        nu = 0.3
        k = np.array([1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8,
                       -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8])
        Ke = E/(1-nu**2) * np.array([
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
    
    def edof(elx, ely):
        n1 = (nely + 1) * elx + ely
        n2 = (nely + 1) * (elx + 1) + ely
        return np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3])
    
    x = volfrac * np.ones((nely, nelx))
    x_phys = x.copy()
    loop = 0
    change = 1.0
    c_history = []
    
    while change > 0.01 and loop < max_iter:
        loop += 1
        x_phys = x.copy()
        
        ndof = 2 * (nelx + 1) * (nely + 1)
        iK = np.zeros(64 * nelx * nely, dtype=np.int32)
        jK = np.zeros(64 * nelx * nely, dtype=np.int32)
        sK = np.zeros(64 * nelx * nely)
        
        for elx in range(nelx):
            for ely in range(nely):
                ed = edof(elx, ely)
                start = 64 * (ely + nely * elx)
                for i in range(8):
                    for j in range(8):
                        iK[start + 8*i + j] = ed[i]
                        jK[start + 8*i + j] = ed[j]
                        sK[start + 8*i + j] = (0.001 + x_phys[ely, elx]**penal) * Ke[i, j]
        
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
                c += (0.001 + x_phys[ely, elx]**penal) * ce
                dc[ely, elx] = -penal * x_phys[ely, elx]**(penal - 1) * ce
        
        dv = np.ones((nely, nelx))
        
        dcn = np.zeros((nely, nelx))
        for i in range(nelx):
            for j in range(nely):
                sum_ = 0.0
                for k in range(max(0, int(i - np.floor(rmin))), min(nelx, int(i + np.floor(rmin) + 1))):
                    for l in range(max(0, int(j - np.floor(rmin))), min(nely, int(j + np.floor(rmin) + 1))):
                        fac = rmin - np.sqrt((i - k)**2 + (j - l)**2)
                        if fac > 0:
                            sum_ += fac
                            dcn[j, i] += fac * x_phys[l, k] * dc[l, k]
                dcn[j, i] /= (x_phys[j, i] * sum_)
        dc = dcn
        
        l1 = 0.0
        l2 = 1e9
        move = 0.2
        xnew = np.zeros_like(x)
        while (l2 - l1) / (l1 + l2) > 1e-4:
            lmid = 0.5 * (l1 + l2)
            for i in range(nelx):
                for j in range(nely):
                    be = dc[j, i] / (dv[j, i] * lmid)
                    be = x[j, i] * np.sqrt(be)
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
        c_history.append(c)
        
        print(f"Iteration {loop:3d}: Compliance = {c:.4f}, Volume = {np.sum(x)/(nelx*nely):.4f}, Change = {change:.4f}")
    
    print("\nOptimization completed!")
    print(f"Final compliance: {c_history[-1]:.4f}")
    print(f"Final volume fraction: {np.sum(x)/(nelx*nely):.4f}")
    
    return x, c_history


if __name__ == "__main__":
    print("=" * 60)
    print("SIMP 拓扑优化 - 悬臂梁 (无GUI)")
    print("=" * 60)
    print()
    
    nelx = 30
    nely = 15
    volfrac = 0.5
    penal = 3.0
    rmin = 1.5
    
    print(f"计算域: {nelx} x {nely} 单元")
    print(f"体积约束: {volfrac}")
    print(f"SIMP惩罚因子: {penal}")
    print(f"滤波器半径: {rmin}")
    print()
    print("=" * 60)
    print("开始优化...")
    print("=" * 60)
    print()
    
    x_final, c_history = simp_topology_optimization(
        nelx=nelx,
        nely=nely,
        volfrac=volfrac,
        penal=penal,
        rmin=rmin,
        max_iter=20
    )
    
    print()
    print("=" * 60)
    print("优化完成!")
    print(f"最终柔顺度: {c_history[-1]:.4f}")
    print(f"最终体积分数: {np.sum(x_final)/(nelx*nely):.4f}")
    print("=" * 60)
