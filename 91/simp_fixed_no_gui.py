import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def simp_topology_optimization(nelx=40, nely=20, volfrac=0.5, penal=3.0, rmin=2.0, max_iter=80,
                                use_projection=True, beta=8.0, eta=0.5, use_gray_penalty=True, gray_weight=0.1):
    """
    修复棋盘格问题的 SIMP 拓扑优化算法 (无GUI版)
    
    核心改进:
    1. 增大默认滤波半径 rmin=2.0 (原1.5)
    2. 标准灵敏度滤波 - 同时滤波 dc 和 dv
    3. Heaviside投影滤波 - 边界锐化，抑制数值振荡
    4. 灰度惩罚项 - 4*x*(1-x) 惩罚中间密度
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
    
    def heaviside_projection(x, beta, eta):
        numerator = np.tanh(beta * eta) + np.tanh(beta * (x - eta))
        denominator = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
        return numerator / denominator
    
    x = volfrac * np.ones((nely, nelx))
    x_phys = x.copy()
    loop = 0
    change = 1.0
    
    print("=" * 70)
    print("SIMP 拓扑优化 - 棋盘格修复版")
    print("=" * 70)
    print(f"计算域: {nelx} x {nely} 单元")
    print(f"体积约束: {volfrac}")
    print(f"SIMP惩罚因子: {penal}")
    print(f"滤波器半径: {rmin}")
    print(f"Heaviside投影: {'开启' if use_projection else '关闭'}, beta={beta}")
    print(f"灰度惩罚: {'开启' if use_gray_penalty else '关闭'}, weight={gray_weight}")
    print("=" * 70)
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
                for i in range(8):
                    for j in range(8):
                        iK[start + 8*i + j] = ed[i]
                        jK[start + 8*i + j] = ed[j]
                        sK[start + 8*i + j] = (1e-9 + x_phys[ely, elx]**penal) * Ke[i, j]
        
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
                c += (1e-9 + x_phys[ely, elx]**penal) * ce
                dc[ely, elx] = -penal * x_phys[ely, elx]**(penal - 1) * ce
        
        if use_projection:
            dproj = beta * (1 - np.tanh(beta * (x - eta))**2) / (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))
            dc = dc * dproj
        
        if use_gray_penalty:
            dc_gray = 4 * gray_weight * (1 - 2 * x_phys) / (nelx * nely)
            dc = dc + dc_gray
        
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
                            sum_dc += fac * dc[l, k]
                            sum_dv += fac * dv[l, k]
                dcn[j, i] = sum_dc / sum_
                dvy[j, i] = sum_dv / sum_
        dc = dcn
        dv = dvy
        
        l1 = 0.0
        l2 = 1e9
        move = 0.15
        xnew = np.zeros_like(x)
        while (l2 - l1) / (l1 + l2) > 1e-7:
            lmid = 0.5 * (l1 + l2)
            for i in range(nelx):
                for j in range(nely):
                    if dc[j, i] < 0:
                        be = (-dc[j, i] / (dv[j, i] * lmid))**(1/(penal-1))
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
        
        if loop % 5 == 0 or loop == 1 or loop == max_iter:
            gray_ratio = np.sum((x_phys > 0.05) & (x_phys < 0.95)) / (nelx * nely)
            print(f"Iter {loop:3d} | Compliance={c:.4f} | Volume={np.sum(x)/(nelx*nely):.4f} | "
                  f"Change={change:.4f} | Gray={gray_ratio:.3f}")
    
    print()
    print("=" * 70)
    print("优化完成!")
    print(f"最终柔顺度: {c:.4f}")
    print(f"最终体积分数: {np.sum(x)/(nelx*nely):.4f}")
    print(f"中间密度单元比例: {gray_ratio:.4f}")
    print("=" * 70)
    
    return x_phys


def diagnose_checkerboard(x):
    """诊断棋盘格现象"""
    nely, nelx = x.shape
    checker_count = 0
    total = 0
    
    for i in range(1, nelx-1):
        for j in range(1, nely-1):
            neighbors = [x[j-1,i], x[j+1,i], x[j,i-1], x[j,i+1]]
            mean_neighbor = np.mean(neighbors)
            if (x[j,i] > 0.9 and mean_neighbor < 0.3) or (x[j,i] < 0.1 and mean_neighbor > 0.7):
                checker_count += 1
            total += 1
    
    checker_ratio = checker_count / total if total > 0 else 0
    print(f"\n棋盘格诊断:")
    print(f"  疑似棋盘格单元: {checker_count}")
    print(f"  棋盘格比例: {checker_ratio:.4f}")
    
    return checker_ratio


if __name__ == "__main__":
    print()
    print("测试修复版本...")
    print()
    x_fixed = simp_topology_optimization(
        nelx=40, nely=20, volfrac=0.5, penal=3.0, rmin=2.0,
        max_iter=60, use_projection=True, beta=8.0, use_gray_penalty=True
    )
    
    diagnose_checkerboard(x_fixed)
    
    print()
    print("=" * 70)
    print("测试无修复版本（对比）...")
    print("=" * 70)
    print()
    
    x_original = simp_topology_optimization(
        nelx=40, nely=20, volfrac=0.5, penal=3.0, rmin=1.2,
        max_iter=60, use_projection=False, use_gray_penalty=False
    )
    
    diagnose_checkerboard(x_original)
