import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def simp_topology_optimization_fixed(nelx=60, nely=30, volfrac=0.5, penal=3.0, rmin=2.0, max_iter=100, 
                                     use_projection=True, beta=8.0, eta=0.5, use_gray_penalty=True, gray_weight=0.1):
    """
    修复棋盘格问题的 SIMP 拓扑优化算法
    
    改进措施:
    1. 标准灵敏度滤波 (Sigmund 2001) - 基础去棋盘格
    2. Heaviside投影滤波 - 边界更清晰，抑制数值振荡
    3. 灰度惩罚项 - 进一步抑制中间密度
    
    参数:
        nelx, nely: x,y方向单元数
        volfrac: 体积约束
        penal: SIMP惩罚因子
        rmin: 滤波器半径 (默认增大到2.0)
        max_iter: 最大迭代次数
        use_projection: 是否使用Heaviside投影滤波
        beta: Heaviside投影参数 (越大越锐利)
        eta: 投影阈值
        use_gray_penalty: 是否使用灰度惩罚
        gray_weight: 灰度惩罚权重
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
        """Heaviside投影函数 - 使边界更锐利"""
        numerator = np.tanh(beta * eta) + np.tanh(beta * (x - eta))
        denominator = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
        return numerator / denominator
    
    x = volfrac * np.ones((nely, nelx))
    x_phys = x.copy()
    loop = 0
    change = 1.0
    c_history = []
    gray_history = []
    
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()
    
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
            gray_measure = np.sum(4 * x_phys * (1 - x_phys)) / (nelx * nely)
            gray_history.append(gray_measure)
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
        c_history.append(c)
        
        if loop % 5 == 0 or loop == 1:
            gray_ratio = np.sum((x_phys > 0.05) & (x_phys < 0.95)) / (nelx * nely)
            print(f"Iter {loop:3d}: Compliance={c:.4f}, Volume={np.sum(x)/(nelx*nely):.4f}, "
                  f"Change={change:.4f}, Gray={gray_ratio:.3f}")
            
            ax1.clear()
            ax1.imshow(1 - x_phys, cmap='gray', origin='upper', vmin=0, vmax=1)
            ax1.set_title(f'Topology (Iter {loop})')
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            ax2.clear()
            ax2.plot(c_history, 'b-', linewidth=2)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Compliance')
            ax2.set_title('Convergence History')
            ax2.grid(True, alpha=0.3)
            
            ax3.clear()
            density_hist = x_phys.flatten()
            ax3.hist(density_hist, bins=20, range=(0, 1), edgecolor='black', alpha=0.7)
            ax3.set_xlabel('Density')
            ax3.set_ylabel('Count')
            ax3.set_title('Density Distribution')
            ax3.grid(True, alpha=0.3)
            
            ax4.clear()
            ax4.imshow(np.abs(x - x_phys), cmap='hot', origin='upper')
            ax4.set_title('Density Change')
            ax4.set_xticks([])
            ax4.set_yticks([])
            
            plt.tight_layout()
            plt.pause(0.05)
    
    plt.ioff()
    
    fig_final, axes_final = plt.subplots(2, 2, figsize=(14, 10))
    af1, af2, af3, af4 = axes_final.flatten()
    
    af1.imshow(1 - x_phys, cmap='gray', origin='upper', vmin=0, vmax=1)
    af1.set_title(f'Final Topology\nCompliance={c:.4f}, Volume={np.sum(x)/(nelx*nely):.4f}')
    af1.set_xticks([])
    af1.set_yticks([])
    
    af2.plot(c_history, 'b-', linewidth=2)
    af2.set_xlabel('Iteration')
    af2.set_ylabel('Compliance')
    af2.set_title('Convergence History')
    af2.grid(True, alpha=0.3)
    
    density_hist = x_phys.flatten()
    af3.hist(density_hist, bins=20, range=(0, 1), edgecolor='black', alpha=0.7)
    af3.set_xlabel('Density')
    af3.set_ylabel('Count')
    gray_ratio = np.sum((x_phys > 0.05) & (x_phys < 0.95)) / (nelx * nely)
    af3.set_title(f'Density Distribution\nGray elements: {gray_ratio:.3f}')
    af3.grid(True, alpha=0.3)
    
    if use_gray_penalty and gray_history:
        af4.plot(gray_history, 'r-', linewidth=2)
        af4.set_xlabel('Iteration')
        af4.set_ylabel('Gray Measure')
        af4.set_title('Gray Element History')
        af4.grid(True, alpha=0.3)
    else:
        af4.imshow(x_phys, cmap='viridis', origin='upper')
        af4.set_title('Density Heatmap')
        af4.set_xticks([])
        af4.set_yticks([])
    
    plt.tight_layout()
    plt.show()
    
    print()
    print("=" * 70)
    print("优化完成!")
    print(f"最终柔顺度: {c_history[-1]:.4f}")
    print(f"最终体积分数: {np.sum(x)/(nelx*nely):.4f}")
    print(f"中间密度单元比例: {gray_ratio:.4f}")
    print("=" * 70)
    
    return x_phys, c_history


def example_comparison():
    """对比有/无棋盘格修复的效果"""
    print("=" * 70)
    print("棋盘格修复对比测试")
    print("=" * 70)
    print()
    
    print("运行修复版本...")
    print()
    
    x_fixed, c_fixed = simp_topology_optimization_fixed(
        nelx=40, nely=20, volfrac=0.5, penal=3.0, rmin=2.0, 
        max_iter=80, use_projection=True, beta=8.0, use_gray_penalty=True
    )
    
    return x_fixed, c_fixed


if __name__ == "__main__":
    example_comparison()
