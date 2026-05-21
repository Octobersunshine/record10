import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


class SIMPTopologyOptimization:
    def __init__(self, nelx=60, nely=30, volfrac=0.5, penal=3.0, rmin=1.5):
        self.nelx = nelx
        self.nely = nely
        self.volfrac = volfrac
        self.penal = penal
        self.rmin = rmin
        self.dc = np.zeros((nely, nelx))
        self.dv = np.ones((nely, nelx))
        self.x = volfrac * np.ones((nely, nelx))
        self.x_phys = self.x.copy()

    def lk(self):
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

    def edof_1(self, x, y):
        n1 = (self.nely + 1) * x + y
        n2 = (self.nely + 1) * (x + 1) + y
        edof = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3])
        return edof

    def fe(self):
        Ke = self.lk()
        ndof = 2 * (self.nelx + 1) * (self.nely + 1)
        iK = np.zeros(64 * self.nelx * self.nely)
        jK = np.zeros(64 * self.nelx * self.nely)
        sK = np.zeros(64 * self.nelx * self.nely)
        for elx in range(self.nelx):
            for ely in range(self.nely):
                edof = self.edof_1(elx, ely)
                start = 64 * (ely + self.nely * elx)
                for i in range(8):
                    for j in range(8):
                        iK[start + 8 * i + j] = edof[i]
                        jK[start + 8 * i + j] = edof[j]
                        sK[start + 8 * i + j] = (0.001 + self.x_phys[ely, elx]**self.penal) * Ke[i, j]
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof))
        K = (K + K.T) / 2.0

        F = np.zeros(ndof)
        F[2 * (self.nely + 1) - 1] = -1.0

        U = np.zeros(ndof)
        fixed = np.union1d(np.arange(0, 2*(self.nely+1), 2), np.arange(1, 2*(self.nely+1), 2))
        free = np.setdiff1d(np.arange(ndof), fixed)
        U[free] = spsolve(K[free][:, free], F[free])
        return U

    def oc(self):
        l1 = 0.0
        l2 = 1e9
        move = 0.2
        xnew = np.zeros_like(self.x)
        while (l2 - l1) / (l1 + l2) > 1e-4:
            lmid = 0.5 * (l1 + l2)
            for elx in range(self.nelx):
                for ely in range(self.nely):
                    be = self.dc[ely, elx] / (self.dv[ely, elx] * lmid)
                    be = self.x[ely, elx] * np.sqrt(be)
                    if be < max(0.0, self.x[ely, elx] - move):
                        xnew[ely, elx] = max(0.0, self.x[ely, elx] - move)
                    elif be > min(1.0, self.x[ely, elx] + move):
                        xnew[ely, elx] = min(1.0, self.x[ely, elx] + move)
                    else:
                        xnew[ely, elx] = be
            if np.sum(xnew) - self.volfrac * self.nelx * self.nely > 0:
                l1 = lmid
            else:
                l2 = lmid
        self.x = xnew.copy()

    def filter(self):
        rmin = self.rmin
        dcn = np.zeros_like(self.dc)
        for i in range(self.nelx):
            for j in range(self.nely):
                sum_ = 0.0
                for k in range(max(0, int(i - np.floor(rmin))), min(self.nelx, int(i + np.floor(rmin) + 1))):
                    for l in range(max(0, int(j - np.floor(rmin))), min(self.nely, int(j + np.floor(rmin) + 1))):
                        fac = rmin - np.sqrt((i - k)**2 + (j - l)**2)
                        if fac > 0:
                            sum_ += fac
                            dcn[j, i] += fac * self.x_phys[l, k] * self.dc[l, k]
                dcn[j, i] /= (self.x_phys[j, i] * sum_)
        self.dc = dcn

    def optimize(self, loop_max=100, plot_interval=10):
        Ke = self.lk()
        loop = 0
        change = 1.0
        c_values = []
        
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        while change > 0.01 and loop < loop_max:
            loop += 1
            self.x_phys = self.x.copy()
            
            U = self.fe()
            
            c = 0.0
            self.dc.fill(0)
            
            for elx in range(self.nelx):
                for ely in range(self.nely):
                    edof = self.edof_1(elx, ely)
                    Ue = U[edof]
                    ce = np.dot(np.dot(Ue.T, Ke), Ue)
                    c += (0.001 + self.x_phys[ely, elx]**self.penal) * ce
                    self.dc[ely, elx] = -self.penal * self.x_phys[ely, elx]**(self.penal - 1) * ce
            
            self.dv = np.ones_like(self.x)
            
            self.filter()
            self.oc()
            
            change = np.max(np.abs(self.x - self.x_phys))
            c_values.append(c)
            
            if loop % plot_interval == 0 or loop == 1:
                print(f"Loop: {loop}, Compliance: {c:.4f}, Volume: {np.sum(self.x)/(self.nelx*self.nely):.4f}, Change: {change:.4f}")
                ax.clear()
                ax.imshow(1 - self.x_phys, cmap='gray', origin='upper')
                ax.set_title(f'SIMP Topology Optimization\nLoop {loop}, Compliance {c:.2f}')
                ax.set_xticks([])
                ax.set_yticks([])
                plt.pause(0.1)
        
        plt.ioff()
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.clear()
        ax2.imshow(1 - self.x_phys, cmap='gray', origin='upper')
        ax2.set_title(f'Final Design\nCompliance: {c:.4f}, Volume Fraction: {np.sum(self.x)/(self.nelx*self.nely):.4f}')
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.show()
        
        return self.x_phys, c_values


def main():
    print("=" * 60)
    print("SIMP Topology Optimization - Cantilever Beam")
    print("=" * 60)
    
    nelx = 60
    nely = 30
    volfrac = 0.5
    penal = 3.0
    rmin = 1.5
    
    print(f"Domain: {nelx} x {nely} elements")
    print(f"Volume Fraction: {volfrac}")
    print(f"SIMP Penalization: {penal}")
    print(f"Filter Radius: {rmin}")
    print("=" * 60)
    
    optimizer = SIMPTopologyOptimization(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal, rmin=rmin)
    x_final, c_history = optimizer.optimize(loop_max=100, plot_interval=5)
    
    print("=" * 60)
    print("Optimization completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
