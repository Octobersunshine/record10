import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time


def tmm(wavelengths, n_list, d_list, n0=1.0, ns=1.5):
    R = np.zeros_like(wavelengths, dtype=float)
    T = np.zeros_like(wavelengths, dtype=float)
    for idx, lambda0 in enumerate(wavelengths):
        k0 = 2 * np.pi / lambda0
        M = np.eye(2, dtype=complex)
        for n, d in zip(n_list, d_list):
            delta = k0 * n * d
            eta = n
            M = M @ np.array([
                [np.cos(delta), -1j * np.sin(delta) / eta],
                [-1j * eta * np.sin(delta), np.cos(delta)]
            ])
        A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        denom = A * ns + B * n0 * ns + C + D * n0
        r = (A * ns + B * n0 * ns - C - D * n0) / denom
        t = 2 * ns / denom
        R[idx] = np.real(np.abs(r) ** 2)
        T[idx] = np.real(np.abs(t) ** 2 * n0 / ns)
    return R, T


class GlobalFilter:
    def __init__(self):
        self.wl = None
        self.target = None
        self.lc = None
        self.bw = None

    def loss(self, params, nl):
        n = np.clip(params[:nl], 1.3, 2.6)
        d = np.clip(params[nl:], 20, 400)
        _, T = tmm(self.wl, n.tolist(), d.tolist())
        lb, ub = self.lc - self.bw / 2, self.lc + self.bw / 2
        inb = (self.wl >= lb) & (self.wl <= ub)
        out = ~inb
        in_loss = np.mean((1.0 - T[inb]) ** 2)
        out_loss = np.mean(T[out] ** 2)
        ew = self.bw * 0.1
        le = (self.wl >= lb - ew) & (self.wl <= lb + ew)
        re = (self.wl >= ub - ew) & (self.wl <= ub + ew)
        edge = np.mean(T[le | re] ** 2) if np.any(le | re) else 0.0
        return 0.35 * in_loss + 0.40 * out_loss + 0.25 * edge

    def pso(self, nl, npop=20, niter=50):
        ndim = 2 * nl
        qw = self.lc / 4.0
        pop = np.zeros((npop, ndim))
        for p in range(npop):
            for i in range(nl):
                base = 2.35 if i % 2 == 0 else 1.38
                pop[p, i] = base + np.random.uniform(-0.2, 0.2)
                pop[p, nl + i] = qw / pop[p, i] + np.random.uniform(-10, 10)
        pop[:, :nl] = np.clip(pop[:, :nl], 1.3, 2.6)
        pop[:, nl:] = np.clip(pop[:, nl:], 20, 400)
        vel = np.random.uniform(-0.3, 0.3, (npop, ndim))
        pbest = pop.copy()
        pval = np.array([self.loss(p, nl) for p in pop])
        gi = np.argmin(pval)
        gbest, gval = pbest[gi].copy(), pval[gi]
        hist = [gval]
        w, c1, c2 = 0.6, 1.8, 1.8
        for it in range(niter):
            r1, r2 = np.random.random((npop, ndim)), np.random.random((npop, ndim))
            vel = w * vel + c1 * r1 * (pbest - pop) + c2 * r2 * (gbest - pop)
            vel = np.clip(vel, -0.5, 0.5)
            pop = pop + vel
            pop[:, :nl] = np.clip(pop[:, :nl], 1.3, 2.6)
            pop[:, nl:] = np.clip(pop[:, nl:], 20, 400)
            for p in range(npop):
                v = self.loss(pop[p], nl)
                if v < pval[p]:
                    pval[p], pbest[p] = v, pop[p].copy()
                    if v < gval:
                        gval, gbest = v, pop[p].copy()
            hist.append(gval)
        return gbest, hist

    def sa(self, nl, niter=200):
        ndim = 2 * nl
        qw = self.lc / 4.0
        cur = np.zeros(ndim)
        for i in range(nl):
            cur[i] = 2.35 if i % 2 == 0 else 1.38
            cur[nl + i] = qw / cur[i]
        cl = self.loss(cur, nl)
        best, bl = cur.copy(), cl
        hist = [bl]
        T0, Tf = 30.0, 0.01
        for it in range(niter):
            temp = T0 * (Tf / T0) ** (it / niter)
            cand = cur + np.random.normal(0, 0.12 * temp / T0, ndim)
            cand[:nl] = np.clip(cand[:nl], 1.3, 2.6)
            cand[nl:] = np.clip(cand[nl:], 20, 400)
            cl2 = self.loss(cand, nl)
            if cl2 < cl or np.random.random() < np.exp(-(cl2 - cl) / max(temp, 1e-10)):
                cur, cl = cand, cl2
                if cl < bl:
                    best, bl = cur.copy(), cl
            hist.append(bl)
        return best, hist

    def design(self, lc, bw, nl=6):
        self.lc, self.bw = lc, bw
        self.wl = np.linspace(lc - 2.5 * bw, lc + 2.5 * bw, 100)
        sigma = bw / (2 * np.sqrt(2 * np.log(2)))
        self.target = np.exp(-((self.wl - lc) ** 2) / (2 * sigma ** 2))

        print(f"Design: λ₀={lc}nm, BW={bw}nm, Layers={nl}")
        start = time.time()

        print("  PSO...", end=" ", flush=True)
        pso_b, pso_h = self.pso(nl, npop=15, niter=40)
        pso_l = self.loss(pso_b, nl)
        print(f"loss={pso_l:.4f}")

        print("  SA...", end=" ", flush=True)
        sa_b, sa_h = self.sa(nl, niter=150)
        sa_l = self.loss(sa_b, nl)
        print(f"loss={sa_l:.4f}")

        best = pso_b if pso_l < sa_l else sa_b
        bl = min(pso_l, sa_l)

        print("  L-BFGS-B...", end=" ", flush=True)
        res = minimize(self.loss, best, args=(nl,), method='L-BFGS-B', options={'maxiter': 200})
        if res.fun < bl:
            best, bl = res.x, res.fun
        print(f"final loss={bl:.4f}")

        print(f"  Time: {time.time() - start:.1f}s")

        n_opt = np.clip(best[:nl], 1.3, 2.6).tolist()
        d_opt = np.clip(best[nl:], 20, 400).tolist()
        R, T = tmm(self.wl, n_opt, d_opt)

        lb, ub = lc - bw / 2, lc + bw / 2
        inb = (self.wl >= lb) & (self.wl <= ub)
        out = ~inb
        max_T = np.max(T)
        peak = self.wl[np.argmax(T)]
        avg_in = np.mean(T[inb])
        avg_out = np.mean(T[out])
        T50 = max_T / 2
        abv = np.where(T >= T50)[0]
        fwhm = self.wl[abv[-1]] - self.wl[abv[0]] if len(abv) > 1 else 0
        rej = 10 * np.log10(avg_in / (avg_out + 1e-12))

        print(f"\n  Peak T: {max_T*100:.1f}% @ {peak:.1f}nm")
        print(f"  Avg in-band: {avg_in*100:.1f}%")
        print(f"  Avg out-of-band: {avg_out*100:.1f}%")
        print(f"  FWHM: {fwhm:.1f}nm (target: {bw}nm)")
        print(f"  Rejection: {rej:.1f}dB")

        return {'n': n_opt, 'd': d_opt, 'T': T, 'R': R, 'peak_T': max_T,
                'avg_in': avg_in, 'avg_out': avg_out, 'fwhm': fwhm, 'rej': rej,
                'hist': pso_h + sa_h + [bl]}


def plot_result(gf, res, lc, bw, fn='test_global.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(gf.wl, res['T']*100, 'b-', lw=2, label='T')
    axes[0].plot(gf.wl, gf.target*100, 'r--', lw=1.5, label='Target')
    axes[0].plot(gf.wl, res['R']*100, 'orange', lw=1.5, alpha=0.6, label='R')
    axes[0].axvline(lc, color='k', ls=':', alpha=0.7)
    lb, ub = lc - bw/2, lc + bw/2
    axes[0].axvspan(lb, ub, alpha=0.15, color='green', label='Passband')
    axes[0].set_xlabel('Wavelength (nm)')
    axes[0].set_ylabel('(%)')
    axes[0].set_title(f'Global Opt Filter (λ₀={lc}nm, BW={bw}nm)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 105)

    x = np.arange(len(res['n']))
    w = 0.35
    ax1 = axes[1]
    ax1.bar(x - w/2, res['n'], w, color='steelblue', label='n')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Refractive Index', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_ylim(1.0, 2.8)
    ax2 = ax1.twinx()
    ax2.bar(x + w/2, res['d'], w, color='salmon', label='d (nm)')
    ax2.set_ylabel('Thickness (nm)', color='salmon')
    ax2.tick_params(axis='y', labelcolor='salmon')
    l1, la1 = ax1.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, loc='upper right')
    axes[1].set_title('Thin Film Stack')
    axes[1].set_xticks(x)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(fn, dpi=120, bbox_inches='tight')
    print(f"\nSaved: {fn}")
    plt.close()


def main():
    print("=" * 50)
    print("Global Optimization Test")
    print("PSO + SA + L-BFGS-B")
    print("=" * 50)

    gf = GlobalFilter()
    res = gf.design(lc=550, bw=40, nl=6)
    plot_result(gf, res, 550, 40)

    print("\nDone!")


if __name__ == "__main__":
    main()
