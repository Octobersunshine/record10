import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time


def tmm_fast(wavelengths, n_list, d_list, n0=1.0, ns=1.5):
    """快速TMM计算"""
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


class RobustFilterDemo:
    """快速演示：鲁棒性分析 + 蒙特卡洛"""

    def __init__(self):
        self.lc = 550
        self.bw = 40
        self.wl = np.linspace(self.lc - 2.5 * self.bw, self.lc + 2.5 * self.bw, 80)
        self.specs = {'min_T': 0.75, 'max_out_T': 0.15, 'fwhm_tol': 0.20}

    def loss(self, params, nl):
        n = np.clip(params[:nl], 1.3, 2.6)
        d = np.clip(params[nl:], 20, 400)
        _, T = tmm_fast(self.wl, n.tolist(), d.tolist())
        lb, ub = self.lc - self.bw / 2, self.lc + self.bw / 2
        inb = (self.wl >= lb) & (self.wl <= ub)
        out = ~inb
        return 0.35 * np.mean((1 - T[inb]) ** 2) + 0.4 * np.mean(T[out] ** 2) + 0.25 * np.mean(T[(self.wl >= lb - 4) & (self.wl <= lb + 4) | (self.wl >= ub - 4) & (self.wl <= ub + 4)] ** 2)

    def design(self, nl=6):
        print(f"设计滤波器: {nl}层膜, {self.lc}nm中心, {self.bw}nm带宽")
        print("-" * 50)

        qw = self.lc / 4.0
        x0 = np.zeros(2 * nl)
        for i in range(nl):
            x0[i] = 2.35 if i % 2 == 0 else 1.38
            x0[nl + i] = qw / x0[i]

        print("  PSO全局搜索...", end=" ")
        np.random.seed(42)
        n_pop, n_iter = 15, 40
        pop = np.tile(x0, (n_pop, 1)) + np.random.normal(0, 0.1, (n_pop, 2 * nl))
        vel = np.random.uniform(-0.2, 0.2, (n_pop, 2 * nl))
        pbest = pop.copy()
        pval = np.array([self.loss(p, nl) for p in pop])
        gbest = pbest[np.argmin(pval)]
        gval = np.min(pval)

        for it in range(n_iter):
            w = 0.9 - 0.5 * it / n_iter
            r1, r2 = np.random.random((n_pop, 2 * nl)), np.random.random((n_pop, 2 * nl))
            vel = w * vel + 2 * r1 * (pbest - pop) + 2 * r2 * (gbest - pop)
            vel = np.clip(vel, -0.3, 0.3)
            pop = pop + vel
            pop[:, :nl] = np.clip(pop[:, :nl], 1.3, 2.6)
            pop[:, nl:] = np.clip(pop[:, nl:], 20, 400)
            for p in range(n_pop):
                v = self.loss(pop[p], nl)
                if v < pval[p]:
                    pval[p], pbest[p] = v, pop[p]
                    if v < gval:
                        gval, gbest = v, pop[p].copy()
        print(f"完成, 损失={gval:.4f}")

        print("  L-BFGS-B微调...", end=" ")
        res = minimize(self.loss, gbest, args=(nl,), method='L-BFGS-B', options={'maxiter': 200})
        print(f"完成, 损失={res.fun:.4f}")

        n_opt = np.clip(res.x[:nl], 1.3, 2.6).tolist()
        d_opt = np.clip(res.x[nl:], 20, 400).tolist()
        return n_opt, d_opt

    def evaluate(self, n_list, d_list):
        R, T = tmm_fast(self.wl, n_list, d_list)
        lb, ub = self.lc - self.bw / 2, self.lc + self.bw / 2
        inb = (self.wl >= lb) & (self.wl <= ub)
        out = ~inb
        max_T, peak = np.max(T), self.wl[np.argmax(T)]
        avg_in, avg_out = np.mean(T[inb]), np.mean(T[out])
        T50 = max_T / 2
        abv = np.where(T >= T50)[0]
        fwhm = self.wl[abv[-1]] - self.wl[abv[0]] if len(abv) > 1 else 0
        ok = (abs(fwhm - self.bw) / self.bw <= self.specs['fwhm_tol'] and
              avg_in >= self.specs['min_T'] and avg_out <= self.specs['max_out_T'])
        return {'max_T': max_T, 'peak': peak, 'avg_in': avg_in, 'avg_out': avg_out,
                'fwhm': fwhm, 'ok': ok, 'R': R, 'T': T}

    def monte_carlo(self, n_list, d_list, err=0.02, n_sample=800):
        print(f"\n蒙特卡洛分析: ±{err*100:.0f}%膜厚误差, {n_sample}样本")
        print("-" * 50)

        nl = len(d_list)
        d_nom = np.array(d_list)
        passes = 0
        all_avg_in, all_avg_out, all_fwhm = [], [], []
        all_d_var = []

        for i in range(n_sample):
            d_var = d_nom * np.random.normal(1.0, err / 3, nl)
            all_d_var.append(d_var)
            m = self.evaluate(n_list, d_var.tolist())
            all_avg_in.append(m['avg_in'])
            all_avg_out.append(m['avg_out'])
            all_fwhm.append(m['fwhm'])
            if m['ok']:
                passes += 1
            if (i + 1) % 200 == 0:
                print(f"  完成 {i+1}/{n_sample}...")

        yld = passes / n_sample
        ci = 1.96 * np.sqrt(yld * (1 - yld) / n_sample)

        all_avg_in, all_avg_out, all_fwhm = map(np.array, [all_avg_in, all_avg_out, all_fwhm])
        all_d_var = np.array(all_d_var)

        print(f"\n  成品率: {yld*100:.1f}% (95% CI: {(yld-ci)*100:.1f}% - {(yld+ci)*100:.1f}%)")
        print(f"  通带透射率: {np.mean(all_avg_in)*100:.1f} ± {np.std(all_avg_in)*100:.1f}%")
        print(f"  带外透射率: {np.mean(all_avg_out)*100:.1f} ± {np.std(all_avg_out)*100:.1f}%")
        print(f"  半高宽: {np.mean(all_fwhm):.1f} ± {np.std(all_fwhm):.1f} nm")

        sens = np.zeros(nl)
        for i in range(nl):
            d_plus = d_nom.copy()
            d_plus[i] *= (1 + 0.01)
            m_plus = self.evaluate(n_list, d_plus.tolist())
            d_minus = d_nom.copy()
            d_minus[i] *= (1 - 0.01)
            m_minus = self.evaluate(n_list, d_minus.tolist())
            sens[i] = abs(m_plus['avg_in'] - m_minus['avg_in'])
        sens = sens / np.max(sens)

        print(f"\n  层灵敏度 (高→低):")
        for idx in np.argsort(sens)[::-1]:
            print(f"    层{idx+1}: {sens[idx]:.3f}")

        tol_rec = np.full(nl, 0.025)
        mean_s = np.mean(sens)
        tight, loose = [], []
        for i in range(nl):
            if sens[i] > mean_s + 0.3 * np.std(sens):
                tol_rec[i] = 0.015
                tight.append(i + 1)
            elif sens[i] < mean_s - 0.3 * np.std(sens):
                tol_rec[i] = 0.035
                loose.append(i + 1)

        print(f"\n  公差建议:")
        if tight:
            print(f"    严格控制层 {tight}: ±1.5%")
        if loose:
            print(f"    放宽控制层 {loose}: ±3.5%")
        print(f"    常规控制层: ±2.5%")

        return {
            'yield': yld, 'yield_ci': (yld - ci, yld + ci),
            'avg_in': all_avg_in, 'avg_out': all_avg_out, 'fwhm': all_fwhm,
            'sensitivity': sens, 'tolerance': tol_rec
        }

    def plot(self, n_list, d_list, metrics, mc_data):
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.25)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.wl, metrics['T'] * 100, 'b-', lw=2, label='Transmittance')
        ax1.plot(self.wl, metrics['R'] * 100, 'orange', lw=1.5, alpha=0.6, label='Reflectance')
        ax1.axvline(self.lc, color='k', ls=':', alpha=0.7)
        lb, ub = self.lc - self.bw / 2, self.lc + self.bw / 2
        ax1.axvspan(lb, ub, alpha=0.15, color='green', label='Passband')
        ax1.axhline(self.specs['min_T'] * 100, color='red', ls='--', alpha=0.5, label='Spec limit')
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('(%)')
        ax1.set_title(f'Filter Spectrum (λ₀={self.lc}nm, BW={self.bw}nm)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 105)

        ax2 = fig.add_subplot(gs[0, 1])
        x = np.arange(len(n_list))
        w = 0.35
        ax2.bar(x - w / 2, n_list, w, color='steelblue', label='n')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Refractive Index', color='steelblue')
        ax2.tick_params(axis='y', labelcolor='steelblue')
        ax2.set_ylim(1.0, 2.8)
        ax2_t = ax2.twinx()
        ax2_t.bar(x + w / 2, d_list, w, color='salmon', label='d (nm)')
        ax2_t.set_ylabel('Thickness (nm)', color='salmon')
        ax2_t.tick_params(axis='y', labelcolor='salmon')
        l1, la1 = ax2.get_legend_handles_labels()
        l2, la2 = ax2_t.get_legend_handles_labels()
        ax2.legend(l1 + l2, la1 + la2, loc='upper right')
        ax2.set_title('Thin Film Stack')
        ax2.set_xticks(x)
        ax2.set_xticklabels([str(i + 1) for i in x])
        ax2.grid(True, alpha=0.3, axis='y')

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(mc_data['avg_in'] * 100, bins=25, alpha=0.7, color='green', label='In-band T')
        ax3.hist(mc_data['avg_out'] * 100, bins=25, alpha=0.7, color='red', label='Out-of-band T')
        ax3.axvline(self.specs['min_T'] * 100, color='darkgreen', ls='--', lw=2, label='In-band spec')
        ax3.axvline(self.specs['max_out_T'] * 100, color='darkred', ls='--', lw=2, label='Out-band spec')
        ax3.set_xlabel('Transmittance (%)')
        ax3.set_ylabel('Count')
        ax3.set_title(f'Performance Distribution (Yield = {mc_data["yield"]*100:.1f}%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        ax4 = fig.add_subplot(gs[1, 1])
        colors = ['red' if s > np.mean(mc_data['sensitivity']) else 'steelblue'
                  for s in mc_data['sensitivity']]
        ax4.bar(x, mc_data['sensitivity'], color=colors, alpha=0.8)
        ax4.axhline(np.mean(mc_data['sensitivity']), color='k', ls='--', alpha=0.7, label='Mean')
        ax4.set_xlabel('Layer')
        ax4.set_ylabel('Normalized Sensitivity')
        ax4.set_title('Layer Sensitivity Analysis')
        ax4.set_xticks(x)
        ax4.set_xticklabels([str(i + 1) for i in x])
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        for i, tol in enumerate(mc_data['tolerance']):
            ax4.text(i, mc_data['sensitivity'][i] + 0.02, f'±{tol*100:.0f}%',
                    ha='center', fontsize=8, rotation=90)

        plt.tight_layout()
        plt.savefig('robust_demo_result.png', dpi=120, bbox_inches='tight')
        print(f"\n结果图已保存: robust_demo_result.png")
        plt.close()


def main():
    print("=" * 60)
    print("  薄膜滤波器鲁棒性分析演示")
    print("  蒙特卡洛模拟 + 灵敏度分析 + 公差设计")
    print("=" * 60)

    demo = RobustFilterDemo()
    n_list, d_list = demo.design(nl=6)

    print("\n优化结果:")
    print(f"{'层号':<6} {'折射率':<12} {'厚度(nm)':<12}")
    print("-" * 35)
    for i, (n, d) in enumerate(zip(n_list, d_list)):
        print(f"{i+1:<6} {n:<12.4f} {d:<12.2f}")

    metrics = demo.evaluate(n_list, d_list)
    print(f"\n标称性能:")
    print(f"  峰值透射率: {metrics['max_T']*100:.2f}% @ {metrics['peak']:.1f}nm")
    print(f"  通带平均: {metrics['avg_in']*100:.2f}%")
    print(f"  带外平均: {metrics['avg_out']*100:.2f}%")
    print(f"  半高宽: {metrics['fwhm']:.1f} nm")
    print(f"  满足规格: {'是' if metrics['ok'] else '否'}")

    mc_data = demo.monte_carlo(n_list, d_list, err=0.02, n_sample=800)
    demo.plot(n_list, d_list, metrics, mc_data)

    print("\n" + "=" * 60)
    print("  分析完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
