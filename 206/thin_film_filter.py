import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class ThinFilmFilter:
    def __init__(self, n_substrate: float = 1.5, n_incident: float = 1.0):
        self.n_substrate = n_substrate
        self.n_incident = n_incident

    def tmm(self, wavelengths: np.ndarray, n_list: List[float], d_list: List[float],
            theta: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        传递矩阵法计算多层薄膜的反射率和透射率
        
        参数:
            wavelengths: 波长数组 (nm)
            n_list: 每层折射率列表
            d_list: 每层厚度列表 (nm)
            theta: 入射角 (弧度)
            
        返回:
            R: 反射率数组
            T: 透射率数组
        """
        n_layers = len(n_list)
        if n_layers != len(d_list):
            raise ValueError("折射率列表和厚度列表长度必须相同")

        R = np.zeros_like(wavelengths, dtype=float)
        T = np.zeros_like(wavelengths, dtype=float)

        n0 = self.n_incident
        ns = self.n_substrate

        cos_theta0 = np.cos(theta)

        for idx, lambda0 in enumerate(wavelengths):
            k0 = 2 * np.pi / lambda0

            M = np.eye(2, dtype=complex)

            for n, d in zip(n_list, d_list):
                sin_theta = n0 * np.sin(theta) / n
                cos_theta = np.sqrt(1 - sin_theta ** 2 + 0j)
                delta = k0 * n * d * cos_theta
                eta = n * cos_theta

                m11 = np.cos(delta)
                m12 = -1j * np.sin(delta) / eta
                m21 = -1j * eta * np.sin(delta)
                m22 = np.cos(delta)
                layer_matrix = np.array([[m11, m12], [m21, m22]])

                M = M @ layer_matrix

            sin_theta_s = n0 * np.sin(theta) / ns
            cos_theta_s = np.sqrt(1 - sin_theta_s ** 2 + 0j)
            eta_s = ns * cos_theta_s
            eta0 = n0 * cos_theta0

            A = M[0, 0]
            B = M[0, 1]
            C = M[1, 0]
            D = M[1, 1]

            r = (A * eta_s + B * eta0 * eta_s - C - D * eta0) / (A * eta_s + B * eta0 * eta_s + C + D * eta0)
            t = 2 * eta_s / (A * eta_s + B * eta0 * eta_s + C + D * eta0)

            R[idx] = np.real(np.abs(r) ** 2)
            T[idx] = np.real(np.abs(t) ** 2 * eta0 / eta_s)

        return R, T

    def design_bandpass(self, lambda_center: float, bandwidth: float,
                        n_layers: int = 10,
                        lambda_min: Optional[float] = None,
                        lambda_max: Optional[float] = None,
                        n_points: int = 500) -> Tuple[List[float], List[float]]:
        """
        设计带通滤波器
        
        参数:
            lambda_center: 中心波长 (nm)
            bandwidth: 带宽 (nm)
            n_layers: 薄膜层数
            lambda_min: 最小波长 (nm)，默认为 lambda_center - 2*bandwidth
            lambda_max: 最大波长 (nm)，默认为 lambda_center + 2*bandwidth
            n_points: 波长采样点数
            
        返回:
            n_list: 优化后的折射率列表
            d_list: 优化后的厚度列表 (nm)
        """
        if lambda_min is None:
            lambda_min = lambda_center - 2 * bandwidth
        if lambda_max is None:
            lambda_max = lambda_center + 2 * bandwidth

        self.lambda_center = lambda_center
        self.bandwidth = bandwidth
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.wavelengths = np.linspace(lambda_min, lambda_max, n_points)

        def target_spectrum(wavelengths, lambda_c, bw):
            sigma = bw / (2 * np.sqrt(2 * np.log(2)))
            return np.exp(-((wavelengths - lambda_c) ** 2) / (2 * sigma ** 2))

        self.target_T = target_spectrum(self.wavelengths, lambda_center, bandwidth)

        n_low = 1.38
        n_high = 2.35

        n_initial = []
        d_initial = []
        quarter_wave = lambda_center / 4.0

        for i in range(n_layers):
            if i % 2 == 0:
                n_initial.append(n_high + np.random.uniform(-0.1, 0.1))
            else:
                n_initial.append(n_low + np.random.uniform(-0.05, 0.05))
            d_initial.append(quarter_wave / n_initial[-1] * (0.8 + np.random.uniform(-0.2, 0.2)))

        def loss(params):
            n_opt = params[:n_layers]
            d_opt = params[n_layers:]

            n_opt = np.clip(n_opt, 1.3, 2.5)
            d_opt = np.clip(d_opt, 10, 500)

            _, T = self.tmm(self.wavelengths, n_opt.tolist(), d_opt.tolist())

            mse = np.mean((T - self.target_T) ** 2)

            lambda_low = lambda_center - bandwidth / 2
            lambda_high = lambda_center + bandwidth / 2
            in_band = (self.wavelengths >= lambda_low) & (self.wavelengths <= lambda_high)
            out_band = ~in_band

            in_band_loss = np.mean((1.0 - T[in_band]) ** 2)
            out_band_loss = np.mean(T[out_band] ** 2)

            total_loss = 0.3 * mse + 0.4 * in_band_loss + 0.3 * out_band_loss

            return total_loss

        x0 = np.concatenate([n_initial, d_initial])

        print(f"开始优化，层数: {n_layers}，中心波长: {lambda_center} nm，带宽: {bandwidth} nm")
        print(f"初始损失值: {loss(x0):.6f}")

        result = minimize(loss, x0, method='L-BFGS-B',
                          options={'maxiter': 2000, 'iprint': 1})

        n_opt = np.clip(result.x[:n_layers], 1.3, 2.5).tolist()
        d_opt = np.clip(result.x[n_layers:], 10, 500).tolist()

        print(f"优化完成，最终损失值: {result.fun:.6f}")

        return n_opt, d_opt

    def evaluate(self, n_list: List[float], d_list: List[float],
                 plot: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        评估滤波器性能
        
        参数:
            n_list: 折射率列表
            d_list: 厚度列表 (nm)
            plot: 是否绘制图表
            
        返回:
            wavelengths: 波长数组
            R: 反射率数组
            T: 透射率数组
        """
        R, T = self.tmm(self.wavelengths, n_list, d_list)

        lambda_low = self.lambda_center - self.bandwidth / 2
        lambda_high = self.lambda_center + self.bandwidth / 2
        in_band = (self.wavelengths >= lambda_low) & (self.wavelengths <= lambda_high)
        out_band = ~in_band

        max_T = np.max(T)
        avg_T_in = np.mean(T[in_band])
        avg_T_out = np.mean(T[out_band])

        T_50 = max_T / 2
        above_50 = np.where(T >= T_50)[0]
        if len(above_50) > 1:
            fwhm = self.wavelengths[above_50[-1]] - self.wavelengths[above_50[0]]
        else:
            fwhm = 0

        print(f"\n滤波器性能评估:")
        print(f"  最大透射率: {max_T * 100:.2f}%")
        print(f"  通带平均透射率: {avg_T_in * 100:.2f}%")
        print(f"  阻带平均透射率: {avg_T_out * 100:.2f}%")
        print(f"  实际半高宽: {fwhm:.1f} nm")
        print(f"  设计半高宽: {self.bandwidth:.1f} nm")
        print(f"  抑制比: {avg_T_in / (avg_T_out + 1e-10):.1f} dB")

        if plot:
            self._plot_spectrum(self.wavelengths, R, T, n_list, d_list)

        return self.wavelengths, R, T

    def _plot_spectrum(self, wavelengths, R, T, n_list, d_list):
        """绘制光谱图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].plot(wavelengths, T * 100, 'b-', linewidth=2, label='透射率')
        axes[0].plot(wavelengths, self.target_T * 100, 'r--', linewidth=1.5, label='目标透射率')
        axes[0].axvline(self.lambda_center, color='k', linestyle=':', alpha=0.7, label='中心波长')
        axes[0].axvspan(self.lambda_center - self.bandwidth / 2,
                        self.lambda_center + self.bandwidth / 2,
                        alpha=0.2, color='green', label='通带')
        axes[0].set_xlabel('波长 (nm)', fontsize=12)
        axes[0].set_ylabel('透射率 (%)', fontsize=12)
        axes[0].set_title(f'带通滤波器光谱 (中心: {self.lambda_center} nm, 带宽: {self.bandwidth} nm)',
                          fontsize=13)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 105)

        axes[0].plot(wavelengths, R * 100, 'orange', linewidth=1.5, alpha=0.6, label='反射率')
        axes[0].legend(fontsize=10)

        layer_indices = np.arange(len(n_list))
        bar_width = 0.35

        ax1 = axes[1]
        bars1 = ax1.bar(layer_indices - bar_width / 2, n_list, bar_width,
                        color='steelblue', label='折射率')
        ax1.set_xlabel('膜层编号', fontsize=12)
        ax1.set_ylabel('折射率', fontsize=12, color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1.set_ylim(1.0, 2.8)

        ax2 = ax1.twinx()
        bars2 = ax2.bar(layer_indices + bar_width / 2, d_list, bar_width,
                        color='salmon', label='厚度 (nm)')
        ax2.set_ylabel('厚度 (nm)', fontsize=12, color='salmon')
        ax2.tick_params(axis='y', labelcolor='salmon')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

        axes[1].set_title('膜系参数', fontsize=13)
        axes[1].set_xticks(layer_indices)
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('bandpass_filter_result.png', dpi=150, bbox_inches='tight')
        print("\n结果图已保存为 'bandpass_filter_result.png'")
        plt.show()


def main():
    filter = ThinFilmFilter(n_substrate=1.5, n_incident=1.0)

    lambda_center = 550.0
    bandwidth = 50.0
    n_layers = 12

    print("=" * 60)
    print("薄膜光学滤波器设计 - 传递矩阵法优化")
    print("=" * 60)

    n_list, d_list = filter.design_bandpass(
        lambda_center=lambda_center,
        bandwidth=bandwidth,
        n_layers=n_layers,
        lambda_min=400,
        lambda_max=700
    )

    print(f"\n优化后的膜系参数:")
    print(f"{'层号':<6} {'折射率':<12} {'厚度 (nm)':<12} {'光学厚度 (nm)':<15}")
    print("-" * 50)
    for i, (n, d) in enumerate(zip(n_list, d_list)):
        optical_thickness = n * d
        print(f"{i + 1:<6} {n:<12.4f} {d:<12.2f} {optical_thickness:<15.1f}")

    wavelengths, R, T = filter.evaluate(n_list, d_list, plot=True)

    print(f"\n{'='*60}")
    print("设计完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
