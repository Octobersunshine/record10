import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.optimize import minimize
from multilayer_earth_model import LayeredEarthModel, Layer, MU0


@dataclass
class MTData:
    frequencies: np.ndarray
    apparent_resistivity: np.ndarray
    phase: np.ndarray
    rho_error: Optional[np.ndarray] = None
    phase_error: Optional[np.ndarray] = None


class MTInversion:
    def __init__(self, mt_data: MTData, n_layers: int = 4):
        self.mt_data = mt_data
        self.n_layers = n_layers
        self.best_model = None
        
        if mt_data.rho_error is None:
            self.rho_error = np.ones_like(mt_data.apparent_resistivity) * 0.05
        else:
            self.rho_error = mt_data.rho_error
            
        if mt_data.phase_error is None:
            self.phase_error = np.ones_like(mt_data.phase) * 2.0
        else:
            self.phase_error = mt_data.phase_error

    def _model_to_params(self, layers: List[Layer]) -> np.ndarray:
        params = []
        for i, layer in enumerate(layers):
            if i < len(layers) - 1:
                params.append(np.log10(layer.thickness))
            params.append(np.log10(layer.resistivity))
        return np.array(params)

    def _params_to_model(self, params: np.ndarray) -> List[Layer]:
        layers = []
        param_idx = 0
        for i in range(self.n_layers):
            if i < self.n_layers - 1:
                thickness = 10 ** params[param_idx]
                param_idx += 1
            else:
                thickness = float('inf')
            resistivity = 10 ** params[param_idx]
            param_idx += 1
            layers.append(Layer(thickness=thickness, resistivity=resistivity))
        return layers

    def _misfit(self, params: np.ndarray) -> float:
        try:
            layers = self._params_to_model(params)
            model = LayeredEarthModel(layers)
            
            rho_pred, phase_pred = model.compute_apparent_resistivity(
                self.mt_data.frequencies
            )
            
            rho_misfit = np.sum(
                ((np.log10(rho_pred) - np.log10(self.mt_data.apparent_resistivity)) / 
                 np.log10(1 + self.rho_error)) ** 2
            )
            
            phase_misfit = np.sum(
                ((phase_pred - self.mt_data.phase) / self.phase_error) ** 2
            )
            
            total_misfit = rho_misfit + phase_misfit
            
            reg = 0.0
            for i in range(1, self.n_layers):
                reg += (np.log10(layers[i].resistivity) - 
                       np.log10(layers[i-1].resistivity)) ** 2
            
            return total_misfit + 0.1 * reg
            
        except (ValueError, FloatingPointError):
            return 1e10

    def invert(self, initial_model: Optional[List[Layer]] = None) -> List[Layer]:
        if initial_model is None:
            initial_layers = []
            for i in range(self.n_layers):
                if i < self.n_layers - 1:
                    thickness = 1000 * (10 ** i)
                else:
                    thickness = float('inf')
                resistivity = 100.0
                initial_layers.append(Layer(thickness=thickness, resistivity=resistivity))
        else:
            initial_layers = initial_model
            
        initial_params = self._model_to_params(initial_layers)
        
        bounds = []
        for i in range(self.n_layers):
            if i < self.n_layers - 1:
                bounds.append((1, 6))
            bounds.append((-1, 4))
        
        result = minimize(
            self._misfit,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'disp': False}
        )
        
        self.best_model = self._params_to_model(result.x)
        return self.best_model

    def compute_predicted_response(self, layers: List[Layer]) -> Tuple[np.ndarray, np.ndarray]:
        model = LayeredEarthModel(layers)
        return model.compute_apparent_resistivity(self.mt_data.frequencies)


def create_sample_mt_data() -> MTData:
    frequencies = np.logspace(-4, 2, 50)
    
    true_layers = [
        Layer(thickness=500, resistivity=100),
        Layer(thickness=5000, resistivity=500),
        Layer(thickness=20000, resistivity=10),
        Layer(thickness=float('inf'), resistivity=1000)
    ]
    
    model = LayeredEarthModel(true_layers)
    rho_app, phase = model.compute_apparent_resistivity(frequencies)
    
    np.random.seed(42)
    rho_noise = rho_app * np.random.normal(0, 0.03, size=rho_app.shape)
    phase_noise = np.random.normal(0, 1.0, size=phase.shape)
    
    return MTData(
        frequencies=frequencies,
        apparent_resistivity=rho_app + rho_noise,
        phase=phase + phase_noise
    )


def save_mt_data(mt_data: MTData, filename: str):
    with open(filename, 'w') as f:
        f.write("# Frequency(Hz)  Apparent_Resistivity(Ohm-m)  Phase(deg)\n")
        for freq, rho, ph in zip(mt_data.frequencies, 
                                   mt_data.apparent_resistivity, 
                                   mt_data.phase):
            f.write(f"{freq:.6e}  {rho:.6e}  {ph:.4f}\n")


def load_mt_data(filename: str) -> MTData:
    data = np.loadtxt(filename, comments='#')
    return MTData(
        frequencies=data[:, 0],
        apparent_resistivity=data[:, 1],
        phase=data[:, 2]
    )


def main():
    print("=" * 60)
    print("MT测深数据反演示例")
    print("=" * 60)
    
    print("\n生成合成MT数据...")
    mt_data = create_sample_mt_data()
    
    print(f"数据点数: {len(mt_data.frequencies)}")
    print(f"频率范围: {mt_data.frequencies.min():.2e} - {mt_data.frequencies.max():.2e} Hz")
    
    print("\n保存MT数据到文件...")
    save_mt_data(mt_data, 'sample_mt_data.txt')
    
    print("\n开始反演...")
    inversion = MTInversion(mt_data, n_layers=4)
    inverted_model = inversion.invert()
    
    print("\n反演得到的地电阻率模型:")
    for i, layer in enumerate(inverted_model):
        if layer.thickness == float('inf'):
            print(f"  层{i+1}: 半空间, 电阻率={layer.resistivity:.1f} Ω·m")
        else:
            print(f"  层{i+1}: 厚度={layer.thickness:.1f}m, 电阻率={layer.resistivity:.1f} Ω·m")
    
    print("\n" + "=" * 60)
    print("真实模型:")
    print("=" * 60)
    true_layers = [
        Layer(thickness=500, resistivity=100),
        Layer(thickness=5000, resistivity=500),
        Layer(thickness=20000, resistivity=10),
        Layer(thickness=float('inf'), resistivity=1000)
    ]
    for i, layer in enumerate(true_layers):
        if layer.thickness == float('inf'):
            print(f"  层{i+1}: 半空间, 电阻率={layer.resistivity:.1f} Ω·m")
        else:
            print(f"  层{i+1}: 厚度={layer.thickness:.1f}m, 电阻率={layer.resistivity:.1f} Ω·m")
    
    rho_pred, phase_pred = inversion.compute_predicted_response(inverted_model)
    
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.loglog(mt_data.frequencies, mt_data.apparent_resistivity, 
               'ko', label='观测数据', markersize=4)
    ax1.loglog(mt_data.frequencies, rho_pred, 'r-', label='反演预测', linewidth=2)
    ax1.set_xlabel('频率 (Hz)')
    ax1.set_ylabel('视电阻率 (Ω·m)')
    ax1.set_title('MT视电阻率拟合')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')
    
    ax2.semilogx(mt_data.frequencies, mt_data.phase, 
                 'ko', label='观测数据', markersize=4)
    ax2.semilogx(mt_data.frequencies, phase_pred, 'r-', label='反演预测', linewidth=2)
    ax2.set_xlabel('频率 (Hz)')
    ax2.set_ylabel('相位 (度)')
    ax2.set_title('MT相位拟合')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('mt_inversion_fit.png', dpi=150, bbox_inches='tight')
    print("\n反演拟合图已保存为: mt_inversion_fit.png")
    
    fig2, ax = plt.subplots(figsize=(8, 10))
    
    depths = []
    resistivities = []
    current_depth = 0
    
    for i, layer in enumerate(inverted_model):
        if layer.thickness != float('inf'):
            depths.extend([current_depth, current_depth + layer.thickness])
            resistivities.extend([layer.resistivity, layer.resistivity])
            current_depth += layer.thickness
        else:
            depths.extend([current_depth, current_depth + 50000])
            resistivities.extend([layer.resistivity, layer.resistivity])
    
    ax.semilogx(resistivities, np.array(depths) / 1000, 'b-', linewidth=2.5, label='反演模型')
    ax.set_xlabel('电阻率 (Ω·m)')
    ax.set_ylabel('深度 (km)')
    ax.set_title('反演得到的地电阻率剖面')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('inverted_resistivity_profile.png', dpi=150, bbox_inches='tight')
    print("电阻率剖面图已保存为: inverted_resistivity_profile.png")


if __name__ == "__main__":
    main()
