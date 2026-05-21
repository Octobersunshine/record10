import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


class EITHybridSolver:
    def __init__(self, mesh, forward_solver, unet_model_path=None, grid_size=64, device='cpu'):
        self.mesh = mesh
        self.forward = forward_solver
        self.grid_size = grid_size
        self.device = device
        self.unet_reconstructor = None
        
        if unet_model_path is not None:
            try:
                from eit_unet import EITFastReconstructor
                self.unet_reconstructor = EITFastReconstructor(
                    unet_model_path, mesh, grid_size, device
                )
                print("U-Net模型加载成功！")
            except Exception as e:
                print(f"U-Net模型加载失败: {e}")
                print("将仅使用传统方法")
        
        self._create_grid()

    def _create_grid(self):
        xi = np.linspace(-1.1, 1.1, self.grid_size)
        yi = np.linspace(-1.1, 1.1, self.grid_size)
        self.XI, self.YI = np.meshgrid(xi, yi)
        self.mask = self.XI**2 + self.YI**2 <= 1.0

    def sigma_to_image(self, sigma):
        elem_centers = np.zeros((self.mesh.n_elements, 2))
        for e_idx in range(self.mesh.n_elements):
            elem_nodes = self.mesh.nodes[self.mesh.elements[e_idx]]
            elem_centers[e_idx] = np.mean(elem_nodes, axis=0)

        zi = griddata(elem_centers, sigma, (self.XI, self.YI), method='cubic', fill_value=1.0)
        zi[~self.mask] = 1.0
        return zi

    def reconstruct_traditional(self, measurements, use_cem=True, max_iter=15, reg_param=5e-3):
        from eit_solver_cem import EITInverseCEM
        
        start_time = time.time()
        
        inverse = EITInverseCEM(self.forward, max_iter=max_iter, reg_sigma=reg_param, reg_z=1e-2, tol=1e-4)
        
        if use_cem:
            z_init = np.ones(self.mesh.n_electrodes) * 0.1
            sigma_recon, z_recon = inverse.reconstruct_joint(measurements, z0=z_init)
        else:
            z_fixed = np.ones(self.mesh.n_electrodes) * 0.001
            sigma_recon = inverse.reconstruct_with_known_z(measurements, z_fixed)
        
        elapsed = time.time() - start_time
        print(f"传统方法重构耗时: {elapsed:.2f}秒")
        
        return sigma_recon, elapsed

    def reconstruct_unet(self, measurements):
        if self.unet_reconstructor is None:
            raise ValueError("U-Net模型未加载")
        
        start_time = time.time()
        
        img_recon = self.unet_reconstructor.reconstruct(measurements)
        sigma_recon = self.unet_reconstructor.reconstruct_to_elements(measurements)
        
        elapsed = time.time() - start_time
        print(f"U-Net快速重构耗时: {elapsed:.4f}秒")
        
        return sigma_recon, img_recon, elapsed

    def reconstruct_hybrid(self, measurements, refine_iter=5):
        if self.unet_reconstructor is None:
            return self.reconstruct_traditional(measurements)
        
        sigma_unet, img_unet, time_unet = self.reconstruct_unet(measurements)
        
        from eit_solver_cem import EITInverseCEM
        start_time = time.time()
        
        inverse = EITInverseCEM(self.forward, max_iter=refine_iter, reg_sigma=1e-3, reg_z=1e-2, tol=1e-4)
        z_init = np.ones(self.mesh.n_electrodes) * 0.1
        sigma_refined, z_refined = inverse.reconstruct_joint(measurements, sigma0=sigma_unet, z0=z_init)
        
        time_refine = time.time() - start_time
        total_time = time_unet + time_refine
        print(f"混合方法总耗时: {total_time:.2f}秒 (U-Net: {time_unet:.3f}s + 精修: {time_refine:.3f}s)")
        
        return sigma_refined, total_time

    def compare_methods(self, measurements, sigma_true, title_prefix=""):
        results = {}
        
        print("\n" + "=" * 60)
        print("方法1: 传统Gauss-Newton + CEM")
        print("=" * 60)
        sigma_trad, time_trad = self.reconstruct_traditional(measurements, use_cem=True)
        img_trad = self.sigma_to_image(sigma_trad)
        error_trad = np.linalg.norm(sigma_trad - sigma_true) / np.linalg.norm(sigma_true)
        results['traditional'] = {'sigma': sigma_trad, 'img': img_trad, 'time': time_trad, 'error': error_trad}
        
        if self.unet_reconstructor is not None:
            print("\n" + "=" * 60)
            print("方法2: U-Net快速重建 (绝对成像)")
            print("=" * 60)
            sigma_unet, img_unet, time_unet = self.reconstruct_unet(measurements)
            error_unet = np.linalg.norm(sigma_unet - sigma_true) / np.linalg.norm(sigma_true)
            results['unet'] = {'sigma': sigma_unet, 'img': img_unet, 'time': time_unet, 'error': error_unet}
            
            print("\n" + "=" * 60)
            print("方法3: 混合方法 (U-Net + Gauss-Newton精修)")
            print("=" * 60)
            sigma_hybrid, time_hybrid = self.reconstruct_hybrid(measurements, refine_iter=5)
            img_hybrid = self.sigma_to_image(sigma_hybrid)
            error_hybrid = np.linalg.norm(sigma_hybrid - sigma_true) / np.linalg.norm(sigma_true)
            results['hybrid'] = {'sigma': sigma_hybrid, 'img': img_hybrid, 'time': time_hybrid, 'error': error_hybrid}
        
        return results


def plot_comprehensive_comparison(mesh, sigma_true, results, save_path='method_comparison.png'):
    n_methods = len(results) + 1
    fig = plt.figure(figsize=(6 * n_methods, 7))
    
    gs = fig.add_gridspec(2, n_methods, hspace=0.3, wspace=0.3)
    
    xi = np.linspace(-1.1, 1.1, 64)
    yi = np.linspace(-1.1, 1.1, 64)
    XI, YI = np.meshgrid(xi, yi)
    circle_mask = XI**2 + YI**2 <= 1.0
    
    elem_centers = np.zeros((mesh.n_elements, 2))
    for e_idx in range(mesh.n_elements):
        elem_nodes = mesh.nodes[mesh.elements[e_idx]]
        elem_centers[e_idx] = np.mean(elem_nodes, axis=0)
    
    img_true = griddata(elem_centers, sigma_true, (XI, YI), method='cubic', fill_value=1.0)
    img_true[~circle_mask] = np.nan
    
    methods = ['真实分布'] + list(results.keys())
    method_names = {
        'true': '真实分布',
        'traditional': '传统Gauss-Newton',
        'unet': 'U-Net快速重建',
        'hybrid': '混合方法'
    }
    
    for i, method in enumerate(methods):
        ax = fig.add_subplot(gs[0, i])
        
        if method == '真实分布':
            img = img_true
            error = 0
            t = 0
        else:
            r = results[method]
            if 'img' in r:
                img = r['img']
            else:
                img = griddata(elem_centers, r['sigma'], (XI, YI), method='cubic', fill_value=1.0)
            img[~circle_mask] = np.nan
            error = r['error'] * 100
            t = r['time']
        
        im = ax.pcolormesh(XI, YI, img, cmap='viridis', shading='auto', vmin=0.5, vmax=3.5)
        ax.set_aspect('equal')
        title = method_names.get(method, method)
        if method != '真实分布':
            title += f"\n误差: {error:.1f}%, 时间: {t:.2f}s"
        ax.set_title(title, fontsize=12)
        
        from matplotlib.patches import Circle
        circle = Circle((0, 0), 1.0, fill=False, color='black', linewidth=2)
        ax.add_patch(circle)
    
    cbar_ax = fig.add_axes([0.92, 0.55, 0.02, 0.35])
    plt.colorbar(im, cax=cbar_ax, label='电导率 (S/m)')
    
    ax_time = fig.add_subplot(gs[1, :])
    method_labels = []
    times = []
    errors = []
    
    for method, r in results.items():
        method_labels.append(method_names.get(method, method))
        times.append(r['time'])
        errors.append(r['error'] * 100)
    
    x = np.arange(len(method_labels))
    width = 0.35
    
    bars1 = ax_time.bar(x - width/2, times, width, label='重建时间 (秒)', alpha=0.7, color='steelblue')
    ax2 = ax_time.twinx()
    bars2 = ax2.bar(x + width/2, errors, width, label='相对误差 (%)', alpha=0.7, color='coral')
    
    ax_time.set_xlabel('重建方法')
    ax_time.set_ylabel('时间 (秒)')
    ax2.set_ylabel('相对误差 (%)')
    ax_time.set_title('各方法性能对比', fontsize=14)
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(method_labels)
    
    lines1, labels1 = ax_time.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_time.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax_time.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('EIT重建方法对比：传统迭代 vs U-Net快速重建 vs 混合方法', fontsize=16, y=0.995)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"对比图已保存: {save_path}")
    plt.close()
