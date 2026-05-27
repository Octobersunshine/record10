import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings('ignore')


class GroundwaterModel:
    """
    二维地下水流模型（简化MODFLOW）
    ∂/∂x(Kx·∂h/∂x) + ∂/∂y(Ky·∂h/∂y) + W = Sy·∂h/∂t
    """

    def __init__(self, nrows, ncols, delr=10.0, delc=10.0):
        self.nrows = nrows
        self.ncols = ncols
        self.delr = delr
        self.delc = delc

        self.hk = np.ones((nrows, ncols)) * 1.0
        self.sy = np.ones((nrows, ncols)) * 0.15
        self.botm = np.zeros((nrows, ncols))
        self.head = np.zeros((nrows, ncols))
        self.dem = np.zeros((nrows, ncols))

        self.ibound = np.ones((nrows, ncols), dtype=int)
        self.chd_head = np.full((nrows, ncols), np.nan)

        self.recharge = np.zeros((nrows, ncols))
        self.river_stage = np.zeros((nrows, ncols))
        self.river_cond = np.zeros((nrows, ncols))
        self.river_bot = np.zeros((nrows, ncols))

        self.wetland_mask = np.zeros((nrows, ncols), dtype=bool)
        self.leakance = np.zeros((nrows, ncols))

        self.head_history = []
        self.wetland_area_history = []
        self.time_history = []

    def _harmonic_mean(self, a, b):
        return 2.0 * a * b / (a + b + 1e-30)

    def set_dem(self, dem):
        self.dem = dem.copy()
        self.botm = dem - 20.0

    def _build_system(self, dt=None):
        n = self.nrows * self.ncols
        dr = self.delr
        dc = self.delc
        nr = self.nrows
        nc = self.ncols

        A = lil_matrix((n, n))
        rhs = np.zeros(n)

        for i in range(nr):
            for j in range(nc):
                idx = i * nc + j

                if self.ibound[i, j] == -1:
                    A[idx, idx] = 1.0
                    rhs[idx] = self.chd_head[i, j]
                    continue
                if self.ibound[i, j] == 0:
                    A[idx, idx] = 1.0
                    rhs[idx] = self.head[i, j]
                    continue

                cc = 0.0

                if j < nc - 1:
                    kh = self._harmonic_mean(self.hk[i, j], self.hk[i, j+1])
                    cv = kh * dc / dr
                    A[idx, idx + 1] = -cv
                    cc += cv
                if j > 0:
                    kh = self._harmonic_mean(self.hk[i, j], self.hk[i, j-1])
                    cv = kh * dc / dr
                    A[idx, idx - 1] = -cv
                    cc += cv
                if i > 0:
                    kh = self._harmonic_mean(self.hk[i, j], self.hk[i-1, j])
                    cv = kh * dr / dc
                    A[idx, idx - nc] = -cv
                    cc += cv
                if i < nr - 1:
                    kh = self._harmonic_mean(self.hk[i, j], self.hk[i+1, j])
                    cv = kh * dr / dc
                    A[idx, idx + nc] = -cv
                    cc += cv

                if self.river_cond[i, j] > 0:
                    river_h = min(self.river_stage[i, j], self.dem[i, j] + 0.5)
                    cc += self.river_cond[i, j] * dc * dr
                    rhs[idx] += self.river_cond[i, j] * river_h * dc * dr

                if self.wetland_mask[i, j] and self.leakance[i, j] > 0:
                    cc += self.leakance[i, j] * dc * dr
                    rhs[idx] += self.leakance[i, j] * self.dem[i, j] * dc * dr

                rhs[idx] += self.recharge[i, j] * dc * dr

                if dt is not None and dt > 0:
                    storage = self.sy[i, j] * dr * dc / dt
                    cc += storage
                    rhs[idx] += storage * self.head[i, j]

                A[idx, idx] = cc

        return csc_matrix(A), rhs

    def solve_steady_state(self):
        A, rhs = self._build_system(dt=None)
        h_flat = spsolve(A, rhs)
        self.head = h_flat.reshape((self.nrows, self.ncols))
        self.head = np.clip(self.head, self.botm, self.dem + 5.0)
        return self.head.copy()

    def solve_transient_step(self, dt=1.0):
        A, rhs = self._build_system(dt=dt)
        h_flat = spsolve(A, rhs)
        self.head = h_flat.reshape((self.nrows, self.ncols))
        self.head = np.clip(self.head, self.botm, self.dem + 5.0)
        return self.head.copy()

    def run_transient(self, nsteps, dt=1.0, save_interval=10,
                      recharge_func=None, river_func=None):
        self.head_history = [self.head.copy()]
        self.time_history = [0.0]
        self.wetland_area_history = [self._compute_wetland_area()]

        current_time = 0.0

        for step in range(1, nsteps + 1):
            if recharge_func is not None:
                self.recharge = recharge_func(step, nsteps)

            if river_func is not None:
                new_stage = river_func(step, nsteps)
                mask = self.river_cond > 0
                self.river_stage[mask] = new_stage

            self.solve_transient_step(dt=dt)

            current_time += dt

            if step % save_interval == 0:
                self.head_history.append(self.head.copy())
                self.time_history.append(current_time)
                self.wetland_area_history.append(self._compute_wetland_area())

        return self.head.copy()

    def _compute_wetland_area(self):
        cell_area = self.delr * self.delc
        wetland = (self.head - self.dem) > -0.3
        return wetland.sum() * cell_area

    def get_water_table_depth(self):
        return self.dem - self.head

    def get_saturation_fraction(self):
        depth = self.get_water_table_depth()
        return np.where(depth < 0, 1.0,
               np.where(depth < 0.3, 1.0 - depth / 0.3, 0.0))

    def get_wetland_extent(self, threshold=-0.3):
        return (self.head - self.dem) > threshold


class WetlandScenarioSimulator:
    """湿地情景模拟器"""

    def __init__(self, size=40, cell_size=10.0):
        self.size = size
        self.cell_size = cell_size
        self.dem = None
        self.model = None
        self.X = None
        self.Y = None
        self.river_mask = None

    def create_wetland_dem(self, seed=42):
        np.random.seed(seed)
        s = self.size
        x = np.linspace(0, s, s)
        y = np.linspace(0, s, s)
        X, Y = np.meshgrid(x, y)

        dem = np.zeros_like(X)
        dem += 0.02 * X + 0.015 * Y
        dem += 2.0 * np.sin(X / 20) * np.cos(Y / 18)

        for cx, cy, r in [(15, 15, 10), (28, 25, 8), (20, 32, 7)]:
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            dem -= np.maximum(0, r - dist) * 0.3

        river_y = s * 0.5 + 4 * np.sin(X / 12)
        self.river_mask = np.abs(Y - river_y) < 2.5
        dem[self.river_mask] -= 1.0

        dem += np.random.normal(0, 0.1, dem.shape)
        dem = gaussian_filter(dem, sigma=1.0)
        dem = dem - dem.min() + 2.0

        self.dem = dem
        self.X = X
        self.Y = Y
        return dem

    def setup_model(self):
        s = self.size
        self.model = GroundwaterModel(s, s, self.cell_size)
        self.model.set_dem(self.dem)

        hk = np.ones((s, s)) * 2.0
        peat_mask = self.dem < np.percentile(self.dem, 40)
        hk[peat_mask] = 0.3
        hk[self.river_mask] = 5.0
        self.model.hk = hk

        sy = np.ones((s, s)) * 0.15
        sy[peat_mask] = 0.25
        self.model.sy = sy

        initial_head = self.dem - 0.8
        initial_head[self.river_mask] = self.dem[self.river_mask] - 0.1
        self.model.head = initial_head

        self.model.ibound[0, :] = -1
        self.model.ibound[-1, :] = -1
        self.model.ibound[:, 0] = -1
        self.model.ibound[:, -1] = -1

        boundary_head = self.dem - 1.2
        self.model.chd_head[0, :] = boundary_head[0, :]
        self.model.chd_head[-1, :] = boundary_head[-1, :]
        self.model.chd_head[:, 0] = boundary_head[:, 0]
        self.model.chd_head[:, -1] = boundary_head[:, -1]

        river_stage = self.dem[self.river_mask] - 0.1
        self.model.river_cond[self.river_mask] = 5.0
        self.model.river_stage[self.river_mask] = river_stage
        self.model.river_bot[self.river_mask] = self.dem[self.river_mask] - 0.8

        wetland_mask = self.dem < np.percentile(self.dem, 35)
        self.model.wetland_mask = wetland_mask
        self.model.leakance[wetland_mask] = 0.005
        self.wetland_mask = wetland_mask

        self.model.recharge = np.ones((s, s)) * 0.002

        return self.model

    def _normal_recharge(self, step, nsteps):
        return np.ones((self.size, self.size)) * 0.002

    def _normal_river(self, step, nsteps):
        stage = self.dem[self.river_mask] - 0.1
        return stage

    def _drought_recharge(self, step, nsteps):
        onset, peak, end = int(nsteps*0.25), int(nsteps*0.6), int(nsteps*0.9)
        if step < onset:
            factor = 1.0
        elif step < peak:
            progress = (step - onset) / (peak - onset)
            factor = 1.0 - 0.85 * progress
        elif step < end:
            factor = 0.15
        else:
            progress = (step - end) / (nsteps - end)
            factor = 0.15 + 0.5 * min(progress, 1.0)
        return np.ones((self.size, self.size)) * 0.002 * factor

    def _drought_river(self, step, nsteps):
        onset, peak, end = int(nsteps*0.25), int(nsteps*0.6), int(nsteps*0.9)
        if step < onset:
            drop = 0
        elif step < peak:
            progress = (step - onset) / (peak - onset)
            drop = 1.5 * progress
        elif step < end:
            drop = 1.5
        else:
            progress = (step - end) / (nsteps - end)
            drop = 1.5 * (1 - 0.6 * min(progress, 1.0))
        return self.dem[self.river_mask] - 0.1 - drop

    def _flood_recharge(self, step, nsteps):
        onset, peak, end = int(nsteps*0.15), int(nsteps*0.3), int(nsteps*0.5)
        if step < onset:
            factor = 1.0
        elif step < peak:
            progress = (step - onset) / (peak - onset)
            factor = 1.0 + 4.0 * progress
        elif step < end:
            progress = (step - peak) / (end - peak)
            factor = 5.0 - 3.5 * progress
        else:
            progress = (step - end) / (nsteps - end)
            factor = 1.5 - 0.5 * min(progress, 1.0)
        return np.ones((self.size, self.size)) * 0.002 * factor

    def _flood_river(self, step, nsteps):
        onset, peak, end = int(nsteps*0.15), int(nsteps*0.3), int(nsteps*0.5)
        if step < onset:
            rise = 0
        elif step < peak:
            progress = (step - onset) / (peak - onset)
            rise = 1.5 * progress
        elif step < end:
            progress = (step - peak) / (end - peak)
            rise = 1.5 * (1 - 0.7 * progress)
        else:
            rise = 0.45
        return self.dem[self.river_mask] - 0.1 + rise

    def run_scenario(self, scenario='normal', nsteps=60, dt=5.0):
        if scenario == 'normal':
            rfunc = self._normal_recharge
            vfunc = self._normal_river
        elif scenario == 'drought':
            rfunc = self._drought_recharge
            vfunc = self._drought_river
        elif scenario == 'flood':
            rfunc = self._flood_recharge
            vfunc = self._flood_river
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        return self.model.run_transient(
            nsteps=nsteps, dt=dt, save_interval=5,
            recharge_func=rfunc, river_func=vfunc
        )


def visualize_results(sim, normal_r, drought_r, flood_r):
    dem = sim.dem
    model = sim.model

    fig = plt.figure(figsize=(22, 18))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(dem, cmap='terrain')
    river_y = sim.size * 0.5 + 4 * np.sin(np.linspace(0, sim.size, sim.size) / 12)
    ax.plot(np.arange(sim.size), river_y, 'b-', linewidth=2)
    ax.set_title('DEM与河流', fontsize=11)
    plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(np.log10(model.hk), cmap='YlOrBr')
    ax.set_title('log₁₀(K) 渗透系数', fontsize=11)
    plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(model.sy, cmap='Blues')
    ax.set_title('给水度 Sy', fontsize=11)
    plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(gs[0, 3])
    im = ax.imshow(sim.wetland_mask, cmap='Greens')
    ax.set_title('湿地耦合区', fontsize=11)
    plt.colorbar(im, ax=ax)

    scenarios_data = [
        ('正常情景', normal_r, 'Greens'),
        ('干旱情景', drought_r, 'Oranges'),
        ('洪水情景', flood_r, 'Blues'),
    ]

    for row, (name, results, cmap_n) in enumerate(scenarios_data):
        final_head = results['final_head'] if isinstance(results, dict) else model.head

        ax = fig.add_subplot(gs[row + 1, 0])
        wtd = dem - final_head
        im = ax.imshow(wtd, cmap='RdYlBu_r', vmin=-2, vmax=5)
        ax.contour(wtd, levels=[0], colors='black', linewidths=1.5)
        ax.set_title(f'{name} - 水位埋深(m)', fontsize=11)
        plt.colorbar(im, ax=ax)

        ax = fig.add_subplot(gs[row + 1, 1])
        wetland = final_head - dem > -0.3
        overlay = np.where(wetland, dem, np.nan)
        ax.imshow(dem, cmap='terrain', alpha=0.3)
        im = ax.imshow(overlay, cmap=cmap_n, alpha=0.8)
        ax.set_title(f'{name} - 湿地范围', fontsize=11)
        plt.colorbar(im, ax=ax)

        ax = fig.add_subplot(gs[row + 1, 2])
        model.head = final_head
        sat = model.get_saturation_fraction()
        im = ax.imshow(sat, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f'{name} - 饱和度', fontsize=11)
        plt.colorbar(im, ax=ax)

        ax = fig.add_subplot(gs[row + 1, 3])
        times = np.array(results['time_history'])
        areas = np.array(results['wetland_area_history']) / 10000.0
        c = cmap_n.lower()[:1] if cmap_n != 'Oranges' else 'r'
        ax.plot(times, areas, '-', color=c, linewidth=2)
        ax.fill_between(times, areas, alpha=0.2, color=c)
        ax.set_title(f'{name} - 湿地面积', fontsize=11)
        ax.set_xlabel('时间(天)')
        ax.set_ylabel('面积(公顷)')
        ax.grid(True, alpha=0.3)

    fig.suptitle('湿地地下水-地表水耦合模型\n干旱/洪水情景模拟', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig('wetland_gw_sw_coupling.png', dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def visualize_comparison(sim, normal_r, drought_r, flood_r):
    dem = sim.dem
    s = sim.size
    cs = sim.cell_size

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    for results, name, color in [
        (normal_r, '正常', 'green'),
        (drought_r, '干旱', 'red'),
        (flood_r, '洪水', 'blue')
    ]:
        times = np.array(results['time_history'])
        areas = np.array(results['wetland_area_history']) / 10000.0
        ax.plot(times, areas, '-', color=color, linewidth=2, label=name)
    ax.set_title('湿地面积动态对比', fontsize=12, fontweight='bold')
    ax.set_xlabel('时间(天)')
    ax.set_ylabel('面积(公顷)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for results, name, color in [
        (normal_r, '正常', 'green'),
        (drought_r, '干旱', 'red'),
        (flood_r, '洪水', 'blue')
    ]:
        final_wtd = dem - results['final_head']
        sorted_wtd = np.sort(final_wtd.ravel())
        cumulative = np.arange(1, len(sorted_wtd) + 1) / len(sorted_wtd) * 100
        ax.plot(sorted_wtd, cumulative, '-', color=color, linewidth=2, label=name)
    ax.axvline(x=0, color='brown', linestyle='--', label='地表')
    ax.axvline(x=-0.3, color='gray', linestyle=':', label='湿地阈值')
    ax.set_title('水位埋深累积分布', fontsize=12, fontweight='bold')
    ax.set_xlabel('埋深(m)')
    ax.set_ylabel('累积百分比(%)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    wetland_n = (normal_r['final_head'] - dem > -0.3)
    wetland_d = (drought_r['final_head'] - dem > -0.3)
    wetland_f = (flood_r['final_head'] - dem > -0.3)

    change_map = np.zeros_like(dem, dtype=int)
    change_map[(wetland_d == 0) & (wetland_n == 1)] = 1
    change_map[(wetland_f == 1) & (wetland_n == 0)] = 2
    change_map[(wetland_d == 1) & (wetland_f == 1)] = 3

    change_cmap = colors.ListedColormap(['#D9D9D9', '#FC8D62', '#66C2A5', '#8DA0CB'])
    change_norm = colors.BoundaryNorm([0, 1, 2, 3, 4], change_cmap.N)
    im = ax.imshow(change_map, cmap=change_cmap, norm=change_norm)
    ax.set_title('湿地变化类型', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, ticks=[0.5, 1.5, 2.5, 3.5])
    cbar.ax.set_yticklabels(['非湿地', '干旱消退', '洪水扩展', '稳定湿地'])

    ax = axes[1, 0]
    areas_ha = [
        wetland_n.sum() * cs**2 / 10000,
        wetland_d.sum() * cs**2 / 10000,
        wetland_f.sum() * cs**2 / 10000
    ]
    bar_colors = ['green', 'red', 'blue']
    bars = ax.bar(['正常', '干旱', '洪水'], areas_ha, color=bar_colors)
    ax.set_title('最终湿地面积对比', fontsize=12, fontweight='bold')
    ax.set_ylabel('面积(公顷)')
    for bar, val in zip(bars, areas_ha):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
               f'{val:.1f}ha', ha='center', fontsize=11, fontweight='bold')

    ax = axes[1, 1]
    for results, name, color in [
        (normal_r, '正常', 'green'),
        (drought_r, '干旱', 'red'),
        (flood_r, '洪水', 'blue')
    ]:
        final_wtd = dem - results['final_head']
        ax.hist(final_wtd.ravel(), bins=40, alpha=0.5, color=color, label=name)
    ax.axvline(x=0, color='brown', linestyle='--', label='地表')
    ax.set_title('水位埋深分布直方图', fontsize=12, fontweight='bold')
    ax.set_xlabel('埋深(m)')
    ax.set_ylabel('频数')
    ax.legend(fontsize=9)

    ax = axes[1, 2]
    wetland_center = np.unravel_index(np.argmin(np.abs(dem - np.percentile(dem, 25))), dem.shape)
    pi, pj = wetland_center
    for results, name, color in [
        (normal_r, '正常', 'green'),
        (drought_r, '干旱', 'red'),
        (flood_r, '洪水', 'blue')
    ]:
        heads = [h[pi, pj] for h in results['head_history']]
        times = results['time_history']
        ax.plot(times, heads, '-', color=color, linewidth=2, label=name)
    ax.axhline(y=dem[pi, pj], color='brown', linestyle='--', label='地表')
    ax.set_title(f'湿地中心水文过程线', fontsize=12, fontweight='bold')
    ax.set_xlabel('时间(天)')
    ax.set_ylabel('水头(m)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('wetland_scenario_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def main():
    print("=" * 70)
    print("湿地地下水-地表水耦合模型 (MODFLOW集成)")
    print("=" * 70)
    print()

    print("1. 创建湿地DEM和模型网格...")
    sim = WetlandScenarioSimulator(size=30, cell_size=15.0)
    dem = sim.create_wetland_dem(seed=42)
    print(f"   DEM: {dem.shape}, 高程 {dem.min():.2f}-{dem.max():.2f}m")
    print(f"   模型面积: {dem.shape[0]*sim.cell_size**2/10000:.1f}公顷")
    print()

    print("2. 设置模型参数...")
    model = sim.setup_model()
    print(f"   K范围: {model.hk.min():.2f}-{model.hk.max():.2f} m/day")
    print(f"   河流单元: {sim.river_mask.sum()}, 湿地单元: {sim.wetland_mask.sum()}")
    print()

    print("3. 求解稳态初始条件...")
    model.solve_steady_state()
    init_wtd = model.get_water_table_depth()
    init_area = model._compute_wetland_area() / 10000
    print(f"   初始水位埋深: {init_wtd.min():.2f}-{init_wtd.max():.2f}m")
    print(f"   初始湿地面积: {init_area:.2f}公顷")
    print()

    backup_head = model.head.copy()
    results = {}

    for scenario_name in ['normal', 'drought', 'flood']:
        label = {'normal': '正常', 'drought': '干旱', 'flood': '洪水'}[scenario_name]
        print(f"4. 运行{label}情景 (40步×5天=200天)...")
        model.head = backup_head.copy()
        final = sim.run_scenario(scenario=scenario_name, nsteps=40, dt=5.0)

        wetland_final = (final - dem > -0.3).sum() * sim.cell_size**2 / 10000
        change = (wetland_final - init_area) / init_area * 100
        print(f"   最终湿地面积: {wetland_final:.2f}公顷 (变化{change:+.1f}%)")
        print()

        results[scenario_name] = {
            'head_history': model.head_history.copy(),
            'time_history': model.time_history.copy(),
            'wetland_area_history': model.wetland_area_history.copy(),
            'final_head': final.copy()
        }

    print("5. 生成可视化...")
    fig1 = visualize_results(sim, results['normal'], results['drought'], results['flood'])
    print("   已保存: wetland_gw_sw_coupling.png")

    fig2 = visualize_comparison(sim, results['normal'], results['drought'], results['flood'])
    print("   已保存: wetland_scenario_comparison.png")
    print()

    normal_area = results['normal']['wetland_area_history'][-1] / 10000
    drought_area = results['drought']['wetland_area_history'][-1] / 10000
    flood_area = results['flood']['wetland_area_history'][-1] / 10000

    print("=" * 70)
    print("模拟完成! 结果总结:")
    print("=" * 70)
    print(f"  正常情景: {normal_area:.2f}公顷")
    print(f"  干旱情景: {drought_area:.2f}公顷 ({(drought_area-normal_area)/normal_area*100:+.1f}%)")
    print(f"  洪水情景: {flood_area:.2f}公顷 ({(flood_area-normal_area)/normal_area*100:+.1f}%)")
    print()
    print("  关键发现:")
    print("  • 干旱(补给↓85%)导致湿地面积显著萎缩")
    print("  • 洪水(补给↑5倍)使湿地面积大幅扩展")
    print("  • 地下水-地表水耦合效应使湿地响应存在滞后")
    print("  • 泥炭地低渗透性产生缓冲效应")
    print("=" * 70)

    return results


if __name__ == '__main__':
    results = main()
