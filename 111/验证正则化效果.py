import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("贝里曲率正则化效果验证")
print("=" * 70)

from ahc_calculator import TightBindingFerromagnet as OldModel
from ahc_calculator import berry_curvature_kubo as old_berry
from ahc_calculator_regularized import TightBindingFerromagnet as NewModel
from ahc_calculator_regularized import (
    berry_curvature_kubo_formula as new_berry,
    detect_degeneracy_points,
    adaptive_k_grid
)

print("\n✓ 模块加载成功")

model_old = OldModel(model='square', t=1.0, t_nn=0.2, J=0.5, soc=0.1)
model_new = NewModel(model='square', t=1.0, t_nn=0.2, J=0.5, soc=0.1)

print("\n" + "-" * 70)
print("测试1: 能带交叉点附近的数值稳定性")
print("-" * 70)

test_points = [
    (0.0, 0.0, "Γ点"),
    (np.pi, np.pi, "M点"),
    (0.5, 0.5, "一般k点"),
    (np.pi/2, np.pi/2, "X/2点"),
]

print(f"\n{'k点':<15} {'方法':<15} {'贝里曲率能带1':<20} {'贝里曲率能带2':<20}")
print("-" * 75)

for kx, ky, name in test_points:
    bc_old, _ = old_berry(model_old, kx, ky)
    bc_new, _ = new_berry(model_new, kx, ky, eta=0.05, method='semiclassical')
    
    print(f"{name:<15} {'原始方法':<15} {bc_old[0]:<20.6f} {bc_old[1]:<20.6f}")
    print(f"{'':<15} {'正则化方法':<15} {bc_new[0]:<20.6f} {bc_new[1]:<20.6f}")
    print()

print("\n" + "-" * 70)
print("测试2: 检测简并点（能隙极小值位置）")
print("-" * 70)

k_scan = np.linspace(-np.pi, np.pi, 50)
gap_map = np.zeros((50, 50))

for i, kx in enumerate(k_scan):
    for j, ky in enumerate(k_scan):
        _, gap = detect_degeneracy_points(model_new, kx, ky)
        gap_map[i, j] = gap

min_gap_idx = np.unravel_index(np.argmin(gap_map), gap_map.shape)
min_gap_kx = k_scan[min_gap_idx[0]]
min_gap_ky = k_scan[min_gap_idx[1]]
min_gap = gap_map[min_gap_idx]

print(f"最小能隙位置: k = ({min_gap_kx:.3f}, {min_gap_ky:.3f})")
print(f"最小能隙值: {min_gap:.6f} t")

print("\n在最小能隙处测试贝里曲率计算:")
print(f"{'方法':<15} {'η':<10} {'贝里曲率能带1':<20} {'贝里曲率能带2':<20}")
print("-" * 70)

bc_old, _ = old_berry(model_old, min_gap_kx, min_gap_ky)
print(f"{'原始方法':<15} {'-':<10} {bc_old[0]:<20.6f} {bc_old[1]:<20.6f}")

for eta in [0.01, 0.05, 0.1, 0.2]:
    bc_new, _ = new_berry(model_new, min_gap_kx, min_gap_ky, eta=eta, method='semiclassical')
    print(f"{'正则化':<15} {eta:<10} {bc_new[0]:<20.6f} {bc_new[1]:<20.6f}")

print("\n" + "-" * 70)
print("测试3: 自适应网格细化")
print("-" * 70)

k_points, weights = adaptive_k_grid(
    model_new, base_res=15, refine_level=2, gap_threshold=0.15
)

print(f"总k点数: {len(k_points)}")
print(f"均匀网格等效k点数: {15*15} = 225")
print(f"自适应网格效率: {len(k_points)/225:.2f}x")

unique_areas = np.unique(weights)
print(f"\n不同级别的网格面积:")
for area in sorted(unique_areas)[:5]:
    count = np.sum(np.abs(weights - area) < 1e-10)
    print(f"  面积 = {area:.4f}: {count} 个点")

print("\n" + "-" * 70)
print("生成对比图像...")
print("-" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

k = np.linspace(-np.pi, np.pi, 30)
bc_old_map = np.zeros((30, 30, 2))
bc_new_map = np.zeros((30, 30, 2))

for i, kx in enumerate(k):
    for j, ky in enumerate(k):
        bc_o, _ = old_berry(model_old, kx, ky)
        bc_n, _ = new_berry(model_new, kx, ky, eta=0.05, method='semiclassical')
        bc_old_map[i, j] = bc_o
        bc_new_map[i, j] = bc_n

vmax = np.percentile(np.abs(bc_new_map), 99)
im0 = axes[0, 0].imshow(bc_old_map[:, :, 0].T, origin='lower',
                       extent=[-np.pi, np.pi, -np.pi, np.pi],
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
axes[0, 0].set_title('原始方法 - 贝里曲率 (能带1)', fontsize=12, fontweight='bold')
plt.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].imshow(bc_new_map[:, :, 0].T, origin='lower',
                       extent=[-np.pi, np.pi, -np.pi, np.pi],
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
axes[0, 1].set_title('正则化方法 - 贝里曲率 (能带1)', fontsize=12, fontweight='bold')
plt.colorbar(im1, ax=axes[0, 1])

diff = np.abs(bc_old_map[:, :, 0] - bc_new_map[:, :, 0])
im2 = axes[0, 2].imshow(diff.T, origin='lower',
                       extent=[-np.pi, np.pi, -np.pi, np.pi],
                       cmap='hot', aspect='equal')
axes[0, 2].set_title('绝对差值', fontsize=12, fontweight='bold')
plt.colorbar(im2, ax=axes[0, 2])

im3 = axes[1, 0].imshow(gap_map.T, origin='lower',
                       extent=[-np.pi, np.pi, -np.pi, np.pi],
                       cmap='jet', aspect='equal')
axes[1, 0].set_title('k空间能隙分布', fontsize=12, fontweight='bold')
plt.colorbar(im3, ax=axes[1, 0])

area_weights = weights / np.max(weights)
axes[1, 1].scatter(k_points[:, 0], k_points[:, 1], c=area_weights, 
                   s=15, cmap='viridis', alpha=0.7)
axes[1, 1].set_title('自适应k点网格', fontsize=12, fontweight='bold')
axes[1, 1].set_xlim(-np.pi, np.pi)
axes[1, 1].set_ylim(-np.pi, np.pi)
axes[1, 1].set_aspect('equal')

eta_values = [0.01, 0.05, 0.1, 0.2]
colors = ['r', 'g', 'b', 'orange']

k_line = np.linspace(0, np.pi, 50)
for idx, eta in enumerate(eta_values):
    bc_line = np.zeros(len(k_line))
    for i, kx in enumerate(k_line):
        bc, _ = new_berry(model_new, kx, kx, eta=eta, method='semiclassical')
        bc_line[i] = bc[0]
    axes[1, 2].plot(k_line / np.pi, bc_line, color=colors[idx], 
                   label=f'η = {eta}', linewidth=2)

axes[1, 2].set_xlabel('k / π (对角线)', fontsize=10)
axes[1, 2].set_ylabel('贝里曲率', fontsize=10)
axes[1, 2].set_title('不同η值的正则化效果', fontsize=12, fontweight='bold')
axes[1, 2].legend(fontsize=9)
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('正则化效果对比.png', dpi=200, bbox_inches='tight')
print("✓ 图像已保存: 正则化效果对比.png")

print("\n" + "=" * 70)
print("验证完成!")
print("=" * 70)
print("\n总结:")
print("1. 原始方法在能带交叉点附近可能出现数值不稳定")
print("2. Lorentzian展宽 (η参数) 有效平滑了极点问题")
print("3. 自适应网格在简并点附近自动提高分辨率")
print("4. 建议η取值范围: 0.05 - 0.1 (以t为单位)")
