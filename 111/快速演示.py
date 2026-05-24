import sys
import numpy as np

print("=" * 60)
print("紧束缚模型 - 异常霍尔电导率计算演示")
print("=" * 60)

try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print("✓ 导入 numpy, matplotlib 成功")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    print("\n请先安装依赖:")
    print("  pip install numpy scipy matplotlib")
    sys.exit(1)

print("\n正在加载主模块...")
from ahc_calculator import TightBindingFerromagnet, berry_curvature_kubo, fermi_distribution

print("✓ 模块加载成功")

print("\n" + "-" * 60)
print("测试1: 哈密顿量计算")
print("-" * 60)

model = TightBindingFerromagnet(
    model='square',
    t=1.0,
    t_nn=0.2,
    J=0.5,
    soc=0.1
)

h = model.hamiltonian(0, 0)
print(f"Γ点 (0,0) 哈密顿量:")
print(f"  形状: {h.shape}")
print(f"  本征值: {np.linalg.eigvalsh(h)}")

print("\n" + "-" * 60)
print("测试2: 贝里曲率计算")
print("-" * 60)

bc, eigvals = berry_curvature_kubo(model, 0.5, 0.5)
print(f"k点 (0.5, 0.5) 处:")
print(f"  本征值: {eigvals}")
print(f"  贝里曲率: {bc}")

print("\n" + "-" * 60)
print("测试3: 完整AHC计算 (简化版)")
print("-" * 60)

k_res = 20
print(f"使用 k_res = {k_res} (快速计算)")

k = np.linspace(-np.pi, np.pi, k_res)
dk = (2 * np.pi / k_res) ** 2

all_bc = np.zeros((k_res, k_res, 2))
all_eigvals = np.zeros((k_res, k_res, 2))

for i, kx in enumerate(k):
    for j, ky in enumerate(k):
        bc, eigvals = berry_curvature_kubo(model, kx, ky)
        all_bc[i, j] = bc
        all_eigvals[i, j] = eigvals

mu_list = np.linspace(-5, 5, 50)
ahc_list = np.zeros(50)

for idx, mu in enumerate(mu_list):
    ahc = 0.0
    for n in range(2):
        occ = fermi_distribution(all_eigvals[:, :, n], mu, 0.02)
        ahc += np.sum(all_bc[:, :, n] * occ) * dk
    ahc_list[idx] = ahc / (2 * np.pi)

print(f"计算完成!")
print(f"  最大 |σ_xy| = {np.max(np.abs(ahc_list)):.6f} e²/h")

print("\n" + "-" * 60)
print("测试4: 生成图像")
print("-" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(mu_list, ahc_list, 'b-', linewidth=2)
axes[0].set_xlabel('费米能 E_F (t)', fontsize=12)
axes[0].set_ylabel('σ_xy (e²/h)', fontsize=12)
axes[0].set_title('异常霍尔电导率', fontsize=14)
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.5)

im1 = axes[1].imshow(all_bc[:, :, 0].T, origin='lower',
                     extent=[-np.pi, np.pi, -np.pi, np.pi],
                     cmap='RdBu_r', aspect='equal')
axes[1].set_title('贝里曲率 - 低能带', fontsize=14)
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(all_bc[:, :, 1].T, origin='lower',
                     extent=[-np.pi, np.pi, -np.pi, np.pi],
                     cmap='RdBu_r', aspect='equal')
axes[2].set_title('贝里曲率 - 高能带', fontsize=14)
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig('快速演示结果.png', dpi=150, bbox_inches='tight')
print("✓ 图像已保存: 快速演示结果.png")

print("\n" + "=" * 60)
print("所有测试通过!")
print("=" * 60)
print("\n完整功能请运行: python ahc_calculator.py")
print("详细说明请查看: 使用说明.md")
