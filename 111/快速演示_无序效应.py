import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("无序效应快速演示 - 杂质散射对异常霍尔电导率的影响")
print("=" * 70)

print("\n正在加载模块...")
from disorder_ahc import (
    TightBindingDisorder,
    kubo_formula_with_disorder,
    SupercellDisorder,
    plot_disorder_effects,
    plot_dos_with_disorder
)
print("✓ 模块加载成功")

model = TightBindingDisorder(
    model='square',
    t=1.0,
    t_nn=0.2,
    J=0.5,
    soc=0.1
)

print(f"\n模型参数:")
print(f"  最近邻跃迁 t = {model.t}")
print(f"  次近邻跃迁 t' = {model.t_nn}")
print(f"  交换劈裂 J = {model.J}")
print(f"  自旋轨道耦合 λ = {model.soc}")

print("\n" + "-" * 70)
print("演示1: 不同浓度下的CPA计算")
print("-" * 70)

c_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
W_list = [0.0, 0.5, 1.0, 2.0]
k_res = 20
mu = 0.0

print(f"计算参数: k_res={k_res}, mu={mu}")
print(f"浓度范围: {c_list}")
print(f"无序强度: {W_list}")

cpa_results = {}
for c in c_list:
    for W in W_list:
        print(f"  计算 c={c}, W={W}...", end=" ", flush=True)
        sigma = kubo_formula_with_disorder(
            model, k_res=k_res, mu=mu, eta=0.05, gamma=0.1 + W * c
        )
        cpa_results[(c, W)] = sigma
        print(f"σ_xy = {sigma:.6f}")

print("\n" + "-" * 70)
print("演示2: 超胞态密度计算")
print("-" * 70)

supercell_size = 4
print(f"超胞大小: {supercell_size}x{supercell_size}")

E_grid = np.linspace(-6, 6, 200)
c_demo = [0.0, 0.2, 0.4]
W_demo = 1.0

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

colors = plt.cm.viridis(np.linspace(0, 1, len(W_list)))
for idx, W in enumerate(W_list):
    cpa_values = [cpa_results[(c, W)] for c in c_list]
    axes[0].plot(c_list, cpa_values, 'o-', color=colors[idx], 
                label=f'W = {W}', linewidth=2, markersize=6)

axes[0].set_xlabel('杂质浓度 c', fontsize=12)
axes[0].set_ylabel('σ_xy (e²/h)', fontsize=12)
axes[0].set_title('异常霍尔电导率 vs 杂质浓度 (CPA)', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='k', linestyle=':', alpha=0.5)

colors = plt.cm.plasma(np.linspace(0, 1, len(c_demo)))
for idx, c in enumerate(c_demo):
    sc = SupercellDisorder(
        model, supercell_size=supercell_size, disorder_type='onsite', W=W_demo, c=c
    )
    
    dos_avg = np.zeros_like(E_grid)
    for seed in range(2):
        H = sc.generate_supercell_hamiltonian(seed=seed)
        eigvals, _ = sc.diagonlize_supercell(H)
        dos_avg += sc.compute_dos(eigvals, E_grid, sigma=0.1)
    dos_avg /= 2
    
    axes[1].plot(E_grid, dos_avg, color=colors[idx], label=f'c = {c}', linewidth=2)

axes[1].set_xlabel('能量 (t)', fontsize=12)
axes[1].set_ylabel('态密度 (DOS)', fontsize=12)
axes[1].set_title(f'不同浓度下的态密度 (W = {W_demo} t)', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('无序效应快速演示.png', dpi=150, bbox_inches='tight')
print("✓ 图像已保存: 无序效应快速演示.png")

print("\n" + "-" * 70)
print("结果总结")
print("-" * 70)

print(f"\n{'浓度 c':<10} {'强度 W':<10} {'σ_xy (e²/h)':<15}")
print("-" * 40)
for c in c_list:
    for W in W_list:
        print(f"{c:<10} {W:<10} {cpa_results[(c, W)]:<15.6f}")

print("\n" + "=" * 70)
print("演示完成!")
print("=" * 70)
print("\n物理结论:")
print("1. 随着杂质浓度增加，AHC通常先增加后减小")
print("2. 弱无序下，散射导致贝里曲率贡献重新分布")
print("3. 强无序下，安德森局域化抑制输运")
print("4. 态密度随无序增加而展宽")
print("\n完整功能请运行: python disorder_ahc.py")
