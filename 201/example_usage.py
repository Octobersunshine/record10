import numpy as np
from phonon_dispersion import Crystal, ForceConstants, PhononCalculator, HighSymmetryPaths, PhononVisualizer, compute_dos


def example_real_system():
    """
    示例：使用真实DFT力常数计算声子色散谱
    """
    print("=" * 60)
    print("示例1: 使用模型力常数计算NaCl结构声子谱")
    print("=" * 60)
    
    a = 5.64
    lattice = np.eye(3) * a
    
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5]
    ])
    
    symbols = ['Na', 'Cl']
    masses = np.array([22.9898, 35.453])
    
    crystal = Crystal(lattice, positions, symbols, masses)
    print(f"\n晶体: NaCl (岩盐结构)")
    print(f"晶格常数: {a} Å")
    print(f"原子数: {crystal.n_atoms}")
    print(f"原子: {symbols}")
    
    fc = ForceConstants(crystal, cutoff=8.0)
    fc.generate_model_fc(spring_constant=12.0)
    print(f"力常数矩阵形状: {fc.fc_matrix.shape}")
    
    phonon = PhononCalculator(crystal, fc)
    
    print("\nΓ点频率 (cm⁻¹):")
    gamma = np.array([0.0, 0.0, 0.0])
    freqs = phonon.compute_frequencies(gamma, convert_to_cm=True)
    for i, f in enumerate(freqs):
        print(f"  模式 {i+1}: {f:.2f} cm⁻¹")
    
    path_points, labels = HighSymmetryPaths.get_sc_path()
    distances, frequencies = phonon.compute_band_structure(path_points, n_points_per_segment=60)
    
    PhononVisualizer.plot_dispersion(
        distances, frequencies, labels,
        title='NaCl Phonon Dispersion (Model Force Constants)',
        save_path='nacl_phonon_dispersion.png'
    )
    
    print(f"\n频率范围: {np.min(frequencies):.2f} - {np.max(frequencies):.2f} cm⁻¹")
    print("NaCl色散图已保存为: nacl_phonon_dispersion.png")
    
    return crystal, fc, phonon


def example_diamond_structure():
    """
    示例：金刚石结构声子计算
    """
    print("\n" + "=" * 60)
    print("示例2: 金刚石结构声子色散谱")
    print("=" * 60)
    
    a = 3.57
    lattice = np.array([
        [0.0, a/2, a/2],
        [a/2, 0.0, a/2],
        [a/2, a/2, 0.0]
    ])
    
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25]
    ])
    
    symbols = ['C', 'C']
    masses = np.array([12.011, 12.011])
    
    crystal = Crystal(lattice, positions, symbols, masses)
    print(f"\n晶体: 金刚石结构")
    print(f"晶格常数: {a} Å")
    
    fc = ForceConstants(crystal, cutoff=3.0)
    fc.generate_model_fc(spring_constant=25.0)
    
    phonon = PhononCalculator(crystal, fc)
    
    path_points, labels = HighSymmetryPaths.get_fcc_path()
    distances, frequencies = phonon.compute_band_structure(path_points, n_points_per_segment=80)
    
    PhononVisualizer.plot_dispersion(
        distances, frequencies, labels,
        title='Diamond Structure Phonon Dispersion',
        save_path='diamond_phonon_dispersion.png'
    )
    
    print("金刚石色散图已保存为: diamond_phonon_dispersion.png")
    
    print("\n计算声子态密度...")
    freq_axis, dos = compute_dos(phonon, n_mesh=12, sigma=4.0)
    PhononVisualizer.plot_dos(
        freq_axis, dos,
        title='Diamond Phonon Density of States',
        save_path='diamond_phonon_dos.png'
    )
    print("金刚石态密度图已保存为: diamond_phonon_dos.png")
    
    return crystal, fc, phonon


def example_save_and_load_fc():
    """
    示例：保存和加载力常数
    """
    print("\n" + "=" * 60)
    print("示例3: 保存和加载力常数文件")
    print("=" * 60)
    
    a = 4.0
    lattice = np.eye(3) * a
    positions = np.array([[0.0, 0.0, 0.0]])
    symbols = ['X']
    masses = np.array([40.0])
    
    crystal = Crystal(lattice, positions, symbols, masses)
    
    fc = ForceConstants(crystal, cutoff=5.0)
    fc.generate_model_fc(spring_constant=10.0)
    
    fc.save_fc_to_file('example_force_constants.npz')
    print("\n力常数已保存到: example_force_constants.npz")
    
    fc_loaded = ForceConstants(crystal)
    fc_loaded.load_fc_from_file('example_force_constants.npz')
    print("力常数已从文件加载")
    
    phonon = PhononCalculator(crystal, fc_loaded)
    gamma = np.array([0.0, 0.0, 0.0])
    freqs = phonon.compute_frequencies(gamma, convert_to_cm=True)
    print(f"Γ点频率 (从加载的力常数计算): {freqs}")
    
    return fc_loaded


def custom_high_symmetry_path():
    """
    示例：自定义高对称路径
    """
    print("\n" + "=" * 60)
    print("示例4: 自定义高对称k点路径")
    print("=" * 60)
    
    a = 5.0
    lattice = np.eye(3) * a
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    masses = np.array([30.0, 40.0])
    
    crystal = Crystal(lattice, positions, ['A', 'B'], masses)
    fc = ForceConstants(crystal, cutoff=6.0)
    fc.generate_model_fc(spring_constant=18.0)
    
    phonon = PhononCalculator(crystal, fc)
    
    G = np.array([0.0, 0.0, 0.0])
    X = np.array([0.5, 0.0, 0.0])
    Y = np.array([0.0, 0.5, 0.0])
    S = np.array([0.5, 0.5, 0.0])
    Z = np.array([0.0, 0.0, 0.5])
    
    custom_path = [G, X, S, Y, G, Z, X]
    custom_labels = ['Γ', 'X', 'S', 'Y', 'Γ', 'Z', 'X']
    
    distances, frequencies = phonon.compute_band_structure(custom_path, n_points_per_segment=50)
    
    PhononVisualizer.plot_dispersion(
        distances, frequencies, custom_labels,
        title='Custom Path Phonon Dispersion',
        save_path='custom_path_phonon.png'
    )
    
    print("自定义路径色散图已保存为: custom_path_phonon.png")
    return phonon


def analysis_phonon_modes():
    """
    示例：分析特定k点的声子模式
    """
    print("\n" + "=" * 60)
    print("示例5: 声子模式详细分析")
    print("=" * 60)
    
    a = 5.43
    lattice = np.array([
        [0.0, a/2, a/2],
        [a/2, 0.0, a/2],
        [a/2, a/2, 0.0]
    ])
    positions = np.array([[0.0, 0.0, 0.0]])
    masses = np.array([28.0855])
    
    crystal = Crystal(lattice, positions, ['Si'], masses)
    fc = ForceConstants(crystal, cutoff=5.0)
    fc.generate_model_fc(spring_constant=20.0)
    phonon = PhononCalculator(crystal, fc)
    
    k_points = {
        'Γ': np.array([0.0, 0.0, 0.0]),
        'X': np.array([0.5, 0.0, 0.5]),
        'L': np.array([0.5, 0.5, 0.5]),
        'K': np.array([0.375, 0.375, 0.75])
    }
    
    print("\n各高对称点的声子频率 (cm⁻¹):")
    print("-" * 50)
    for name, k in k_points.items():
        freqs = phonon.compute_frequencies(k, convert_to_cm=True)
        print(f"{name}: {[f'{f:.1f}' for f in freqs]}")
    
    return phonon


if __name__ == "__main__":
    example_real_system()
    example_diamond_structure()
    example_save_and_load_fc()
    custom_high_symmetry_path()
    analysis_phonon_modes()
    
    print("\n" + "=" * 60)
    print("所有示例计算完成!")
    print("生成的图片文件:")
    print("  - nacl_phonon_dispersion.png")
    print("  - diamond_phonon_dispersion.png")
    print("  - diamond_phonon_dos.png")
    print("  - custom_path_phonon.png")
    print("=" * 60)
