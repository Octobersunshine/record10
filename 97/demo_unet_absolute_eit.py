import numpy as np
import matplotlib.pyplot as plt
import os

from eit_solver_cem import EITMesh, EITForwardCEM
from eit_unet import (
    EIT_UNet, EITDataset, EITDataGenerator, 
    EITTrainer, plot_reconstruction_comparison
)
from eit_hybrid_solver import EITHybridSolver, plot_comprehensive_comparison


def create_phantom(mesh, phantom_type='complex'):
    sigma = np.ones(mesh.n_elements)
    
    elem_centers = np.zeros((mesh.n_elements, 2))
    for e_idx in range(mesh.n_elements):
        elem_nodes = mesh.nodes[mesh.elements[e_idx]]
        elem_centers[e_idx] = np.mean(elem_nodes, axis=0)
    
    if phantom_type == 'complex':
        centers = [(0.35, 0.0), (-0.25, 0.25), (0.0, -0.3)]
        radii = [0.15, 0.12, 0.18]
        conductivities = [3.5, 0.4, 2.5]
        
        for (cx, cy), r, cond in zip(centers, radii, conductivities):
            dist = np.sqrt((elem_centers[:, 0] - cx)**2 + (elem_centers[:, 1] - cy)**2)
            sigma[dist < r] = cond
    
    elif phantom_type == 'single':
        cx, cy = 0.3, 0.2
        r = 0.25
        dist = np.sqrt((elem_centers[:, 0] - cx)**2 + (elem_centers[:, 1] - cy)**2)
        sigma[dist < r] = 3.0
    
    elif phantom_type == 'two_circles':
        cx1, cy1, r1 = 0.35, 0.0, 0.18
        cx2, cy2, r2 = -0.25, 0.25, 0.15
        dist1 = np.sqrt((elem_centers[:, 0] - cx1)**2 + (elem_centers[:, 1] - cy1)**2)
        dist2 = np.sqrt((elem_centers[:, 0] - cx2)**2 + (elem_centers[:, 1] - cy2)**2)
        sigma[dist1 < r1] = 3.5
        sigma[dist2 < r2] = 0.5
    
    return sigma


def train_unet_model(n_train_samples=500, n_val_samples=100, grid_size=64, num_epochs=30, model_path='eit_unet_demo.pth'):
    print("=" * 70)
    print("  U-Net训练流程：生成数据 -> 训练模型 -> 验证")
    print("=" * 70)
    
    print("\n步骤 1: 创建网格和正向求解器...")
    mesh = EITMesh(n_radius=5, n_angles=12, r=1.0, electrode_angle_width=0.3)
    forward = EITForwardCEM(mesh)
    print(f"  网格: {mesh.n_nodes}节点, {mesh.n_elements}单元, {mesh.n_electrodes}电极")
    
    print("\n步骤 2: 生成训练数据 (绝对成像，无需基线)...")
    print("  生成多样化的电导率分布用于泛化学习")
    data_gen = EITDataGenerator(mesh, forward, grid_size=grid_size)
    
    print("  生成训练集...")
    train_meas, train_imgs = data_gen.generate_dataset(
        n_samples=n_train_samples, use_cem=True, contact_impedance_std=0.2
    )
    
    print("  生成验证集...")
    val_meas, val_imgs = data_gen.generate_dataset(
        n_samples=n_val_samples, use_cem=True, contact_impedance_std=0.2
    )
    
    print(f"  训练集大小: {train_meas.shape}")
    print(f"  验证集大小: {val_meas.shape}")
    
    print("\n步骤 3: 创建数据加载器...")
    train_dataset = EITDataset(train_meas, train_imgs)
    val_dataset = EITDataset(val_meas, val_imgs)
    
    train_loader = __import__('torch.utils.data').utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=0
    )
    val_loader = __import__('torch.utils.data').utils.data.DataLoader(
        val_dataset, batch_size=16, shuffle=False, num_workers=0
    )
    
    print("\n步骤 4: 初始化U-Net模型...")
    n_measurements = mesh.n_electrodes * (mesh.n_electrodes - 2)
    model = EIT_UNet(n_measurements=n_measurements, grid_size=grid_size)
    print(f"  输入维度: {n_measurements} 测量值")
    print(f"  输出维度: {grid_size}x{grid_size} 电导率图像")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")
    
    print("\n步骤 5: 开始训练...")
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    print(f"  训练设备: {device}")
    
    trainer = EITTrainer(model, device=device)
    best_loss = trainer.train(
        train_loader, val_loader, 
        num_epochs=num_epochs, lr=1e-4,
        save_path=model_path
    )
    print(f"  最佳验证损失: {best_loss:.6f}")
    
    print("\n步骤 6: 绘制训练历史...")
    trainer.plot_training_history('unet_training_history.png')
    print("  训练曲线已保存: unet_training_history.png")
    
    return model_path, mesh, forward, data_gen


def demo_comparison_without_training(model_path=None):
    print("=" * 70)
    print("  快速演示：使用预训练模型对比各方法")
    print("  (如果没有预训练模型，将只展示传统方法)")
    print("=" * 70)
    
    print("\n步骤 1: 初始化网格和求解器...")
    mesh = EITMesh(n_radius=5, n_angles=12, r=1.0, electrode_angle_width=0.3)
    forward = EITForwardCEM(mesh)
    
    print("\n步骤 2: 创建测试样本 (复杂多异常分布)...")
    sigma_true = create_phantom(mesh, phantom_type='complex')
    
    print("\n步骤 3: 模拟测量数据 (带噪声和接触阻抗变化)...")
    z = np.ones(mesh.n_electrodes) * 0.1
    z[:6] *= 3.0
    z[9:] *= 2.0
    
    measurements = forward.simulate_measurements(sigma_true, z)
    noise = 0.01 * np.random.randn(len(measurements))
    measurements_noisy = measurements + noise
    
    print(f"  测量点数: {len(measurements)}")
    print(f"  接触阻抗范围: [{z.min():.3f}, {z.max():.3f}]")
    
    print("\n步骤 4: 创建混合求解器...")
    hybrid_solver = EITHybridSolver(
        mesh, forward, 
        unet_model_path=model_path if os.path.exists(model_path) else None
    )
    
    print("\n步骤 5: 运行各方法对比...")
    results = hybrid_solver.compare_methods(measurements_noisy, sigma_true)
    
    print("\n步骤 6: 生成对比图...")
    plot_comprehensive_comparison(mesh, sigma_true, results, 'eit_method_comparison.png')
    
    print("\n" + "=" * 70)
    print("  性能总结")
    print("=" * 70)
    for method, r in results.items():
        name = {'traditional': '传统Gauss-Newton', 'unet': 'U-Net快速', 'hybrid': '混合方法'}
        print(f"  {name.get(method, method)}:")
        print(f"    相对误差: {r['error']*100:.2f}%")
        print(f"    计算时间: {r['time']:.3f}秒")
        if 'traditional' in results and method != 'traditional':
            speedup = results['traditional']['time'] / r['time']
            print(f"    加速比: {speedup:.1f}x")
    print("=" * 70)
    
    return results


def demo_absolute_imaging_concept():
    print("\n" + "=" * 70)
    print("  绝对成像 (Absolute EIT) 概念演示")
    print("=" * 70)
    print("\n传统差分成像 (Differential EIT):")
    print("  - 需要基线参考（空场测量）")
    print("  - 重建的是相对变化量 Δσ")
    print("  - 优点：对系统误差鲁棒")
    print("  - 缺点：需要基线，无法得到绝对电导率值")
    
    print("\n绝对成像 (Absolute EIT):")
    print("  - 无需基线，单次测量直接重建")
    print("  - 得到绝对电导率值 σ")
    print("  - 挑战：病态问题、系统误差、接触阻抗")
    print("  - U-Net方案：通过大量数据学习端到端映射，")
    print("             隐式建模系统误差，实现绝对成像")
    
    print("\nU-Net绝对成像的优势:")
    print("  ✓ 无需参考基线测量")
    print("  ✓ 鲁棒于接触阻抗变化（训练时随机化）")
    print("  ✓ 重建速度快（前向传播 < 10ms）")
    print("  ✓ 端到端学习，自动提取特征")
    print("=" * 70)


def main():
    print("\n" + "=" * 70)
    print("  EIT绝对成像与U-Net快速重建演示程序")
    print("=" * 70)
    
    model_path = 'eit_unet_demo.pth'
    
    choice = input("\n请选择演示模式:\n  1. 完整流程：训练U-Net + 对比演示\n  2. 快速演示：仅对比各方法（使用预训练模型）\n  3. 仅展示绝对成像概念\n\n请输入选择 (1/2/3): ").strip()
    
    if choice == '1':
        n_samples = input(f"训练样本数 (默认500): ").strip()
        n_samples = int(n_samples) if n_samples.isdigit() else 500
        
        n_epochs = input(f"训练轮数 (默认20): ").strip()
        n_epochs = int(n_epochs) if n_epochs.isdigit() else 20
        
        model_path, mesh, forward, data_gen = train_unet_model(
            n_train_samples=n_samples, 
            n_val_samples=max(50, n_samples//5),
            num_epochs=n_epochs,
            model_path=model_path
        )
        
        input("\n训练完成！按回车继续对比演示...")
        demo_comparison_without_training(model_path)
        
    elif choice == '2':
        demo_comparison_without_training(model_path)
        
    elif choice == '3':
        demo_absolute_imaging_concept()
        
    else:
        print("无效选择，运行快速演示...")
        demo_comparison_without_training(model_path)
    
    print("\n演示完成！查看生成的图片文件：")
    print("  - eit_method_comparison.png: 各方法对比")
    print("  - unet_training_history.png: 训练曲线")
    plt.show()


if __name__ == "__main__":
    main()
