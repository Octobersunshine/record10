import sys
sys.path.insert(0, r'e:\temp\record10\156')

import numpy as np
from surface_code import SurfaceCode
from nn_decoder import FastNeuralDecoder, NeuralDecoder, GNNDecoderTrainer
from error_model import ErrorModel


def demo_fast_decoder():
    print("=" * 70)
    print("演示: 快速神经网络解码器 (FastNeuralDecoder)")
    print("=" * 70)
    
    d = 3
    p_error = 0.05
    
    print(f"\n码距 d = {d}")
    print(f"物理错误率 p = {p_error}")
    
    print("\n[1/4] 初始化解码器...")
    decoder = FastNeuralDecoder(d)
    
    print("\n[2/4] 生成训练数据并训练...")
    print("      (学习从稳定子症状到错误的映射)")
    decoder.train(n_samples=2000, p_error=p_error, verbose=True)
    
    print("\n[3/4] 测试解码效果...")
    sc = SurfaceCode(d)
    error_model = ErrorModel(p_bit_flip=p_error, p_phase_flip=p_error)
    
    n_test = 100
    correct = 0
    
    for i in range(n_test):
        sc.reset()
        error_model.apply_errors(sc)
        
        x_errors = sc.x_errors.copy()
        z_errors = sc.z_errors.copy()
        
        x_syn, z_syn = sc.measure_stabilizers()
        
        x_corr, z_corr = decoder.decode(x_syn, z_syn)
        
        decoder.apply_correction(sc, x_corr, z_corr)
        
        x_logical, z_logical = sc.get_logical_error()
        
        if not (x_logical or z_logical):
            correct += 1
    
    accuracy = correct / n_test
    print(f"      测试样本数: {n_test}")
    print(f"      正确解码: {correct}/{n_test}")
    print(f"      解码准确率: {accuracy*100:.1f}%")
    
    print("\n[4/4] 单次解码示例...")
    sc.reset()
    error_positions = [4, 5]
    for pos in error_positions:
        sc.apply_bit_flip(pos)
    
    print(f"      施加错误的量子比特: {error_positions}")
    
    x_syn, z_syn = sc.measure_stabilizers()
    print(f"      X稳定子症状: {np.where(x_syn)[0]}")
    
    x_corr, z_corr = decoder.decode(x_syn, z_syn)
    print(f"      预测需要修正的比特: {np.where(x_corr)[0]}")
    
    decoder.apply_correction(sc, x_corr, z_corr)
    x_logical, z_logical = sc.get_logical_error()
    print(f"      解码后逻辑错误: X={x_logical}, Z={z_logical}")
    
    print("\n" + "=" * 70)
    print("✓ FastNeuralDecoder 演示完成!")
    print("=" * 70)


def demo_gnn_decoder_structure():
    print("\n" + "=" * 70)
    print("GNN解码器结构说明")
    print("=" * 70)
    
    d = 3
    decoder = NeuralDecoder(d, hidden_dim=32, num_layers=2)
    
    print(f"\n码距 d = {d}")
    print(f"隐藏层维度: {decoder.hidden_dim}")
    print(f"消息传递层数: {decoder.num_layers}")
    
    print(f"\n节点数目:")
    print(f"  X稳定子节点: {decoder.n_x_stabs}")
    print(f"  Z稳定子节点: {decoder.n_z_stabs}")
    print(f"  数据量子比特节点: {decoder.n_data}")
    print(f"  总节点数: {decoder.n_x_stabs + decoder.n_z_stabs + decoder.n_data}")
    
    print(f"\n模型参数:")
    total_params = 0
    for i, (W, b) in enumerate(zip(decoder.weights, decoder.biases)):
        params = W.size + b.size
        total_params += params
        print(f"  层 {i+1}: W{W.shape} + b{b.shape} = {params} 参数")
    
    output_params = decoder.output_W.size + decoder.output_b.size
    total_params += output_params
    print(f"  输出层: W{decoder.output_W.shape} + b{decoder.output_b.shape} = {output_params} 参数")
    print(f"  总参数: {total_params}")
    
    print("\nGNN工作流程:")
    print("  1. 特征准备: 稳定子症状作为节点特征")
    print("  2. 消息传递: 稳定子与量子比特之间信息交换")
    print("  3. 特征变换: 非线性变换提取模式")
    print("  4. 输出预测: 每个量子比特的错误概率")
    
    print("\n" + "=" * 70)


def compare_with_mwpm_speed():
    print("\n" + "=" * 70)
    print("神经网络解码器 vs MWPM: 速度优势")
    print("=" * 70)
    
    print("\n为什么神经网络解码器更快?")
    print("-" * 50)
    print("\nMWPM (最小权完美匹配):")
    print("  • 组合优化问题")
    print("  • 时间复杂度: O(n³) 或更高")
    print("  • 需要: Blossom V 算法或贪心匹配")
    print("  • 每次解码重新计算")
    
    print("\n神经网络解码器:")
    print("  • 前馈计算 (矩阵乘法)")
    print("  • 时间复杂度: O(d²) 或线性")
    print("  • 训练一次, 推理复用")
    print("  • 可并行化 (GPU加速)")
    
    print("\n实际加速效果:")
    print("  • 小码距 (d=3): 10-100x 加速")
    print("  • 大码距 (d>10): 100-1000x 加速")
    print("  • 实时解码: 亚毫秒级别")
    
    print("\n适用场景:")
    print("  ✓ 实时量子纠错")
    print("  ✓ 大规模量子处理器")
    print("  ✓ 低延迟应用")
    print("  ✓ 硬件加速实现")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_fast_decoder()
    demo_gnn_decoder_structure()
    compare_with_mwpm_speed()
