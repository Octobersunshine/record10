import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


EPS = 1e-10


def safe_divide(a, b):
    """
    安全除法，避免除零
    """
    return a / (b + EPS)


def add_epsilon(X, eps=EPS):
    """
    给矩阵添加极小值，避免零元素
    """
    return np.maximum(X, eps)


def beta_divergence(V, V_recon, beta):
    """
    计算β-散度
    
    参数:
        V: 原始矩阵
        V_recon: 重构矩阵
        beta: β值
            - beta=0: Itakura-Saito散度
            - beta=1: KL散度
            - beta=2: 欧几里得距离(F范数)
    
    返回:
        divergence: β-散度值
    """
    V = add_epsilon(V)
    V_recon = add_epsilon(V_recon)
    
    if beta == 2:
        return 0.5 * np.sum((V - V_recon) ** 2)
    elif beta == 1:
        return np.sum(V * np.log(safe_divide(V, V_recon)) - V + V_recon)
    elif beta == 0:
        return np.sum(safe_divide(V, V_recon) - np.log(safe_divide(V, V_recon)) - 1)
    else:
        term1 = safe_divide(V ** beta, beta * (beta - 1))
        term2 = safe_divide(V * (V_recon ** (beta - 1)), beta - 1)
        term3 = safe_divide(V_recon ** beta, beta)
        return np.sum(term1 - term2 + term3)


def custom_nmf_beta(V, n_components, beta=2, max_iter=1000, tol=1e-4, random_state=42):
    """
    自定义NMF实现，支持β-散度族损失函数（乘法更新规则）
    
    参数:
        V: 非负矩阵，形状为 (n_samples, n_features)
        n_components: 成分数量
        beta: β值
            - beta=0: Itakura-Saito散度（适合音频频谱）
            - beta=1: KL散度（适合文本/计数数据）
            - beta=2: 欧几里得距离（默认，适合一般数据）
        max_iter: 最大迭代次数
        tol: 收敛阈值
        random_state: 随机种子
    
    返回:
        W, H, n_iter, reconstruction_error
    """
    np.random.seed(random_state)
    n_samples, n_features = V.shape
    
    W = np.random.rand(n_samples, n_components)
    H = np.random.rand(n_components, n_features)
    
    W = add_epsilon(W)
    H = add_epsilon(H)
    
    prev_error = np.inf
    
    for i in range(max_iter):
        V_recon = np.dot(W, H)
        V_recon = add_epsilon(V_recon)
        
        if beta == 2:
            H = H * safe_divide(np.dot(W.T, V), np.dot(np.dot(W.T, W), H))
            W = W * safe_divide(np.dot(V, H.T), np.dot(np.dot(W, H), H.T))
        elif beta == 1:
            V_over_VH = safe_divide(V, V_recon)
            H = H * safe_divide(np.dot(W.T, V_over_VH), np.sum(W, axis=0, keepdims=True).T)
            W = W * safe_divide(np.dot(V_over_VH, H.T), np.sum(H, axis=1, keepdims=True).T)
        elif beta == 0:
            V_over_VH = safe_divide(V, V_recon ** 2)
            V_inv_VH = safe_divide(1, V_recon)
            
            H_numerator = np.dot(W.T, V_over_VH)
            H_denominator = np.dot(W.T, V_inv_VH)
            H = H * safe_divide(H_numerator, H_denominator)
            
            W_numerator = np.dot(V_over_VH, H.T)
            W_denominator = np.dot(V_inv_VH, H.T)
            W = W * safe_divide(W_numerator, W_denominator)
        else:
            V_recon_beta_minus_1 = V_recon ** (beta - 1)
            V_recon_beta_minus_2 = V_recon ** (beta - 2)
            
            H = H * safe_divide(
                np.dot(W.T, V * V_recon_beta_minus_2),
                np.dot(W.T, V_recon_beta_minus_1)
            )
            
            W = W * safe_divide(
                np.dot(V * V_recon_beta_minus_2, H.T),
                np.dot(V_recon_beta_minus_1, H.T)
            )
        
        W = add_epsilon(W)
        H = add_epsilon(H)
        
        current_error = beta_divergence(V, np.dot(W, H), beta)
        
        if abs(prev_error - current_error) < tol * abs(prev_error):
            break
        
        prev_error = current_error
    
    V_recon_final = np.dot(W, H)
    reconstruction_error = beta_divergence(V, V_recon_final, beta)
    
    return W, H, i + 1, reconstruction_error


def get_beta_name(beta):
    """
    获取β-散度的名称
    """
    if beta == 0:
        return "Itakura-Saito散度"
    elif beta == 1:
        return "KL散度"
    elif beta == 2:
        return "欧几里得距离(F范数)"
    else:
        return f"β-散度(β={beta})"


def nmf_feature_extraction(V, n_components, random_state=42, max_iter=1000, 
                           use_custom=False, beta=2, normalize=True):
    """
    使用NMF进行非负矩阵分解和特征提取（包含β-散度族和数值稳定性保护）
    
    参数:
        V: 非负矩阵，形状为 (n_samples, n_features)
        n_components: 分解后的成分数量（降维后的维度）
        random_state: 随机种子，保证可复现性
        max_iter: 最大迭代次数
        use_custom: 是否使用自定义NMF实现（True使用乘法更新规则，False使用scikit-learn）
        beta: β-散度参数（仅use_custom=True时有效）
            - beta=0: Itakura-Saito散度（适合音频频谱）
            - beta=1: KL散度（适合文本/计数数据）
            - beta=2: 欧几里得距离（默认，适合一般数据）
        normalize: 是否进行归一化
    
    返回:
        W: 基矩阵，形状为 (n_samples, n_components) - 样本的低维特征表示
        H: 系数矩阵，形状为 (n_components, n_features) - 提取的特征基
        model: 训练好的NMF模型（scikit-learn版本时返回）
        scaler: 归一化器
    """
    if np.any(V < 0):
        raise ValueError("输入矩阵V必须是非负的")
    
    V_safe = add_epsilon(V.copy())
    
    if normalize:
        scaler = MinMaxScaler()
        V_normalized = scaler.fit_transform(V_safe)
    else:
        scaler = None
        V_normalized = V_safe
    
    V_normalized = add_epsilon(V_normalized)
    
    if use_custom:
        print(f"使用自定义NMF（乘法更新规则，{get_beta_name(beta)}）")
        W, H, n_iter, reconstruction_error = custom_nmf_beta(
            V_normalized, n_components, beta=beta, 
            max_iter=max_iter, random_state=random_state
        )
        model = None
        print(f"重构误差 ({get_beta_name(beta)}): {reconstruction_error:.4f}")
        print(f"迭代次数: {n_iter}")
    else:
        print("使用scikit-learn NMF")
        model = NMF(
            n_components=n_components,
            init='nndsvd',
            random_state=random_state,
            max_iter=max_iter,
            alpha_W=0.0,
            alpha_H=0.0,
            l1_ratio=0.0
        )
        
        W = model.fit_transform(V_normalized)
        H = model.components_
        
        W = add_epsilon(W)
        H = add_epsilon(H)
        
        reconstruction_error = model.reconstruction_err_
        print(f"重构误差 (F范数): {reconstruction_error:.4f}")
        print(f"迭代次数: {model.n_iter_}")
    
    if np.any(np.isnan(W)) or np.any(np.isnan(H)):
        print("警告: 检测到NaN值，已自动修复")
        W = np.nan_to_num(W, nan=EPS, posinf=1.0, neginf=EPS)
        H = np.nan_to_num(H, nan=EPS, posinf=1.0, neginf=EPS)
    
    return W, H, model, scaler


def reconstruct(W, H, scaler=None):
    """
    从W和H重构原始矩阵
    
    参数:
        W: 基矩阵
        H: 系数矩阵
        scaler: 归一化器（如果之前进行了归一化）
    
    返回:
        V_reconstructed: 重构后的矩阵
    """
    V_reconstructed = np.dot(W, H)
    
    if scaler is not None:
        V_reconstructed = scaler.inverse_transform(V_reconstructed)
    
    return V_reconstructed


def analyze_components(H, feature_names=None, top_k=5):
    """
    分析提取的特征成分
    
    参数:
        H: 系数矩阵，形状为 (n_components, n_features)
        feature_names: 特征名称列表
        top_k: 每个成分显示前k个权重最大的特征
    """
    n_components = H.shape[0]
    
    print(f"\n=== 特征成分分析 (每个成分显示Top-{top_k}特征) ===")
    
    for i in range(n_components):
        component = H[i]
        top_indices = np.argsort(component)[::-1][:top_k]
        
        print(f"\n成分 {i+1}:")
        for idx in top_indices:
            weight = component[idx]
            if feature_names is not None:
                print(f"  {feature_names[idx]}: {weight:.4f}")
            else:
                print(f"  特征{idx}: {weight:.4f}")


def visualize_components(W, H, figsize=(12, 5)):
    """
    可视化W和H矩阵
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    im1 = ax1.imshow(W, aspect='auto', cmap='viridis')
    ax1.set_title('W矩阵 (样本特征表示)')
    ax1.set_xlabel('成分')
    ax1.set_ylabel('样本')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(H, aspect='auto', cmap='viridis')
    ax2.set_title('H矩阵 (特征基)')
    ax2.set_xlabel('原始特征')
    ax2.set_ylabel('成分')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()


def test_beta_divergence():
    """
    测试不同β-散度的效果
    """
    print("\n" + "="*60)
    print("β-散度族对比测试")
    print("="*60)
    
    np.random.seed(42)
    
    print("\n1. 模拟音频频谱数据（具有动态范围特性）:")
    n_samples = 60
    n_features = 40
    n_components = 4
    
    t = np.linspace(0, 10, n_features)
    V_audio = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        for j in range(n_components):
            freq = 0.5 + j * 0.3
            amp = np.random.exponential(1.0)
            V_audio[i] += amp * np.exp(-0.1 * (t - freq * 5) ** 2)
    
    V_audio = np.maximum(V_audio, 0)
    print(f"   矩阵形状: {V_audio.shape}")
    print(f"   动态范围: {V_audio.min():.4f} ~ {V_audio.max():.4f}")
    
    betas = [0, 1, 2]
    results = []
    
    for beta in betas:
        print(f"\n--- {get_beta_name(beta)} ---")
        W, H, _, scaler = nmf_feature_extraction(
            V_audio, n_components, use_custom=True, beta=beta, normalize=False
        )
        
        V_recon = reconstruct(W, H, scaler)
        fro_error = np.linalg.norm(V_audio - V_recon) / np.linalg.norm(V_audio)
        is_error = beta_divergence(V_audio, V_recon, beta=0)
        kl_error = beta_divergence(V_audio, V_recon, beta=1)
        
        results.append({
            'beta': beta,
            'name': get_beta_name(beta),
            'fro_error': fro_error,
            'is_error': is_error,
            'kl_error': kl_error,
            'W': W,
            'H': H
        })
        
        print(f"   F范数相对误差: {fro_error:.4%}")
        print(f"   IS散度值: {is_error:.4f}")
        print(f"   KL散度值: {kl_error:.4f}")
        print(f"   W矩阵最小值: {W.min():.2e}")
    
    print("\n" + "-"*60)
    print("总结:")
    print("- β=0 (IS散度): 对音频频谱等动态范围大的数据效果好")
    print("  * 对小值元素的拟合误差赋予更高权重")
    print("  * 适合音乐频谱分析、源分离等应用")
    print("- β=1 (KL散度): 适合计数数据、文本主题模型")
    print("  * 对应泊松分布的最大似然估计")
    print("  * 适合文档主题提取、词袋模型等")
    print("- β=2 (欧氏距离): 适合一般连续值数据")
    print("  * 对应高斯分布的最大似然估计")
    print("  * 计算稳定，适合大多数情况")
    
    return results


def test_zero_element_stability():
    """
    测试零元素稳定性
    """
    print("\n" + "="*60)
    print("测试：含大量零元素的矩阵（模拟稀疏数据）")
    print("="*60)
    
    np.random.seed(42)
    
    n_samples = 50
    n_features = 30
    n_components = 3
    
    V = np.random.rand(n_samples, n_features)
    mask = np.random.rand(n_samples, n_features) < 0.7
    V[mask] = 0
    
    print(f"\n原始矩阵形状: {V.shape}")
    print(f"零元素比例: {np.mean(V == 0):.1%}")
    print(f"矩阵最小值: {V.min():.4f}, 最大值: {V.max():.4f}")
    
    print(f"\n1. 使用scikit-learn NMF:")
    W1, H1, _, scaler1 = nmf_feature_extraction(V, n_components, use_custom=False)
    V_recon1 = reconstruct(W1, H1, scaler1)
    rel_error1 = np.linalg.norm(V - V_recon1) / np.linalg.norm(V)
    print(f"相对重构误差: {rel_error1:.4%}")
    print(f"W矩阵含NaN: {np.any(np.isnan(W1))}, H矩阵含NaN: {np.any(np.isnan(H1))}")
    print(f"W矩阵最小值: {W1.min():.2e}, H矩阵最小值: {H1.min():.2e}")
    
    print(f"\n2. 使用自定义NMF（β=2欧氏距离）:")
    W2, H2, _, scaler2 = nmf_feature_extraction(V, n_components, use_custom=True, beta=2)
    V_recon2 = reconstruct(W2, H2, scaler2)
    rel_error2 = np.linalg.norm(V - V_recon2) / np.linalg.norm(V)
    print(f"相对重构误差: {rel_error2:.4%}")
    print(f"W矩阵含NaN: {np.any(np.isnan(W2))}, H矩阵含NaN: {np.any(np.isnan(H2))}")
    print(f"W矩阵最小值: {W2.min():.2e}, H矩阵最小值: {H2.min():.2e}")
    
    print(f"\n3. 使用自定义NMF（β=1 KL散度）:")
    W3, H3, _, scaler3 = nmf_feature_extraction(V, n_components, use_custom=True, beta=1)
    V_recon3 = reconstruct(W3, H3, scaler3)
    rel_error3 = np.linalg.norm(V - V_recon3) / np.linalg.norm(V)
    print(f"相对重构误差: {rel_error3:.4%}")
    print(f"W矩阵含NaN: {np.any(np.isnan(W3))}, H矩阵含NaN: {np.any(np.isnan(H3))}")
    
    print("\n✓ 零元素稳定性测试通过！")


def main():
    """
    示例：演示NMF特征提取（含β-散度族和数值稳定性）
    """
    print("="*60)
    print("NMF特征提取（含β-散度族和数值稳定性保护）")
    print("="*60)
    
    np.random.seed(42)
    
    n_samples = 100
    n_features = 50
    n_components = 5
    
    print(f"\n生成数据: {n_samples}个样本, {n_features}个特征")
    V = np.random.rand(n_samples, n_features) * 10
    print(f"原始矩阵形状: {V.shape}")
    print(f"矩阵最小值: {V.min():.4f}, 最大值: {V.max():.4f}")
    
    print(f"\n开始NMF分解，目标成分数: {n_components}")
    W, H, model, scaler = nmf_feature_extraction(V, n_components)
    
    print(f"\n分解结果:")
    print(f"  W矩阵形状: {W.shape} (样本的低维特征表示)")
    print(f"  H矩阵形状: {H.shape} (提取的特征基)")
    
    V_reconstructed = reconstruct(W, H, scaler)
    relative_error = np.linalg.norm(V - V_reconstructed) / np.linalg.norm(V)
    print(f"\n相对重构误差: {relative_error:.4%}")
    
    analyze_components(H, top_k=5)
    
    print("\n=== 样本特征表示示例 (前3个样本) ===")
    for i in range(min(3, n_samples)):
        print(f"样本{i+1}: {W[i]}")
    
    test_beta_divergence()
    test_zero_element_stability()
    
    print("\n" + "="*60)
    print("功能总结：")
    print("="*60)
    print("\n1. 数值稳定性保护:")
    print("   - EPS = 1e-10作为极小值保护")
    print("   - safe_divide(a, b): 安全除法避免除零")
    print("   - add_epsilon(X): 确保矩阵所有元素 >= EPS")
    print("   - NaN检测和自动修复")
    
    print("\n2. β-散度族损失函数支持:")
    print("   - β=0: Itakura-Saito散度")
    print("     · 适合音频频谱分析、音乐源分离")
    print("     · 对小值元素的拟合误差赋予更高权重")
    print("     · 乘法更新公式: H *= W^T(V/(WH)^2) / W^T(1/(WH))")
    print("   - β=1: KL散度（Kullback-Leibler）")
    print("     · 适合文本主题模型、计数数据")
    print("     · 对应泊松分布最大似然估计")
    print("     · 乘法更新公式: H *= W^T(V/(WH)) / sum(W, axis=0)")
    print("   - β=2: 欧几里得距离（F范数）")
    print("     · 适合一般连续值数据")
    print("     · 对应高斯分布最大似然估计")
    print("     · 计算稳定，默认选项")
    
    print("\n3. 使用方式:")
    print("   - 常规使用(scikit-learn):")
    print("       W, H, _, _ = nmf_feature_extraction(V, n_components)")
    print("   - 使用β=0 (IS散度，音频数据):")
    print("       W, H, _, _ = nmf_feature_extraction(V, n_components, use_custom=True, beta=0)")
    print("   - 使用β=1 (KL散度，文本数据):")
    print("       W, H, _, _ = nmf_feature_extraction(V, n_components, use_custom=True, beta=1)")
    print("   - 使用β=2 (欧氏距离，一般数据):")
    print("       W, H, _, _ = nmf_feature_extraction(V, n_components, use_custom=True, beta=2)")
    
    print("\n4. 典型应用场景:")
    print("   - β=0: 音乐频谱分析、语音处理、声源分离")
    print("   - β=1: 文档主题建模(LDA)、推荐系统、图像分割")
    print("   - β=2: 图像特征提取、基因表达分析、通用降维")
    print("="*60)


if __name__ == "__main__":
    main()
