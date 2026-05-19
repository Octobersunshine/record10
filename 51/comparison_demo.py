import numpy as np
import matplotlib.pyplot as plt
from mcmc_changepoint_detection import MCMCChangepointDetection
from improved_mcmc_changepoint import ImprovedMCMCChangepointDetection


def generate_challenging_data():
    """生成具有挑战性的数据，容易产生多峰后验"""
    np.random.seed(123)
    
    n = 300
    data = np.zeros(n)
    
    data[0:80] = np.random.normal(0, 1, 80)
    data[80:150] = np.random.normal(2, 1.2, 70)
    data[150:220] = np.random.normal(2.5, 1.0, 70)
    data[220:] = np.random.normal(-1, 1.1, 80)
    
    true_cp = [80, 150, 220]
    return data, true_cp


def compare_models():
    data, true_cp = generate_challenging_data()
    
    print("="*70)
    print("变点检测模型比较：解决变点数目可辨识性问题")
    print("="*70)
    print(f"\n真实变点位置: {true_cp}")
    print(f"真实变点数量: {len(true_cp)}")
    
    print("\n" + "-"*70)
    print("模型1: 原始MCMC (简单泊松先验 + 条件似然)")
    print("-"*70)
    
    model1 = MCMCChangepointDetection(data, max_changepoints=8)
    model1.run_mcmc(n_iterations=20000, burn_in=5000, thin=10)
    
    ncp_dist1 = np.zeros(9)
    for s in model1.samples:
        if len(s) <= 8:
            ncp_dist1[len(s)] += 1
    ncp_dist1 /= len(model1.samples)
    
    print("\n变点数量后验分布:")
    for i in range(9):
        if ncp_dist1[i] > 0.01:
            print(f"  {i} 个变点: {ncp_dist1[i]:.4f}")
    
    peaks1 = [i for i in range(9) if ncp_dist1[i] > 0.15]
    print(f"\n主要峰值: {peaks1} 个变点")
    if len(peaks1) > 1:
        print("⚠ 检测到多峰后验分布!")
    
    probs1 = np.zeros(len(data))
    for s in model1.samples:
        for cp in s:
            if 0 <= cp < len(data):
                probs1[cp] += 1
    probs1 /= len(model1.samples)
    
    high_prob1 = [(i, p) for i, p in enumerate(probs1) if p > 0.3]
    print(f"\n高概率变点数量 (p>0.3): {len(high_prob1)}")
    
    print("\n" + "-"*70)
    print("模型2: 改进MCMC (正则化先验 + 边际似然 + BMA)")
    print("-"*70)
    
    model2 = ImprovedMCMCChangepointDetection(
        data, 
        max_changepoints=8,
        prior_type='regularized_poisson',
        penalty_strength=2.0
    )
    model2.run_mcmc(n_iterations=20000, burn_in=5000, thin=10, verbose=False)
    
    model2.compute_posterior_probs()
    
    print("\n变点数量后验分布:")
    for i in range(9):
        if model2.ncp_posterior[i] > 0.01:
            print(f"  {i} 个变点: {model2.ncp_posterior[i]:.4f}")
    
    peaks2 = [i for i in range(9) if model2.ncp_posterior[i] > 0.15]
    print(f"\n主要峰值: {peaks2} 个变点")
    if len(peaks2) == 1:
        print("✓ 单峰后验分布!")
    elif len(peaks2) > 1:
        print(f"⚠ 仍有 {len(peaks2)} 个峰值")
    
    high_prob2 = [(i, p) for i, p in enumerate(model2.posterior_probs) if p > 0.3]
    print(f"\n高概率变点数量 (p>0.3): {len(high_prob2)}")
    
    print("\n" + "="*70)
    print("改进效果总结")
    print("="*70)
    
    print(f"\n变点数量峰值: {len(peaks1)} → {len(peaks2)}")
    if len(peaks1) > len(peaks2):
        print("✓ 成功减少了多峰问题")
    
    print(f"\n高概率变点数: {len(high_prob1)} → {len(high_prob2)}")
    if len(high_prob1) > len(high_prob2):
        print("✓ 减少了虚假变点")
    
    print(f"\n真实变点的后验概率:")
    for cp in true_cp:
        p1 = probs1[cp] if cp < len(probs1) else 0
        p2 = model2.posterior_probs[cp] if cp < len(model2.posterior_probs) else 0
        print(f"  位置 {cp}: {p1:.3f} → {p2:.3f} {'↑' if p2 > p1 else '↓'}")
    
    print("\n" + "="*70)
    print("建议")
    print("="*70)
    print("1. 总是检查变点数量的后验分布是否有多个峰值")
    print("2. 使用贝叶斯模型平均(BMA)而不是仅依赖MAP估计")
    print("3. 对不确定的数据，使用更强的正则化先验")
    print("4. 报告变点的后验概率，而不仅仅是点估计")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    axes[0].plot(data, 'b-', linewidth=1, alpha=0.7)
    for cp in true_cp:
        axes[0].axvline(x=cp, color='k', linestyle='--', linewidth=2, label='True Changepoint' if cp == true_cp[0] else "")
    axes[0].set_ylabel('Value')
    axes[0].set_title('Time Series Data with True Changepoints')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(probs1, 'r-', linewidth=2, label='Original Model')
    axes[1].fill_between(range(len(data)), probs1, alpha=0.2, color='red')
    axes[1].set_ylabel('Probability')
    axes[1].set_title('Original Model - Changepoint Posterior Probability')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)
    
    axes[2].plot(model2.posterior_probs, 'g-', linewidth=2, label='Improved Model (with BMA)')
    axes[2].fill_between(range(len(data)), model2.posterior_probs, alpha=0.2, color='green')
    axes[2].set_xlabel('Position')
    axes[2].set_ylabel('Probability')
    axes[2].set_title('Improved Model - Changepoint Posterior Probability')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.show()
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    axes2[0].bar(range(9), ncp_dist1, color='red', alpha=0.6, width=0.6)
    axes2[0].set_xlabel('Number of Changepoints')
    axes2[0].set_ylabel('Probability')
    axes2[0].set_title('Original Model - Number of Changepoints Distribution')
    axes2[0].grid(True, alpha=0.3)
    axes2[0].set_xticks(range(9))
    
    axes2[1].bar(range(9), model2.ncp_posterior, color='green', alpha=0.6, width=0.6)
    axes2[1].set_xlabel('Number of Changepoints')
    axes2[1].set_ylabel('Probability')
    axes2[1].set_title('Improved Model - Number of Changepoints Distribution')
    axes2[1].grid(True, alpha=0.3)
    axes2[1].set_xticks(range(9))
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    compare_models()
