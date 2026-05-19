import numpy as np
from mcmc_changepoint_detection import MCMCChangepointDetection, generate_test_data


def main():
    np.random.seed(42)
    
    print("=== 示例1: 基本使用 ===")
    data1 = generate_test_data(n=150, changepoints=[50, 100], means=[0, 4, 1], stds=[1, 1, 0.8])
    
    bcd1 = MCMCChangepointDetection(data1, max_changepoints=5)
    print("运行MCMC采样...")
    bcd1.run_mcmc(n_iterations=15000, burn_in=3000, thin=5)
    
    posterior_probs = bcd1.compute_posterior_probs()
    map_cp = bcd1.get_map_estimate()
    
    print(f"\n真实变点位置: [50, 100]")
    print(f"MAP估计变点位置: {map_cp}")
    print(f"\n高概率变点位置 (>0.3):")
    high_prob = [(i, p) for i, p in enumerate(posterior_probs) if p > 0.3]
    for pos, prob in sorted(high_prob, key=lambda x: -x[1]):
        print(f"  位置 {pos}: 概率 = {prob:.3f}")
    
    print("\n" + "="*50)
    print("\n=== 示例2: 自定义时间序列 ===")
    
    def my_time_series(n=100):
        data = np.zeros(n)
        data[:30] = np.random.normal(0, 1, 30)
        data[30:70] = np.random.normal(5, 1.2, 40)
        data[70:] = np.random.normal(-2, 0.8, 30)
        return data
    
    data2 = my_time_series(100)
    bcd2 = MCMCChangepointDetection(data2, max_changepoints=3)
    print("运行MCMC采样...")
    bcd2.run_mcmc(n_iterations=10000, burn_in=2000, thin=5)
    
    probs2 = bcd2.compute_posterior_probs()
    map2 = bcd2.get_map_estimate()
    
    print(f"\n真实变点位置: [30, 70]")
    print(f"MAP估计变点位置: {map2}")
    
    print("\n=== 返回值说明 ===")
    print("- run_mcmc(): 返回所有MCMC样本列表")
    print("- compute_posterior_probs(): 返回每个位置的变点后验概率数组")
    print("- get_map_estimate(): 返回最大后验估计的变点位置列表")
    
    print("\n=== 简单调用示例 ===")
    print("""
    # 最简调用方式
    your_data = [...]  # 你的时间序列数据
    model = MCMCChangepointDetection(your_data)
    model.run_mcmc()
    posterior_probabilities = model.compute_posterior_probs()
    print(posterior_probabilities)  # 每个位置的变点后验概率
    """)


if __name__ == '__main__':
    main()
