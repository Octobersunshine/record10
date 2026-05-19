import requests
import numpy as np

def test_default_nlags():
    """测试默认延迟阶数计算"""
    from main import calculate_default_nlags
    
    print("=" * 60)
    print("测试默认延迟阶数计算策略:")
    print("=" * 60)
    
    test_cases = [10, 20, 30, 50, 100, 200, 500]
    for nobs in test_cases:
        nlags = calculate_default_nlags(nobs)
        print(f"序列长度={nobs:3d}, 默认延迟阶数={nlags:3d}")
    
    print()

def generate_white_noise(length=200):
    """生成白噪声序列用于测试"""
    return np.random.randn(length).tolist()

def generate_autocorrelated_series(length=200):
    """生成自相关序列（非白噪声）用于测试"""
    np.random.seed(42)
    noise = np.random.randn(length)
    ar1 = np.zeros(length)
    ar1[0] = noise[0]
    for i in range(1, length):
        ar1[i] = 0.8 * ar1[i-1] + noise[i]
    return ar1.tolist()

def generate_long_period_series(length=200, period=30):
    """生成长周期时间序列用于测试"""
    np.random.seed(42)
    t = np.arange(length)
    signal = np.sin(2 * np.pi * t / period) + 0.2 * np.random.randn(length)
    return signal.tolist()

def print_ljung_box_summary(result):
    """打印 Ljung-Box 检验摘要"""
    lb = result['ljung_box']
    print(f"\nLjung-Box 白噪声检验结果:")
    print(f"  显著性水平: {lb['significance_level']}")
    print(f"  是否为白噪声: {lb['is_white_noise']}")
    print(f"  最小p值: {min(lb['p_values']):.6f}")
    print(f"\n  关键延迟阶数的p值:")
    display_lags = [1, 5, 10, 20]
    for lag in display_lags:
        if lag <= len(lb['p_values']):
            idx = lag - 1
            p_val = lb['p_values'][idx]
            significance = "*" if p_val < lb['significance_level'] else ""
            print(f"    lag={lag:2d}: p={p_val:.6f} {significance}")
    print("  (* 表示 p < 显著性水平，拒绝白噪声假设)")

def test_api():
    url = "http://localhost:8000/api/correlation"
    
    print("=" * 60)
    print("测试1: 长周期序列 (长度=200, 周期=30) - 非白噪声")
    print("=" * 60)
    
    long_series = generate_long_period_series(length=200, period=30)
    
    payload = {
        "data": long_series,
        "alpha": 0.05
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        max_lag = max(result['acf']['lags'])
        print(f"使用的延迟阶数: {max_lag}")
        print(f"ACF计算的阶数足够捕捉周期为30的信号: {max_lag >= 30}")
        
        print("\n前40阶ACF值 (显示周期特征):")
        acf_values = result['acf']['acf_values'][:41]
        for i in range(0, 41, 5):
            print(f"  lag={i:2d}: acf={acf_values[i]:.4f}")
        
        print_ljung_box_summary(result)
        print(f"\n预期: is_white_noise=False (周期序列不是白噪声)")
        
    except Exception as e:
        print(f"测试失败: {e}")
        return
    
    print("\n" + "=" * 60)
    print("测试2: 白噪声序列")
    print("=" * 60)
    
    white_noise = generate_white_noise(length=100)
    
    payload = {
        "data": white_noise,
        "alpha": 0.05
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        print_ljung_box_summary(result)
        
    except Exception as e:
        print(f"测试失败: {e}")
        return
    
    print("\n" + "=" * 60)
    print("测试3: AR(1) 自相关序列 - 非白噪声")
    print("=" * 60)
    
    ar_series = generate_autocorrelated_series(length=200)
    
    payload = {
        "data": ar_series,
        "alpha": 0.05
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        print_ljung_box_summary(result)
        print(f"\n预期: is_white_noise=False (AR序列不是白噪声)")
        
    except Exception as e:
        print(f"测试失败: {e}")
        return
    
    print("\n" + "=" * 60)
    print("测试4: 自定义延迟阶数和显著性水平")
    print("=" * 60)
    
    custom_series = generate_long_period_series(length=100, period=25)
    
    payload = {
        "data": custom_series,
        "nlags": 40,
        "alpha": 0.01
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        max_lag = max(result['acf']['lags'])
        print(f"自定义延迟阶数: {max_lag}")
        print(f"使用的显著性水平: {result['ljung_box']['significance_level']}")
        print(f"是否为白噪声: {result['ljung_box']['is_white_noise']}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        return
    
    print("\n" + "=" * 60)
    print("所有测试成功!")
    print("=" * 60)

if __name__ == "__main__":
    test_default_nlags()
    test_api()