import numpy as np
from scipy.stats import norm


def black_scholes_monte_carlo(S0, K, T, r, sigma, num_simulations=100000, seed=42):
    """
    蒙特卡洛模拟欧式看涨期权定价（Black-Scholes模型）
    使用Box-Muller变换生成独立正态随机数，确保z1和z2都被使用，避免相关性问题
    
    参数:
        S0: 标的资产当前价格
        K: 行权价格
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率
        num_simulations: 模拟次数
        seed: 随机种子
    
    返回:
        option_price: 期权价格
        delta: 期权Delta
    """
    np.random.seed(seed)
    
    n_pairs = (num_simulations + 1) // 2
    u1 = np.random.uniform(0, 1, n_pairs)
    u2 = np.random.uniform(0, 1, n_pairs)
    
    R = np.sqrt(-2 * np.log(u1))
    theta = 2 * np.pi * u2
    
    z1 = R * np.cos(theta)
    z2 = R * np.sin(theta)
    
    z = np.concatenate([z1, z2])[:num_simulations]
    
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    
    payoff = np.maximum(ST - K, 0)
    
    option_price = np.exp(-r * T) * np.mean(payoff)
    
    indicator = (ST > K).astype(float)
    delta = np.exp(-r * T) * np.mean(indicator * ST / S0)
    
    return option_price, delta


def black_scholes_analytical(S0, K, T, r, sigma):
    """
    Black-Scholes解析解，用于验证蒙特卡洛结果
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    
    return call_price, delta


def laguerre_basis(x, degree=3):
    """
    Laguerre多项式基函数，用于LSM回归
    """
    basis = [np.ones_like(x)]
    if degree >= 1:
        basis.append(np.exp(-x / 2))
    if degree >= 2:
        basis.append(np.exp(-x / 2) * (1 - x))
    if degree >= 3:
        basis.append(np.exp(-x / 2) * (1 - 2 * x + x**2 / 2))
    return np.column_stack(basis)


def bermudan_lsm(S0, K, T, r, sigma, exercise_dates, num_simulations=100000, seed=42):
    """
    最小二乘蒙特卡洛（LSM）定价百慕大看涨期权
    
    参数:
        S0: 标的资产当前价格
        K: 行权价格
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率
        exercise_dates: 行权日期列表（相对于现在的年数，从小到大排列，必须包含T）
        num_simulations: 模拟次数
        seed: 随机种子
    
    返回:
        option_price: 期权价格
    """
    np.random.seed(seed)
    
    exercise_dates = np.array(sorted(exercise_dates))
    if not np.isclose(exercise_dates[-1], T):
        raise ValueError("最后一个行权日必须等于到期时间T")
    
    dt = np.diff(np.concatenate(([0], exercise_dates)))
    num_exercises = len(exercise_dates)
    
    S = np.zeros((num_simulations, num_exercises))
    S[:, 0] = S0
    
    n_pairs = (num_simulations + 1) // 2
    
    for i in range(1, num_exercises):
        u1 = np.random.uniform(0, 1, n_pairs)
        u2 = np.random.uniform(0, 1, n_pairs)
        R = np.sqrt(-2 * np.log(u1))
        theta = 2 * np.pi * u2
        z1 = R * np.cos(theta)
        z2 = R * np.sin(theta)
        z = np.concatenate([z1, z2])[:num_simulations]
        
        S[:, i] = S[:, i-1] * np.exp((r - 0.5 * sigma**2) * dt[i] + sigma * np.sqrt(dt[i]) * z)
    
    payoff = np.maximum(S - K, 0)
    
    V = payoff[:, -1].copy()
    
    for t in range(num_exercises - 2, -1, -1):
        dt_step = exercise_dates[t+1] - exercise_dates[t]
        
        in_the_money = payoff[:, t] > 0
        
        if np.sum(in_the_money) > 0:
            X = S[in_the_money, t] / K
            Y = V[in_the_money] * np.exp(-r * dt_step)
            
            basis = laguerre_basis(X)
            
            beta = np.linalg.lstsq(basis, Y, rcond=None)[0]
            continuation = basis @ beta
            
            exercise_now = payoff[in_the_money, t]
            
            V[in_the_money] = np.where(exercise_now > continuation, exercise_now, 
                                      V[in_the_money] * np.exp(-r * dt_step))
            
            V[~in_the_money] = V[~in_the_money] * np.exp(-r * dt_step)
        else:
            V = V * np.exp(-r * dt_step)
    
    option_price = np.mean(V)
    
    return option_price


def verify_random_numbers(num_pairs=100000, seed=42):
    """
    验证Box-Muller生成的随机数的独立性和统计特性
    """
    np.random.seed(seed)
    
    u1 = np.random.uniform(0, 1, num_pairs)
    u2 = np.random.uniform(0, 1, num_pairs)
    
    R = np.sqrt(-2 * np.log(u1))
    theta = 2 * np.pi * u2
    
    z1 = R * np.cos(theta)
    z2 = R * np.sin(theta)
    
    correlation = np.corrcoef(z1, z2)[0, 1]
    
    print("=" * 50)
    print("随机数统计特性验证")
    print("=" * 50)
    print(f"z1 均值: {np.mean(z1):.6f} (期望值: 0)")
    print(f"z1 方差: {np.var(z1):.6f} (期望值: 1)")
    print(f"z2 均值: {np.mean(z2):.6f} (期望值: 0)")
    print(f"z2 方差: {np.var(z2):.6f} (期望值: 1)")
    print(f"z1与z2相关系数: {correlation:.6f} (期望值: 0)")
    print("=" * 50)
    
    return correlation


if __name__ == "__main__":
    S0 = 100
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2
    
    verify_random_numbers()
    
    mc_price, mc_delta = black_scholes_monte_carlo(S0, K, T, r, sigma, num_simulations=100000)
    bs_price, bs_delta = black_scholes_analytical(S0, K, T, r, sigma)
    
    print("=" * 50)
    print("欧式看涨期权定价结果")
    print("=" * 50)
    print(f"参数: S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")
    print("-" * 50)
    print(f"蒙特卡洛模拟价格: {mc_price:.6f}")
    print(f"Black-Scholes解析价格: {bs_price:.6f}")
    print(f"价格差异: {abs(mc_price - bs_price):.6f}")
    print("-" * 50)
    print(f"蒙特卡洛模拟Delta: {mc_delta:.6f}")
    print(f"Black-Scholes解析Delta: {bs_delta:.6f}")
    print(f"Delta差异: {abs(mc_delta - bs_delta):.6f}")
    print("=" * 50)
    
    print()
    print("=" * 50)
    print("百慕大看涨期权LSM定价结果")
    print("=" * 50)
    print(f"参数: S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")
    print("-" * 50)
    
    exercise_dates_1 = [0.5, 1.0]
    bermudan_price_1 = bermudan_lsm(S0, K, T, r, sigma, exercise_dates_1, num_simulations=50000)
    print(f"2个行权日 {exercise_dates_1}: {bermudan_price_1:.6f}")
    
    exercise_dates_2 = [0.25, 0.5, 0.75, 1.0]
    bermudan_price_2 = bermudan_lsm(S0, K, T, r, sigma, exercise_dates_2, num_simulations=50000)
    print(f"4个行权日 {exercise_dates_2}: {bermudan_price_2:.6f}")
    
    exercise_dates_12 = [i/12 for i in range(1, 13)]
    bermudan_price_12 = bermudan_lsm(S0, K, T, r, sigma, exercise_dates_12, num_simulations=50000)
    print(f"12个行权日 (每月): {bermudan_price_12:.6f}")
    
    print("-" * 50)
    print(f"欧式期权价格 (参考): {bs_price:.6f}")
    print(f"美式期权价格应 >= 欧式价格 (行权机会更多)")
    print("=" * 50)
