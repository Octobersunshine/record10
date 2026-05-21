import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

def simple_garch_forecast(returns, horizon=5, dist='t'):
    """
    简化版GARCH(1,1)拟合和波动率预测 (使用厚尾分布)
    
    参数:
        returns: 收益率序列
        horizon: 预测期数
        dist: 残差分布: 't' (学生t分布, 默认), 'skewt' (偏t分布), 'normal'
    
    返回:
        forecast_volatility: 预测的波动率序列
    """
    dist_mapping = {
        'normal': 'Normal',
        't': 'StudentsT',
        'skewt': 'SkewStudent'
    }
    
    returns_scaled = returns * 100
    
    model = arch_model(
        returns_scaled, 
        vol='Garch', 
        p=1, 
        q=1,
        dist=dist_mapping.get(dist.lower(), 'StudentsT')
    )
    result = model.fit(disp='off')
    
    forecast = result.forecast(horizon=horizon)
    forecast_vol = np.sqrt(forecast.variance.iloc[-1]) / 100
    
    return forecast_vol, result

if __name__ == "__main__":
    np.random.seed(42)
    
    n = 500
    omega, alpha, beta = 0.1, 0.15, 0.8
    nu = 5
    
    sim_returns = np.zeros(n)
    sim_volatility = np.zeros(n)
    for i in range(1, n):
        sim_volatility[i] = np.sqrt(omega + alpha * sim_returns[i-1]**2 + beta * sim_volatility[i-1]**2)
        sim_returns[i] = sim_volatility[i] * stats.t.rvs(df=nu, size=1)[0]
    
    print("使用学生t分布的GARCH(1,1)模型:")
    forecast, result = simple_garch_forecast(sim_returns, horizon=5, dist='t')
    
    print(f"\n估计的自由度 nu: {result.params['nu']:.4f}")
    print(f"真实自由度 nu: {nu}")
    print("\n波动率预测:")
    print(forecast)
