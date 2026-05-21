"""
已实现波动率 (Realized Volatility) 计算与Realized GARCH使用示例

本示例展示如何:
1. 从高频数据计算各种已实现波动率测度
2. 使用Realized GARCH进行波动率预测
3. 对比标准GARCH和Realized GARCH的效果
"""

import numpy as np
import pandas as pd
from scipy import stats


def calculate_realized_volatility(intraday_returns, freq='5min', method='rv'):
    """
    从高频收益率计算已实现波动率
    
    参数:
        intraday_returns: 高频收益率 (pd.Series, 带时间戳索引)
        freq: 采样频率
        method: 计算方法
            - 'rv': 已实现方差 (Realized Variance)
            - 'rv5': 5分钟已实现波动率 (标准)
            - 'rk': 已实现核估计 (Realized Kernel, 考虑市场微观结构噪声)
            - 'bv': 双幂变差 (Bi-power Variation, 稳健于跳跃)
    
    返回:
        rv_daily: 日度已实现波动率序列
    """
    if method == 'rv':
        rv_daily = (intraday_returns ** 2).resample('D').sum()
    elif method == 'rk':
        def realized_kernel(x):
            n = len(x)
            if n < 2:
                return np.nan
            H = min(10, n // 10)
            rk = np.sum(x ** 2)
            for h in range(1, H + 1):
                weight = 1 - h / (H + 1)
                rk += 2 * weight * np.sum(x[h:] * x[:-h])
            return rk
        rv_daily = intraday_returns.resample('D').apply(realized_kernel)
    elif method == 'bv':
        def bipower_variation(x):
            n = len(x)
            if n < 2:
                return np.nan
            mu1 = np.sqrt(2 / np.pi)
            bv = (np.pi / 2) * np.sum(np.abs(x[1:]) * np.abs(x[:-1]))
            return bv
        rv_daily = intraday_returns.resample('D').apply(bipower_variation)
    else:
        rv_daily = (intraday_returns ** 2).resample('D').sum()
    
    return rv_daily.dropna()


def prepare_realized_garch_data(csv_path=None, intraday_freq='5min'):
    """
    准备Realized GARCH所需的数据
    
    参数:
        csv_path: 高频数据CSV路径，格式应为:
                  timestamp, return, price
        intraday_freq: 日内频率
    
    返回:
        daily_returns: 日收益率
        realized_measures: 已实现波动率测度字典
    """
    if csv_path is None:
        print("使用模拟高频数据...")
        from garch_model import simulate_high_frequency_data
        daily_returns, rv, rk = simulate_high_frequency_data(
            n_days=500, n_intraday=78
        )
        realized_measures = {
            'RV': rv,
            'RK': rk
        }
        return daily_returns, realized_measures
    
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    if 'return' not in df.columns:
        if 'price' in df.columns:
            df['return'] = np.log(df['price'] / df['price'].shift(1))
            df = df.dropna()
        else:
            raise ValueError("数据需要包含'return'或'price'列")
    
    intraday_returns = df['return']
    
    daily_returns = intraday_returns.resample('D').sum()
    daily_returns = daily_returns[daily_returns != 0]
    
    realized_measures = {}
    for method in ['rv', 'rk', 'bv']:
        rv = calculate_realized_volatility(intraday_returns, method=method)
        realized_measures[method.upper()] = rv
    
    return daily_returns, realized_measures


def run_realized_garch_demo():
    """运行Realized GARCH完整演示"""
    print("=" * 70)
    print("Realized GARCH 完整演示 - 利用高频数据提高预测精度")
    print("=" * 70)
    
    daily_returns, realized_measures = prepare_realized_garch_data()
    
    print("\n可用的已实现波动率测度:")
    for name, measure in realized_measures.items():
        print(f"  - {name}: {len(measure)}个观测值")
    
    from garch_model import compare_garch_realized_garch
    
    print("\n" + "=" * 70)
    print("方案1: 使用标准已实现方差 (RV)")
    print("=" * 70)
    results_rv = compare_garch_realized_garch(
        returns=daily_returns,
        realized_measure=realized_measures['RV'],
        forecast_horizon=5
    )
    
    if 'RK' in realized_measures:
        print("\n" + "=" * 70)
        print("方案2: 使用已实现核估计 (RK) - 更稳健，考虑微观结构噪声")
        print("=" * 70)
        results_rk = compare_garch_realized_garch(
            returns=daily_returns,
            realized_measure=realized_measures['RK'],
            forecast_horizon=5
        )
    
    print("\n" + "=" * 70)
    print("实践建议:")
    print("=" * 70)
    print("""
    1. 数据频率选择:
       - 5分钟是标准选择（平衡精度与噪声）
       - 过低（如1小时）损失信息
       - 过高（如1分钟）微观结构噪声增大
    
    2. 已实现测度选择:
       - RV: 标准，计算简单，但受跳跃和噪声影响
       - RK: 推荐，对市场微观结构噪声更稳健
       - BV: 稳健于跳跃，但效率略低
    
    3. 样本大小:
       - 至少需要1年（250交易日）数据
       - 2-3年数据效果更佳
    
    4. 典型应用:
       - 期权定价: 需要精确波动率预测
       - 风险管理: VaR计算更准确
       - 组合优化: 协方差矩阵估计更精确
    """)


if __name__ == "__main__":
    run_realized_garch_demo()
