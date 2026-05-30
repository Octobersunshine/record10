import numpy as np
from typing import Callable, Tuple


def monte_carlo_integrate(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    N: int
) -> Tuple[float, float]:
    """
    一维蒙特卡洛积分
    
    参数:
        f: 被积函数，接受numpy数组输入
        a: 积分区间左端点
        b: 积分区间右端点
        N: 随机采样点数
    
    返回:
        (积分近似值, 估计误差(标准差))
    """
    x = np.random.uniform(a, b, N)
    fx = f(x)
    integral = (b - a) * np.mean(fx)
    error = (b - a) * np.std(fx, ddof=1) / np.sqrt(N)
    return integral, error


if __name__ == "__main__":
    def f(x):
        return x ** 2
    
    a, b = 0, 1
    N = 100000
    
    integral, error = monte_carlo_integrate(f, a, b, N)
    exact = 1/3
    
    print(f"被积函数: f(x) = x^2")
    print(f"积分区间: [{a}, {b}]")
    print(f"采样点数: {N}")
    print(f"蒙特卡洛积分结果: {integral:.6f}")
    print(f"精确值: {exact:.6f}")
    print(f"估计误差: {error:.6f}")
    print(f"实际误差: {abs(integral - exact):.6f}")
