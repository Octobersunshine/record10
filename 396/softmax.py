import numpy as np


def softmax(x, T=1.0, axis=-1):
    x = np.array(x, dtype=np.float64)
    x_scaled = x / T
    x_max = np.max(x_scaled, axis=axis, keepdims=True)
    exp_x = np.exp(x_scaled - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_softmax(x, T=1.0, axis=-1):
    x = np.array(x, dtype=np.float64)
    x_scaled = x / T
    x_max = np.max(x_scaled, axis=axis, keepdims=True)
    log_sum_exp = x_max + np.log(np.sum(np.exp(x_scaled - x_max), axis=axis, keepdims=True))
    return x_scaled - log_sum_exp


def entropy(probs):
    probs = np.array(probs, dtype=np.float64)
    probs = np.clip(probs, 1e-12, None)
    return -np.sum(probs * np.log(probs), axis=-1)


def softmax_with_info(x, T=1.0, axis=-1):
    probs = softmax(x, T=T, axis=axis)
    log_probs = log_softmax(x, T=T, axis=axis)
    ent = entropy(probs)
    return {
        "probs": probs,
        "log_probs": log_probs,
        "entropy": ent,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("测试1: 温度参数T对Softmax的影响")
    print("=" * 60)
    x = np.array([2.0, 1.0, 0.1])
    print(f"原始向量: {x}\n")

    for T in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        info = softmax_with_info(x, T=T)
        print(f"T={T:>4.1f} | Softmax: {info['probs']} | 熵: {info['entropy']:.4f}")

    print(f"\n最大可能熵 (均匀分布): {np.log(len(x)):.4f}")

    print("\n" + "=" * 60)
    print("测试2: 批量Softmax (多个向量同时计算)")
    print("=" * 60)
    batch = np.array([
        [2.0, 1.0, 0.1],
        [1000.0, 1001.0, 999.0],
        [-1000.0, -1001.0, -999.0],
        [1.0, 1.0, 1.0],
    ])
    print(f"输入批次 (shape={batch.shape}):\n{batch}\n")

    info = softmax_with_info(batch, T=1.0)
    print(f"Softmax结果:\n{info['probs']}")
    print(f"每行和: {np.sum(info['probs'], axis=-1)}")
    print(f"每行熵: {info['entropy']}")

    print("\n" + "=" * 60)
    print("测试3: 批量Softmax + 温度T=0.5")
    print("=" * 60)
    info = softmax_with_info(batch, T=0.5)
    print(f"Softmax结果:\n{info['probs']}")
    print(f"每行和: {np.sum(info['probs'], axis=-1)}")
    print(f"每行熵: {info['entropy']}")

    print("\n" + "=" * 60)
    print("测试4: 单向量输入 (向后兼容)")
    print("=" * 60)
    result = softmax([2.0, 1.0, 0.1])
    print(f"Softmax([2.0, 1.0, 0.1]) = {result}")
    print(f"和: {np.sum(result)}")

    print("\n" + "=" * 60)
    print("温度参数原理说明:")
    print("=" * 60)
    print("softmax(x, T) = exp(x/T) / sum(exp(x/T))")
    print("T→0: 分布趋近one-hot (argmax), 熵→0")
    print("T=1: 标准Softmax")
    print("T→∞: 分布趋近均匀, 熵→ln(n)")
