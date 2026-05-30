def knapsack_dp_2d(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], values[i - 1] + dp[i - 1][w - weights[i - 1]])
            else:
                dp[i][w] = dp[i - 1][w]
    
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(i - 1)
            w -= weights[i - 1]
    
    selected.reverse()
    return dp[n][capacity], selected


def knapsack_dp_1d(weights, values, capacity):
    n = len(weights)
    dp = [0] * (capacity + 1)
    selected = [[False] * (capacity + 1) for _ in range(n)]
    
    for i in range(n):
        for w in range(capacity, weights[i] - 1, -1):
            if dp[w - weights[i]] + values[i] > dp[w]:
                dp[w] = dp[w - weights[i]] + values[i]
                selected[i][w] = True
    
    selected_items = []
    w = capacity
    for i in range(n - 1, -1, -1):
        if selected[i][w]:
            selected_items.append(i)
            w -= weights[i]
    
    selected_items.reverse()
    return dp[capacity], selected_items


if __name__ == "__main__":
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 8
    
    print("=" * 60)
    print("测试用例 1: 小规模数据")
    print("=" * 60)
    print(f"物品重量: {weights}")
    print(f"物品价值: {values}")
    print(f"背包容量: {capacity}")
    
    max_value_2d, selected_2d = knapsack_dp_2d(weights, values, capacity)
    max_value_1d, selected_1d = knapsack_dp_1d(weights, values, capacity)
    
    print(f"\n二维DP结果:")
    print(f"  最大价值: {max_value_2d}")
    print(f"  选中物品索引: {selected_2d}")
    
    print(f"\n一维DP结果:")
    print(f"  最大价值: {max_value_1d}")
    print(f"  选中物品索引: {selected_1d}")
    
    print(f"\n结果一致性验证: {'通过' if max_value_2d == max_value_1d else '失败'}")
    
    print("\n" + "=" * 60)
    print("测试用例 2: 中等规模数据")
    print("=" * 60)
    
    import random
    random.seed(42)
    n = 50
    capacity_m = 200
    weights_m = [random.randint(1, 30) for _ in range(n)]
    values_m = [random.randint(1, 50) for _ in range(n)]
    
    max_value_2d_m, selected_2d_m = knapsack_dp_2d(weights_m, values_m, capacity_m)
    max_value_1d_m, selected_1d_m = knapsack_dp_1d(weights_m, values_m, capacity_m)
    
    print(f"物品数量: {n}, 背包容量: {capacity_m}")
    print(f"二维DP最大价值: {max_value_2d_m}")
    print(f"一维DP最大价值: {max_value_1d_m}")
    print(f"结果一致性: {'通过' if max_value_2d_m == max_value_1d_m else '失败'}")
    
    weight_2d = sum(weights_m[i] for i in selected_2d_m)
    value_2d = sum(values_m[i] for i in selected_2d_m)
    weight_1d = sum(weights_m[i] for i in selected_1d_m)
    value_1d = sum(values_m[i] for i in selected_1d_m)
    
    print(f"\n二维DP选中物品: 总重量={weight_2d}, 总价值={value_2d}")
    print(f"一维DP选中物品: 总重量={weight_1d}, 总价值={value_1d}")
    
    print("\n" + "=" * 60)
    print("内存复杂度分析")
    print("=" * 60)
    n_big = 1000
    c_big = 10000
    import sys
    dp_2d_est = (n_big + 1) * (c_big + 1) * 4 / 1024 / 1024
    dp_1d_est = (c_big + 1) * 4 / 1024 / 1024
    sel_1d_est = n_big * (c_big + 1) / 8 / 1024 / 1024
    
    print(f"当 n={n_big}, C={c_big} 时:")
    print(f"  二维DP内存: {dp_2d_est:.1f} MB (仅DP数组)")
    print(f"  一维DP内存: {dp_1d_est:.1f} MB (DP数组) + {sel_1d_est:.1f} MB (选择标记)")
    print(f"  空间优化率: {(1 - (dp_1d_est + sel_1d_est) / dp_2d_est) * 100:.1f}%")
