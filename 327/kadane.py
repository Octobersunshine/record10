def kadane_algorithm(nums):
    if not nums:
        return None, None, None, None
    
    max_current = max_global = nums[0]
    start = 0
    end = 0
    temp_start = 0
    
    for i in range(1, len(nums)):
        if max_current + nums[i] < nums[i]:
            max_current = nums[i]
            temp_start = i
        else:
            max_current += nums[i]
        
        if max_current > max_global:
            max_global = max_current
            start = temp_start
            end = i
    
    return max_global, start, end, nums[start:end+1]


def kadane_2d(matrix):
    if not matrix or not matrix[0]:
        return None, None, None, None
    
    rows = len(matrix)
    cols = len(matrix[0])
    
    max_sum = None
    top_left = None
    bottom_right = None
    
    for top in range(rows):
        temp = [0] * cols
        for bottom in range(top, rows):
            for c in range(cols):
                temp[c] += matrix[bottom][c]
            
            row_sum, col_start, col_end, _ = kadane_algorithm(temp)
            
            if max_sum is None or row_sum > max_sum:
                max_sum = row_sum
                top_left = (top, col_start)
                bottom_right = (bottom, col_end)
    
    sub_matrix = []
    for r in range(top_left[0], bottom_right[0] + 1):
        sub_matrix.append(matrix[r][top_left[1]:bottom_right[1] + 1])
    
    return max_sum, top_left, bottom_right, sub_matrix


if __name__ == "__main__":
    print("=" * 50)
    print("一维 Kadane 算法测试")
    print("=" * 50)
    
    test_cases = [
        [],
        [-2, 1, -3, 4, -1, 2, 1, -5, 4],
        [1],
        [-5],
        [5, 4, -1, 7, 8],
        [-1, -2, -3, -4],
        [-5, -3, -1, -2, -4],
        [-2, -3, 4, -1, -2, 1, 5, -3],
        [1, 2, 3, 4, 5],
        [-1, 2, 3, -4, 5, 10],
        [2, -1, 2],
        [-3, -1, -2]
    ]
    
    for nums in test_cases:
        result = kadane_algorithm(nums)
        print(f"数组: {nums}")
        if result[0] is None:
            print("空数组，无结果")
        else:
            max_sum, start, end, sub_arr = result
            print(f"最大子数组和: {max_sum}")
            print(f"起始索引: {start}, 结束索引: {end}")
            print(f"子数组: {sub_arr}")
        print()
    
    print("=" * 50)
    print("二维 Kadane 算法测试（最大子矩阵和）")
    print("=" * 50)
    
    matrix1 = [
        [1, -2, -1, 4],
        [-8, 3, 4, 2],
        [3, 8, 10, -8],
        [-4, -1, 1, 7]
    ]
    
    matrix2 = [
        [2, 1, -3],
        [-4, 2, 5],
        [6, -1, 3]
    ]
    
    matrix3 = [
        [-1, -2],
        [-3, -4]
    ]
    
    matrix4 = [
        [1, 2],
        [3, 4]
    ]
    
    matrix5 = [[5]]
    
    matrices = [
        ("全类型矩阵", matrix1),
        ("混合矩阵", matrix2),
        ("全负数矩阵", matrix3),
        ("全正数矩阵", matrix4),
        ("单元素矩阵", matrix5)
    ]
    
    for name, mat in matrices:
        print(f"{name}:")
        for row in mat:
            print(f"  {row}")
        
        max_sum, tl, br, sub_mat = kadane_2d(mat)
        print(f"最大子矩阵和: {max_sum}")
        print(f"左上坐标: {tl}, 右下坐标: {br}")
        print(f"子矩阵:")
        for row in sub_mat:
            print(f"  {row}")
        print()
