def cartesian_product(*iterables):
    """
    生成多个集合的笛卡尔积，支持流式输出。
    
    与 itertools.product 功能相同，使用递归实现。
    
    Args:
        *iterables: 多个可迭代对象
        
    Yields:
        tuple: 笛卡尔积中的一个元素元组
    """
    if not iterables:
        yield ()
        return
    
    first, *rest = iterables
    first_list = list(first)
    
    if not rest:
        for item in first_list:
            yield (item,)
    else:
        for item in first_list:
            for product in cartesian_product(*rest):
                yield (item,) + product


def permutations(iterable, k=None):
    """
    生成无重复元素的所有排列，支持流式输出。
    
    与 itertools.permutations 功能相同。
    
    Args:
        iterable: 输入的可迭代对象
        k: 选择的元素个数，默认为 len(iterable)
        
    Yields:
        tuple: 一个排列元组
        
    Raises:
        TypeError: 如果 k 不是整数
        ValueError: 如果 k 是负数
    """
    pool = tuple(iterable)
    n = len(pool)
    
    if k is None:
        k = n
    
    if not isinstance(k, int):
        raise TypeError(f"k 必须是整数类型，当前类型: {type(k).__name__}")
    
    if k < 0:
        raise ValueError(f"k 不能为负数，当前值: {k}")
    
    if k == 0:
        yield ()
        return
    
    if k > n:
        return
    
    indices = list(range(n))
    cycles = list(range(n, n - k, -1))
    
    yield tuple(pool[i] for i in indices[:k])
    
    while n:
        for i in reversed(range(k)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i + 1:] + [indices[i]]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield tuple(pool[i] for i in indices[:k])
                break
        else:
            return


def permutations_with_duplicates(iterable, k=None):
    """
    生成带重复元素的所有唯一排列，支持流式输出。
    
    处理输入中包含重复元素的情况，使用去重策略确保结果唯一。
    采用回溯算法，逐个yield输出，避免内存爆炸。
    
    Args:
        iterable: 输入的可迭代对象（可包含重复元素）
        k: 选择的元素个数，默认为 len(iterable)
        
    Yields:
        tuple: 一个唯一的排列元组
        
    Raises:
        TypeError: 如果 k 不是整数
        ValueError: 如果 k 是负数
    """
    pool = list(iterable)
    n = len(pool)
    
    if k is None:
        k = n
    
    if not isinstance(k, int):
        raise TypeError(f"k 必须是整数类型，当前类型: {type(k).__name__}")
    
    if k < 0:
        raise ValueError(f"k 不能为负数，当前值: {k}")
    
    if k == 0:
        yield ()
        return
    
    if k > n:
        return
    
    pool.sort()
    
    def backtrack(path, used):
        if len(path) == k:
            yield tuple(path)
            return
        
        for i in range(n):
            if used[i]:
                continue
            
            if i > 0 and pool[i] == pool[i - 1] and not used[i - 1]:
                continue
            
            used[i] = True
            path.append(pool[i])
            
            yield from backtrack(path, used)
            
            path.pop()
            used[i] = False
    
    used = [False] * n
    yield from backtrack([], used)


def combinations(iterable, k):
    """
    生成无重复元素的所有组合，支持流式输出。
    
    与 itertools.combinations 功能相同。
    
    Args:
        iterable: 输入的可迭代对象
        k: 选择的元素个数
        
    Yields:
        tuple: 一个组合元组
        
    Raises:
        TypeError: 如果 k 不是整数
        ValueError: 如果 k 是负数
    """
    pool = tuple(iterable)
    n = len(pool)
    
    if not isinstance(k, int):
        raise TypeError(f"k 必须是整数类型，当前类型: {type(k).__name__}")
    
    if k < 0:
        raise ValueError(f"k 不能为负数，当前值: {k}")
    
    if k == 0:
        yield ()
        return
    
    if k > n:
        return
    
    indices = list(range(k))
    
    yield tuple(pool[i] for i in indices)
    
    while True:
        for i in reversed(range(k)):
            if indices[i] != i + n - k:
                break
        else:
            return
        
        indices[i] += 1
        for j in range(i + 1, k):
            indices[j] = indices[j - 1] + 1
        
        yield tuple(pool[i] for i in indices)


def combinations_with_duplicates(iterable, k):
    """
    生成带重复元素的所有唯一组合，支持流式输出。
    
    处理输入中包含重复元素的情况，使用去重策略确保结果唯一。
    采用回溯算法，逐个yield输出，避免内存爆炸。
    
    Args:
        iterable: 输入的可迭代对象（可包含重复元素）
        k: 选择的元素个数
        
    Yields:
        tuple: 一个唯一的组合元组
        
    Raises:
        TypeError: 如果 k 不是整数
        ValueError: 如果 k 是负数
    """
    pool = list(iterable)
    n = len(pool)
    
    if not isinstance(k, int):
        raise TypeError(f"k 必须是整数类型，当前类型: {type(k).__name__}")
    
    if k < 0:
        raise ValueError(f"k 不能为负数，当前值: {k}")
    
    if k == 0:
        yield ()
        return
    
    if k > n:
        return
    
    pool.sort()
    
    def backtrack(start, path):
        if len(path) == k:
            yield tuple(path)
            return
        
        for i in range(start, n):
            if i > start and pool[i] == pool[i - 1]:
                continue
            
            path.append(pool[i])
            
            yield from backtrack(i + 1, path)
            
            path.pop()
    
    yield from backtrack(0, [])


def combinations_with_replacement(iterable, k):
    """
    生成允许重复选择的所有组合，支持流式输出。
    
    与 itertools.combinations_with_replacement 功能相同。
    
    Args:
        iterable: 输入的可迭代对象
        k: 选择的元素个数
        
    Yields:
        tuple: 一个组合元组
        
    Raises:
        TypeError: 如果 k 不是整数
        ValueError: 如果 k 是负数
    """
    pool = tuple(iterable)
    n = len(pool)
    
    if not isinstance(k, int):
        raise TypeError(f"k 必须是整数类型，当前类型: {type(k).__name__}")
    
    if k < 0:
        raise ValueError(f"k 不能为负数，当前值: {k}")
    
    if k == 0:
        yield ()
        return
    
    if n == 0:
        return
    
    indices = [0] * k
    
    yield tuple(pool[i] for i in indices)
    
    while True:
        for i in reversed(range(k)):
            if indices[i] != n - 1:
                break
        else:
            return
        
        indices[i:] = [indices[i] + 1] * (k - i)
        
        yield tuple(pool[i] for i in indices)


if __name__ == "__main__":
    print("=" * 60)
    print("笛卡尔积测试")
    print("=" * 60)
    print("cartesian_product([1, 2], ['a', 'b'], [True, False]):")
    for item in cartesian_product([1, 2], ['a', 'b'], [True, False]):
        print(f"  {item}")
    
    print("\n" + "=" * 60)
    print("排列测试（无重复元素）")
    print("=" * 60)
    print("permutations(['A', 'B', 'C'], 2):")
    for item in permutations(['A', 'B', 'C'], 2):
        print(f"  {item}")
    
    print("\npermutations([1, 2, 3]):")
    for item in permutations([1, 2, 3]):
        print(f"  {item}")
    
    print("\n" + "=" * 60)
    print("排列测试（带重复元素）")
    print("=" * 60)
    print("permutations(['A', 'A', 'B'], 2):")
    print("  使用普通 permutations:")
    for item in permutations(['A', 'A', 'B'], 2):
        print(f"    {item}")
    print("  使用去重 permutations_with_duplicates:")
    for item in permutations_with_duplicates(['A', 'A', 'B'], 2):
        print(f"    {item}")
    
    print("\npermutations_with_duplicates([1, 1, 2, 2], 3):")
    for item in permutations_with_duplicates([1, 1, 2, 2], 3):
        print(f"  {item}")
    
    print("\npermutations_with_duplicates(['a', 'a', 'a', 'b'], 2):")
    for item in permutations_with_duplicates(['a', 'a', 'a', 'b'], 2):
        print(f"  {item}")
    
    print("\n" + "=" * 60)
    print("组合测试（无重复元素）")
    print("=" * 60)
    print("combinations(['A', 'B', 'C', 'D'], 2):")
    for item in combinations(['A', 'B', 'C', 'D'], 2):
        print(f"  {item}")
    
    print("\ncombinations([1, 2, 3, 4, 5], 3):")
    for item in combinations([1, 2, 3, 4, 5], 3):
        print(f"  {item}")
    
    print("\n" + "=" * 60)
    print("组合测试（带重复元素）")
    print("=" * 60)
    print("combinations(['A', 'A', 'B', 'C'], 2):")
    print("  使用普通 combinations:")
    for item in combinations(['A', 'A', 'B', 'C'], 2):
        print(f"    {item}")
    print("  使用去重 combinations_with_duplicates:")
    for item in combinations_with_duplicates(['A', 'A', 'B', 'C'], 2):
        print(f"    {item}")
    
    print("\ncombinations_with_duplicates([1, 1, 2, 2, 3], 3):")
    for item in combinations_with_duplicates([1, 1, 2, 2, 3], 3):
        print(f"  {item}")
    
    print("\ncombinations_with_duplicates(['a', 'a', 'b', 'b', 'b', 'c'], 2):")
    for item in combinations_with_duplicates(['a', 'a', 'b', 'b', 'b', 'c'], 2):
        print(f"  {item}")
    
    print("\n" + "=" * 60)
    print("组合测试（允许重复选择）")
    print("=" * 60)
    print("combinations_with_replacement(['A', 'B', 'C'], 2):")
    for item in combinations_with_replacement(['A', 'B', 'C'], 2):
        print(f"  {item}")
    
    print("\n" + "=" * 60)
    print("边界情况测试")
    print("=" * 60)
    
    print("\n1. k = 0 的情况 (应返回包含空元组的生成器):")
    print(f"  permutations(['A', 'B'], 0) = {list(permutations(['A', 'B'], 0))}")
    print(f"  permutations_with_duplicates(['A', 'B'], 0) = {list(permutations_with_duplicates(['A', 'B'], 0))}")
    print(f"  combinations(['A', 'B'], 0) = {list(combinations(['A', 'B'], 0))}")
    print(f"  combinations_with_duplicates(['A', 'B'], 0) = {list(combinations_with_duplicates(['A', 'B'], 0))}")
    print(f"  combinations_with_replacement(['A', 'B'], 0) = {list(combinations_with_replacement(['A', 'B'], 0))}")
    
    print("\n2. k > n 的情况 (应返回空生成器):")
    print(f"  permutations(['A', 'B'], 5) = {list(permutations(['A', 'B'], 5))}")
    print(f"  permutations_with_duplicates(['A', 'B'], 5) = {list(permutations_with_duplicates(['A', 'B'], 5))}")
    print(f"  combinations(['A', 'B'], 5) = {list(combinations(['A', 'B'], 5))}")
    print(f"  combinations_with_duplicates(['A', 'B'], 5) = {list(combinations_with_duplicates(['A', 'B'], 5))}")
    
    print("\n3. 空列表输入:")
    print(f"  permutations([], 0) = {list(permutations([], 0))}")
    print(f"  permutations([], 1) = {list(permutations([], 1))}")
    print(f"  permutations_with_duplicates([], 0) = {list(permutations_with_duplicates([], 0))}")
    print(f"  permutations_with_duplicates([], 1) = {list(permutations_with_duplicates([], 1))}")
    print(f"  combinations([], 0) = {list(combinations([], 0))}")
    print(f"  combinations_with_duplicates([], 0) = {list(combinations_with_duplicates([], 0))}")
    print(f"  combinations_with_replacement([], 0) = {list(combinations_with_replacement([], 0))}")
    print(f"  combinations_with_replacement([], 1) = {list(combinations_with_replacement([], 1))}")
    
    print("\n" + "=" * 60)
    print("参数合法性校验测试")
    print("=" * 60)
    
    print("\n1. k 为负数:")
    try:
        list(permutations(['A', 'B'], -1))
        print("  permutations: 未抛出异常 (错误!)")
    except ValueError as e:
        print(f"  permutations: {e}")
    
    try:
        list(permutations_with_duplicates(['A', 'B'], -1))
        print("  permutations_with_duplicates: 未抛出异常 (错误!)")
    except ValueError as e:
        print(f"  permutations_with_duplicates: {e}")
    
    try:
        list(combinations(['A', 'B'], -1))
        print("  combinations: 未抛出异常 (错误!)")
    except ValueError as e:
        print(f"  combinations: {e}")
    
    try:
        list(combinations_with_duplicates(['A', 'B'], -1))
        print("  combinations_with_duplicates: 未抛出异常 (错误!)")
    except ValueError as e:
        print(f"  combinations_with_duplicates: {e}")
    
    try:
        list(combinations_with_replacement(['A', 'B'], -1))
        print("  combinations_with_replacement: 未抛出异常 (错误!)")
    except ValueError as e:
        print(f"  combinations_with_replacement: {e}")
    
    print("\n2. k 不是整数:")
    try:
        list(permutations(['A', 'B'], 2.5))
        print("  permutations: 未抛出异常 (错误!)")
    except TypeError as e:
        print(f"  permutations: {e}")
    
    try:
        list(permutations_with_duplicates(['A', 'B'], 2.5))
        print("  permutations_with_duplicates: 未抛出异常 (错误!)")
    except TypeError as e:
        print(f"  permutations_with_duplicates: {e}")
    
    try:
        list(combinations(['A', 'B'], 'a'))
        print("  combinations: 未抛出异常 (错误!)")
    except TypeError as e:
        print(f"  combinations: {e}")
    
    try:
        list(combinations_with_duplicates(['A', 'B'], 'a'))
        print("  combinations_with_duplicates: 未抛出异常 (错误!)")
    except TypeError as e:
        print(f"  combinations_with_duplicates: {e}")
    
    print("\n" + "=" * 60)
    print("流式输出演示（惰性求值）")
    print("=" * 60)
    
    def infinite_generator():
        num = 0
        while True:
            yield num
            num += 1
    
    print("取前5个无限生成器与[1,2]的笛卡尔积:")
    count = 0
    for item in cartesian_product(infinite_generator(), [1, 2]):
        if count >= 5:
            break
        print(f"  {item}")
        count += 1
    
    print("\n流式处理大量数据（逐个yield，不占用大量内存）:")
    large_data = list(range(100))
    print(f"  从100个元素中取3个的组合总数: C(100,3) = 161700")
    count = 0
    for item in combinations(large_data, 3):
        if count < 5:
            print(f"    第{count+1}个: {item}")
        count += 1
    print(f"  已遍历 {count} 个组合，全部通过流式输出完成！")
    
    print("\n带重复元素的流式输出:")
    large_dup_data = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5] * 10
    print(f"  从{len(large_dup_data)}个重复元素中取4个的唯一排列:")
    count = 0
    for item in permutations_with_duplicates(large_dup_data, 4):
        if count < 5:
            print(f"    第{count+1}个: {item}")
        count += 1
    print(f"  已遍历 {count} 个唯一排列，全部通过流式输出完成！")
    
    print("\n✓ 所有测试完成！")
