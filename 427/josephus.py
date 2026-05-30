def josephus_recursive(n: int, k: int, index: int = 0) -> int:
    if n == 0:
        return -1
    if k == 0:
        return n - 1 + index
    if n == 1:
        return index
    return (josephus_recursive(n - 1, k, 0) + k) % n + index


def josephus_iterative(n: int, k: int, index: int = 0) -> int:
    if n == 0:
        return -1
    if k == 0:
        return n - 1 + index
    res = 0
    for i in range(2, n + 1):
        res = (res + k) % i
    return res + index


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def josephus_circular_linked_list(n: int, k: int, index: int = 0) -> int:
    if n == 0:
        return -1
    if k == 0:
        return n - 1 + index
    if n == 1:
        return index
    
    head = ListNode(index)
    current = head
    for i in range(1, n):
        current.next = ListNode(i + index)
        current = current.next
    current.next = head
    
    prev = current
    current = head
    while n > 1:
        for _ in range(k - 1):
            prev = current
            current = current.next
        prev.next = current.next
        current = prev.next
        n -= 1
    
    return current.val


def josephus_enhanced(n: int, k, index: int = 0, return_details: bool = False):
    if n == 0:
        if return_details:
            return -1, [], []
        return -1
    
    is_sequence = isinstance(k, (list, tuple))
    
    if is_sequence and len(k) == 0:
        if return_details:
            return -1, [], []
        return -1
    
    if not is_sequence and k == 0:
        if return_details:
            elimination_order = list(range(index, n - 1 + index))
            remaining_counts = list(range(n - 1, 0, -1))
            return n - 1 + index, elimination_order, remaining_counts
        return n - 1 + index
    
    if n == 1:
        if return_details:
            return index, [], [n]
        return index
    
    head = ListNode(index)
    current = head
    for i in range(1, n):
        current.next = ListNode(i + index)
        current = current.next
    current.next = head
    
    prev = current
    current = head
    elimination_order = []
    remaining_counts = []
    step_idx = 0
    original_n = n
    
    while n > 1:
        if is_sequence:
            current_k = k[step_idx % len(k)]
            step_idx += 1
        else:
            current_k = k
        
        if current_k == 0:
            elimination_order.append(current.val)
            prev.next = current.next
            current = prev.next
        else:
            for _ in range(current_k - 1):
                prev = current
                current = current.next
            elimination_order.append(current.val)
            prev.next = current.next
            current = prev.next
        
        n -= 1
        remaining_counts.append(n)
    
    survivor = current.val
    
    if return_details:
        return survivor, elimination_order, remaining_counts
    return survivor


if __name__ == "__main__":
    test_cases = [
        (5, 2, 0),
        (5, 2, 1),
        (10, 3, 0),
        (10, 3, 1),
        (1, 5, 0),
        (1, 5, 1),
        (0, 5, 0),
        (0, 5, 1),
        (5, 0, 0),
        (5, 0, 1),
        (3, 0, 0),
        (10, 0, 0),
        (10, 0, 1),
        (0, 0, 0),
        (7, 4, 0),
        (7, 4, 1),
    ]
    
    print("=" * 70)
    print("基础功能测试 - 固定步长")
    print("=" * 70)
    print(f"{'n':>3} {'k':>3} {'base':>5} | {'递归':>5} {'迭代':>5} {'链表':>5} {'增强版':>6}")
    print("-" * 70)
    all_pass = True
    for n, k, idx in test_cases:
        r1 = josephus_recursive(n, k, idx)
        r2 = josephus_iterative(n, k, idx)
        r3 = josephus_circular_linked_list(n, k, idx)
        r4 = josephus_enhanced(n, k, idx)
        status = "✓" if r1 == r2 == r3 == r4 else "✗"
        if r1 != r2 or r2 != r3 or r3 != r4:
            all_pass = False
        print(f"{n:>3} {k:>3} {idx:>5} | {r1:>5} {r2:>5} {r3:>5} {r4:>6}  {status}")
    print("-" * 70)
    print(f"基础测试结果: {'全部通过' if all_pass else '存在不一致'}")
    print()
    
    print("=" * 70)
    print("增强功能测试 - 动态步长序列 & 详细信息")
    print("=" * 70)
    
    enhanced_tests = [
        (5, [1, 2, 3, 4], 0, "动态步长递增"),
        (5, [2, 2, 2, 2], 0, "固定步长序列等价k=2"),
        (5, [0, 0, 0, 0], 0, "全0步长序列"),
        (6, [3, 1, 4, 1, 5], 0, "动态步长混合"),
        (5, [2, 0, 3], 1, "1-index 动态步长含0"),
        (0, [1, 2, 3], 0, "n=0 边界"),
        (1, [5], 0, "n=1 边界"),
    ]
    
    for n, k_seq, idx, desc in enhanced_tests:
        survivor, elim_order, remaining = josephus_enhanced(n, k_seq, idx, return_details=True)
        print(f"\n测试: {desc}")
        print(f"  n={n}, k序列={k_seq}, base={idx}")
        print(f"  最后幸存者: {survivor}")
        print(f"  移除顺序: {elim_order}")
        print(f"  每步剩余: {remaining}")
        if len(elim_order) > 0 and len(remaining) > 0:
            for i, (el, rem) in enumerate(zip(elim_order, remaining), 1):
                print(f"    第{i}步: 移除 {el}, 剩余 {rem} 人")
    
    print()
    print("=" * 70)
    print("边界条件说明:")
    print("=" * 70)
    print("  n=0           → 返回 -1")
    print("  k=0           → 步长为0，每次移除当前人，最后剩下第 n-1 个")
    print("  k为序列       → 循环使用步长序列，支持动态变化")
    print("  base          → 0 表示 0-index，1 表示 1-index")
    print("  return_details→ True 时返回 (幸存者, 移除顺序, 剩余人数)")
