def _get_k(k_seq, step):
    if isinstance(k_seq, (list, tuple)):
        return k_seq[step % len(k_seq)]
    return k_seq


def josephus_recursive(n, k):
    if n == 0:
        return -1, [], []

    if isinstance(k, (list, tuple)):
        if not k:
            raise ValueError("步长序列不能为空")
        if len(k) > 1:
            people = list(range(n))
            removal_order = []
            remaining_counts = [n]
            idx = 0
            for step in range(n - 1):
                current_k = _get_k(k, step)
                if current_k == 0:
                    removal_order = list(range(n - 1))
                    remaining_counts = list(range(n, 0, -1))
                    return n - 1, removal_order, remaining_counts
                idx = (idx + current_k - 1) % len(people)
                removed = people.pop(idx)
                removal_order.append(removed)
                remaining_counts.append(len(people))
            return people[0], removal_order, remaining_counts
        k = k[0]

    if k == 0:
        removal_order = list(range(n - 1))
        remaining_counts = list(range(n, 0, -1))
        return n - 1, removal_order, remaining_counts

    if n == 1:
        return 0, [], [1]

    sub_survivor, sub_order, sub_remaining = josephus_recursive(n - 1, k)

    removed = (k - 1) % n
    removal_order = [removed] + [(x + k) % n for x in sub_order]
    survivor = (sub_survivor + k) % n
    remaining_counts = [n] + sub_remaining

    return survivor, removal_order, remaining_counts


def josephus_iterative(n, k):
    if n == 0:
        return -1, [], []

    if isinstance(k, (list, tuple)) and not k:
        raise ValueError("步长序列不能为空")

    k0 = _get_k(k, 0) if n > 1 else 0
    if k0 == 0 and n > 1:
        removal_order = list(range(n - 1))
        remaining_counts = list(range(n, 0, -1))
        return n - 1, removal_order, remaining_counts

    people = list(range(n))
    removal_order = []
    remaining_counts = [n]
    idx = 0

    for step in range(n - 1):
        current_k = _get_k(k, step)
        idx = (idx + current_k - 1) % len(people)
        removed = people.pop(idx)
        removal_order.append(removed)
        remaining_counts.append(len(people))

    return people[0], removal_order, remaining_counts


def josephus_linked_list(n, k):
    if n == 0:
        return -1, [], []

    if isinstance(k, (list, tuple)) and not k:
        raise ValueError("步长序列不能为空")

    k0 = _get_k(k, 0) if n > 1 else 0
    if k0 == 0 and n > 1:
        removal_order = list(range(n - 1))
        remaining_counts = list(range(n, 0, -1))
        return n - 1, removal_order, remaining_counts

    class Node:
        __slots__ = ('val', 'next')

        def __init__(self, val):
            self.val = val
            self.next = None

    head = Node(0)
    cur = head
    for i in range(1, n):
        cur.next = Node(i)
        cur = cur.next
    cur.next = head

    prev = cur
    cur = head
    removal_order = []
    remaining_counts = [n]

    for step in range(n - 1):
        current_k = _get_k(k, step)
        for _ in range(current_k - 1):
            prev = cur
            cur = cur.next
        removal_order.append(cur.val)
        prev.next = cur.next
        cur = cur.next
        remaining_counts.append(n - step - 1)

    return cur.val, removal_order, remaining_counts


if __name__ == '__main__':
    cases = [
        (7, 3),
        (5, 0),
        (0, 3),
        (1, 0),
        (6, [1, 2, 3]),
        (8, (2, 4)),
    ]

    for n, k in cases:
        print(f"\n=== n={n}, k={k} ===")
        try:
            s1, o1, r1 = josephus_recursive(n, k)
            s2, o2, r2 = josephus_iterative(n, k)
            s3, o3, r3 = josephus_linked_list(n, k)

            print(f"幸存者: 递归={s1}  迭代={s2}  链表={s3}")
            print(f"移除顺序: {o2}")
            print(f"剩余人数: {r2}")

            assert s1 == s2 == s3, f"幸存者不一致: {s1}, {s2}, {s3}"
            assert o1 == o2 == o3, f"移除顺序不一致: {o1}, {o2}, {o3}"
            assert r1 == r2 == r3, f"剩余人数不一致: {r1}, {r2}, {r3}"
            print("✓ 三种方法结果一致")
        except Exception as e:
            print(f"错误: {e}")
