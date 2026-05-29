class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def _build_linked_list(values):
    if not values:
        return None
    head = ListNode(values[0])
    current = head
    for v in values[1:]:
        current.next = ListNode(v)
        current = current.next
    return head


def _linked_list_to_list(head):
    result = []
    current = head
    while current:
        result.append(current.val)
        current = current.next
    return result


def _validate_input(values):
    if not isinstance(values, list):
        raise TypeError("输入必须为列表类型")
    for i, v in enumerate(values):
        if isinstance(v, (list, dict, set)):
            raise ValueError(f"索引 {i} 的值类型不合法，不能为列表、字典或集合")


def _validate_range(m, n, length):
    if not isinstance(m, int) or not isinstance(n, int):
        raise TypeError("m和n必须为整数")
    if m < 1 or n < 1:
        raise ValueError("m和n必须为正整数 (1-indexed)")
    if m > n:
        raise ValueError("m不能大于n")
    if m > length:
        raise ValueError(f"m={m} 超出链表长度 {length}")


def reverse_linked_list(values):
    _validate_input(values)

    if len(values) <= 1:
        return {"reversed": values.copy(), "access_order": values.copy()}

    head = _build_linked_list(values)
    access_order = []

    prev = None
    current = head

    while current:
        access_order.append(current.val)
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node

    reversed_list = _linked_list_to_list(prev)
    return {"reversed": reversed_list, "access_order": access_order}


def reverse_linked_list_recursive(values):
    _validate_input(values)

    if len(values) <= 1:
        return {
            "reversed": values.copy(),
            "access_order": values.copy(),
            "max_stack_depth": len(values),
        }

    head = _build_linked_list(values)
    access_order = []
    max_depth = [0]

    def _reverse(node, depth):
        max_depth[0] = max(max_depth[0], depth)
        if node.next is None:
            access_order.append(node.val)
            return node
        access_order.append(node.val)
        new_head = _reverse(node.next, depth + 1)
        node.next.next = node
        node.next = None
        return new_head

    new_head = _reverse(head, 1)
    reversed_list = _linked_list_to_list(new_head)

    return {
        "reversed": reversed_list,
        "access_order": access_order,
        "max_stack_depth": max_depth[0],
    }


def reverse_between(values, m, n):
    _validate_input(values)
    _validate_range(m, n, len(values))

    if len(values) <= 1 or m == n:
        return {"reversed": values.copy(), "access_order": []}

    head = _build_linked_list(values)
    access_order = []

    dummy = ListNode(0, head)
    prev_segment = dummy

    for _ in range(m - 1):
        prev_segment = prev_segment.next

    prev = None
    current = prev_segment.next
    tail_of_reversed = current

    for _ in range(n - m + 1):
        access_order.append(current.val)
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node

    prev_segment.next = prev
    tail_of_reversed.next = current

    reversed_list = _linked_list_to_list(dummy.next)
    return {"reversed": reversed_list, "access_order": access_order}


def compare_complexity(values):
    _validate_input(values)
    n = len(values)

    iterative = reverse_linked_list(values)
    recursive = reverse_linked_list_recursive(values)

    print(f"链表长度: {n}")
    print(f"迭代法 - 额外空间复杂度: O(1)")
    print(f"递归法 - 栈深度(实际): {recursive['max_stack_depth']}")
    print(f"递归法 - 额外空间复杂度: O({n})  (递归栈深度与链表长度成正比)")
    print(f"迭代法反转结果: {iterative['reversed']}")
    print(f"递归法反转结果: {recursive['reversed']}")
    print(f"迭代法访问顺序: {iterative['access_order']}")
    print(f"递归法访问顺序: {recursive['access_order']}")
    print(f"两种方法结果一致: {iterative['reversed'] == recursive['reversed']}")


if __name__ == "__main__":
    print("=" * 50)
    print("1. 迭代法全链表反转")
    print("=" * 50)
    for values in [[1, 2, 3, 4, 5], [10, 20, 30], [7], []]:
        result = reverse_linked_list(values)
        print(f"输入: {values}")
        print(f"反转结果: {result['reversed']}")
        print(f"访问顺序: {result['access_order']}")
        print("-" * 40)

    print("\n" + "=" * 50)
    print("2. 递归法全链表反转")
    print("=" * 50)
    for values in [[1, 2, 3, 4, 5], [10, 20, 30], [7], []]:
        result = reverse_linked_list_recursive(values)
        print(f"输入: {values}")
        print(f"反转结果: {result['reversed']}")
        print(f"访问顺序: {result['access_order']}")
        print(f"递归栈深度: {result['max_stack_depth']}")
        print("-" * 40)

    print("\n" + "=" * 50)
    print("3. 区间反转 (reverse_between)")
    print("=" * 50)
    range_cases = [
        ([1, 2, 3, 4, 5], 2, 4),
        ([1, 2, 3, 4, 5], 1, 5),
        ([10, 20, 30, 40], 1, 3),
        ([5, 6, 7], 2, 2),
        ([1, 2], 1, 2),
    ]
    for values, m, n in range_cases:
        result = reverse_between(values, m, n)
        print(f"输入: {values}, m={m}, n={n}")
        print(f"反转结果: {result['reversed']}")
        print(f"访问顺序: {result['access_order']}")
        print("-" * 40)

    print("\n" + "=" * 50)
    print("4. 迭代法 vs 递归法 空间复杂度对比")
    print("=" * 50)
    compare_complexity([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    print("\n" + "=" * 50)
    print("5. 非法输入测试")
    print("=" * 50)
    invalid_tests = [
        ("非列表输入", "not a list", None, None),
        ("m超出范围", [1, 2, 3], 5, 3),
        ("m > n", [1, 2, 3], 3, 1),
        ("m为0", [1, 2, 3], 0, 2),
    ]
    for desc, values, m, n in invalid_tests:
        try:
            if m is not None:
                reverse_between(values, m, n)
            else:
                reverse_linked_list(values)
            print(f"{desc}: 未抛出异常 (错误)")
        except (TypeError, ValueError) as e:
            print(f"{desc}: 正确抛出 {type(e).__name__}: {e}")
