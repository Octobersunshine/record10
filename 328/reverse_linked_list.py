from typing import List, Optional, Tuple


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def array_to_linked_list(arr: List[int]) -> Optional[ListNode]:
    if not arr:
        return None
    head = ListNode(arr[0])
    current = head
    for val in arr[1:]:
        current.next = ListNode(val)
        current = current.next
    return head


def linked_list_to_array(head: Optional[ListNode]) -> List[int]:
    result = []
    current = head
    while current:
        result.append(current.val)
        current = current.next
    return result


def reverse_linked_list_iterative(head: Optional[ListNode]) -> Tuple[Optional[ListNode], List[int], int]:
    visit_order = []
    space_used = 0

    if not head:
        return None, visit_order, space_used

    if not head.next:
        visit_order.append(head.val)
        space_used = 1
        return head, visit_order, space_used

    prev = None
    current = head

    while current:
        visit_order.append(current.val)
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node

    space_used = 3
    return prev, visit_order, space_used


def reverse_linked_list_recursive(head: Optional[ListNode]) -> Tuple[Optional[ListNode], List[int], int]:
    visit_order = []
    max_depth = [0]

    if not head:
        return None, visit_order, 0

    def _reverse(node: ListNode, depth: int) -> ListNode:
        max_depth[0] = max(max_depth[0], depth)
        visit_order.append(node.val)

        if not node.next:
            return node

        new_head = _reverse(node.next, depth + 1)
        node.next.next = node
        node.next = None
        return new_head

    result = _reverse(head, 1)
    return result, visit_order, max_depth[0]


def reverse_between_iterative(head: Optional[ListNode], m: int, n: int) -> Tuple[Optional[ListNode], List[int], int]:
    visit_order = []
    space_used = 0

    if not head or m >= n:
        if head:
            visit_order.append(head.val)
        return head, visit_order, space_used

    dummy = ListNode(0, head)
    prev_node = dummy

    for _ in range(m - 1):
        if prev_node.next:
            visit_order.append(prev_node.next.val)
        prev_node = prev_node.next

    reverse_tail = prev_node.next
    current = reverse_tail
    space_used = 4

    for _ in range(n - m + 1):
        visit_order.append(current.val)
        next_node = current.next
        current.next = prev_node.next
        prev_node.next = current
        current = next_node

    reverse_tail.next = current

    return dummy.next, visit_order, space_used


def reverse_between_recursive(head: Optional[ListNode], m: int, n: int) -> Tuple[Optional[ListNode], List[int], int]:
    visit_order = []
    max_depth = [0]
    successor = [None]

    if not head or m >= n:
        if head:
            visit_order.append(head.val)
        return head, visit_order, 0

    def _reverse_n(node: ListNode, count: int, depth: int) -> ListNode:
        max_depth[0] = max(max_depth[0], depth)
        visit_order.append(node.val)

        if count == 1:
            successor[0] = node.next
            return node

        new_head = _reverse_n(node.next, count - 1, depth + 1)
        node.next.next = node
        node.next = successor[0]
        return new_head

    if m == 1:
        result = _reverse_n(head, n - m + 1, 1)
        return result, visit_order, max_depth[0]

    head.next, sub_visit, sub_depth = reverse_between_recursive(head.next, m - 1, n - 1)
    visit_order.extend(sub_visit)
    max_depth[0] = max(max_depth[0], sub_depth + 1)
    return head, visit_order, max_depth[0]


def validate_input(arr: List[int], m: int = None, n: int = None) -> bool:
    if not isinstance(arr, list):
        print("错误: 输入必须是列表类型")
        return False

    for item in arr:
        if not isinstance(item, int):
            print(f"错误: 元素 '{item}' 不是整数类型")
            return False

    if m is not None and n is not None:
        if not isinstance(m, int) or not isinstance(n, int):
            print("错误: m和n必须是整数")
            return False
        if m < 1 or n < 1:
            print("错误: m和n必须大于0")
            return False
        if m > n:
            print("错误: m不能大于n")
            return False
        if m > len(arr):
            print("错误: m超出链表长度")
            return False

    return True


def reverse_api(arr: List[int]) -> List[int]:
    print("输入数组:", arr)

    if not validate_input(arr):
        print("反转后数组:", [])
        return []

    if len(arr) <= 1:
        print("节点访问顺序:", arr)
        print("反转后数组:", arr.copy())
        return arr.copy()

    head = array_to_linked_list(arr)

    iter_head, iter_visit, iter_space = reverse_linked_list_iterative(head)
    iter_result = linked_list_to_array(iter_head)
    print(f"[迭代法] 节点访问顺序: {iter_visit}")
    print(f"[迭代法] 反转后数组: {iter_result}")
    print(f"[迭代法] 空间复杂度: O({iter_space}) (3个指针变量)")

    head = array_to_linked_list(arr)
    rec_head, rec_visit, rec_space = reverse_linked_list_recursive(head)
    rec_result = linked_list_to_array(rec_head)
    print(f"[递归法] 节点访问顺序: {rec_visit}")
    print(f"[递归法] 反转后数组: {rec_result}")
    print(f"[递归法] 空间复杂度: O({rec_space}) (递归栈最大深度)")

    print(f"空间对比: 迭代法 O({iter_space}) vs 递归法 O({rec_space})")

    return iter_result


def reverse_between_api(arr: List[int], m: int, n: int) -> List[int]:
    print(f"输入数组: {arr}, 区间: [{m}, {n}]")

    if not validate_input(arr, m, n):
        print("区间反转后数组:", [])
        return []

    if len(arr) <= 1:
        print("节点访问顺序:", arr)
        print("区间反转后数组:", arr.copy())
        return arr.copy()

    head = array_to_linked_list(arr)

    iter_head, iter_visit, iter_space = reverse_between_iterative(head, m, n)
    iter_result = linked_list_to_array(iter_head)
    print(f"[迭代法] 节点访问顺序: {iter_visit}")
    print(f"[迭代法] 区间反转后数组: {iter_result}")
    print(f"[迭代法] 空间复杂度: O({iter_space}) (常数个指针变量)")

    head = array_to_linked_list(arr)
    rec_head, rec_visit, rec_space = reverse_between_recursive(head, m, n)
    rec_result = linked_list_to_array(rec_head)
    print(f"[递归法] 节点访问顺序: {rec_visit}")
    print(f"[递归法] 区间反转后数组: {rec_result}")
    print(f"[递归法] 空间复杂度: O({rec_space}) (递归栈最大深度)")

    print(f"空间对比: 迭代法 O({iter_space}) vs 递归法 O({rec_space})")

    return iter_result


if __name__ == "__main__":
    print("=" * 50)
    print("全链表反转测试")
    print("=" * 50)
    for arr in [[1, 2, 3, 4, 5], [10, 20, 30], [5], []]:
        print(f"\n--- 输入: {arr} ---")
        reverse_api(arr)

    print("\n" + "=" * 50)
    print("区间反转测试")
    print("=" * 50)
    for arr, m, n in [([1, 2, 3, 4, 5], 2, 4), ([1, 2, 3, 4, 5], 1, 5), ([3, 5], 1, 2)]:
        print()
        reverse_between_api(arr, m, n)
