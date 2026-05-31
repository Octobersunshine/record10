class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def build_linked_list(arr, cycle_start_idx):
    if not arr:
        return None
    nodes = [ListNode(v) for v in arr]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    if 0 <= cycle_start_idx < len(nodes):
        nodes[-1].next = nodes[cycle_start_idx]
    return nodes[0]


def detect_cycle(head):
    if head is None:
        return False, None, 0, []

    slow = head
    fast = head
    has_cycle = False
    meet_node = None

    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            has_cycle = True
            meet_node = slow
            break

    if not has_cycle:
        return False, None, 0, []

    cycle_length = 1
    curr = meet_node.next
    while curr is not meet_node:
        cycle_length += 1
        curr = curr.next

    slow = head
    fast = meet_node
    while slow is not fast:
        slow = slow.next
        fast = fast.next
    entry = slow

    cycle_nodes = []
    curr = entry
    for _ in range(cycle_length):
        cycle_nodes.append(curr.val)
        curr = curr.next

    return True, entry, cycle_length, cycle_nodes


if __name__ == "__main__":
    test_cases = [
        ([3, 2, 0, -4], 1),
        ([1, 2], 0),
        ([1], -1),
        ([], -1),
        ([1, 2, 3, 4, 5], 2),
        ([1], 0),
        ([5, 4, 3, 2, 1], 4),
        ([10, 20], 1),
    ]

    for arr, idx in test_cases:
        head = build_linked_list(arr, idx)
        has_cycle, entry, cycle_length, cycle_nodes = detect_cycle(head)
        if has_cycle:
            print(f"arr={arr}, cycle_start_idx={idx} -> 有环, 入口节点值: {entry.val}, 环长度: {cycle_length}, 环节点列表: {cycle_nodes}")
        else:
            print(f"arr={arr}, cycle_start_idx={idx} -> 无环")
