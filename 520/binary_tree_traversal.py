from collections import deque
from typing import Any, Dict, List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(arr: List[Optional[int]]) -> Optional[TreeNode]:
    if not arr or arr[0] is None:
        return None
    nodes = [TreeNode(val) if val is not None else None for val in arr]
    n = len(nodes)
    for i in range(n):
        if nodes[i] is not None:
            left = 2 * i + 1
            right = 2 * i + 2
            if left < n:
                nodes[i].left = nodes[left]
            if right < n:
                nodes[i].right = nodes[right]
    return nodes[0]


def preorder_recursive(root: Optional[TreeNode]) -> List[int]:
    result = []
    def dfs(node):
        if node:
            result.append(node.val)
            dfs(node.left)
            dfs(node.right)
    dfs(root)
    return result


def preorder_iterative(root: Optional[TreeNode]) -> List[int]:
    if not root:
        return []
    result = []
    stack = [root]
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return result


def inorder_recursive(root: Optional[TreeNode]) -> List[int]:
    result = []
    def dfs(node):
        if node:
            dfs(node.left)
            result.append(node.val)
            dfs(node.right)
    dfs(root)
    return result


def inorder_iterative(root: Optional[TreeNode]) -> List[int]:
    if not root:
        return []
    result = []
    stack = []
    current = root
    while current or stack:
        while current:
            stack.append(current)
            current = current.left
        current = stack.pop()
        result.append(current.val)
        current = current.right
    return result


def postorder_recursive(root: Optional[TreeNode]) -> List[int]:
    result = []
    def dfs(node):
        if node:
            dfs(node.left)
            dfs(node.right)
            result.append(node.val)
    dfs(root)
    return result


def postorder_iterative(root: Optional[TreeNode]) -> List[int]:
    if not root:
        return []
    result = []
    stack = [root]
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
    return result[::-1]


def preorder_morris(root: Optional[TreeNode]) -> List[int]:
    result = []
    current = root
    while current:
        if not current.left:
            result.append(current.val)
            current = current.right
        else:
            predecessor = current.left
            while predecessor.right and predecessor.right is not current:
                predecessor = predecessor.right
            if not predecessor.right:
                result.append(current.val)
                predecessor.right = current
                current = current.left
            else:
                predecessor.right = None
                current = current.right
    return result


def inorder_morris(root: Optional[TreeNode]) -> List[int]:
    result = []
    current = root
    while current:
        if not current.left:
            result.append(current.val)
            current = current.right
        else:
            predecessor = current.left
            while predecessor.right and predecessor.right is not current:
                predecessor = predecessor.right
            if not predecessor.right:
                predecessor.right = current
                current = current.left
            else:
                predecessor.right = None
                result.append(current.val)
                current = current.right
    return result


def postorder_morris(root: Optional[TreeNode]) -> List[int]:
    result = []
    dummy = TreeNode(0)
    dummy.left = root
    current = dummy
    while current:
        if not current.left:
            current = current.right
        else:
            predecessor = current.left
            while predecessor.right and predecessor.right is not current:
                predecessor = predecessor.right
            if not predecessor.right:
                predecessor.right = current
                current = current.left
            else:
                predecessor.right = None
                temp = []
                node = current.left
                while node:
                    temp.append(node.val)
                    node = node.right
                result.extend(reversed(temp))
                current = current.right
    return result


def level_order(root: Optional[TreeNode]) -> List[int]:
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        result.append(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return result


def level_order_by_level(root: Optional[TreeNode]) -> List[List[int]]:
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        level_size = len(queue)
        level = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result


def build_tree_from_preorder_inorder(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    if not preorder or not inorder:
        return None
    inorder_map = {val: idx for idx, val in enumerate(inorder)}
    pre_idx = [0]

    def helper(left: int, right: int) -> Optional[TreeNode]:
        if left > right:
            return None
        root_val = preorder[pre_idx[0]]
        pre_idx[0] += 1
        root = TreeNode(root_val)
        mid = inorder_map[root_val]
        root.left = helper(left, mid - 1)
        root.right = helper(mid + 1, right)
        return root

    return helper(0, len(inorder) - 1)


def to_visualization_json(root: Optional[TreeNode]) -> dict:
    def assign_coordinates(node: Optional[TreeNode], x_counter: List[int], y: int, positions: dict) -> None:
        if not node:
            return
        assign_coordinates(node.left, x_counter, y + 1, positions)
        positions[id(node)] = {'x': x_counter[0], 'y': y}
        x_counter[0] += 1
        assign_coordinates(node.right, x_counter, y + 1, positions)

    if not root:
        return {'root': None, 'nodes': [], 'edges': []}

    positions = {}
    assign_coordinates(root, [0], 0, positions)

    nodes = []
    edges = []
    queue = deque([(root, None, None)])
    while queue:
        node, parent_id, relation = queue.popleft()
        node_id = id(node)
        pos = positions[node_id]
        node_data = {
            'id': node_id,
            'val': node.val,
            'x': pos['x'],
            'y': pos['y']
        }
        nodes.append(node_data)
        if parent_id is not None:
            edges.append({
                'source': parent_id,
                'target': node_id,
                'relation': relation
            })
        if node.left:
            queue.append((node.left, node_id, 'left'))
        if node.right:
            queue.append((node.right, node_id, 'right'))

    return {
        'root': id(root),
        'nodes': nodes,
        'edges': edges
    }


def traversal(arr: List[Optional[int]]) -> Dict[str, Any]:
    root = build_tree(arr)
    return {
        'preorder_recursive': preorder_recursive(root),
        'preorder_iterative': preorder_iterative(root),
        'preorder_morris': preorder_morris(root),
        'inorder_recursive': inorder_recursive(root),
        'inorder_iterative': inorder_iterative(root),
        'inorder_morris': inorder_morris(root),
        'postorder_recursive': postorder_recursive(root),
        'postorder_iterative': postorder_iterative(root),
        'postorder_morris': postorder_morris(root),
        'level_order': level_order(root),
        'level_order_by_level': level_order_by_level(root),
        'visualization': to_visualization_json(root),
    }


def traversal_from_preorder_inorder(preorder: List[int], inorder: List[int]) -> Dict[str, Any]:
    root = build_tree_from_preorder_inorder(preorder, inorder)
    return {
        'preorder_recursive': preorder_recursive(root),
        'preorder_iterative': preorder_iterative(root),
        'preorder_morris': preorder_morris(root),
        'inorder_recursive': inorder_recursive(root),
        'inorder_iterative': inorder_iterative(root),
        'inorder_morris': inorder_morris(root),
        'postorder_recursive': postorder_recursive(root),
        'postorder_iterative': postorder_iterative(root),
        'postorder_morris': postorder_morris(root),
        'level_order': level_order(root),
        'level_order_by_level': level_order_by_level(root),
        'visualization': to_visualization_json(root),
    }


if __name__ == '__main__':
    import json

    test_cases = [
        [1, None, 2, None, None, 3],
        [1, 2, 3, 4, 5, 6, 7],
        [],
        [1],
        [None],
        [3, 9, 20, None, None, 15, 7],
        [1, None, 2, None, None, 3, 4],
    ]

    for i, arr in enumerate(test_cases):
        print(f"Test case {i + 1}: {arr}")
        result = traversal(arr)
        for key, value in result.items():
            if key == 'visualization':
                continue
            print(f"  {key}: {value}")
        print(f"  visualization nodes: {len(result['visualization']['nodes'])}")
        print()

    print("=" * 50)
    print("Testing build_tree_from_preorder_inorder:")
    print("=" * 50)

    rebuild_tests = [
        ([3, 9, 20, 15, 7], [9, 3, 15, 20, 7]),
        ([1, 2, 4, 5, 3, 6, 7], [4, 2, 5, 1, 6, 3, 7]),
        ([1, 2, 3], [2, 1, 3]),
        ([1], [1]),
        ([], []),
    ]

    for i, (pre, ino) in enumerate(rebuild_tests):
        print(f"\nRebuild test {i + 1}:")
        print(f"  preorder: {pre}")
        print(f"  inorder:  {ino}")
        result = traversal_from_preorder_inorder(pre, ino)
        for key, value in result.items():
            if key == 'visualization':
                continue
            print(f"  {key}: {value}")
        vis_json = json.dumps(result['visualization'], indent=2, ensure_ascii=False)
        print(f"  visualization: {vis_json}")
        print()
