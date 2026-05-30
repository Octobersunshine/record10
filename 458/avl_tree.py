from typing import Optional, List


class AVLNode:
    def __init__(self, key: int):
        self.key = key
        self.left: Optional["AVLNode"] = None
        self.right: Optional["AVLNode"] = None
        self.height = 1


class AVLTree:
    def __init__(self):
        self.root: Optional[AVLNode] = None
        self.size = 0

    def _get_height(self, node: Optional[AVLNode]) -> int:
        if not node:
            return 0
        return node.height

    def _get_balance(self, node: Optional[AVLNode]) -> int:
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)

    def _rotate_right(self, y: AVLNode) -> AVLNode:
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))
        return x

    def _rotate_left(self, x: AVLNode) -> AVLNode:
        y = x.right
        T2 = y.left
        y.left = x
        x.right = T2
        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        return y

    def _insert(self, node: Optional[AVLNode], key: int) -> AVLNode:
        if not node:
            return AVLNode(key)
        if key < node.key:
            node.left = self._insert(node.left, key)
        elif key > node.key:
            node.right = self._insert(node.right, key)
        else:
            return node
        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))
        balance = self._get_balance(node)
        if balance > 1 and key < node.left.key:
            return self._rotate_right(node)
        if balance < -1 and key > node.right.key:
            return self._rotate_left(node)
        if balance > 1 and key > node.left.key:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        if balance < -1 and key < node.right.key:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
        return node

    def insert(self, key: int) -> None:
        if not self.search(key):
            self.root = self._insert(self.root, key)
            self.size += 1

    def _get_min_value_node(self, node: AVLNode) -> AVLNode:
        current = node
        while current.left:
            current = current.left
        return current

    def _delete(self, node: Optional[AVLNode], key: int) -> Optional[AVLNode]:
        if not node:
            return node
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            if not node.left:
                temp = node.right
                node = None
                return temp
            elif not node.right:
                temp = node.left
                node = None
                return temp
            temp = self._get_min_value_node(node.right)
            node.key = temp.key
            node.right = self._delete(node.right, temp.key)
        if not node:
            return node
        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))
        balance = self._get_balance(node)
        if balance > 1 and self._get_balance(node.left) >= 0:
            return self._rotate_right(node)
        if balance > 1 and self._get_balance(node.left) < 0:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        if balance < -1 and self._get_balance(node.right) <= 0:
            return self._rotate_left(node)
        if balance < -1 and self._get_balance(node.right) > 0:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
        return node

    def delete(self, key: int) -> bool:
        if not self.search(key):
            return False
        self.root = self._delete(self.root, key)
        self.size -= 1
        return True

    def _search(self, node: Optional[AVLNode], key: int) -> bool:
        if not node:
            return False
        if key == node.key:
            return True
        elif key < node.key:
            return self._search(node.left, key)
        else:
            return self._search(node.right, key)

    def search(self, key: int) -> bool:
        return self._search(self.root, key)

    def _range_query(self, node: Optional[AVLNode], start: int, end: int, result: List[int]) -> None:
        if not node:
            return
        if start < node.key:
            self._range_query(node.left, start, end, result)
        if start <= node.key <= end:
            result.append(node.key)
        if end > node.key:
            self._range_query(node.right, start, end, result)

    def range_query(self, start: int, end: int) -> List[int]:
        result: List[int] = []
        self._range_query(self.root, start, end, result)
        return result

    def count_nodes(self) -> int:
        return self.size
