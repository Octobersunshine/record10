class AVLNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1


class AVLTree:
    def __init__(self):
        self.root = None

    def _height(self, node):
        return node.height if node else 0

    def _balance_factor(self, node):
        return self._height(node.left) - self._height(node.right) if node else 0

    def _update_height(self, node):
        if node:
            node.height = 1 + max(self._height(node.left), self._height(node.right))

    def _right_rotate(self, y):
        x = y.left
        t2 = x.right
        x.right = y
        y.left = t2
        self._update_height(y)
        self._update_height(x)
        return x

    def _left_rotate(self, x):
        y = x.right
        t2 = y.left
        y.left = x
        x.right = t2
        self._update_height(x)
        self._update_height(y)
        return y

    def _rebalance(self, node):
        self._update_height(node)
        bf = self._balance_factor(node)

        left_bf = self._balance_factor(node.left)
        right_bf = self._balance_factor(node.right)

        if bf > 1:
            if left_bf >= 0:
                return self._right_rotate(node)
            else:
                node.left = self._left_rotate(node.left)
                return self._right_rotate(node)

        if bf < -1:
            if right_bf <= 0:
                return self._left_rotate(node)
            else:
                node.right = self._right_rotate(node.right)
                return self._left_rotate(node)

        return node

    def insert(self, key):
        self.root = self._insert(self.root, key)
        return self.get_info()

    def _insert(self, node, key):
        if not node:
            return AVLNode(key)
        if key < node.key:
            node.left = self._insert(node.left, key)
        elif key > node.key:
            node.right = self._insert(node.right, key)
        else:
            return node
        return self._rebalance(node)

    def delete(self, key):
        self.root = self._delete(self.root, key)
        return self.get_info()

    def _delete(self, node, key):
        if not node:
            return None

        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            if not node.left and not node.right:
                return None
            elif not node.left:
                return node.right
            elif not node.right:
                return node.left
            else:
                successor = self._min_node(node.right)
                node.key = successor.key
                node.right = self._delete(node.right, successor.key)

        return self._rebalance(node)

    def _min_node(self, node):
        current = node
        while current.left:
            current = current.left
        return current

    def search(self, key):
        node = self._search(self.root, key)
        if node is None:
            return False
        return True

    def _search(self, node, key):
        if not node or node.key == key:
            return node
        if key < node.key:
            return self._search(node.left, key)
        return self._search(node.right, key)

    def inorder(self):
        result = []
        self._inorder(self.root, result)
        return result

    def _inorder(self, node, result):
        if node:
            self._inorder(node.left, result)
            result.append(node.key)
            self._inorder(node.right, result)

    def range_query(self, min_key, max_key):
        result = []
        self._range_query(self.root, min_key, max_key, result)
        return result

    def _range_query(self, node, min_key, max_key, result):
        if not node:
            return
        if min_key < node.key:
            self._range_query(node.left, min_key, max_key, result)
        if min_key <= node.key <= max_key:
            result.append(node.key)
        if max_key > node.key:
            self._range_query(node.right, min_key, max_key, result)

    def get_max_balance_factor(self):
        return self._get_max_bf(self.root)

    def _get_max_bf(self, node):
        if not node:
            return 0
        left_max = self._get_max_bf(node.left)
        right_max = self._get_max_bf(node.right)
        current = abs(self._balance_factor(node))
        return max(current, left_max, right_max)

    def get_height(self):
        return self._height(self.root)

    def get_balance_factor(self):
        return self._balance_factor(self.root)

    def get_info(self):
        return {
            "height": self.get_height(),
            "balance_factor": self.get_balance_factor(),
            "max_balance_factor": self.get_max_balance_factor(),
            "inorder": self.inorder(),
        }


if __name__ == "__main__":
    tree = AVLTree()

    print("=== 插入操作 ===")
    for key in [10, 20, 30, 40, 50, 25]:
        info = tree.insert(key)
        print(f"插入 {key}: 高度={info['height']}, 平衡因子={info['balance_factor']}, 中序={info['inorder']}")

    print("\n=== 查找操作 ===")
    for key in [25, 35]:
        found = tree.search(key)
        print(f"查找 {key}: {'找到' if found else '未找到'}")

    print("\n=== 删除操作 ===")
    for key in [30, 10]:
        info = tree.delete(key)
        print(f"删除 {key}: 高度={info['height']}, 平衡因子={info['balance_factor']}, 中序={info['inorder']}")

    print("\n=== 最终状态 ===")
    info = tree.get_info()
    print(f"高度={info['height']}, 平衡因子={info['balance_factor']}, 中序={info['inorder']}")
