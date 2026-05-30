RED = True
BLACK = False


class RBNode:
    def __init__(self, key, color=RED):
        self.key = key
        self.color = color
        self.left = None
        self.right = None
        self.parent = None


class RBTree:
    def __init__(self):
        self.NIL = RBNode(None, BLACK)
        self.NIL.left = self.NIL
        self.NIL.right = self.NIL
        self.NIL.parent = self.NIL
        self.root = self.NIL

    def _is_red(self, node):
        return node is not self.NIL and node.color == RED

    def _left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left is not self.NIL:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is self.NIL:
            self.root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _right_rotate(self, y):
        x = y.left
        y.left = x.right
        if x.right is not self.NIL:
            x.right.parent = y
        x.parent = y.parent
        if y.parent is self.NIL:
            self.root = x
        elif y is y.parent.left:
            y.parent.left = x
        else:
            y.parent.right = x
        x.right = y
        y.parent = x

    def insert(self, key):
        node = RBNode(key)
        node.left = self.NIL
        node.right = self.NIL
        node.parent = self.NIL

        y = self.NIL
        x = self.root
        while x is not self.NIL:
            y = x
            if node.key < x.key:
                x = x.left
            elif node.key > x.key:
                x = x.right
            else:
                return self.get_info()

        node.parent = y
        if y is self.NIL:
            self.root = node
        elif node.key < y.key:
            y.left = node
        else:
            y.right = node

        self._insert_fixup(node)
        return self.get_info()

    def _insert_fixup(self, z):
        while self._is_red(z.parent):
            if z.parent is z.parent.parent.left:
                y = z.parent.parent.right
                if self._is_red(y):
                    z.parent.color = BLACK
                    y.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent
                else:
                    if z is z.parent.right:
                        z = z.parent
                        self._left_rotate(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._right_rotate(z.parent.parent)
            else:
                y = z.parent.parent.left
                if self._is_red(y):
                    z.parent.color = BLACK
                    y.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent
                else:
                    if z is z.parent.left:
                        z = z.parent
                        self._right_rotate(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._left_rotate(z.parent.parent)
        self.root.color = BLACK

    def delete(self, key):
        z = self._search_node(self.root, key)
        if z is self.NIL:
            return self.get_info()

        y = z
        y_original_color = y.color
        if z.left is self.NIL:
            x = z.right
            self._transplant(z, z.right)
        elif z.right is self.NIL:
            x = z.left
            self._transplant(z, z.left)
        else:
            y = self._min_node(z.right)
            y_original_color = y.color
            x = y.right
            if y.parent is z:
                x.parent = y
            else:
                self._transplant(y, y.right)
                y.right = z.right
                y.right.parent = y
            self._transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color

        if y_original_color == BLACK:
            self._delete_fixup(x)
        return self.get_info()

    def _transplant(self, u, v):
        if u.parent is self.NIL:
            self.root = v
        elif u is u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def _delete_fixup(self, x):
        while x is not self.root and not self._is_red(x):
            if x is x.parent.left:
                w = x.parent.right
                if self._is_red(w):
                    w.color = BLACK
                    x.parent.color = RED
                    self._left_rotate(x.parent)
                    w = x.parent.right
                if not self._is_red(w.left) and not self._is_red(w.right):
                    w.color = RED
                    x = x.parent
                else:
                    if not self._is_red(w.right):
                        w.left.color = BLACK
                        w.color = RED
                        self._right_rotate(w)
                        w = x.parent.right
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.right.color = BLACK
                    self._left_rotate(x.parent)
                    x = self.root
            else:
                w = x.parent.left
                if self._is_red(w):
                    w.color = BLACK
                    x.parent.color = RED
                    self._right_rotate(x.parent)
                    w = x.parent.left
                if not self._is_red(w.left) and not self._is_red(w.right):
                    w.color = RED
                    x = x.parent
                else:
                    if not self._is_red(w.left):
                        w.right.color = BLACK
                        w.color = RED
                        self._left_rotate(w)
                        w = x.parent.left
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.left.color = BLACK
                    self._right_rotate(x.parent)
                    x = self.root
        x.color = BLACK

    def _min_node(self, node):
        while node.left is not self.NIL:
            node = node.left
        return node

    def search(self, key):
        node = self._search_node(self.root, key)
        if node is self.NIL:
            return False
        return True

    def _search_node(self, node, key):
        while node is not self.NIL and key != node.key:
            if key < node.key:
                node = node.left
            else:
                node = node.right
        return node

    def inorder(self):
        result = []
        self._inorder(self.root, result)
        return result

    def _inorder(self, node, result):
        if node is not self.NIL:
            self._inorder(node.left, result)
            result.append(node.key)
            self._inorder(node.right, result)

    def range_query(self, min_key, max_key):
        result = []
        self._range_query(self.root, min_key, max_key, result)
        return result

    def _range_query(self, node, min_key, max_key, result):
        if node is self.NIL:
            return
        if min_key < node.key:
            self._range_query(node.left, min_key, max_key, result)
        if min_key <= node.key <= max_key:
            result.append(node.key)
        if max_key > node.key:
            self._range_query(node.right, min_key, max_key, result)

    def get_height(self):
        return self._get_height(self.root)

    def _get_height(self, node):
        if node is self.NIL:
            return 0
        return 1 + max(self._get_height(node.left), self._get_height(node.right))

    def _node_balance_factor(self, node):
        if node is self.NIL:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)

    def get_balance_factor(self):
        return self._node_balance_factor(self.root)

    def get_max_balance_factor(self):
        return self._get_max_bf(self.root)

    def _get_max_bf(self, node):
        if node is self.NIL:
            return 0
        left_max = self._get_max_bf(node.left)
        right_max = self._get_max_bf(node.right)
        current = abs(self._node_balance_factor(node))
        return max(current, left_max, right_max)

    def get_info(self):
        return {
            "height": self.get_height(),
            "balance_factor": self.get_balance_factor(),
            "max_balance_factor": self.get_max_balance_factor(),
            "inorder": self.inorder(),
        }


if __name__ == "__main__":
    tree = RBTree()

    print("=== 插入操作 ===")
    for key in [10, 20, 30, 40, 50, 25]:
        info = tree.insert(key)
        print(f"插入 {key}: 高度={info['height']}, BF={info['balance_factor']}, "
              f"最大BF={info['max_balance_factor']}, 中序={info['inorder']}")

    print("\n=== 范围查询 ===")
    result = tree.range_query(20, 40)
    print(f"范围查询 [20, 40]: {result}")

    print("\n=== 查找操作 ===")
    for key in [25, 35]:
        found = tree.search(key)
        print(f"查找 {key}: {'找到' if found else '未找到'}")

    print("\n=== 删除操作 ===")
    for key in [30, 10]:
        info = tree.delete(key)
        print(f"删除 {key}: 高度={info['height']}, BF={info['balance_factor']}, "
              f"最大BF={info['max_balance_factor']}, 中序={info['inorder']}")
