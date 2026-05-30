import time
import random
from typing import Optional


class SegmentTreeSum:
    def __init__(self, data):
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size *= 2
        self.tree = [0] * (2 * self.size)
        self.lazy = [0] * (2 * self.size)
        for i in range(self.n):
            self.tree[self.size + i] = data[i]
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def _push(self, node, l, r):
        if self.lazy[node] != 0 and node < self.size:
            mid = (l + r) // 2
            self.tree[2 * node] += self.lazy[node] * (mid - l + 1)
            self.lazy[2 * node] += self.lazy[node]
            self.tree[2 * node + 1] += self.lazy[node] * (r - mid)
            self.lazy[2 * node + 1] += self.lazy[node]
            self.lazy[node] = 0

    def update_point(self, pos, value):
        if pos < 0 or pos >= self.n:
            raise IndexError(f"Position {pos} out of range [0, {self.n - 1}]")
        pos += self.size
        self.tree[pos] = value
        pos //= 2
        while pos >= 1:
            new_val = self.tree[2 * pos] + self.tree[2 * pos + 1]
            if self.tree[pos] == new_val:
                break
            self.tree[pos] = new_val
            pos //= 2

    def update_range(self, ql, qr, val):
        if ql < 0:
            ql = 0
        if qr >= self.n:
            qr = self.n - 1
        if ql > qr:
            return
        self._update_range(1, 0, self.size - 1, ql, qr, val)

    def _update_range(self, node, l, r, ql, qr, val):
        if qr < l or r < ql:
            return
        if ql <= l and r <= qr:
            self.tree[node] += val * (r - l + 1)
            self.lazy[node] += val
            return
        self._push(node, l, r)
        mid = (l + r) // 2
        self._update_range(2 * node, l, mid, ql, qr, val)
        self._update_range(2 * node + 1, mid + 1, r, ql, qr, val)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def _query_range(self, node, l, r, ql, qr):
        if qr < l or r < ql:
            return 0
        if ql <= l and r <= qr:
            return self.tree[node]
        self._push(node, l, r)
        mid = (l + r) // 2
        left_sum = self._query_range(2 * node, l, mid, ql, qr)
        right_sum = self._query_range(2 * node + 1, mid + 1, r, ql, qr)
        return left_sum + right_sum

    def query_range(self, ql, qr):
        if ql < 0:
            ql = 0
        if qr >= self.n:
            qr = self.n - 1
        if ql > qr:
            return 0
        return self._query_range(1, 0, self.size - 1, ql, qr)


class SegmentTreeMinMax:
    def __init__(self, data, is_max=True):
        self.n = len(data)
        self.is_max = is_max
        self.size = 1
        while self.size < self.n:
            self.size *= 2
        init_val = float('-inf') if is_max else float('inf')
        self.tree = [init_val] * (2 * self.size)
        self.lazy = [0] * (2 * self.size)
        for i in range(self.n):
            self.tree[self.size + i] = data[i]
        for i in range(self.size - 1, 0, -1):
            if is_max:
                self.tree[i] = max(self.tree[2 * i], self.tree[2 * i + 1])
            else:
                self.tree[i] = min(self.tree[2 * i], self.tree[2 * i + 1])

    def _push(self, node, l, r):
        if self.lazy[node] != 0 and node < self.size:
            self.tree[2 * node] += self.lazy[node]
            self.lazy[2 * node] += self.lazy[node]
            self.tree[2 * node + 1] += self.lazy[node]
            self.lazy[2 * node + 1] += self.lazy[node]
            self.lazy[node] = 0

    def update_point(self, pos, value):
        if pos < 0 or pos >= self.n:
            raise IndexError(f"Position {pos} out of range [0, {self.n - 1}]")
        pos += self.size
        self.tree[pos] = value
        pos //= 2
        while pos >= 1:
            if self.is_max:
                new_val = max(self.tree[2 * pos], self.tree[2 * pos + 1])
            else:
                new_val = min(self.tree[2 * pos], self.tree[2 * pos + 1])
            if self.tree[pos] == new_val:
                break
            self.tree[pos] = new_val
            pos //= 2

    def update_range(self, ql, qr, val):
        if ql < 0:
            ql = 0
        if qr >= self.n:
            qr = self.n - 1
        if ql > qr:
            return
        self._update_range(1, 0, self.size - 1, ql, qr, val)

    def _update_range(self, node, l, r, ql, qr, val):
        if qr < l or r < ql:
            return
        if ql <= l and r <= qr:
            self.tree[node] += val
            self.lazy[node] += val
            return
        self._push(node, l, r)
        mid = (l + r) // 2
        self._update_range(2 * node, l, mid, ql, qr, val)
        self._update_range(2 * node + 1, mid + 1, r, ql, qr, val)
        if self.is_max:
            self.tree[node] = max(self.tree[2 * node], self.tree[2 * node + 1])
        else:
            self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def _query_range(self, node, l, r, ql, qr):
        if qr < l or r < ql:
            return float('-inf') if self.is_max else float('inf')
        if ql <= l and r <= qr:
            return self.tree[node]
        self._push(node, l, r)
        mid = (l + r) // 2
        left_val = self._query_range(2 * node, l, mid, ql, qr)
        right_val = self._query_range(2 * node + 1, mid + 1, r, ql, qr)
        if self.is_max:
            return max(left_val, right_val)
        else:
            return min(left_val, right_val)

    def query_range(self, ql, qr):
        if ql < 0:
            ql = 0
        if qr >= self.n:
            qr = self.n - 1
        if ql > qr:
            return float('-inf') if self.is_max else float('inf')
        return self._query_range(1, 0, self.size - 1, ql, qr)


class DynamicSegmentTreeNode:
    __slots__ = ['sum_val', 'min_val', 'max_val', 'lazy', 'left', 'right']
    
    def __init__(self):
        self.sum_val = 0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.lazy = 0
        self.left: Optional[DynamicSegmentTreeNode] = None
        self.right: Optional[DynamicSegmentTreeNode] = None


class DynamicSegmentTree:
    def __init__(self, max_bound):
        self.max_bound = max_bound
        self.root = DynamicSegmentTreeNode()

    def _push(self, node, l, r):
        if node.lazy == 0 or l == r:
            return
        if not node.left:
            node.left = DynamicSegmentTreeNode()
        if not node.right:
            node.right = DynamicSegmentTreeNode()
        mid = (l + r) // 2
        node.left.sum_val += node.lazy * (mid - l + 1)
        node.left.min_val += node.lazy
        node.left.max_val += node.lazy
        node.left.lazy += node.lazy
        node.right.sum_val += node.lazy * (r - mid)
        node.right.min_val += node.lazy
        node.right.max_val += node.lazy
        node.right.lazy += node.lazy
        node.lazy = 0

    def _update_node(self, node):
        if not node.left and not node.right:
            return
        if node.left and node.right:
            node.sum_val = node.left.sum_val + node.right.sum_val
            node.min_val = min(node.left.min_val, node.right.min_val)
            node.max_val = max(node.left.max_val, node.right.max_val)
        elif node.left:
            node.sum_val = node.left.sum_val
            node.min_val = node.left.min_val
            node.max_val = node.left.max_val
        else:
            node.sum_val = node.right.sum_val
            node.min_val = node.right.min_val
            node.max_val = node.right.max_val

    def update_point(self, pos, value):
        if pos < 0 or pos > self.max_bound:
            raise IndexError(f"Position {pos} out of range [0, {self.max_bound}]")
        self._update_point(self.root, 0, self.max_bound, pos, value)

    def _update_point(self, node, l, r, pos, value):
        if l == r:
            node.sum_val = value
            node.min_val = value
            node.max_val = value
            return
        self._push(node, l, r)
        mid = (l + r) // 2
        if pos <= mid:
            if not node.left:
                node.left = DynamicSegmentTreeNode()
            self._update_point(node.left, l, mid, pos, value)
        else:
            if not node.right:
                node.right = DynamicSegmentTreeNode()
            self._update_point(node.right, mid + 1, r, pos, value)
        self._update_node(node)

    def update_range(self, ql, qr, val):
        if ql < 0:
            ql = 0
        if qr > self.max_bound:
            qr = self.max_bound
        if ql > qr:
            return
        self._update_range(self.root, 0, self.max_bound, ql, qr, val)

    def _update_range(self, node, l, r, ql, qr, val):
        if qr < l or r < ql:
            return
        if ql <= l and r <= qr:
            node.sum_val += val * (r - l + 1)
            node.min_val += val
            node.max_val += val
            node.lazy += val
            return
        self._push(node, l, r)
        mid = (l + r) // 2
        if not node.left:
            node.left = DynamicSegmentTreeNode()
        if not node.right:
            node.right = DynamicSegmentTreeNode()
        self._update_range(node.left, l, mid, ql, qr, val)
        self._update_range(node.right, mid + 1, r, ql, qr, val)
        self._update_node(node)

    def query_sum(self, ql, qr):
        if ql < 0:
            ql = 0
        if qr > self.max_bound:
            qr = self.max_bound
        if ql > qr:
            return 0
        return self._query_sum(self.root, 0, self.max_bound, ql, qr)

    def _query_sum(self, node, l, r, ql, qr):
        if not node or qr < l or r < ql:
            return 0
        if ql <= l and r <= qr:
            return node.sum_val
        self._push(node, l, r)
        mid = (l + r) // 2
        return self._query_sum(node.left, l, mid, ql, qr) + \
               self._query_sum(node.right, mid + 1, r, ql, qr)

    def query_min(self, ql, qr):
        if ql < 0:
            ql = 0
        if qr > self.max_bound:
            qr = self.max_bound
        if ql > qr:
            return float('inf')
        return self._query_min(self.root, 0, self.max_bound, ql, qr)

    def _query_min(self, node, l, r, ql, qr):
        if not node or qr < l or r < ql:
            return float('inf')
        if ql <= l and r <= qr:
            return node.min_val
        self._push(node, l, r)
        mid = (l + r) // 2
        return min(self._query_min(node.left, l, mid, ql, qr),
                   self._query_min(node.right, mid + 1, r, ql, qr))

    def query_max(self, ql, qr):
        if ql < 0:
            ql = 0
        if qr > self.max_bound:
            qr = self.max_bound
        if ql > qr:
            return float('-inf')
        return self._query_max(self.root, 0, self.max_bound, ql, qr)

    def _query_max(self, node, l, r, ql, qr):
        if not node or qr < l or r < ql:
            return float('-inf')
        if ql <= l and r <= qr:
            return node.max_val
        self._push(node, l, r)
        mid = (l + r) // 2
        return max(self._query_max(node.left, l, mid, ql, qr),
                   self._query_max(node.right, mid + 1, r, ql, qr))


class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)

    def update(self, idx, delta):
        idx += 1
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx

    def query_prefix(self, idx):
        idx += 1
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

    def query_range(self, l, r):
        if l < 0:
            l = 0
        if r >= self.n:
            r = self.n - 1
        if l > r:
            return 0
        return self.query_prefix(r) - self.query_prefix(l - 1)


class FenwickTreeRange:
    def __init__(self, size):
        self.n = size
        self.B1 = [0] * (self.n + 1)
        self.B2 = [0] * (self.n + 1)

    def _add(self, B, idx, delta):
        idx += 1
        while idx <= self.n:
            B[idx] += delta
            idx += idx & -idx

    def _query(self, B, idx):
        idx += 1
        res = 0
        while idx > 0:
            res += B[idx]
            idx -= idx & -idx
        return res

    def update_range(self, l, r, val):
        if l < 0:
            l = 0
        if r >= self.n:
            r = self.n - 1
        if l > r:
            return
        self._add(self.B1, l, val)
        self._add(self.B1, r + 1, -val)
        self._add(self.B2, l, val * (l - 1))
        self._add(self.B2, r + 1, -val * r)

    def _prefix_sum(self, idx):
        if idx < 0:
            return 0
        return self._query(self.B1, idx) * idx - self._query(self.B2, idx)

    def query_range(self, l, r):
        if l < 0:
            l = 0
        if r >= self.n:
            r = self.n - 1
        if l > r:
            return 0
        return self._prefix_sum(r) - self._prefix_sum(l - 1)


if __name__ == "__main__":
    print("=== 线段树功能测试 ===")
    data = [1, 3, 5, 7, 9, 11]
    
    st_sum = SegmentTreeSum(data)
    print("原始数组:", data)
    print("区间和 [0,5]:", st_sum.query_range(0, 5))
    
    st_sum.update_range(1, 3, 2)
    print("区间 [1,3] 加 2 后数组:", [st_sum.query_range(i, i) for i in range(6)])
    print("区间和 [0,5]:", st_sum.query_range(0, 5))
    
    st_max = SegmentTreeMinMax(data, is_max=True)
    st_min = SegmentTreeMinMax(data, is_max=False)
    print("\n区间最大值 [1,4]:", st_max.query_range(1, 4))
    print("区间最小值 [1,4]:", st_min.query_range(1, 4))
    
    st_max.update_range(2, 3, 5)
    print("区间 [2,3] 加 5 后最大值 [1,4]:", st_max.query_range(1, 4))
    
    print("\n=== 动态开点线段树测试 ===")
    dst = DynamicSegmentTree(10**9)
    dst.update_point(10000, 42)
    dst.update_point(20000, 58)
    dst.update_range(15000, 25000, 10)
    print("位置 10000 的值:", dst.query_sum(10000, 10000))
    print("位置 20000 的值:", dst.query_sum(20000, 20000))
    print("区间和 [10000, 20000]:", dst.query_sum(10000, 20000))
    print("区间最大值 [10000, 20000]:", dst.query_max(10000, 20000))
    
    print("\n=== 性能对比测试 ===")
    SIZE = 100000
    OPS = 10000
    
    random_data = [random.randint(1, 1000) for _ in range(SIZE)]
    
    st_start = time.time()
    st = SegmentTreeSum(random_data)
    for _ in range(OPS):
        l = random.randint(0, SIZE - 1)
        r = random.randint(l, SIZE - 1)
        val = random.randint(1, 10)
        st.update_range(l, r, val)
        _ = st.query_range(l, r)
    st_time = time.time() - st_start
    
    ft_start = time.time()
    ft = FenwickTreeRange(SIZE)
    for i in range(SIZE):
        ft.update_range(i, i, random_data[i])
    for _ in range(OPS):
        l = random.randint(0, SIZE - 1)
        r = random.randint(l, SIZE - 1)
        val = random.randint(1, 10)
        ft.update_range(l, r, val)
        _ = ft.query_range(l, r)
    ft_time = time.time() - ft_start
    
    print(f"数据规模: {SIZE}, 操作次数: {OPS}")
    print(f"线段树耗时: {st_time:.4f} 秒")
    print(f"树状数组耗时: {ft_time:.4f} 秒")
    print(f"树状数组比线段树快: {st_time/ft_time:.2f}x")
    
    print("\n=== 稀疏场景对比 ===")
    SPARSE_SIZE = 10**9
    SPARSE_OPS = 5000
    
    dst_start = time.time()
    dst2 = DynamicSegmentTree(SPARSE_SIZE)
    positions = random.sample(range(SPARSE_SIZE), SPARSE_OPS)
    for pos in positions:
        dst2.update_point(pos, random.randint(1, 1000))
    for _ in range(SPARSE_OPS):
        pos = random.choice(positions)
        _ = dst2.query_sum(pos, pos + 1000)
    dst_time = time.time() - dst_start
    
    print(f"索引范围: 1e9, 实际操作: {SPARSE_OPS} 次")
    print(f"动态开点线段树耗时: {dst_time:.4f} 秒")
    print(f"普通线段树无法处理此规模（需要预分配内存）")
