class SegmentTree:
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


if __name__ == "__main__":
    data = [1, 3, 5, 7, 9, 11]
    st = SegmentTree(data)
    
    print("=== 基础功能测试 ===")
    print("原始数组:", data)
    print("查询区间 [0, 5] 和:", st.query_range(0, 5), "  (预期: 36)")
    print("查询区间 [1, 3] 和:", st.query_range(1, 3), "  (预期: 15)")
    print("查询区间 [2, 4] 和:", st.query_range(2, 4), "  (预期: 21)")
    
    st.update_point(2, 6)
    print("\n单点更新: 位置 2 的值改为 6")
    print("更新后数组:", [st.query_range(i, i) for i in range(len(data))])
    print("查询区间 [0, 5] 和:", st.query_range(0, 5), "  (预期: 37)")
    print("查询区间 [1, 3] 和:", st.query_range(1, 3), "  (预期: 16)")
    
    st.update_point(0, 10)
    print("\n单点更新: 位置 0 的值改为 10")
    print("更新后数组:", [st.query_range(i, i) for i in range(len(data))])
    print("查询区间 [0, 2] 和:", st.query_range(0, 2), "  (预期: 19)")
    
    print("\n=== 边界条件测试 ===")
    print("查询单点 [0]:", st.query_range(0, 0), "  (预期: 10)")
    print("查询单点 [5]:", st.query_range(5, 5), "  (预期: 11)")
    print("查询越界左 [-1, 2]:", st.query_range(-1, 2), "  (预期: 19)")
    print("查询越界右 [3, 10]:", st.query_range(3, 10), "  (预期: 27)")
    print("查询完全越界左 [-5, -1]:", st.query_range(-5, -1), "  (预期: 0)")
    print("查询完全越界右 [10, 20]:", st.query_range(10, 20), "  (预期: 0)")
    
    print("\n=== 连续更新测试 ===")
    st2 = SegmentTree([0] * 5)
    print("初始数组:", [st2.query_range(i, i) for i in range(5)])
    for i in range(5):
        st2.update_point(i, i + 1)
    print("连续更新后:", [st2.query_range(i, i) for i in range(5)])
    print("总和:", st2.query_range(0, 4), "  (预期: 15)")
    
    print("\n=== 不同大小数组测试 ===")
    data3 = [10]
    st3 = SegmentTree(data3)
    print("单元素数组:", data3)
    print("查询 [0, 0]:", st3.query_range(0, 0), "  (预期: 10)")
    st3.update_point(0, 100)
    print("更新后查询:", st3.query_range(0, 0), "  (预期: 100)")
    
    data4 = list(range(1, 9))
    st4 = SegmentTree(data4)
    print("\n2的幂大小数组 (8个元素):", data4)
    print("查询 [0, 7] 和:", st4.query_range(0, 7), "  (预期: 36)")
    print("查询 [3, 5] 和:", st4.query_range(3, 5), "  (预期: 15)")
    
    print("\n=== 错误处理测试 ===")
    try:
        st.update_point(-1, 99)
        print("错误: 应该抛出异常")
    except IndexError as e:
        print("正确捕获越界更新:", str(e))
    try:
        st.update_point(100, 99)
        print("错误: 应该抛出异常")
    except IndexError as e:
        print("正确捕获越界更新:", str(e))
