class UnionFind:
    def __init__(self, n=0, strategy="size", weighted=False):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        self.count = n
        self.strategy = strategy
        self.weighted = weighted
        self.weight = [0] * n

    def add(self):
        new_id = len(self.parent)
        self.parent.append(new_id)
        self.rank.append(0)
        self.size.append(1)
        self.count += 1
        self.weight.append(0)
        return new_id

    def find(self, x):
        if self.parent[x] != x:
            orig_parent = self.parent[x]
            self.parent[x] = self.find(self.parent[x])
            if self.weighted:
                self.weight[x] += self.weight[orig_parent]
        return self.parent[x]

    def union(self, a, b, w=0):
        if a == b:
            return False
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            if self.weighted and self._get_weight(a, b) != w:
                return False
            return False
        if self.strategy == "size":
            if self.size[root_a] < self.size[root_b]:
                root_a, root_b = root_b, root_a
                if self.weighted:
                    w = -w
                    a, b = b, a
            self.parent[root_b] = root_a
            self.size[root_a] += self.size[root_b]
            if self.weighted:
                self.weight[root_b] = self.weight[a] - self.weight[b] + w
        else:
            if self.rank[root_a] < self.rank[root_b]:
                self.parent[root_a] = root_b
                if self.weighted:
                    self.weight[root_a] = self.weight[b] - self.weight[a] - w
            else:
                self.parent[root_b] = root_a
                if self.weighted:
                    self.weight[root_b] = self.weight[a] - self.weight[b] + w
                if self.rank[root_a] == self.rank[root_b]:
                    self.rank[root_a] += 1
        self.count -= 1
        return True

    def _get_weight(self, a, b):
        return self.weight[a] - self.weight[b]

    def get_distance(self, a, b):
        if self.find(a) != self.find(b):
            return None
        return self.weight[a] - self.weight[b]

    def connected(self, a, b):
        return self.find(a) == self.find(b)

    def roots(self):
        return [self.find(i) for i in range(len(self.parent))]

    def components(self):
        comp = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in comp:
                comp[root] = []
            comp[root].append(i)
        return comp


if __name__ == "__main__":
    print("=" * 50)
    print("测试1: 动态添加节点")
    print("=" * 50)
    uf = UnionFind(3)
    print(f"初始3个节点，集合数量: {uf.count}")
    new_id = uf.add()
    print(f"添加节点 {new_id} 后，集合数量: {uf.count}")
    new_id2 = uf.add()
    print(f"添加节点 {new_id2} 后，集合数量: {uf.count}")
    uf.union(0, 3)
    uf.union(1, 4)
    print(f"合并后连通分量: {uf.components()}")

    print("\n" + "=" * 50)
    print("测试2: 带权并查集（食物链关系）")
    print("=" * 50)
    uf2 = UnionFind(6, weighted=True)
    uf2.union(0, 1, 1)
    uf2.union(1, 2, 1)
    print(f"0 -> 1 距离: {uf2.get_distance(0, 1)} (应为1)")
    print(f"1 -> 2 距离: {uf2.get_distance(1, 2)} (应为1)")
    print(f"0 -> 2 距离: {uf2.get_distance(0, 2)} (应为2)")
    print(f"2 -> 0 距离: {uf2.get_distance(2, 0)} (应为-2)")
    print(f"0 -> 3 距离: {uf2.get_distance(0, 3)} (应为None, 不连通)")

    print("\n" + "=" * 50)
    print("测试3: 连通分量详细信息")
    print("=" * 50)
    uf3 = UnionFind(10)
    uf3.union(0, 1)
    uf3.union(2, 3)
    uf3.union(1, 3)
    uf3.union(4, 5)
    uf3.union(5, 6)
    uf3.union(7, 8)
    comp = uf3.components()
    print(f"连通分量数: {len(comp)}")
    for root, nodes in comp.items():
        print(f"  根 {root}: 节点 {nodes}, 大小 {len(nodes)}")

    print("\n" + "=" * 50)
    print("测试4: 带权并查集冲突检测")
    print("=" * 50)
    uf4 = UnionFind(3, weighted=True)
    print(f"union(0,1,1) 返回: {uf4.union(0, 1, 1)}")
    print(f"union(1,2,1) 返回: {uf4.union(1, 2, 1)}")
    print(f"union(0,2,3) 返回: {uf4.union(0, 2, 3)} (应为False, 因为0->2应为2，冲突)")
