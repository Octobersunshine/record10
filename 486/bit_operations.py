def _check_bit_position(i):
    if i < 0 or i > 63:
        raise ValueError(f"位位置 i 必须在 0-63 范围内，当前值: {i}")

def get_bit(n, i):
    _check_bit_position(i)
    return (n >> i) & 1

def set_bit(n, i):
    _check_bit_position(i)
    return n | (1 << i)

def clear_bit(n, i):
    _check_bit_position(i)
    return n & ~(1 << i)

def toggle_bit(n, i):
    _check_bit_position(i)
    return n ^ (1 << i)

def count_ones(n):
    count = 0
    if n < 0:
        n &= (1 << 64) - 1
    while n:
        n &= n - 1
        count += 1
    return count


def hamming_distance(a, b):
    return count_ones(a ^ b)


class BitArray:
    def __init__(self, size):
        if size <= 0:
            raise ValueError(f"位图大小必须为正整数，当前值: {size}")
        self._size = size
        self._data = bytearray((size + 7) >> 3)

    def __len__(self):
        return self._size

    def _check_index(self, i):
        if i < 0 or i >= self._size:
            raise IndexError(f"位位置 i 必须在 0-{self._size - 1} 范围内，当前值: {i}")

    def get(self, i):
        self._check_index(i)
        return (self._data[i >> 3] >> (i & 7)) & 1

    def set(self, i):
        self._check_index(i)
        self._data[i >> 3] |= 1 << (i & 7)

    def clear(self, i):
        self._check_index(i)
        self._data[i >> 3] &= ~(1 << (i & 7))

    def toggle(self, i):
        self._check_index(i)
        self._data[i >> 3] ^= 1 << (i & 7)

    def count_ones(self):
        return sum(bin(b).count('1') for b in self._data)

    def to_int(self):
        return int.from_bytes(self._data, 'little')

    def __repr__(self):
        bits = ''.join(str(self.get(i)) for i in range(self._size - 1, -1, -1))
        return f"BitArray({bits})"


class BitSet:
    def __init__(self, elements=None):
        self._bits = 0
        if elements:
            for e in elements:
                self.add(e)

    def add(self, i):
        if i < 0:
            raise ValueError(f"集合元素不能为负数，当前值: {i}")
        self._bits |= 1 << i

    def remove(self, i):
        if i < 0:
            raise ValueError(f"集合元素不能为负数，当前值: {i}")
        self._bits &= ~(1 << i)

    def contains(self, i):
        if i < 0:
            return False
        return (self._bits >> i) & 1 == 1

    def union(self, other):
        result = BitSet()
        result._bits = self._bits | other._bits
        return result

    def intersection(self, other):
        result = BitSet()
        result._bits = self._bits & other._bits
        return result

    def difference(self, other):
        result = BitSet()
        result._bits = self._bits & ~other._bits
        return result

    def symmetric_difference(self, other):
        result = BitSet()
        result._bits = self._bits ^ other._bits
        return result

    def elements(self):
        bits = self._bits
        result = []
        pos = 0
        while bits:
            if bits & 1:
                result.append(pos)
            bits >>= 1
            pos += 1
        return result

    def __len__(self):
        return count_ones(self._bits)

    def __repr__(self):
        return f"BitSet({self.elements()})"

    def __eq__(self, other):
        if not isinstance(other, BitSet):
            return NotImplemented
        return self._bits == other._bits


if __name__ == "__main__":
    n = int(input("请输入整数 n: "))
    i = int(input("请输入位位置 i: "))

    try:
        print(f"原始值 n        = {n} (二进制: {bin(n)})")
        print(f"获取第 {i} 位    = {get_bit(n, i)}")
        print(f"设置第 {i} 位为1 = {set_bit(n, i)} (二进制: {bin(set_bit(n, i))})")
        print(f"清除第 {i} 位为0 = {clear_bit(n, i)} (二进制: {bin(clear_bit(n, i))})")
        print(f"翻转第 {i} 位    = {toggle_bit(n, i)} (二进制: {bin(toggle_bit(n, i))})")
        print(f"1 的个数         = {count_ones(n)}")
    except ValueError as e:
        print(f"错误: {e}")

    print("\n--- 汉明距离 ---")
    a, b = 10, 13
    print(f"hamming_distance({a}, {b}) = {hamming_distance(a, b)}")
    print(f"  {a} = {bin(a)}, {b} = {bin(b)}, 异或 = {bin(a ^ b)}")

    print("\n--- BitArray 大位图操作 ---")
    ba = BitArray(128)
    ba.set(0)
    ba.set(5)
    ba.set(127)
    ba.toggle(5)
    print(f"BitArray(128): set(0), set(5), set(127), toggle(5)")
    print(f"  get(0)={ba.get(0)}, get(5)={ba.get(5)}, get(127)={ba.get(127)}")
    print(f"  1 的个数 = {ba.count_ones()}")
    print(f"  转为整数 = {ba.to_int()}")

    print("\n--- BitSet 集合操作 ---")
    s1 = BitSet([0, 1, 3, 5])
    s2 = BitSet([1, 2, 3, 7])
    print(f"s1 = {s1}")
    print(f"s2 = {s2}")
    print(f"并集     s1 ∪ s2  = {s1.union(s2)}")
    print(f"交集     s1 ∩ s2  = {s1.intersection(s2)}")
    print(f"差集     s1 - s2  = {s1.difference(s2)}")
    print(f"对称差集 s1 △ s2  = {s1.symmetric_difference(s2)}")
