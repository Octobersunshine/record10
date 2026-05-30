import logging
import json
import time
import random
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TOMBSTONE = object()


class HashTableBase(ABC):
    def __init__(self, initial_capacity: int = 16, load_factor_threshold: float = 0.75):
        self.capacity = initial_capacity
        self.size = 0
        self.load_factor_threshold = load_factor_threshold
        self._init_storage()

    @abstractmethod
    def _init_storage(self):
        pass

    def _hash(self, key: Any, capacity: Optional[int] = None) -> int:
        cap = capacity if capacity is not None else self.capacity
        return hash(key) % cap

    def _load_factor(self) -> float:
        return self.size / self.capacity

    @abstractmethod
    def _resize(self):
        pass

    @abstractmethod
    def put(self, key: Any, value: Any):
        pass

    @abstractmethod
    def get(self, key: Any) -> Any:
        pass

    @abstractmethod
    def delete(self, key: Any) -> Optional[bool]:
        pass

    @abstractmethod
    def _find_index(self, key: Any) -> Optional[int]:
        pass

    @abstractmethod
    def get_state(self) -> str:
        pass

    @abstractmethod
    def _get_all_items(self) -> List[Tuple[Any, Any]]:
        pass

    def __len__(self) -> int:
        return self.size

    def __str__(self) -> str:
        return self.get_state()

    def __contains__(self, key: Any) -> bool:
        try:
            self.get(key)
            return True
        except KeyError:
            return False

    def serialize(self, filepath: str):
        data = {
            "strategy": self.__class__.__name__,
            "capacity": self.capacity,
            "size": self.size,
            "load_factor_threshold": self.load_factor_threshold,
            "items": [
                {"key": self._serialize_key(k), "value": v}
                for k, v in self._get_all_items()
            ]
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"HashTable serialized to {filepath}")

    def _serialize_key(self, key: Any) -> Any:
        return key

    @classmethod
    def deserialize(cls, filepath: str) -> 'HashTableBase':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        strategy_map = {
            'ChainingHashTable': ChainingHashTable,
            'LinearProbingHashTable': LinearProbingHashTable,
            'QuadraticProbingHashTable': QuadraticProbingHashTable,
        }

        strategy = data["strategy"]
        if strategy not in strategy_map:
            raise ValueError(f"Unknown strategy: {strategy}")

        ht_class = strategy_map[strategy]
        ht = ht_class(
            initial_capacity=data["capacity"],
            load_factor_threshold=data["load_factor_threshold"]
        )
        ht.size = data["size"]

        for item in data["items"]:
            ht.put(item["key"], item["value"])

        logger.info(f"HashTable deserialized from {filepath}")
        return ht


class ChainingHashTable(HashTableBase):
    def _init_storage(self):
        self.buckets = [[] for _ in range(self.capacity)]

    def _get_all_items(self) -> List[Tuple[Any, Any]]:
        items = []
        for bucket in self.buckets:
            items.extend(bucket)
        return items

    def _resize(self):
        old_buckets = self.buckets
        new_capacity = self.capacity * 2
        new_buckets = [[] for _ in range(new_capacity)]
        for bucket in old_buckets:
            for key, value in bucket:
                new_index = self._hash(key, new_capacity)
                new_buckets[new_index].append((key, value))
        self.capacity = new_capacity
        self.buckets = new_buckets
        logger.info(f"ChainingHashTable resized to capacity {self.capacity}")

    def _find_index(self, key: Any) -> Optional[Tuple[int, int]]:
        index = self._hash(key)
        bucket = self.buckets[index]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                return (index, i)
        return None

    def put(self, key: Any, value: Any):
        index = self._hash(key)
        bucket = self.buckets[index]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
        self.size += 1
        if self._load_factor() > self.load_factor_threshold:
            self._resize()

    def get(self, key: Any) -> Any:
        result = self._find_index(key)
        if result is not None:
            bucket_idx, item_idx = result
            return self.buckets[bucket_idx][item_idx][1]
        raise KeyError(f"Key '{key}' not found")

    def delete(self, key: Any) -> Optional[bool]:
        result = self._find_index(key)
        if result is not None:
            bucket_idx, item_idx = result
            del self.buckets[bucket_idx][item_idx]
            self.size -= 1
            logger.info(f"Key '{key}' deleted successfully")
            return True
        logger.warning(f"Attempted to delete non-existent key '{key}'")
        return None

    def get_state(self) -> str:
        state = []
        for i, bucket in enumerate(self.buckets):
            if bucket:
                entries = [f"{k}: {v}" for k, v in bucket]
                state.append(f"Bucket {i}: [{', '.join(entries)}]")
            else:
                state.append(f"Bucket {i}: []")
        return "\n".join(state)


class OpenAddressingHashTable(HashTableBase):
    def _init_storage(self):
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity

    def _get_all_items(self) -> List[Tuple[Any, Any]]:
        items = []
        for i in range(self.capacity):
            if self.keys[i] is not None and self.keys[i] is not TOMBSTONE:
                items.append((self.keys[i], self.values[i]))
        return items

    @abstractmethod
    def _probe(self, key: Any, capacity: int, step: int) -> int:
        pass

    def _find_index(self, key: Any) -> Optional[int]:
        for step in range(self.capacity):
            index = self._probe(key, self.capacity, step)
            if self.keys[index] is None:
                return None
            if self.keys[index] is TOMBSTONE:
                continue
            if self.keys[index] == key:
                return index
        return None

    def _find_insert_index(self, key: Any) -> Optional[int]:
        first_tombstone = None
        for step in range(self.capacity):
            index = self._probe(key, self.capacity, step)
            if self.keys[index] is None:
                return first_tombstone if first_tombstone is not None else index
            if self.keys[index] is TOMBSTONE:
                if first_tombstone is None:
                    first_tombstone = index
                continue
            if self.keys[index] == key:
                return index
        return None

    def _resize(self):
        old_keys = self.keys
        old_values = self.values
        new_capacity = self.capacity * 2
        self.capacity = new_capacity
        self.keys = [None] * new_capacity
        self.values = [None] * new_capacity
        self.size = 0
        for i in range(len(old_keys)):
            if old_keys[i] is not None and old_keys[i] is not TOMBSTONE:
                self.put(old_keys[i], old_values[i])
        logger.info(f"{self.__class__.__name__} resized to capacity {self.capacity}")

    def put(self, key: Any, value: Any):
        index = self._find_insert_index(key)
        if index is None:
            raise RuntimeError("HashTable is full")
        if self.keys[index] is None or self.keys[index] is TOMBSTONE:
            self.size += 1
        self.keys[index] = key
        self.values[index] = value
        if self._load_factor() > self.load_factor_threshold:
            self._resize()

    def get(self, key: Any) -> Any:
        index = self._find_index(key)
        if index is not None:
            return self.values[index]
        raise KeyError(f"Key '{key}' not found")

    def delete(self, key: Any) -> Optional[bool]:
        index = self._find_index(key)
        if index is not None:
            self.keys[index] = TOMBSTONE
            self.values[index] = None
            self.size -= 1
            logger.info(f"Key '{key}' deleted successfully")
            return True
        logger.warning(f"Attempted to delete non-existent key '{key}'")
        return None

    def get_state(self) -> str:
        state = []
        for i in range(self.capacity):
            if self.keys[i] is None:
                state.append(f"Slot {i}: [empty]")
            elif self.keys[i] is TOMBSTONE:
                state.append(f"Slot {i}: [tombstone]")
            else:
                state.append(f"Slot {i}: [{self.keys[i]}: {self.values[i]}]")
        return "\n".join(state)


class LinearProbingHashTable(OpenAddressingHashTable):
    def _probe(self, key: Any, capacity: int, step: int) -> int:
        return (self._hash(key, capacity) + step) % capacity


class QuadraticProbingHashTable(OpenAddressingHashTable):
    def _probe(self, key: Any, capacity: int, step: int) -> int:
        offset = step * (step + 1) // 2
        return (self._hash(key, capacity) + offset) % capacity


def compare_lookup_efficiency(num_operations: int = 10000):
    strategies = [
        ("分离链接法", ChainingHashTable),
        ("线性探测", LinearProbingHashTable),
        ("二次探测", QuadraticProbingHashTable),
    ]

    print("\n" + "=" * 70)
    print("哈希表不同冲突解决策略查找效率对比")
    print("=" * 70)
    print(f"测试规模: {num_operations} 次操作")
    print()

    keys = [f"key_{i}" for i in range(num_operations)]
    values = [random.randint(1, 1000000) for _ in range(num_operations)]
    lookup_keys = random.sample(keys, num_operations)
    random.shuffle(lookup_keys)

    results = []

    for name, ht_class in strategies:
        ht = ht_class(initial_capacity=16)

        insert_start = time.perf_counter()
        for k, v in zip(keys, values):
            ht.put(k, v)
        insert_time = time.perf_counter() - insert_start

        lookup_start = time.perf_counter()
        for k in lookup_keys:
            ht.get(k)
        lookup_time = time.perf_counter() - lookup_start

        mixed_start = time.perf_counter()
        for k in lookup_keys[:num_operations // 2]:
            ht.delete(k)
        for i in range(num_operations // 2):
            ht.put(f"new_key_{i}", values[i])
        for k in lookup_keys[num_operations // 2:]:
            if k in ht:
                ht.get(k)
        mixed_time = time.perf_counter() - mixed_start

        results.append({
            "name": name,
            "class": ht_class,
            "insert_time": insert_time,
            "lookup_time": lookup_time,
            "mixed_time": mixed_time,
            "final_capacity": ht.capacity,
            "final_size": ht.size,
        })

        print(f"【{name}】")
        print(f"  最终容量: {ht.capacity}, 最终大小: {ht.size}")
        print(f"  插入 {num_operations} 次: {insert_time:.4f} 秒")
        print(f"  查找 {num_operations} 次: {lookup_time:.4f} 秒")
        print(f"  混合操作 {num_operations} 次: {mixed_time:.4f} 秒")
        print()

    print("-" * 70)
    print("效率对比:")
    print("-" * 70)

    base_lookup = min(r["lookup_time"] for r in results)
    for r in results:
        speedup = base_lookup / r["lookup_time"] if r["lookup_time"] > 0 else 0
        marker = "★ 最快" if r["lookup_time"] == base_lookup else ""
        print(f"{r['name']:10} 查找效率: 相对 {speedup:.2f}x {marker}")

    print("-" * 70)
    print(f"负载因子阈值: 0.75, 初始容量: 16")
    print("=" * 70)

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("1. 分离链接法测试")
    print("=" * 70)
    ht_chain = ChainingHashTable(initial_capacity=4)
    for k, v in [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5)]:
        ht_chain.put(k, v)
    print(f"分离链接法容量: {ht_chain.capacity}, 大小: {ht_chain.size}")
    print(ht_chain)
    print()

    print("=" * 70)
    print("2. 线性探测测试")
    print("=" * 70)
    ht_linear = LinearProbingHashTable(initial_capacity=4)
    for k, v in [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5)]:
        ht_linear.put(k, v)
    print(f"线性探测容量: {ht_linear.capacity}, 大小: {ht_linear.size}")
    print(ht_linear)
    print(f"get('c') = {ht_linear.get('c')}")
    ht_linear.delete("b")
    print(f"delete('b') 后大小: {ht_linear.size}")
    print()

    print("=" * 70)
    print("3. 二次探测测试")
    print("=" * 70)
    ht_quad = QuadraticProbingHashTable(initial_capacity=4)
    for k, v in [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5)]:
        ht_quad.put(k, v)
    print(f"二次探测容量: {ht_quad.capacity}, 大小: {ht_quad.size}")
    print(ht_quad)
    print()

    print("=" * 70)
    print("4. JSON序列化/反序列化测试")
    print("=" * 70)
    test_file = "hash_table_test.json"
    ht_test = ChainingHashTable(initial_capacity=8)
    test_data = {"name": "Alice", "age": 25, "score": 95.5, "active": True}
    for k, v in test_data.items():
        ht_test.put(k, v)
    print(f"原始哈希表大小: {ht_test.size}")
    ht_test.serialize(test_file)

    ht_restored = HashTableBase.deserialize(test_file)
    print(f"恢复后哈希表类型: {ht_restored.__class__.__name__}")
    print(f"恢复后哈希表大小: {ht_restored.size}")
    for k in test_data.keys():
        print(f"  get('{k}') = {ht_restored.get(k)}")
    print()

    print("=" * 70)
    print("5. 删除不存在key测试")
    print("=" * 70)
    result = ht_chain.delete("not_exist")
    print(f"delete('not_exist') 返回: {result}")
    print()

    compare_lookup_efficiency(num_operations=5000)
