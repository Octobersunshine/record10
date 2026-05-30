import time
import sys
import random
from typing import List, Dict, Any
from skiplist import SkipList
from avl_tree import AVLTree


def get_memory_size(obj, seen=None):
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, SkipList):
        size += get_memory_size(obj.header, seen)
        node = obj.header.forward[0] if obj.header.forward else None
        while node:
            size += get_memory_size(node, seen)
            node = node.forward[0] if node.forward else None
    elif isinstance(obj, SkipListNode):
        size += sum(get_memory_size(x, seen) for x in obj.forward)
    elif isinstance(obj, AVLTree):
        size += get_memory_size(obj.root, seen)
    elif isinstance(obj, AVLNode):
        size += get_memory_size(obj.left, seen)
        size += get_memory_size(obj.right, seen)
    elif hasattr(obj, '__dict__'):
        size += get_memory_size(obj.__dict__, seen)
    elif isinstance(obj, dict):
        size += sum(get_memory_size(k, seen) + get_memory_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(get_memory_size(i, seen) for i in obj)
    return size


from skiplist import SkipListNode
from avl_tree import AVLNode


def benchmark(n: int = 10000) -> Dict[str, Any]:
    random.seed(42)
    keys = list(range(n))
    random.shuffle(keys)
    search_keys = random.sample(keys, min(1000, n))
    range_queries = [(random.randint(0, n//2), random.randint(n//2, n)) for _ in range(100)]

    sl = SkipList()
    avl = AVLTree()

    print("=" * 60)
    print(f"性能对比测试 (n={n})")
    print("=" * 60)

    results: Dict[str, Any] = {"skiplist": {}, "avl": {}}

    print("\n【插入测试】")
    t0 = time.time()
    for k in keys:
        sl.insert(k)
    t_sl_insert = time.time() - t0
    print(f"  跳表: {t_sl_insert:.4f}s")
    results["skiplist"]["insert"] = t_sl_insert

    t0 = time.time()
    for k in keys:
        avl.insert(k)
    t_avl_insert = time.time() - t0
    print(f"  AVL : {t_avl_insert:.4f}s")
    results["avl"]["insert"] = t_avl_insert

    print("\n【查找测试】")
    t0 = time.time()
    for k in search_keys:
        sl.search(k)
    t_sl_search = time.time() - t0
    print(f"  跳表: {t_sl_search:.4f}s")
    results["skiplist"]["search"] = t_sl_search

    t0 = time.time()
    for k in search_keys:
        avl.search(k)
    t_avl_search = time.time() - t0
    print(f"  AVL : {t_avl_search:.4f}s")
    results["avl"]["search"] = t_avl_search

    print("\n【范围查询测试】")
    t0 = time.time()
    for s, e in range_queries:
        sl.range_query(s, e)
    t_sl_range = time.time() - t0
    print(f"  跳表: {t_sl_range:.4f}s")
    results["skiplist"]["range"] = t_sl_range

    t0 = time.time()
    for s, e in range_queries:
        avl.range_query(s, e)
    t_avl_range = time.time() - t0
    print(f"  AVL : {t_avl_range:.4f}s")
    results["avl"]["range"] = t_avl_range

    print("\n【删除测试】")
    delete_keys = random.sample(keys, min(1000, n))
    t0 = time.time()
    for k in delete_keys:
        sl.delete(k)
    t_sl_delete = time.time() - t0
    print(f"  跳表: {t_sl_delete:.4f}s")
    results["skiplist"]["delete"] = t_sl_delete

    t0 = time.time()
    for k in delete_keys:
        avl.delete(k)
    t_avl_delete = time.time() - t0
    print(f"  AVL : {t_avl_delete:.4f}s")
    results["avl"]["delete"] = t_avl_delete

    print("\n【内存占用】")
    for k in delete_keys:
        sl.insert(k)
        avl.insert(k)

    mem_sl = get_memory_size(sl)
    mem_avl = get_memory_size(avl)
    print(f"  跳表: {mem_sl / 1024:.2f} KB")
    print(f"  AVL : {mem_avl / 1024:.2f} KB")
    results["skiplist"]["memory"] = mem_sl
    results["avl"]["memory"] = mem_avl

    print("\n" + "=" * 60)
    print("对比总结:")
    print("-" * 60)
    print(f"{'操作':<10} {'跳表(s)':>10} {'AVL(s)':>10} {'跳表/AVL':>12}")
    print("-" * 50)
    for op in ["insert", "search", "range", "delete"]:
        ratio = results["skiplist"][op] / results["avl"][op]
        print(f"{op:<10} {results['skiplist'][op]:>10.4f} {results['avl'][op]:>10.4f} {ratio:>12.2f}x")
    mem_ratio = mem_sl / mem_avl
    print(f"{'内存(KB)':<10} {mem_sl/1024:>10.2f} {mem_avl/1024:>10.2f} {mem_ratio:>12.2f}x")
    print("=" * 60)

    return results


def test_range_query():
    print("\n" + "=" * 60)
    print("范围查询功能测试")
    print("=" * 60)
    sl = SkipList()
    avl = AVLTree()
    for k in [3, 7, 1, 9, 5, 12, 15, 20, 11]:
        sl.insert(k)
        avl.insert(k)
    print(f"数据: {sorted([3,7,1,9,5,12,15,20,11])}")
    print(f"范围 [5, 15]:")
    print(f"  跳表: {sl.range_query(5, 15)}")
    print(f"  AVL : {avl.range_query(5, 15)}")


def test_serialization():
    print("\n" + "=" * 60)
    print("序列化功能测试")
    print("=" * 60)
    sl = SkipList()
    for k in [3, 7, 1, 9, 5]:
        sl.insert(k)
    print("原始跳表:")
    info = sl.get_structure()
    for lvl in range(info["max_level"] + 1):
        print(f"  层 {lvl}: {info['layers'][lvl]}")
    sl.save_to_file("skiplist_test.json")
    sl2 = SkipList.load_from_file("skiplist_test.json")
    print("加载后跳表:")
    info2 = sl2.get_structure()
    for lvl in range(info2["max_level"] + 1):
        print(f"  层 {lvl}: {info2['layers'][lvl]}")
    print(f"数据一致: {sorted(sl.range_query(0, 100)) == sorted(sl2.range_query(0, 100))}")


if __name__ == "__main__":
    test_range_query()
    test_serialization()
    benchmark(10000)
