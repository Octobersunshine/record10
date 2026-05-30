from avl_tree import AVLTree
from rb_tree import RBTree
import random


def test_range_query_avl():
    print("=" * 60)
    print("测试1：AVL 树范围查询")
    print("=" * 60)
    tree = AVLTree()
    for key in [10, 20, 30, 40, 50, 25, 35]:
        tree.insert(key)

    result = tree.range_query(20, 40)
    print(f"范围查询 [20, 40]: {result}")
    assert result == [20, 25, 30, 35, 40], f"AVL 范围查询结果错误: {result}"

    result2 = tree.range_query(5, 15)
    print(f"范围查询 [5, 15]: {result2}")
    assert result2 == [10], f"AVL 范围查询结果错误: {result2}"

    result3 = tree.range_query(100, 200)
    print(f"范围查询 [100, 200]: {result3}")
    assert result3 == [], f"AVL 范围查询结果错误: {result3}"
    print("✓ AVL 范围查询测试通过\n")


def test_range_query_rb():
    print("=" * 60)
    print("测试2：红黑树范围查询")
    print("=" * 60)
    tree = RBTree()
    for key in [10, 20, 30, 40, 50, 25, 35]:
        tree.insert(key)

    result = tree.range_query(20, 40)
    print(f"范围查询 [20, 40]: {result}")
    assert result == [20, 25, 30, 35, 40], f"红黑树范围查询结果错误: {result}"

    result2 = tree.range_query(5, 15)
    print(f"范围查询 [5, 15]: {result2}")
    assert result2 == [10], f"红黑树范围查询结果错误: {result2}"

    result3 = tree.range_query(100, 200)
    print(f"范围查询 [100, 200]: {result3}")
    assert result3 == [], f"红黑树范围查询结果错误: {result3}"
    print("✓ 红黑树范围查询测试通过\n")


def test_max_balance_factor_avl():
    print("=" * 60)
    print("测试3：AVL 树最大平衡因子（始终 ≤ 1）")
    print("=" * 60)
    tree = AVLTree()
    keys = list(range(1, 51))
    random.shuffle(keys)
    random.seed(42)
    keys = list(range(1, 51))
    random.shuffle(keys)

    for key in keys:
        tree.insert(key)

    max_bf = tree.get_max_balance_factor()
    print(f"插入 1-50 后，AVL 最大平衡因子绝对值: {max_bf}")
    assert max_bf <= 1, f"AVL 最大平衡因子应该 ≤ 1，实际是 {max_bf}"

    for key in [10, 20, 30, 40, 50]:
        tree.delete(key)

    max_bf = tree.get_max_balance_factor()
    print(f"删除 5 个节点后，AVL 最大平衡因子绝对值: {max_bf}")
    assert max_bf <= 1, f"AVL 删除后最大平衡因子应该 ≤ 1，实际是 {max_bf}"
    print("✓ AVL 最大平衡因子测试通过\n")


def test_max_balance_factor_rb():
    print("=" * 60)
    print("测试4：红黑树最大平衡因子（宽松平衡，允许 > 1）")
    print("=" * 60)
    tree = RBTree()
    random.seed(42)
    keys = list(range(1, 51))
    random.shuffle(keys)

    for key in keys:
        tree.insert(key)

    max_bf = tree.get_max_balance_factor()
    print(f"插入 1-50 后，红黑树最大平衡因子绝对值: {max_bf}")

    for key in [10, 20, 30, 40, 50]:
        tree.delete(key)

    max_bf = tree.get_max_balance_factor()
    print(f"删除 5 个节点后，红黑树最大平衡因子绝对值: {max_bf}")
    print("✓ 红黑树最大平衡因子测试通过\n")


def test_compare_avl_vs_rb():
    print("=" * 60)
    print("测试5：AVL vs 红黑树对比（相同插入序列）")
    print("=" * 60)
    avl = AVLTree()
    rb = RBTree()

    random.seed(123)
    keys = list(range(1, 101))
    random.shuffle(keys)

    for key in keys:
        avl.insert(key)
        rb.insert(key)

    avl_info = avl.get_info()
    rb_info = rb.get_info()

    print(f"{'指标':<20} {'AVL树':>10} {'红黑树':>10}")
    print("-" * 42)
    print(f"{'高度':<20} {avl_info['height']:>10} {rb_info['height']:>10}")
    print(f"{'根节点平衡因子':<14} {avl_info['balance_factor']:>10} {rb_info['balance_factor']:>10}")
    print(f"{'最大平衡因子':<14} {avl_info['max_balance_factor']:>10} {rb_info['max_balance_factor']:>10}")

    assert avl_info['height'] <= rb_info['height'], \
        f"AVL 树高度({avl_info['height']})应该 ≤ 红黑树高度({rb_info['height']})"
    assert avl_info['max_balance_factor'] <= 1, \
        f"AVL 最大平衡因子应该 ≤ 1，实际是 {avl_info['max_balance_factor']}"
    assert avl_info['inorder'] == rb_info['inorder'], \
        "AVL 和红黑树的中序遍历结果应该相同"

    print(f"\n✓ AVL 高度({avl_info['height']}) ≤ 红黑树高度({rb_info['height']})")
    print(f"✓ AVL 最大BF({avl_info['max_balance_factor']}) ≤ 1 (严格平衡)")
    print(f"✓ 两树中序遍历一致 (均为 {len(avl_info['inorder'])} 个节点)\n")


def test_compare_after_deletion():
    print("=" * 60)
    print("测试6：删除操作后 AVL vs 红黑树对比")
    print("=" * 60)
    avl = AVLTree()
    rb = RBTree()

    random.seed(456)
    keys = list(range(1, 31))
    random.shuffle(keys)

    for key in keys:
        avl.insert(key)
        rb.insert(key)

    delete_keys = [5, 15, 25, 10, 20]
    for key in delete_keys:
        avl.delete(key)
        rb.delete(key)

    avl_info = avl.get_info()
    rb_info = rb.get_info()

    print(f"删除 {delete_keys} 后:")
    print(f"{'指标':<20} {'AVL树':>10} {'红黑树':>10}")
    print("-" * 42)
    print(f"{'高度':<20} {avl_info['height']:>10} {rb_info['height']:>10}")
    print(f"{'最大平衡因子':<14} {avl_info['max_balance_factor']:>10} {rb_info['max_balance_factor']:>10}")
    print(f"{'中序遍历长度':<14} {len(avl_info['inorder']):>10} {len(rb_info['inorder']):>10}")

    assert avl_info['max_balance_factor'] <= 1
    assert avl_info['inorder'] == rb_info['inorder']
    print(f"\n✓ 删除后 AVL 最大BF ≤ 1 (严格平衡)")
    print(f"✓ 两树中序遍历一致\n")


def test_range_query_after_modification():
    print("=" * 60)
    print("测试7：插入/删除后范围查询对比")
    print("=" * 60)
    avl = AVLTree()
    rb = RBTree()

    for key in [50, 30, 70, 20, 40, 60, 80, 10, 35, 55, 75, 90]:
        avl.insert(key)
        rb.insert(key)

    avl.delete(50)
    rb.delete(50)

    avl_result = avl.range_query(30, 75)
    rb_result = rb.range_query(30, 75)
    print(f"AVL  范围查询 [30, 75]: {avl_result}")
    print(f"红黑树 范围查询 [30, 75]: {rb_result}")

    assert avl_result == rb_result, f"范围查询结果不一致: AVL={avl_result}, RB={rb_result}"
    assert avl_result == [30, 35, 40, 55, 60, 70, 75], f"范围查询结果错误: {avl_result}"
    print("✓ 删除后范围查询测试通过\n")


if __name__ == "__main__":
    test_range_query_avl()
    test_range_query_rb()
    test_max_balance_factor_avl()
    test_max_balance_factor_rb()
    test_compare_avl_vs_rb()
    test_compare_after_deletion()
    test_range_query_after_modification()

    print("=" * 60)
    print("所有对比测试通过！✓")
    print("=" * 60)
