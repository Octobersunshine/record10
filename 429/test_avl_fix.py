from avl_tree import AVLTree


def test_delete_double_rotation():
    print("=" * 60)
    print("测试1：删除后触发 LR 双旋场景")
    print("=" * 60)
    tree = AVLTree()
    
    for key in [50, 30, 60, 20, 40, 35]:
        info = tree.insert(key)
        print(f"插入 {key}: 高度={info['height']}, BF={info['balance_factor']}, 中序={info['inorder']}")
    
    print("\n删除 60（删除后应触发 LR 双旋）:")
    info = tree.delete(60)
    print(f"删除后: 高度={info['height']}, BF={info['balance_factor']}, 中序={info['inorder']}")
    assert info['balance_factor'] in [-1, 0, 1], f"平衡因子应该在[-1,0,1]，实际是{info['balance_factor']}"
    assert info['inorder'] == [20, 30, 35, 40, 50], f"中序遍历错误"
    print("✓ LR 双旋测试通过\n")


def test_delete_rl_rotation():
    print("=" * 60)
    print("测试2：删除后触发 RL 双旋场景")
    print("=" * 60)
    tree = AVLTree()
    
    for key in [30, 20, 50, 40, 60, 45]:
        info = tree.insert(key)
        print(f"插入 {key}: 高度={info['height']}, BF={info['balance_factor']}, 中序={info['inorder']}")
    
    print("\n删除 20（删除后应触发 RL 双旋）:")
    info = tree.delete(20)
    print(f"删除后: 高度={info['height']}, BF={info['balance_factor']}, 中序={info['inorder']}")
    assert info['balance_factor'] in [-1, 0, 1], f"平衡因子应该在[-1,0,1]，实际是{info['balance_factor']}"
    assert info['inorder'] == [30, 40, 45, 50, 60], f"中序遍历错误"
    print("✓ RL 双旋测试通过\n")


def test_delete_single_rotation():
    print("=" * 60)
    print("测试3：删除后触发 LL 单旋场景")
    print("=" * 60)
    tree = AVLTree()
    
    for key in [30, 20, 40, 10, 25]:
        info = tree.insert(key)
        print(f"插入 {key}: 高度={info['height']}, BF={info['balance_factor']}, 中序={info['inorder']}")
    
    print("\n删除 40（删除后应触发 LL 单旋）:")
    info = tree.delete(40)
    print(f"删除后: 高度={info['height']}, BF={info['balance_factor']}, 中序={info['inorder']}")
    assert info['balance_factor'] in [-1, 0, 1], f"平衡因子应该在[-1,0,1]，实际是{info['balance_factor']}"
    assert info['inorder'] == [10, 20, 25, 30], f"中序遍历错误"
    print("✓ LL 单旋测试通过\n")


def test_delete_rr_rotation():
    print("=" * 60)
    print("测试4：删除后触发 RR 单旋场景")
    print("=" * 60)
    tree = AVLTree()
    
    for key in [20, 10, 40, 30, 50]:
        info = tree.insert(key)
        print(f"插入 {key}: 高度={info['height']}, BF={info['balance_factor']}, 中序={info['inorder']}")
    
    print("\n删除 10（删除后应触发 RR 单旋）:")
    info = tree.delete(10)
    print(f"删除后: 高度={info['height']}, BF={info['balance_factor']}, 中序={info['inorder']}")
    assert info['balance_factor'] in [-1, 0, 1], f"平衡因子应该在[-1,0,1]，实际是{info['balance_factor']}"
    assert info['inorder'] == [20, 30, 40, 50], f"中序遍历错误"
    print("✓ RR 单旋测试通过\n")


def test_search():
    print("=" * 60)
    print("测试5：查找功能测试")
    print("=" * 60)
    tree = AVLTree()
    
    for key in [10, 20, 30]:
        tree.insert(key)
    
    result1 = tree.search(20)
    print(f"查找存在的节点 20: 返回值={result1}, 类型={type(result1).__name__}")
    assert result1 is True, f"查找存在的节点应该返回 True，实际返回 {result1}"
    
    result2 = tree.search(99)
    print(f"查找不存在的节点 99: 返回值={result2}, 类型={type(result2).__name__}")
    assert result2 is False, f"查找不存在的节点应该返回 False，实际返回 {result2}"
    assert result2 is not None, f"查找不存在的节点不应该返回 None"
    
    print("✓ 查找功能测试通过\n")


def test_delete_various_cases():
    print("=" * 60)
    print("测试6：各种删除场景综合测试")
    print("=" * 60)
    tree = AVLTree()
    
    keys = [9, 5, 10, 0, 6, 11, -1, 1, 2]
    for key in keys:
        info = tree.insert(key)
        print(f"插入 {key:2d}: 高度={info['height']}, BF={info['balance_factor']:2d}, 中序={info['inorder']}")
    
    print("\n开始删除测试：")
    delete_keys = [10, 9, 6, 5, 0, -1, 1, 2, 11]
    for key in delete_keys:
        info = tree.delete(key)
        print(f"删除 {key:2d}: 高度={info['height']}, BF={info['balance_factor']:2d}, 中序={info['inorder']}")
        assert info['balance_factor'] in [-1, 0, 1], f"删除{key}后平衡因子应该在[-1,0,1]，实际是{info['balance_factor']}"
    
    info = tree.get_info()
    assert info['inorder'] == [], f"删除所有节点后树应为空，实际是{info['inorder']}"
    print("✓ 综合删除测试通过\n")


def test_delete_with_two_children():
    print("=" * 60)
    print("测试7：删除有两个子节点的节点（需要后继替换）")
    print("=" * 60)
    tree = AVLTree()
    
    for key in [50, 30, 70, 20, 40, 60, 80]:
        info = tree.insert(key)
        print(f"插入 {key}: 高度={info['height']}, BF={info['balance_factor']}, 中序={info['inorder']}")
    
    print("\n删除 50（有两个子节点，需要后继替换）:")
    info = tree.delete(50)
    print(f"删除后: 高度={info['height']}, BF={info['balance_factor']}, 中序={info['inorder']}")
    assert info['balance_factor'] in [-1, 0, 1], f"平衡因子应该在[-1,0,1]，实际是{info['balance_factor']}"
    assert info['inorder'] == [20, 30, 40, 60, 70, 80], f"中序遍历错误"
    assert tree.search(50) is False, "删除后 50 应该不存在"
    assert tree.search(60) is True, "后继节点 60 应该存在"
    print("✓ 删除两个子节点测试通过\n")


if __name__ == "__main__":
    test_delete_double_rotation()
    test_delete_rl_rotation()
    test_delete_single_rotation()
    test_delete_rr_rotation()
    test_search()
    test_delete_various_cases()
    test_delete_with_two_children()
    
    print("=" * 60)
    print("所有测试通过！✓")
    print("=" * 60)
