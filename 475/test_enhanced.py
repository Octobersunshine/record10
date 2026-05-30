import warnings
from keyset_pagination import KeysetPaginator, CursorEncoder


test_data = [
    {"id": 1, "name": "张三", "score": 90},
    {"id": 2, "name": "李四", "score": 90},
    {"id": 3, "name": "王五", "score": 90},
    {"id": 4, "name": "赵六", "score": 85},
    {"id": 5, "name": "钱七", "score": 85},
]


def test_auto_append_unique_key():
    print("=" * 70)
    print("测试1: 自动添加唯一键字段（id）")
    print("=" * 70)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        paginator = KeysetPaginator(
            sort_fields=[("score", "desc")],
            auto_append_unique_key=True
        )

        print(f"\n原始排序字段: [('score', 'desc')]")
        print(f"处理后排序字段: {paginator.sort_fields}")
        print(f"触发的警告数: {len(w)}")
        for warning in w:
            print(f"  警告: {warning.message}")

    page_size = 2
    cursor = None
    all_results = []

    for page_num in range(1, 6):
        result = paginator.paginate(
            items=test_data,
            page_size=page_size,
            cursor=cursor,
            is_forward=True
        )

        print(f"\n第 {page_num} 页:")
        print(f"  数据 (score, id, name): {[(item['score'], item['id'], item['name']) for item in result.data]}")
        print(f"  下一页游标: {result.next_cursor}")
        if result.next_cursor:
            decoded = CursorEncoder.decode(result.next_cursor)
            print(f"  游标解码值: {decoded}")

        all_results.extend(result.data)

        if not result.has_next:
            break

        cursor = result.next_cursor

    print(f"\n总共获取的记录数: {len(all_results)}")
    print(f"原始数据记录数: {len(test_data)}")
    all_ids = [item['id'] for item in all_results]
    original_ids = [item['id'] for item in test_data]
    missing_ids = set(original_ids) - set(all_ids)
    print(f"缺失的记录ID: {missing_ids if missing_ids else '无'}")


def test_no_auto_append_with_warning():
    print("\n" + "=" * 70)
    print("测试2: 不自动添加唯一键，但显示警告")
    print("=" * 70)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        paginator = KeysetPaginator(
            sort_fields=[("score", "desc")],
            auto_append_unique_key=False,
            warn_on_duplicate_risk=True
        )

        print(f"\n排序字段: {paginator.sort_fields}")
        print(f"触发的警告数: {len(w)}")
        for warning in w:
            print(f"  警告: {warning.message}")

    page_size = 2
    cursor = None
    all_results = []

    for page_num in range(1, 6):
        result = paginator.paginate(
            items=test_data,
            page_size=page_size,
            cursor=cursor,
            is_forward=True
        )

        all_results.extend(result.data)

        if not result.has_next:
            break

        cursor = result.next_cursor

    all_ids = [item['id'] for item in all_results]
    original_ids = [item['id'] for item in test_data]
    missing_ids = set(original_ids) - set(all_ids)
    print(f"\n缺失的记录ID: {missing_ids if missing_ids else '无'}")
    if missing_ids:
        print(f"  验证: 如预期，因为没有唯一键，在重复值处分页会遗漏数据")


def test_manual_unique_key_last():
    print("\n" + "=" * 70)
    print("测试3: 手动指定唯一键在最后（最佳实践）")
    print("=" * 70)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        paginator = KeysetPaginator(
            sort_fields=[("score", "desc"), ("id", "asc")],
            auto_append_unique_key=True
        )

        print(f"\n排序字段: {paginator.sort_fields}")
        print(f"触发的警告数: {len(w)} (应为0，因为唯一键已在最后)")

    page_size = 2
    cursor = None
    all_results = []

    for page_num in range(1, 6):
        result = paginator.paginate(
            items=test_data,
            page_size=page_size,
            cursor=cursor,
            is_forward=True
        )

        all_results.extend(result.data)

        if not result.has_next:
            break

        cursor = result.next_cursor

    all_ids = [item['id'] for item in all_results]
    original_ids = [item['id'] for item in test_data]
    missing_ids = set(original_ids) - set(all_ids)
    print(f"\n缺失的记录ID: {missing_ids if missing_ids else '无'}")


def test_unique_key_not_last_warning():
    print("\n" + "=" * 70)
    print("测试4: 唯一键不在最后的警告")
    print("=" * 70)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        paginator = KeysetPaginator(
            sort_fields=[("id", "asc"), ("score", "desc")],
            auto_append_unique_key=False,
            warn_on_duplicate_risk=True
        )

        print(f"\n排序字段: {paginator.sort_fields}")
        print(f"触发的警告数: {len(w)}")
        for warning in w:
            print(f"  警告: {warning.message}")


def test_first_page_no_cursor():
    print("\n" + "=" * 70)
    print("测试5: 首页查询（cursor为None）")
    print("=" * 70)

    paginator = KeysetPaginator(
        sort_fields=[("score", "desc"), ("id", "asc")],
        auto_append_unique_key=False,
        warn_on_duplicate_risk=False
    )

    result = paginator.paginate(
        items=test_data,
        page_size=3,
        cursor=None,
        is_forward=True
    )

    print(f"\n首页数据 (score, id, name): {[(item['score'], item['id'], item['name']) for item in result.data]}")
    print(f"has_next: {result.has_next}")
    print(f"has_previous: {result.has_previous} (应为 False)")
    print(f"next_cursor: {result.next_cursor}")
    print(f"previous_cursor: {result.previous_cursor} (应为 None)")

    decoded = CursorEncoder.decode(result.next_cursor)
    print(f"next_cursor解码值: {decoded}")


def test_cursor_validation():
    print("\n" + "=" * 70)
    print("测试6: Cursor验证")
    print("=" * 70)

    paginator = KeysetPaginator(
        sort_fields=[("score", "desc"), ("id", "asc")],
        auto_append_unique_key=False,
        warn_on_duplicate_risk=False
    )

    print(f"\n排序字段数: {len(paginator.sort_fields)}")

    valid_cursor = CursorEncoder.encode((90, 2))
    try:
        values = paginator.validate_cursor(valid_cursor)
        print(f"有效cursor解码: {values} - 验证通过")
    except ValueError as e:
        print(f"验证失败: {e}")

    invalid_cursor = CursorEncoder.encode((90,))
    try:
        values = paginator.validate_cursor(invalid_cursor)
        print(f"无效cursor解码: {values} - 验证通过（不应发生）")
    except ValueError as e:
        print(f"无效cursor验证失败（预期）: {e}")


def test_invalid_page_size():
    print("\n" + "=" * 70)
    print("测试7: 无效page_size验证")
    print("=" * 70)

    paginator = KeysetPaginator(
        sort_fields=[("id", "asc")],
        auto_append_unique_key=False,
        warn_on_duplicate_risk=False
    )

    try:
        paginator.paginate(test_data, page_size=0)
        print("page_size=0 未抛出异常（错误）")
    except ValueError as e:
        print(f"page_size=0 抛出异常（预期）: {e}")

    try:
        paginator.paginate(test_data, page_size=-1)
        print("page_size=-1 未抛出异常（错误）")
    except ValueError as e:
        print(f"page_size=-1 抛出异常（预期）: {e}")


def test_custom_unique_key():
    print("\n" + "=" * 70)
    print("测试8: 自定义唯一键字段名")
    print("=" * 70)

    custom_data = [
        {"uuid": 101, "name": "张三", "score": 90},
        {"uuid": 102, "name": "李四", "score": 90},
        {"uuid": 103, "name": "王五", "score": 85},
    ]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        paginator = KeysetPaginator(
            sort_fields=[("score", "desc")],
            unique_key="uuid",
            auto_append_unique_key=True
        )

        print(f"\n排序字段: {paginator.sort_fields}")
        print(f"唯一键字段: {paginator.unique_key}")

    result = paginator.paginate(custom_data, page_size=2, cursor=None)
    print(f"\n首页数据 (score, uuid, name): {[(item['score'], item['uuid'], item['name']) for item in result.data]}")
    print(f"下一页游标解码: {CursorEncoder.decode(result.next_cursor)}")


if __name__ == "__main__":
    test_auto_append_unique_key()
    test_no_auto_append_with_warning()
    test_manual_unique_key_last()
    test_unique_key_not_last_warning()
    test_first_page_no_cursor()
    test_cursor_validation()
    test_invalid_page_size()
    test_custom_unique_key()
