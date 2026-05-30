from keyset_pagination import KeysetPaginator, CursorEncoder


test_data = [
    {"id": 1, "name": "张三", "score": 90},
    {"id": 2, "name": "李四", "score": 90},
    {"id": 3, "name": "王五", "score": 90},
    {"id": 4, "name": "赵六", "score": 85},
    {"id": 5, "name": "钱七", "score": 85},
    {"id": 6, "name": "孙八", "score": 85},
    {"id": 7, "name": "周九", "score": 80},
    {"id": 8, "name": "吴十", "score": 80},
    {"id": 9, "name": "郑一", "score": 80},
    {"id": 10, "name": "王二", "score": 75},
]


def test_single_field_with_duplicates():
    print("=" * 70)
    print("测试1: 单字段排序（有重复值）- 可能遗漏数据")
    print("=" * 70)

    paginator = KeysetPaginator(sort_fields=[("score", "desc")])

    page_size = 3
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


def test_multi_field_with_unique_id():
    print("\n" + "=" * 70)
    print("测试2: 多字段排序（score + id）- 确保唯一性")
    print("=" * 70)

    paginator = KeysetPaginator(sort_fields=[("score", "desc"), ("id", "asc")])

    page_size = 3
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


def test_sql_generation_multi_field():
    print("\n" + "=" * 70)
    print("测试3: 多字段排序SQL生成")
    print("=" * 70)

    paginator = KeysetPaginator(sort_fields=[("score", "desc"), ("id", "asc")])
    base_query = "SELECT id, name, score FROM users"

    cursor = CursorEncoder.encode((90, 3))
    print(f"\n游标值 (score=90, id=3):")
    query, params = paginator.build_sql_query(
        base_query=base_query,
        page_size=10,
        cursor=cursor,
        is_forward=True
    )
    print(f"  SQL: {query}")
    print(f"  参数: {params}")

    print(f"\n向后翻页SQL:")
    query, params = paginator.build_sql_query(
        base_query=base_query,
        page_size=10,
        cursor=cursor,
        is_forward=False
    )
    print(f"  SQL: {query}")
    print(f"  参数: {params}")


def test_first_page_no_cursor():
    print("\n" + "=" * 70)
    print("测试4: 首页查询（cursor为None）")
    print("=" * 70)

    paginator = KeysetPaginator(sort_fields=[("score", "desc"), ("id", "asc")])

    result = paginator.paginate(
        items=test_data,
        page_size=5,
        cursor=None,
        is_forward=True
    )

    print(f"\n首页数据 (score, id, name): {[(item['score'], item['id'], item['name']) for item in result.data]}")
    print(f"has_previous: {result.has_previous} (应为 False)")
    print(f"previous_cursor: {result.previous_cursor} (应为 None)")


def test_backward_pagination():
    print("\n" + "=" * 70)
    print("测试5: 双向翻页验证")
    print("=" * 70)

    paginator = KeysetPaginator(sort_fields=[("score", "desc"), ("id", "asc")])

    result_page1 = paginator.paginate(test_data, 3, None, True)
    print(f"\n第1页: {[(item['score'], item['id']) for item in result_page1.data]}")

    result_page2 = paginator.paginate(test_data, 3, result_page1.next_cursor, True)
    print(f"第2页: {[(item['score'], item['id']) for item in result_page2.data]}")

    result_page1_again = paginator.paginate(test_data, 3, result_page2.previous_cursor, False)
    print(f"从第2页翻回第1页: {[(item['score'], item['id']) for item in result_page1_again.data]}")

    is_same = result_page1.data == result_page1_again.data
    print(f"数据一致: {is_same}")


if __name__ == "__main__":
    test_single_field_with_duplicates()
    test_multi_field_with_unique_id()
    test_sql_generation_multi_field()
    test_first_page_no_cursor()
    test_backward_pagination()
