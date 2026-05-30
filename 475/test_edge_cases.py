from keyset_pagination import KeysetPaginator, CursorEncoder


test_data = [
    {"id": 1, "name": "张三", "score": 90},
    {"id": 2, "name": "李四", "score": 90},
    {"id": 3, "name": "王五", "score": 90},
    {"id": 4, "name": "赵六", "score": 85},
    {"id": 5, "name": "钱七", "score": 85},
]


def test_single_field_duplicate_boundary():
    print("=" * 70)
    print("测试: 单字段排序，分页边界在重复值中间（应遗漏数据）")
    print("=" * 70)

    paginator = KeysetPaginator(sort_fields=[("score", "desc")])

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
    if missing_ids:
        print(f"  问题: 因为cursor只包含score=90，下一页查询score < 90，漏掉了score=90的id=3!")


def test_multi_field_duplicate_boundary():
    print("\n" + "=" * 70)
    print("测试: 多字段排序（score + id），分页边界在重复值中间（不应遗漏）")
    print("=" * 70)

    paginator = KeysetPaginator(sort_fields=[("score", "desc"), ("id", "asc")])

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
    if not missing_ids:
        print(f"  正确: cursor包含(score, id)=(90,2)，查询条件是score < 90 OR (score=90 AND id > 2)，能正确获取id=3!")


def test_sql_comparison():
    print("\n" + "=" * 70)
    print("SQL对比: 单字段 vs 多字段")
    print("=" * 70)

    base_query = "SELECT id, name, score FROM users"

    print("\n--- 单字段排序 (score DESC) ---")
    paginator1 = KeysetPaginator(sort_fields=[("score", "desc")])
    cursor1 = CursorEncoder.encode((90,))
    query1, params1 = paginator1.build_sql_query(base_query, 10, cursor1, True)
    print(f"  游标: (90,)")
    print(f"  SQL: {query1}")
    print(f"  问题: 漏掉所有score=90且id > 2的记录!")

    print("\n--- 多字段排序 (score DESC, id ASC) ---")
    paginator2 = KeysetPaginator(sort_fields=[("score", "desc"), ("id", "asc")])
    cursor2 = CursorEncoder.encode((90, 2))
    query2, params2 = paginator2.build_sql_query(base_query, 10, cursor2, True)
    print(f"  游标: (90, 2)")
    print(f"  SQL: {query2}")
    print(f"  正确: 包含score=90且id > 2的记录!")


if __name__ == "__main__":
    test_single_field_duplicate_boundary()
    test_multi_field_duplicate_boundary()
    test_sql_comparison()
