import warnings
from keyset_pagination import (
    UnifiedPaginator,
    OffsetPaginator,
    KeysetPaginator,
    PaginationRecommender,
    PaginationMethod,
    PageResult,
    PageInfo,
    CursorEncoder
)


test_data = [
    {"id": 1, "name": "张三", "score": 95},
    {"id": 2, "name": "李四", "score": 88},
    {"id": 3, "name": "王五", "score": 92},
    {"id": 4, "name": "赵六", "score": 85},
    {"id": 5, "name": "钱七", "score": 90},
    {"id": 6, "name": "孙八", "score": 87},
    {"id": 7, "name": "周九", "score": 93},
    {"id": 8, "name": "吴十", "score": 89},
    {"id": 9, "name": "郑一", "score": 91},
    {"id": 10, "name": "王二", "score": 86},
    {"id": 11, "name": "陈三", "score": 94},
    {"id": 12, "name": "林四", "score": 84},
    {"id": 13, "name": "黄五", "score": 83},
    {"id": 14, "name": "刘六", "score": 82},
    {"id": 15, "name": "杨七", "score": 81},
]


def test_offset_paginator():
    print("=" * 70)
    print("测试1: OffsetPaginator - 传统OFFSET分页")
    print("=" * 70)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paginator = OffsetPaginator(
            sort_fields=[("score", "desc"), ("id", "asc")],
            auto_append_unique_key=False,
            warn_on_duplicate_risk=False
        )

    print("\n--- 按page分页 ---")
    for page in range(1, 6):
        result = paginator.paginate(
            items=test_data,
            page_size=3,
            page=page
        )

        print(f"\n第 {page} 页:")
        print(f"  数据 (score, id, name): {[(item['score'], item['id'], item['name']) for item in result.data]}")
        print(f"  page_info: page={result.page_info.page}, total_items={result.page_info.total_items}, "
              f"total_pages={result.page_info.total_pages}")
        print(f"  has_next={result.page_info.has_next}, has_previous={result.page_info.has_previous}")
        print(f"  offset={result.page_info.offset}, method={result.page_info.method.value}")

        if not result.page_info.has_next:
            break

    print("\n--- 按offset分页 ---")
    result = paginator.paginate(
        items=test_data,
        page_size=3,
        offset=6
    )
    print(f"\noffset=6, page_size=3:")
    print(f"  数据: {[(item['score'], item['id'], item['name']) for item in result.data]}")
    print(f"  自动计算page: {result.page_info.page}")

    print("\n--- 手动指定total_items ---")
    result = paginator.paginate(
        items=test_data[:3],
        page_size=3,
        page=1,
        total_items=1000
    )
    print(f"  数据长度: {len(result.data)}")
    print(f"  total_items: {result.page_info.total_items} (手动指定)")
    print(f"  total_pages: {result.page_info.total_pages}")


def test_pagination_recommender():
    print("\n" + "=" * 70)
    print("测试2: PaginationRecommender - 智能推荐")
    print("=" * 70)

    recommender = PaginationRecommender(
        offset_threshold=100,
        large_offset_threshold=1000
    )

    test_cases = [
        {"page": 1, "page_size": 10, "desc": "首页，小偏移"},
        {"page": 5, "page_size": 10, "desc": "第5页，偏移40"},
        {"page": 11, "page_size": 10, "desc": "第11页，偏移100（达到阈值）"},
        {"page": 50, "page_size": 20, "desc": "第50页，偏移980（接近大偏移阈值）"},
        {"page": 101, "page_size": 10, "desc": "第101页，偏移1000（达到大偏移阈值）"},
        {"page": 200, "page_size": 50, "desc": "第200页，偏移9950（大偏移）"},
    ]

    for test_case in test_cases:
        method, recommendation = recommender.recommend(
            page=test_case["page"],
            page_size=test_case["page_size"]
        )
        offset = (test_case["page"] - 1) * test_case["page_size"]
        print(f"\n{test_case['desc']} (offset={offset}):")
        print(f"  推荐方式: {method.value}")
        print(f"  推荐理由: {recommendation}")

    print("\n--- 有cursor时强制keyset ---")
    method, recommendation = recommender.recommend(page=1, page_size=10, has_cursor=True)
    print(f"  推荐方式: {method.value}")
    print(f"  推荐理由: {recommendation}")

    print("\n--- prefer_keyset=True ---")
    method, recommendation = recommender.recommend(page=2, page_size=10, prefer_keyset=True)
    print(f"  推荐方式: {method.value}")
    print(f"  推荐理由: {recommendation}")


def test_keyset_with_estimate():
    print("\n" + "=" * 70)
    print("测试3: KeysetPaginator - 带总记录数估算")
    print("=" * 70)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paginator = KeysetPaginator(
            sort_fields=[("score", "desc"), ("id", "asc")],
            auto_append_unique_key=False,
            warn_on_duplicate_risk=False
        )

    print("\n--- 首页，estimate_total=True (精确计数) ---")
    result = paginator.paginate(
        items=test_data,
        page_size=5,
        cursor=None,
        page=1,
        estimate_total=True
    )
    print(f"  数据: {[(item['score'], item['id']) for item in result.data]}")
    print(f"  total_items: {result.page_info.total_items}")
    print(f"  total_pages: {result.page_info.total_pages}")
    print(f"  estimated: {result.page_info.estimated}")

    print("\n--- 第2页，estimate_total=True (估算) ---")
    cursor = result.next_cursor
    result_page2 = paginator.paginate(
        items=test_data,
        page_size=5,
        cursor=cursor,
        page=2,
        estimate_total=True
    )
    print(f"  数据: {[(item['score'], item['id']) for item in result_page2.data]}")
    print(f"  total_items: {result_page2.page_info.total_items}")
    print(f"  total_pages: {result_page2.page_info.total_pages}")
    print(f"  estimated: {result_page2.page_info.estimated}")

    print("\n--- 不估算 ---")
    result_no_est = paginator.paginate(
        items=test_data,
        page_size=5,
        cursor=cursor,
        page=2,
        estimate_total=False
    )
    print(f"  total_items: {result_no_est.page_info.total_items} (None)")
    print(f"  estimated: {result_no_est.page_info.estimated}")


def test_unified_paginator_auto():
    print("\n" + "=" * 70)
    print("测试4: UnifiedPaginator - 自动选择分页方式")
    print("=" * 70)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paginator = UnifiedPaginator(
            sort_fields=[("score", "desc"), ("id", "asc")],
            auto_append_unique_key=False,
            offset_threshold=5,
            large_offset_threshold=10
        )

    print("\n--- 第1页（小偏移，自动选择OFFSET）---")
    result = paginator.paginate(
        items=test_data,
        page_size=3,
        page=1
    )
    print(f"  分页方式: {result.page_info.method.value}")
    print(f"  推荐理由: {result.page_info.recommendation}")
    print(f"  数据: {[(item['score'], item['id']) for item in result.data]}")

    print("\n--- 第10页（大偏移，自动选择KEYSET）---")
    result = paginator.paginate(
        items=test_data,
        page_size=3,
        page=10
    )
    print(f"  分页方式: {result.page_info.method.value}")
    print(f"  推荐理由: {result.page_info.recommendation}")
    print(f"  数据: {[(item['score'], item['id']) for item in result.data]}")

    print("\n--- 有cursor（强制KEYSET）---")
    cursor = CursorEncoder.encode((90, 5))
    result = paginator.paginate(
        items=test_data,
        page_size=3,
        cursor=cursor
    )
    print(f"  分页方式: {result.page_info.method.value}")
    print(f"  推荐理由: {result.page_info.recommendation}")
    print(f"  数据: {[(item['score'], item['id']) for item in result.data]}")


def test_unified_paginator_force_method():
    print("\n" + "=" * 70)
    print("测试5: UnifiedPaginator - 强制指定分页方式")
    print("=" * 70)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paginator = UnifiedPaginator(
            sort_fields=[("score", "desc"), ("id", "asc")],
            auto_append_unique_key=False
        )

    print("\n--- 强制使用OFFSET ---")
    result = paginator.paginate(
        items=test_data,
        page_size=3,
        page=2,
        method=PaginationMethod.OFFSET
    )
    print(f"  分页方式: {result.page_info.method.value}")
    print(f"  推荐理由: {result.page_info.recommendation}")
    print(f"  数据: {[(item['score'], item['id']) for item in result.data]}")
    print(f"  offset: {result.page_info.offset}")

    print("\n--- 强制使用KEYSET ---")
    result = paginator.paginate(
        items=test_data,
        page_size=3,
        page=2,
        method=PaginationMethod.KEYSET,
        estimate_total=True
    )
    print(f"  分页方式: {result.page_info.method.value}")
    print(f"  推荐理由: {result.page_info.recommendation}")
    print(f"  数据: {[(item['score'], item['id']) for item in result.data]}")
    print(f"  next_cursor: {result.next_cursor}")


def test_sql_generation():
    print("\n" + "=" * 70)
    print("测试6: SQL生成 - 两种分页方式")
    print("=" * 70)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paginator = UnifiedPaginator(
            sort_fields=[("score", "desc"), ("id", "asc")],
            auto_append_unique_key=False,
            offset_threshold=100
        )

    base_query = "SELECT id, name, score FROM users"

    print("\n--- OFFSET分页SQL ---")
    query, params, count_query, method, recommendation = paginator.build_sql_query(
        base_query=base_query,
        page_size=10,
        page=5,
        method=PaginationMethod.OFFSET
    )
    print(f"  方式: {method.value}")
    print(f"  查询: {query}")
    print(f"  参数: {params}")
    print(f"  COUNT查询: {count_query}")

    print("\n--- KEYSET分页SQL ---")
    cursor = CursorEncoder.encode((90, 5))
    query, params, count_query, method, recommendation = paginator.build_sql_query(
        base_query=base_query,
        page_size=10,
        cursor=cursor,
        method=PaginationMethod.KEYSET
    )
    print(f"  方式: {method.value}")
    print(f"  查询: {query}")
    print(f"  参数: {params}")
    print(f"  COUNT查询: {count_query}")

    print("\n--- 自动选择（大偏移）---")
    query, params, count_query, method, recommendation = paginator.build_sql_query(
        base_query=base_query,
        page_size=10,
        page=200
    )
    print(f"  方式: {method.value}")
    print(f"  推荐理由: {recommendation}")
    print(f"  查询: {query}")


def test_total_items_with_unified():
    print("\n" + "=" * 70)
    print("测试7: 总记录数处理 - UnifiedPaginator")
    print("=" * 70)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paginator = UnifiedPaginator(
            sort_fields=[("score", "desc"), ("id", "asc")],
            auto_append_unique_key=False
        )

    print("\n--- OFFSET分页，手动指定total_items ---")
    result = paginator.paginate(
        items=test_data,
        page_size=5,
        page=1,
        total_items=5000
    )
    print(f"  方法: {result.page_info.method.value}")
    print(f"  total_items: {result.page_info.total_items}")
    print(f"  total_pages: {result.page_info.total_pages}")
    print(f"  estimated: {result.page_info.estimated}")

    print("\n--- KEYSET分页，手动指定total_items ---")
    cursor = CursorEncoder.encode((90, 5))
    result = paginator.paginate(
        items=test_data,
        page_size=5,
        cursor=cursor,
        page=2,
        total_items=5000
    )
    print(f"  方法: {result.page_info.method.value}")
    print(f"  total_items: {result.page_info.total_items}")
    print(f"  total_pages: {result.page_info.total_pages}")
    print(f"  estimated: {result.page_info.estimated} (手动指定后为False)")

    print("\n--- KEYSET分页，自动估算 ---")
    result = paginator.paginate(
        items=test_data,
        page_size=5,
        cursor=cursor,
        page=2,
        estimate_total=True
    )
    print(f"  方法: {result.page_info.method.value}")
    print(f"  total_items: {result.page_info.total_items}")
    print(f"  total_pages: {result.page_info.total_pages}")
    print(f"  estimated: {result.page_info.estimated}")


def test_page_info_structure():
    print("\n" + "=" * 70)
    print("测试8: PageInfo和PageResult结构")
    print("=" * 70)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paginator = UnifiedPaginator(
            sort_fields=[("score", "desc"), ("id", "asc")],
            auto_append_unique_key=False
        )

    result = paginator.paginate(
        items=test_data,
        page_size=3,
        page=2
    )

    print("\nPageResult结构:")
    print(f"  data类型: {type(result.data).__name__}, 长度: {len(result.data)}")
    print(f"  next_cursor: {result.next_cursor}")
    print(f"  previous_cursor: {result.previous_cursor}")
    print(f"  extra: {result.extra}")

    print("\nPageInfo结构:")
    info = result.page_info
    print(f"  page: {info.page}")
    print(f"  page_size: {info.page_size}")
    print(f"  total_items: {info.total_items}")
    print(f"  total_pages: {info.total_pages}")
    print(f"  has_next: {info.has_next}")
    print(f"  has_previous: {info.has_previous}")
    print(f"  next_page: {info.next_page}")
    print(f"  previous_page: {info.previous_page}")
    print(f"  offset: {info.offset}")
    print(f"  method: {info.method.value}")
    print(f"  recommendation: {info.recommendation}")
    print(f"  estimated: {info.estimated}")


if __name__ == "__main__":
    test_offset_paginator()
    test_pagination_recommender()
    test_keyset_with_estimate()
    test_unified_paginator_auto()
    test_unified_paginator_force_method()
    test_sql_generation()
    test_total_items_with_unified()
    test_page_info_structure()
