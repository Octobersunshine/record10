import warnings
from keyset_pagination import (
    UnifiedPaginator,
    PaginationMethod,
    CursorEncoder
)


sample_data = [
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
]


def demo_unified_paginator():
    print("=" * 70)
    print("示例1: UnifiedPaginator - 智能选择分页方式")
    print("=" * 70)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paginator = UnifiedPaginator(
            sort_fields=[("score", "desc"), ("id", "asc")],
            auto_append_unique_key=False,
            offset_threshold=5,
            large_offset_threshold=20
        )

    print("\n1. 第1页（小偏移，自动选择OFFSET）:")
    result = paginator.paginate(
        items=sample_data,
        page_size=3,
        page=1,
        total_items=len(sample_data)
    )
    print(f"   分页方式: {result.page_info.method.value}")
    print(f"   推荐理由: {result.page_info.recommendation}")
    print(f"   数据: {[(item['score'], item['id'], item['name']) for item in result.data]}")
    print(f"   total_items: {result.page_info.total_items}, total_pages: {result.page_info.total_pages}")

    print("\n2. 第10页（大偏移，自动选择KEYSET）:")
    result = paginator.paginate(
        items=sample_data,
        page_size=3,
        page=10
    )
    print(f"   分页方式: {result.page_info.method.value}")
    print(f"   推荐理由: {result.page_info.recommendation}")
    print(f"   数据: {[(item['score'], item['id'], item['name']) for item in result.data]}")
    print(f"   next_cursor: {result.next_cursor}")

    print("\n3. 使用cursor（强制KEYSET）:")
    cursor = CursorEncoder.encode((90, 5))
    result = paginator.paginate(
        items=sample_data,
        page_size=3,
        cursor=cursor
    )
    print(f"   分页方式: {result.page_info.method.value}")
    print(f"   推荐理由: {result.page_info.recommendation}")
    print(f"   数据: {[(item['score'], item['id'], item['name']) for item in result.data]}")


def demo_offset_pagination():
    print("\n" + "=" * 70)
    print("示例2: 传统OFFSET分页")
    print("=" * 70)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paginator = UnifiedPaginator(
            sort_fields=[("score", "desc"), ("id", "asc")],
            auto_append_unique_key=False
        )

    print("\n1. 按page分页（第3页，每页3条）:")
    result = paginator.paginate(
        items=sample_data,
        page_size=3,
        page=3,
        method=PaginationMethod.OFFSET
    )
    print(f"   数据: {[(item['score'], item['id'], item['name']) for item in result.data]}")
    print(f"   page={result.page_info.page}, offset={result.page_info.offset}")
    print(f"   has_next={result.page_info.has_next}, has_previous={result.page_info.has_previous}")
    print(f"   next_page={result.page_info.next_page}, previous_page={result.page_info.previous_page}")

    print("\n2. 按offset分页（offset=5，每页3条）:")
    result = paginator.paginate(
        items=sample_data,
        page_size=3,
        offset=5,
        method=PaginationMethod.OFFSET
    )
    print(f"   数据: {[(item['score'], item['id'], item['name']) for item in result.data]}")
    print(f"   自动计算page={result.page_info.page}")


def demo_keyset_pagination():
    print("\n" + "=" * 70)
    print("示例3: 游标分页（Keyset Pagination）")
    print("=" * 70)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paginator = UnifiedPaginator(
            sort_fields=[("score", "desc"), ("id", "asc")],
            auto_append_unique_key=False
        )

    print("\n1. 顺序翻页（从首页开始）:")
    page_size = 4
    cursor = None

    for page_num in range(1, 5):
        result = paginator.paginate(
            items=sample_data,
            page_size=page_size,
            cursor=cursor,
            page=page_num,
            method=PaginationMethod.KEYSET,
            estimate_total=True
        )

        print(f"\n   第 {page_num} 页:")
        print(f"      数据: {[(item['score'], item['id'], item['name']) for item in result.data]}")
        print(f"      next_cursor: {result.next_cursor}")
        print(f"      total_items: {result.page_info.total_items}"
              f"{' (估算)' if result.page_info.estimated else ' (精确)'}")

        if not result.page_info.has_next:
            break

        cursor = result.next_cursor

    print("\n2. 带总记录数估算:")
    cursor = CursorEncoder.encode((90, 5))
    result = paginator.paginate(
        items=sample_data,
        page_size=4,
        cursor=cursor,
        page=2,
        method=PaginationMethod.KEYSET,
        estimate_total=True
    )
    print(f"   数据: {[(item['score'], item['id'], item['name']) for item in result.data]}")
    print(f"   total_items: {result.page_info.total_items}")
    print(f"   total_pages: {result.page_info.total_pages}")
    print(f"   estimated: {result.page_info.estimated}")


def demo_sql_generation():
    print("\n" + "=" * 70)
    print("示例4: SQL查询生成")
    print("=" * 70)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paginator = UnifiedPaginator(
            sort_fields=[("score", "desc"), ("id", "asc")],
            auto_append_unique_key=False,
            offset_threshold=100
        )

    base_query = "SELECT id, name, score FROM users"

    print("\n1. OFFSET分页SQL:")
    query, params, count_query, method, recommendation = paginator.build_sql_query(
        base_query=base_query,
        page_size=10,
        page=3,
        method=PaginationMethod.OFFSET
    )
    print(f"   方式: {method.value}")
    print(f"   查询: {query}")
    print(f"   COUNT查询: {count_query}")

    print("\n2. KEYSET分页SQL（带cursor）:")
    cursor = CursorEncoder.encode((90, 5))
    query, params, count_query, method, recommendation = paginator.build_sql_query(
        base_query=base_query,
        page_size=10,
        cursor=cursor,
        method=PaginationMethod.KEYSET
    )
    print(f"   方式: {method.value}")
    print(f"   查询: {query}")
    print(f"   参数: {params}")

    print("\n3. 大偏移自动推荐KEYSET:")
    query, params, count_query, method, recommendation = paginator.build_sql_query(
        base_query=base_query,
        page_size=20,
        page=100
    )
    print(f"   方式: {method.value}")
    print(f"   推荐理由: {recommendation}")
    print(f"   查询: {query}")


def demo_auto_recommendation():
    print("\n" + "=" * 70)
    print("示例5: 智能推荐演示")
    print("=" * 70)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paginator = UnifiedPaginator(
            sort_fields=[("score", "desc"), ("id", "asc")],
            auto_append_unique_key=False,
            offset_threshold=100,
            large_offset_threshold=1000
        )

    test_cases = [
        (1, 10, "第1页，小偏移"),
        (5, 10, "第5页，偏移40"),
        (11, 10, "第11页，偏移100（达到阈值）"),
        (51, 20, "第51页，偏移1000（达到大偏移阈值）"),
        (200, 50, "第200页，偏移9950（大偏移）"),
    ]

    for page, page_size, desc in test_cases:
        result = paginator.paginate(
            items=sample_data,
            page_size=page_size,
            page=page
        )
        offset = (page - 1) * page_size
        print(f"\n{desc} (offset={offset}):")
        print(f"   选择方式: {result.page_info.method.value}")
        print(f"   推荐理由: {result.page_info.recommendation}")


def demo_page_info_details():
    print("\n" + "=" * 70)
    print("示例6: PageInfo完整信息")
    print("=" * 70)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paginator = UnifiedPaginator(
            sort_fields=[("score", "desc"), ("id", "asc")],
            auto_append_unique_key=False
        )

    result = paginator.paginate(
        items=sample_data,
        page_size=4,
        page=2,
        total_items=1000
    )

    info = result.page_info
    print("\nPageInfo完整字段:")
    print(f"  page: {info.page}                  # 当前页码")
    print(f"  page_size: {info.page_size}            # 每页大小")
    print(f"  total_items: {info.total_items}         # 总记录数")
    print(f"  total_pages: {info.total_pages}         # 总页数")
    print(f"  has_next: {info.has_next}           # 是否有下一页")
    print(f"  has_previous: {info.has_previous}       # 是否有上一页")
    print(f"  next_page: {info.next_page}             # 下一页页码")
    print(f"  previous_page: {info.previous_page}         # 上一页页码")
    print(f"  offset: {info.offset}               # 偏移量")
    print(f"  method: {info.method.value}             # 分页方式")
    print(f"  recommendation: {info.recommendation}  # 推荐说明")
    print(f"  estimated: {info.estimated}            # 是否为估算值")

    print(f"\n数据: {[(item['id'], item['name']) for item in result.data]}")


if __name__ == "__main__":
    demo_unified_paginator()
    demo_offset_pagination()
    demo_keyset_pagination()
    demo_sql_generation()
    demo_auto_recommendation()
    demo_page_info_details()
