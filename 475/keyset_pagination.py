import base64
import json
import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


UNIQUE_KEY_FIELDS = {'id', 'uuid', 'pk', 'primary_key'}
DEFAULT_OFFSET_THRESHOLD = 1000
DEFAULT_LARGE_OFFSET_THRESHOLD = 10000


class PaginationMethod(str, Enum):
    OFFSET = "offset"
    KEYSET = "keyset"


@dataclass
class PageInfo:
    page: Optional[int] = None
    page_size: int = 10
    total_items: Optional[int] = None
    total_pages: Optional[int] = None
    has_next: bool = False
    has_previous: bool = False
    next_page: Optional[int] = None
    previous_page: Optional[int] = None
    offset: Optional[int] = None
    method: PaginationMethod = PaginationMethod.OFFSET
    recommendation: Optional[str] = None
    estimated: bool = False


@dataclass
class PageResult:
    data: List[Any]
    page_info: PageInfo
    next_cursor: Optional[str] = None
    previous_cursor: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class CursorEncoder:
    @staticmethod
    def encode(values: Tuple[Any, ...]) -> str:
        cursor_data = json.dumps(values, ensure_ascii=False)
        return base64.urlsafe_b64encode(cursor_data.encode('utf-8')).decode('utf-8')

    @staticmethod
    def decode(cursor: str) -> Tuple[Any, ...]:
        try:
            decoded = base64.urlsafe_b64decode(cursor.encode('utf-8'))
            return tuple(json.loads(decoded.decode('utf-8')))
        except Exception as e:
            raise ValueError(f"Invalid cursor format: {e}")


class TotalCountEstimator:
    @staticmethod
    def estimate_exponential(
        items: List[Any],
        cursor_values: Tuple[Any, ...],
        sort_fields: List[Tuple[str, str]],
        sample_size: int = 100
    ) -> Tuple[int, bool]:
        if not items:
            return 0, False

        if len(items) < sample_size:
            return len(items), False

        sorted_items = items[:sample_size]
        if len(sort_fields) == 0:
            return len(items), False

        field_name, direction = sort_fields[0]
        is_desc = direction.lower() == 'desc'

        values = []
        for item in sorted_items:
            if isinstance(item, dict):
                val = item.get(field_name)
            else:
                val = getattr(item, field_name, None)
            if val is not None and isinstance(val, (int, float)):
                values.append(val)

        if len(values) < 2:
            return len(items), False

        values.sort(reverse=is_desc)

        cursor_val = cursor_values[0]
        if not isinstance(cursor_val, (int, float)):
            return len(items), True

        min_val = min(values)
        max_val = max(values)
        value_range = max_val - min_val

        if value_range <= 0:
            return len(items), True

        if is_desc:
            position_ratio = abs(cursor_val - min_val) / value_range
        else:
            position_ratio = abs(max_val - cursor_val) / value_range
        position_ratio = max(0.01, min(0.99, position_ratio))

        if is_desc:
            processed_count = len([v for v in values if v < cursor_val])
        else:
            processed_count = len([v for v in values if v > cursor_val])
        remaining_estimate = int(processed_count / position_ratio) - processed_count
        total_estimate = processed_count + remaining_estimate

        return max(len(items), total_estimate), True

    @staticmethod
    def estimate_count_approx(
        base_query: str,
        cursor: Optional[str] = None,
        sample_percent: int = 10
    ) -> Tuple[str, bool]:
        if sample_percent <= 0 or sample_percent > 100:
            sample_percent = 10

        approx_query = f"SELECT COUNT(*) FROM ({base_query}) AS sub TABLESAMPLE SYSTEM ({sample_percent})"
        estimate_query = f"SELECT ({sample_percent} / 100) * COUNT(*) FROM ({base_query}) AS sub"

        return estimate_query, True

    @staticmethod
    def estimate_from_last_page(
        page: int,
        page_size: int,
        last_page_cursor: Optional[str]
    ) -> Tuple[int, bool]:
        if last_page_cursor is None:
            return page * page_size, True

        estimate = page * page_size
        return estimate, True


class PaginationRecommender:
    def __init__(
        self,
        offset_threshold: int = DEFAULT_OFFSET_THRESHOLD,
        large_offset_threshold: int = DEFAULT_LARGE_OFFSET_THRESHOLD
    ):
        self.offset_threshold = offset_threshold
        self.large_offset_threshold = large_offset_threshold

    def recommend(
        self,
        page: Optional[int] = None,
        offset: Optional[int] = None,
        page_size: int = 10,
        total_items: Optional[int] = None,
        has_cursor: bool = False,
        prefer_keyset: bool = False
    ) -> Tuple[PaginationMethod, str]:
        if has_cursor:
            return PaginationMethod.KEYSET, "使用游标分页，性能稳定"

        current_offset = offset if offset is not None else ((page - 1) * page_size if page else 0)

        if prefer_keyset and current_offset > 0:
            return PaginationMethod.KEYSET, f"偏移量 {current_offset} 较大，推荐使用游标分页避免性能问题"

        if current_offset >= self.large_offset_threshold:
            return PaginationMethod.KEYSET, f"偏移量 {current_offset} 超过 {self.large_offset_threshold}，强烈建议使用游标分页"

        if current_offset >= self.offset_threshold:
            return (PaginationMethod.KEYSET,
                    f"偏移量 {current_offset} 超过 {self.offset_threshold}，游标分页性能更优")

        return PaginationMethod.OFFSET, f"偏移量 {current_offset} 较小，OFFSET分页足够高效"


class OffsetPaginator:
    def __init__(
        self,
        sort_fields: List[Tuple[str, str]],
        unique_key: Optional[str] = 'id',
        auto_append_unique_key: bool = True,
        warn_on_duplicate_risk: bool = True
    ):
        self.sort_fields = self._process_sort_fields(
            sort_fields,
            unique_key,
            auto_append_unique_key,
            warn_on_duplicate_risk
        )
        self.field_names = [field for field, _ in self.sort_fields]
        self.unique_key = unique_key

    def _process_sort_fields(
        self,
        sort_fields: List[Tuple[str, str]],
        unique_key: Optional[str],
        auto_append_unique_key: bool,
        warn_on_duplicate_risk: bool
    ) -> List[Tuple[str, str]]:
        if not sort_fields:
            raise ValueError("sort_fields cannot be empty")

        field_names = [field.lower() for field, _ in sort_fields]
        last_field = field_names[-1]

        has_unique_key = any(
            field.lower() in UNIQUE_KEY_FIELDS
            for field in field_names
        )

        last_is_unique = (
            unique_key and last_field == unique_key.lower()
        ) or last_field in UNIQUE_KEY_FIELDS

        if not last_is_unique and warn_on_duplicate_risk:
            if has_unique_key:
                warnings.warn(
                    f"排序字段包含唯一键但不在最后位置，建议将唯一键字段移到最后。"
                    f"当前排序: {[f for f, _ in sort_fields]}",
                    UserWarning
                )
            else:
                warnings.warn(
                    f"排序字段组合不包含唯一键字段（如id），当存在重复值时可能导致分页不稳定。"
                    f"建议添加唯一键字段作为最后一个排序条件。当前排序: {[f for f, _ in sort_fields]}",
                    UserWarning
                )

        if auto_append_unique_key and unique_key and not has_unique_key:
            processed = list(sort_fields)
            processed.append((unique_key, 'asc'))
            warnings.warn(
                f"已自动添加唯一键字段 '{unique_key}' 作为最后一个排序条件。"
                f"最终排序: {[f for f, _ in processed]}",
                UserWarning
            )
            return processed

        return list(sort_fields)

    def _sort_items(self, items: List[Any], reverse: bool = False) -> List[Any]:
        sorted_items = items[:]

        for field, direction in reversed(self.sort_fields):
            is_desc = direction.lower() == 'desc'
            if reverse:
                is_desc = not is_desc

            sorted_items.sort(
                key=lambda item: (
                    item[field] if isinstance(item, dict) else getattr(item, field)
                ),
                reverse=is_desc
            )

        return sorted_items

    def paginate(
        self,
        items: List[Any],
        page_size: int,
        page: Optional[int] = None,
        offset: Optional[int] = None,
        total_items: Optional[int] = None
    ) -> PageResult:
        if page_size <= 0:
            raise ValueError("page_size must be greater than 0")

        sorted_items = self._sort_items(items)

        if page is not None and offset is None:
            if page <= 0:
                raise ValueError("page must be greater than 0")
            offset = (page - 1) * page_size
        elif offset is not None and page is None:
            if offset < 0:
                raise ValueError("offset must be greater than or equal to 0")
            page = offset // page_size + 1
        elif page is None and offset is None:
            page = 1
            offset = 0

        actual_total = total_items if total_items is not None else len(sorted_items)
        total_pages = math.ceil(actual_total / page_size) if actual_total > 0 else 0

        page_data = sorted_items[offset:offset + page_size]

        has_next = offset + page_size < actual_total
        has_previous = offset > 0

        page_info = PageInfo(
            page=page,
            page_size=page_size,
            total_items=actual_total,
            total_pages=total_pages,
            has_next=has_next,
            has_previous=has_previous,
            next_page=page + 1 if has_next else None,
            previous_page=page - 1 if has_previous else None,
            offset=offset,
            method=PaginationMethod.OFFSET
        )

        return PageResult(
            data=page_data,
            page_info=page_info
        )

    def build_sql_query(
        self,
        base_query: str,
        page_size: int,
        page: Optional[int] = None,
        offset: Optional[int] = None,
        count_total: bool = True
    ) -> Tuple[str, Tuple[Any, ...], Optional[str]]:
        if page_size <= 0:
            raise ValueError("page_size must be greater than 0")

        if page is not None and offset is None:
            if page <= 0:
                raise ValueError("page must be greater than 0")
            offset = (page - 1) * page_size
        elif offset is not None and page is None:
            if offset < 0:
                raise ValueError("offset must be greater than or equal to 0")
        elif page is None and offset is None:
            offset = 0

        order_by_parts = []
        for field, direction in self.sort_fields:
            order_by_parts.append(f"{field} {direction}")
        order_by = ", ".join(order_by_parts)

        query = f"{base_query} ORDER BY {order_by} LIMIT {page_size} OFFSET {offset}"

        count_query = None
        if count_total:
            count_query = f"SELECT COUNT(*) FROM ({base_query}) AS total"

        return query, (), count_query


class KeysetPaginator:
    def __init__(
        self,
        sort_fields: List[Tuple[str, str]],
        unique_key: Optional[str] = 'id',
        auto_append_unique_key: bool = True,
        warn_on_duplicate_risk: bool = True
    ):
        self.sort_fields = self._process_sort_fields(
            sort_fields,
            unique_key,
            auto_append_unique_key,
            warn_on_duplicate_risk
        )
        self.field_names = [field for field, _ in self.sort_fields]
        self.unique_key = unique_key
        self.estimator = TotalCountEstimator()

    def _process_sort_fields(
        self,
        sort_fields: List[Tuple[str, str]],
        unique_key: Optional[str],
        auto_append_unique_key: bool,
        warn_on_duplicate_risk: bool
    ) -> List[Tuple[str, str]]:
        if not sort_fields:
            raise ValueError("sort_fields cannot be empty")

        field_names = [field.lower() for field, _ in sort_fields]
        last_field = field_names[-1]

        has_unique_key = any(
            field.lower() in UNIQUE_KEY_FIELDS
            for field in field_names
        )

        last_is_unique = (
            unique_key and last_field == unique_key.lower()
        ) or last_field in UNIQUE_KEY_FIELDS

        if not last_is_unique and warn_on_duplicate_risk:
            if has_unique_key:
                warnings.warn(
                    f"排序字段包含唯一键但不在最后位置，建议将唯一键字段移到最后以避免重复值导致的数据遗漏。"
                    f"当前排序: {[f for f, _ in sort_fields]}",
                    UserWarning
                )
            else:
                warnings.warn(
                    f"排序字段组合不包含唯一键字段（如id），当存在重复值时可能导致数据遗漏。"
                    f"建议添加唯一键字段作为最后一个排序条件。当前排序: {[f for f, _ in sort_fields]}",
                    UserWarning
                )

        if auto_append_unique_key and unique_key and not has_unique_key:
            processed = list(sort_fields)
            processed.append((unique_key, 'asc'))
            warnings.warn(
                f"已自动添加唯一键字段 '{unique_key}' 作为最后一个排序条件以确保分页正确性。"
                f"最终排序: {[f for f, _ in processed]}",
                UserWarning
            )
            return processed

        return list(sort_fields)

    def validate_cursor(self, cursor: str) -> Tuple[Any, ...]:
        values = CursorEncoder.decode(cursor)
        if len(values) != len(self.sort_fields):
            raise ValueError(
                f"Cursor值数量({len(values)})与排序字段数量({len(self.sort_fields)})不匹配"
            )
        return values

    def _get_cursor_values(self, item: Any) -> Tuple[Any, ...]:
        if isinstance(item, dict):
            return tuple(item[field] for field in self.field_names)
        else:
            return tuple(getattr(item, field) for field in self.field_names)

    def _build_where_clause(self, cursor_values: Tuple[Any, ...], is_forward: bool) -> Tuple[str, Tuple[Any, ...]]:
        conditions = []
        params = []

        for i, (field, direction) in enumerate(self.sort_fields):
            is_desc = direction.lower() == 'desc'

            if i == 0:
                if is_forward:
                    op = '<' if is_desc else '>'
                else:
                    op = '>' if is_desc else '<'
                conditions.append(f"{field} {op} ?")
                params.append(cursor_values[i])
            else:
                prev_conditions = " AND ".join(
                    f"{self.sort_fields[j][0]} = ?" for j in range(i)
                )
                if is_forward:
                    op = '<' if is_desc else '>'
                else:
                    op = '>' if is_desc else '<'
                conditions.append(f"({prev_conditions} AND {field} {op} ?)")
                params.extend(cursor_values[:i])
                params.append(cursor_values[i])

        where_clause = " OR ".join(conditions)
        return where_clause, tuple(params)

    def build_sql_query(
        self,
        base_query: str,
        page_size: int,
        cursor: Optional[str] = None,
        is_forward: bool = True,
        count_total: bool = False
    ) -> Tuple[str, Tuple[Any, ...], Optional[str]]:
        if page_size <= 0:
            raise ValueError("page_size must be greater than 0")

        where_clause = ""
        params = ()

        if cursor:
            cursor_values = self.validate_cursor(cursor)
            where_clause, params = self._build_where_clause(cursor_values, is_forward)

        order_by_parts = []
        for field, direction in self.sort_fields:
            if not is_forward:
                direction = 'DESC' if direction.upper() == 'ASC' else 'ASC'
            order_by_parts.append(f"{field} {direction}")
        order_by = ", ".join(order_by_parts)

        query = base_query
        if where_clause:
            query += f" WHERE {where_clause}"
        query += f" ORDER BY {order_by}"
        query += f" LIMIT {page_size + 1}"

        count_query = None
        if count_total:
            count_query, _ = self.estimator.estimate_count_approx(base_query, cursor)

        return query, params, count_query

    def paginate(
        self,
        items: List[Any],
        page_size: int,
        cursor: Optional[str] = None,
        is_forward: bool = True,
        page: Optional[int] = None,
        estimate_total: bool = False
    ) -> PageResult:
        if page_size <= 0:
            raise ValueError("page_size must be greater than 0")

        original_cursor = cursor
        original_items = items[:]

        if cursor:
            cursor_values = self.validate_cursor(cursor)
            items = self._filter_by_cursor(items, cursor_values, is_forward)
            items = self._sort_items(items, reverse=not is_forward)
        else:
            items = self._sort_items(items)
            cursor_values = self._get_cursor_values(items[0]) if items else ()

        has_more = len(items) > page_size
        page_data = items[:page_size] if has_more else items

        if not is_forward:
            page_data = list(reversed(page_data))

        next_cursor = None
        previous_cursor = None

        if page_data:
            if has_more:
                last_item = page_data[-1]
                next_cursor = CursorEncoder.encode(self._get_cursor_values(last_item))

            if cursor is not None:
                first_item = page_data[0]
                previous_cursor = CursorEncoder.encode(self._get_cursor_values(first_item))

        if not is_forward:
            next_cursor, previous_cursor = previous_cursor, next_cursor
            has_more, has_previous = (cursor is not None), has_more
        else:
            has_previous = cursor is not None

        total_items = None
        total_pages = None
        estimated = False

        if estimate_total:
            if cursor is None and page == 1:
                total_items = len(original_items)
                total_pages = math.ceil(total_items / page_size) if total_items > 0 else 0
                estimated = False
            else:
                estimated_count, is_estimated = self.estimator.estimate_exponential(
                    original_items,
                    cursor_values,
                    self.sort_fields
                )
                total_items = estimated_count
                total_pages = math.ceil(total_items / page_size) if total_items > 0 else None
                estimated = is_estimated

        current_page = page if page is not None else 1
        offset = (current_page - 1) * page_size if current_page > 0 else 0

        page_info = PageInfo(
            page=current_page,
            page_size=page_size,
            total_items=total_items,
            total_pages=total_pages,
            has_next=has_more,
            has_previous=has_previous,
            next_page=current_page + 1 if has_more else None,
            previous_page=current_page - 1 if has_previous else None,
            offset=offset,
            method=PaginationMethod.KEYSET,
            estimated=estimated
        )

        return PageResult(
            data=page_data,
            page_info=page_info,
            next_cursor=next_cursor,
            previous_cursor=previous_cursor,
            extra={
                'cursor_values': self._get_cursor_values(page_data[0]) if page_data else None
            }
        )

    def _filter_by_cursor(
        self,
        items: List[Any],
        cursor_values: Tuple[Any, ...],
        is_forward: bool
    ) -> List[Any]:
        filtered = []
        for item in items:
            item_values = self._get_cursor_values(item)
            if self._compare_keys(item_values, cursor_values, is_forward):
                filtered.append(item)
        return filtered

    def _compare_keys(
        self,
        item_values: Tuple[Any, ...],
        cursor_values: Tuple[Any, ...],
        is_forward: bool
    ) -> bool:
        for i, (item_val, cursor_val, (_, direction)) in enumerate(
            zip(item_values, cursor_values, self.sort_fields)
        ):
            is_desc = direction.lower() == 'desc'

            if item_val == cursor_val:
                continue

            if is_forward:
                if is_desc:
                    return item_val < cursor_val
                else:
                    return item_val > cursor_val
            else:
                if is_desc:
                    return item_val > cursor_val
                else:
                    return item_val < cursor_val

        return False

    def _sort_items(self, items: List[Any], reverse: bool = False) -> List[Any]:
        sorted_items = items[:]

        for field, direction in reversed(self.sort_fields):
            is_desc = direction.lower() == 'desc'
            if reverse:
                is_desc = not is_desc

            sorted_items.sort(
                key=lambda item: (
                    item[field] if isinstance(item, dict) else getattr(item, field)
                ),
                reverse=is_desc
            )

        return sorted_items


class UnifiedPaginator:
    def __init__(
        self,
        sort_fields: List[Tuple[str, str]],
        unique_key: Optional[str] = 'id',
        auto_append_unique_key: bool = True,
        offset_threshold: int = DEFAULT_OFFSET_THRESHOLD,
        large_offset_threshold: int = DEFAULT_LARGE_OFFSET_THRESHOLD
    ):
        self.sort_fields = sort_fields
        self.unique_key = unique_key
        self.auto_append_unique_key = auto_append_unique_key

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.offset_paginator = OffsetPaginator(
                sort_fields=sort_fields,
                unique_key=unique_key,
                auto_append_unique_key=auto_append_unique_key,
                warn_on_duplicate_risk=False
            )
            self.keyset_paginator = KeysetPaginator(
                sort_fields=sort_fields,
                unique_key=unique_key,
                auto_append_unique_key=auto_append_unique_key,
                warn_on_duplicate_risk=False
            )

        self.recommender = PaginationRecommender(
            offset_threshold=offset_threshold,
            large_offset_threshold=large_offset_threshold
        )

    def paginate(
        self,
        items: List[Any],
        page_size: int,
        page: Optional[int] = None,
        offset: Optional[int] = None,
        cursor: Optional[str] = None,
        is_forward: bool = True,
        method: Optional[Union[PaginationMethod, str]] = None,
        total_items: Optional[int] = None,
        estimate_total: bool = False,
        prefer_keyset: bool = False
    ) -> PageResult:
        has_cursor = cursor is not None

        if method is None:
            recommended_method, recommendation = self.recommender.recommend(
                page=page,
                offset=offset,
                page_size=page_size,
                total_items=total_items,
                has_cursor=has_cursor,
                prefer_keyset=prefer_keyset
            )
            method = recommended_method
        else:
            if isinstance(method, str):
                method = PaginationMethod(method)
            recommendation = f"使用指定的分页方式: {method.value}"

        if method == PaginationMethod.KEYSET or has_cursor:
            result = self.keyset_paginator.paginate(
                items=items,
                page_size=page_size,
                cursor=cursor,
                is_forward=is_forward,
                page=page,
                estimate_total=estimate_total
            )
            result.page_info.recommendation = recommendation
            if total_items is not None:
                result.page_info.total_items = total_items
                result.page_info.total_pages = math.ceil(total_items / page_size) if total_items > 0 else 0
                result.page_info.estimated = False
            return result
        else:
            result = self.offset_paginator.paginate(
                items=items,
                page_size=page_size,
                page=page,
                offset=offset,
                total_items=total_items
            )
            result.page_info.recommendation = recommendation
            return result

    def build_sql_query(
        self,
        base_query: str,
        page_size: int,
        page: Optional[int] = None,
        offset: Optional[int] = None,
        cursor: Optional[str] = None,
        is_forward: bool = True,
        method: Optional[Union[PaginationMethod, str]] = None,
        count_total: bool = True,
        prefer_keyset: bool = False
    ) -> Tuple[str, Tuple[Any, ...], Optional[str], PaginationMethod, str]:
        has_cursor = cursor is not None

        if method is None:
            recommended_method, recommendation = self.recommender.recommend(
                page=page,
                offset=offset,
                page_size=page_size,
                has_cursor=has_cursor,
                prefer_keyset=prefer_keyset
            )
            method = recommended_method
        else:
            if isinstance(method, str):
                method = PaginationMethod(method)
            recommendation = f"使用指定的分页方式: {method.value}"

        if method == PaginationMethod.KEYSET or has_cursor:
            query, params, count_query = self.keyset_paginator.build_sql_query(
                base_query=base_query,
                page_size=page_size,
                cursor=cursor,
                is_forward=is_forward,
                count_total=count_total
            )
        else:
            query, params, count_query = self.offset_paginator.build_sql_query(
                base_query=base_query,
                page_size=page_size,
                page=page,
                offset=offset,
                count_total=count_total
            )

        return query, params, count_query, method, recommendation
