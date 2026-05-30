from datetime import datetime, timedelta, timezone, tzinfo
from typing import List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import calendar
import warnings


class FieldType(Enum):
    VALUES = "values"
    LAST_DAY = "last_day"
    NEAREST_WEEKDAY = "nearest_weekday"
    LAST_WEEKDAY = "last_weekday_of_month"
    NTH_WEEKDAY = "nth_weekday_of_month"


@dataclass
class FieldSpec:
    field_type: FieldType
    values: Optional[List[int]] = None
    param: Optional[int] = None
    param2: Optional[int] = None

    def has_fixed_values(self) -> bool:
        return self.field_type == FieldType.VALUES and self.values is not None

    def get_all_values(self, min_val: int, max_val: int) -> List[int]:
        if self.has_fixed_values():
            return self.values
        return list(range(min_val, max_val + 1))


class CronParser:
    FIELD_NAMES_6 = ['second', 'minute', 'hour', 'day', 'month', 'weekday']
    FIELD_NAMES_7 = ['second', 'minute', 'hour', 'day', 'month', 'weekday', 'year']
    FIELD_RANGES = {
        'second': (0, 59),
        'minute': (0, 59),
        'hour': (0, 23),
        'day': (1, 31),
        'month': (1, 12),
        'weekday': (0, 6),
        'year': (1970, 2099)
    }

    DAY_WEEKDAY_MODES = ('or', 'and')

    def __init__(self,
                 expression: str,
                 day_weekday_mode: str = 'or',
                 tz: Optional[Union[str, tzinfo]] = None):
        if day_weekday_mode not in self.DAY_WEEKDAY_MODES:
            raise ValueError(f"day_weekday_mode 必须为 {self.DAY_WEEKDAY_MODES} 之一，当前为 '{day_weekday_mode}'")

        self.expression = expression.strip()
        self.day_weekday_mode = day_weekday_mode
        self.tz = self._resolve_timezone(tz)
        self.num_fields = 6
        self.field_names = self.FIELD_NAMES_6

        parts = self.expression.split()
        if len(parts) == 7:
            self.num_fields = 7
            self.field_names = self.FIELD_NAMES_7
        elif len(parts) != 6:
            raise ValueError(f"Cron表达式必须包含6或7个字段，当前为{len(parts)}个")

        self.fields: dict = {}
        self._parse_expression(parts)
        self._validate_day_weekday()

    def _resolve_timezone(self, tz: Optional[Union[str, tzinfo]]) -> Optional[tzinfo]:
        if tz is None:
            return None
        if isinstance(tz, tzinfo):
            return tz
        if isinstance(tz, str):
            if tz.startswith('UTC') or tz.startswith('+') or tz.startswith('-'):
                try:
                    if tz == 'UTC':
                        return timezone.utc
                    sign = 1 if tz[0] == '+' else -1
                    parts = tz[1:].split(':')
                    hours = int(parts[0])
                    minutes = int(parts[1]) if len(parts) > 1 else 0
                    return timezone(timedelta(hours=sign * hours, minutes=sign * minutes))
                except Exception:
                    raise ValueError(f"无法解析时区: {tz}")
            else:
                raise ValueError(f"不支持的时区格式: {tz}，请使用 'UTC', '+08:00', '-05:00' 等格式")
        return None

    def _convert_to_tz(self, dt: datetime) -> datetime:
        if self.tz is None:
            return dt
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(self.tz).replace(tzinfo=None)

    def _convert_from_tz(self, dt: datetime) -> datetime:
        if self.tz is None:
            return dt
        dt_tz = dt.replace(tzinfo=self.tz)
        return dt_tz.astimezone(timezone.utc).replace(tzinfo=None)

    def _to_expression_tz(self, dt: datetime) -> datetime:
        if self.tz is None:
            return dt
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(self.tz).replace(tzinfo=None)

    def _from_expression_tz(self, dt: datetime) -> datetime:
        if self.tz is None:
            return dt
        dt_tz = dt.replace(tzinfo=self.tz)
        return dt_tz.astimezone(timezone.utc).replace(tzinfo=None)

    def _parse_expression(self, parts: List[str]):
        for i, field_name in enumerate(self.field_names):
            self.fields[field_name] = self._parse_field(parts[i], field_name)
        if self.num_fields == 6:
            self.fields['year'] = FieldSpec(
                FieldType.VALUES,
                values=list(range(self.FIELD_RANGES['year'][0], self.FIELD_RANGES['year'][1] + 1))
            )

    def _validate_day_weekday(self):
        day_spec = self.fields['day']
        weekday_spec = self.fields['weekday']

        if day_spec.field_type == FieldType.VALUES and day_spec.values is None:
            if weekday_spec.field_type == FieldType.VALUES and weekday_spec.values is None:
                raise ValueError("'day' 和 'weekday' 不能同时为 '?'")

        day_restricted = self._is_field_restricted('day')
        weekday_restricted = self._is_field_restricted('weekday')

        if day_restricted and weekday_restricted:
            warnings.warn(
                f"Cron表达式 '{self.expression}' 中 'day' 和 'weekday' 同时指定了具体值，"
                f"当前按 '{self.day_weekday_mode}' 关系处理。"
                f"标准Cron规范为'或'关系（任一匹配即执行），"
                f"若需'与'关系（同时匹配才执行）请设置 day_weekday_mode='and'",
                UserWarning,
                stacklevel=4
            )

    def _parse_field(self, field_str: str, field_name: str) -> FieldSpec:
        min_val, max_val = self.FIELD_RANGES[field_name]

        if field_str == '?':
            if field_name not in ('day', 'weekday'):
                raise ValueError(f"'?' 只能用于 'day' 或 'weekday' 字段，不能用于 '{field_name}'")
            return FieldSpec(FieldType.VALUES, values=None)

        if field_str == 'L':
            if field_name == 'day':
                return FieldSpec(FieldType.LAST_DAY)
            elif field_name == 'weekday':
                return FieldSpec(FieldType.LAST_WEEKDAY, param=None)
            else:
                raise ValueError(f"'L' 只能用于 'day' 或 'weekday' 字段，不能用于 '{field_name}'")

        if field_str.endswith('L') and field_name == 'weekday':
            try:
                weekday = int(field_str[:-1])
                if weekday < min_val or weekday > max_val:
                    raise ValueError(f"周几值 {weekday} 超出范围 [{min_val}, {max_val}]")
                return FieldSpec(FieldType.LAST_WEEKDAY, param=weekday)
            except ValueError:
                raise ValueError(f"无效的周字段L格式: {field_str}，应为 '1L'-'6L' 或 '0L'")

        if field_str.endswith('W') and field_name == 'day':
            try:
                day = int(field_str[:-1])
                if day < min_val or day > max_val:
                    raise ValueError(f"日期值 {day} 超出范围 [{min_val}, {max_val}]")
                return FieldSpec(FieldType.NEAREST_WEEKDAY, param=day)
            except ValueError:
                raise ValueError(f"无效的日字段W格式: {field_str}，应为 '1W'-'31W'")

        if '#' in field_str and field_name == 'weekday':
            try:
                weekday_part, nth_part = field_str.split('#', 1)
                weekday = int(weekday_part)
                nth = int(nth_part)
                if weekday < min_val or weekday > max_val:
                    raise ValueError(f"周几值 {weekday} 超出范围 [{min_val}, {max_val}]")
                if nth < 1 or nth > 5:
                    raise ValueError(f"第n个周几的值 {nth} 超出范围 [1, 5]")
                return FieldSpec(FieldType.NTH_WEEKDAY, param=weekday, param2=nth)
            except ValueError as e:
                if "超出范围" in str(e):
                    raise
                raise ValueError(f"无效的周字段#格式: {field_str}，应为 '1#3' 形式（周几#第几个）")

        if field_str == '*':
            return FieldSpec(FieldType.VALUES, values=list(range(min_val, max_val + 1)))

        result: Set[int] = set()

        for part in field_str.split(','):
            values = self._parse_part(part, min_val, max_val, field_name)
            for v in values:
                if v < min_val or v > max_val:
                    raise ValueError(f"字段 '{field_name}' 的值 {v} 超出范围 [{min_val}, {max_val}]")
                result.add(v)

        return FieldSpec(FieldType.VALUES, values=sorted(result))

    def _parse_part(self, part: str, min_val: int, max_val: int, field_name: str) -> List[int]:
        if part == 'L' or part.endswith('L') or part.endswith('W') or '#' in part:
            raise ValueError(f"无效的表达式片段: {part}")

        if '/' in part:
            range_part, step_part = part.split('/', 1)
            step = int(step_part)
            if step <= 0:
                raise ValueError(f"步长必须为正整数，当前为 {step}")

            if range_part == '*':
                start, end = min_val, max_val
            elif '-' in range_part:
                start, end = map(int, range_part.split('-', 1))
            else:
                start = int(range_part)
                end = max_val

            return list(range(start, end + 1, step))

        if '-' in part:
            start, end = map(int, part.split('-', 1))
            return list(range(start, end + 1))

        return [int(part)]

    def _to_cron_weekday(self, py_weekday: int) -> int:
        return (py_weekday + 1) % 7

    def _py_weekday_from_cron(self, cron_weekday: int) -> int:
        return (cron_weekday - 1) % 7

    def _is_leap_year(self, year: int) -> bool:
        return calendar.isleap(year)

    def _days_in_month(self, year: int, month: int) -> int:
        return calendar.monthrange(year, month)[1]

    def _last_day_of_month(self, year: int, month: int) -> int:
        return self._days_in_month(year, month)

    def _nearest_weekday(self, year: int, month: int, target_day: int) -> int:
        days_in_month = self._days_in_month(year, month)
        day = min(target_day, days_in_month)
        dt = datetime(year, month, day)
        py_wd = dt.weekday()

        if py_wd < 5:
            return day

        if py_wd == 5:
            if day > 1:
                return day - 1
            else:
                return day + 2

        if py_wd == 6:
            if day < days_in_month:
                return day + 1
            else:
                return day - 2

        return day

    def _nth_weekday_of_month(self, year: int, month: int, cron_weekday: int, nth: int) -> Optional[int]:
        py_wd = self._py_weekday_from_cron(cron_weekday)
        cal = calendar.monthcalendar(year, month)
        count = 0
        for week in cal:
            if week[py_wd] != 0:
                count += 1
                if count == nth:
                    return week[py_wd]
        return None

    def _last_weekday_of_month(self, year: int, month: int, cron_weekday: Optional[int]) -> int:
        py_wd = self._py_weekday_from_cron(cron_weekday) if cron_weekday is not None else None
        cal = calendar.monthcalendar(year, month)

        if py_wd is None:
            for week in reversed(cal):
                for day in reversed(week):
                    if day != 0:
                        return day
            return self._days_in_month(year, month)
        else:
            for week in reversed(cal):
                if week[py_wd] != 0:
                    return week[py_wd]
            return self._days_in_month(year, month)

    def _find_next_value(self, current: int, allowed: List[int], min_val: int, max_val: int) -> tuple:
        for val in allowed:
            if val >= current:
                return val, False
        return allowed[0], True

    def _is_field_restricted(self, field_name: str) -> bool:
        spec = self.fields[field_name]
        if spec.field_type != FieldType.VALUES:
            return True
        if spec.values is None:
            return False
        min_val, max_val = self.FIELD_RANGES[field_name]
        return len(spec.values) < (max_val - min_val + 1)

    def _matches_day(self, year: int, month: int, day: int) -> bool:
        day_spec = self.fields['day']
        weekday_spec = self.fields['weekday']

        dt = datetime(year, month, day)
        cron_weekday = self._to_cron_weekday(dt.weekday())

        day_restricted = self._is_field_restricted('day')
        weekday_restricted = self._is_field_restricted('weekday')

        day_match = self._day_matches(year, month, day, day_spec)
        weekday_match = self._weekday_matches(year, month, day, cron_weekday, weekday_spec)

        if day_restricted and weekday_restricted:
            if self.day_weekday_mode == 'and':
                return day_match and weekday_match
            else:
                return day_match or weekday_match
        elif day_restricted:
            return day_match
        elif weekday_restricted:
            return weekday_match
        else:
            return True

    def _day_matches(self, year: int, month: int, day: int, spec: FieldSpec) -> bool:
        if spec.field_type == FieldType.VALUES:
            if spec.values is None:
                return True
            return day in spec.values
        elif spec.field_type == FieldType.LAST_DAY:
            return day == self._last_day_of_month(year, month)
        elif spec.field_type == FieldType.NEAREST_WEEKDAY:
            target = spec.param
            nearest = self._nearest_weekday(year, month, target)
            return day == nearest
        elif spec.field_type == FieldType.LAST_WEEKDAY:
            return False
        elif spec.field_type == FieldType.NTH_WEEKDAY:
            return False
        return False

    def _weekday_matches(self, year: int, month: int, day: int, cron_weekday: int, spec: FieldSpec) -> bool:
        if spec.field_type == FieldType.VALUES:
            if spec.values is None:
                return True
            return cron_weekday in spec.values
        elif spec.field_type == FieldType.LAST_WEEKDAY:
            target = spec.param
            last_day = self._last_weekday_of_month(year, month, target)
            return day == last_day
        elif spec.field_type == FieldType.NTH_WEEKDAY:
            target_wd = spec.param
            nth = spec.param2
            expected_day = self._nth_weekday_of_month(year, month, target_wd, nth)
            return expected_day is not None and day == expected_day
        return False

    def _get_candidate_days(self, year: int, month: int) -> Set[int]:
        candidates: Set[int] = set()
        days_in_month = self._days_in_month(year, month)
        day_spec = self.fields['day']
        weekday_spec = self.fields['weekday']

        if day_spec.field_type == FieldType.VALUES:
            if day_spec.values is not None:
                for d in day_spec.values:
                    if 1 <= d <= days_in_month:
                        candidates.add(d)
            else:
                for d in range(1, days_in_month + 1):
                    candidates.add(d)
        elif day_spec.field_type == FieldType.LAST_DAY:
            candidates.add(self._last_day_of_month(year, month))
        elif day_spec.field_type == FieldType.NEAREST_WEEKDAY:
            nearest = self._nearest_weekday(year, month, day_spec.param)
            candidates.add(nearest)

        if weekday_spec.field_type == FieldType.VALUES:
            if weekday_spec.values is not None:
                for cron_wd in weekday_spec.values:
                    py_wd = self._py_weekday_from_cron(cron_wd)
                    cal = calendar.monthcalendar(year, month)
                    for week in cal:
                        d = week[py_wd]
                        if d != 0:
                            candidates.add(d)
        elif weekday_spec.field_type == FieldType.LAST_WEEKDAY:
            d = self._last_weekday_of_month(year, month, weekday_spec.param)
            candidates.add(d)
        elif weekday_spec.field_type == FieldType.NTH_WEEKDAY:
            d = self._nth_weekday_of_month(year, month, weekday_spec.param, weekday_spec.param2)
            if d is not None:
                candidates.add(d)

        if not candidates:
            candidates = set(range(1, days_in_month + 1))

        return candidates

    def get_next_execution(self, base_time: Optional[datetime] = None) -> datetime:
        if base_time is None:
            base_time = datetime.now()

        if self.tz is not None:
            base_time = self._to_expression_tz(base_time)

        current = base_time + timedelta(seconds=1)
        current = current.replace(microsecond=0)

        max_years = 5
        for _ in range(max_years * 12):
            result = self._try_build_time(current)
            if result:
                return result

            if current.month == 12:
                current = datetime(current.year + 1, 1, 1, 0, 0, 0)
            else:
                current = datetime(current.year, current.month + 1, 1, 0, 0, 0)

        raise ValueError(f"在未来{max_years}年内未找到匹配的执行时间")

    def _try_build_time(self, start: datetime) -> Optional[datetime]:
        year = start.year
        month = start.month
        day = start.day
        hour = start.hour
        minute = start.minute
        second = start.second

        year_min, year_max = self.FIELD_RANGES['year']

        for _ in range(60):
            if year > year_max or year < year_min:
                return None

            if self.fields['year'].has_fixed_values():
                allowed_years = self.fields['year'].values
                new_year, carry = self._find_next_value(year, allowed_years, year_min, year_max)
                if carry and new_year < year:
                    return None
                if new_year != year:
                    year = new_year
                    month = 1
                    day = 1
                    hour = 0
                    minute = 0
                    second = 0
                    continue

            for _ in range(12):
                if month > 12:
                    month = 1
                    year += 1
                    break

                if self.fields['month'].has_fixed_values():
                    new_month, carry = self._find_next_value(month, self.fields['month'].values, 1, 12)
                    if carry:
                        month = 1
                        year += 1
                        day = 1
                        hour = 0
                        minute = 0
                        second = 0
                        break
                    if new_month != month:
                        month = new_month
                        day = 1
                        hour = 0
                        minute = 0
                        second = 0
                        continue

                days_in_month = self._days_in_month(year, month)
                candidate_days = sorted(self._get_candidate_days(year, month))
                candidate_days = [d for d in candidate_days if 1 <= d <= days_in_month]

                if not candidate_days:
                    month += 1
                    day = 1
                    hour = 0
                    minute = 0
                    second = 0
                    continue

                for _ in range(len(candidate_days) + 2):
                    found_day = None
                    for cand_day in candidate_days:
                        if cand_day >= day and self._matches_day(year, month, cand_day):
                            found_day = cand_day
                            break

                    if found_day is None:
                        break

                    if found_day != day:
                        day = found_day
                        hour = 0
                        minute = 0
                        second = 0

                    if day > days_in_month:
                        break

                    result = self._try_build_hms(year, month, day, hour, minute, second)
                    if result is not None:
                        return result

                    day += 1
                    hour = 0
                    minute = 0
                    second = 0

                month += 1
                day = 1
                hour = 0
                minute = 0
                second = 0

        return None

    def _try_build_hms(self, year: int, month: int, day: int,
                       hour: int, minute: int, second: int) -> Optional[datetime]:
        for _ in range(24):
            if hour >= 24:
                return None

            if self.fields['hour'].has_fixed_values():
                new_hour, carry = self._find_next_value(hour, self.fields['hour'].values, 0, 23)
                if carry:
                    return None
                if new_hour != hour:
                    hour = new_hour
                    minute = 0
                    second = 0

            for _ in range(60):
                if minute >= 60:
                    hour += 1
                    minute = 0
                    second = 0
                    break

                if self.fields['minute'].has_fixed_values():
                    new_minute, carry = self._find_next_value(minute, self.fields['minute'].values, 0, 59)
                    if carry:
                        minute = 0
                        second = 0
                        hour += 1
                        break
                    if new_minute != minute:
                        minute = new_minute
                        second = 0

                for _ in range(60):
                    if second >= 60:
                        minute += 1
                        second = 0
                        break

                    if self.fields['second'].has_fixed_values():
                        new_second, carry = self._find_next_value(second, self.fields['second'].values, 0, 59)
                        if carry:
                            second = 0
                            minute += 1
                            break
                        second = new_second

                    try:
                        return datetime(year, month, day, hour, minute, second)
                    except ValueError:
                        return None

                if second >= 60:
                    minute += 1
                    second = 0
                    continue

            if minute >= 60:
                hour += 1
                minute = 0
                second = 0
                continue

        return None

    def get_next_n_executions(self, n: int, base_time: Optional[datetime] = None) -> List[datetime]:
        if n <= 0:
            raise ValueError("n 必须为正整数")

        result = []
        current = base_time or datetime.now()

        for _ in range(n):
            next_time = self.get_next_execution(current)
            result.append(next_time)
            current = next_time

        return result

    def explain(self) -> str:
        explanations = []
        field_labels = ['秒', '分', '时', '日', '月', '周']
        if self.num_fields == 7:
            field_labels.append('年')

        for i, field_name in enumerate(self.field_names):
            spec = self.fields[field_name]
            label = field_labels[i]

            if spec.field_type == FieldType.VALUES:
                if spec.values is None:
                    explanations.append(f"{label}: 不指定")
                else:
                    min_val, max_val = self.FIELD_RANGES[field_name]
                    if len(spec.values) == max_val - min_val + 1:
                        explanations.append(f"{label}: 每{label}")
                    elif len(spec.values) <= 12:
                        explanations.append(f"{label}: {', '.join(map(str, spec.values))}")
                    else:
                        explanations.append(f"{label}: {spec.values[0]}-{spec.values[-1]} 共{len(spec.values)}个值")
            elif spec.field_type == FieldType.LAST_DAY:
                explanations.append(f"{label}: 每月最后一天")
            elif spec.field_type == FieldType.NEAREST_WEEKDAY:
                explanations.append(f"{label}: 每月{spec.param}号最近的工作日")
            elif spec.field_type == FieldType.LAST_WEEKDAY:
                if spec.param is None:
                    explanations.append(f"{label}: 每月最后一天")
                else:
                    weekday_names = ['日', '一', '二', '三', '四', '五', '六']
                    explanations.append(f"{label}: 每月最后一个周{weekday_names[spec.param]}")
            elif spec.field_type == FieldType.NTH_WEEKDAY:
                weekday_names = ['日', '一', '二', '三', '四', '五', '六']
                nth_map = {1: '一', 2: '二', 3: '三', 4: '四', 5: '五'}
                explanations.append(f"{label}: 每月第{nth_map.get(spec.param2, spec.param2)}个周{weekday_names[spec.param]}")

        if self._is_field_restricted('day') and self._is_field_restricted('weekday'):
            mode_desc = '或（任一匹配即执行）' if self.day_weekday_mode == 'or' else '与（同时匹配才执行）'
            explanations.append(f"日周关系: {mode_desc}")

        if self.tz is not None:
            explanations.append(f"时区: {self.tz}")

        return '\n'.join(explanations)


def main():
    weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    base_time = datetime(2024, 1, 15, 10, 30, 45)
    print(f"基准时间: {base_time}")
    print(f"基准时间是{weekday_names[base_time.weekday()]}\n")

    print("=" * 70)
    print("扩展Cron表达式测试 (L, W, #, 年, 时区)")
    print("=" * 70)

    extended_cases = [
        ("0 0 0 L * *", "每月最后一天 00:00:00 执行", None, 10),
        ("0 0 12 15W * *", "每月15号最近的工作日 12:00 执行", None, 10),
        ("0 0 10 * * 1#3", "每月第三个周一 10:00 执行", None, 10),
        ("0 30 9 * * 5L", "每月最后一个周五 09:30 执行", None, 10),
        ("0 0 0 L * * 2024-2026", "2024-2026年每月最后一天执行", None, 10),
        ("0 0 9 1 1 * 2024,2025,2026", "2024-2026年的元旦上午9点", None, 10),
        ("0 0 12 * * *", "每天中午12点，+08:00时区", "+08:00", 10),
        ("0 0 9 15 * 1", "日=15 或 周=1（OR模式）", None, 10),
    ]

    for expr, desc, tz, n in extended_cases:
        print(f"\n表达式: {expr}")
        print(f"描述: {desc}")
        if tz:
            print(f"时区: {tz}")
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                parser = CronParser(expr, tz=tz)
                if caught:
                    for w in caught:
                        print(f"⚠ 警告: {w.message}")
            print(parser.explain())
            times = parser.get_next_n_executions(n, base_time)
            print(f"未来{n}次执行时间:")
            for i, t in enumerate(times, 1):
                tz_str = f" {tz}" if tz else ""
                print(f"  {i:2d}. {t}{tz_str} ({weekday_names[t.weekday()]}, 日={t.day})")
        except Exception as e:
            print(f"错误: {e}")
        print("-" * 70)

    print("\n" + "=" * 70)
    print("时区转换演示")
    print("=" * 70)

    expr = "0 30 9 * * *"
    base_utc = datetime(2024, 1, 15, 2, 0, 0)
    print(f"\nUTC基准时间: {base_utc}")
    print(f"表达式: {expr} (每天 09:30)")

    for tz_str in ["+00:00", "+08:00", "-05:00"]:
        parser = CronParser(expr, tz=tz_str)
        next_t = parser.get_next_execution(base_utc)
        print(f"  时区 {tz_str}: 下次执行 = {next_t}")


if __name__ == "__main__":
    main()
