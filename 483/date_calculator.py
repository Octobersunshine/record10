from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Any, Union
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class DateType(Enum):
    WORKDAY = "workday"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"


CHINESE_HOLIDAYS: dict[int, set[date]] = {
    2024: {
        date(2024, 1, 1),
        date(2024, 2, 10), date(2024, 2, 11), date(2024, 2, 12),
        date(2024, 2, 13), date(2024, 2, 14), date(2024, 2, 15),
        date(2024, 2, 16), date(2024, 2, 17),
        date(2024, 4, 4), date(2024, 4, 5), date(2024, 4, 6),
        date(2024, 5, 1), date(2024, 5, 2), date(2024, 5, 3),
        date(2024, 5, 4), date(2024, 5, 5),
        date(2024, 6, 8), date(2024, 6, 9), date(2024, 6, 10),
        date(2024, 9, 15), date(2024, 9, 16), date(2024, 9, 17),
        date(2024, 10, 1), date(2024, 10, 2), date(2024, 10, 3),
        date(2024, 10, 4), date(2024, 10, 5), date(2024, 10, 6),
        date(2024, 10, 7),
    },
    2025: {
        date(2025, 1, 1),
        date(2025, 1, 28), date(2025, 1, 29), date(2025, 1, 30),
        date(2025, 1, 31), date(2025, 2, 1), date(2025, 2, 2),
        date(2025, 2, 3), date(2025, 2, 4),
        date(2025, 4, 4), date(2025, 4, 5), date(2025, 4, 6),
        date(2025, 5, 1), date(2025, 5, 2), date(2025, 5, 3),
        date(2025, 5, 4), date(2025, 5, 5),
        date(2025, 5, 31), date(2025, 6, 1), date(2025, 6, 2),
        date(2025, 10, 1), date(2025, 10, 2), date(2025, 10, 3),
        date(2025, 10, 4), date(2025, 10, 5), date(2025, 10, 6),
        date(2025, 10, 7), date(2025, 10, 8),
    },
    2026: {
        date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3),
        date(2026, 2, 17), date(2026, 2, 18), date(2026, 2, 19),
        date(2026, 2, 20), date(2026, 2, 21), date(2026, 2, 22),
        date(2026, 2, 23),
        date(2026, 4, 4), date(2026, 4, 5), date(2026, 4, 6),
        date(2026, 5, 1), date(2026, 5, 2), date(2026, 5, 3),
        date(2026, 5, 4), date(2026, 5, 5),
        date(2026, 6, 19), date(2026, 6, 20), date(2026, 6, 21),
        date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 3),
        date(2026, 10, 4), date(2026, 10, 5), date(2026, 10, 6),
        date(2026, 10, 7),
    },
}

ADJUSTED_WORKDAYS: dict[int, set[date]] = {
    2024: {
        date(2024, 2, 4), date(2024, 2, 18),
        date(2024, 4, 7), date(2024, 4, 28),
        date(2024, 5, 11), date(2024, 9, 14),
        date(2024, 9, 29), date(2024, 10, 12),
    },
    2025: {
        date(2025, 1, 26), date(2025, 2, 8),
        date(2025, 4, 27), date(2025, 9, 28),
        date(2025, 10, 11),
    },
    2026: {
        date(2026, 2, 14), date(2026, 2, 15),
        date(2026, 4, 26), date(2026, 9, 27),
        date(2026, 10, 10),
    },
}

_CN_NUM: dict[str, int] = {
    "零": 0, "〇": 0,
    "一": 1, "壹": 1,
    "二": 2, "贰": 2, "两": 2,
    "三": 3, "叁": 3,
    "四": 4, "肆": 4,
    "五": 5, "伍": 5,
    "六": 6, "陆": 6,
    "七": 7, "柒": 7,
    "八": 8, "捌": 8,
    "九": 9, "玖": 9,
    "十": 10, "拾": 10,
}

_WEEKDAY_CN: dict[str, int] = {
    "一": 0, "二": 1, "三": 2, "四": 3,
    "五": 4, "六": 5, "日": 6, "天": 6,
}


def _cn_to_int(s: str) -> int | None:
    if s.isdigit():
        return int(s)
    if s in _CN_NUM:
        return _CN_NUM[s]
    if "十" in s or "拾" in s:
        parts = re.split(r"[十拾]", s, maxsplit=1)
        tens = _CN_NUM.get(parts[0], 0) if parts[0] else 1
        ones = _CN_NUM.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return tens * 10 + ones
    return None


def _resolve_tz(tz: Union[str, ZoneInfo, None]) -> ZoneInfo:
    if tz is None:
        return ZoneInfo("Asia/Shanghai")
    if isinstance(tz, ZoneInfo):
        return tz
    if isinstance(tz, str):
        try:
            return ZoneInfo(tz)
        except ZoneInfoNotFoundError:
            raise ValueError(f"无效的 IANA 时区标识: {tz!r}")
    raise TypeError(f"不支持时区类型: {type(tz)}，请传入 IANA 时区字符串或 ZoneInfo 对象")


def _date_to_timestamp(d: date, tz: ZoneInfo) -> float:
    dt = datetime(d.year, d.month, d.day, tzinfo=tz)
    return dt.timestamp()


class DateCalculator:

    def __init__(
        self,
        timezone: Union[str, ZoneInfo, None] = None,
        holidays: dict[int, set[date]] | None = None,
        adjusted_workdays: dict[int, set[date]] | None = None,
    ):
        self._tz = _resolve_tz(timezone)
        self._holidays = holidays if holidays is not None else CHINESE_HOLIDAYS
        self._adjusted = adjusted_workdays if adjusted_workdays is not None else ADJUSTED_WORKDAYS

    @property
    def timezone(self) -> ZoneInfo:
        return self._tz

    @property
    def timezone_str(self) -> str:
        return str(self._tz)

    def _today(self) -> date:
        utc_now = datetime.now(timezone.utc)
        local_now = utc_now.astimezone(self._tz)
        return local_now.date()

    def _parse_date(self, d: Union[date, datetime, str]) -> date:
        if isinstance(d, datetime):
            if d.tzinfo is not None:
                local_dt = d.astimezone(self._tz)
                return local_dt.date()
            return d.date()
        if isinstance(d, date):
            return d
        if isinstance(d, str):
            if "T" in d or " " in d:
                try:
                    dt = datetime.fromisoformat(d)
                except ValueError:
                    dt = datetime.strptime(d, "%Y-%m-%d %H:%M:%S")
                if dt.tzinfo is not None:
                    local_dt = dt.astimezone(self._tz)
                    return local_dt.date()
                return dt.date()
            try:
                return date.fromisoformat(d)
            except ValueError:
                pass
            parsed = self._try_parse_natural(d)
            if parsed is not None:
                return parsed
            raise ValueError(f"无法解析日期: {d!r}")
        raise TypeError(f"不支持日期类型: {type(d)}，请传入 date/datetime 对象、ISO 格式字符串或自然语言表达式")

    def _get_holidays(self, year: int) -> set[date]:
        return self._holidays.get(year, set())

    def _get_adjusted(self, year: int) -> set[date]:
        return self._adjusted.get(year, set())

    def _try_parse_natural(self, expr: str) -> date | None:
        expr = expr.strip()
        today = self._today()
        m: re.Match | None

        m = re.match(r"^今天$", expr)
        if m:
            return today

        m = re.match(r"^明天$", expr)
        if m:
            return today + timedelta(days=1)

        m = re.match(r"^后天$", expr)
        if m:
            return today + timedelta(days=2)

        m = re.match(r"^大后天$", expr)
        if m:
            return today + timedelta(days=3)

        m = re.match(r"^昨天$", expr)
        if m:
            return today + timedelta(days=-1)

        m = re.match(r"^前天$", expr)
        if m:
            return today + timedelta(days=-2)

        m = re.match(r"^大前天$", expr)
        if m:
            return today + timedelta(days=-3)

        m = re.match(r"^(\d+|[一二两三四五六七八九十]+)(?:天|日)后$", expr)
        if m:
            n = _cn_to_int(m.group(1))
            if n is not None:
                return today + timedelta(days=n)

        m = re.match(r"^(\d+|[一二两三四五六七八九十]+)(?:天|日)前$", expr)
        if m:
            n = _cn_to_int(m.group(1))
            if n is not None:
                return today + timedelta(days=-n)

        m = re.match(r"^(\d+|[一二两三四五六七八九十]+)周后$", expr)
        if m:
            n = _cn_to_int(m.group(1))
            if n is not None:
                return today + timedelta(weeks=n)

        m = re.match(r"^(\d+|[一二两三四五六七八九十]+)周前$", expr)
        if m:
            n = _cn_to_int(m.group(1))
            if n is not None:
                return today + timedelta(weeks=-n)

        m = re.match(r"^(\d+|[一二两三四五六七八九十]+)个?月后$", expr)
        if m:
            n = _cn_to_int(m.group(1))
            if n is not None:
                return self._add_months(today, n)

        m = re.match(r"^(\d+|[一二两三四五六七八九十]+)个?月前$", expr)
        if m:
            n = _cn_to_int(m.group(1))
            if n is not None:
                return self._add_months(today, -n)

        m = re.match(r"^(\d+)年后$", expr)
        if m:
            n = int(m.group(1))
            return date(today.year + n, today.month, today.day)

        m = re.match(r"^(\d+)年前$", expr)
        if m:
            n = int(m.group(1))
            return date(today.year - n, today.month, today.day)

        m = re.match(r"^下?上?个?星期([一二三四五六日天])$", expr)
        if m:
            return None

        m = re.match(r"^下个?星期([一二三四五六日天])$", expr)
        if m:
            target_wd = _WEEKDAY_CN.get(m.group(1))
            if target_wd is not None:
                return self._next_weekday(today, target_wd)

        m = re.match(r"^上个?星期([一二三四五六日天])$", expr)
        if m:
            target_wd = _WEEKDAY_CN.get(m.group(1))
            if target_wd is not None:
                return self._prev_weekday(today, target_wd)

        m = re.match(r"^本星期([一二三四五六日天])$", expr)
        if m:
            target_wd = _WEEKDAY_CN.get(m.group(1))
            if target_wd is not None:
                return self._this_weekday(today, target_wd)

        m = re.match(r"^下个?周([一二三四五六日天])$", expr)
        if m:
            target_wd = _WEEKDAY_CN.get(m.group(1))
            if target_wd is not None:
                return self._next_weekday(today, target_wd)

        m = re.match(r"^上个?周([一二三四五六日天])$", expr)
        if m:
            target_wd = _WEEKDAY_CN.get(m.group(1))
            if target_wd is not None:
                return self._prev_weekday(today, target_wd)

        m = re.match(r"^本周([一二三四五六日天])$", expr)
        if m:
            target_wd = _WEEKDAY_CN.get(m.group(1))
            if target_wd is not None:
                return self._this_weekday(today, target_wd)

        m = re.match(r"^下个?月(\d+|[一二三四五六七八九十]+)[号日]?$", expr)
        if m:
            day = _cn_to_int(m.group(1))
            if day is not None:
                nm = self._add_months(today, 1)
                return self._safe_date(nm.year, nm.month, day)

        m = re.match(r"^上个?月(\d+|[一二三四五六七八九十]+)[号日]?$", expr)
        if m:
            day = _cn_to_int(m.group(1))
            if day is not None:
                pm = self._add_months(today, -1)
                return self._safe_date(pm.year, pm.month, day)

        m = re.match(r"^本月(\d+|[一二三四五六七八九十]+)[号日]?$", expr)
        if m:
            day = _cn_to_int(m.group(1))
            if day is not None:
                return self._safe_date(today.year, today.month, day)

        m = re.match(r"^(\d{4})年(\d+|[一二三四五六七八九十]+)月(\d+|[一二三四五六七八九十]+)[号日]?$", expr)
        if m:
            year = int(m.group(1))
            month = _cn_to_int(m.group(2))
            day = _cn_to_int(m.group(3))
            if month is not None and day is not None:
                return self._safe_date(year, month, day)

        m = re.match(r"^明年(\d+|[一二两三四五六七八九十]+)月(\d+|[一二三四五六七八九十]+)[号日]?$", expr)
        if m:
            month = _cn_to_int(m.group(1))
            day = _cn_to_int(m.group(2))
            if month is not None and day is not None:
                return self._safe_date(today.year + 1, month, day)

        m = re.match(r"^去年(\d+|[一二三四五六七八九十]+)月(\d+|[一二三四五六七八九十]+)[号日]?$", expr)
        if m:
            month = _cn_to_int(m.group(1))
            day = _cn_to_int(m.group(2))
            if month is not None and day is not None:
                return self._safe_date(today.year - 1, month, day)

        m = re.match(r"^(\d+|[一二三四五六七八九十]+)个?工作日后$", expr)
        if m:
            n = _cn_to_int(m.group(1))
            if n is not None:
                current = today
                for _ in range(n):
                    current += timedelta(days=1)
                    while not self.is_workday(current)["is_workday"]:
                        current += timedelta(days=1)
                return current

        m = re.match(r"^(\d+|[一二三四五六七八九十]+)个?工作日前$", expr)
        if m:
            n = _cn_to_int(m.group(1))
            if n is not None:
                current = today
                for _ in range(n):
                    current -= timedelta(days=1)
                    while not self.is_workday(current)["is_workday"]:
                        current -= timedelta(days=1)
                return current

        return None

    @staticmethod
    def _add_months(d: date, months: int) -> date:
        month = d.month - 1 + months
        year = d.year + month // 12
        month = month % 12 + 1
        return DateCalculator._safe_date(year, month, d.day)

    @staticmethod
    def _safe_date(year: int, month: int, day: int) -> date:
        import calendar
        max_day = calendar.monthrange(year, month)[1]
        day = min(day, max_day)
        return date(year, month, day)

    @staticmethod
    def _next_weekday(today: date, target: int) -> date:
        current_wd = today.weekday()
        delta = (target - current_wd) % 7
        if delta == 0:
            delta = 7
        return today + timedelta(days=delta)

    @staticmethod
    def _prev_weekday(today: date, target: int) -> date:
        current_wd = today.weekday()
        delta = (current_wd - target) % 7
        if delta == 0:
            delta = 7
        return today - timedelta(days=delta)

    @staticmethod
    def _this_weekday(today: date, target: int) -> date:
        current_wd = today.weekday()
        delta = target - current_wd
        return today + timedelta(days=delta)

    def parse_natural_date(self, expr: str, base: Union[date, datetime, str, None] = None) -> dict:
        expr = expr.strip()
        if base is not None:
            saved_today = self._today
            base_date = self._parse_date(base)
            self._today = lambda: base_date

        try:
            parsed = self._try_parse_natural(expr)
            if parsed is None:
                raise ValueError(f"无法识别的自然语言表达式: {expr!r}")
        finally:
            if base is not None:
                self._today = saved_today

        ts = _date_to_timestamp(parsed, self._tz)
        return {
            "expression": expr,
            "date": parsed.isoformat(),
            "timestamp": ts,
            "weekday": parsed.weekday(),
            "weekday_name": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][parsed.weekday()],
            "timezone": self.timezone_str,
        }

    def batch_calculate(self, operations: list[dict]) -> list[dict]:
        results: list[dict] = []
        for i, op in enumerate(operations):
            try:
                result = self._execute_operation(op)
                result["index"] = i
                result["status"] = "success"
            except Exception as e:
                result = {"index": i, "status": "error", "error": str(e), "operation": op}
            results.append(result)
        return results

    def _execute_operation(self, op: dict) -> dict:
        method = op.get("method")
        if not method:
            raise ValueError("缺少 method 字段")
        params = op.get("params", {})
        dispatch = {
            "date_diff": self.date_diff,
            "add_days": self.add_days,
            "add_workdays": self.add_workdays,
            "is_holiday": self.is_holiday,
            "is_weekend": self.is_weekend,
            "is_workday": self.is_workday,
            "date_type": self.date_type,
            "workdays_between": self.workdays_between,
            "now": self.now,
            "parse_natural_date": self.parse_natural_date,
        }
        fn = dispatch.get(method)
        if fn is None:
            raise ValueError(f"未知方法: {method!r}")
        return fn(**params)

    def now(self) -> dict:
        utc_now = datetime.now(timezone.utc)
        local_now = utc_now.astimezone(self._tz)
        today = local_now.date()
        return {
            "utc_time": utc_now.isoformat(),
            "local_time": local_now.isoformat(),
            "local_date": today.isoformat(),
            "timezone": self.timezone_str,
            "utc_offset": local_now.strftime("%z"),
        }

    def date_diff(
        self,
        start: Union[date, datetime, str],
        end: Union[date, datetime, str],
    ) -> dict:
        start = self._parse_date(start)
        end = self._parse_date(end)
        delta = end - start
        total_days = delta.days
        abs_days = abs(total_days)

        weeks = abs_days // 7
        remaining = abs_days % 7

        return {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "total_days": total_days,
            "abs_days": abs_days,
            "weeks": weeks,
            "remaining_days": remaining,
            "direction": "正向" if total_days >= 0 else "反向",
            "timezone": self.timezone_str,
        }

    def add_days(
        self,
        base: Union[date, datetime, str],
        days: int,
    ) -> dict:
        base = self._parse_date(base)
        target = base + timedelta(days=days)
        ts = _date_to_timestamp(target, self._tz)
        return {
            "base": base.isoformat(),
            "days": days,
            "result": target.isoformat(),
            "timestamp": ts,
            "timezone": self.timezone_str,
        }

    def add_workdays(
        self,
        base: Union[date, datetime, str],
        days: int,
    ) -> dict:
        base = self._parse_date(base)
        current = base
        step = 1 if days >= 0 else -1
        remaining = abs(days)
        while remaining > 0:
            current += timedelta(days=step)
            if self.is_workday(current)["is_workday"]:
                remaining -= 1
        ts = _date_to_timestamp(current, self._tz)
        return {
            "base": base.isoformat(),
            "workdays": days,
            "result": current.isoformat(),
            "timestamp": ts,
            "timezone": self.timezone_str,
        }

    def is_holiday(self, d: Union[date, datetime, str]) -> dict:
        d = self._parse_date(d)
        holidays = self._get_holidays(d.year)
        is_hol = d in holidays
        name = ""
        if is_hol:
            name = self._get_holiday_name(d)
        return {
            "date": d.isoformat(),
            "is_holiday": is_hol,
            "holiday_name": name if name else None,
            "timezone": self.timezone_str,
        }

    def is_weekend(self, d: Union[date, datetime, str]) -> dict:
        d = self._parse_date(d)
        is_we = d.weekday() >= 5
        return {
            "date": d.isoformat(),
            "weekday": d.weekday(),
            "weekday_name": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][d.weekday()],
            "is_weekend": is_we,
            "timezone": self.timezone_str,
        }

    def is_workday(self, d: Union[date, datetime, str]) -> dict:
        d = self._parse_date(d)
        adjusted = self._get_adjusted(d.year)
        holidays = self._get_holidays(d.year)

        if d in adjusted:
            is_wd = True
            detail = "调休工作日"
        elif d in holidays:
            is_wd = False
            detail = "法定节假日"
        elif d.weekday() >= 5:
            is_wd = False
            detail = "周末"
        else:
            is_wd = True
            detail = "常规工作日"

        return {
            "date": d.isoformat(),
            "weekday": d.weekday(),
            "weekday_name": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][d.weekday()],
            "is_workday": is_wd,
            "detail": detail,
            "timezone": self.timezone_str,
        }

    def date_type(self, d: Union[date, datetime, str]) -> dict:
        d = self._parse_date(d)
        adjusted = self._get_adjusted(d.year)
        holidays = self._get_holidays(d.year)

        if d in adjusted:
            dtype = DateType.WORKDAY
            detail = "调休工作日"
        elif d in holidays:
            dtype = DateType.HOLIDAY
            detail = "法定节假日"
        elif d.weekday() >= 5:
            dtype = DateType.WEEKEND
            detail = "周末"
        else:
            dtype = DateType.WORKDAY
            detail = "常规工作日"

        return {
            "date": d.isoformat(),
            "type": dtype.value,
            "detail": detail,
            "timezone": self.timezone_str,
        }

    def workdays_between(
        self,
        start: Union[date, datetime, str],
        end: Union[date, datetime, str],
    ) -> dict:
        start = self._parse_date(start)
        end = self._parse_date(end)
        if start > end:
            start, end = end, start
            swapped = True
        else:
            swapped = False

        count = 0
        current = start
        while current <= end:
            if self.is_workday(current)["is_workday"]:
                count += 1
            current += timedelta(days=1)

        return {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "workdays": count,
            "timezone": self.timezone_str,
        }

    @staticmethod
    def _get_holiday_name(d: date) -> str:
        m, day = d.month, d.day
        if m == 1 and day == 1:
            return "元旦"
        if m == 1 and 1 <= day <= 3:
            return "元旦假期"
        if m == 2 and 10 <= day <= 25:
            return "春节假期"
        if m == 1 and 28 <= day <= 31:
            return "春节假期"
        if m == 4 and 4 <= day <= 6:
            return "清明节假期"
        if m == 5 and 1 <= day <= 5:
            return "劳动节假期"
        if m == 6 and 8 <= day <= 10:
            return "端午节假期"
        if m == 6 and 19 <= day <= 21:
            return "端午节假期"
        if m == 9 and 15 <= day <= 17:
            return "中秋节假期"
        if m == 10 and 1 <= day <= 8:
            return "国庆节假期"
        return "法定节假日"


def main():
    calc = DateCalculator(timezone="Asia/Shanghai")

    print("=" * 60)
    print("日期计算 API 演示（含自然语言解析 + 批量计算）")
    print("=" * 60)

    today_info = calc.now()
    print(f"\n今天: {today_info['local_date']} ({calc.parse_natural_date('今天')['weekday_name']})")

    print("\n🗣️ 1. 自然语言日期解析")
    expressions = [
        "今天", "明天", "后天", "大后天",
        "昨天", "前天", "大前天",
        "3天后", "10天前",
        "2周后", "一周前",
        "3个月后", "一个月前",
        "下周一", "上周五", "本周三",
        "下个月15号", "上个月28号", "本月10号",
        "明年3月15号", "去年6月1日",
        "5个工作日后", "3个工作日前",
    ]
    for expr in expressions:
        try:
            r = calc.parse_natural_date(expr)
            print(f"   {expr:12s} → {r['date']} ({r['weekday_name']}) 时间戳={r['timestamp']}")
        except ValueError as e:
            print(f"   {expr:12s} → 解析失败: {e}")

    print("\n🗣️ 2. 自然语言日期解析（指定基准日期）")
    r = calc.parse_natural_date("下周一", base="2024-10-01")
    print(f"   基准 2024-10-01 的 下周一 → {r['date']} ({r['weekday_name']})")
    r = calc.parse_natural_date("3天后", base="2025-01-28")
    print(f"   基准 2025-01-28 的 3天后 → {r['date']} ({r['weekday_name']})")

    print("\n🗣️ 3. 自然语言作为其他 API 的输入")
    r = calc.is_workday("下周一")
    print(f"   下周一: {r['weekday_name']}, {r['detail']}, 工作日={r['is_workday']}")
    r = calc.date_diff("今天", "下个月15号")
    print(f"   今天 → 下个月15号: {r['total_days']}天")
    r = calc.add_days("下周一", 5)
    print(f"   下周一 + 5天 = {r['result']}")

    print("\n📦 4. 批量计算")
    batch_ops = [
        {"method": "parse_natural_date", "params": {"expr": "下周一"}},
        {"method": "parse_natural_date", "params": {"expr": "3天后"}},
        {"method": "parse_natural_date", "params": {"expr": "下个月15号"}},
        {"method": "is_workday", "params": {"d": "2024-10-01"}},
        {"method": "is_workday", "params": {"d": "2024-02-04"}},
        {"method": "date_diff", "params": {"start": "今天", "end": "下个月15号"}},
        {"method": "add_workdays", "params": {"base": "今天", "days": 5}},
        {"method": "workdays_between", "params": {"start": "2024-09-30", "end": "2024-10-07"}},
        {"method": "now", "params": {}},
        {"method": "is_holiday", "params": {"d": "2024-10-01"}},
    ]
    results = calc.batch_calculate(batch_ops)
    for r in results:
        status = r["status"]
        idx = r["index"]
        if status == "success":
            method = batch_ops[idx]["method"]
            data = {k: v for k, v in r.items() if k not in ("index", "status")}
            print(f"   [{idx}] {method}: {data}")
        else:
            print(f"   [{idx}] 错误: {r['error']}")

    print("\n📅 5. 原有功能验证")
    r = calc.date_diff("2024-01-01", "2024-12-31")
    print(f"   日期差: 2024-01-01 → 2024-12-31 = {r['total_days']}天")

    r = calc.add_workdays("2024-09-30", 5)
    print(f"   工作日加: 2024-09-30 + 5个工作日 = {r['result']}")

    print("\n" + "=" * 60)
    print("演示完成")


if __name__ == "__main__":
    main()
