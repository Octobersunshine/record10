import re
from datetime import datetime, timedelta
from typing import Optional


WEEKDAY_MAP = {
    "一": 0, "二": 1, "三": 2, "四": 3,
    "五": 4, "六": 5, "日": 6, "天": 6,
}

PERIOD_MAP = {
    "凌晨": 0, "早上": 0, "早晨": 0, "上午": 0,
    "中午": 12, "下午": 12, "傍晚": 12, "晚上": 12, "夜晚": 12, "夜里": 12,
}


def _resolve_hour(hour: int, period: Optional[str]) -> int:
    if period and period in PERIOD_MAP:
        offset = PERIOD_MAP[period]
        if offset == 12 and hour < 12:
            hour += 12
        if offset == 0 and hour == 12:
            hour = 0
    return hour


def _apply_relative_days(base: datetime, day_text: str) -> datetime:
    base_date = base.date()
    offset_map = {
        "前天": -2, "昨天": -1, "今天": 0,
        "明天": 1, "后天": 2, "大后天": 3,
    }
    offset = offset_map.get(day_text, 0)
    target_date = base_date + timedelta(days=offset)
    return datetime(target_date.year, target_date.month, target_date.day)


def _apply_weekday(base: datetime, direction: str, weekday_cn: str) -> datetime:
    target = WEEKDAY_MAP.get(weekday_cn)
    if target is None:
        return datetime(base.year, base.month, base.day)
    base_date = base.date()
    current = base_date.weekday()
    if direction == "下":
        delta = (target - current) % 7
        if delta == 0:
            delta = 7
    elif direction == "上":
        delta = -((current - target) % 7)
        if delta == 0:
            delta = -7
    else:
        delta = (target - current) % 7
    target_date = base_date + timedelta(days=delta)
    return datetime(target_date.year, target_date.month, target_date.day)


def _apply_delta(base: datetime, text: str) -> Optional[datetime]:
    m = re.search(r"(\d+)\s*([秒分钟小时天周个月年]+)[以之]?(前|后)", text)
    if not m:
        return None
    value = int(m.group(1))
    unit = m.group(2)
    direction = m.group(3)
    sign = -1 if direction == "前" else 1

    base = base.replace(microsecond=0)

    if unit.startswith("秒"):
        return base + timedelta(seconds=sign * value)
    if unit.startswith("分"):
        result = base.replace(second=0) + timedelta(minutes=sign * value)
        return result.replace(second=0)
    if unit.startswith("小时"):
        return base + timedelta(hours=sign * value)
    if unit.startswith("天") or unit.startswith("日"):
        return base + timedelta(days=sign * value)
    if unit.startswith("周") or unit.startswith("星期"):
        return base + timedelta(weeks=sign * value)
    if "月" in unit:
        total_months = base.month + sign * value
        new_year = base.year + (total_months - 1) // 12
        new_month = (total_months - 1) % 12 + 1
        try:
            return base.replace(year=new_year, month=new_month)
        except ValueError:
            return base.replace(year=new_year, month=new_month, day=28)
    if unit.startswith("年"):
        try:
            return base.replace(year=base.year + sign * value)
        except ValueError:
            return base.replace(year=base.year + sign * value, day=28)
    return None


def parse_chinese_time(text: str, base: Optional[datetime] = None) -> Optional[str]:
    if base is None:
        base = datetime.now()

    base = base.replace(microsecond=0)
    text = text.strip()

    result = _try_delta(base, text)
    if result:
        return result.replace(microsecond=0).isoformat()

    result = _try_weekday_time(base, text)
    if result:
        return result.replace(microsecond=0).isoformat()

    result = _try_relative_day_time(base, text)
    if result:
        return result.replace(microsecond=0).isoformat()

    result = _try_absolute_date_time(base, text)
    if result:
        return result.replace(microsecond=0).isoformat()

    result = _try_time_only(base, text)
    if result:
        return result.replace(microsecond=0).isoformat()

    return _try_dateparser(text)


def _try_delta(base: datetime, text: str) -> Optional[datetime]:
    return _apply_delta(base, text)


def _try_weekday_time(base: datetime, text: str) -> Optional[datetime]:
    m = re.match(
        r"(上|下)?(?:周|星期)(一|二|三|四|五|六|日|天)"
        r"(?:\s*(凌晨|早上|早晨|上午|中午|下午|傍晚|晚上|夜晚|夜里))?"
        r"(?:\s*(\d{1,2}))?"
        r"(?:\s*[点时:：])?"
        r"(?:\s*(\d{1,2}))?"
        r"(?:\s*半)?"
        r"(?:\s*分?)?",
        text,
    )
    if not m:
        return None

    direction = m.group(1) or ""
    weekday_cn = m.group(2)
    period = m.group(3)
    hour_str = m.group(4)
    minute_str = m.group(5)

    result = _apply_weekday(base, direction, weekday_cn)
    result = result.replace(hour=0, minute=0, second=0, microsecond=0)

    if hour_str:
        hour = int(hour_str)
        hour = _resolve_hour(hour, period)
        minute = int(minute_str) if minute_str else 0
        result = result.replace(hour=hour, minute=minute)

    if "半" in text and not minute_str:
        if hour_str:
            result = result.replace(minute=30)

    return result


def _try_relative_day_time(base: datetime, text: str) -> Optional[datetime]:
    m = re.match(
        r"(大后天|后天|明天|今天|昨天|前天)"
        r"(?:\s*(凌晨|早上|早晨|上午|中午|下午|傍晚|晚上|夜晚|夜里))?"
        r"(?:\s*(\d{1,2}))?"
        r"(?:\s*[点时:：])?"
        r"(?:\s*(\d{1,2}))?"
        r"(?:\s*半)?"
        r"(?:\s*分?)?",
        text,
    )
    if not m:
        return None

    day_text = m.group(1)
    period = m.group(2)
    hour_str = m.group(3)
    minute_str = m.group(4)

    result = _apply_relative_days(base, day_text)
    result = result.replace(hour=0, minute=0, second=0, microsecond=0)

    if hour_str:
        hour = int(hour_str)
        hour = _resolve_hour(hour, period)
        minute = int(minute_str) if minute_str else 0
        result = result.replace(hour=hour, minute=minute)

    if "半" in text and not minute_str:
        if hour_str:
            result = result.replace(minute=30)

    return result


def _try_absolute_date_time(base: datetime, text: str) -> Optional[datetime]:
    m = re.match(
        r"(\d{4})\s*[年/\-\.]\s*(\d{1,2})\s*[月/\-\.]\s*(\d{1,2})\s*[日号]?"
        r"(?:\s*(凌晨|早上|早晨|上午|中午|下午|傍晚|晚上|夜晚|夜里))?"
        r"(?:\s*(\d{1,2}))?"
        r"(?:\s*[点时:：])?"
        r"(?:\s*(\d{1,2}))?"
        r"(?:\s*[分:：])?"
        r"(?:\s*(\d{1,2}))?\s*[秒]?",
        text,
    )
    if not m:
        m2 = re.match(
            r"(\d{1,2})\s*[月/\-\.]\s*(\d{1,2})\s*[日号]?"
            r"(?:\s*(凌晨|早上|早晨|上午|中午|下午|傍晚|晚上|夜晚|夜里))?"
            r"(?:\s*(\d{1,2}))?"
            r"(?:\s*[点时:：])?"
            r"(?:\s*(\d{1,2}))?",
            text,
        )
        if not m2:
            return None
        month = int(m2.group(1))
        day = int(m2.group(2))
        period = m2.group(3)
        hour_str = m2.group(4)
        minute_str = m2.group(5)
        year = base.year
        result = base.replace(month=month, day=day, hour=0, minute=0, second=0, microsecond=0)
    else:
        year = int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3))
        period = m.group(4)
        hour_str = m.group(5)
        minute_str = m.group(6)
        second_str = m.group(7)
        result = datetime(year, month, day)

    if hour_str:
        hour = int(hour_str)
        hour = _resolve_hour(hour, period)
        minute = int(minute_str) if minute_str else 0
        second = int(second_str) if 'second_str' in dir() and second_str else 0
        result = result.replace(hour=hour, minute=minute, second=second)

    if "半" in text and not minute_str:
        if hour_str:
            result = result.replace(minute=30)

    return result


def _try_time_only(base: datetime, text: str) -> Optional[datetime]:
    m = re.match(
        r"(凌晨|早上|早晨|上午|中午|下午|傍晚|晚上|夜晚|夜里)?"
        r"\s*(\d{1,2})"
        r"\s*[点时:：]"
        r"(?:\s*(\d{1,2}))?"
        r"(?:\s*[分:：])?"
        r"(?:\s*(\d{1,2}))?\s*[秒]?"
        r"(?:\s*半)?",
        text,
    )
    if not m:
        return None

    period = m.group(1)
    hour_str = m.group(2)
    minute_str = m.group(3)
    second_str = m.group(4)

    if not hour_str:
        return None

    hour = int(hour_str)
    hour = _resolve_hour(hour, period)
    minute = int(minute_str) if minute_str else 0
    second = int(second_str) if second_str else 0

    result = base.replace(hour=hour, minute=minute, second=second, microsecond=0)

    if "半" in text and not minute_str:
        result = result.replace(minute=30)

    return result


def _try_dateparser(text: str) -> Optional[str]:
    try:
        import dateparser
        settings = {
            "LANGUAGES": ["zh"],
            "PREFER_DATES_FROM": "future",
            "RETURN_AS_TIMEZONE_AWARE": False,
        }
        result = dateparser.parse(text, settings=settings)
        if result:
            return result.replace(microsecond=0).isoformat()
    except ImportError:
        pass
    return None


def _parse_time_component(base: datetime, text: str) -> Optional[datetime]:
    result = _try_time_only(base, text)
    if result:
        return result
    m = re.match(
        r"(凌晨|早上|早晨|上午|中午|下午|傍晚|晚上|夜晚|夜里)?\s*(\d{1,2})",
        text.strip(),
    )
    if m:
        return _try_time_only(base, text.strip())
    return None


def parse_time_range(text: str, base: Optional[datetime] = None) -> Optional[dict]:
    if base is None:
        base = datetime.now()
    base = base.replace(microsecond=0)
    text = text.strip()

    separators = [r"\s*[~\-]\s*", r"\s*到\s*", r"\s*至\s*", r"\s*-\s*"]
    for sep in separators:
        m = re.split(sep, text)
        if len(m) == 2:
            start_text, end_text = m[0].strip(), m[1].strip()
            start_dt = _parse_time_component(base, start_text)
            end_dt = _parse_time_component(base, end_text)
            if start_dt and end_dt:
                return {
                    "type": "range",
                    "start": start_dt.replace(microsecond=0).isoformat(),
                    "end": end_dt.replace(microsecond=0).isoformat(),
                }
    return None


def _next_weekday_occurrence(base: datetime, weekday: int, hour: int, minute: int) -> datetime:
    base_date = base.date()
    current_weekday = base_date.weekday()
    delta = (weekday - current_weekday) % 7
    if delta == 0:
        candidate = datetime(base_date.year, base_date.month, base_date.day, hour, minute)
        if candidate > base:
            return candidate
        delta = 7
    target_date = base_date + timedelta(days=delta)
    return datetime(target_date.year, target_date.month, target_date.day, hour, minute)


def parse_recurring_weekly(text: str, base: Optional[datetime] = None, count: int = 5) -> Optional[dict]:
    if base is None:
        base = datetime.now()
    base = base.replace(microsecond=0)
    text = text.strip()

    m = re.match(
        r"每\s*周\s*([一二三四五六日天])\s*"
        r"(凌晨|早上|早晨|上午|中午|下午|傍晚|晚上|夜晚|夜里)?\s*"
        r"(\d{1,2})?\s*[点时]?\s*(\d{1,2})?\s*(?:半)?\s*分?",
        text,
    )
    if not m:
        return None

    weekday_cn = m.group(1)
    period = m.group(2)
    hour_str = m.group(3)
    minute_str = m.group(4)

    weekday = WEEKDAY_MAP.get(weekday_cn)
    if weekday is None:
        return None

    hour = 0
    minute = 0
    if hour_str:
        hour = int(hour_str)
        hour = _resolve_hour(hour, period)
        minute = int(minute_str) if minute_str else 0

    if "半" in text and not minute_str:
        minute = 30

    occurrences = []
    current = base
    for _ in range(count):
        next_dt = _next_weekday_occurrence(current, weekday, hour, minute)
        occurrences.append(next_dt.replace(microsecond=0).isoformat())
        current = next_dt + timedelta(days=1)

    return {
        "type": "recurring",
        "frequency": "weekly",
        "weekday": weekday_cn,
        "time": f"{hour:02d}:{minute:02d}",
        "next_occurrences": occurrences,
    }


def _last_day_of_month(year: int, month: int) -> int:
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    return (next_month - timedelta(days=1)).day


def _next_monthly_occurrence(base: datetime, day_spec: str, hour: int, minute: int) -> datetime:
    base_date = base.date()
    current_year, current_month = base_date.year, base_date.month

    for offset in range(0, 12):
        check_year = current_year + (current_month + offset - 1) // 12
        check_month = (current_month + offset - 1) % 12 + 1

        if day_spec == "最后一天":
            day = _last_day_of_month(check_year, check_month)
        else:
            try:
                day = int(day_spec)
                last_day = _last_day_of_month(check_year, check_month)
                if day > last_day:
                    continue
            except ValueError:
                day = int(day_spec)

        candidate = datetime(check_year, check_month, day, hour, minute)
        if candidate > base:
            return candidate

    return datetime(current_year + 1, current_month, 1, hour, minute)


def parse_recurring_monthly(text: str, base: Optional[datetime] = None, count: int = 5) -> Optional[dict]:
    if base is None:
        base = datetime.now()
    base = base.replace(microsecond=0)
    text = text.strip()

    m = re.match(
        r"每\s*月\s*(最后一天|\d{1,2})\s*[号日]?\s*"
        r"(凌晨|早上|早晨|上午|中午|下午|傍晚|晚上|夜晚|夜里)?\s*"
        r"(\d{1,2})?\s*[点时]?\s*(\d{1,2})?\s*(?:半)?\s*分?",
        text,
    )
    if not m:
        return None

    day_spec = m.group(1)
    period = m.group(2)
    hour_str = m.group(3)
    minute_str = m.group(4)

    hour = 0
    minute = 0
    if hour_str:
        hour = int(hour_str)
        hour = _resolve_hour(hour, period)
        minute = int(minute_str) if minute_str else 0

    if "半" in text and not minute_str:
        minute = 30

    occurrences = []
    current = base
    for _ in range(count):
        next_dt = _next_monthly_occurrence(current, day_spec, hour, minute)
        occurrences.append(next_dt.replace(microsecond=0).isoformat())
        current = next_dt + timedelta(days=1)

    return {
        "type": "recurring",
        "frequency": "monthly",
        "day": day_spec,
        "time": f"{hour:02d}:{minute:02d}",
        "next_occurrences": occurrences,
    }


def parse_time(text: str, base: Optional[datetime] = None, recurring_count: int = 5) -> Optional[dict]:
    if base is None:
        base = datetime.now()
    base = base.replace(microsecond=0)
    text = text.strip()

    weekly = parse_recurring_weekly(text, base=base, count=recurring_count)
    if weekly:
        return weekly

    monthly = parse_recurring_monthly(text, base=base, count=recurring_count)
    if monthly:
        return monthly

    time_range = parse_time_range(text, base=base)
    if time_range:
        return time_range

    single = parse_chinese_time(text, base=base)
    if single:
        return {
            "type": "single",
            "datetime": single,
        }

    return None


if __name__ == "__main__":
    test_base = datetime(2026, 5, 31, 14, 25, 37)

    single_cases = [
        ("明天下午3点", test_base),
        ("下周一上午10点半", test_base),
        ("5分钟后", test_base),
        ("5分钟前", test_base),
        ("1天前", test_base),
        ("后天", test_base),
        ("昨天", test_base),
        ("前天", test_base),
    ]

    range_cases = [
        ("上午9点到下午5点", test_base),
        ("10:00-18:00", test_base),
        ("早上8点半至晚上9点", test_base),
    ]

    weekly_cases = [
        ("每周一上午9点", test_base),
        ("每周五下午6点半", test_base),
    ]

    monthly_cases = [
        ("每月15号上午10点", test_base),
        ("每月最后一天下午5点", test_base),
    ]

    print(f"基准时间: {test_base.isoformat()}")
    print("=" * 60)

    print("【单点时间】")
    for case, base in single_cases:
        result = parse_time(case, base=base)
        print(f"  {case:<20s}  =>  {result}")
    print()

    print("【时间范围】")
    for case, base in range_cases:
        result = parse_time(case, base=base)
        print(f"  {case:<20s}  =>  {result}")
    print()

    print("【每周重复】")
    for case, base in weekly_cases:
        result = parse_time(case, base=base, recurring_count=3)
        print(f"  {case}")
        if result:
            for i, occ in enumerate(result["next_occurrences"], 1):
                print(f"    第{i}次: {occ}")
    print()

    print("【每月重复】")
    for case, base in monthly_cases:
        result = parse_time(case, base=base, recurring_count=3)
        print(f"  {case}")
        if result:
            for i, occ in enumerate(result["next_occurrences"], 1):
                print(f"    第{i}次: {occ}")
