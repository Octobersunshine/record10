from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime, formatdate
import re
import time


class TimeDelta:
    UNITS = [
        ('年', 'years', 365 * 24 * 3600),
        ('个月', 'months', 30 * 24 * 3600),
        ('周', 'weeks', 7 * 24 * 3600),
        ('天', 'days', 24 * 3600),
        ('小时', 'hours', 3600),
        ('分钟', 'minutes', 60),
        ('秒', 'seconds', 1),
    ]

    EN_UNITS = {
        'year': ('年', 365 * 24 * 3600), 'years': ('年', 365 * 24 * 3600),
        'month': ('个月', 30 * 24 * 3600), 'months': ('个月', 30 * 24 * 3600),
        'week': ('周', 7 * 24 * 3600), 'weeks': ('周', 7 * 24 * 3600),
        'day': ('天', 24 * 3600), 'days': ('天', 24 * 3600),
        'hour': ('小时', 3600), 'hours': ('小时', 3600),
        'minute': ('分钟', 60), 'minutes': ('分钟', 60),
        'second': ('秒', 1), 'seconds': ('秒', 1),
    }

    def __init__(self, delta_seconds):
        self.total_seconds = delta_seconds

    @property
    def is_future(self):
        return self.total_seconds > 0

    @property
    def is_past(self):
        return self.total_seconds < 0

    def humanize(self, lang='zh', precision=2):
        abs_seconds = abs(self.total_seconds)
        parts = []
        remaining = abs_seconds

        if lang == 'en':
            return self._humanize_en(abs_seconds, precision)

        for label, _, unit_seconds in self.UNITS:
            if remaining >= unit_seconds:
                count = int(remaining // unit_seconds)
                remaining %= unit_seconds
                parts.append(f"{count}{label}")
                if len(parts) >= precision:
                    break

        if not parts:
            return "刚刚"

        direction = "后" if self.is_future else "前"
        return "".join(parts) + direction

    def _humanize_en(self, abs_seconds, precision):
        parts = []
        remaining = abs_seconds

        for label, _, unit_seconds in self.UNITS:
            if remaining >= unit_seconds:
                count = int(remaining // unit_seconds)
                remaining %= unit_seconds
                en_name = self.UNITS[0][1] if label == '年' else \
                          self.UNITS[1][1] if label == '个月' else \
                          self.UNITS[2][1] if label == '周' else \
                          self.UNITS[3][1] if label == '天' else \
                          self.UNITS[4][1] if label == '小时' else \
                          self.UNITS[5][1] if label == '分钟' else 'seconds'
                suffix = '' if count == 1 else 's'
                name_map = {
                    'years': 'year', 'months': 'month', 'weeks': 'week',
                    'days': 'day', 'hours': 'hour', 'minutes': 'minute', 'seconds': 'second'
                }
                parts.append(f"{count} {name_map.get(en_name, en_name)}{suffix}")
                if len(parts) >= precision:
                    break

        if not parts:
            return "just now"

        direction = "from now" if self.is_future else "ago"
        return ", ".join(parts) + " " + direction

    def __repr__(self):
        sign = "+" if self.is_future else ""
        return f"TimeDelta({sign}{self.total_seconds}s) = {self.humanize()}"


class TimeConverter:
    DATE_ORDER_MDY = 'MDY'
    DATE_ORDER_DMY = 'DMY'
    DATE_ORDER_YMD = 'YMD'

    COMMON_FORMATS = [
        ('iso8601', r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', '%Y-%m-%d %H:%M:%S'),
        ('iso8601_tz', r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}', None),
        ('iso8601_z', r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}Z', None),
        ('rfc2822', r'^[A-Z][a-z]{2}, \d{1,2} [A-Z][a-z]{2} \d{4} \d{2}:\d{2}:\d{2}', None),
        ('timestamp', r'^\d{10}(\.\d+)?$', None),
        ('timestamp_ms', r'^\d{13}$', None),
        ('ymd_slash', r'^\d{4}/\d{1,2}/\d{1,2}( \d{2}:\d{2}(:\d{2})?)?$', '%Y/%m/%d %H:%M:%S'),
        ('ymd_dash', r'^\d{4}-\d{1,2}-\d{1,2}( \d{2}:\d{2}(:\d{2})?)?$', '%Y-%m-%d %H:%M:%S'),
        ('dmy_slash', r'^\d{1,2}/\d{1,2}/\d{4}( \d{2}:\d{2}(:\d{2})?)?$', '%d/%m/%Y %H:%M:%S'),
        ('mdy_slash', r'^\d{1,2}/\d{1,2}/\d{4}( \d{2}:\d{2}(:\d{2})?)?$', '%m/%d/%Y %H:%M:%S'),
        ('dmy_dash', r'^\d{1,2}-\d{1,2}-\d{4}( \d{2}:\d{2}(:\d{2})?)?$', '%d-%m-%Y %H:%M:%S'),
        ('mdy_dash', r'^\d{1,2}-\d{1,2}-\d{4}( \d{2}:\d{2}(:\d{2})?)?$', '%m-%d-%Y %H:%M:%S'),
        ('chinese_ymd', r'^\d{4}年\d{1,2}月\d{1,2}日( \d{1,2}时\d{1,2}分(\d{1,2}秒)?)?$', '%Y年%m月%d日 %H时%M分%S秒'),
    ]

    RELATIVE_ZH_PATTERN = re.compile(
        r'(\d+)\s*'
        r'(年|个月|月|周|星期|天|日|小时|时|分钟|分|秒)'
        r'(前|后|以前|以后|之前|之后)?'
    )

    RELATIVE_EN_PATTERN = re.compile(
        r'(\d+)\s*'
        r'(years?|months?|weeks?|days?|hours?|minutes?|mins?|seconds?|secs?)'
        r'\s*(ago|from\s+now|later|before)?',
        re.IGNORECASE
    )

    ZH_UNIT_TO_SECONDS = {
        '年': 365 * 24 * 3600,
        '个月': 30 * 24 * 3600,
        '月': 30 * 24 * 3600,
        '周': 7 * 24 * 3600,
        '星期': 7 * 24 * 3600,
        '天': 24 * 3600,
        '日': 24 * 3600,
        '小时': 3600,
        '时': 3600,
        '分钟': 60,
        '分': 60,
        '秒': 1,
    }

    EN_UNIT_TO_SECONDS = {
        'year': 365 * 24 * 3600, 'years': 365 * 24 * 3600,
        'month': 30 * 24 * 3600, 'months': 30 * 24 * 3600,
        'week': 7 * 24 * 3600, 'weeks': 7 * 24 * 3600,
        'day': 24 * 3600, 'days': 24 * 3600,
        'hour': 3600, 'hours': 3600,
        'minute': 60, 'minutes': 60, 'min': 60, 'mins': 60,
        'second': 1, 'seconds': 1, 'sec': 1, 'secs': 1,
    }

    def __init__(self, tz_offset_hours=0, date_order=None):
        self.tz = timezone(timedelta(hours=tz_offset_hours))
        self.date_order = date_order

    def set_timezone(self, tz_offset_hours):
        self.tz = timezone(timedelta(hours=tz_offset_hours))

    def set_date_order(self, date_order):
        self.date_order = date_order

    def time_diff(self, time1, time2=None, lang='zh', precision=2):
        if time2 is None:
            time2 = datetime.now(self.tz)
        elif isinstance(time2, str):
            time2, _ = self.parse_auto(time2)
        if isinstance(time1, str):
            time1, _ = self.parse_auto(time1)
        delta_seconds = (time1 - time2).total_seconds()
        td = TimeDelta(delta_seconds)
        return td.humanize(lang=lang, precision=precision), td

    def humanize_diff(self, dt, base=None, lang='zh', precision=2):
        if base is None:
            base = datetime.now(self.tz)
        elif isinstance(base, str):
            base, _ = self.parse_auto(base)
        if isinstance(dt, str):
            dt, _ = self.parse_auto(dt)
        delta_seconds = (dt - base).total_seconds()
        td = TimeDelta(delta_seconds)
        return td.humanize(lang=lang, precision=precision)

    def parse_relative(self, relative_str, base=None):
        if base is None:
            base = datetime.now(self.tz)
        elif isinstance(base, str):
            base, _ = self.parse_auto(base)

        zh_matches = self.RELATIVE_ZH_PATTERN.findall(relative_str)
        if zh_matches:
            return self._parse_relative_zh(relative_str, zh_matches, base)

        en_matches = self.RELATIVE_EN_PATTERN.findall(relative_str)
        if en_matches:
            return self._parse_relative_en(relative_str, en_matches, base)

        raise ValueError(f"无法解析相对时间表达式: {relative_str}")

    def _parse_relative_zh(self, relative_str, matches, base):
        total_delta = 0
        has_direction = False
        is_future = False

        for value_str, unit, direction in matches:
            value = int(value_str)
            seconds = self.ZH_UNIT_TO_SECONDS.get(unit)
            if seconds is None:
                raise ValueError(f"未知的时间单位: {unit}")
            total_delta += value * seconds
            if direction:
                has_direction = True
                is_future = direction in ('后', '以后', '之后')

        if not has_direction:
            is_future = '后' in relative_str or '以后' in relative_str or '之后' in relative_str
            if not is_future:
                is_future = False

        if is_future:
            return base + timedelta(seconds=total_delta)
        else:
            return base - timedelta(seconds=total_delta)

    def _parse_relative_en(self, relative_str, matches, base):
        total_delta = 0
        is_future = False
        has_direction = False

        for value_str, unit, direction in matches:
            value = int(value_str)
            unit_lower = unit.lower()
            seconds = self.EN_UNIT_TO_SECONDS.get(unit_lower)
            if seconds is None:
                raise ValueError(f"Unknown time unit: {unit}")
            total_delta += value * seconds
            if direction:
                has_direction = True
                dir_lower = direction.lower().strip()
                if 'from now' in dir_lower or 'later' in dir_lower:
                    is_future = True
                elif 'ago' in dir_lower or 'before' in dir_lower:
                    is_future = False

        if not has_direction:
            is_future = 'from now' in relative_str.lower() or 'later' in relative_str.lower()

        if is_future:
            return base + timedelta(seconds=total_delta)
        else:
            return base - timedelta(seconds=total_delta)

    def relative_to(self, relative_str, target_format='iso8601', custom_format=None,
                    base=None, tz_offset_hours=None):
        if tz_offset_hours is not None:
            self.set_timezone(tz_offset_hours)
        result_dt = self.parse_relative(relative_str, base)
        return self._format_datetime(result_dt, target_format, custom_format)

    def batch_convert(self, time_strings, source_format='auto', target_format='iso8601',
                      input_format=None, custom_format=None, tz_offset_hours=None,
                      date_order=None):
        if tz_offset_hours is not None:
            self.set_timezone(tz_offset_hours)
        if date_order is not None:
            self.set_date_order(date_order)

        results = []
        errors = []

        for time_str in time_strings:
            try:
                if source_format == 'auto':
                    result = self.convert(time_str, 'auto', target_format,
                                         custom_format=custom_format)
                else:
                    result = self.convert(time_str, source_format, target_format,
                                         input_format=input_format,
                                         custom_format=custom_format)
                results.append({'input': time_str, 'output': result, 'error': None})
            except Exception as e:
                results.append({'input': time_str, 'output': None, 'error': str(e)})
                errors.append(time_str)

        return results

    def batch_convert_formats(self, time_str, target_formats, source_format='auto',
                              input_format=None, custom_formats=None,
                              tz_offset_hours=None, date_order=None):
        if tz_offset_hours is not None:
            self.set_timezone(tz_offset_hours)
        if date_order is not None:
            self.set_date_order(date_order)

        if source_format == 'auto':
            dt, detected = self.parse_auto(time_str)
        elif source_format == 'timestamp':
            dt = self.timestamp_to_datetime(float(time_str), self.tz)
        elif source_format == 'iso8601':
            dt = self._parse_iso8601(time_str)
        elif source_format == 'rfc2822':
            dt = parsedate_to_datetime(time_str).astimezone(self.tz)
        elif source_format == 'custom':
            if input_format is None:
                raise ValueError("input_format must be provided for 'custom' source format")
            dt = self._parse_with_format(time_str, input_format)
        else:
            raise ValueError(f"Unsupported source format: {source_format}")

        results = {}
        for i, fmt in enumerate(target_formats):
            if fmt == 'custom' and custom_formats:
                cf = custom_formats[i] if i < len(custom_formats) else custom_formats[-1]
                results[f"custom:{cf}"] = self._format_datetime(dt, fmt, cf)
            else:
                results[fmt] = self._format_datetime(dt, fmt)

        return results

    def sniff_format(self, time_str):
        time_str = time_str.strip()
        
        for fmt_name, pattern, strftime_fmt in self.COMMON_FORMATS:
            if re.match(pattern, time_str):
                if fmt_name in ['dmy_slash', 'mdy_slash', 'dmy_dash', 'mdy_dash']:
                    return self._resolve_ambiguous_format(time_str, fmt_name)
                return fmt_name, strftime_fmt
        
        return None, None

    def _resolve_ambiguous_format(self, time_str, fmt_name):
        is_slash = '/' in time_str
        parts = time_str.split()[0].split('/' if is_slash else '-')
        first, second = int(parts[0]), int(parts[1])
        
        if self.date_order == self.DATE_ORDER_MDY:
            fmt = f'%m/%d/%Y %H:%M:%S' if is_slash else f'%m-%d-%Y %H:%M:%S'
            return 'mdy_slash' if is_slash else 'mdy_dash', fmt
        elif self.date_order == self.DATE_ORDER_DMY:
            fmt = f'%d/%m/%Y %H:%M:%S' if is_slash else f'%d-%m-%Y %H:%M:%S'
            return 'dmy_slash' if is_slash else 'dmy_dash', fmt
        
        if first > 12:
            fmt = f'%d/%m/%Y %H:%M:%S' if is_slash else f'%d-%m-%Y %H:%M:%S'
            return 'dmy_slash' if is_slash else 'dmy_dash', fmt
        elif second > 12:
            fmt = f'%m/%d/%Y %H:%M:%S' if is_slash else f'%m-%d-%Y %H:%M:%S'
            return 'mdy_slash' if is_slash else 'mdy_dash', fmt
        
        raise ValueError(
            f"日期格式存在歧义: '{time_str}'。"
            f"请使用 set_date_order() 指定日期顺序："
            f"TimeConverter.DATE_ORDER_MDY (月/日/年) 或 "
            f"TimeConverter.DATE_ORDER_DMY (日/月/年)"
        )

    def parse_auto(self, time_str):
        fmt_name, strftime_fmt = self.sniff_format(time_str)
        
        if fmt_name is None:
            raise ValueError(f"无法识别时间格式: {time_str}")
        
        if fmt_name == 'timestamp':
            return self.timestamp_to_datetime(float(time_str), self.tz), 'timestamp'
        elif fmt_name == 'timestamp_ms':
            return self.timestamp_to_datetime(float(time_str) / 1000, self.tz), 'timestamp_ms'
        elif fmt_name in ['iso8601_tz', 'iso8601_z', 'iso8601']:
            return self._parse_iso8601(time_str), 'iso8601'
        elif fmt_name == 'rfc2822':
            return parsedate_to_datetime(time_str).astimezone(self.tz), 'rfc2822'
        else:
            return self._parse_with_format(time_str, strftime_fmt), fmt_name

    def _parse_iso8601(self, time_str):
        dt = datetime.fromisoformat(time_str.replace('T', ' ').replace('Z', '+00:00'))
        return dt.astimezone(self.tz)

    def _parse_with_format(self, time_str, fmt):
        format_variants = [
            fmt,
            fmt.replace(':%S', ''),
            fmt.replace(' %H:%M:%S', ''),
            fmt.replace(' %H:%M', ''),
        ]
        for variant in format_variants:
            try:
                dt = datetime.strptime(time_str, variant)
                return dt.replace(tzinfo=self.tz)
            except ValueError:
                continue
        raise ValueError(f"无法解析时间字符串: {time_str}")

    @staticmethod
    def timestamp_to_datetime(timestamp, tz=None):
        if tz is None:
            tz = timezone.utc
        return datetime.fromtimestamp(timestamp, tz=tz)

    @staticmethod
    def datetime_to_timestamp(dt):
        return dt.timestamp()

    def from_timestamp(self, timestamp, target_format='iso8601', custom_format=None):
        dt = self.timestamp_to_datetime(timestamp, self.tz)
        return self._format_datetime(dt, target_format, custom_format)

    def from_iso8601(self, iso_str, target_format='timestamp', custom_format=None):
        dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        dt = dt.astimezone(self.tz)
        return self._format_datetime(dt, target_format, custom_format)

    def from_rfc2822(self, rfc_str, target_format='timestamp', custom_format=None):
        dt = parsedate_to_datetime(rfc_str)
        dt = dt.astimezone(self.tz)
        return self._format_datetime(dt, target_format, custom_format)

    def from_custom_format(self, time_str, input_format, target_format='timestamp', custom_format=None):
        dt = datetime.strptime(time_str, input_format)
        dt = dt.replace(tzinfo=self.tz)
        return self._format_datetime(dt, target_format, custom_format)

    def _format_datetime(self, dt, target_format, custom_format=None):
        if target_format == 'timestamp':
            return str(dt.timestamp())
        elif target_format == 'iso8601':
            return dt.isoformat()
        elif target_format == 'rfc2822':
            return formatdate(dt.timestamp(), localtime=False, usegmt=True)
        elif target_format == 'custom':
            if custom_format is None:
                raise ValueError("custom_format must be provided for 'custom' target format")
            return dt.strftime(custom_format)
        else:
            raise ValueError(f"Unsupported target format: {target_format}")

    def convert(self, time_str, source_format, target_format, 
                input_format=None, custom_format=None, tz_offset_hours=None,
                date_order=None):
        if tz_offset_hours is not None:
            self.set_timezone(tz_offset_hours)
        if date_order is not None:
            self.set_date_order(date_order)

        if source_format == 'auto':
            dt, detected_format = self.parse_auto(time_str)
            return self._format_datetime(dt, target_format, custom_format)
        elif source_format == 'timestamp':
            return self.from_timestamp(float(time_str), target_format, custom_format)
        elif source_format == 'iso8601':
            return self.from_iso8601(time_str, target_format, custom_format)
        elif source_format == 'rfc2822':
            return self.from_rfc2822(time_str, target_format, custom_format)
        elif source_format == 'custom':
            if input_format is None:
                raise ValueError("input_format must be provided for 'custom' source format")
            return self.from_custom_format(time_str, input_format, target_format, custom_format)
        else:
            raise ValueError(f"Unsupported source format: {source_format}")

    def auto_convert(self, time_str, target_format='iso8601', custom_format=None, 
                     tz_offset_hours=None, date_order=None):
        if tz_offset_hours is not None:
            self.set_timezone(tz_offset_hours)
        if date_order is not None:
            self.set_date_order(date_order)
        
        dt, detected_format = self.parse_auto(time_str)
        return self._format_datetime(dt, target_format, custom_format), detected_format


if __name__ == '__main__':
    converter = TimeConverter(tz_offset_hours=8)

    print("=" * 60)
    print("人类可读时间差测试")
    print("=" * 60)

    now = datetime.now(converter.tz)
    past = now - timedelta(days=3, hours=5, minutes=30)
    future = now + timedelta(hours=2, minutes=15)
    far_past = now - timedelta(days=400)
    far_future = now + timedelta(days=180, hours=12)

    desc, td = converter.time_diff(past, now)
    print(f"  过去时间 (3天5小时30分前): {desc}")
    desc, td = converter.time_diff(future, now)
    print(f"  未来时间 (2小时15分后):   {desc}")
    desc, td = converter.time_diff(far_past, now, precision=3)
    print(f"  远过去 (400天前):         {desc} (精度3)")
    desc, td = converter.time_diff(far_future, now, precision=3)
    print(f"  远未来 (180天12小时后):   {desc} (精度3)")
    desc, td = converter.time_diff(now, now)
    print(f"  刚刚:                     {desc}")

    print("\n英文输出:")
    desc, td = converter.time_diff(past, now, lang='en')
    print(f"  3天5小时30分前: {desc}")
    desc, td = converter.time_diff(future, now, lang='en')
    print(f"  2小时15分后:   {desc}")
    desc, td = converter.time_diff(now, now, lang='en')
    print(f"  刚刚:          {desc}")

    print("\n字符串输入:")
    desc = converter.humanize_diff("2024-01-15 10:30:00", "2024-01-20 10:30:00")
    print(f"  2024-01-15 vs 2024-01-20: {desc}")
    desc = converter.humanize_diff("2024-01-25 10:30:00", "2024-01-20 10:30:00")
    print(f"  2024-01-25 vs 2024-01-20: {desc}")

    print("\n" + "=" * 60)
    print("相对时间计算测试")
    print("=" * 60)

    base_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=converter.tz)

    zh_exprs = [
        "3天前", "2小时后", "1周前", "5分钟后",
        "1年3个月前", "2天3小时后",
    ]
    print("\n中文相对时间表达式 (基准: 2024-01-15 12:00:00):")
    for expr in zh_exprs:
        result = converter.parse_relative(expr, base=base_time)
        print(f"  {expr:<15} -> {result.strftime('%Y-%m-%d %H:%M:%S')}")

    en_exprs = [
        "3 days ago", "2 hours from now", "1 week ago",
        "5 minutes ago", "1 year 2 months ago",
    ]
    print("\n英文相对时间表达式 (基准: 2024-01-15 12:00:00):")
    for expr in en_exprs:
        result = converter.parse_relative(expr, base=base_time)
        print(f"  {expr:<25} -> {result.strftime('%Y-%m-%d %H:%M:%S')}")

    print("\nrelative_to 直接输出目标格式:")
    result = converter.relative_to("3天前", 'custom', '%Y-%m-%d %H:%M:%S', base=base_time)
    print(f"  3天前 -> {result}")
    result = converter.relative_to("2小时后", 'iso8601', base=base_time)
    print(f"  2小时后 -> {result}")

    print("\n" + "=" * 60)
    print("批量时间格式转换测试")
    print("=" * 60)

    batch_inputs = [
        "1705314600",
        "2024-01-15 10:30:00",
        "Mon, 15 Jan 2024 10:30:00 GMT",
        "2024/01/15 18:30:00",
        "invalid_time_string",
    ]

    print("\nbatch_convert (批量相同格式转换):")
    results = converter.batch_convert(batch_inputs, 'auto', 'custom',
                                      custom_format='%Y/%m/%d %H:%M:%S')
    for r in results:
        if r['error']:
            print(f"  {r['input']:<35} -> 错误: {r['error']}")
        else:
            print(f"  {r['input']:<35} -> {r['output']}")

    print("\nbatch_convert_formats (单时间 -> 多格式):")
    formats = ['timestamp', 'iso8601', 'rfc2822', 'custom', 'custom']
    custom_fmts = ['%Y年%m月%d日 %H:%M:%S', '%Y/%m/%d']
    multi_results = converter.batch_convert_formats(
        "2024-01-15 18:30:00", formats, 'auto',
        custom_formats=custom_fmts
    )
    for fmt_name, value in multi_results.items():
        print(f"  {fmt_name:<30} -> {value}")

    print("\n测试完成!")
