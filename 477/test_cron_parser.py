import unittest
import warnings
from datetime import datetime
from cron_parser import CronParser


class TestCronParser(unittest.TestCase):

    def test_every_second(self):
        parser = CronParser("* * * * * *")
        base = datetime(2024, 1, 15, 10, 30, 45)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time, datetime(2024, 1, 15, 10, 30, 46))

    def test_every_minute(self):
        parser = CronParser("0 * * * * *")
        base = datetime(2024, 1, 15, 10, 30, 45)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time, datetime(2024, 1, 15, 10, 31, 0))

    def test_every_hour(self):
        parser = CronParser("0 0 * * * *")
        base = datetime(2024, 1, 15, 10, 30, 45)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time, datetime(2024, 1, 15, 11, 0, 0))

    def test_daily_at_noon(self):
        parser = CronParser("0 0 12 * * *")
        base = datetime(2024, 1, 15, 10, 30, 45)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time, datetime(2024, 1, 15, 12, 0, 0))

    def test_workday_morning(self):
        parser = CronParser("0 30 9 * * 1-5")
        base = datetime(2024, 1, 15, 10, 30, 45)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time, datetime(2024, 1, 16, 9, 30, 0))
        self.assertEqual(next_time.weekday(), 1)

    def test_workday_morning_before_time(self):
        parser = CronParser("0 30 9 * * 1-5")
        base = datetime(2024, 1, 15, 8, 0, 0)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time, datetime(2024, 1, 15, 9, 30, 0))
        self.assertEqual(next_time.weekday(), 0)

    def test_every_5_minutes(self):
        parser = CronParser("0 */5 * * * *")
        base = datetime(2024, 1, 15, 10, 30, 45)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time, datetime(2024, 1, 15, 10, 35, 0))

    def test_monthly_first_day(self):
        parser = CronParser("0 0 9 1 * *")
        base = datetime(2024, 1, 15, 10, 30, 45)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time, datetime(2024, 2, 1, 9, 0, 0))

    def test_weekly_monday_question_mark(self):
        parser = CronParser("0 0 9 ? * 1")
        base = datetime(2024, 1, 15, 10, 30, 45)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time.weekday(), 0)
        self.assertEqual(next_time.day, 22)

    def test_yearly_jan_1st(self):
        parser = CronParser("0 0 9 1 1 *")
        base = datetime(2024, 1, 15, 10, 30, 45)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time, datetime(2025, 1, 1, 9, 0, 0))

    def test_weekend_afternoon(self):
        parser = CronParser("30 45 14 * * 0,6")
        base = datetime(2024, 1, 15, 10, 30, 45)
        next_time = parser.get_next_execution(base)
        self.assertIn(next_time.weekday(), [5, 6])
        self.assertEqual(next_time.hour, 14)
        self.assertEqual(next_time.minute, 45)
        self.assertEqual(next_time.second, 30)

    def test_range_with_step(self):
        parser = CronParser("0 0-30/10 9 * * *")
        base = datetime(2024, 1, 15, 10, 30, 45)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time, datetime(2024, 1, 16, 9, 0, 0))

        next_2 = parser.get_next_execution(next_time)
        self.assertEqual(next_2, datetime(2024, 1, 16, 9, 10, 0))

        next_3 = parser.get_next_execution(next_2)
        self.assertEqual(next_3, datetime(2024, 1, 16, 9, 20, 0))

    def test_value_list(self):
        parser = CronParser("0 0,15,30,45 * * * *")
        base = datetime(2024, 1, 15, 10, 30, 45)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time, datetime(2024, 1, 15, 10, 45, 0))

    def test_question_mark_invalid_field(self):
        with self.assertRaises(ValueError) as ctx:
            CronParser("? * * * * *")
        self.assertIn("只能用于", str(ctx.exception))

    def test_both_day_and_weekday_question(self):
        with self.assertRaises(ValueError) as ctx:
            CronParser("* * * ? * ?")
        self.assertIn("不能同时为", str(ctx.exception))

    def test_invalid_field_count(self):
        with self.assertRaises(ValueError) as ctx:
            CronParser("* * * * *")
        self.assertIn("6或7个字段", str(ctx.exception))

    def test_value_out_of_range(self):
        with self.assertRaises(ValueError) as ctx:
            CronParser("60 * * * * *")
        self.assertIn("超出范围", str(ctx.exception))

    def test_negative_step(self):
        with self.assertRaises(ValueError) as ctx:
            CronParser("0 */-5 * * * *")
        self.assertIn("步长必须为正整数", str(ctx.exception))

    def test_get_next_n_executions(self):
        parser = CronParser("0 0 * * * *")
        base = datetime(2024, 1, 15, 10, 30, 45)
        times = parser.get_next_n_executions(5, base)
        self.assertEqual(len(times), 5)
        for i, t in enumerate(times):
            self.assertEqual(t.minute, 0)
            self.assertEqual(t.second, 0)
            self.assertEqual(t.hour, 11 + i)

    def test_cross_year(self):
        parser = CronParser("0 0 0 31 12 *")
        base = datetime(2024, 1, 15, 10, 30, 45)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time, datetime(2024, 12, 31, 0, 0, 0))

    def test_leap_year_feb_29(self):
        parser = CronParser("0 0 12 29 2 *")
        base = datetime(2024, 1, 15, 10, 30, 45)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time, datetime(2024, 2, 29, 12, 0, 0))

        base2 = datetime(2024, 3, 1, 0, 0, 0)
        next_time2 = parser.get_next_execution(base2)
        self.assertEqual(next_time2, datetime(2028, 2, 29, 12, 0, 0))

    def test_day_and_weekday_both_specified(self):
        parser = CronParser("0 0 12 15 * 1")
        base = datetime(2024, 1, 14, 10, 0, 0)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time.day, 15)

    def test_weekday_conversion(self):
        parser = CronParser("* * * * * 0")
        base = datetime(2024, 1, 13, 10, 0, 0)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time.weekday(), 6)

    def test_weekday_saturday(self):
        parser = CronParser("* * * * * 6")
        base = datetime(2024, 1, 13, 10, 0, 0)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time.weekday(), 5)

    def test_complex_expression(self):
        parser = CronParser("30 15,45 9-17/2 * * 1,3,5")
        base = datetime(2024, 1, 15, 10, 0, 0)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time.hour, 11)
        self.assertEqual(next_time.minute, 15)
        self.assertEqual(next_time.second, 30)
        self.assertIn(next_time.weekday(), [0, 2, 4])

    def test_explain(self):
        parser = CronParser("0 30 9 * * 1-5")
        explanation = parser.explain()
        self.assertIn("秒: 0", explanation)
        self.assertIn("分: 30", explanation)
        self.assertIn("时: 9", explanation)
        self.assertIn("周: 1, 2, 3, 4, 5", explanation)

    def test_explain_with_day_weekday_conflict(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            parser = CronParser("0 0 12 15 * 1", day_weekday_mode='or')
        explanation = parser.explain()
        self.assertIn("日周关系", explanation)
        self.assertIn("或", explanation)

    def test_explain_with_day_weekday_and_mode(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            parser = CronParser("0 0 12 15 * 1", day_weekday_mode='and')
        explanation = parser.explain()
        self.assertIn("日周关系", explanation)
        self.assertIn("与", explanation)


class TestDayWeekdayConflict(unittest.TestCase):

    def test_or_mode_matches_either(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            parser = CronParser("0 0 12 15 * 1", day_weekday_mode='or')

        base = datetime(2024, 1, 14, 10, 0, 0)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time.day, 15)
        self.assertEqual(next_time.month, 1)

    def test_and_mode_requires_both(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            parser = CronParser("0 0 12 15 * 1", day_weekday_mode='and')

        base = datetime(2024, 1, 14, 10, 0, 0)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time.day, 15)
        self.assertEqual(next_time.weekday(), 0)
        self.assertEqual(next_time.month, 1)
        self.assertEqual(next_time.year, 2024)

    def test_and_mode_no_match_in_month_skips(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            parser = CronParser("0 0 12 31 * 1", day_weekday_mode='and')

        base = datetime(2024, 2, 1, 0, 0, 0)
        next_time = parser.get_next_execution(base)
        dt = next_time
        self.assertEqual(dt.day, 31)
        self.assertEqual(dt.weekday(), 0)

    def test_or_mode_default(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            parser = CronParser("0 0 12 15 * 1")

        self.assertEqual(parser.day_weekday_mode, 'or')

        base = datetime(2024, 1, 14, 10, 0, 0)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time.day, 15)

    def test_invalid_mode(self):
        with self.assertRaises(ValueError) as ctx:
            CronParser("* * * * * *", day_weekday_mode='xor')
        self.assertIn("day_weekday_mode", str(ctx.exception))

    def test_warning_on_both_specified(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            CronParser("0 0 12 15 * 1")

        self.assertTrue(len(caught) >= 1)
        self.assertTrue(issubclass(caught[0].category, UserWarning))
        self.assertIn("同时指定", str(caught[0].message))
        self.assertIn("or", str(caught[0].message))

    def test_warning_mentions_and_mode(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            CronParser("0 0 12 15 * 1")

        self.assertTrue(len(caught) >= 1)
        self.assertIn("and", str(caught[0].message))

    def test_no_warning_when_day_is_wildcard(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            CronParser("0 0 12 * * 1")

        conflict_warnings = [w for w in caught if "同时指定" in str(w.message)]
        self.assertEqual(len(conflict_warnings), 0)

    def test_no_warning_when_weekday_is_wildcard(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            CronParser("0 0 12 15 * *")

        conflict_warnings = [w for w in caught if "同时指定" in str(w.message)]
        self.assertEqual(len(conflict_warnings), 0)

    def test_no_warning_when_question_mark_used(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            CronParser("0 0 12 ? * 1")

        conflict_warnings = [w for w in caught if "同时指定" in str(w.message)]
        self.assertEqual(len(conflict_warnings), 0)

    def test_or_vs_and_produces_different_results(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            parser_or = CronParser("0 0 12 15 * 3", day_weekday_mode='or')
            parser_and = CronParser("0 0 12 15 * 3", day_weekday_mode='and')

        base = datetime(2024, 1, 14, 10, 0, 0)
        next_or = parser_or.get_next_execution(base)
        next_and = parser_and.get_next_execution(base)

        self.assertNotEqual(next_or, next_and)
        self.assertLess(next_or, next_and)

    def test_or_mode_includes_non_15th_mondays(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            parser = CronParser("0 0 12 15 * 1", day_weekday_mode='or')

        base = datetime(2024, 1, 14, 10, 0, 0)
        times = parser.get_next_n_executions(5, base)
        has_non_15th = any(t.day != 15 for t in times)
        has_non_monday_15th = any(t.day == 15 and t.weekday() != 0 for t in times)
        self.assertTrue(has_non_15th or has_non_monday_15th)

    def test_and_mode_only_matches_15th_mondays(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            parser = CronParser("0 0 12 15 * 1", day_weekday_mode='and')

        base = datetime(2024, 1, 14, 10, 0, 0)
        times = parser.get_next_n_executions(5, base)
        for t in times:
            self.assertEqual(t.day, 15)
            self.assertEqual(t.weekday(), 0)


class TestCronExtensions(unittest.TestCase):

    def test_last_day_of_month_L(self):
        parser = CronParser("0 0 0 L * *")
        base = datetime(2024, 1, 15, 10, 0, 0)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time, datetime(2024, 1, 31, 0, 0, 0))

    def test_last_day_february_leap(self):
        parser = CronParser("0 0 0 L * *")
        base = datetime(2024, 2, 1, 0, 0, 0)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time, datetime(2024, 2, 29, 0, 0, 0))

    def test_last_day_february_non_leap(self):
        parser = CronParser("0 0 0 L * *")
        base = datetime(2023, 2, 1, 0, 0, 0)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time, datetime(2023, 2, 28, 0, 0, 0))

    def test_nearest_weekday_weekday(self):
        parser = CronParser("0 0 12 15W * *")
        base = datetime(2024, 1, 1, 0, 0, 0)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time.day, 15)
        self.assertEqual(next_time.month, 1)
        self.assertLess(next_time.weekday(), 5)

    def test_nearest_weekday_saturday(self):
        parser = CronParser("0 0 12 1W * *")
        base = datetime(2025, 2, 1, 0, 0, 0)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time.day, 3)
        self.assertEqual(next_time.weekday(), 0)

    def test_nearest_weekday_sunday(self):
        parser = CronParser("0 0 12 16W * *")
        base = datetime(2024, 2, 1, 0, 0, 0)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time.day, 16)
        self.assertEqual(next_time.weekday(), 4)

    def test_nth_weekday_third_monday(self):
        parser = CronParser("0 0 10 * * 1#3")
        base = datetime(2024, 1, 1, 0, 0, 0)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time.day, 15)
        self.assertEqual(next_time.month, 1)
        self.assertEqual(next_time.weekday(), 0)

    def test_nth_weekday_fifth_friday(self):
        parser = CronParser("0 0 0 * * 5#5")
        base = datetime(2024, 1, 1, 0, 0, 0)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time.day, 29)
        self.assertEqual(next_time.month, 3)
        self.assertEqual(next_time.weekday(), 4)

    def test_last_weekday_L_friday(self):
        parser = CronParser("0 30 9 * * 5L")
        base = datetime(2024, 1, 1, 0, 0, 0)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time.day, 26)
        self.assertEqual(next_time.month, 1)
        self.assertEqual(next_time.weekday(), 4)

    def test_last_weekday_L_sunday(self):
        parser = CronParser("0 0 0 * * 0L")
        base = datetime(2024, 1, 1, 0, 0, 0)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time.day, 28)
        self.assertEqual(next_time.month, 1)
        self.assertEqual(next_time.weekday(), 6)

    def test_seven_fields_with_year(self):
        parser = CronParser("0 0 9 1 1 * 2024-2026")
        base = datetime(2024, 1, 15, 0, 0, 0)
        times = parser.get_next_n_executions(2, base)
        self.assertEqual(times[0], datetime(2025, 1, 1, 9, 0, 0))
        self.assertEqual(times[1], datetime(2026, 1, 1, 9, 0, 0))
        with self.assertRaises(ValueError):
            parser.get_next_execution(times[1])

    def test_seven_fields_year_list(self):
        parser = CronParser("0 0 9 1 1 * 2024,2026,2028")
        base = datetime(2024, 1, 15, 0, 0, 0)
        times = parser.get_next_n_executions(2, base)
        self.assertEqual(times[0].year, 2026)
        self.assertEqual(times[1].year, 2028)

    def test_six_fields_backward_compatible(self):
        parser6 = CronParser("0 30 9 * * 1-5")
        parser7 = CronParser("0 30 9 * * 1-5 *")
        base = datetime(2024, 1, 15, 10, 0, 0)
        t6 = parser6.get_next_execution(base)
        t7 = parser7.get_next_execution(base)
        self.assertEqual(t6, t7)

    def test_invalid_field_count_8(self):
        with self.assertRaises(ValueError) as ctx:
            CronParser("* * * * * * * *")
        self.assertIn("6或7个字段", str(ctx.exception))

    def test_timezone_positive_offset(self):
        parser = CronParser("0 30 9 * * *", tz="+08:00")
        base_utc = datetime(2024, 1, 15, 1, 0, 0)
        next_time = parser.get_next_execution(base_utc)
        self.assertEqual(next_time.hour, 9)
        self.assertEqual(next_time.minute, 30)

    def test_timezone_negative_offset(self):
        parser = CronParser("0 30 9 * * *", tz="-05:00")
        base_utc = datetime(2024, 1, 15, 14, 0, 0)
        next_time = parser.get_next_execution(base_utc)
        self.assertEqual(next_time.hour, 9)
        self.assertEqual(next_time.minute, 30)

    def test_timezone_utc(self):
        parser = CronParser("0 30 9 * * *", tz="UTC")
        base = datetime(2024, 1, 15, 8, 0, 0)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time, datetime(2024, 1, 15, 9, 30, 0))

    def test_explain_L(self):
        parser = CronParser("0 0 0 L * *")
        explanation = parser.explain()
        self.assertIn("每月最后一天", explanation)

    def test_explain_W(self):
        parser = CronParser("0 0 12 15W * *")
        explanation = parser.explain()
        self.assertIn("15号最近的工作日", explanation)

    def test_explain_hash(self):
        parser = CronParser("0 0 10 * * 1#3")
        explanation = parser.explain()
        self.assertIn("第三个周一", explanation)

    def test_explain_L_weekday(self):
        parser = CronParser("0 30 9 * * 5L")
        explanation = parser.explain()
        self.assertIn("最后一个周五", explanation)

    def test_explain_year(self):
        parser = CronParser("0 0 9 1 1 * 2024-2026")
        explanation = parser.explain()
        self.assertIn("年", explanation)

    def test_explain_timezone(self):
        parser = CronParser("0 0 9 * * *", tz="+08:00")
        explanation = parser.explain()
        self.assertIn("时区", explanation)

    def test_L_invalid_field(self):
        with self.assertRaises(ValueError) as ctx:
            CronParser("L * * * * *")
        self.assertIn("只能用于", str(ctx.exception))

    def test_W_invalid_field(self):
        with self.assertRaises(ValueError) as ctx:
            CronParser("* W * * * *")
        self.assertIn("无效", str(ctx.exception))

    def test_hash_invalid_field(self):
        with self.assertRaises(ValueError) as ctx:
            CronParser("* * * 1#3 * *")
        self.assertIn("无效", str(ctx.exception))

    def test_hash_nth_out_of_range(self):
        with self.assertRaises(ValueError) as ctx:
            CronParser("* * * * * 1#6")
        self.assertIn("超出范围", str(ctx.exception))

    def test_invalid_timezone_format(self):
        with self.assertRaises(ValueError) as ctx:
            CronParser("* * * * * *", tz="invalid")
        self.assertIn("不支持的时区格式", str(ctx.exception))

    def test_get_next_10_executions(self):
        parser = CronParser("0 0 0 L * *")
        base = datetime(2024, 1, 1, 0, 0, 0)
        times = parser.get_next_n_executions(10, base)
        self.assertEqual(len(times), 10)
        for i in range(len(times) - 1):
            self.assertLess(times[i], times[i + 1])

    def test_combined_L_and_W(self):
        with self.assertRaises(ValueError) as ctx:
            CronParser("0 0 12 LW * *")
        self.assertIn("无效", str(ctx.exception))

    def test_last_weekday_no_param(self):
        parser = CronParser("0 0 0 * * L")
        base = datetime(2024, 1, 1, 0, 0, 0)
        next_time = parser.get_next_execution(base)
        self.assertEqual(next_time.day, 31)

    def test_invalid_L_format(self):
        with self.assertRaises(ValueError):
            CronParser("* * * * * XL")

    def test_invalid_W_format(self):
        with self.assertRaises(ValueError):
            CronParser("* * * XW * *")


if __name__ == '__main__':
    unittest.main(verbosity=2)
