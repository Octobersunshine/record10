from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import threading

try:
    from zoneinfo import ZoneInfo
    ZONEINFO_AVAILABLE = True
except ImportError:
    try:
        from backports.zoneinfo import ZoneInfo
        ZONEINFO_AVAILABLE = True
    except ImportError:
        ZONEINFO_AVAILABLE = False


class QuotaPeriod(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class ResetStrategy(Enum):
    FIXED_TIME = "fixed_time"
    ROLLING_WINDOW = "rolling_window"


class QuotaLevel(Enum):
    GLOBAL = "global"
    TENANT = "tenant"
    USER = "user"


def _ensure_timezone(dt: datetime, tz: Optional[Any] = None) -> datetime:
    if dt.tzinfo is None:
        if tz is not None:
            return dt.replace(tzinfo=tz)
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _to_timezone(dt: datetime, tz: Any) -> datetime:
    dt = _ensure_timezone(dt)
    return dt.astimezone(tz)


class QuotaUsage:
    def __init__(self, used: int = 0, last_reset: Optional[datetime] = None,
                 window_start: Optional[datetime] = None,
                 tz: Optional[Any] = None):
        self.used = used
        self.borrowed: int = 0
        now = datetime.now(tz or timezone.utc)
        self.last_reset = last_reset or now
        self.window_start = window_start or now


class QuotaConfig:
    def __init__(self, limit: int, period: QuotaPeriod, strategy: ResetStrategy,
                 fixed_reset_hour: int = 0, fixed_reset_minute: int = 0,
                 fixed_reset_weekday: int = 0, fixed_reset_day: int = 1,
                 tz: Optional[Any] = None,
                 level: QuotaLevel = QuotaLevel.USER,
                 parent_id: Optional[str] = None,
                 max_borrow: int = 0):
        self.limit = limit
        self.period = period
        self.strategy = strategy
        self.fixed_reset_hour = fixed_reset_hour
        self.fixed_reset_minute = fixed_reset_minute
        self.fixed_reset_weekday = fixed_reset_weekday
        self.fixed_reset_day = fixed_reset_day
        self.timezone = tz or timezone.utc
        self.level = level
        self.parent_id = parent_id
        self.max_borrow = max_borrow


class QuotaResult:
    def __init__(self, allowed: bool, remaining: int, limit: int,
                 reset_time: datetime, used: int,
                 borrowed: int = 0, max_borrow: int = 0,
                 level_results: Optional[Dict[str, 'QuotaResult']] = None):
        self.allowed = allowed
        self.remaining = remaining
        self.limit = limit
        self.reset_time = reset_time
        self.used = used
        self.borrowed = borrowed
        self.max_borrow = max_borrow
        self.level_results = level_results or {}

    @property
    def usage_rate(self) -> float:
        if self.limit == 0:
            return 0.0
        return self.used / self.limit

    @property
    def effective_remaining(self) -> int:
        return self.remaining - self.borrowed

    def to_dict(self) -> Dict:
        d = {
            "allowed": self.allowed,
            "remaining": self.remaining,
            "limit": self.limit,
            "reset_time": self.reset_time.isoformat(),
            "used": self.used,
            "borrowed": self.borrowed,
            "max_borrow": self.max_borrow,
            "usage_rate": round(self.usage_rate, 4),
            "effective_remaining": self.effective_remaining,
        }
        if self.level_results:
            d["level_results"] = {
                k: v.to_dict() for k, v in self.level_results.items()
            }
        return d


class UsageRecord:
    def __init__(self, timestamp: datetime, amount: int, cumulative_used: int):
        self.timestamp = timestamp
        self.amount = amount
        self.cumulative_used = cumulative_used


class TrendPrediction:
    def __init__(self, current_rate: float, predicted_usage_at_reset: int,
                 predicted_rate_at_reset: float, will_exhaust: bool,
                 estimated_exhaust_time: Optional[datetime],
                 avg_consumption_per_hour: float):
        self.current_rate = current_rate
        self.predicted_usage_at_reset = predicted_usage_at_reset
        self.predicted_rate_at_reset = predicted_rate_at_reset
        self.will_exhaust = will_exhaust
        self.estimated_exhaust_time = estimated_exhaust_time
        self.avg_consumption_per_hour = avg_consumption_per_hour

    def to_dict(self) -> Dict:
        return {
            "current_rate": round(self.current_rate, 4),
            "predicted_usage_at_reset": self.predicted_usage_at_reset,
            "predicted_rate_at_reset": round(self.predicted_rate_at_reset, 4),
            "will_exhaust": self.will_exhaust,
            "estimated_exhaust_time": (
                self.estimated_exhaust_time.isoformat()
                if self.estimated_exhaust_time else None
            ),
            "avg_consumption_per_hour": round(self.avg_consumption_per_hour, 2),
        }


class BorrowRecord:
    def __init__(self, amount: int, timestamp: datetime,
                 reason: str = "", repaid: int = 0):
        self.amount = amount
        self.timestamp = timestamp
        self.reason = reason
        self.repaid = repaid

    @property
    def outstanding(self) -> int:
        return self.amount - self.repaid


class ResetStrategyBase(ABC):
    @abstractmethod
    def check_and_reset(self, usage: QuotaUsage, config: QuotaConfig,
                        current_time: datetime) -> QuotaUsage:
        pass

    @abstractmethod
    def get_next_reset_time(self, config: QuotaConfig,
                            current_time: datetime) -> datetime:
        pass


class FixedTimeResetStrategy(ResetStrategyBase):
    def _get_last_reset_time(self, config: QuotaConfig,
                             current_time: datetime) -> datetime:
        tz = config.timezone
        current_local = _to_timezone(current_time, tz)

        if config.period == QuotaPeriod.DAILY:
            reset_today = current_local.replace(
                hour=config.fixed_reset_hour,
                minute=config.fixed_reset_minute,
                second=0, microsecond=0
            )
            if current_local >= reset_today:
                return reset_today
            return reset_today - timedelta(days=1)

        elif config.period == QuotaPeriod.WEEKLY:
            days_behind = current_local.weekday() - config.fixed_reset_weekday
            if days_behind < 0:
                days_behind += 7
            last_week = current_local - timedelta(days=days_behind)
            last_reset = last_week.replace(
                hour=config.fixed_reset_hour,
                minute=config.fixed_reset_minute,
                second=0, microsecond=0
            )
            if current_local < last_reset:
                last_reset -= timedelta(days=7)
            return last_reset

        elif config.period == QuotaPeriod.MONTHLY:
            reset_this_month = current_local.replace(
                day=config.fixed_reset_day,
                hour=config.fixed_reset_hour,
                minute=config.fixed_reset_minute,
                second=0, microsecond=0
            )
            if current_local >= reset_this_month:
                return reset_this_month
            if current_local.month == 1:
                return reset_this_month.replace(
                    year=current_local.year - 1, month=12)
            return reset_this_month.replace(month=current_local.month - 1)

        return current_local

    def check_and_reset(self, usage: QuotaUsage, config: QuotaConfig,
                        current_time: datetime) -> QuotaUsage:
        tz = config.timezone
        current_local = _to_timezone(current_time, tz)
        usage_last_reset_local = _to_timezone(usage.last_reset, tz)

        if current_local < usage_last_reset_local:
            usage.last_reset = current_local

        last_reset_time = self._get_last_reset_time(config, current_local)

        if _to_timezone(usage.last_reset, tz) < last_reset_time:
            usage.used = 0
            usage.borrowed = 0
            usage.last_reset = current_local

        return usage

    def get_next_reset_time(self, config: QuotaConfig,
                            current_time: datetime) -> datetime:
        tz = config.timezone
        current_local = _to_timezone(current_time, tz)

        if config.period == QuotaPeriod.DAILY:
            reset_today = current_local.replace(
                hour=config.fixed_reset_hour,
                minute=config.fixed_reset_minute,
                second=0, microsecond=0
            )
            if current_local < reset_today:
                return reset_today
            return reset_today + timedelta(days=1)

        elif config.period == QuotaPeriod.WEEKLY:
            days_ahead = config.fixed_reset_weekday - current_local.weekday()
            if days_ahead < 0 or (days_ahead == 0 and
                current_local >= current_local.replace(
                    hour=config.fixed_reset_hour,
                    minute=config.fixed_reset_minute,
                    second=0, microsecond=0
                )):
                days_ahead += 7
            next_week = current_local + timedelta(days=days_ahead)
            return next_week.replace(
                hour=config.fixed_reset_hour,
                minute=config.fixed_reset_minute,
                second=0, microsecond=0
            )

        elif config.period == QuotaPeriod.MONTHLY:
            reset_this_month = current_local.replace(
                day=config.fixed_reset_day,
                hour=config.fixed_reset_hour,
                minute=config.fixed_reset_minute,
                second=0, microsecond=0
            )
            if current_local < reset_this_month:
                return reset_this_month
            if current_local.month == 12:
                return reset_this_month.replace(
                    year=current_local.year + 1, month=1)
            return reset_this_month.replace(month=current_local.month + 1)

        return current_local


class RollingWindowResetStrategy(ResetStrategyBase):
    def check_and_reset(self, usage: QuotaUsage, config: QuotaConfig,
                        current_time: datetime) -> QuotaUsage:
        tz = config.timezone
        current_local = _to_timezone(current_time, tz)
        window_duration = self._get_window_duration(config.period)

        window_start_local = _to_timezone(usage.window_start, tz)
        if current_local < window_start_local:
            usage.window_start = current_local
            window_start_local = current_local

        window_end = window_start_local + window_duration

        if current_local >= window_end:
            full_windows_passed = int(
                (current_local - window_start_local).total_seconds() //
                window_duration.total_seconds()
            )
            usage.window_start = window_start_local + timedelta(
                seconds=full_windows_passed * window_duration.total_seconds()
            )
            usage.used = 0
            usage.borrowed = 0
            usage.last_reset = current_local

        return usage

    def get_next_reset_time(self, config: QuotaConfig,
                            current_time: datetime) -> datetime:
        tz = config.timezone
        current_local = _to_timezone(current_time, tz)
        window_duration = self._get_window_duration(config.period)
        return current_local + window_duration

    def _get_window_duration(self, period: QuotaPeriod) -> timedelta:
        if period == QuotaPeriod.DAILY:
            return timedelta(days=1)
        elif period == QuotaPeriod.WEEKLY:
            return timedelta(weeks=1)
        elif period == QuotaPeriod.MONTHLY:
            return timedelta(days=30)
        return timedelta(days=1)


class HierarchicalQuotaManager:
    def __init__(self, default_timezone: Optional[Any] = None,
                 max_history_per_quota: int = 1000):
        self._quotas: Dict[str, QuotaConfig] = {}
        self._usage: Dict[str, QuotaUsage] = {}
        self._strategies: Dict[ResetStrategy, ResetStrategyBase] = {
            ResetStrategy.FIXED_TIME: FixedTimeResetStrategy(),
            ResetStrategy.ROLLING_WINDOW: RollingWindowResetStrategy()
        }
        self._default_timezone = default_timezone or timezone.utc
        self._lock = threading.RLock()
        self._children: Dict[str, List[str]] = {}
        self._borrow_records: Dict[str, List[BorrowRecord]] = {}
        self._usage_history: Dict[str, deque] = {}
        self._max_history = max_history_per_quota

    def create_quota(self, quota_id: str, limit: int, period: QuotaPeriod,
                     strategy: ResetStrategy, **kwargs) -> None:
        with self._lock:
            if 'tz' not in kwargs:
                kwargs['tz'] = self._default_timezone
            config = QuotaConfig(
                limit=limit, period=period, strategy=strategy, **kwargs
            )
            self._quotas[quota_id] = config
            self._usage[quota_id] = QuotaUsage(tz=config.timezone)
            self._children[quota_id] = []
            self._borrow_records[quota_id] = []
            self._usage_history[quota_id] = deque(
                maxlen=self._max_history)

            if config.parent_id and config.parent_id in self._children:
                self._children[config.parent_id].append(quota_id)

    def delete_quota(self, quota_id: str) -> None:
        with self._lock:
            config = self._quotas.get(quota_id)
            if config and config.parent_id and config.parent_id in self._children:
                children = self._children[config.parent_id]
                if quota_id in children:
                    children.remove(quota_id)
            self._quotas.pop(quota_id, None)
            self._usage.pop(quota_id, None)
            self._children.pop(quota_id, None)
            self._borrow_records.pop(quota_id, None)
            self._usage_history.pop(quota_id, None)

    def _get_ancestor_chain(self, quota_id: str) -> List[str]:
        chain = []
        current = quota_id
        while current:
            chain.append(current)
            config = self._quotas.get(current)
            if config and config.parent_id:
                current = config.parent_id
            else:
                break
        return chain

    def _get_effective_limit(self, quota_id: str) -> int:
        config = self._quotas.get(quota_id)
        if not config:
            return 0

        own_limit = config.limit
        chain = self._get_ancestor_chain(quota_id)

        for ancestor_id in chain:
            if ancestor_id == quota_id:
                continue
            ancestor_config = self._quotas.get(ancestor_id)
            ancestor_usage = self._usage.get(ancestor_id)
            if ancestor_config and ancestor_usage:
                ancestor_available = (
                    ancestor_config.limit - ancestor_usage.used + usage.used
                    if (usage := self._usage.get(quota_id)) else own_limit
                )
                if ancestor_available < own_limit:
                    own_limit = ancestor_available

        return own_limit

    def consume(self, quota_id: str, amount: int = 1,
                current_time: Optional[datetime] = None) -> QuotaResult:
        with self._lock:
            if quota_id not in self._quotas:
                raise ValueError(f"Quota '{quota_id}' does not exist")

            config = self._quotas[quota_id]
            usage = self._usage[quota_id]
            now = current_time or datetime.now(config.timezone)
            strategy = self._strategies[config.strategy]

            usage = strategy.check_and_reset(usage, config, now)
            reset_time = strategy.get_next_reset_time(config, now)

            chain = self._get_ancestor_chain(quota_id)
            level_results: Dict[str, QuotaResult] = {}

            effective_limit = self._get_effective_limit(quota_id)
            can_consume = (usage.used + amount <= effective_limit)
            can_borrow = (
                not can_consume and
                config.max_borrow > 0 and
                usage.borrowed + (usage.used + amount - config.limit) <= config.max_borrow
            )

            if can_consume:
                usage.used += amount
                allowed = True
                for ancestor_id in chain:
                    if ancestor_id == quota_id:
                        continue
                    ancestor_usage = self._usage.get(ancestor_id)
                    if ancestor_usage:
                        ancestor_usage.used += amount
            elif can_borrow:
                borrow_needed = usage.used + amount - config.limit
                usage.used += amount
                usage.borrowed += borrow_needed
                self._record_borrow(quota_id, borrow_needed, now)
                allowed = True
                for ancestor_id in chain:
                    if ancestor_id == quota_id:
                        continue
                    ancestor_usage = self._usage.get(ancestor_id)
                    if ancestor_usage:
                        ancestor_usage.used += amount
            else:
                allowed = False

            self._record_usage(quota_id, amount, usage.used, now)

            for ancestor_id in chain:
                if ancestor_id == quota_id:
                    continue
                a_config = self._quotas.get(ancestor_id)
                a_usage = self._usage.get(ancestor_id)
                if a_config and a_usage:
                    a_remaining = a_config.limit - a_usage.used
                    a_reset = strategy.get_next_reset_time(a_config, now)
                    level_results[ancestor_id] = QuotaResult(
                        allowed=allowed,
                        remaining=a_remaining,
                        limit=a_config.limit,
                        reset_time=a_reset,
                        used=a_usage.used,
                        borrowed=a_usage.borrowed,
                        max_borrow=a_config.max_borrow,
                    )

            remaining = config.limit - usage.used

            return QuotaResult(
                allowed=allowed,
                remaining=remaining,
                limit=config.limit,
                reset_time=reset_time,
                used=usage.used,
                borrowed=usage.borrowed,
                max_borrow=config.max_borrow,
                level_results=level_results,
            )

    def check(self, quota_id: str,
              current_time: Optional[datetime] = None) -> QuotaResult:
        with self._lock:
            if quota_id not in self._quotas:
                raise ValueError(f"Quota '{quota_id}' does not exist")

            config = self._quotas[quota_id]
            usage = self._usage[quota_id]
            now = current_time or datetime.now(config.timezone)
            strategy = self._strategies[config.strategy]

            usage = strategy.check_and_reset(usage, config, now)
            reset_time = strategy.get_next_reset_time(config, now)

            effective_limit = self._get_effective_limit(quota_id)
            remaining = effective_limit - usage.used

            chain = self._get_ancestor_chain(quota_id)
            level_results: Dict[str, QuotaResult] = {}
            for ancestor_id in chain:
                if ancestor_id == quota_id:
                    continue
                a_config = self._quotas.get(ancestor_id)
                a_usage = self._usage.get(ancestor_id)
                if a_config and a_usage:
                    a_remaining = a_config.limit - a_usage.used
                    a_reset = strategy.get_next_reset_time(a_config, now)
                    level_results[ancestor_id] = QuotaResult(
                        allowed=a_remaining > 0,
                        remaining=a_remaining,
                        limit=a_config.limit,
                        reset_time=a_reset,
                        used=a_usage.used,
                        borrowed=a_usage.borrowed,
                        max_borrow=a_config.max_borrow,
                    )

            return QuotaResult(
                allowed=remaining > 0,
                remaining=remaining,
                limit=config.limit,
                reset_time=reset_time,
                used=usage.used,
                borrowed=usage.borrowed,
                max_borrow=config.max_borrow,
                level_results=level_results,
            )

    def borrow(self, quota_id: str, amount: int,
               reason: str = "",
               current_time: Optional[datetime] = None) -> QuotaResult:
        with self._lock:
            if quota_id not in self._quotas:
                raise ValueError(f"Quota '{quota_id}' does not exist")

            config = self._quotas[quota_id]
            if config.max_borrow <= 0:
                raise ValueError(
                    f"Quota '{quota_id}' does not allow borrowing")

            usage = self._usage[quota_id]
            now = current_time or datetime.now(config.timezone)
            strategy = self._strategies[config.strategy]

            usage = strategy.check_and_reset(usage, config, now)
            reset_time = strategy.get_next_reset_time(config, now)

            if usage.borrowed + amount > config.max_borrow:
                allowed = False
            else:
                usage.used += amount
                usage.borrowed += amount
                self._record_borrow(quota_id, amount, now, reason)
                allowed = True
                self._record_usage(quota_id, amount, usage.used, now)

            remaining = config.limit - usage.used

            return QuotaResult(
                allowed=allowed,
                remaining=remaining,
                limit=config.limit,
                reset_time=reset_time,
                used=usage.used,
                borrowed=usage.borrowed,
                max_borrow=config.max_borrow,
            )

    def repay(self, quota_id: str, amount: int,
              current_time: Optional[datetime] = None) -> QuotaResult:
        with self._lock:
            if quota_id not in self._quotas:
                raise ValueError(f"Quota '{quota_id}' does not exist")

            config = self._quotas[quota_id]
            usage = self._usage[quota_id]
            now = current_time or datetime.now(config.timezone)
            strategy = self._strategies[config.strategy]

            usage = strategy.check_and_reset(usage, config, now)
            reset_time = strategy.get_next_reset_time(config, now)

            actual_repay = min(amount, usage.borrowed)
            usage.borrowed -= actual_repay
            usage.used = max(0, usage.used - actual_repay)

            records = self._borrow_records.get(quota_id, [])
            remaining_repay = actual_repay
            for record in reversed(records):
                if remaining_repay <= 0:
                    break
                outstanding = record.outstanding
                repay_for_this = min(remaining_repay, outstanding)
                record.repaid += repay_for_this
                remaining_repay -= repay_for_this

            remaining = config.limit - usage.used

            return QuotaResult(
                allowed=True,
                remaining=remaining,
                limit=config.limit,
                reset_time=reset_time,
                used=usage.used,
                borrowed=usage.borrowed,
                max_borrow=config.max_borrow,
            )

    def get_borrow_records(self, quota_id: str) -> List[Dict]:
        with self._lock:
            records = self._borrow_records.get(quota_id, [])
            return [
                {
                    "amount": r.amount,
                    "timestamp": r.timestamp.isoformat(),
                    "reason": r.reason,
                    "repaid": r.repaid,
                    "outstanding": r.outstanding,
                }
                for r in records
            ]

    def _record_borrow(self, quota_id: str, amount: int,
                       timestamp: datetime, reason: str = ""):
        records = self._borrow_records.get(quota_id, [])
        records.append(BorrowRecord(
            amount=amount, timestamp=timestamp, reason=reason
        ))

    def _record_usage(self, quota_id: str, amount: int,
                      cumulative: int, timestamp: datetime):
        history = self._usage_history.get(quota_id)
        if history is not None:
            history.append(UsageRecord(
                timestamp=timestamp,
                amount=amount,
                cumulative_used=cumulative,
            ))

    def predict_trend(self, quota_id: str,
                      current_time: Optional[datetime] = None
                      ) -> TrendPrediction:
        with self._lock:
            if quota_id not in self._quotas:
                raise ValueError(f"Quota '{quota_id}' does not exist")

            config = self._quotas[quota_id]
            usage = self._usage[quota_id]
            now = current_time or datetime.now(config.timezone)
            strategy = self._strategies[config.strategy]

            usage = strategy.check_and_reset(usage, config, now)
            reset_time = strategy.get_next_reset_time(config, now)

            history = self._usage_history.get(quota_id, deque())

            current_rate = usage.used / config.limit if config.limit > 0 else 0.0

            if len(history) < 2:
                time_to_reset = (reset_time - now).total_seconds()
                avg_per_hour = usage.used / max(
                    (now - usage.last_reset).total_seconds() / 3600, 0.001
                ) if usage.used > 0 else 0.0
                predicted_usage = min(
                    int(avg_per_hour * time_to_reset / 3600),
                    config.limit
                )
                predicted_rate = (
                    predicted_usage / config.limit if config.limit > 0 else 0.0
                )
                will_exhaust = predicted_usage >= config.limit
                exhaust_time = None
                if avg_per_hour > 0:
                    remaining_quota = config.limit - usage.used
                    hours_to_exhaust = remaining_quota / avg_per_hour
                    exhaust_time = now + timedelta(hours=hours_to_exhaust)
                    if exhaust_time > reset_time:
                        exhaust_time = None
                return TrendPrediction(
                    current_rate=current_rate,
                    predicted_usage_at_reset=predicted_usage,
                    predicted_rate_at_reset=predicted_rate,
                    will_exhaust=will_exhaust,
                    estimated_exhaust_time=exhaust_time,
                    avg_consumption_per_hour=avg_per_hour,
                )

            records = list(history)
            first_ts = records[0].timestamp
            last_ts = records[-1].timestamp
            elapsed_hours = max(
                (last_ts - first_ts).total_seconds() / 3600, 0.001
            )

            total_consumed = sum(r.amount for r in records)
            avg_per_hour = total_consumed / elapsed_hours

            time_to_reset = (reset_time - now).total_seconds()
            hours_to_reset = time_to_reset / 3600

            projected_future = avg_per_hour * hours_to_reset
            predicted_usage = min(
                int(usage.used + projected_future), config.limit
            )
            predicted_rate = (
                predicted_usage / config.limit if config.limit > 0 else 0.0
            )

            will_exhaust = predicted_usage >= config.limit
            exhaust_time = None
            if avg_per_hour > 0 and usage.used < config.limit:
                remaining_quota = config.limit - usage.used
                hours_to_exhaust = remaining_quota / avg_per_hour
                exhaust_time = now + timedelta(hours=hours_to_exhaust)
                if exhaust_time > reset_time:
                    exhaust_time = None

            return TrendPrediction(
                current_rate=current_rate,
                predicted_usage_at_reset=predicted_usage,
                predicted_rate_at_reset=predicted_rate,
                will_exhaust=will_exhaust,
                estimated_exhaust_time=exhaust_time,
                avg_consumption_per_hour=avg_per_hour,
            )

    def get_children(self, quota_id: str) -> List[str]:
        with self._lock:
            return list(self._children.get(quota_id, []))

    def get_usage_history(self, quota_id: str) -> List[Dict]:
        with self._lock:
            history = self._usage_history.get(quota_id, deque())
            return [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "amount": r.amount,
                    "cumulative_used": r.cumulative_used,
                }
                for r in history
            ]

    def reset(self, quota_id: str) -> None:
        with self._lock:
            if quota_id in self._usage:
                tz = (self._quotas[quota_id].timezone
                      if quota_id in self._quotas else None)
                self._usage[quota_id] = QuotaUsage(tz=tz)
                self._borrow_records[quota_id] = []

    def get_quota_config(self, quota_id: str) -> Optional[QuotaConfig]:
        return self._quotas.get(quota_id)
