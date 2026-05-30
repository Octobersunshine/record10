import time
from threading import Lock
from typing import Optional, Union
from .sliding_window_circular import SlidingWindowCircularLimiter


class MultiDimensionLimiter:
    ALGORITHM_NAME = 'multi_dimension'

    DIMENSION_API = 'api'
    DIMENSION_USER = 'user'
    DIMENSION_IP = 'ip'
    DIMENSION_API_USER = 'api:user'
    DIMENSION_API_IP = 'api:ip'
    DIMENSION_USER_IP = 'user:ip'
    DIMENSION_API_USER_IP = 'api:user:ip'

    ALL_DIMENSIONS = [
        DIMENSION_API, DIMENSION_USER, DIMENSION_IP,
        DIMENSION_API_USER, DIMENSION_API_IP,
        DIMENSION_USER_IP, DIMENSION_API_USER_IP
    ]

    def __init__(self, default_qps: int = 10, window_size: int = 1):
        self.default_qps = default_qps
        self.window_size = window_size
        self._limiters: dict[str, SlidingWindowCircularLimiter] = {}
        self._dimension_rules: dict[str, dict] = {}
        self._lock = Lock()

    def _get_or_create_limiter(self, dimension: str) -> SlidingWindowCircularLimiter:
        if dimension not in self._limiters:
            self._limiters[dimension] = SlidingWindowCircularLimiter(
                default_qps=self.default_qps,
                window_size=self.window_size
            )
        return self._limiters[dimension]

    def _build_key(self, dimension: str, api_path: str = '',
                   user_id: str = '', ip: str = '') -> str:
        parts = [dimension]
        if dimension in (self.DIMENSION_API, self.DIMENSION_API_USER,
                         self.DIMENSION_API_IP, self.DIMENSION_API_USER_IP):
            if api_path:
                parts.append(f"api={api_path}")
        if dimension in (self.DIMENSION_USER, self.DIMENSION_API_USER,
                         self.DIMENSION_USER_IP, self.DIMENSION_API_USER_IP):
            if user_id:
                parts.append(f"user={user_id}")
        if dimension in (self.DIMENSION_IP, self.DIMENSION_API_IP,
                         self.DIMENSION_USER_IP, self.DIMENSION_API_USER_IP):
            if ip:
                parts.append(f"ip={ip}")
        return '|'.join(parts)

    def set_dimension_rule(self, dimension: str, qps: int,
                           api_pattern: str = None) -> None:
        if dimension not in self.ALL_DIMENSIONS:
            raise ValueError(
                f"Invalid dimension: {dimension}. "
                f"Must be one of: {self.ALL_DIMENSIONS}"
            )

        with self._lock:
            limiter = self._get_or_create_limiter(dimension)
            rule_key = api_pattern or '*'
            limiter.set_limit(rule_key, qps)
            self._dimension_rules[f"{dimension}:{rule_key}"] = {
                'dimension': dimension,
                'qps': qps,
                'api_pattern': api_pattern
            }

    def remove_dimension_rule(self, dimension: str, api_pattern: str = None) -> None:
        rule_key = api_pattern or '*'
        rule_key_full = f"{dimension}:{rule_key}"
        with self._lock:
            if rule_key_full in self._dimension_rules:
                del self._dimension_rules[rule_key_full]
            if dimension in self._limiters:
                self._limiters[dimension].remove_limit(rule_key)

    def get_all_rules(self) -> dict:
        return dict(self._dimension_rules)

    def _get_qps_for_api(self, dimension: str, api_path: str) -> int:
        if dimension not in self._limiters:
            return self.default_qps
        limiter = self._limiters[dimension]
        specific = limiter.get_limit(api_path)
        wildcard = limiter.get_limit('*')
        if specific != self.default_qps:
            return specific
        if wildcard != self.default_qps:
            return wildcard
        return self.default_qps

    def set_limit(self, api_path: str, qps: int) -> None:
        self.set_dimension_rule(self.DIMENSION_API, qps, api_pattern=api_path)

    def get_limit(self, api_path: str) -> Union[int, dict]:
        rules = self._get_rules_for_dimension(self.DIMENSION_API)
        for rule in rules:
            if rule.get('api_pattern') == api_path:
                return rule['qps']
        for rule in rules:
            if rule.get('api_pattern') in (None, '*'):
                return rule['qps']
        return self.default_qps

    def get_all_limits(self) -> dict:
        result = {}
        for key, rule in self._dimension_rules.items():
            result[key] = rule['qps']
        return result

    def remove_limit(self, api_path: str) -> None:
        self.remove_dimension_rule(self.DIMENSION_API, api_pattern=api_path)

    def allow_request(self, api_path: str = '', user_id: str = '',
                      ip: str = '') -> tuple[bool, dict]:
        now = time.time()
        all_results = {}
        overall_allowed = True
        most_restrictive = None
        min_retry = float('inf')

        active_dimensions = set()
        for rule_key, rule in self._dimension_rules.items():
            active_dimensions.add(rule['dimension'])

        if not active_dimensions:
            active_dimensions = {self.DIMENSION_API}

        for dimension in active_dimensions:
            limiter = self._get_or_create_limiter(dimension)
            key = self._build_key(dimension, api_path, user_id, ip)

            effective_qps = self._get_qps_for_api(dimension, api_path)
            limiter.set_limit(key, effective_qps)

            stats = limiter.get_stats(key)
            current_qps = stats.get('current_qps', 0)
            current_remaining = stats.get('remaining', effective_qps)

            allowed = current_remaining > 0

            all_results[dimension] = {
                'allowed': allowed,
                'key': key,
                'qps': current_qps,
                'limit_qps': effective_qps,
                'remaining': current_remaining
            }

            if not allowed:
                overall_allowed = False
                retry = 1.0 / effective_qps if effective_qps > 0 else 1.0
                if retry < min_retry:
                    min_retry = retry
                    most_restrictive = dimension

        if overall_allowed:
            for dimension in active_dimensions:
                limiter = self._get_or_create_limiter(dimension)
                key = self._build_key(dimension, api_path, user_id, ip)
                limiter.allow_request(key)

                all_results[dimension]['allowed'] = True

        result = {
            'allowed': overall_allowed,
            'algorithm': self.ALGORITHM_NAME,
            'dimensions_checked': len(all_results),
            'dimensions': all_results,
            'memory_fixed': True
        }

        if not overall_allowed:
            result['retry_after'] = max(0, min_retry)
            result['blocked_by'] = most_restrictive
            result['limit_qps'] = all_results[most_restrictive]['limit_qps']
            result['current_qps'] = all_results[most_restrictive]['qps']
            result['remaining'] = 0
        else:
            result['retry_after'] = 0
            min_remaining = min(
                (d['remaining'] - 1 for d in all_results.values()),
                default=0
            )
            result['remaining'] = max(0, min_remaining)
            if all_results:
                first_dim = next(iter(all_results.values()))
                result['limit_qps'] = first_dim['limit_qps']
                result['current_qps'] = first_dim['qps']

        return overall_allowed, result

    def get_stats(self, api_path: str = '', user_id: str = '',
                  ip: str = '') -> dict:
        stats = {}
        for dimension, limiter in self._limiters.items():
            key = self._build_key(dimension, api_path, user_id, ip)
            stats[dimension] = limiter.get_stats(key)
        return {
            'algorithm': self.ALGORITHM_NAME,
            'dimensions': stats,
            'active_rules': len(self._dimension_rules)
        }

    def get_memory_info(self) -> dict:
        total = 0
        for limiter in self._limiters.values():
            info = limiter.get_memory_info()
            total += info.get('memory_usage_bytes', 0)
        return {
            'memory_fixed': True,
            'total_memory_bytes': total,
            'active_dimensions': len(self._limiters),
            'active_rules': len(self._dimension_rules)
        }

    def reset(self, dimension: str = None) -> None:
        with self._lock:
            if dimension:
                if dimension in self._limiters:
                    self._limiters[dimension].reset()
            else:
                for limiter in self._limiters.values():
                    limiter.reset()
