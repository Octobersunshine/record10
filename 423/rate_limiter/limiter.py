from typing import Union, Optional
from .sliding_window import SlidingWindowLimiter
from .sliding_window_circular import SlidingWindowCircularLimiter
from .sliding_window_redis import SlidingWindowRedisLimiter, REDIS_AVAILABLE
from .token_bucket import TokenBucketLimiter
from .multi_dimension import MultiDimensionLimiter
from .distributed_redis import DistributedRateLimiter
from .dynamic_throttle import DynamicThrottleManager


class RateLimiter:
    ALGORITHM_SLIDING_WINDOW = 'sliding_window'
    ALGORITHM_SLIDING_WINDOW_CIRCULAR = 'sliding_window_circular'
    ALGORITHM_SLIDING_WINDOW_REDIS = 'sliding_window_redis'
    ALGORITHM_TOKEN_BUCKET = 'token_bucket'
    ALGORITHM_MULTI_DIMENSION = 'multi_dimension'
    ALGORITHM_DISTRIBUTED_REDIS = 'distributed_redis'

    def __init__(self, algorithm: str = ALGORITHM_SLIDING_WINDOW_CIRCULAR,
                 default_qps: int = 10, **kwargs):
        self.algorithm = algorithm
        self.default_qps = default_qps
        self._limiter = self._create_limiter(algorithm, default_qps, **kwargs)
        self._dynamic_throttle: Optional[DynamicThrottleManager] = None
        self._dynamic_enabled = kwargs.get('dynamic_throttle', False)

        if self._dynamic_enabled:
            self._dynamic_throttle = DynamicThrottleManager(
                base_qps=default_qps,
                check_interval=kwargs.get('dynamic_check_interval', 5.0),
                min_qps=kwargs.get('dynamic_min_qps', 1),
                max_qps=kwargs.get('dynamic_max_qps'),
                cooldown_seconds=kwargs.get('dynamic_cooldown', 30.0),
                multipliers=kwargs.get('dynamic_multipliers')
            )

    def _create_limiter(self, algorithm: str, default_qps: int, **kwargs):
        if algorithm == self.ALGORITHM_SLIDING_WINDOW:
            window_size = kwargs.get('window_size', 1)
            return SlidingWindowLimiter(default_qps=default_qps, window_size=window_size)
        elif algorithm == self.ALGORITHM_SLIDING_WINDOW_CIRCULAR:
            window_size = kwargs.get('window_size', 1)
            return SlidingWindowCircularLimiter(default_qps=default_qps, window_size=window_size)
        elif algorithm == self.ALGORITHM_SLIDING_WINDOW_REDIS:
            if not REDIS_AVAILABLE:
                raise ImportError("Redis library not installed. pip install redis")
            return SlidingWindowRedisLimiter(
                default_qps=default_qps,
                window_size=kwargs.get('window_size', 1),
                redis_host=kwargs.get('redis_host', 'localhost'),
                redis_port=kwargs.get('redis_port', 6379),
                redis_db=kwargs.get('redis_db', 0),
                redis_password=kwargs.get('redis_password'),
                redis_prefix=kwargs.get('redis_prefix', 'rate_limit:')
            )
        elif algorithm == self.ALGORITHM_TOKEN_BUCKET:
            burst = kwargs.get('burst', default_qps)
            return TokenBucketLimiter(default_qps=default_qps, burst=burst)
        elif algorithm == self.ALGORITHM_MULTI_DIMENSION:
            return MultiDimensionLimiter(
                default_qps=default_qps,
                window_size=kwargs.get('window_size', 1)
            )
        elif algorithm == self.ALGORITHM_DISTRIBUTED_REDIS:
            if not REDIS_AVAILABLE:
                raise ImportError("Redis library not installed. pip install redis")
            return DistributedRateLimiter(
                default_qps=default_qps,
                window_size=kwargs.get('window_size', 1),
                redis_host=kwargs.get('redis_host', 'localhost'),
                redis_port=kwargs.get('redis_port', 6379),
                redis_db=kwargs.get('redis_db', 0),
                redis_password=kwargs.get('redis_password'),
                redis_prefix=kwargs.get('redis_prefix', 'rl:'),
                redis_timeout=kwargs.get('redis_timeout', 1.0)
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {self.available_algorithms}")

    def set_limit(self, api_path: str, qps: int, **kwargs) -> None:
        if self.algorithm == self.ALGORITHM_TOKEN_BUCKET:
            burst = kwargs.get('burst', qps)
            self._limiter.set_limit(api_path, qps, burst)
        else:
            self._limiter.set_limit(api_path, qps)

    def get_limit(self, api_path: str) -> Union[int, dict]:
        return self._limiter.get_limit(api_path)

    def get_all_limits(self) -> dict:
        return self._limiter.get_all_limits()

    def remove_limit(self, api_path: str) -> None:
        self._limiter.remove_limit(api_path)

    def allow_request(self, api_path: str, **kwargs) -> tuple[bool, dict]:
        user_id = kwargs.get('user_id', '')
        ip = kwargs.get('ip', '')

        if self.algorithm in (self.ALGORITHM_MULTI_DIMENSION,
                              self.ALGORITHM_DISTRIBUTED_REDIS):
            allowed, info = self._limiter.allow_request(
                api_path=api_path, user_id=user_id, ip=ip
            )
        else:
            allowed, info = self._limiter.allow_request(api_path)

        if self._dynamic_throttle and self._dynamic_throttle.enabled:
            base_qps = info.get('limit_qps', self.default_qps)
            effective_qps = self._dynamic_throttle.get_effective_qps(base_qps)
            info['dynamic_adjusted'] = True
            info['dynamic_base_qps'] = base_qps
            info['dynamic_effective_qps'] = effective_qps
            info['dynamic_multiplier'] = self._dynamic_throttle._current_multiplier

        if self._dynamic_throttle:
            self._dynamic_throttle.record_request(is_error=not allowed)

        return allowed, info

    def get_stats(self, api_path: str = '', **kwargs) -> dict:
        if self.algorithm in (self.ALGORITHM_MULTI_DIMENSION,
                              self.ALGORITHM_DISTRIBUTED_REDIS):
            return self._limiter.get_stats(
                api_path=api_path,
                user_id=kwargs.get('user_id', ''),
                ip=kwargs.get('ip', '')
            )
        return self._limiter.get_stats(api_path)

    def reset(self, api_path: str = None) -> None:
        self._limiter.reset(api_path)

    def switch_algorithm(self, algorithm: str, **kwargs) -> None:
        old_limits = {}
        old_dimension_rules = {}

        is_old_multi_dim = hasattr(self._limiter, 'get_all_rules')
        is_new_multi_dim = algorithm in (self.ALGORITHM_MULTI_DIMENSION,
                                          self.ALGORITHM_DISTRIBUTED_REDIS)

        if is_old_multi_dim:
            old_dimension_rules = self._limiter.get_all_rules()
        else:
            old_limits = self.get_all_limits()

        self.algorithm = algorithm
        self._limiter = self._create_limiter(algorithm, self.default_qps, **kwargs)

        if is_new_multi_dim:
            if old_dimension_rules and hasattr(self._limiter, 'set_dimension_rule'):
                for rule_key, rule in old_dimension_rules.items():
                    try:
                        self._limiter.set_dimension_rule(
                            rule['dimension'],
                            rule['qps'],
                            api_pattern=rule.get('api_pattern')
                        )
                    except Exception:
                        pass
        else:
            for api_path, limit in old_limits.items():
                try:
                    if isinstance(limit, dict):
                        self.set_limit(api_path, limit['qps'], burst=limit.get('burst'))
                    else:
                        self.set_limit(api_path, limit)
                except Exception:
                    pass

    def set_dimension_rule(self, dimension: str, qps: int, **kwargs) -> None:
        if self.algorithm not in (self.ALGORITHM_MULTI_DIMENSION,
                                  self.ALGORITHM_DISTRIBUTED_REDIS):
            raise ValueError(
                f"Dimension rules require multi_dimension or distributed_redis algorithm. "
                f"Current: {self.algorithm}"
            )
        api_pattern = kwargs.get('api_pattern')
        if self.algorithm == self.ALGORITHM_DISTRIBUTED_REDIS:
            burst = kwargs.get('burst', qps)
            strategy = kwargs.get('strategy', 'sliding_window')
            self._limiter.set_dimension_rule(
                dimension, qps, api_pattern=api_pattern,
                burst=burst, strategy=strategy
            )
        else:
            self._limiter.set_dimension_rule(dimension, qps, api_pattern=api_pattern)

    def remove_dimension_rule(self, dimension: str, api_pattern: str = None) -> None:
        if self.algorithm not in (self.ALGORITHM_MULTI_DIMENSION,
                                  self.ALGORITHM_DISTRIBUTED_REDIS):
            raise ValueError("Dimension rules require multi_dimension or distributed_redis algorithm")
        self._limiter.remove_dimension_rule(dimension, api_pattern=api_pattern)

    def get_all_dimension_rules(self) -> dict:
        if hasattr(self._limiter, 'get_all_rules'):
            return self._limiter.get_all_rules()
        return {}

    def get_dynamic_status(self) -> dict:
        if self._dynamic_throttle:
            return self._dynamic_throttle.get_status()
        return {'enabled': False}

    def set_dynamic_throttle(self, enabled: bool = True, **kwargs) -> dict:
        if enabled and self._dynamic_throttle is None:
            self._dynamic_throttle = DynamicThrottleManager(
                base_qps=kwargs.get('base_qps', self.default_qps),
                check_interval=kwargs.get('check_interval', 5.0),
                min_qps=kwargs.get('min_qps', 1),
                max_qps=kwargs.get('max_qps'),
                cooldown_seconds=kwargs.get('cooldown_seconds', 30.0),
                multipliers=kwargs.get('multipliers')
            )
            self._dynamic_throttle.start()
        elif enabled and self._dynamic_throttle:
            self._dynamic_throttle.enabled = True
            if not self._dynamic_throttle._monitor._monitor_thread or \
               not self._dynamic_throttle._monitor._monitor_thread.is_alive():
                self._dynamic_throttle.start()
        elif self._dynamic_throttle:
            self._dynamic_throttle.enabled = False

        return self.get_dynamic_status()

    def force_dynamic_adjust(self, multiplier: float, level: str = None) -> dict:
        if not self._dynamic_throttle:
            return {'error': 'Dynamic throttle not initialized'}
        return self._dynamic_throttle.force_adjust(multiplier, level)

    def get_memory_info(self) -> dict:
        if hasattr(self._limiter, 'get_memory_info'):
            return self._limiter.get_memory_info()
        return {'memory_fixed': False, 'note': 'Memory usage grows with request count'}

    @property
    def available_algorithms(self) -> list[str]:
        algos = [
            self.ALGORITHM_SLIDING_WINDOW_CIRCULAR,
            self.ALGORITHM_SLIDING_WINDOW,
            self.ALGORITHM_TOKEN_BUCKET,
            self.ALGORITHM_MULTI_DIMENSION,
        ]
        if REDIS_AVAILABLE:
            algos.append(self.ALGORITHM_SLIDING_WINDOW_REDIS)
            algos.append(self.ALGORITHM_DISTRIBUTED_REDIS)
        return algos
