import os
import uuid
import json
import threading
from abc import ABC, abstractmethod

try:
    from cachetools import TTLCache
    _HAS_CACHETOOLS = True
except ImportError:
    _HAS_CACHETOOLS = False

try:
    import redis
    _HAS_REDIS = True
except ImportError:
    _HAS_REDIS = False


class CaptchaCacheBackend(ABC):
    @abstractmethod
    def put(self, token, value, ttl_seconds):
        pass

    @abstractmethod
    def pop(self, token):
        pass

    @abstractmethod
    def size(self):
        pass


class MemoryTTLCache(CaptchaCacheBackend):
    def __init__(self, maxsize=10000):
        if not _HAS_CACHETOOLS:
            raise ImportError("cachetools is required for MemoryTTLCache. Install with: pip install cachetools")
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._caches = {}

    def _get_cache(self, ttl):
        if ttl not in self._caches:
            with self._lock:
                if ttl not in self._caches:
                    self._caches[ttl] = TTLCache(maxsize=self._maxsize, ttl=ttl)
        return self._caches[ttl]

    def put(self, token, value, ttl_seconds):
        cache = self._get_cache(ttl_seconds)
        with self._lock:
            cache[token] = value
        return token

    def pop(self, token):
        with self._lock:
            for cache in self._caches.values():
                if token in cache:
                    return cache.pop(token, None)
        return None

    def size(self):
        with self._lock:
            return sum(len(c) for c in self._caches.values())


class RedisCache(CaptchaCacheBackend):
    def __init__(self, redis_url=None, key_prefix="captcha:"):
        if not _HAS_REDIS:
            raise ImportError("redis is required for RedisCache. Install with: pip install redis")
        self._key_prefix = key_prefix
        self._client = redis.Redis.from_url(
            redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
            decode_responses=True
        )
        self._client.ping()

    def put(self, token, value, ttl_seconds):
        key = self._key_prefix + token
        self._client.setex(key, ttl_seconds, json.dumps(value))
        return token

    def pop(self, token):
        key = self._key_prefix + token
        pipe = self._client.pipeline()
        pipe.get(key)
        pipe.delete(key)
        result, _ = pipe.execute()
        if result is None:
            return None
        return json.loads(result)

    def size(self):
        keys = list(self._client.scan_iter(match=self._key_prefix + '*'))
        return len(keys)


class CaptchaCache:
    def __init__(self, ttl=300, backend=None):
        self._ttl = ttl
        self._backend = backend or self._default_backend()

    def _default_backend(self):
        redis_url = os.getenv('REDIS_URL')
        if redis_url and _HAS_REDIS:
            return RedisCache(redis_url=redis_url)
        return MemoryTTLCache()

    def put(self, code, captcha_type='text'):
        token = uuid.uuid4().hex
        self._backend.put(token, {'code': code, 'type': captcha_type}, self._ttl)
        return token

    def verify(self, token, user_input, tolerance=5):
        entry = self._backend.pop(token)
        if entry is None:
            return False, 'token不存在或已过期'

        stored_code = entry['code']
        captcha_type = entry.get('type', 'text')

        if captcha_type == 'slide':
            try:
                stored_x = int(stored_code)
                input_x = int(user_input)
                if abs(stored_x - input_x) <= tolerance:
                    return True, '验证成功'
                return False, '验证码错误'
            except (ValueError, TypeError):
                return False, '验证码错误'

        if stored_code.lower() == user_input.lower():
            return True, '验证成功'
        return False, '验证码错误'

    def size(self):
        return self._backend.size()


_default_ttl = int(os.getenv('CAPTCHA_TTL', '300'))
captcha_cache = CaptchaCache(ttl=_default_ttl)
