#!/usr/bin/env python3
import socket
import sys
import time
import random
import argparse
import threading
from collections import OrderedDict
try:
    import dns.resolver
    import dns.exception
    DNSPYTHON_AVAILABLE = True
except ImportError:
    DNSPYTHON_AVAILABLE = False

DEFAULT_TIMEOUT = 3
DEFAULT_RETRIES = 2
DEFAULT_TTL = 300
DEFAULT_CACHE_MAX = 1000
HOT_DOMAIN_THRESHOLD = 3
HOT_DOMAIN_TTL_MULTIPLIER = 2


def log_warning(msg):
    print(f"[WARNING] {msg}", file=sys.stderr)


def log_info(msg):
    print(f"[INFO] {msg}", file=sys.stderr)


class DNSCache:
    def __init__(self, max_size=DEFAULT_CACHE_MAX):
        self._cache = OrderedDict()
        self._access_count = {}
        self._lock = threading.Lock()
        self.max_size = max_size

    def _make_key(self, domain, record_type, dns_server):
        return (domain.lower(), record_type, dns_server or '')

    def get(self, domain, record_type, dns_server=None):
        key = self._make_key(domain, record_type, dns_server)
        with self._lock:
            if key not in self._cache:
                return None
            entry = self._cache[key]
            if time.time() > entry['expire_at']:
                del self._cache[key]
                self._access_count.pop(key, None)
                return None
            self._access_count[key] = self._access_count.get(key, 0) + 1
            self._cache.move_to_end(key)
            return entry

    def put(self, domain, record_type, ips, ttl, error, dns_server=None):
        key = self._make_key(domain, record_type, dns_server)
        with self._lock:
            access_count = self._access_count.get(key, 0)
            effective_ttl = ttl
            if access_count >= HOT_DOMAIN_THRESHOLD:
                effective_ttl = ttl * HOT_DOMAIN_TTL_MULTIPLIER
            expire_at = time.time() + effective_ttl
            entry = {
                'ips': ips,
                'ttl': ttl,
                'effective_ttl': effective_ttl,
                'expire_at': expire_at,
                'error': error,
                'cached_at': time.time(),
                'hot': access_count >= HOT_DOMAIN_THRESHOLD,
            }
            self._cache[key] = entry
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_size:
                oldest_key, _ = self._cache.popitem(last=False)
                self._access_count.pop(oldest_key, None)

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._access_count.clear()

    def stats(self):
        with self._lock:
            total = len(self._cache)
            hot = sum(1 for e in self._cache.values() if e['hot'])
            expired = sum(1 for e in self._cache.values() if time.time() > e['expire_at'])
            return {'total': total, 'hot': hot, 'expired': expired}

    def cleanup_expired(self):
        with self._lock:
            now = time.time()
            expired_keys = [k for k, v in self._cache.items() if now > v['expire_at']]
            for k in expired_keys:
                del self._cache[k]
                self._access_count.pop(k, None)
            return len(expired_keys)


_global_cache = DNSCache()


def resolve_with_socket(domain, record_type='A', timeout=DEFAULT_TIMEOUT):
    try:
        family = socket.AF_INET if record_type == 'A' else socket.AF_INET6
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(timeout)
        try:
            results = socket.getaddrinfo(domain, None, family, socket.SOCK_STREAM)
            ips = list(set(result[4][0] for result in results))
            return ips, None, DEFAULT_TTL
        finally:
            socket.setdefaulttimeout(old_timeout)
    except socket.timeout:
        msg = f"socket 解析 {domain} ({record_type}) 超时 ({timeout}s)"
        log_warning(msg)
        return [], msg, 0
    except socket.gaierror as e:
        msg = f"socket 解析 {domain} ({record_type}) 失败: {e}"
        log_warning(msg)
        return [], msg, 0


def resolve_with_dnspython(domain, record_type='A', dns_server=None,
                           timeout=DEFAULT_TIMEOUT, retries=DEFAULT_RETRIES):
    resolver = dns.resolver.Resolver()
    if dns_server:
        resolver.nameservers = [dns_server]
    resolver.timeout = timeout
    resolver.lifetime = timeout * (retries + 1)

    for attempt in range(1, retries + 1):
        try:
            answers = resolver.resolve(domain, record_type)
            ips = [str(rdata) for rdata in answers]
            ttl = answers.rrset.ttl if answers.rrset else DEFAULT_TTL
            return ips, None, ttl
        except dns.exception.Timeout:
            msg = f"dnspython 解析 {domain} ({record_type}) 超时 (尝试 {attempt}/{retries}, {timeout}s)"
            log_warning(msg)
        except dns.exception.DNSException as e:
            msg = f"dnspython 解析 {domain} ({record_type}) 失败: {e}"
            log_warning(msg)
            return [], msg, 0

    final_msg = f"dnspython 解析 {domain} ({record_type}) 在 {retries} 次重试后仍超时"
    log_warning(final_msg)
    return [], final_msg, 0


def apply_load_balance(ips, strategy='none'):
    if not ips or strategy == 'none':
        return ips
    if strategy == 'random':
        shuffled = ips[:]
        random.shuffle(shuffled)
        return shuffled
    if strategy == 'round-robin':
        return ips[1:] + ips[:1]
    return ips


def dns_resolve(domain, dns_server=None, use_socket=False,
                timeout=DEFAULT_TIMEOUT, retries=DEFAULT_RETRIES,
                cache=None, lb_strategy='none'):
    if cache is None:
        cache = _global_cache

    results = {
        'domain': domain,
        'dns_server': dns_server or 'system default',
        'A': [],
        'AAAA': [],
        'A_error': None,
        'AAAA_error': None,
        'A_ttl': None,
        'AAAA_ttl': None,
        'A_from_cache': False,
        'AAAA_from_cache': False,
    }

    for record_type in ('A', 'AAAA'):
        cached = cache.get(domain, record_type, dns_server)
        if cached is not None:
            ips = apply_load_balance(cached['ips'], lb_strategy)
            results[record_type] = ips
            results[f'{record_type}_error'] = cached['error']
            results[f'{record_type}_ttl'] = cached['effective_ttl']
            results[f'{record_type}_from_cache'] = True
            continue

        if use_socket or not DNSPYTHON_AVAILABLE:
            ips, error, ttl = resolve_with_socket(domain, record_type, timeout)
        else:
            ips, error, ttl = resolve_with_dnspython(
                domain, record_type, dns_server, timeout, retries)

        cache.put(domain, record_type, ips, ttl, error, dns_server)
        ips = apply_load_balance(ips, lb_strategy)
        results[record_type] = ips
        results[f'{record_type}_error'] = error
        results[f'{record_type}_ttl'] = ttl

    return results


def batch_resolve(domains, dns_server=None, use_socket=False,
                  timeout=DEFAULT_TIMEOUT, retries=DEFAULT_RETRIES,
                  cache=None, lb_strategy='none'):
    all_results = []
    for domain in domains:
        result = dns_resolve(domain, dns_server, use_socket,
                             timeout, retries, cache, lb_strategy)
        all_results.append(result)
    return all_results


def print_results(results):
    cache_tag = lambda from_cache: " (缓存)" if from_cache else ""
    ttl_tag = lambda ttl: f" [TTL: {ttl:.0f}s]" if ttl is not None else ""

    print(f"\nDNS 解析结果")
    print(f"域名: {results['domain']}")
    print(f"DNS 服务器: {results['dns_server']}")
    print("-" * 50)

    for record_type, label in [('A', 'A 记录 (IPv4)'), ('AAAA', 'AAAA 记录 (IPv6)')]:
        ips = results[record_type]
        error = results[f'{record_type}_error']
        ttl = results[f'{record_type}_ttl']
        from_cache = results[f'{record_type}_from_cache']
        print(f"\n{label}{cache_tag(from_cache)}{ttl_tag(ttl)}:")
        if ips:
            for ip in ips:
                print(f"  {ip}")
        elif error:
            print(f"  解析失败: {error}")
        else:
            print("  无记录")
    print()


def print_cache_stats(cache):
    stats = cache.stats()
    print(f"\n缓存统计")
    print("-" * 30)
    print(f"  总条目数: {stats['total']}")
    print(f"  热点域名: {stats['hot']}")
    print(f"  已过期:   {stats['expired']}")
    print()


def main():
    parser = argparse.ArgumentParser(description='DNS 解析工具 - 查询 A 记录和 AAAA 记录')
    parser.add_argument('domains', nargs='+', help='要解析的域名 (支持多个)')
    parser.add_argument('-s', '--server', help='指定 DNS 服务器 (如: 8.8.8.8)', default=None)
    parser.add_argument('--socket', action='store_true', help='使用 socket 库 (不支持自定义 DNS 服务器)')
    parser.add_argument('-t', '--timeout', type=float, default=DEFAULT_TIMEOUT,
                        help=f'超时时间，单位秒 (默认: {DEFAULT_TIMEOUT})')
    parser.add_argument('-r', '--retries', type=int, default=DEFAULT_RETRIES,
                        help=f'重试次数 (默认: {DEFAULT_RETRIES}, 仅 dnspython 模式)')
    parser.add_argument('--lb', choices=['none', 'random', 'round-robin'], default='none',
                        help='负载均衡策略: none=原始顺序, random=随机排序, round-robin=轮询 (默认: none)')
    parser.add_argument('--no-cache', action='store_true', help='禁用 DNS 缓存')
    parser.add_argument('--cache-stats', action='store_true', help='显示缓存统计信息')
    parser.add_argument('--cache-max', type=int, default=DEFAULT_CACHE_MAX,
                        help=f'缓存最大条目数 (默认: {DEFAULT_CACHE_MAX})')

    args = parser.parse_args()

    if args.socket and args.server:
        print("警告: 使用 socket 模式时无法指定自定义 DNS 服务器，将忽略 --server 参数")

    if not DNSPYTHON_AVAILABLE and not args.socket:
        print("提示: dnspython 未安装，将使用 socket 模式")
        print("安装 dnspython 以获得更多功能: pip install dnspython")
        args.socket = True

    cache = None if args.no_cache else DNSCache(max_size=args.cache_max)

    all_results = batch_resolve(
        args.domains, args.server, args.socket,
        args.timeout, args.retries, cache, args.lb)

    for results in all_results:
        print_results(results)

    if args.cache_stats and cache:
        print_cache_stats(cache)


if __name__ == '__main__':
    main()
