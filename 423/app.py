import json
from functools import wraps
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from rate_limiter import RateLimiter, REDIS_AVAILABLE

app = Flask(__name__)
CORS(app)

rate_limiter = RateLimiter(
    algorithm=RateLimiter.ALGORITHM_MULTI_DIMENSION,
    default_qps=10,
    window_size=1
)

EXCLUDE_PATHS = [
    '/api/limits',
    '/api/limits/',
    '/api/stats',
    '/api/stats/',
    '/api/algorithm',
    '/api/algorithm/',
    '/api/memory',
    '/api/memory/',
    '/api/reset',
    '/api/reset/',
    '/api/dimensions',
    '/api/dimensions/',
    '/api/dynamic',
    '/api/dynamic/',
    '/api/health',
    '/api/health/'
]


def rate_limit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        api_path = request.path
        method = request.method

        if any(api_path.startswith(p) for p in EXCLUDE_PATHS):
            return func(*args, **kwargs)

        key = f"{method}:{api_path}"
        allowed, info = rate_limiter.allow_request(key)

        headers = {
            'X-RateLimit-Limit': str(info['limit_qps']),
            'X-RateLimit-Remaining': str(info['remaining']),
            'X-RateLimit-Used': str(info['limit_qps'] - info['remaining']),
            'X-RateLimit-Algorithm': info['algorithm'],
        }

        if not allowed:
            headers['Retry-After'] = str(max(1, int(info['retry_after'])))
            error_response = {
                'error': 'Too Many Requests',
                'status_code': 429,
                'message': f"Rate limit exceeded for {key}. Limit: {info['limit_qps']} QPS",
                'details': {
                    'api_path': key,
                    'limit_qps': info['limit_qps'],
                    'current_qps': info['current_qps'],
                    'retry_after_seconds': max(1, int(info['retry_after'])),
                    'algorithm': info['algorithm']
                }
            }
            response = Response(
                response=json.dumps(error_response, ensure_ascii=False),
                status=429,
                mimetype='application/json'
            )
            for k, v in headers.items():
                response.headers[k] = v
            return response

        response = func(*args, **kwargs)
        if isinstance(response, tuple):
            resp, status = response[0], response[1]
            if isinstance(resp, dict):
                resp = jsonify(resp)
            for k, v in headers.items():
                resp.headers[k] = v
            return resp, status
        elif isinstance(response, dict):
            resp = jsonify(response)
            for k, v in headers.items():
                resp.headers[k] = v
            return resp
        else:
            for k, v in headers.items():
                response.headers[k] = v
            return response

    return wrapper


def _extract_client_ip():
    forwarded = request.headers.get('X-Forwarded-For', '')
    if forwarded:
        return forwarded.split(',')[0].strip()
    real_ip = request.headers.get('X-Real-IP', '')
    if real_ip:
        return real_ip.strip()
    return request.remote_addr or '0.0.0.0'


def _extract_user_id():
    user_id = request.headers.get('X-User-ID', '')
    if not user_id:
        user_id = request.args.get('user_id', '')
    if not user_id:
        auth = request.headers.get('Authorization', '')
        if auth.startswith('Bearer '):
            user_id = f"token:{auth[7:20]}"
    return user_id


@app.before_request
def before_request():
    api_path = request.path
    method = request.method

    if any(api_path.startswith(p) for p in EXCLUDE_PATHS):
        return None

    key = f"{method}:{api_path}"
    user_id = _extract_user_id()
    ip = _extract_client_ip()

    allowed, info = rate_limiter.allow_request(key, user_id=user_id, ip=ip)

    request.rate_limit_info = info

    if not allowed:
        headers = {
            'X-RateLimit-Limit': str(info.get('limit_qps', rate_limiter.default_qps)),
            'X-RateLimit-Remaining': '0',
            'X-RateLimit-Used': str(info.get('limit_qps', rate_limiter.default_qps)),
            'X-RateLimit-Algorithm': info.get('algorithm', ''),
            'Retry-After': str(max(1, int(info.get('retry_after', 1))))
        }
        if info.get('blocked_by'):
            headers['X-RateLimit-BlockedBy'] = info['blocked_by']
        if info.get('dynamic_adjusted'):
            headers['X-RateLimit-Dynamic'] = 'true'
            headers['X-RateLimit-Effective-QPS'] = str(info.get('dynamic_effective_qps', ''))

        error_response = {
            'error': 'Too Many Requests',
            'status_code': 429,
            'message': f"Rate limit exceeded for {key}. Limit: {info.get('limit_qps', '?')} QPS",
            'details': {
                'api_path': key,
                'limit_qps': info.get('limit_qps'),
                'current_qps': info.get('current_qps'),
                'retry_after_seconds': max(1, int(info.get('retry_after', 1))),
                'algorithm': info.get('algorithm'),
                'blocked_by': info.get('blocked_by'),
                'user_id': user_id if user_id else None,
                'ip': ip
            }
        }
        if info.get('dynamic_adjusted'):
            error_response['details']['dynamic_adjusted'] = True
            error_response['details']['dynamic_effective_qps'] = info.get('dynamic_effective_qps')

        response = Response(
            response=json.dumps(error_response, ensure_ascii=False),
            status=429,
            mimetype='application/json'
        )
        for k, v in headers.items():
            response.headers[k] = v
        return response
    return None


@app.after_request
def after_request(response):
    if hasattr(request, 'rate_limit_info'):
        info = request.rate_limit_info
        response.headers['X-RateLimit-Limit'] = str(info['limit_qps'])
        response.headers['X-RateLimit-Remaining'] = str(info['remaining'])
        response.headers['X-RateLimit-Used'] = str(info['limit_qps'] - info['remaining'])
        response.headers['X-RateLimit-Algorithm'] = info['algorithm']
    return response


@app.route('/api/limits', methods=['GET'])
def get_limits():
    limits = rate_limiter.get_all_limits()
    return {
        'success': True,
        'data': {
            'algorithm': rate_limiter.algorithm,
            'default_qps': rate_limiter.default_qps,
            'limits': limits,
            'available_algorithms': rate_limiter.available_algorithms
        }
    }


@app.route('/api/limits/<path:api_path>', methods=['GET'])
def get_limit(api_path):
    if not api_path.startswith('/'):
        api_path = '/' + api_path
    method = request.args.get('method', 'GET')
    key = f"{method}:{api_path}"
    limit = rate_limiter.get_limit(key)
    stats = rate_limiter.get_stats(key)
    return {
        'success': True,
        'data': {
            'api_path': key,
            'limit': limit,
            'stats': stats
        }
    }


@app.route('/api/limits', methods=['POST'])
def set_limit():
    data = request.get_json()
    if not data:
        return {'success': False, 'error': 'Request body is required'}, 400

    api_path = data.get('api_path')
    qps = data.get('qps')
    method = data.get('method', 'GET')
    burst = data.get('burst')

    if not api_path:
        return {'success': False, 'error': 'api_path is required'}, 400
    if qps is None:
        return {'success': False, 'error': 'qps is required'}, 400
    if not isinstance(qps, int) or qps < 0:
        return {'success': False, 'error': 'qps must be a non-negative integer'}, 400

    key = f"{method}:{api_path}"

    kwargs = {}
    if burst is not None:
        if not isinstance(burst, int) or burst < 0:
            return {'success': False, 'error': 'burst must be a non-negative integer'}, 400
        kwargs['burst'] = burst

    rate_limiter.set_limit(key, qps, **kwargs)

    return {
        'success': True,
        'data': {
            'api_path': key,
            'qps': qps,
            'burst': burst or qps,
            'limit': rate_limiter.get_limit(key)
        }
    }, 201


@app.route('/api/limits/<path:api_path>', methods=['DELETE'])
def remove_limit(api_path):
    if not api_path.startswith('/'):
        api_path = '/' + api_path
    method = request.args.get('method', 'GET')
    key = f"{method}:{api_path}"
    rate_limiter.remove_limit(key)
    return {
        'success': True,
        'data': {
            'api_path': key,
            'message': 'Limit removed'
        }
    }


@app.route('/api/limits', methods=['PUT'])
def update_limit():
    data = request.get_json()
    if not data:
        return {'success': False, 'error': 'Request body is required'}, 400

    api_path = data.get('api_path')
    qps = data.get('qps')
    method = data.get('method', 'GET')
    burst = data.get('burst')

    if not api_path:
        return {'success': False, 'error': 'api_path is required'}, 400
    if qps is None:
        return {'success': False, 'error': 'qps is required'}, 400
    if not isinstance(qps, int) or qps < 0:
        return {'success': False, 'error': 'qps must be a non-negative integer'}, 400

    key = f"{method}:{api_path}"

    kwargs = {}
    if burst is not None:
        if not isinstance(burst, int) or burst < 0:
            return {'success': False, 'error': 'burst must be a non-negative integer'}, 400
        kwargs['burst'] = burst

    rate_limiter.set_limit(key, qps, **kwargs)

    return {
        'success': True,
        'data': {
            'api_path': key,
            'qps': qps,
            'burst': burst or qps,
            'limit': rate_limiter.get_limit(key)
        }
    }


@app.route('/api/stats', methods=['GET'])
def get_all_stats():
    stats = {}
    for key in rate_limiter.get_all_limits().keys():
        stats[key] = rate_limiter.get_stats(key)
    return {
        'success': True,
        'data': {
            'algorithm': rate_limiter.algorithm,
            'stats': stats
        }
    }


@app.route('/api/stats/<path:api_path>', methods=['GET'])
def get_api_stats(api_path):
    if not api_path.startswith('/'):
        api_path = '/' + api_path
    method = request.args.get('method', 'GET')
    key = f"{method}:{api_path}"
    stats = rate_limiter.get_stats(key)
    return {
        'success': True,
        'data': stats
    }


@app.route('/api/algorithm', methods=['GET'])
def get_algorithm():
    return {
        'success': True,
        'data': {
            'current_algorithm': rate_limiter.algorithm,
            'available_algorithms': rate_limiter.available_algorithms,
            'default_qps': rate_limiter.default_qps
        }
    }


@app.route('/api/algorithm', methods=['PUT'])
def switch_algorithm():
    data = request.get_json()
    if not data:
        return {'success': False, 'error': 'Request body is required'}, 400

    algorithm = data.get('algorithm')
    if not algorithm:
        return {'success': False, 'error': 'algorithm is required'}, 400
    if algorithm not in rate_limiter.available_algorithms:
        return {
            'success': False,
            'error': f'Invalid algorithm. Must be one of: {rate_limiter.available_algorithms}'
        }, 400

    kwargs = {}

    if algorithm in [RateLimiter.ALGORITHM_SLIDING_WINDOW,
                     RateLimiter.ALGORITHM_SLIDING_WINDOW_CIRCULAR,
                     RateLimiter.ALGORITHM_SLIDING_WINDOW_REDIS]:
        window_size = data.get('window_size')
        if window_size is not None:
            kwargs['window_size'] = window_size

    if algorithm == RateLimiter.ALGORITHM_SLIDING_WINDOW_REDIS:
        redis_host = data.get('redis_host')
        redis_port = data.get('redis_port')
        redis_db = data.get('redis_db')
        redis_password = data.get('redis_password')
        redis_prefix = data.get('redis_prefix')

        if redis_host:
            kwargs['redis_host'] = redis_host
        if redis_port is not None:
            kwargs['redis_port'] = redis_port
        if redis_db is not None:
            kwargs['redis_db'] = redis_db
        if redis_password:
            kwargs['redis_password'] = redis_password
        if redis_prefix:
            kwargs['redis_prefix'] = redis_prefix

    elif algorithm == RateLimiter.ALGORITHM_TOKEN_BUCKET:
        burst = data.get('burst')
        if burst is not None:
            kwargs['burst'] = burst

    try:
        rate_limiter.switch_algorithm(algorithm, **kwargs)
    except Exception as e:
        return {'success': False, 'error': str(e)}, 400

    memory_info = rate_limiter.get_memory_info()

    return {
        'success': True,
        'data': {
            'current_algorithm': rate_limiter.algorithm,
            'default_qps': rate_limiter.default_qps,
            'limits': rate_limiter.get_all_limits(),
            'memory_info': memory_info
        }
    }


@app.route('/api/memory', methods=['GET'])
def get_memory_info():
    memory_info = rate_limiter.get_memory_info()
    return {
        'success': True,
        'data': {
            'algorithm': rate_limiter.algorithm,
            'memory_info': memory_info,
            'algorithm_features': {
                'sliding_window': {
                    'description': 'Original sliding window using deque',
                    'memory_fixed': False,
                    'note': 'Memory grows with request count. Not recommended for high concurrency.'
                },
                'sliding_window_circular': {
                    'description': 'Optimized sliding window using fixed-size circular array',
                    'memory_fixed': True,
                    'note': 'O(1) memory usage, no growth with request count. Recommended for high concurrency.'
                },
                'sliding_window_redis': {
                    'description': 'Distributed sliding window using Redis sorted set',
                    'memory_fixed': True,
                    'distributed': True,
                    'note': 'Suitable for distributed systems. Requires Redis.'
                },
                'token_bucket': {
                    'description': 'Token bucket algorithm',
                    'memory_fixed': True,
                    'note': 'Supports burst traffic. Fixed memory per API.'
                }
            }
        }
    }


@app.route('/api/reset', methods=['POST'])
def reset_stats():
    data = request.get_json(silent=True) or {}
    api_path = data.get('api_path')
    method = data.get('method', 'GET')

    if api_path:
        key = f"{method}:{api_path}"
        rate_limiter.reset(key)
        message = f"Stats reset for {key}"
    else:
        rate_limiter.reset()
        message = "All stats reset"

    return {
        'success': True,
        'data': {
            'message': message
        }
    }


@app.route('/api/dimensions', methods=['GET'])
def get_dimension_rules():
    rules = rate_limiter.get_all_dimension_rules()
    return {
        'success': True,
        'data': {
            'algorithm': rate_limiter.algorithm,
            'dimension_rules': rules,
            'available_dimensions': [
                {'name': 'api', 'description': 'Rate limit by API path only'},
                {'name': 'user', 'description': 'Rate limit by user ID only'},
                {'name': 'ip', 'description': 'Rate limit by client IP only'},
                {'name': 'api:user', 'description': 'Rate limit by API + User combination'},
                {'name': 'api:ip', 'description': 'Rate limit by API + IP combination'},
                {'name': 'api:user:ip', 'description': 'Rate limit by API + User + IP combination'},
            ]
        }
    }


@app.route('/api/dimensions', methods=['POST'])
def set_dimension_rule():
    data = request.get_json()
    if not data:
        return {'success': False, 'error': 'Request body is required'}, 400

    dimension = data.get('dimension')
    qps = data.get('qps')
    api_pattern = data.get('api_pattern')
    burst = data.get('burst')
    strategy = data.get('strategy', 'sliding_window')

    if not dimension:
        return {'success': False, 'error': 'dimension is required'}, 400
    if qps is None:
        return {'success': False, 'error': 'qps is required'}, 400
    if not isinstance(qps, int) or qps < 0:
        return {'success': False, 'error': 'qps must be a non-negative integer'}, 400

    try:
        kwargs = {'api_pattern': api_pattern}
        if burst is not None:
            kwargs['burst'] = burst
        if strategy:
            kwargs['strategy'] = strategy
        rate_limiter.set_dimension_rule(dimension, qps, **kwargs)
    except ValueError as e:
        return {'success': False, 'error': str(e)}, 400

    return {
        'success': True,
        'data': {
            'dimension': dimension,
            'qps': qps,
            'api_pattern': api_pattern,
            'burst': burst,
            'strategy': strategy
        }
    }, 201


@app.route('/api/dimensions', methods=['DELETE'])
def remove_dimension_rule():
    data = request.get_json(silent=True) or {}
    dimension = data.get('dimension')
    api_pattern = data.get('api_pattern')

    if not dimension:
        return {'success': False, 'error': 'dimension is required'}, 400

    try:
        rate_limiter.remove_dimension_rule(dimension, api_pattern=api_pattern)
    except ValueError as e:
        return {'success': False, 'error': str(e)}, 400

    return {
        'success': True,
        'data': {
            'dimension': dimension,
            'api_pattern': api_pattern,
            'message': 'Dimension rule removed'
        }
    }


@app.route('/api/dynamic', methods=['GET'])
def get_dynamic_status():
    status = rate_limiter.get_dynamic_status()
    return {
        'success': True,
        'data': status
    }


@app.route('/api/dynamic', methods=['POST'])
def enable_dynamic_throttle():
    data = request.get_json(silent=True) or {}
    enabled = data.get('enabled', True)

    kwargs = {}
    for key in ['base_qps', 'check_interval', 'min_qps', 'max_qps',
                'cooldown_seconds', 'multipliers']:
        if key in data:
            kwargs[key] = data[key]

    status = rate_limiter.set_dynamic_throttle(enabled=enabled, **kwargs)
    return {
        'success': True,
        'data': status
    }


@app.route('/api/dynamic/adjust', methods=['POST'])
def force_dynamic_adjust():
    data = request.get_json()
    if not data:
        return {'success': False, 'error': 'Request body is required'}, 400

    multiplier = data.get('multiplier')
    level = data.get('level')

    if multiplier is None:
        return {'success': False, 'error': 'multiplier is required'}, 400

    result = rate_limiter.force_dynamic_adjust(multiplier, level)
    return {
        'success': True,
        'data': result
    }


@app.route('/api/users', methods=['GET'])
@rate_limit
def get_users():
    return {
        'success': True,
        'data': [
            {'id': 1, 'name': 'Alice'},
            {'id': 2, 'name': 'Bob'},
            {'id': 3, 'name': 'Charlie'}
        ]
    }


@app.route('/api/users', methods=['POST'])
@rate_limit
def create_user():
    data = request.get_json() or {}
    return {
        'success': True,
        'data': {
            'id': 4,
            'name': data.get('name', 'New User')
        }
    }, 201


@app.route('/api/orders', methods=['GET'])
@rate_limit
def get_orders():
    return {
        'success': True,
        'data': [
            {'id': 101, 'user_id': 1, 'amount': 99.99},
            {'id': 102, 'user_id': 2, 'amount': 199.99}
        ]
    }


@app.route('/api/health', methods=['GET'])
def health_check():
    return {
        'success': True,
        'data': {
            'status': 'healthy',
            'algorithm': rate_limiter.algorithm,
            'default_qps': rate_limiter.default_qps
        }
    }


@app.errorhandler(404)
def not_found(error):
    return {
        'success': False,
        'error': 'Not Found',
        'status_code': 404
    }, 404


@app.errorhandler(405)
def method_not_allowed(error):
    return {
        'success': False,
        'error': 'Method Not Allowed',
        'status_code': 405
    }, 405


@app.errorhandler(500)
def internal_error(error):
    return {
        'success': False,
        'error': 'Internal Server Error',
        'status_code': 500
    }, 500


if __name__ == '__main__':
    memory_info = rate_limiter.get_memory_info()
    dynamic_status = rate_limiter.get_dynamic_status()

    print("=" * 70)
    print("API Rate Limiter Server (Multi-Dimension + Dynamic Throttle)")
    print("=" * 70)
    print(f"Algorithm: {rate_limiter.algorithm}")
    print(f"Default QPS: {rate_limiter.default_qps}")
    print(f"Dynamic Throttle: {'ON' if dynamic_status.get('enabled') else 'OFF'}")
    print(f"Server: http://127.0.0.1:5000")
    print("=" * 70)
    print("\nAvailable Algorithms:")
    for algo in rate_limiter.available_algorithms:
        marker = "  ✓ DEFAULT" if algo == rate_limiter.algorithm else "   "
        if algo == 'multi_dimension':
            print(f"  {algo:<30} {marker} [Multi-Dim]")
        elif algo == 'distributed_redis':
            print(f"  {algo:<30} {marker} [Distributed+Multi-Dim, Redis]")
        elif algo == 'sliding_window_circular':
            print(f"  {algo:<30} {marker} [Fixed Memory]")
        elif algo == 'sliding_window_redis':
            print(f"  {algo:<30} {marker} [Distributed, Redis]")
        elif algo == 'token_bucket':
            print(f"  {algo:<30} {marker} [Burst Support]")
        else:
            print(f"  {algo:<30} {marker} [Basic]")
    print("=" * 70)
    print("\nAvailable API Endpoints:")
    print("  GET  /api/health          - Health check")
    print("  GET  /api/users           - Get users (multi-dim rate limited)")
    print("  POST /api/users           - Create user (multi-dim rate limited)")
    print("  GET  /api/orders          - Get orders (multi-dim rate limited)")
    print("\nRate Limit Configuration:")
    print("  GET    /api/limits        - Get all limits")
    print("  POST   /api/limits        - Set limit for API")
    print("  PUT    /api/limits        - Update limit")
    print("  DELETE /api/limits/<path> - Remove limit")
    print("\nMulti-Dimension Rules:")
    print("  GET    /api/dimensions    - Get dimension rules")
    print("  POST   /api/dimensions    - Set dimension rule")
    print("  DELETE /api/dimensions    - Remove dimension rule")
    print("\nDynamic Throttle:")
    print("  GET    /api/dynamic       - Get dynamic throttle status")
    print("  POST   /api/dynamic       - Enable/disable dynamic throttle")
    print("  POST   /api/dynamic/adjust- Force adjust multiplier")
    print("\nSystem:")
    print("  GET    /api/stats         - Get all stats")
    print("  GET    /api/algorithm     - Get/set algorithm")
    print("  GET    /api/memory        - Get memory info")
    print("  POST   /api/reset         - Reset stats")
    print("=" * 70)
    print("\nMulti-Dimension Rate Limiting:")
    print("  Dimensions: api, user, ip, api:user, api:ip, api:user:ip")
    print("  User ID: X-User-ID header or user_id query param")
    print("  Client IP: X-Forwarded-For > X-Real-IP > remote_addr")
    print("=" * 70)
    app.run(host='0.0.0.0', port=5000, debug=False)
