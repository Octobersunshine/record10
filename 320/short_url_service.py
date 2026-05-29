import string
import threading
from datetime import datetime
from flask import Flask, request, redirect, jsonify

app = Flask(__name__)

url_mapping = {}
visit_logs = {}
counter = 0
counter_lock = threading.Lock()
stats_lock = threading.Lock()

BASE62_CHARS = string.ascii_letters + string.digits


def base62_encode(num):
    if num == 0:
        return BASE62_CHARS[0]
    result = []
    while num > 0:
        num, remainder = divmod(num, 62)
        result.append(BASE62_CHARS[remainder])
    return ''.join(reversed(result))


def generate_short_code():
    global counter
    with counter_lock:
        counter += 1
        current_id = counter
    code = base62_encode(current_id)
    return code


def is_valid_alias(alias):
    if not alias or len(alias) < 1 or len(alias) > 20:
        return False
    return all(c in BASE62_CHARS or c in '-_' for c in alias)


@app.route('/shorten', methods=['POST'])
def shorten_url():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'URL is required'}), 400

    long_url = data['url']
    custom_alias = data.get('alias', '').strip()

    if custom_alias:
        if not is_valid_alias(custom_alias):
            return jsonify({'error': 'Invalid alias. Use 1-20 chars: letters, digits, -, _'}), 400
        if custom_alias in url_mapping:
            return jsonify({'error': 'Alias already exists'}), 409
        short_code = custom_alias
    else:
        short_code = generate_short_code()

    url_mapping[short_code] = {
        'url': long_url,
        'created_at': datetime.now().isoformat(),
        'visit_count': 0
    }
    visit_logs[short_code] = []

    short_url = f'http://localhost:5000/{short_code}'
    return jsonify({
        'long_url': long_url,
        'short_url': short_url,
        'short_code': short_code,
        'created_at': url_mapping[short_code]['created_at']
    }), 201


@app.route('/<short_code>', methods=['GET'])
def redirect_to_long_url(short_code):
    if short_code not in url_mapping:
        return jsonify({'error': 'Short URL not found'}), 404

    mapping = url_mapping[short_code]
    long_url = mapping['url']
    client_ip = request.remote_addr
    user_agent = request.headers.get('User-Agent', '')
    referer = request.headers.get('Referer', '')

    with stats_lock:
        mapping['visit_count'] += 1
        visit_logs[short_code].append({
            'timestamp': datetime.now().isoformat(),
            'ip': client_ip,
            'user_agent': user_agent,
            'referer': referer
        })

    return redirect(long_url, code=302)


@app.route('/stats/<short_code>', methods=['GET'])
def get_stats(short_code):
    if short_code not in url_mapping:
        return jsonify({'error': 'Short URL not found'}), 404

    mapping = url_mapping[short_code]
    logs = visit_logs.get(short_code, [])

    ip_counts = {}
    for log in logs:
        ip = log['ip']
        ip_counts[ip] = ip_counts.get(ip, 0) + 1

    top_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    recent_visits = logs[-20:]

    return jsonify({
        'short_code': short_code,
        'long_url': mapping['url'],
        'created_at': mapping['created_at'],
        'total_visits': mapping['visit_count'],
        'unique_visitors': len(ip_counts),
        'top_ips': [{'ip': ip, 'count': count} for ip, count in top_ips],
        'recent_visits': recent_visits,
        'all_visits': logs
    }), 200


@app.route('/stats', methods=['GET'])
def get_all_stats():
    all_stats = []
    for code, mapping in url_mapping.items():
        logs = visit_logs.get(code, [])
        ip_counts = {}
        for log in logs:
            ip = log['ip']
            ip_counts[ip] = ip_counts.get(ip, 0) + 1

        all_stats.append({
            'short_code': code,
            'long_url': mapping['url'],
            'created_at': mapping['created_at'],
            'total_visits': mapping['visit_count'],
            'unique_visitors': len(ip_counts)
        })

    all_stats.sort(key=lambda x: x['total_visits'], reverse=True)

    return jsonify({
        'total_short_urls': len(url_mapping),
        'total_visits_all': sum(s['total_visits'] for s in all_stats),
        'rankings': all_stats
    }), 200


if __name__ == '__main__':
    app.run(debug=True, port=5000)
