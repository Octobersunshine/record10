from flask import Flask, request, jsonify
from flask_cors import CORS
from ip_lookup import IPLookup
import os

app = Flask(__name__)
CORS(app)

ip_lookup = IPLookup()


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'city_db_loaded': ip_lookup.city_reader is not None,
        'asn_db_loaded': ip_lookup.asn_reader is not None,
        'cache': ip_lookup.cache_stats()
    })


@app.route('/api/ip/<ip>', methods=['GET'])
def lookup_ip(ip):
    result = ip_lookup.lookup(ip)
    if result['success']:
        status_code = 200
    elif result.get('error') == 'Invalid IP address':
        status_code = 400
    else:
        status_code = 500
    return jsonify(result), status_code


@app.route('/api/ip/batch', methods=['POST'])
def batch_lookup():
    data = request.get_json()
    if not data or 'ips' not in data:
        return jsonify({
            'success': False,
            'error': 'Missing "ips" field in request body'
        }), 400

    ips = data['ips']
    if not isinstance(ips, list):
        return jsonify({
            'success': False,
            'error': '"ips" must be an array'
        }), 400

    if len(ips) > 1000:
        return jsonify({
            'success': False,
            'error': 'Batch lookup limited to 1000 IPs per request'
        }), 400

    results = ip_lookup.batch_lookup(ips)
    return jsonify({
        'success': True,
        'total': len(results),
        'results': results
    })


@app.route('/api/ip', methods=['POST'])
def lookup_post():
    data = request.get_json()
    if not data or 'ip' not in data:
        return jsonify({
            'success': False,
            'error': 'Missing "ip" field in request body'
        }), 400

    result = ip_lookup.lookup(data['ip'])
    if result['success']:
        status_code = 200
    elif result.get('error') == 'Invalid IP address':
        status_code = 400
    else:
        status_code = 500
    return jsonify(result), status_code


@app.route('/api/cache/stats', methods=['GET'])
def cache_stats():
    return jsonify(ip_lookup.cache_stats())


@app.route('/api/cache/clear', methods=['POST'])
def cache_clear():
    ip_lookup.cache_clear()
    return jsonify({'success': True, 'message': 'Cache cleared'})


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    app.run(host=host, port=port, debug=debug)
