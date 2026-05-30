import threading
import time
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from models import db, IPList
from ip_utils import is_valid_ip_or_cidr, ip_in_list
from config import Config
from ban_manager import BanManager

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ip_list.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)


def _parse_datetime(datetime_str):
    if not datetime_str:
        return None
    try:
        if isinstance(datetime_str, (int, float)):
            return datetime.utcfromtimestamp(datetime_str)
        return datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
    except (ValueError, TypeError):
        return None


def _check_ip_logic(ip):
    all_entries = IPList.query.all()
    entries_dict = []
    for entry in all_entries:
        if entry.list_type == 'blacklist' and entry.is_expired():
            continue
        entries_dict.append(entry.to_dict())

    blacklist_entries = [e for e in entries_dict if e['list_type'] == 'blacklist']
    whitelist_entries = [e for e in entries_dict if e['list_type'] == 'whitelist']

    black_matches = ip_in_list(ip, blacklist_entries)
    white_matches = ip_in_list(ip, whitelist_entries)

    result = {
        'ip': ip,
        'matched_blacklist': [],
        'matched_whitelist': [],
        'has_conflict': False,
        'priority_rule': 'blacklist_first',
        'allowed': True,
        'final_decision': 'allow',
        'reason': 'no_match_default_allow'
    }

    for match in black_matches:
        result['matched_blacklist'].append({
            'id': match['id'],
            'ip_address': match['ip_address'],
            'is_cidr': match['is_cidr'],
            'description': match['description'],
            'expires_at': match.get('expires_at'),
            'is_autoban': match.get('is_autoban', False)
        })

    for match in white_matches:
        result['matched_whitelist'].append({
            'id': match['id'],
            'ip_address': match['ip_address'],
            'is_cidr': match['is_cidr'],
            'description': match['description'],
            'expires_at': match.get('expires_at'),
            'is_autoban': match.get('is_autoban', False)
        })

    if black_matches and white_matches:
        result['has_conflict'] = True
        result['allowed'] = False
        result['final_decision'] = 'deny'
        result['reason'] = 'conflict_blacklist_priority'
    elif black_matches:
        result['allowed'] = False
        result['final_decision'] = 'deny'
        result['reason'] = 'matched_blacklist'
    elif white_matches:
        result['allowed'] = True
        result['final_decision'] = 'allow'
        result['reason'] = 'matched_whitelist'

    return result


@app.route('/api/ip/check', methods=['GET'])
def check_ip():
    ip = request.args.get('ip')
    if not ip:
        return jsonify({'error': 'IP address is required'}), 400

    valid, _ = is_valid_ip_or_cidr(ip)
    if not valid:
        return jsonify({'error': 'Invalid IP address'}), 400

    result = _check_ip_logic(ip)

    response = {
        'ip': result['ip'],
        'allowed': result['allowed'],
        'reason': result['reason'],
        'matched_blacklist': result['matched_blacklist'],
        'matched_whitelist': result['matched_whitelist'],
        'has_conflict': result['has_conflict'],
        'priority_rule': result['priority_rule'],
        'final_decision': result['final_decision']
    }

    return jsonify(response), 403 if not result['allowed'] else 200


@app.route('/api/ip/verify', methods=['GET', 'POST'])
def verify_ip():
    if request.method == 'POST':
        data = request.get_json()
        if not data or 'ip' not in data:
            return jsonify({'error': 'ip is required in request body'}), 400
        ip = data['ip']
    else:
        ip = request.args.get('ip')
        if not ip:
            return jsonify({'error': 'IP address is required'}), 400

    valid, _ = is_valid_ip_or_cidr(ip)
    if not valid:
        return jsonify({'error': 'Invalid IP address'}), 400

    result = _check_ip_logic(ip)
    failure_count = BanManager.get_failure_count(ip)

    response = {
        'ip': result['ip'],
        'allowed': result['allowed'],
        'reason': result['reason'],
        'final_decision': result['final_decision'],
        'has_conflict': result['has_conflict'],
        'priority_rule': result['priority_rule'],
        'current_failure_count': failure_count,
        'matched_rules': {
            'blacklist': {
                'count': len(result['matched_blacklist']),
                'entries': result['matched_blacklist']
            },
            'whitelist': {
                'count': len(result['matched_whitelist']),
                'entries': result['matched_whitelist']
            }
        },
        'verification_details': {
            'is_blacklisted': len(result['matched_blacklist']) > 0,
            'is_whitelisted': len(result['matched_whitelist']) > 0,
            'conflict_resolved_by': 'blacklist_priority' if result['has_conflict'] else None
        }
    }

    return jsonify(response), 200


@app.route('/api/ip', methods=['POST'])
def create_ip_entry():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400

    ip_address = data.get('ip_address')
    list_type = data.get('list_type')
    description = data.get('description', '')
    duration_seconds = data.get('duration_seconds')
    expires_at = _parse_datetime(data.get('expires_at'))

    if not ip_address or not list_type:
        return jsonify({'error': 'ip_address and list_type are required'}), 400

    if list_type not in ['whitelist', 'blacklist']:
        return jsonify({'error': 'list_type must be whitelist or blacklist'}), 400

    valid, entry_type = is_valid_ip_or_cidr(ip_address)
    if not valid:
        return jsonify({'error': 'Invalid IP address or CIDR format'}), 400

    existing = IPList.query.filter_by(ip_address=ip_address, list_type=list_type).first()
    if existing:
        return jsonify({'error': 'IP/CIDR already exists in this list type'}), 409

    if duration_seconds and list_type == 'blacklist':
        expires_at = datetime.utcnow() + timedelta(seconds=int(duration_seconds))

    new_entry = IPList(
        ip_address=ip_address,
        list_type=list_type,
        is_cidr=(entry_type == 'cidr'),
        description=description,
        expires_at=expires_at,
        is_autoban=False
    )

    db.session.add(new_entry)

    if list_type == 'blacklist':
        from models import BanStats
        ban_stat = BanStats(
            ip_address=ip_address,
            failure_count=0,
            ban_duration_seconds=int(duration_seconds) if duration_seconds else None,
            is_automatic=False,
            is_active=True,
            reason=description or '手动封禁'
        )
        db.session.add(ban_stat)

    db.session.commit()

    return jsonify(new_entry.to_dict()), 201


@app.route('/api/ip', methods=['GET'])
def list_ip_entries():
    list_type = request.args.get('list_type')
    include_expired = request.args.get('include_expired', 'false').lower() == 'true'
    query = IPList.query

    if list_type:
        if list_type not in ['whitelist', 'blacklist']:
            return jsonify({'error': 'list_type must be whitelist or blacklist'}), 400
        query = query.filter_by(list_type=list_type)

    entries = query.order_by(IPList.created_at.desc()).all()

    if not include_expired:
        entries = [e for e in entries if not (e.list_type == 'blacklist' and e.is_expired())]

    return jsonify([entry.to_dict() for entry in entries]), 200


@app.route('/api/ip/<int:entry_id>', methods=['GET'])
def get_ip_entry(entry_id):
    entry = IPList.query.get(entry_id)
    if not entry:
        return jsonify({'error': 'Entry not found'}), 404
    return jsonify(entry.to_dict()), 200


@app.route('/api/ip/<int:entry_id>', methods=['PUT'])
def update_ip_entry(entry_id):
    entry = IPList.query.get(entry_id)
    if not entry:
        return jsonify({'error': 'Entry not found'}), 404

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400

    if 'ip_address' in data and data['ip_address'] != entry.ip_address:
        valid, entry_type = is_valid_ip_or_cidr(data['ip_address'])
        if not valid:
            return jsonify({'error': 'Invalid IP address or CIDR format'}), 400

        target_list_type = data.get('list_type', entry.list_type)
        existing = IPList.query.filter_by(ip_address=data['ip_address'], list_type=target_list_type).first()
        if existing and existing.id != entry_id:
            return jsonify({'error': 'IP/CIDR already exists in this list type'}), 409

        entry.ip_address = data['ip_address']
        entry.is_cidr = (entry_type == 'cidr')

    if 'list_type' in data:
        if data['list_type'] not in ['whitelist', 'blacklist']:
            return jsonify({'error': 'list_type must be whitelist or blacklist'}), 400
        entry.list_type = data['list_type']

    if 'description' in data:
        entry.description = data['description']

    if 'duration_seconds' in data and entry.list_type == 'blacklist':
        if data['duration_seconds'] is None:
            entry.expires_at = None
        else:
            entry.expires_at = datetime.utcnow() + timedelta(seconds=int(data['duration_seconds']))

    if 'expires_at' in data:
        entry.expires_at = _parse_datetime(data['expires_at'])

    db.session.commit()
    return jsonify(entry.to_dict()), 200


@app.route('/api/ip/<int:entry_id>', methods=['DELETE'])
def delete_ip_entry(entry_id):
    entry = IPList.query.get(entry_id)
    if not entry:
        return jsonify({'error': 'Entry not found'}), 404

    if entry.list_type == 'blacklist':
        from models import BanStats
        ban_stat = BanStats.query.filter_by(
            ip_address=entry.ip_address,
            is_active=True
        ).first()
        if ban_stat:
            ban_stat.is_active = False
            ban_stat.unban_time = datetime.utcnow()

    db.session.delete(entry)
    db.session.commit()

    BanManager._send_notification(
        'manual_unban',
        entry.ip_address,
        f"IP {entry.ip_address} 已手动解封"
    )

    return jsonify({'message': 'Entry deleted successfully'}), 200


@app.route('/api/failure', methods=['POST'])
def record_failure():
    data = request.get_json() or {}
    ip = data.get('ip') or request.remote_addr

    if not ip:
        return jsonify({'error': 'IP address is required'}), 400

    valid, _ = is_valid_ip_or_cidr(ip)
    if not valid:
        return jsonify({'error': 'Invalid IP address'}), 400

    result = BanManager.record_failure(
        ip_address=ip,
        failure_reason=data.get('reason'),
        request_path=data.get('path'),
        user_agent=data.get('user_agent')
    )

    current_count = BanManager.get_failure_count(ip)

    response = {
        'ip': ip,
        'recorded': True,
        'current_failure_count': current_count,
        'threshold': Config.MAX_FAILURE_ATTEMPTS,
        'window_seconds': Config.FAILURE_WINDOW_SECONDS
    }

    if result:
        response['auto_banned'] = True
        response['ban_details'] = result
    else:
        response['auto_banned'] = False

    return jsonify(response), 200


@app.route('/api/stats/ban', methods=['GET'])
def get_ban_stats():
    ip = request.args.get('ip')
    limit = int(request.args.get('limit', 100))

    stats = BanManager.get_ban_stats(ip_address=ip, limit=limit)
    return jsonify([s.to_dict() for s in stats]), 200


@app.route('/api/stats/ban/summary', methods=['GET'])
def get_ban_summary():
    summary = BanManager.get_ban_summary()
    return jsonify(summary), 200


@app.route('/api/bans/active', methods=['GET'])
def get_active_bans():
    bans = BanManager.get_active_bans()
    return jsonify([b.to_dict() for b in bans]), 200


@app.route('/api/cleanup', methods=['POST'])
def manual_cleanup():
    cleaned = BanManager.cleanup_expired_bans()
    return jsonify({
        'cleaned_count': cleaned,
        'message': f'Cleaned {cleaned} expired bans'
    }), 200


@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify(Config.get_config()), 200


@app.route('/api/config', methods=['PUT'])
def update_config():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400

    updated = Config.update_config(**data)
    return jsonify(updated), 200


@app.route('/api/notifications', methods=['GET'])
def get_notifications():
    unread_only = request.args.get('unread_only', 'false').lower() == 'true'
    limit = int(request.args.get('limit', 100))

    notifications = BanManager.get_notifications(unread_only=unread_only, limit=limit)
    return jsonify([n.to_dict() for n in notifications]), 200


@app.route('/api/notifications/<int:notification_id>/read', methods=['POST'])
def mark_notification_read(notification_id):
    success = BanManager.mark_notification_read(notification_id)
    if not success:
        return jsonify({'error': 'Notification not found'}), 404
    return jsonify({'message': 'Marked as read'}), 200


@app.route('/api/notifications/read-all', methods=['POST'])
def mark_all_notifications_read():
    BanManager.mark_all_notifications_read()
    return jsonify({'message': 'All notifications marked as read'}), 200


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'error': 'Method not allowed'}), 405


def start_cleanup_thread():
    def cleanup_loop():
        with app.app_context():
            while True:
                try:
                    cleaned = BanManager.cleanup_expired_bans()
                    if cleaned > 0:
                        print(f"[CLEANUP] Cleaned {cleaned} expired bans")
                except Exception as e:
                    print(f"[CLEANUP] Error: {e}")
                time.sleep(Config.CLEANUP_INTERVAL_SECONDS)

    thread = threading.Thread(target=cleanup_loop, daemon=True)
    thread.start()
    print(f"[CLEANUP] Started cleanup thread (interval: {Config.CLEANUP_INTERVAL_SECONDS}s)")


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    start_cleanup_thread()
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
