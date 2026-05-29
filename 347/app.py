from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from models import db, IPList, BanAttempt, BanLog
from ip_utils import is_valid_ip, check_ip_access, find_conflicts

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ip_list.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

AUTO_BAN_ENABLED = True
AUTO_BAN_THRESHOLD = 5
AUTO_BAN_WINDOW_SECONDS = 300
AUTO_BAN_DURATION_SECONDS = 3600

db.init_app(app)

with app.app_context():
    db.create_all()


def _cleanup_expired_bans():
    now = datetime.utcnow()
    expired = IPList.query.filter(
        IPList.list_type == 'blacklist',
        IPList.is_temporary == True,
        IPList.expires_at < now
    ).all()

    for ban in expired:
        log = BanLog(
            ip_address=ban.ip_address,
            event_type='unban_expired',
            reason='Temporary ban expired',
            source='auto_cleanup'
        )
        db.session.add(log)
        db.session.delete(ban)

    db.session.commit()
    return len(expired)


def _record_ban_log(ip, event_type, reason, source='api', **kwargs):
    log = BanLog(
        ip_address=ip,
        event_type=event_type,
        reason=reason,
        source=source,
        **kwargs
    )
    db.session.add(log)
    db.session.commit()
    return log


def _auto_ban_if_needed(ip, attempt):
    if not AUTO_BAN_ENABLED:
        return None

    now = datetime.utcnow()
    window_start = now - timedelta(seconds=AUTO_BAN_WINDOW_SECONDS)

    if attempt.window_start < window_start:
        attempt.failure_count = 1
        attempt.window_start = now
    else:
        attempt.failure_count += 1

    if attempt.failure_count >= AUTO_BAN_THRESHOLD:
        existing = IPList.query.filter_by(
            ip_address=ip,
            list_type='blacklist'
        ).first()

        if existing and not existing.is_expired():
            return None

        expires_at = now + timedelta(seconds=AUTO_BAN_DURATION_SECONDS)

        if existing:
            existing.expires_at = expires_at
            existing.is_temporary = True
            existing.description = f'Auto-banned: {AUTO_BAN_THRESHOLD} failures in {AUTO_BAN_WINDOW_SECONDS}s window'
            ban_entry = existing
        else:
            ban_entry = IPList(
                ip_address=ip,
                list_type='blacklist',
                is_cidr=False,
                is_temporary=True,
                expires_at=expires_at,
                description=f'Auto-banned: {AUTO_BAN_THRESHOLD} failures in {AUTO_BAN_WINDOW_SECONDS}s window'
            )
            db.session.add(ban_entry)

        _record_ban_log(
            ip, 'auto_ban',
            f'Automatically banned after {attempt.failure_count} failures',
            source='auto_ban',
            ban_duration_seconds=AUTO_BAN_DURATION_SECONDS,
            threshold=AUTO_BAN_THRESHOLD,
            failure_count=attempt.failure_count,
            expires_at=expires_at
        )

        attempt.failure_count = 0
        attempt.window_start = now

        return ban_entry

    return None


@app.route('/api/ip/check', methods=['POST'])
def check_ip():
    data = request.get_json()
    if not data or 'ip' not in data:
        return jsonify({'error': 'IP address is required'}), 400

    ip = data['ip']
    valid, _ = is_valid_ip(ip)
    if not valid:
        return jsonify({'error': 'Invalid IP address'}), 400

    _cleanup_expired_bans()
    all_rules = IPList.query.all()
    result = check_ip_access(ip, all_rules)

    return jsonify(result)


@app.route('/api/ip/check/batch', methods=['POST'])
def check_ip_batch():
    data = request.get_json()
    if not data or 'ips' not in data:
        return jsonify({'error': 'ips list is required'}), 400

    ips = data['ips']
    if not isinstance(ips, list):
        return jsonify({'error': 'ips must be a list'}), 400

    _cleanup_expired_bans()
    all_rules = IPList.query.all()
    results = []

    for ip in ips:
        valid, _ = is_valid_ip(ip)
        if not valid:
            results.append({
                'ip': ip,
                'status': 'error',
                'allowed': False,
                'decided_by': 'error',
                'decided_rule': None,
                'reason': 'Invalid IP address',
                'matched_blacklist': [],
                'matched_whitelist': [],
                'expired_blacklist': [],
                'has_conflict': False
            })
            continue

        results.append(check_ip_access(ip, all_rules))

    summary = {
        'total': len(results),
        'allowed': sum(1 for r in results if r.get('allowed')),
        'rejected': sum(1 for r in results if not r.get('allowed') and r.get('status') != 'error'),
        'errors': sum(1 for r in results if r.get('status') == 'error'),
        'conflicts': sum(1 for r in results if r.get('has_conflict')),
        'expired_matches': sum(len(r.get('expired_blacklist', [])) for r in results)
    }

    return jsonify({
        'results': results,
        'summary': summary
    })


@app.route('/api/ip/fail', methods=['POST'])
def record_failure():
    data = request.get_json()
    if not data or 'ip' not in data:
        return jsonify({'error': 'IP address is required'}), 400

    ip = data['ip']
    valid, is_cidr = is_valid_ip(ip)
    if not valid or is_cidr:
        return jsonify({'error': 'Invalid single IP address'}), 400

    reason = data.get('reason', 'Authentication failure')

    attempt = BanAttempt.query.filter_by(ip_address=ip).first()
    if not attempt:
        now = datetime.utcnow()
        attempt = BanAttempt(
            ip_address=ip,
            failure_count=0,
            window_start=now,
            first_attempt_at=now,
            last_attempt_at=now
        )
        db.session.add(attempt)

    ban_created = _auto_ban_if_needed(ip, attempt)
    db.session.commit()

    notification = None
    if ban_created:
        notification = {
            'level': 'warning',
            'type': 'auto_ban',
            'message': f'IP {ip} has been automatically banned',
            'ban': {
                'ip_address': ip,
                'expires_at': ban_created.expires_at.isoformat(),
                'duration_seconds': AUTO_BAN_DURATION_SECONDS,
                'failure_count': attempt.failure_count
            }
        }

    return jsonify({
        'ip': ip,
        'failure_count': attempt.failure_count,
        'threshold': AUTO_BAN_THRESHOLD,
        'window_seconds': AUTO_BAN_WINDOW_SECONDS,
        'auto_ban_triggered': ban_created is not None,
        'notification': notification
    })


@app.route('/api/ip/ban/stats', methods=['GET'])
def get_ban_stats():
    _cleanup_expired_bans()
    now = datetime.utcnow()

    total_bans = IPList.query.filter_by(list_type='blacklist').count()
    active_bans = IPList.query.filter(
        IPList.list_type == 'blacklist',
        (IPList.expires_at == None) | (IPList.expires_at > now)
    ).count()
    temp_bans = IPList.query.filter_by(
        list_type='blacklist',
        is_temporary=True
    ).count()
    permanent_bans = total_bans - temp_bans

    auto_bans_count = BanLog.query.filter_by(
        event_type='auto_ban'
    ).count()
    expired_unbans_count = BanLog.query.filter_by(
        event_type='unban_expired'
    ).count()

    last_24h = now - timedelta(hours=24)
    bans_24h = BanLog.query.filter(
        BanLog.event_type.in_(['auto_ban', 'manual_ban']),
        BanLog.created_at >= last_24h
    ).count()

    top_failing_ips = db.session.query(
        BanAttempt.ip_address,
        BanAttempt.failure_count
    ).order_by(
        BanAttempt.failure_count.desc()
    ).limit(10).all()

    return jsonify({
        'current': {
            'total_blacklist': total_bans,
            'active_bans': active_bans,
            'temporary_bans': temp_bans,
            'permanent_bans': permanent_bans
        },
        'auto_ban': {
            'enabled': AUTO_BAN_ENABLED,
            'threshold': AUTO_BAN_THRESHOLD,
            'window_seconds': AUTO_BAN_WINDOW_SECONDS,
            'ban_duration_seconds': AUTO_BAN_DURATION_SECONDS,
            'total_auto_banned': auto_bans_count,
            'bans_last_24h': bans_24h
        },
        'history': {
            'total_auto_bans': auto_bans_count,
            'total_expired_unbans': expired_unbans_count
        },
        'top_failing_ips': [
            {'ip': ip, 'failure_count': count}
            for ip, count in top_failing_ips
        ]
    })


@app.route('/api/ip/ban/logs', methods=['GET'])
def get_ban_logs():
    limit = request.args.get('limit', default=100, type=int)
    offset = request.args.get('offset', default=0, type=int)
    event_type = request.args.get('event_type', type=str)
    ip = request.args.get('ip', type=str)

    query = BanLog.query

    if event_type:
        query = query.filter_by(event_type=event_type)
    if ip:
        query = query.filter(BanLog.ip_address.like(f'%{ip}%'))

    total = query.count()
    logs = query.order_by(BanLog.created_at.desc()).offset(offset).limit(limit).all()

    return jsonify({
        'total': total,
        'limit': limit,
        'offset': offset,
        'logs': [log.to_dict() for log in logs]
    })


@app.route('/api/ip/ban/cleanup', methods=['POST'])
def cleanup_expired():
    count = _cleanup_expired_bans()
    return jsonify({
        'cleaned_count': count,
        'message': f'Cleaned up {count} expired ban(s)'
    })


@app.route('/api/ip/conflicts', methods=['GET'])
def get_conflicts():
    _cleanup_expired_bans()
    all_rules = IPList.query.all()
    conflicts = find_conflicts(all_rules)

    return jsonify({
        'total_conflicts': len(conflicts),
        'priority_rule': 'blacklist takes priority over whitelist',
        'conflicts': conflicts
    })


@app.route('/api/ip', methods=['POST'])
def create_ip_entry():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400

    if 'ip_address' not in data:
        return jsonify({'error': 'ip_address is required'}), 400

    if 'list_type' not in data:
        return jsonify({'error': 'list_type is required'}), 400

    ip_address = data['ip_address'].strip()
    list_type = data['list_type'].strip().lower()

    if list_type not in ['whitelist', 'blacklist']:
        return jsonify({'error': 'list_type must be whitelist or blacklist'}), 400

    valid, is_cidr = is_valid_ip(ip_address)
    if not valid:
        return jsonify({'error': 'Invalid IP address or CIDR'}), 400

    existing = IPList.query.filter_by(ip_address=ip_address, list_type=list_type).first()
    if existing:
        return jsonify({'error': f'IP {ip_address} already exists in {list_type}'}), 409

    description = data.get('description', '')
    expires_at = None
    is_temporary = False

    ttl_seconds = data.get('ttl_seconds')
    if ttl_seconds is not None:
        ttl_seconds = int(ttl_seconds)
    if ttl_seconds and list_type == 'blacklist':
        expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        is_temporary = True

    entry = IPList(
        ip_address=ip_address,
        list_type=list_type,
        is_cidr=is_cidr,
        description=description,
        expires_at=expires_at,
        is_temporary=is_temporary
    )

    db.session.add(entry)
    db.session.commit()

    if list_type == 'blacklist':
        _record_ban_log(
            ip_address, 'manual_ban',
            description or 'Manually added to blacklist',
            source='api',
            ban_duration_seconds=ttl_seconds,
            expires_at=expires_at
        )

    conflict_warning = None
    opposite_type = 'whitelist' if list_type == 'blacklist' else 'blacklist'
    opposite = IPList.query.filter_by(ip_address=ip_address, list_type=opposite_type).first()
    if opposite:
        conflict_warning = {
            'warning': f'IP {ip_address} exists in both blacklist and whitelist',
            'resolution': 'blacklist takes priority',
            'conflicting_rule': opposite.to_dict()
        }

    notification = None
    if list_type == 'blacklist':
        notification = {
            'level': 'info',
            'type': 'manual_ban',
            'message': f'IP {ip_address} added to blacklist',
            'expires_at': expires_at.isoformat() if expires_at else None
        }

    response = entry.to_dict()
    if conflict_warning:
        response['conflict_warning'] = conflict_warning
    if notification:
        response['notification'] = notification

    return jsonify(response), 201


@app.route('/api/ip', methods=['GET'])
def list_ip_entries():
    list_type = request.args.get('list_type', type=str)
    is_cidr = request.args.get('is_cidr', type=lambda v: v.lower() == 'true')
    is_temporary = request.args.get('is_temporary', type=lambda v: v.lower() == 'true')
    is_expired = request.args.get('is_expired', type=lambda v: v.lower() == 'true')

    _cleanup_expired_bans()
    query = IPList.query

    if list_type:
        list_type = list_type.lower()
        if list_type in ['whitelist', 'blacklist']:
            query = query.filter_by(list_type=list_type)

    if is_cidr is not None:
        query = query.filter_by(is_cidr=is_cidr)

    if is_temporary is not None:
        query = query.filter_by(is_temporary=is_temporary)

    if is_expired is not None:
        now = datetime.utcnow()
        if is_expired:
            query = query.filter(IPList.expires_at < now)
        else:
            query = query.filter((IPList.expires_at == None) | (IPList.expires_at >= now))

    entries = query.order_by(IPList.created_at.desc()).all()

    return jsonify([entry.to_dict() for entry in entries])


@app.route('/api/ip/<int:entry_id>', methods=['GET'])
def get_ip_entry(entry_id):
    entry = IPList.query.get(entry_id)
    if not entry:
        return jsonify({'error': 'Entry not found'}), 404

    return jsonify(entry.to_dict())


@app.route('/api/ip/<int:entry_id>', methods=['PUT'])
def update_ip_entry(entry_id):
    entry = IPList.query.get(entry_id)
    if not entry:
        return jsonify({'error': 'Entry not found'}), 404

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400

    if 'list_type' in data:
        list_type = data['list_type'].strip().lower()
        if list_type not in ['whitelist', 'blacklist']:
            return jsonify({'error': 'list_type must be whitelist or blacklist'}), 400
        entry.list_type = list_type

    if 'description' in data:
        entry.description = data['description']

    if 'ttl_seconds' in data and entry.list_type == 'blacklist':
        ttl_seconds = data['ttl_seconds']
        if ttl_seconds:
            entry.expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
            entry.is_temporary = True
        else:
            entry.expires_at = None
            entry.is_temporary = False

    if 'ip_address' in data:
        ip_address = data['ip_address'].strip()
        valid, is_cidr = is_valid_ip(ip_address)
        if not valid:
            return jsonify({'error': 'Invalid IP address or CIDR'}), 400

        existing = IPList.query.filter(
            IPList.ip_address == ip_address,
            IPList.list_type == entry.list_type,
            IPList.id != entry_id
        ).first()
        if existing:
            return jsonify({'error': f'IP {ip_address} already exists in {entry.list_type}'}), 409

        entry.ip_address = ip_address
        entry.is_cidr = is_cidr

    db.session.commit()

    return jsonify(entry.to_dict())


@app.route('/api/ip/<int:entry_id>', methods=['DELETE'])
def delete_ip_entry(entry_id):
    entry = IPList.query.get(entry_id)
    if not entry:
        return jsonify({'error': 'Entry not found'}), 404

    ip = entry.ip_address
    list_type = entry.list_type

    db.session.delete(entry)
    db.session.commit()

    if list_type == 'blacklist':
        _record_ban_log(
            ip, 'manual_unban',
            'Manually removed from blacklist',
            source='api'
        )

    return jsonify({'message': 'Entry deleted successfully'})


@app.route('/api/ip/batch', methods=['POST'])
def batch_create_entries():
    data = request.get_json()
    if not data or 'entries' not in data:
        return jsonify({'error': 'entries list is required'}), 400

    entries = data['entries']
    if not isinstance(entries, list):
        return jsonify({'error': 'entries must be a list'}), 400

    results = {'success': [], 'failed': [], 'conflict_warnings': [], 'notifications': []}

    for idx, item in enumerate(entries):
        if 'ip_address' not in item or 'list_type' not in item:
            results['failed'].append({
                'index': idx,
                'error': 'ip_address and list_type are required'
            })
            continue

        ip_address = item['ip_address'].strip()
        list_type = item['list_type'].strip().lower()

        if list_type not in ['whitelist', 'blacklist']:
            results['failed'].append({
                'index': idx,
                'ip': ip_address,
                'error': 'list_type must be whitelist or blacklist'
            })
            continue

        valid, is_cidr = is_valid_ip(ip_address)
        if not valid:
            results['failed'].append({
                'index': idx,
                'ip': ip_address,
                'error': 'Invalid IP address or CIDR'
            })
            continue

        existing = IPList.query.filter_by(ip_address=ip_address, list_type=list_type).first()
        if existing:
            results['failed'].append({
                'index': idx,
                'ip': ip_address,
                'error': f'IP {ip_address} already exists in {list_type}'
            })
            continue

        description = item.get('description', '')
        ttl_seconds = item.get('ttl_seconds')
        expires_at = None
        is_temporary = False

        if ttl_seconds and list_type == 'blacklist':
            expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
            is_temporary = True

        entry = IPList(
            ip_address=ip_address,
            list_type=list_type,
            is_cidr=is_cidr,
            description=description,
            expires_at=expires_at,
            is_temporary=is_temporary
        )

        db.session.add(entry)
        results['success'].append({
            'index': idx,
            'ip': ip_address,
            'list_type': list_type,
            'is_cidr': is_cidr,
            'expires_at': expires_at.isoformat() if expires_at else None
        })

        if list_type == 'blacklist':
            _record_ban_log(
                ip_address, 'manual_ban',
                description or 'Manually added to blacklist',
                source='batch_api',
                ban_duration_seconds=ttl_seconds,
                expires_at=expires_at
            )
            results['notifications'].append({
                'index': idx,
                'level': 'info',
                'type': 'manual_ban',
                'message': f'IP {ip_address} added to blacklist'
            })

        opposite_type = 'whitelist' if list_type == 'blacklist' else 'blacklist'
        opposite = IPList.query.filter_by(ip_address=ip_address, list_type=opposite_type).first()
        if opposite:
            results['conflict_warnings'].append({
                'index': idx,
                'ip': ip_address,
                'warning': f'IP {ip_address} exists in both blacklist and whitelist',
                'resolution': 'blacklist takes priority'
            })

    db.session.commit()

    return jsonify(results), 201


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
