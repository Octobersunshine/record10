import ipaddress
from datetime import datetime


def is_valid_ip(ip_str):
    try:
        ipaddress.ip_address(ip_str)
        return True, False
    except ValueError:
        pass

    try:
        ipaddress.ip_network(ip_str, strict=False)
        return True, True
    except ValueError:
        pass

    return False, False


def ip_matches(ip, rule_ip, is_cidr):
    try:
        target_ip = ipaddress.ip_address(ip)
    except ValueError:
        return False

    if is_cidr:
        try:
            network = ipaddress.ip_network(rule_ip, strict=False)
            return target_ip in network
        except ValueError:
            return False
    else:
        try:
            rule_ip_obj = ipaddress.ip_address(rule_ip)
            return target_ip == rule_ip_obj
        except ValueError:
            return False


def _rule_to_dict(rule):
    return {
        'id': rule.id,
        'ip_address': rule.ip_address,
        'list_type': rule.list_type,
        'is_cidr': rule.is_cidr,
        'description': rule.description,
        'is_temporary': getattr(rule, 'is_temporary', False),
        'expires_at': rule.expires_at.isoformat() if getattr(rule, 'expires_at', None) else None
    }


def _is_rule_expired(rule):
    if not hasattr(rule, 'expires_at') or rule.expires_at is None:
        return False
    return datetime.utcnow() > rule.expires_at


def check_ip_access(ip, all_rules, filter_expired=True):
    matched_blacklist = []
    matched_whitelist = []
    expired_blacklist = []

    for rule in all_rules:
        if ip_matches(ip, rule.ip_address, rule.is_cidr):
            if rule.list_type == 'blacklist':
                if filter_expired and _is_rule_expired(rule):
                    expired_blacklist.append(_rule_to_dict(rule))
                else:
                    matched_blacklist.append(_rule_to_dict(rule))
            else:
                matched_whitelist.append(_rule_to_dict(rule))

    has_conflict = len(matched_blacklist) > 0 and len(matched_whitelist) > 0

    if matched_blacklist:
        status = 'rejected'
        decided_by = 'blacklist'
        decided_rule = matched_blacklist[0]
        reason = f'IP {ip} is in blacklist (matched: {decided_rule["ip_address"]})'
        if has_conflict:
            reason += f' [CONFLICT: also matched {len(matched_whitelist)} whitelist rule(s), blacklist takes priority]'
    elif matched_whitelist:
        status = 'allowed'
        decided_by = 'whitelist'
        decided_rule = matched_whitelist[0]
        reason = f'IP {ip} is in whitelist (matched: {decided_rule["ip_address"]})'
    else:
        has_whitelist = any(r.list_type == 'whitelist' for r in all_rules if not _is_rule_expired(r))
        if has_whitelist:
            status = 'rejected'
            decided_by = 'default'
            decided_rule = None
            reason = f'IP {ip} is not in any whitelist (default deny)'
        else:
            status = 'allowed'
            decided_by = 'default'
            decided_rule = None
            reason = f'IP {ip} is allowed (no restriction configured)'

    return {
        'ip': ip,
        'status': status,
        'allowed': status == 'allowed',
        'decided_by': decided_by,
        'decided_rule': decided_rule,
        'reason': reason,
        'matched_blacklist': matched_blacklist,
        'matched_whitelist': matched_whitelist,
        'expired_blacklist': expired_blacklist,
        'has_conflict': has_conflict
    }


def find_conflicts(rules):
    active_rules = [r for r in rules if not _is_rule_expired(r)]
    conflicts = []
    blacklist_rules = [r for r in active_rules if r.list_type == 'blacklist']
    whitelist_rules = [r for r in active_rules if r.list_type == 'whitelist']

    for br in blacklist_rules:
        for wr in whitelist_rules:
            if _rules_overlap(br, wr):
                conflicts.append({
                    'blacklist_rule': _rule_to_dict(br),
                    'whitelist_rule': _rule_to_dict(wr),
                    'resolution': 'blacklist takes priority',
                    'blacklist_rule_id': br.id,
                    'whitelist_rule_id': wr.id
                })

    return conflicts


def _rules_overlap(rule_a, rule_b):
    if rule_a.is_cidr and rule_b.is_cidr:
        try:
            net_a = ipaddress.ip_network(rule_a.ip_address, strict=False)
            net_b = ipaddress.ip_network(rule_b.ip_address, strict=False)
            return net_a.overlaps(net_b)
        except ValueError:
            return False
    elif rule_a.is_cidr:
        try:
            net_a = ipaddress.ip_network(rule_a.ip_address, strict=False)
            ip_b = ipaddress.ip_address(rule_b.ip_address)
            return ip_b in net_a
        except ValueError:
            return False
    elif rule_b.is_cidr:
        try:
            net_b = ipaddress.ip_network(rule_b.ip_address, strict=False)
            ip_a = ipaddress.ip_address(rule_a.ip_address)
            return ip_a in net_b
        except ValueError:
            return False
    else:
        return rule_a.ip_address == rule_b.ip_address
