import ipaddress
from typing import Tuple, Optional


def is_valid_ip(ip_str: str) -> bool:
    try:
        ipaddress.ip_address(ip_str)
        return True
    except ValueError:
        return False


def is_valid_cidr(cidr_str: str) -> bool:
    try:
        ipaddress.ip_network(cidr_str, strict=False)
        return '/' in cidr_str
    except ValueError:
        return False


def is_valid_ip_or_cidr(value: str) -> Tuple[bool, Optional[str]]:
    if is_valid_ip(value):
        return True, 'ip'
    elif is_valid_cidr(value):
        return True, 'cidr'
    return False, None


def ip_matches_cidr(ip_str: str, cidr_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
        network = ipaddress.ip_network(cidr_str, strict=False)
        return ip in network
    except ValueError:
        return False


def ip_in_list(ip_str: str, ip_entries: list) -> list:
    matches = []
    for entry in ip_entries:
        entry_ip = entry['ip_address']
        if entry['is_cidr']:
            if ip_matches_cidr(ip_str, entry_ip):
                matches.append(entry)
        else:
            if ip_str == entry_ip:
                matches.append(entry)
    return matches
