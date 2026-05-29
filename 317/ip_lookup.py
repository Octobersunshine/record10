import maxminddb
import os
import ipaddress
import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Union


CLOUD_PROVIDERS = {
    'amazon': {'name': 'AWS', 'type': 'cloud', 'keywords': ['amazon', 'aws', 'amazonaws', 'ec2', 's3', 'cloudfront']},
    'google': {'name': 'Google Cloud', 'type': 'cloud', 'keywords': ['google', 'gcp', 'googlecloud', 'google cloud']},
    'microsoft': {'name': 'Azure', 'type': 'cloud', 'keywords': ['microsoft', 'azure', 'microsoft azure']},
    'alibaba': {'name': 'Alibaba Cloud', 'type': 'cloud', 'keywords': ['alibaba', 'aliyun', 'alidns', 'alicloud']},
    'tencent': {'name': 'Tencent Cloud', 'type': 'cloud', 'keywords': ['tencent', 'tencent cloud']},
    'huawei': {'name': 'Huawei Cloud', 'type': 'cloud', 'keywords': ['huawei', 'huawei cloud', 'huawei cloud']},
    'oracle': {'name': 'Oracle Cloud', 'type': 'cloud', 'keywords': ['oracle', 'oracle cloud']},
    'digitalocean': {'name': 'DigitalOcean', 'type': 'cloud', 'keywords': ['digitalocean', 'digital ocean']},
    'linode': {'name': 'Linode (Akamai)', 'type': 'cloud', 'keywords': ['linode', 'akamai']},
    'vultr': {'name': 'Vultr', 'type': 'cloud', 'keywords': ['vultr']},
    'ibm': {'name': 'IBM Cloud', 'type': 'cloud', 'keywords': ['ibm', 'softlayer', 'ibm cloud']},
    'cloudflare': {'name': 'Cloudflare', 'type': 'cdn', 'keywords': ['cloudflare']},
    'akamai': {'name': 'Akamai', 'type': 'cdn', 'keywords': ['akamai', 'akamai technologies', 'linode']},
    'fastly': {'name': 'Fastly', 'type': 'cdn', 'keywords': ['fastly']},
    'cdnetworks': {'name': 'CDNetworks', 'type': 'cdn', 'keywords': ['cdnetworks']},
    'stackpath': {'name': 'StackPath', 'type': 'cdn', 'keywords': ['stackpath', 'highwinds']},
    'imperva': {'name': 'Imperva', 'type': 'cdn', 'keywords': ['imperva', 'incapsula']},
    'quadranet': {'name': 'QuadraNet', 'type': 'datacenter', 'keywords': ['quadranet']},
    'ovh': {'name': 'OVH', 'type': 'datacenter', 'keywords': ['ovh', 'ovh sas', 'ovhcloud']},
    'hetzner': {'name': 'Hetzner', 'type': 'datacenter', 'keywords': ['hetzner']},
    'leaseweb': {'name': 'LeaseWeb', 'type': 'datacenter', 'keywords': ['leaseweb']},
    'choopa': {'name': 'Choopa/Vultr', 'type': 'datacenter', 'keywords': ['choopa']},
    'cogent': {'name': 'Cogent', 'type': 'datacenter', 'keywords': ['cogent']},
    'zenlayer': {'name': 'Zenlayer', 'type': 'datacenter', 'keywords': ['zenlayer']},
}

CDN_ASN_SET = {
    13335,  # Cloudflare
    20940,  # Akamai
    54113,  # Fastly
    20446,  # Highwinds/StackPath
    19551,  # Incapsula/Imperva
    36692,  # CDNetworks
    16509,  # Amazon/AWS (CloudFront)
    14618,  # Amazon
    8075,   # Microsoft
    15169,  # Google
    37963,  # Alibaba
    45090,  # Tencent
    55990,  # Huawei
    7922,   # Comcast
}

DATACENTER_KEYWORDS = [
    'hosting', 'data center', 'datacenter', 'colocation', 'colocate',
    'server farm', 'serverfarm', 'idc', 'vps', 'dedicated server',
    'cloud computing', 'infrastructure', 'data centre',
]


class LRUCache:
    def __init__(self, maxsize: int = 1000):
        self._maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Dict]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, value: Dict):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                self._cache[key] = value
                if len(self._cache) > self._maxsize:
                    self._cache.popitem(last=False)

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def stats(self) -> Dict:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0
            return {
                'size': len(self._cache),
                'maxsize': self._maxsize,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': round(hit_rate, 2)
            }


class IPLookup:
    def __init__(self, city_db_path: str = None, asn_db_path: str = None, cache_size: int = 1000):
        self.city_db_path = city_db_path or os.getenv('GEOLITE2_CITY_DB', 'GeoLite2-City.mmdb')
        self.asn_db_path = asn_db_path or os.getenv('GEOLITE2_ASN_DB', 'GeoLite2-ASN.mmdb')
        self.city_reader = None
        self.asn_reader = None
        self._cache = LRUCache(maxsize=cache_size)
        self._init_readers()

    def _init_readers(self):
        if os.path.exists(self.city_db_path):
            self.city_reader = maxminddb.open_database(self.city_db_path)
        if os.path.exists(self.asn_db_path):
            self.asn_reader = maxminddb.open_database(self.asn_db_path)

    def _parse_ip(self, ip: str):
        try:
            addr = ipaddress.IPv4Address(ip)
            return addr, 'IPv4'
        except ipaddress.AddressValueError:
            pass
        try:
            addr = ipaddress.IPv6Address(ip)
            return addr, 'IPv6'
        except ipaddress.AddressValueError:
            pass
        return None, None

    def _is_private_ip(self, addr) -> bool:
        return addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved

    def _detect_attribution(self, isp_name: str, asn: int) -> Dict:
        attribution = {
            'is_cdn': False,
            'is_cloud': False,
            'is_datacenter': False,
            'provider': '',
            'network_type': 'residential'
        }

        if not isp_name:
            return attribution

        isp_lower = isp_name.lower()

        for provider_info in CLOUD_PROVIDERS.values():
            for keyword in provider_info['keywords']:
                if keyword in isp_lower:
                    attribution['provider'] = provider_info['name']
                    if provider_info['type'] == 'cdn':
                        attribution['is_cdn'] = True
                        attribution['network_type'] = 'cdn'
                    elif provider_info['type'] == 'cloud':
                        attribution['is_cloud'] = True
                        attribution['network_type'] = 'cloud'
                    elif provider_info['type'] == 'datacenter':
                        attribution['is_datacenter'] = True
                        attribution['network_type'] = 'datacenter'
                    return attribution

        if asn in CDN_ASN_SET:
            attribution['is_cdn'] = True
            attribution['network_type'] = 'cdn'
            return attribution

        for keyword in DATACENTER_KEYWORDS:
            if keyword in isp_lower:
                attribution['is_datacenter'] = True
                attribution['network_type'] = 'datacenter'
                attribution['provider'] = isp_name
                return attribution

        return attribution

    def _get_country(self, city_data: Dict) -> Dict:
        country_info = city_data.get('country', {})
        return {
            'code': country_info.get('iso_code', ''),
            'name': country_info.get('names', {}).get('zh-CN', country_info.get('names', {}).get('en', '')),
            'name_en': country_info.get('names', {}).get('en', '')
        }

    def _get_city(self, city_data: Dict) -> Dict:
        city_info = city_data.get('city', {})
        return {
            'name': city_info.get('names', {}).get('zh-CN', city_info.get('names', {}).get('en', '')),
            'name_en': city_info.get('names', {}).get('en', '')
        }

    def _get_subdivision(self, city_data: Dict) -> Dict:
        subdivisions = city_data.get('subdivisions', [])
        if subdivisions:
            sub = subdivisions[0]
            return {
                'code': sub.get('iso_code', ''),
                'name': sub.get('names', {}).get('zh-CN', sub.get('names', {}).get('en', '')),
                'name_en': sub.get('names', {}).get('en', '')
            }
        return {'code': '', 'name': '', 'name_en': ''}

    def _get_location(self, city_data: Dict) -> Dict:
        location = city_data.get('location', {})
        return {
            'latitude': location.get('latitude', 0.0),
            'longitude': location.get('longitude', 0.0),
            'time_zone': location.get('time_zone', ''),
            'accuracy_radius': location.get('accuracy_radius', 0)
        }

    def _get_isp(self, asn_data: Dict) -> Dict:
        if not asn_data:
            return {'asn': 0, 'isp': '', 'organization': ''}
        return {
            'asn': asn_data.get('autonomous_system_number', 0),
            'isp': asn_data.get('autonomous_system_organization', ''),
            'organization': asn_data.get('autonomous_system_organization', '')
        }

    def lookup(self, ip: str) -> Dict:
        cached = self._cache.get(ip)
        if cached is not None:
            return cached

        addr, ip_type = self._parse_ip(ip)

        if addr is None:
            result = {
                'ip': ip,
                'success': False,
                'error': 'Invalid IP address'
            }
            return result

        base_result = {
            'ip': ip,
            'ip_type': ip_type,
            'success': True,
            'country': {'code': '', 'name': '', 'name_en': ''},
            'city': {'name': '', 'name_en': ''},
            'subdivision': {'code': '', 'name': '', 'name_en': ''},
            'location': {'latitude': 0.0, 'longitude': 0.0, 'time_zone': '', 'accuracy_radius': 0},
            'isp': {'asn': 0, 'isp': '', 'organization': ''},
            'attribution': {
                'is_cdn': False,
                'is_cloud': False,
                'is_datacenter': False,
                'provider': '',
                'network_type': 'residential'
            }
        }

        if ip_type == 'IPv6':
            base_result['success'] = True
            base_result['message'] = 'IPv6 address detected, only address type is supported'
            self._cache.put(ip, base_result)
            return base_result

        if self._is_private_ip(addr):
            base_result['success'] = True
            base_result['message'] = '私有地址'
            base_result['attribution']['network_type'] = 'private'
            self._cache.put(ip, base_result)
            return base_result

        if self.city_reader:
            try:
                city_data = self.city_reader.get(ip)
                if city_data:
                    base_result['country'] = self._get_country(city_data)
                    base_result['city'] = self._get_city(city_data)
                    base_result['subdivision'] = self._get_subdivision(city_data)
                    base_result['location'] = self._get_location(city_data)
            except Exception as e:
                base_result['success'] = False
                base_result['error'] = f'City DB lookup error: {str(e)}'

        if self.asn_reader:
            try:
                asn_data = self.asn_reader.get(ip)
                base_result['isp'] = self._get_isp(asn_data)
                isp_name = base_result['isp'].get('isp', '')
                asn = base_result['isp'].get('asn', 0)
                base_result['attribution'] = self._detect_attribution(isp_name, asn)
            except Exception as e:
                if 'error' not in base_result:
                    base_result['success'] = False
                    base_result['error'] = f'ASN DB lookup error: {str(e)}'

        self._cache.put(ip, base_result)
        return base_result

    def batch_lookup(self, ips: List[str]) -> List[Dict]:
        results = []
        for ip in ips:
            results.append(self.lookup(ip))
        return results

    def cache_stats(self) -> Dict:
        return self._cache.stats

    def cache_clear(self):
        self._cache.clear()

    def close(self):
        if self.city_reader:
            self.city_reader.close()
        if self.asn_reader:
            self.asn_reader.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
