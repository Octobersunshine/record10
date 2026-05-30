import hashlib
import requests
import sqlite3
import os
import time
from typing import List, Dict, Optional, Tuple


class PasswordBreachChecker:
    def __init__(self, db_path: str = "breach_data.db"):
        self.db_path = db_path
        self.api_url = "https://api.pwnedpasswords.com/range/"
        self.user_agent = "PasswordStrengthChecker/1.0"
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS breached_passwords (
                hash_prefix TEXT,
                hash_suffix TEXT,
                full_hash TEXT PRIMARY KEY,
                breach_count INTEGER,
                last_checked INTEGER,
                last_breach_date TEXT
            )
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_hash_prefix 
            ON breached_passwords(hash_prefix)
        ''')
        conn.commit()
        conn.close()

    @staticmethod
    def _get_sha1_hash(password: str) -> str:
        return hashlib.sha1(password.encode('utf-8')).hexdigest().upper()

    def _check_api(self, hash_prefix: str) -> List[Tuple[str, int]]:
        try:
            url = f"{self.api_url}{hash_prefix}"
            headers = {'User-Agent': self.user_agent}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                return []
            
            results = []
            for line in response.text.splitlines():
                if ':' in line:
                    suffix, count = line.split(':')
                    results.append((suffix.strip(), int(count)))
            
            return results
        except (requests.RequestException, ValueError):
            return []

    def _save_to_db(self, hash_prefix: str, results: List[Tuple[str, int]]):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        current_time = int(time.time())
        
        for suffix, count in results:
            full_hash = f"{hash_prefix}{suffix}"
            cursor.execute('''
                INSERT OR REPLACE INTO breached_passwords 
                (hash_prefix, hash_suffix, full_hash, breach_count, last_checked)
                VALUES (?, ?, ?, ?, ?)
            ''', (hash_prefix, suffix, full_hash, count, current_time))
        
        conn.commit()
        conn.close()

    def _query_db(self, full_hash: str) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT breach_count, last_checked, last_breach_date
            FROM breached_passwords 
            WHERE full_hash = ?
        ''', (full_hash,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'breach_count': result[0],
                'last_checked': result[1],
                'last_breach_date': result[2]
            }
        return None

    def check_password(self, password: str, use_cache: bool = True) -> Dict:
        full_hash = self._get_sha1_hash(password)
        hash_prefix = full_hash[:5]
        hash_suffix = full_hash[5:]
        
        if use_cache:
            cached = self._query_db(full_hash)
            if cached is not None:
                return {
                    'password': password,
                    'full_hash': full_hash,
                    'breached': cached['breach_count'] > 0,
                    'breach_count': cached['breach_count'],
                    'last_checked': cached['last_checked'],
                    'last_breach_date': cached.get('last_breach_date'),
                    'source': 'cache'
                }
        
        api_results = self._check_api(hash_prefix)
        
        if api_results:
            self._save_to_db(hash_prefix, api_results)
        
        breach_count = 0
        for suffix, count in api_results:
            if suffix == hash_suffix:
                breach_count = count
                break
        
        return {
            'password': password,
            'full_hash': full_hash,
            'breached': breach_count > 0,
            'breach_count': breach_count,
            'last_checked': int(time.time()),
            'last_breach_date': None,
            'source': 'api' if api_results else 'offline'
        }

    def check_passwords_batch(self, passwords: List[str], use_cache: bool = True) -> List[Dict]:
        results = []
        hash_to_password = {}
        uncached_hashes = set()
        
        for password in passwords:
            full_hash = self._get_sha1_hash(password)
            hash_to_password[full_hash] = password
            
            if use_cache:
                cached = self._query_db(full_hash)
                if cached is not None:
                    results.append({
                        'password': password,
                        'full_hash': full_hash,
                        'breached': cached['breach_count'] > 0,
                        'breach_count': cached['breach_count'],
                        'last_checked': cached['last_checked'],
                        'last_breach_date': cached.get('last_breach_date'),
                        'source': 'cache'
                    })
                    continue
            
            uncached_hashes.add(full_hash)
        
        prefix_groups = {}
        for full_hash in uncached_hashes:
            prefix = full_hash[:5]
            if prefix not in prefix_groups:
                prefix_groups[prefix] = []
            prefix_groups[prefix].append(full_hash)
        
        for prefix, full_hashes in prefix_groups.items():
            api_results = self._check_api(prefix)
            
            if api_results:
                self._save_to_db(prefix, api_results)
            
            suffix_to_count = {suffix: count for suffix, count in api_results}
            
            for full_hash in full_hashes:
                suffix = full_hash[5:]
                breach_count = suffix_to_count.get(suffix, 0)
                password = hash_to_password[full_hash]
                
                results.append({
                    'password': password,
                    'full_hash': full_hash,
                    'breached': breach_count > 0,
                    'breach_count': breach_count,
                    'last_checked': int(time.time()),
                    'last_breach_date': None,
                    'source': 'api' if api_results else 'offline'
                })
        
        password_order = {pwd: idx for idx, pwd in enumerate(passwords)}
        results.sort(key=lambda x: password_order.get(x['password'], 0))
        
        return results

    def get_statistics(self) -> Dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM breached_passwords')
        total_records = cursor.fetchone()[0]
        
        cursor.execute('SELECT SUM(breach_count) FROM breached_passwords WHERE breach_count > 0')
        total_breaches = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT MAX(last_checked) FROM breached_passwords')
        last_checked = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_cached_records': total_records,
            'total_breach_instances': total_breaches,
            'last_sync_timestamp': last_checked,
            'database_path': os.path.abspath(self.db_path)
        }

    def clear_cache(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM breached_passwords')
        conn.commit()
        conn.close()


def check_password_breach(password: str, use_cache: bool = True) -> Dict:
    checker = PasswordBreachChecker()
    return checker.check_password(password, use_cache)


def check_passwords_breach_batch(passwords: List[str], use_cache: bool = True) -> List[Dict]:
    checker = PasswordBreachChecker()
    return checker.check_passwords_batch(passwords, use_cache)
