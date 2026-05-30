import re
import hashlib
import random as _random
import string
from typing import Any, Dict, Optional, Union


_DIGITS = string.digits
_LOWERS = string.ascii_lowercase
_UPPERS = string.ascii_uppercase


def mask_phone(phone: str) -> str:
    if not phone or len(phone) < 7:
        return phone
    return phone[:3] + '*' * 4 + phone[-4:]


def mask_id_card(id_card: str) -> str:
    if not id_card or len(id_card) < 8:
        return id_card
    return id_card[:4] + '*' * 3 + id_card[-4:]


def mask_email(email: str) -> str:
    if not email or '@' not in email:
        return email
    username, domain = email.split('@', 1)
    if len(username) <= 2:
        return email
    masked_username = username[:2] + '*' * 3
    return f"{masked_username}@{domain}"


def mask_regex(value: str, pattern: str, replace: str) -> str:
    if not value:
        return value
    return re.sub(pattern, replace, value)


def _build_sub_table(key: str, charset: str) -> Dict[str, str]:
    h = hashlib.sha256((key + charset).encode()).hexdigest()
    seed = int(h, 16) % (2 ** 32)
    rng = _random.Random(seed)
    shuffled = list(charset)
    rng.shuffle(shuffled)
    return dict(zip(charset, shuffled))


def _build_rev_table(key: str, charset: str) -> Dict[str, str]:
    fwd = _build_sub_table(key, charset)
    return {v: k for k, v in fwd.items()}


def _classify_char(ch: str) -> Optional[str]:
    if ch in _DIGITS:
        return _DIGITS
    if ch in _LOWERS:
        return _LOWERS
    if ch in _UPPERS:
        return _UPPERS
    return None


def mask_encrypt(value: str, key: str) -> str:
    if not value or not key:
        return value
    tables: Dict[str, Dict[str, str]] = {}
    result = []
    for ch in value:
        charset = _classify_char(ch)
        if charset is None:
            result.append(ch)
        else:
            if charset not in tables:
                tables[charset] = _build_sub_table(key, charset)
            result.append(tables[charset][ch])
    return ''.join(result)


def mask_decrypt(value: str, key: str) -> str:
    if not value or not key:
        return value
    tables: Dict[str, Dict[str, str]] = {}
    result = []
    for ch in value:
        charset = _classify_char(ch)
        if charset is None:
            result.append(ch)
        else:
            if charset not in tables:
                tables[charset] = _build_rev_table(key, charset)
            result.append(tables[charset][ch])
    return ''.join(result)


_LAST_NAMES = list('赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜')
_FIRST_NAMES = list('伟芳敏静丽强磊军洋勇艳杰娟涛超明华丹巧辉力梅鑫桂英珍贞莉瑞峰建文辉')

_FAKE_DOMAINS = ['example.com', 'test.org', 'mail.net', 'demo.cn', 'fake.io']


def _random_phone() -> str:
    rng = _random.Random()
    prefix = rng.choice(['130', '131', '132', '133', '135', '136', '137', '138', '139',
                         '150', '151', '152', '155', '156', '157', '158', '159',
                         '180', '181', '182', '183', '185', '186', '187', '188', '189'])
    suffix = ''.join([str(rng.randint(0, 9)) for _ in range(8)])
    return prefix + suffix


def _random_id_card() -> str:
    rng = _random.Random()
    area = ''.join([str(rng.randint(0, 9)) for _ in range(6)])
    date = ''.join([str(rng.randint(0, 9)) for _ in range(8)])
    seq = ''.join([str(rng.randint(0, 9)) for _ in range(3)])
    check = str(rng.randint(0, 9))
    return area + date + seq + check


def _random_email() -> str:
    rng = _random.Random()
    length = rng.randint(4, 10)
    username = ''.join(rng.choices(string.ascii_lowercase + string.digits, k=length))
    domain = rng.choice(_FAKE_DOMAINS)
    return f'{username}@{domain}'


def _random_name() -> str:
    rng = _random.Random()
    last = rng.choice(_LAST_NAMES)
    first_len = rng.choice([1, 2])
    first = ''.join(rng.choices(_FIRST_NAMES, k=first_len))
    return last + first


def _random_default(value: str) -> str:
    rng = _random.Random()
    result = []
    for ch in value:
        charset = _classify_char(ch)
        if charset:
            result.append(rng.choice(charset))
        else:
            result.append(ch)
    return ''.join(result)


def mask_random(value: str, category: str = 'default') -> str:
    if not value:
        return value
    generators = {
        'phone': _random_phone,
        'id_card': _random_id_card,
        'email': _random_email,
        'name': _random_name,
    }
    gen = generators.get(category)
    if gen:
        return gen()
    return _random_default(value)


def mask_value(value: str, mask_config: Union[str, Dict]) -> str:
    if isinstance(mask_config, str):
        mask_functions = {
            'phone': mask_phone,
            'id_card': mask_id_card,
            'email': mask_email,
        }
        mask_func = mask_functions.get(mask_config)
        if mask_func:
            return mask_func(value)
        return value
    elif isinstance(mask_config, dict):
        mask_type = mask_config.get('type')
        if mask_type == 'regex':
            pattern = mask_config.get('pattern', '')
            replace = mask_config.get('replace', '')
            return mask_regex(value, pattern, replace)
        elif mask_type == 'encrypt':
            key = mask_config.get('key', '')
            return mask_encrypt(value, key)
        elif mask_type == 'random':
            category = mask_config.get('category', 'default')
            return mask_random(value, category)
        elif mask_type in ('phone', 'id_card', 'email'):
            builtins = {'phone': mask_phone, 'id_card': mask_id_card, 'email': mask_email}
            return builtins[mask_type](value)
        return value
    return value


def _match_path(pattern: str, path: str) -> bool:
    pattern_parts = pattern.split('.')
    path_parts = path.split('.')
    if len(pattern_parts) != len(path_parts):
        return False
    for p_part, path_part in zip(pattern_parts, path_parts):
        if p_part == '*':
            continue
        if p_part != path_part:
            return False
    return True


def _resolve_mask_type(key: str, path: str, simple_mappings: Dict, path_mappings: Dict) -> Any:
    for pattern, mask_type in path_mappings.items():
        if _match_path(pattern, path):
            return mask_type
    return simple_mappings.get(key)


def mask_json_data(data: Any, field_mappings: Dict, _path: str = '') -> Any:
    simple_mappings = {k: v for k, v in field_mappings.items() if '.' not in k}
    path_mappings = {k: v for k, v in field_mappings.items() if '.' in k}

    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            new_path = f'{_path}.{key}' if _path else key
            mask_config = _resolve_mask_type(key, new_path, simple_mappings, path_mappings)

            if mask_config is not None and isinstance(value, str):
                result[key] = mask_value(value, mask_config)
            elif isinstance(value, dict):
                result[key] = mask_json_data(value, field_mappings, new_path)
            elif isinstance(value, list):
                result[key] = mask_json_data(value, field_mappings, f'{new_path}.*')
            else:
                result[key] = value
        return result
    elif isinstance(data, list):
        result = []
        for item in data:
            if isinstance(item, dict):
                item_path = _path if _path else '*'
                result.append(mask_json_data(item, field_mappings, item_path))
            elif isinstance(item, list):
                item_path = f'{_path}.*' if _path else '*.*'
                result.append(mask_json_data(item, field_mappings, item_path))
            else:
                result.append(item)
        return result
    else:
        return data


def unmask_json_data(data: Any, field_mappings: Dict, _path: str = '') -> Any:
    simple_mappings = {k: v for k, v in field_mappings.items() if '.' not in k}
    path_mappings = {k: v for k, v in field_mappings.items() if '.' in k}

    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            new_path = f'{_path}.{key}' if _path else key
            mask_config = _resolve_mask_type(key, new_path, simple_mappings, path_mappings)

            if isinstance(mask_config, dict) and mask_config.get('type') == 'encrypt' and isinstance(value, str):
                enc_key = mask_config.get('key', '')
                result[key] = mask_decrypt(value, enc_key)
            elif isinstance(value, dict):
                result[key] = unmask_json_data(value, field_mappings, new_path)
            elif isinstance(value, list):
                result[key] = unmask_json_data(value, field_mappings, f'{new_path}.*')
            else:
                result[key] = value
        return result
    elif isinstance(data, list):
        result = []
        for item in data:
            if isinstance(item, dict):
                item_path = _path if _path else '*'
                result.append(unmask_json_data(item, field_mappings, item_path))
            elif isinstance(item, list):
                item_path = f'{_path}.*' if _path else '*.*'
                result.append(unmask_json_data(item, field_mappings, item_path))
            else:
                result.append(item)
        return result
    else:
        return data
