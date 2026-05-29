import threading
import time
from dataclasses import dataclass
from decimal import Decimal, getcontext
from typing import Dict, Optional, Tuple, Union
from urllib.request import urlopen, Request
from urllib.error import URLError
import json

getcontext().prec = 50


@dataclass
class ConversionResult:
    value: Decimal
    formula: str


class CurrencyRateProvider:
    _API_URL = 'https://open.er-api.com/v6/latest/USD'
    _CACHE_DURATION = 3600
    _rates: Dict[str, Decimal] = {}
    _last_fetch: float = 0
    _lock = threading.Lock()
    _base_currency = 'USD'

    @classmethod
    def get_rate(cls, currency: str) -> Optional[Decimal]:
        currency = currency.upper()
        rates = cls._get_rates()
        if currency == cls._base_currency:
            return Decimal('1.0')
        return rates.get(currency)

    @classmethod
    def get_rates(cls) -> Dict[str, Decimal]:
        return dict(cls._get_rates())

    @classmethod
    def _get_rates(cls) -> Dict[str, Decimal]:
        now = time.time()
        if cls._rates and (now - cls._last_fetch) < cls._CACHE_DURATION:
            return cls._rates
        fetched = cls._fetch_rates()
        if fetched is not None:
            return fetched
        if cls._rates:
            return cls._rates
        return cls._get_default_rates()

    @classmethod
    def _fetch_rates(cls) -> Optional[Dict[str, Decimal]]:
        with cls._lock:
            now = time.time()
            if cls._rates and (now - cls._last_fetch) < cls._CACHE_DURATION:
                return cls._rates
            try:
                req = Request(cls._API_URL, headers={'User-Agent': 'UnitConverter/1.0'})
                with urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode('utf-8'))
                if data.get('result') == 'success' and 'rates' in data:
                    rates = {}
                    for code, rate in data['rates'].items():
                        rates[code.upper()] = Decimal(str(rate))
                    rates[cls._base_currency] = Decimal('1.0')
                    cls._rates = rates
                    cls._last_fetch = now
                    return cls._rates
            except (URLError, json.JSONDecodeError, KeyError, OSError):
                pass
            return None

    @classmethod
    def _get_default_rates(cls) -> Dict[str, Decimal]:
        return {
            'USD': Decimal('1.0'),
            'CNY': Decimal('7.24'),
            'EUR': Decimal('0.92'),
            'GBP': Decimal('0.79'),
            'JPY': Decimal('154.5'),
            'KRW': Decimal('1360.0'),
            'HKD': Decimal('7.82'),
            'TWD': Decimal('32.5'),
            'SGD': Decimal('1.34'),
            'AUD': Decimal('1.53'),
            'CAD': Decimal('1.37'),
            'CHF': Decimal('0.88'),
        }

    @classmethod
    def force_refresh(cls) -> bool:
        with cls._lock:
            try:
                req = Request(cls._API_URL, headers={'User-Agent': 'UnitConverter/1.0'})
                with urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode('utf-8'))
                if data.get('result') == 'success' and 'rates' in data:
                    rates = {}
                    for code, rate in data['rates'].items():
                        rates[code.upper()] = Decimal(str(rate))
                    rates[cls._base_currency] = Decimal('1.0')
                    cls._rates = rates
                    cls._last_fetch = time.time()
                    return True
            except (URLError, json.JSONDecodeError, KeyError, OSError):
                pass
            return False

    @classmethod
    def set_rates(cls, rates: Dict[str, Union[float, str, Decimal]]):
        with cls._lock:
            converted = {}
            for code, rate in rates.items():
                converted[code.upper()] = Decimal(str(rate))
            converted[cls._base_currency] = Decimal('1.0')
            cls._rates = converted
            cls._last_fetch = time.time()

    @classmethod
    def is_cache_expired(cls) -> bool:
        if not cls._rates:
            return True
        return (time.time() - cls._last_fetch) >= cls._CACHE_DURATION

    @classmethod
    def get_supported_currencies(cls) -> Dict[str, str]:
        rates = cls._get_rates()
        names = {
            'USD': '美元', 'CNY': '人民币', 'EUR': '欧元', 'GBP': '英镑',
            'JPY': '日元', 'KRW': '韩元', 'HKD': '港币', 'TWD': '新台币',
            'SGD': '新加坡元', 'AUD': '澳元', 'CAD': '加元', 'CHF': '瑞士法郎',
        }
        result = {}
        for code in rates:
            result[code] = names.get(code, code)
        return result


class UnitConverter:
    """单位换算API，支持长度、重量、温度、存储、货币等常见单位互转，支持自定义单位注册和换算公式返回"""

    _LENGTH = {
        'm': Decimal('1.0'),
        'km': Decimal('1000.0'),
        'mi': Decimal('1609.344'),
    }

    _WEIGHT = {
        'g': Decimal('1.0'),
        'kg': Decimal('1000.0'),
        'lb': Decimal('453.59237'),
    }

    _STORAGE = {
        'mb': Decimal('1.0'),
        'gb': Decimal('1024.0'),
        'tb': Decimal('1048576.0'),
    }

    _CATEGORIES = {
        'length': ('m', 'km', 'mi'),
        'weight': ('g', 'kg', 'lb'),
        'temperature': ('c', 'f'),
        'storage': ('mb', 'gb', 'tb'),
        'currency': (),
    }

    _UNIT_NAMES = {
        'length': {'m': '米', 'km': '千米', 'mi': '英里'},
        'weight': {'g': '克', 'kg': '千克', 'lb': '磅'},
        'temperature': {'c': '摄氏度', 'f': '华氏度'},
        'storage': {'mb': '兆字节', 'gb': '吉字节', 'tb': '太字节'},
        'currency': {},
    }

    _CUSTOM_CATEGORIES: Dict[str, Dict[str, Decimal]] = {}
    _CUSTOM_UNIT_NAMES: Dict[str, Dict[str, str]] = {}

    @classmethod
    def convert(cls, value: Union[float, str, Decimal], from_unit: str, to_unit: str,
                decimal_places: int = 2) -> Decimal:
        result = cls.convert_with_formula(value, from_unit, to_unit, decimal_places)
        return result.value

    @classmethod
    def convert_with_formula(cls, value: Union[float, str, Decimal], from_unit: str, to_unit: str,
                             decimal_places: int = 2) -> ConversionResult:
        original_from = from_unit
        original_to = to_unit
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()

        dec_value = Decimal(str(value))

        if from_unit == to_unit:
            rounded = cls._round_result(dec_value, decimal_places)
            return ConversionResult(value=rounded, formula=f"{dec_value} {original_from} = {rounded} {original_to}")

        category = cls._get_category(from_unit, to_unit)

        if category == 'temperature':
            result, formula = cls._convert_temperature_with_formula(dec_value, from_unit, to_unit, original_from, original_to)
        elif category == 'currency':
            result, formula = cls._convert_currency_with_formula(dec_value, from_unit, to_unit, original_from, original_to)
        else:
            result, formula = cls._convert_linear_with_formula(dec_value, from_unit, to_unit, category, original_from, original_to)

        rounded = cls._round_result(result, decimal_places)
        return ConversionResult(value=rounded, formula=formula)

    @classmethod
    def _round_result(cls, value: Decimal, decimal_places: int) -> Decimal:
        if decimal_places < 0:
            raise ValueError("decimal_places 不能为负数")
        return value.quantize(Decimal('1.' + '0' * decimal_places))

    @classmethod
    def _convert_linear(cls, value: Decimal, from_unit: str, to_unit: str, category: str) -> Decimal:
        result, _ = cls._convert_linear_with_formula(value, from_unit, to_unit, category, from_unit, to_unit)
        return result

    @classmethod
    def _convert_linear_with_formula(cls, value: Decimal, from_unit: str, to_unit: str,
                                     category: str, display_from: str, display_to: str) -> Tuple[Decimal, str]:
        unit_map = cls._get_unit_map(category)

        from_factor = unit_map[from_unit]
        to_factor = unit_map[to_unit]
        ratio = from_factor / to_factor
        result = value * ratio

        formula = f"1 {display_from} = {ratio} {display_to}; {value} × {ratio} = {result} {display_to}"
        return result, formula

    @classmethod
    def _convert_temperature(cls, value: Decimal, from_unit: str, to_unit: str) -> Decimal:
        result, _ = cls._convert_temperature_with_formula(value, from_unit, to_unit, from_unit, to_unit)
        return result

    @classmethod
    def _convert_temperature_with_formula(cls, value: Decimal, from_unit: str, to_unit: str,
                                          display_from: str, display_to: str) -> Tuple[Decimal, str]:
        nine = Decimal('9')
        five = Decimal('5')
        thirty_two = Decimal('32')

        if from_unit == 'c' and to_unit == 'f':
            result = value * nine / five + thirty_two
            formula = f"°F = °C × 9/5 + 32; {value} × 9/5 + 32 = {result} {display_to}"
            return result, formula
        elif from_unit == 'f' and to_unit == 'c':
            result = (value - thirty_two) * five / nine
            formula = f"°C = (°F - 32) × 5/9; ({value} - 32) × 5/9 = {result} {display_to}"
            return result, formula
        raise ValueError(f"不支持的温度转换: {from_unit} -> {to_unit}")

    @classmethod
    def _convert_currency_with_formula(cls, value: Decimal, from_unit: str, to_unit: str,
                                       display_from: str, display_to: str) -> Tuple[Decimal, str]:
        from_upper = from_unit.upper()
        to_upper = to_unit.upper()

        from_rate = CurrencyRateProvider.get_rate(from_upper)
        to_rate = CurrencyRateProvider.get_rate(to_upper)

        if from_rate is None:
            raise ValueError(f"不支持的货币代码: {display_from}")
        if to_rate is None:
            raise ValueError(f"不支持的货币代码: {display_to}")

        cross_rate = to_rate / from_rate
        result = value * cross_rate

        formula = f"1 {display_from} = {cross_rate} {display_to} (汇率: 1 {from_upper} = {from_rate} USD, 1 USD = {to_rate} {to_upper}); {value} × {cross_rate} = {result} {display_to}"
        return result, formula

    @classmethod
    def _get_unit_map(cls, category: str) -> Dict[str, Decimal]:
        if category in cls._CUSTOM_CATEGORIES:
            return cls._CUSTOM_CATEGORIES[category]

        built_in = {
            'length': cls._LENGTH,
            'weight': cls._WEIGHT,
            'storage': cls._STORAGE,
        }
        if category in built_in:
            return built_in[category]
        raise ValueError(f"未知的单位类别: {category}")

    @classmethod
    def _get_category(cls, unit1: str, unit2: str) -> str:
        for category, units in cls._CATEGORIES.items():
            if category == 'currency':
                continue
            if unit1 in units and unit2 in units:
                return category

        if cls._is_currency(unit1) and cls._is_currency(unit2):
            return 'currency'

        for category in cls._CUSTOM_CATEGORIES:
            if unit1 in cls._CUSTOM_CATEGORIES[category] and unit2 in cls._CUSTOM_CATEGORIES[category]:
                return category

        raise ValueError(f"单位不兼容或不支持: {unit1}, {unit2}")

    @classmethod
    def _is_currency(cls, unit: str) -> bool:
        return CurrencyRateProvider.get_rate(unit.upper()) is not None

    @classmethod
    def register_unit(cls, category: str, unit: str, factor: Union[float, str, Decimal],
                      display_name: str = None) -> None:
        category_lower = category.lower()
        unit_lower = unit.lower()
        dec_factor = Decimal(str(factor))

        if category_lower == 'temperature':
            raise ValueError("温度类别不支持自定义注册（非线性换算）")
        if category_lower == 'currency':
            raise ValueError("货币类别请使用 CurrencyRateProvider.set_rates() 设置")

        if category_lower in ('length', 'weight', 'storage'):
            unit_map = cls._get_unit_map(category_lower)
            unit_map[unit_lower] = dec_factor
            if display_name:
                cls._UNIT_NAMES[category_lower][unit_lower] = display_name
            elif unit_lower not in cls._UNIT_NAMES.get(category_lower, {}):
                cls._UNIT_NAMES.setdefault(category_lower, {})[unit_lower] = unit_lower
            if unit_lower not in cls._CATEGORIES[category_lower]:
                cls._CATEGORIES[category_lower] = cls._CATEGORIES[category_lower] + (unit_lower,)
        else:
            if category_lower not in cls._CUSTOM_CATEGORIES:
                cls._CUSTOM_CATEGORIES[category_lower] = {}
                cls._CUSTOM_UNIT_NAMES[category_lower] = {}
                cls._CATEGORIES[category_lower] = ()
            cls._CUSTOM_CATEGORIES[category_lower][unit_lower] = dec_factor
            if display_name:
                cls._CUSTOM_UNIT_NAMES[category_lower][unit_lower] = display_name
            else:
                cls._CUSTOM_UNIT_NAMES[category_lower][unit_lower] = unit_lower
            if unit_lower not in cls._CATEGORIES[category_lower]:
                cls._CATEGORIES[category_lower] = cls._CATEGORIES[category_lower] + (unit_lower,)

    @classmethod
    def unregister_unit(cls, category: str, unit: str) -> None:
        category_lower = category.lower()
        unit_lower = unit.lower()

        if category_lower in ('temperature', 'currency'):
            raise ValueError(f"不允许从 {category} 类别中移除单位")

        if category_lower in ('length', 'weight', 'storage'):
            built_in_units = {
                'length': cls._LENGTH,
                'weight': cls._WEIGHT,
                'storage': cls._STORAGE,
            }
            unit_map = built_in_units[category_lower]
            if unit_lower in unit_map:
                del unit_map[unit_lower]
            if unit_lower in cls._UNIT_NAMES.get(category_lower, {}):
                del cls._UNIT_NAMES[category_lower][unit_lower]
            current = list(cls._CATEGORIES[category_lower])
            if unit_lower in current:
                current.remove(unit_lower)
                cls._CATEGORIES[category_lower] = tuple(current)
        elif category_lower in cls._CUSTOM_CATEGORIES:
            if unit_lower in cls._CUSTOM_CATEGORIES[category_lower]:
                del cls._CUSTOM_CATEGORIES[category_lower][unit_lower]
            if unit_lower in cls._CUSTOM_UNIT_NAMES.get(category_lower, {}):
                del cls._CUSTOM_UNIT_NAMES[category_lower][unit_lower]
            current = list(cls._CATEGORIES[category_lower])
            if unit_lower in current:
                current.remove(unit_lower)
                cls._CATEGORIES[category_lower] = tuple(current)
            if not cls._CUSTOM_CATEGORIES[category_lower]:
                del cls._CUSTOM_CATEGORIES[category_lower]
                del cls._CUSTOM_UNIT_NAMES[category_lower]
                del cls._CATEGORIES[category_lower]
        else:
            raise ValueError(f"未知的类别: {category}")

    @classmethod
    def get_supported_units(cls) -> dict:
        result = {}
        for category, names in cls._UNIT_NAMES.items():
            if category == 'currency':
                result['currency'] = CurrencyRateProvider.get_supported_currencies()
            else:
                result[category] = dict(names)

        for category, names in cls._CUSTOM_UNIT_NAMES.items():
            result[category] = dict(names)

        return result

    @classmethod
    def currency(cls, value: Union[float, str, Decimal], from_unit: str, to_unit: str,
                 decimal_places: int = 2) -> Decimal:
        return cls.convert(value, from_unit, to_unit, decimal_places)

    @classmethod
    def length(cls, value: Union[float, str, Decimal], from_unit: str, to_unit: str,
               decimal_places: int = 2) -> Decimal:
        return cls.convert(value, from_unit, to_unit, decimal_places)

    @classmethod
    def weight(cls, value: Union[float, str, Decimal], from_unit: str, to_unit: str,
               decimal_places: int = 2) -> Decimal:
        return cls.convert(value, from_unit, to_unit, decimal_places)

    @classmethod
    def temperature(cls, value: Union[float, str, Decimal], from_unit: str, to_unit: str,
                    decimal_places: int = 2) -> Decimal:
        return cls.convert(value, from_unit, to_unit, decimal_places)

    @classmethod
    def storage(cls, value: Union[float, str, Decimal], from_unit: str, to_unit: str,
                decimal_places: int = 2) -> Decimal:
        return cls.convert(value, from_unit, to_unit, decimal_places)
