from faker import Faker
from typing import Any, Dict, List, Set, Optional, Callable
import random
import uuid
import re
from datetime import datetime, timedelta

fake = Faker('zh_CN')

TYPE_MAPPING = {
    'name': lambda: fake.name(),
    'address': lambda: fake.address(),
    'email': lambda: fake.email(),
    'phone': lambda: fake.phone_number(),
    'mobile': lambda: fake.phone_number(),
    'date': lambda: fake.date(),
    'datetime': lambda: fake.date_time(),
    'string': lambda: fake.word(),
    'integer': lambda: random.randint(1, 1000),
    'number': lambda: round(random.uniform(0, 10000), 2),
    'boolean': lambda: random.choice([True, False]),
    'uuid': lambda: str(uuid.uuid4()),
    'guid': lambda: str(uuid.uuid4()),
}

PROPERTY_TYPE_MAP = {
    '姓名': 'name',
    'name': 'name',
    '地址': 'address',
    'address': 'address',
    '邮箱': 'email',
    'email': 'email',
    '手机号': 'mobile',
    'phone': 'phone',
    'mobile': 'mobile',
    '日期': 'date',
    'date': 'date',
    'datetime': 'datetime',
    '时间': 'datetime',
    'uuid': 'uuid',
    'guid': 'uuid',
}

FAKER_METHOD_MAP = {
    'name': 'name',
    'address': 'address',
    'email': 'email',
    'phone_number': 'phone_number',
    'phone': 'phone_number',
    'mobile': 'phone_number',
    'date': 'date',
    'date_time': 'date_time',
    'datetime': 'date_time',
    'word': 'word',
    'text': 'text',
    'sentence': 'sentence',
    'paragraph': 'paragraph',
    'url': 'url',
    'domain_name': 'domain_name',
    'ip_address': 'ipv4',
    'ipv4': 'ipv4',
    'ipv6': 'ipv6',
    'company': 'company',
    'job': 'job',
    'ssn': 'ssn',
    'user_name': 'user_name',
    'username': 'user_name',
    'password': 'password',
    'first_name': 'first_name',
    'last_name': 'last_name',
    'city': 'city',
    'country': 'country',
    'street_address': 'street_address',
    'postcode': 'postcode',
    'latitude': 'latitude',
    'longitude': 'longitude',
    'random_int': 'random_int',
    'random_digit': 'random_digit',
    'random_element': 'random_element',
    'uuid4': 'uuid4',
    'uuid': 'uuid4',
}

ID_FIELD_NAMES = {'id', 'uuid', 'guid', 'id号', '编号', '标识'}

TEMPLATES = {
    'user': {
        'name': '用户表',
        'description': '用户信息表模板，包含基本用户信息',
        'schema': {
            'type': 'object',
            'properties': {
                'id': {'type': 'string'},
                'username': {'type': 'string', 'faker': 'user_name', 'unique': True},
                'email': {'type': 'string', 'format': 'email', 'unique': True},
                'phone': {'type': 'string', 'title': '手机号'},
                'real_name': {'type': 'string', 'title': '姓名'},
                'nickname': {'type': 'string', 'faker': 'word'},
                'age': {'type': 'integer', 'faker': {'type': 'random_int', 'params': {'min': 18, 'max': 80}}},
                'gender': {'type': 'string', 'enum': ['男', '女', '未知']},
                'avatar': {'type': 'string', 'faker': 'url'},
                'address': {'type': 'string', 'title': '地址'},
                'city': {'type': 'string', 'faker': 'city'},
                'province': {'type': 'string', 'faker': 'province'},
                'register_time': {'type': 'string', 'format': 'date-time'},
                'last_login': {'type': 'string', 'format': 'date-time'},
                'is_active': {'type': 'boolean'},
                'is_vip': {'type': 'boolean'}
            }
        }
    },
    'order': {
        'name': '订单表',
        'description': '订单信息表模板，支持关联用户ID',
        'schema': {
            'type': 'object',
            'properties': {
                'id': {'type': 'string'},
                'order_no': {'type': 'string', 'unique': True, 'eval': "'ORD' + str(uuid.uuid4())[:8].upper() + str(random.randint(1000, 9999))"},
                'user_id': {'type': 'string', 'description': '关联用户ID'},
                'product_id': {'type': 'string', 'description': '关联商品ID'},
                'product_name': {'type': 'string', 'faker': 'word'},
                'product_price': {'type': 'number', 'faker': {'type': 'random_int', 'params': {'min': 10, 'max': 9999}}},
                'quantity': {'type': 'integer', 'faker': {'type': 'random_int', 'params': {'min': 1, 'max': 10}}},
                'total_amount': {'type': 'number', 'eval': 'product_price * quantity'},
                'discount_amount': {'type': 'number', 'faker': {'type': 'random_int', 'params': {'min': 0, 'max': 100}}},
                'pay_amount': {'type': 'number', 'eval': 'total_amount - discount_amount if total_amount > discount_amount else 0'},
                'status': {'type': 'string', 'enum': ['待支付', '已支付', '已发货', '已完成', '已取消', '已退款']},
                'pay_method': {'type': 'string', 'enum': ['微信支付', '支付宝', '银行卡', '货到付款']},
                'create_time': {'type': 'string', 'format': 'date-time'},
                'pay_time': {'type': 'string', 'format': 'date-time'},
                'ship_time': {'type': 'string', 'format': 'date-time'},
                'receive_address': {'type': 'string', 'title': '地址'},
                'receive_name': {'type': 'string', 'title': '姓名'},
                'receive_phone': {'type': 'string', 'title': '手机号'},
                'remark': {'type': 'string', 'faker': 'sentence'}
            }
        }
    },
    'product': {
        'name': '商品表',
        'description': '商品信息表模板',
        'schema': {
            'type': 'object',
            'properties': {
                'id': {'type': 'string'},
                'product_name': {'type': 'string', 'faker': 'word'},
                'product_title': {'type': 'string', 'faker': 'sentence'},
                'category': {'type': 'string', 'enum': ['电子产品', '服装鞋帽', '食品饮料', '家居用品', '美妆护肤', '运动户外', '图书音像', '母婴用品']},
                'brand': {'type': 'string', 'faker': 'company'},
                'price': {'type': 'number', 'faker': {'type': 'random_int', 'params': {'min': 10, 'max': 9999}}},
                'original_price': {'type': 'number', 'eval': 'price + random.randint(50, 500)'},
                'stock': {'type': 'integer', 'faker': {'type': 'random_int', 'params': {'min': 0, 'max': 1000}}},
                'sales': {'type': 'integer', 'faker': {'type': 'random_int', 'params': {'min': 0, 'max': 5000}}},
                'status': {'type': 'string', 'enum': ['上架', '下架', '售罄']},
                'description': {'type': 'string', 'faker': 'paragraph'},
                'image': {'type': 'string', 'faker': 'url'},
                'images': {'type': 'array', 'items': {'type': 'string', 'faker': 'url'}, 'minItems': 1, 'maxItems': 5},
                'weight': {'type': 'number', 'faker': {'type': 'random_int', 'params': {'min': 1, 'max': 5000}}},
                'unit': {'type': 'string', 'enum': ['件', '个', '台', '套', '盒', '瓶', '包', '袋']},
                'create_time': {'type': 'string', 'format': 'date-time'},
                'update_time': {'type': 'string', 'format': 'date-time'},
                'is_hot': {'type': 'boolean'},
                'is_new': {'type': 'boolean'}
            }
        }
    },
    'address': {
        'name': '收货地址表',
        'description': '用户收货地址模板',
        'schema': {
            'type': 'object',
            'properties': {
                'id': {'type': 'string'},
                'user_id': {'type': 'string', 'description': '关联用户ID'},
                'name': {'type': 'string', 'title': '姓名'},
                'phone': {'type': 'string', 'title': '手机号'},
                'province': {'type': 'string', 'faker': 'province'},
                'city': {'type': 'string', 'faker': 'city'},
                'district': {'type': 'string', 'faker': 'word'},
                'detail': {'type': 'string', 'title': '地址'},
                'postcode': {'type': 'string', 'faker': 'postcode'},
                'is_default': {'type': 'boolean'},
                'tag': {'type': 'string', 'enum': ['家', '公司', '学校', '其他']},
                'create_time': {'type': 'string', 'format': 'date-time'}
            }
        }
    }
}


class MockDataGenerator:
    def __init__(self):
        self.unique_values: Dict[str, Set[Any]] = {}
        self.max_retries = 1000
        self.current_object: Dict[str, Any] = {}
        self.index: int = 0
        self.ref_data: Dict[str, List[Any]] = {}

    def reset(self):
        self.unique_values.clear()
        self.current_object = {}
        self.index = 0
        self.ref_data.clear()

    def set_ref_data(self, ref_name: str, data: List[Any]):
        self.ref_data[ref_name] = data

    def is_id_field(self, prop_key: str) -> bool:
        return prop_key.lower() in ID_FIELD_NAMES

    def generate_uuid(self) -> str:
        return str(uuid.uuid4())

    def get_faker_value(self, faker_type: str, params: Dict[str, Any] = None) -> Any:
        params = params or {}
        
        if faker_type.lower() in ['uuid', 'uuid4', 'guid']:
            return self.generate_uuid()
        
        method_name = FAKER_METHOD_MAP.get(faker_type, faker_type)
        if hasattr(fake, method_name):
            method = getattr(fake, method_name)
            try:
                return method(**params)
            except Exception:
                return method()
        return fake.word()

    def eval_expression(self, expression: str) -> Any:
        safe_builtins = {
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'len': len,
            'max': max,
            'min': min,
            'sum': sum,
            'abs': abs,
            'round': round,
        }
        safe_globals = {
            '__builtins__': safe_builtins,
            'random': random,
            'uuid': uuid,
            'datetime': datetime,
            'timedelta': timedelta,
            'fake': fake,
            'index': self.index,
            **self.current_object
        }
        
        try:
            return eval(expression, safe_globals)
        except Exception as e:
            return f"[eval_error: {str(e)}]"

    def resolve_ref(self, ref_config: Any) -> Any:
        if isinstance(ref_config, str):
            ref_name = ref_config
            if ref_name in self.ref_data and self.ref_data[ref_name]:
                return random.choice(self.ref_data[ref_name])
        elif isinstance(ref_config, dict):
            ref_name = ref_config.get('name')
            field = ref_config.get('field')
            if ref_name in self.ref_data and self.ref_data[ref_name]:
                selected = random.choice(self.ref_data[ref_name])
                if field and isinstance(selected, dict):
                    return selected.get(field)
                return selected
        return None

    def generate_base_value(self, property_schema: Dict[str, Any], prop_key: str = '') -> Any:
        if 'eval' in property_schema:
            return self.eval_expression(property_schema['eval'])

        if 'ref' in property_schema:
            ref_value = self.resolve_ref(property_schema['ref'])
            if ref_value is not None:
                return ref_value

        if 'faker' in property_schema:
            faker_config = property_schema['faker']
            if isinstance(faker_config, str):
                return self.get_faker_value(faker_config)
            elif isinstance(faker_config, dict):
                faker_type = faker_config.get('type', faker_config.get('method', 'word'))
                params = faker_config.get('params', {})
                return self.get_faker_value(faker_type, params)

        if 'example' in property_schema:
            return property_schema['example']

        if 'enum' in property_schema:
            return random.choice(property_schema['enum'])

        prop_type = property_schema.get('type', 'string')
        prop_format = property_schema.get('format', '')
        prop_name = property_schema.get('title', property_schema.get('description', ''))

        if self.is_id_field(prop_key):
            return self.generate_uuid()

        name_candidates = [prop_key, prop_name]
        for candidate in name_candidates:
            if candidate:
                mapped_type = PROPERTY_TYPE_MAP.get(candidate.lower()) or PROPERTY_TYPE_MAP.get(candidate)
                if mapped_type and mapped_type in TYPE_MAPPING:
                    return TYPE_MAPPING[mapped_type]()

        if prop_format:
            format_map = {
                'email': lambda: fake.email(),
                'date': lambda: fake.date(),
                'date-time': lambda: fake.date_time(),
                'uri': lambda: fake.url(),
                'url': lambda: fake.url(),
                'hostname': lambda: fake.domain_name(),
                'ipv4': lambda: fake.ipv4(),
                'ipv6': lambda: fake.ipv6(),
                'uuid': lambda: self.generate_uuid(),
            }
            if prop_format in format_map:
                return format_map[prop_format]()

        type_generators = {
            'string': lambda: fake.word(),
            'integer': lambda: random.randint(1, 1000),
            'number': lambda: round(random.uniform(0, 10000), 2),
            'boolean': lambda: random.choice([True, False]),
            'null': lambda: None,
        }

        if prop_type in type_generators:
            return type_generators[prop_type]()

        return fake.word()

    def generate_unique_value(self, property_schema: Dict[str, Any], prop_key: str, field_path: str) -> Any:
        if field_path not in self.unique_values:
            self.unique_values[field_path] = set()

        existing_values = self.unique_values[field_path]
        
        for _ in range(self.max_retries):
            value = self.generate_base_value(property_schema, prop_key)
            if value not in existing_values:
                existing_values.add(value)
                return value
        
        raise ValueError(f"无法在 {self.max_retries} 次尝试内为字段 {field_path} 生成唯一值")

    def generate_value(self, property_schema: Dict[str, Any], prop_key: str = '', field_path: str = '') -> Any:
        is_unique = property_schema.get('unique', False) or self.is_id_field(prop_key)
        
        if is_unique:
            return self.generate_unique_value(property_schema, prop_key, field_path or prop_key)
        else:
            return self.generate_base_value(property_schema, prop_key)

    def generate_object(self, schema: Dict[str, Any], parent_path: str = '') -> Dict[str, Any]:
        result = {}
        self.current_object = result
        properties = schema.get('properties', {})

        simple_props = {}
        eval_props = {}
        for prop_name, prop_schema in properties.items():
            if 'eval' in prop_schema:
                eval_props[prop_name] = prop_schema
            else:
                simple_props[prop_name] = prop_schema

        for prop_name, prop_schema in simple_props.items():
            current_path = f"{parent_path}.{prop_name}" if parent_path else prop_name
            
            if prop_schema.get('type') == 'object':
                result[prop_name] = self.generate_object(prop_schema, current_path)
            elif prop_schema.get('type') == 'array':
                items_schema = prop_schema.get('items', {})
                min_items = prop_schema.get('minItems', 1)
                max_items = prop_schema.get('maxItems', 5)
                count = random.randint(min_items, max_items)
                result[prop_name] = [
                    self.generate_value(items_schema, prop_name, f"{current_path}[{i}]")
                    for i in range(count)
                ]
            else:
                result[prop_name] = self.generate_value(prop_schema, prop_name, current_path)

        for prop_name, prop_schema in eval_props.items():
            current_path = f"{parent_path}.{prop_name}" if parent_path else prop_name
            is_unique = prop_schema.get('unique', False)
            if is_unique:
                result[prop_name] = self.generate_unique_value(prop_schema, prop_name, current_path)
            else:
                result[prop_name] = self.generate_value(prop_schema, prop_name, current_path)

        return result

    def generate_mock_data(self, schema: Dict[str, Any], count: int = 1, refs: Dict[str, List[Any]] = None) -> List[Dict[str, Any]]:
        if count < 1:
            count = 1
        if count > 100:
            count = 100

        self.reset()
        
        if refs:
            for ref_name, ref_values in refs.items():
                self.set_ref_data(ref_name, ref_values)

        result = []
        for i in range(count):
            self.index = i
            result.append(self.generate_object(schema))

        return result

    def generate_related_data(self, relations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        self.reset()
        results = {}

        for relation in relations:
            name = relation.get('name')
            schema = relation.get('schema')
            count = relation.get('count', 10)
            refs = relation.get('refs', {})

            resolved_refs = {}
            for ref_key, ref_value in refs.items():
                if isinstance(ref_value, str) and ref_value in results:
                    resolved_refs[ref_key] = results[ref_value]
                else:
                    resolved_refs[ref_key] = ref_value

            results[name] = self.generate_mock_data(schema, count, resolved_refs)

        return results


_generator = MockDataGenerator()


def generate_mock_data(schema: Dict[str, Any], count: int = 1, refs: Dict[str, List[Any]] = None) -> List[Dict[str, Any]]:
    return _generator.generate_mock_data(schema, count, refs)


def generate_related_data(relations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    return _generator.generate_related_data(relations)


def get_template(template_name: str) -> Optional[Dict[str, Any]]:
    return TEMPLATES.get(template_name)


def get_all_templates() -> Dict[str, Dict[str, Any]]:
    return {name: {k: v for k, v in template.items() if k != 'schema'} for name, template in TEMPLATES.items()}
