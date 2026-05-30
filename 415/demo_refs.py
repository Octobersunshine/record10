import json
import os
import sys

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from json_schema_validator import (
    validate,
    validate_json,
    JsonSchemaValidator,
    ReferenceResolver,
    ReferenceResolutionError,
)


def print_result(title, result_dict):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(json.dumps(result_dict, indent=2, ensure_ascii=False))


def demo_local_ref():
    """演示本地 $ref 引用 #/definitions/..."""
    schema = {
        "definitions": {
            "user": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "minimum": 1},
                    "name": {"type": "string", "minLength": 1},
                    "email": {"type": "string", "format": "email"},
                },
                "required": ["id", "name"],
            },
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
        },
        "type": "object",
        "properties": {
            "owner": {"$ref": "#/definitions/user"},
            "manager": {"$ref": "#/definitions/user"},
            "location": {"$ref": "#/definitions/address"},
        },
        "required": ["owner"],
    }

    valid_data = {
        "owner": {"id": 1, "name": "Alice", "email": "alice@example.com"},
        "manager": {"id": 2, "name": "Bob"},
        "location": {"street": "Main St", "city": "Beijing"},
    }
    print_result("1. 本地 $ref 引用 - 通过", validate_json(valid_data, schema))

    invalid_data = {
        "owner": {"id": 0, "name": "", "email": "not-an-email"},
    }
    print_result("2. 本地 $ref 引用 - 失败", validate_json(invalid_data, schema))


def demo_nested_ref():
    """演示嵌套 $ref 引用，$ref 指向的 schema 中又包含 $ref"""
    schema = {
        "definitions": {
            "name": {"type": "string", "minLength": 1},
            "person": {
                "type": "object",
                "properties": {
                    "first_name": {"$ref": "#/definitions/name"},
                    "last_name": {"$ref": "#/definitions/name"},
                    "age": {"type": "integer", "minimum": 0},
                },
                "required": ["first_name", "last_name"],
            },
            "employee": {
                "type": "object",
                "properties": {
                    "personal": {"$ref": "#/definitions/person"},
                    "department": {"type": "string"},
                    "reports_to": {"$ref": "#/definitions/employee"},
                },
                "required": ["personal", "department"],
            },
        },
        "type": "object",
        "properties": {
            "primary_contact": {"$ref": "#/definitions/employee"},
        },
        "required": ["primary_contact"],
    }

    valid_data = {
        "primary_contact": {
            "personal": {"first_name": "John", "last_name": "Doe", "age": 30},
            "department": "Engineering",
            "reports_to": {
                "personal": {"first_name": "Jane", "last_name": "Smith", "age": 40},
                "department": "Management",
            },
        },
    }
    print_result("3. 嵌套 $ref 引用 - 通过", validate_json(valid_data, schema))

    invalid_data = {
        "primary_contact": {
            "personal": {"first_name": "", "last_name": "Doe"},
            "department": "Engineering",
        },
    }
    print_result("4. 嵌套 $ref 引用 - first_name 为空", validate_json(invalid_data, schema))


def demo_file_ref():
    """演示文件 $ref 引用，通过 file:// URI 加载外部 schema"""
    schemas_dir = os.path.join(os.path.dirname(__file__), "schemas")
    base_uri = f"file:///{schemas_dir.replace(os.sep, '/')}/"

    resolver = ReferenceResolver(base_uri=base_uri)

    with open(os.path.join(schemas_dir, "order_schema.json"), "r", encoding="utf-8") as f:
        order_schema = json.load(f)

    validator = JsonSchemaValidator(order_schema, resolver=resolver, base_uri=base_uri)

    valid_order = {
        "id": "ORD-001",
        "customer": {
            "shipping_address": {
                "street": "123 Main St",
                "city": "Beijing",
                "zip": "100000",
            },
        },
        "items": [
            {"name": "Widget", "price": 19.99, "sku": "WID-001"},
            {"name": "Gadget", "price": 29.99, "sku": "GAD-002"},
        ],
        "total": 49.98,
    }
    print_result("5. 文件 $ref 引用 - 通过", validator.validate(valid_order).to_dict())

    invalid_order = {
        "id": "ORD-001",
        "customer": {
            "shipping_address": {
                "street": "123 Main St",
                "city": "Beijing",
                "zip": "abc",
            },
        },
        "items": [
            {"name": "", "price": -10, "sku": "invalid!"},
        ],
        "total": 0,
    }
    print_result("6. 文件 $ref 引用 - 多个错误", validator.validate(invalid_order).to_dict())

    print(f"\n  缓存中的 schema: {list(resolver._cache.keys())}")
    print(f"  (文件引用已缓存，后续校验无需重新加载)")


def demo_circular_ref_detection():
    """演示循环引用检测"""
    circular_schema_a = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "ref_b": {"$ref": "#/definitions/b"},
        },
        "definitions": {
            "b": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "ref_a": {"$ref": "#"},
                },
            },
        },
    }

    print("\n" + "="*70)
    print("  7. 循环引用检测")
    print("="*70)

    try:
        resolver = ReferenceResolver()
        resolver.resolve("#", circular_schema_a, "")
        print("  ✓ 解析顶层 schema 成功（无循环）")
    except ReferenceResolutionError as e:
        print(f"  ✗ 解析错误: {e}")

    try:
        resolver2 = ReferenceResolver()
        resolver2.resolve("#/definitions/b", circular_schema_a, "")
        print("  ✓ 解析 definitions/b 成功")
    except ReferenceResolutionError as e:
        print(f"  ✗ 解析错误: {e}")

    print("\n  --- 显式循环引用测试 ---")
    direct_circular = {
        "definitions": {
            "a": {"$ref": "#/definitions/b"},
            "b": {"$ref": "#/definitions/a"},
        },
        "type": "object",
        "properties": {
            "data": {"$ref": "#/definitions/a"},
        },
    }

    try:
        resolver3 = ReferenceResolver()
        resolver3.resolve("#/definitions/a", direct_circular, "")
        print("  ✗ 应该检测到循环引用，但没有！")
    except ReferenceResolutionError as e:
        print(f"  ✓ 正确检测到循环引用: {e}")

    try:
        validator = JsonSchemaValidator(direct_circular)
        result = validator.validate({"data": {"x": 1}})
        print(f"  验证器处理循环引用: valid={result.valid}, errors={len(result.errors)}")
        for err in result.errors:
            print(f"    - {err.validator}: {err.message}")
    except Exception as e:
        print(f"  验证器抛出异常: {e}")


def demo_ref_in_array_items():
    """演示数组 items 中的 $ref"""
    schema = {
        "definitions": {
            "tag": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "pattern": "^[a-z]+$"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
            },
        },
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "tags": {
                "type": "array",
                "items": {"$ref": "#/definitions/tag"},
                "minItems": 1,
            },
        },
        "required": ["name", "tags"],
    }

    valid_data = {
        "name": "test",
        "tags": [
            {"key": "env", "value": "prod"},
            {"key": "version", "value": "1.0"},
        ],
    }
    print_result("8. 数组 items 中的 $ref - 通过", validate_json(valid_data, schema))

    invalid_data = {
        "name": "test",
        "tags": [
            {"key": "ENV", "value": "prod"},
            {"value": "missing-key"},
        ],
    }
    print_result("9. 数组 items 中的 $ref - 失败", validate_json(invalid_data, schema))


def demo_ref_with_other_keywords():
    """演示 $ref 与其他关键字组合 (注: JSON Schema 中 $ref 通常独占, 但我们先解析 $ref)"""
    schema = {
        "definitions": {
            "positive_int": {
                "type": "integer",
                "minimum": 1,
            },
        },
        "type": "object",
        "properties": {
            "count": {"$ref": "#/definitions/positive_int"},
        },
        "required": ["count"],
    }

    valid_data = {"count": 5}
    print_result("10. $ref 独立使用 - 通过", validate_json(valid_data, schema))

    invalid_data = {"count": 0}
    print_result("11. $ref 独立使用 - 失败", validate_json(invalid_data, schema))


def demo_invalid_ref():
    """演示无效引用的错误处理"""
    schema = {
        "definitions": {
            "valid": {"type": "string"},
        },
        "type": "object",
        "properties": {
            "good": {"$ref": "#/definitions/valid"},
            "bad": {"$ref": "#/definitions/nonexistent"},
        },
    }

    data = {"good": "hello", "bad": "world"}
    print_result("12. 无效 $ref 引用检测", validate_json(data, schema))


def demo_pointer_escaping():
    """演示 JSON Pointer 的 ~1 / ~0 转义"""
    schema = {
        "definitions": {
            "a/b": {"type": "string", "minLength": 5},
            "c~d": {"type": "integer", "minimum": 10},
        },
        "type": "object",
        "properties": {
            "slash": {"$ref": "#/definitions/a~1b"},
            "tilde": {"$ref": "#/definitions/c~0d"},
        },
        "required": ["slash", "tilde"],
    }

    valid_data = {"slash": "hello", "tilde": 20}
    print_result("13. JSON Pointer 转义 (~1, ~0) - 通过", validate_json(valid_data, schema))

    invalid_data = {"slash": "hi", "tilde": 5}
    print_result("14. JSON Pointer 转义 - 失败", validate_json(invalid_data, schema))


def demo_external_ref_with_fragment():
    """演示带片段的外部文件引用"""
    schemas_dir = os.path.join(os.path.dirname(__file__), "schemas")
    base_uri = f"file:///{schemas_dir.replace(os.sep, '/')}/"

    schema = {
        "type": "object",
        "properties": {
            "primary_addr": {
                "$ref": "customer_schema.json#/definitions/address"
            },
        },
        "required": ["primary_addr"],
    }

    resolver = ReferenceResolver(base_uri=base_uri)
    validator = JsonSchemaValidator(schema, resolver=resolver, base_uri=base_uri)

    valid_data = {
        "primary_addr": {
            "street": "123 Main St",
            "city": "Beijing",
            "zip": "100000",
        },
    }
    print_result("15. 带片段的外部文件引用 - 通过", validator.validate(valid_data).to_dict())

    invalid_data = {
        "primary_addr": {
            "street": "123 Main St",
            "city": "Beijing",
            "zip": "abc",
        },
    }
    print_result("16. 带片段的外部文件引用 - 失败", validator.validate(invalid_data).to_dict())


def demo_ref_in_anyof():
    """演示组合关键字中的 $ref"""
    schema = {
        "definitions": {
            "string_id": {"type": "string", "pattern": "^[A-Z]+-\\d+$"},
            "int_id": {"type": "integer", "minimum": 1},
        },
        "type": "object",
        "properties": {
            "id": {
                "anyOf": [
                    {"$ref": "#/definitions/string_id"},
                    {"$ref": "#/definitions/int_id"},
                ]
            },
        },
        "required": ["id"],
    }

    print_result("17. anyOf 中的 $ref - 字符串ID", validate_json({"id": "ABC-123"}, schema))
    print_result("18. anyOf 中的 $ref - 数字ID", validate_json({"id": 456}, schema))
    print_result("19. anyOf 中的 $ref - 都不匹配", validate_json({"id": "invalid"}, schema))


def demo_manual_cache():
    """演示手动缓存 schema"""
    external_schema = {
        "definitions": {
            "coordinate": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number", "minimum": -90, "maximum": 90},
                    "lng": {"type": "number", "minimum": -180, "maximum": 180},
                },
                "required": ["lat", "lng"],
            },
        },
    }

    resolver = ReferenceResolver()
    resolver.cache_schema("https://example.com/geo.json", external_schema)

    schema = {
        "type": "object",
        "properties": {
            "origin": {"$ref": "https://example.com/geo.json#/definitions/coordinate"},
            "destination": {"$ref": "https://example.com/geo.json#/definitions/coordinate"},
        },
        "required": ["origin"],
    }

    validator = JsonSchemaValidator(schema, resolver=resolver)

    valid_data = {
        "origin": {"lat": 39.9, "lng": 116.4},
        "destination": {"lat": 31.2, "lng": 121.5},
    }
    print_result("20. 手动缓存远程 schema - 通过", validator.validate(valid_data).to_dict())

    print(f"\n  缓存内容: {list(resolver._cache.keys())}")
    resolver.clear_cache()
    print(f"  清空后: {list(resolver._cache.keys())}")


if __name__ == "__main__":
    demo_local_ref()
    demo_nested_ref()
    demo_file_ref()
    demo_circular_ref_detection()
    demo_ref_in_array_items()
    demo_ref_with_other_keywords()
    demo_invalid_ref()
    demo_pointer_escaping()
    demo_external_ref_with_fragment()
    demo_ref_in_anyof()
    demo_manual_cache()

    print("\n" + "="*70)
    print("  所有 $ref 引用解析演示完成")
    print("="*70)
