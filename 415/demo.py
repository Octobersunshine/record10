import json
from json_schema_validator import validate, validate_json, JsonSchemaValidator


def print_result(title, result_dict):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(json.dumps(result_dict, indent=2, ensure_ascii=False))


def demo_type_validation():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "score": {"type": "number"},
            "active": {"type": "boolean"},
            "tags": {"type": "array"},
            "metadata": {"type": "object"},
        },
        "required": ["name", "age"],
    }

    valid_data = {
        "name": "Alice",
        "age": 30,
        "score": 95.5,
        "active": True,
        "tags": ["admin"],
        "metadata": {"role": "lead"},
    }
    print_result("1. 类型校验 - 通过", validate_json(valid_data, schema))

    invalid_data = {
        "name": 123,
        "age": "thirty",
        "score": "high",
    }
    print_result("2. 类型校验 - 失败", validate_json(invalid_data, schema))


def demo_required_fields():
    schema = {
        "type": "object",
        "required": ["username", "email", "password"],
        "properties": {
            "username": {"type": "string"},
            "email": {"type": "string", "format": "email"},
            "password": {"type": "string", "minLength": 8},
        },
    }

    valid_data = {
        "username": "bob",
        "email": "bob@example.com",
        "password": "secure123",
    }
    print_result("3. 必填字段校验 - 通过", validate_json(valid_data, schema))

    invalid_data = {"username": "bob", "email": "bob@example.com"}
    print_result("4. 必填字段校验 - 缺少password", validate_json(invalid_data, schema))


def demo_numeric_range():
    schema = {
        "type": "object",
        "properties": {
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "score": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 100},
            "quantity": {"type": "integer", "multipleOf": 5},
        },
    }

    valid_data = {"age": 25, "score": 99.9, "quantity": 20}
    print_result("5. 数值范围校验 - 通过", validate_json(valid_data, schema))

    invalid_data = {"age": -1, "score": 0, "quantity": 13}
    print_result("6. 数值范围校验 - 失败", validate_json(invalid_data, schema))


def demo_string_pattern():
    schema = {
        "type": "object",
        "properties": {
            "phone": {"type": "string", "pattern": r"^\d{3}-\d{4}-\d{4}$"},
            "email": {"type": "string", "format": "email"},
            "website": {"type": "string", "format": "uri"},
            "bio": {"type": "string", "minLength": 10, "maxLength": 200},
            "date": {"type": "string", "format": "date"},
        },
    }

    valid_data = {
        "phone": "138-1234-5678",
        "email": "user@example.com",
        "website": "https://example.com",
        "bio": "A passionate developer",
        "date": "2025-01-15",
    }
    print_result("7. 字符串模式校验 - 通过", validate_json(valid_data, schema))

    invalid_data = {
        "phone": "abc-xxxx-yyyy",
        "email": "not-an-email",
        "website": "not a url",
        "bio": "Hi",
        "date": "2025/01/15",
    }
    print_result("8. 字符串模式校验 - 失败", validate_json(invalid_data, schema))


def demo_enum_const():
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["active", "inactive", "pending"]},
            "version": {"const": "1.0"},
        },
    }

    valid_data = {"status": "active", "version": "1.0"}
    print_result("9. 枚举/常量校验 - 通过", validate_json(valid_data, schema))

    invalid_data = {"status": "deleted", "version": "2.0"}
    print_result("10. 枚举/常量校验 - 失败", validate_json(invalid_data, schema))


def demo_nested_and_array():
    schema = {
        "type": "object",
        "required": ["name", "orders"],
        "properties": {
            "name": {"type": "string"},
            "orders": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["id", "amount"],
                    "properties": {
                        "id": {"type": "string", "pattern": r"^ORD-\d+$"},
                        "amount": {"type": "number", "minimum": 0},
                        "items": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string"},
                        },
                    },
                },
            },
            "address": {
                "type": "object",
                "required": ["city"],
                "properties": {
                    "city": {"type": "string"},
                    "zip": {"type": "string", "pattern": r"^\d{6}$"},
                },
                "additionalProperties": False,
            },
        },
    }

    valid_data = {
        "name": "Charlie",
        "orders": [
            {"id": "ORD-001", "amount": 99.9, "items": ["book"]},
        ],
        "address": {"city": "Beijing", "zip": "100000"},
    }
    print_result("11. 嵌套对象/数组校验 - 通过", validate_json(valid_data, schema))

    invalid_data = {
        "name": "Charlie",
        "orders": [],
        "address": {"city": 123, "zip": "abc", "extra": "not allowed"},
    }
    print_result("12. 嵌套对象/数组校验 - 失败", validate_json(invalid_data, schema))


def demo_combinators():
    schema = {
        "type": "object",
        "properties": {
            "priority": {
                "anyOf": [
                    {"type": "integer", "minimum": 1, "maximum": 5},
                    {"type": "string", "enum": ["low", "medium", "high"]},
                ]
            },
            "config": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "object"},
                ]
            },
            "nickname": {
                "not": {"const": "admin"},
            },
        },
    }

    valid_data = {"priority": 3, "config": "simple", "nickname": "user1"}
    print_result("13. 组合校验(anyOf/oneOf/not) - 通过", validate_json(valid_data, schema))

    invalid_data = {"priority": 0, "config": 123, "nickname": "admin"}
    print_result("14. 组合校验(anyOf/oneOf/not) - 失败", validate_json(invalid_data, schema))


def demo_if_then_else():
    schema = {
        "type": "object",
        "properties": {
            "member_type": {"type": "string"},
            "discount": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "if": {
            "properties": {"member_type": {"const": "premium"}},
        },
        "then": {
            "properties": {"discount": {"type": "number", "minimum": 0.1}},
        },
        "else": {
            "properties": {"discount": {"type": "number", "maximum": 0.05}},
        },
    }

    valid_data = {"member_type": "premium", "discount": 0.2}
    print_result("15. 条件校验(if/then/else) - Premium通过", validate_json(valid_data, schema))

    invalid_data = {"member_type": "premium", "discount": 0.02}
    print_result("16. 条件校验(if/then/else) - Premium折扣太低", validate_json(invalid_data, schema))

    valid_normal = {"member_type": "normal", "discount": 0.03}
    print_result("17. 条件校验(if/then/else) - Normal通过", validate_json(valid_normal, schema))

    invalid_normal = {"member_type": "normal", "discount": 0.5}
    print_result("18. 条件校验(if/then/else) - Normal折扣太高", validate_json(invalid_normal, schema))


def demo_result_api():
    schema = {"type": "object", "required": ["id"], "properties": {"id": {"type": "integer"}}}
    result = validate({"id": "not-int"}, schema)
    print(f"\nbool(result) => {bool(result)}")
    print(f"result.valid => {result.valid}")
    print(f"result.errors => {result.errors}")
    for err in result.errors:
        print(f"  - path={err.path}, msg={err.message}, validator={err.validator}")


if __name__ == "__main__":
    demo_type_validation()
    demo_required_fields()
    demo_numeric_range()
    demo_string_pattern()
    demo_enum_const()
    demo_nested_and_array()
    demo_combinators()
    demo_if_then_else()
    demo_result_api()
