import json
import sys

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from json_schema_validator import (
    JsonSchemaValidator,
    ReferenceResolver,
    ReferenceResolutionError,
    ValidationError,
    detect_draft,
    infer_schema,
    infer_schema_from_samples,
    SchemaInferer,
    CustomValidatorFunc,
)


def print_result(title, data):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    if isinstance(data, dict) and "errors" in data:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(data, indent=2, ensure_ascii=False))


def demo_draft_detection():
    """演示 Schema 草案自动检测"""

    print("\n" + "="*70)
    print("  1. Schema 草案自动检测")
    print("="*70)

    schemas_to_test = [
        ({"$schema": "http://json-schema.org/draft-04/schema#"}, "draft-04 显式声明"),
        ({"$schema": "http://json-schema.org/draft-07/schema#"}, "draft-07 显式声明"),
        ({"$schema": "https://json-schema.org/draft/2020-12/schema"}, "2020-12 显式声明"),
        ({"const": "fixed_value"}, "含 const (推断 draft-06+)"),
        ({"if": {}, "then": {}, "else": {}}, "含 if/then/else (推断 draft-07+)"),
        ({"exclusiveMinimum": 0, "exclusiveMaximum": 100}, "数值型 exclusive* (推断 draft-06+)"),
        ({"type": "object", "properties": {}}, "基础 (默认为 draft-04)"),
    ]

    for schema, description in schemas_to_test:
        draft = detect_draft(schema)
        validator = JsonSchemaValidator(schema)
        print(f"  [{description}]")
        print(f"    检测结果: {draft}")
        print(f"    验证器使用: {validator.draft}")
        print()


def demo_draft_04_exclusive_bool():
    """演示 draft-04 的布尔型 exclusiveMaximum/exclusiveMinimum"""

    print("="*70)
    print("  2. draft-04 布尔型 exclusive 校验")
    print("="*70)

    schema = {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "type": "object",
        "properties": {
            "score": {
                "type": "number",
                "minimum": 0,
                "exclusiveMinimum": True,
                "maximum": 100,
                "exclusiveMaximum": True,
            },
        },
    }

    validator = JsonSchemaValidator(schema)
    print(f"  Schema 草案: {validator.draft}")
    print(f"  exclusive 模式: {validator._feature_mode('exclusiveMaximum')}")

    test_cases = [
        ({"score": 50}, "50 - 通过"),
        ({"score": 0}, "0 - 被 exclusiveMinimum 拒绝"),
        ({"score": 100}, "100 - 被 exclusiveMaximum 拒绝"),
        ({"score": -1}, "-1 - 小于 minimum"),
    ]

    for data, desc in test_cases:
        result = validator.validate(data)
        status = "PASS" if result.valid else "FAIL"
        print(f"  [{status}] {desc}")


def demo_draft_feature_control():
    """演示不同草案的特性控制"""

    print("\n" + "="*70)
    print("  3. 不同草案的特性控制")
    print("="*70)

    schema_with_const = {
        "type": "object",
        "properties": {
            "version": {"const": "1.0"},
        },
    }

    print("  测试 const 关键字在不同草案中的行为:")

    validator_04 = JsonSchemaValidator(schema_with_const, draft="draft-04")
    validator_07 = JsonSchemaValidator(schema_with_const, draft="draft-07")

    data = {"version": "2.0"}
    result_04 = validator_04.validate(data)
    result_07 = validator_07.validate(data)

    print(f"    draft-04: const 被忽略 -> valid={result_04.valid}")
    print(f"    draft-07: const 生效 -> valid={result_07.valid}")


def demo_custom_keyword_global():
    """演示全局自定义关键字"""

    print("\n" + "="*70)
    print("  4. 全局自定义关键字校验")
    print("="*70)

    def validate_password_strength(
        data: Any,
        schema_value: Any,
        path: str,
        validator: JsonSchemaValidator,
        errors: list,
    ) -> None:
        if not isinstance(data, str):
            return
        if schema_value is True:
            has_upper = any(c.isupper() for c in data)
            has_lower = any(c.islower() for c in data)
            has_digit = any(c.isdigit() for c in data)
            has_special = any(c in "!@#$%^&*" for c in data)
            if not (has_upper and has_lower and has_digit and has_special):
                errors.append(
                    ValidationError(
                        path=path,
                        message="Password must contain uppercase, lowercase, digit, and special character",
                        validator="passwordStrength",
                        value=data,
                        schema=schema_value,
                    )
                )

    JsonSchemaValidator.register_keyword("passwordStrength", validate_password_strength)

    schema = {
        "type": "object",
        "properties": {
            "password": {
                "type": "string",
                "minLength": 8,
                "passwordStrength": True,
            },
        },
        "required": ["password"],
    }

    validator = JsonSchemaValidator(schema)

    test_cases = [
        ({"password": "weak"}, "弱密码 - 失败"),
        ({"password": "Strong1!"}, "强密码 - 通过"),
        ({"password": "Short1!"}, "长度不足 - 失败"),
        ({"password": "nouppercase1!"}, "无大写 - 失败"),
    ]

    for data, desc in test_cases:
        result = validator.validate(data)
        status = "PASS" if result.valid else "FAIL"
        print(f"  [{status}] {desc}")
        for err in result.errors[:1]:
            print(f"         {err.message[:60]}")


def demo_custom_keyword_local():
    """演示实例级自定义关键字"""

    print("\n" + "="*70)
    print("  5. 实例级自定义关键字 (不影响全局)")
    print("="*70)

    def validate_even_number(
        data: Any,
        schema_value: Any,
        path: str,
        validator: JsonSchemaValidator,
        errors: list,
    ) -> None:
        if not isinstance(data, (int, float)):
            return
        if schema_value is True and isinstance(data, int) and data % 2 != 0:
            errors.append(
                ValidationError(
                    path=path,
                    message=f"Number {data} must be even",
                    validator="evenNumber",
                    value=data,
                    schema=schema_value,
                )
            )

    schema = {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "evenNumber": True},
        },
    }

    validator1 = JsonSchemaValidator(schema)
    validator1.register_local_keyword("evenNumber", validate_even_number)

    validator2 = JsonSchemaValidator(schema)

    data = {"count": 3}

    result1 = validator1.validate(data)
    result2 = validator2.validate(data)

    print(f"  数据: {data}")
    print(f"  有自定义关键字的验证器: valid={result1.valid}")
    for err in result1.errors:
        print(f"    -> {err.message}")
    print(f"  无自定义关键字的验证器: valid={result2.valid}")


def demo_schema_inference_single():
    """演示从单个数据推断 Schema"""

    print("\n" + "="*70)
    print("  6. 从单个数据推断 Schema")
    print("="*70)

    data = {
        "id": 12345,
        "username": "john_doe",
        "email": "john@example.com",
        "age": 25,
        "active": True,
        "scores": [95.5, 87.3, 92.0],
        "profile": {
            "first_name": "John",
            "last_name": "Doe",
            "birthday": "1998-05-15",
        },
        "uuid": "550e8400-e29b-41d4-a716-446655440000",
    }

    inferred = infer_schema(data)
    print(json.dumps(inferred, indent=2, ensure_ascii=False))


def demo_schema_inference_multiple():
    """演示从多个样本数据推断 Schema"""

    print("\n" + "="*70)
    print("  7. 从多个样本推断 Schema (合并)")
    print("="*70)

    samples = [
        {"id": 1, "name": "Alice", "age": 25, "role": "user"},
        {"id": 2, "name": "Bob Smith", "age": 30},
        {"id": 3, "name": "Carol", "age": 28, "role": "admin", "extra": "data"},
    ]

    inferred = infer_schema_from_samples(samples)
    print("  三个样本:")
    for i, s in enumerate(samples):
        print(f"    {i+1}. {s}")
    print("\n  推断结果:")
    print(json.dumps(inferred, indent=2, ensure_ascii=False))

    print(f"\n  注意:")
    print(f"    - name 的 minLength={inferred['properties']['name']['minLength']}")
    print(f"    - name 的 maxLength={inferred['properties']['name']['maxLength']}")
    print(f"    - age 的 minimum={inferred['properties']['age']['minimum']}")
    print(f"    - age 的 maximum={inferred['properties']['age']['maximum']}")
    print(f"    - required={inferred.get('required')}")


def demo_schema_inference_types():
    """演示不同类型的推断"""

    print("\n" + "="*70)
    print("  8. 各种数据类型的 Schema 推断")
    print("="*70)

    test_cases = [
        ("null", None),
        ("boolean", True),
        ("integer", 42),
        ("number", 3.14),
        ("string", "hello world"),
        ("email string", "user@example.com"),
        ("date string", "2025-01-15"),
        ("uuid string", "550e8400-e29b-41d4-a716-446655440000"),
        ("array", [1, 2, 3]),
        ("object", {"x": 1, "y": 2}),
    ]

    for name, data in test_cases:
        inferred = infer_schema(data, include_schema_uri=False)
        inferred.pop("$inferred", None)
        print(f"  {name}: {json.dumps(inferred, ensure_ascii=False)}")


def demo_inferer_class():
    """演示 SchemaInferer 类的流式使用"""

    print("\n" + "="*70)
    print("  9. SchemaInferer 流式添加样本")
    print("="*70)

    inferer = SchemaInferer(
        detect_formats=True,
        detect_ranges=True,
        infer_required=True,
    )

    stream_data = [
        {"type": "event", "timestamp": 1700000000, "level": "INFO"},
        {"type": "event", "timestamp": 1700000001, "level": "DEBUG"},
        {"type": "event", "timestamp": 1700000002, "level": "WARN", "message": "warning text"},
        {"type": "event", "timestamp": 1700000003, "level": "ERROR", "message": "error occurred"},
    ]

    for i, data in enumerate(stream_data):
        inferer.add_sample(data)
        if i == 1 or i == 3:
            partial = inferer.to_schema(include_schema_uri=False)
            partial.pop("$inferred", None)
            print(f"  第 {i+1} 个样本后:")
            print(f"    {json.dumps(partial, ensure_ascii=False)}")

    print(f"\n  最终结果:")
    final = inferer.to_schema()
    print(json.dumps(final, indent=2, ensure_ascii=False))


def demo_infer_then_validate():
    """演示推断 Schema 后立即用于验证"""

    print("\n" + "="*70)
    print("  10. 推断 Schema 后立即用于验证")
    print("="*70)

    good_data = {"id": 1, "name": "Alice", "age": 25}
    inferred = infer_schema(good_data)
    print("  学习数据:", good_data)
    print("  推断 Schema 并验证新数据:")

    validator = JsonSchemaValidator(inferred)

    test_cases = [
        ({"id": 2, "name": "Bob", "age": 30}, "符合推断 - PASS"),
        ({"id": "wrong", "name": "Charlie", "age": 35}, "id 类型错误 - FAIL"),
        ({"id": 3, "age": 40}, "缺少 name - FAIL"),
    ]

    for data, desc in test_cases:
        result = validator.validate(data)
        status = "PASS" if result.valid else "FAIL"
        print(f"    [{status}] {desc}")
        if not result.valid:
            for err in result.errors:
                print(f"        -> {err.message}")


def demo_custom_keyword_business_rule():
    """演示复杂的业务规则自定义关键字"""

    print("\n" + "="*70)
    print("  11. 复杂业务规则自定义关键字")
    print("="*70)

    def validate_order_total(
        data: Any,
        schema_value: Any,
        path: str,
        validator: JsonSchemaValidator,
        errors: list,
    ) -> None:
        if not isinstance(data, dict):
            return
        items = data.get("items", [])
        total = data.get("total", 0)
        if not isinstance(items, list) or not isinstance(total, (int, float)):
            return

        calculated_total = sum(
            item.get("price", 0) * item.get("quantity", 1)
            for item in items
            if isinstance(item, dict)
        )

        tolerance = schema_value if isinstance(schema_value, (int, float)) else 0.01
        if abs(calculated_total - total) > tolerance:
            errors.append(
                ValidationError(
                    path=path,
                    message=f"Order total {total} does not match calculated {calculated_total}",
                    validator="orderTotalMatchesItems",
                    value=data,
                    schema=schema_value,
                )
            )

    schema = {
        "type": "object",
        "orderTotalMatchesItems": 0.01,
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "price": {"type": "number", "minimum": 0},
                        "quantity": {"type": "integer", "minimum": 1},
                    },
                    "required": ["name", "price"],
                },
            },
            "total": {"type": "number", "minimum": 0},
        },
        "required": ["items", "total"],
    }

    JsonSchemaValidator.register_keyword("orderTotalMatchesItems", validate_order_total)
    validator = JsonSchemaValidator(schema)

    valid_order = {
        "items": [
            {"name": "Apple", "price": 1.5, "quantity": 2},
            {"name": "Banana", "price": 0.75, "quantity": 3},
        ],
        "total": 5.25,
    }

    invalid_order = {
        "items": [
            {"name": "Apple", "price": 1.5, "quantity": 2},
            {"name": "Banana", "price": 0.75, "quantity": 3},
        ],
        "total": 10.00,
    }

    result1 = validator.validate(valid_order)
    result2 = validator.validate(invalid_order)

    print(f"  正确订单: valid={result1.valid}")
    print(f"  错误订单: valid={result2.valid}")
    for err in result2.errors:
        print(f"    -> {err.message}")


if __name__ == "__main__":
    demo_draft_detection()
    demo_draft_04_exclusive_bool()
    demo_draft_feature_control()
    demo_custom_keyword_global()
    demo_custom_keyword_local()
    demo_schema_inference_single()
    demo_schema_inference_multiple()
    demo_schema_inference_types()
    demo_inferer_class()
    demo_infer_then_validate()
    demo_custom_keyword_business_rule()

    print("\n" + "="*70)
    print("  所有新功能演示完成")
    print("="*70)
