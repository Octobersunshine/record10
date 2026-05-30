import requests
import json

BASE_URL = "http://127.0.0.1:5000"


def test_health_check():
    print("=== 测试健康检查接口 ===")
    try:
        response = requests.get(f"{BASE_URL}/api/mock/health")
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
        assert response.status_code == 200
        print("✅ 健康检查测试通过\n")
    except Exception as e:
        print(f"❌ 健康检查测试失败: {e}\n")


def test_schema_info():
    print("=== 测试Schema信息接口 ===")
    try:
        response = requests.get(f"{BASE_URL}/api/mock/schema")
        print(f"状态码: {response.status_code}")
        data = response.json()
        print(f"支持的数据类型: {data['data']['supported_types']}")
        print(f"支持的Faker方法数量: {len(data['data']['faker_methods'])}")
        assert response.status_code == 200
        print("✅ Schema信息测试通过\n")
    except Exception as e:
        print(f"❌ Schema信息测试失败: {e}\n")


def test_generate_user_data():
    print("=== 测试生成用户数据（姓名、地址、邮箱、手机号、日期） ===")
    schema = {
        "type": "object",
        "properties": {
            "id": {
                "type": "integer",
                "faker": "random_int"
            },
            "name": {
                "type": "string",
                "title": "姓名"
            },
            "email": {
                "type": "string",
                "format": "email"
            },
            "phone": {
                "type": "string",
                "title": "手机号"
            },
            "address": {
                "type": "string",
                "title": "地址"
            },
            "birthday": {
                "type": "string",
                "format": "date"
            },
            "created_at": {
                "type": "string",
                "format": "date-time"
            }
        }
    }

    payload = {
        "schema": schema,
        "count": 5
    }

    try:
        response = requests.post(
            f"{BASE_URL}/api/mock/generate",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"状态码: {response.status_code}")
        data = response.json()
        print(f"生成数据条数: {data['total']}")
        print(f"第一条数据: {json.dumps(data['data'][0], ensure_ascii=False, indent=2)}")
        assert response.status_code == 200
        assert data['total'] == 5
        print("✅ 用户数据生成测试通过\n")
    except Exception as e:
        print(f"❌ 用户数据生成测试失败: {e}\n")


def test_generate_with_faker_config():
    print("=== 测试使用Faker配置生成数据 ===")
    schema = {
        "type": "object",
        "properties": {
            "username": {
                "type": "string",
                "faker": "user_name"
            },
            "age": {
                "type": "integer",
                "faker": {
                    "type": "random_int",
                    "params": {"min": 18, "max": 65}
                }
            },
            "company": {
                "type": "string",
                "faker": "company"
            },
            "job": {
                "type": "string",
                "faker": "job"
            },
            "score": {
                "type": "number",
                "faker": {
                    "type": "random_int",
                    "params": {"min": 0, "max": 100}
                }
            },
            "is_active": {
                "type": "boolean"
            },
            "tags": {
                "type": "array",
                "items": {
                    "type": "string",
                    "faker": "word"
                },
                "minItems": 2,
                "maxItems": 5
            }
        }
    }

    payload = {
        "schema": schema,
        "count": 3
    }

    try:
        response = requests.post(
            f"{BASE_URL}/api/mock/generate",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"状态码: {response.status_code}")
        data = response.json()
        print(f"生成数据条数: {data['total']}")
        for i, item in enumerate(data['data']):
            print(f"第{i+1}条 - 用户名: {item['username']}, 年龄: {item['age']}, 公司: {item['company']}, 标签数量: {len(item['tags'])}")
        assert response.status_code == 200
        assert data['total'] == 3
        print("✅ Faker配置数据生成测试通过\n")
    except Exception as e:
        print(f"❌ Faker配置数据生成测试失败: {e}\n")


def test_generate_100_records():
    print("=== 测试生成100条数据（最大限制） ===")
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "title": "姓名"},
            "email": {"type": "string", "format": "email"}
        }
    }

    payload = {
        "schema": schema,
        "count": 100
    }

    try:
        response = requests.post(
            f"{BASE_URL}/api/mock/generate",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"状态码: {response.status_code}")
        data = response.json()
        print(f"生成数据条数: {data['total']}")
        assert response.status_code == 200
        assert data['total'] == 100
        print("✅ 100条数据生成测试通过\n")
    except Exception as e:
        print(f"❌ 100条数据生成测试失败: {e}\n")


def test_invalid_count():
    print("=== 测试无效数量参数 ===")
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "title": "姓名"}
        }
    }

    payload = {
        "schema": schema,
        "count": 200
    }

    try:
        response = requests.post(
            f"{BASE_URL}/api/mock/generate",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"状态码: {response.status_code}")
        data = response.json()
        print(f"错误信息: {data['message']}")
        assert response.status_code == 400
        print("✅ 无效数量参数测试通过\n")
    except Exception as e:
        print(f"❌ 无效数量参数测试失败: {e}\n")


def test_missing_schema():
    print("=== 测试缺少schema参数 ===")
    payload = {
        "count": 5
    }

    try:
        response = requests.post(
            f"{BASE_URL}/api/mock/generate",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"状态码: {response.status_code}")
        data = response.json()
        print(f"错误信息: {data['message']}")
        assert response.status_code == 400
        print("✅ 缺少schema参数测试通过\n")
    except Exception as e:
        print(f"❌ 缺少schema参数测试失败: {e}\n")


def test_chinese_property_names():
    print("=== 测试中文属性名自动识别 ===")
    schema = {
        "type": "object",
        "properties": {
            "用户ID": {
                "type": "integer",
                "faker": "random_int"
            },
            "姓名": {
                "type": "string"
            },
            "邮箱": {
                "type": "string"
            },
            "手机号": {
                "type": "string"
            },
            "地址": {
                "type": "string"
            },
            "日期": {
                "type": "string"
            }
        }
    }

    payload = {
        "schema": schema,
        "count": 3
    }

    try:
        response = requests.post(
            f"{BASE_URL}/api/mock/generate",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"状态码: {response.status_code}")
        data = response.json()
        print(f"生成数据条数: {data['total']}")
        print(f"第一条数据: {json.dumps(data['data'][0], ensure_ascii=False, indent=2)}")
        assert response.status_code == 200
        assert data['total'] == 3
        print("✅ 中文属性名自动识别测试通过\n")
    except Exception as e:
        print(f"❌ 中文属性名自动识别测试失败: {e}\n")


def test_id_auto_uuid():
    print("=== 测试ID字段自动生成UUID ===")
    schema = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string"
            },
            "name": {
                "type": "string",
                "title": "姓名"
            }
        }
    }

    payload = {
        "schema": schema,
        "count": 5
    }

    try:
        response = requests.post(
            f"{BASE_URL}/api/mock/generate",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"状态码: {response.status_code}")
        data = response.json()
        print(f"生成数据条数: {data['total']}")
        
        ids = [item['id'] for item in data['data']]
        unique_ids = set(ids)
        
        print(f"所有ID: {ids[:2]}...")
        print(f"ID数量: {len(ids)}, 唯一ID数量: {len(unique_ids)}")
        
        assert response.status_code == 200
        assert data['total'] == 5
        assert len(ids) == len(unique_ids), "ID存在重复！"
        for id_val in ids:
            assert len(id_val) == 36, f"ID {id_val} 不是有效的UUID格式"
        print("✅ ID字段自动生成UUID测试通过\n")
    except Exception as e:
        print(f"❌ ID字段自动生成UUID测试失败: {e}\n")


def test_unique_constraint():
    print("=== 测试unique唯一性约束 ===")
    schema = {
        "type": "object",
        "properties": {
            "username": {
                "type": "string",
                "faker": "user_name",
                "unique": True
            },
            "email": {
                "type": "string",
                "format": "email",
                "unique": True
            },
            "name": {
                "type": "string",
                "title": "姓名"
            }
        }
    }

    payload = {
        "schema": schema,
        "count": 20
    }

    try:
        response = requests.post(
            f"{BASE_URL}/api/mock/generate",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"状态码: {response.status_code}")
        data = response.json()
        print(f"生成数据条数: {data['total']}")
        
        usernames = [item['username'] for item in data['data']]
        emails = [item['email'] for item in data['data']]
        
        unique_usernames = set(usernames)
        unique_emails = set(emails)
        
        print(f"用户名数量: {len(usernames)}, 唯用户名数量: {len(unique_usernames)}")
        print(f"邮箱数量: {len(emails)}, 唯一邮箱数量: {len(unique_emails)}")
        
        assert response.status_code == 200
        assert data['total'] == 20
        assert len(usernames) == len(unique_usernames), "用户名存在重复！"
        assert len(emails) == len(unique_emails), "邮箱存在重复！"
        print("✅ unique唯一性约束测试通过\n")
    except Exception as e:
        print(f"❌ unique唯一性约束测试失败: {e}\n")


def test_100_unique_ids():
    print("=== 测试100条数据的ID唯一性（压力测试） ===")
    schema = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string"
            },
            "uuid": {
                "type": "string"
            },
            "name": {
                "type": "string",
                "title": "姓名"
            }
        }
    }

    payload = {
        "schema": schema,
        "count": 100
    }

    try:
        response = requests.post(
            f"{BASE_URL}/api/mock/generate",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"状态码: {response.status_code}")
        data = response.json()
        print(f"生成数据条数: {data['total']}")
        
        ids = [item['id'] for item in data['data']]
        uuids = [item['uuid'] for item in data['data']]
        
        unique_ids = set(ids)
        unique_uuids = set(uuids)
        
        print(f"ID数量: {len(ids)}, 唯一ID数量: {len(unique_ids)}")
        print(f"UUID数量: {len(uuids)}, 唯一UUID数量: {len(unique_uuids)}")
        
        assert response.status_code == 200
        assert data['total'] == 100
        assert len(ids) == len(unique_ids), "id字段存在重复！"
        assert len(uuids) == len(unique_uuids), "uuid字段存在重复！"
        print("✅ 100条数据ID唯一性测试通过\n")
    except Exception as e:
        print(f"❌ 100条数据ID唯一性测试失败: {e}\n")


def test_chinese_id_field_names():
    print("=== 测试中文ID字段名自动识别 ===")
    schema = {
        "type": "object",
        "properties": {
            "编号": {
                "type": "string"
            },
            "标识": {
                "type": "string"
            },
            "姓名": {
                "type": "string"
            }
        }
    }

    payload = {
        "schema": schema,
        "count": 5
    }

    try:
        response = requests.post(
            f"{BASE_URL}/api/mock/generate",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"状态码: {response.status_code}")
        data = response.json()
        print(f"生成数据条数: {data['total']}")
        print(f"第一条数据: {json.dumps(data['data'][0], ensure_ascii=False, indent=2)}")
        
        ids = [item['编号'] for item in data['data']]
        unique_ids = set(ids)
        
        assert response.status_code == 200
        assert data['total'] == 5
        assert len(ids) == len(unique_ids), "编号字段存在重复！"
        for id_val in ids:
            assert len(id_val) == 36, f"编号 {id_val} 不是有效的UUID格式"
        print("✅ 中文ID字段名自动识别测试通过\n")
    except Exception as e:
        print(f"❌ 中文ID字段名自动识别测试失败: {e}\n")


def test_eval_expression():
    print("=== 测试Python表达式生成数据 ===")
    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "price": {"type": "number", "faker": {"type": "random_int", "params": {"min": 10, "max": 100}}},
            "quantity": {"type": "integer", "faker": {"type": "random_int", "params": {"min": 1, "max": 10}}},
            "total": {"type": "number", "eval": "price * quantity"},
            "discount": {"type": "number", "eval": "total * 0.1 if total > 100 else 0"},
            "final_price": {"type": "number", "eval": "total - discount"},
            "order_no": {"type": "string", "eval": "'ORD' + str(index + 1000)"},
            "custom_code": {"type": "string", "eval": "fake.bothify(text='??-#####')"}
        }
    }

    payload = {
        "schema": schema,
        "count": 5
    }

    try:
        response = requests.post(
            f"{BASE_URL}/api/mock/generate",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"状态码: {response.status_code}")
        data = response.json()
        print(f"生成数据条数: {data['total']}")
        
        for i, item in enumerate(data['data'][:2]):
            print(f"第{i+1}条 - 价格:{item['price']}, 数量:{item['quantity']}, 总价:{item['total']}, 折扣:{item['discount']}, 实付:{item['final_price']}, 订单号:{item['order_no']}")
            assert item['total'] == item['price'] * item['quantity'], "总价计算错误"
            assert item['final_price'] == item['total'] - item['discount'], "实付计算错误"
        
        assert response.status_code == 200
        assert data['total'] == 5
        print("✅ Python表达式生成数据测试通过\n")
    except Exception as e:
        print(f"❌ Python表达式生成数据测试失败: {e}\n")


def test_list_templates():
    print("=== 测试获取模板列表 ===")
    try:
        response = requests.get(f"{BASE_URL}/api/mock/templates")
        print(f"状态码: {response.status_code}")
        data = response.json()
        print(f"模板数量: {data['total']}")
        print(f"模板列表: {list(data['data'].keys())}")
        
        assert response.status_code == 200
        assert data['total'] > 0
        assert 'user' in data['data']
        assert 'order' in data['data']
        assert 'product' in data['data']
        print("✅ 获取模板列表测试通过\n")
    except Exception as e:
        print(f"❌ 获取模板列表测试失败: {e}\n")


def test_get_template_detail():
    print("=== 测试获取模板详情 ===")
    try:
        response = requests.get(f"{BASE_URL}/api/mock/template/user")
        print(f"状态码: {response.status_code}")
        data = response.json()
        print(f"模板名称: {data['data']['name']}")
        print(f"模板描述: {data['data']['description']}")
        print(f"Schema字段数量: {len(data['data']['schema']['properties'])}")
        
        assert response.status_code == 200
        assert data['data']['name'] == '用户表'
        print("✅ 获取模板详情测试通过\n")
    except Exception as e:
        print(f"❌ 获取模板详情测试失败: {e}\n")


def test_generate_by_template():
    print("=== 测试使用模板生成数据 ===")
    try:
        payload = {
            "template": "user",
            "count": 3
        }
        response = requests.post(
            f"{BASE_URL}/api/mock/generate/template",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"状态码: {response.status_code}")
        data = response.json()
        print(f"生成数据条数: {data['total']}")
        print(f"使用模板: {data['template']}")
        print(f"第一条数据字段: {list(data['data'][0].keys())}")
        
        assert response.status_code == 200
        assert data['total'] == 3
        assert 'id' in data['data'][0]
        assert 'username' in data['data'][0]
        assert 'email' in data['data'][0]
        print("✅ 使用模板生成数据测试通过\n")
    except Exception as e:
        print(f"❌ 使用模板生成数据测试失败: {e}\n")


def test_ref_data_with_refs():
    print("=== 测试使用refs参数关联数据 ===")
    user_ids = [
        "user-" + str(i) for i in range(1, 6)
    ]
    
    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "user_id": {"type": "string", "ref": "user_ids"},
            "order_no": {"type": "string", "eval": "'ORD' + str(random.randint(10000, 99999))"},
            "amount": {"type": "number", "faker": {"type": "random_int", "params": {"min": 10, "max": 1000}}}
        }
    }

    payload = {
        "schema": schema,
        "count": 10,
        "refs": {
            "user_ids": user_ids
        }
    }

    try:
        response = requests.post(
            f"{BASE_URL}/api/mock/generate",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"状态码: {response.status_code}")
        data = response.json()
        print(f"生成数据条数: {data['total']}")
        
        generated_user_ids = [item['user_id'] for item in data['data']]
        print(f"生成的user_id示例: {generated_user_ids[:5]}")
        print(f"所有user_id都在预设列表中: {all(uid in user_ids for uid in generated_user_ids)}")
        
        assert response.status_code == 200
        assert data['total'] == 10
        assert all(uid in user_ids for uid in generated_user_ids), "存在不在预设列表中的user_id"
        print("✅ 使用refs参数关联数据测试通过\n")
    except Exception as e:
        print(f"❌ 使用refs参数关联数据测试失败: {e}\n")


def test_order_template_with_refs():
    print("=== 测试订单模板关联用户数据 ===")
    try:
        user_response = requests.post(
            f"{BASE_URL}/api/mock/generate/template",
            json={"template": "user", "count": 5},
            headers={"Content-Type": "application/json"}
        )
        users = user_response.json()['data']
        user_ids = [u['id'] for u in users]
        print(f"生成用户数量: {len(users)}")
        
        order_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "order_no": {"type": "string", "unique": True, "eval": "'ORD' + str(uuid.uuid4())[:8].upper() + str(random.randint(1000, 9999))"},
                "user_id": {"type": "string", "ref": "users"},
                "product_name": {"type": "string", "faker": "word"},
                "total_amount": {"type": "number", "faker": {"type": "random_int", "params": {"min": 10, "max": 9999}}},
                "status": {"type": "string", "enum": ["待支付", "已支付", "已发货", "已完成"]}
            }
        }

        payload = {
            "schema": order_schema,
            "count": 20,
            "refs": {
                "users": user_ids
            }
        }
        
        response = requests.post(
            f"{BASE_URL}/api/mock/generate",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"状态码: {response.status_code}")
        data = response.json()
        print(f"生成订单数量: {data['total']}")
        
        order_user_ids = [item['user_id'] for item in data['data']]
        print(f"订单关联的用户ID都有效: {all(uid in user_ids for uid in order_user_ids)}")
        print(f"关联到的不同用户数: {len(set(order_user_ids))}")
        
        assert response.status_code == 200
        assert data['total'] == 20
        assert all(uid in user_ids for uid in order_user_ids), "存在无效的用户ID"
        print("✅ 订单模板关联用户数据测试通过\n")
    except Exception as e:
        print(f"❌ 订单模板关联用户数据测试失败: {e}\n")


def test_generate_related_data():
    print("=== 测试关联数据生成接口 ===")
    from mock_generator import get_template
    
    user_template = get_template('user')
    order_template = get_template('order')
    
    order_schema = order_template['schema'].copy()
    order_schema['properties']['user_id']['ref'] = {'name': 'users', 'field': 'id'}
    order_schema['properties']['product_id']['ref'] = {'name': 'products', 'field': 'id'}
    
    payload = {
        "relations": [
            {
                "name": "users",
                "schema": user_template['schema'],
                "count": 5
            },
            {
                "name": "products",
                "schema": get_template('product')['schema'],
                "count": 10
            },
            {
                "name": "orders",
                "schema": order_schema,
                "count": 20,
                "refs": {
                    "users": "users",
                    "products": "products"
                }
            }
        ]
    }

    try:
        response = requests.post(
            f"{BASE_URL}/api/mock/generate/related",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"状态码: {response.status_code}")
        data = response.json()
        print(f"生成的数据组: {list(data['totals'].keys())}")
        print(f"各数据量: {data['totals']}")
        
        users = data['data']['users']
        products = data['data']['products']
        orders = data['data']['orders']
        
        user_ids = [u['id'] for u in users]
        product_ids = [p['id'] for p in products]
        order_user_ids = [o['user_id'] for o in orders]
        order_product_ids = [o['product_id'] for o in orders]
        
        print(f"订单关联用户有效: {all(uid in user_ids for uid in order_user_ids)}")
        print(f"订单关联商品有效: {all(pid in product_ids for pid in order_product_ids)}")
        
        assert response.status_code == 200
        assert data['totals']['users'] == 5
        assert data['totals']['products'] == 10
        assert data['totals']['orders'] == 20
        assert all(uid in user_ids for uid in order_user_ids), "存在无效的用户ID关联"
        assert all(pid in product_ids for pid in order_product_ids), "存在无效的商品ID关联"
        print("✅ 关联数据生成接口测试通过\n")
    except Exception as e:
        print(f"❌ 关联数据生成接口测试失败: {e}\n")


if __name__ == "__main__":
    print("开始运行API测试...\n")

    test_health_check()
    test_schema_info()
    test_generate_user_data()
    test_generate_with_faker_config()
    test_generate_100_records()
    test_invalid_count()
    test_missing_schema()
    test_chinese_property_names()
    test_id_auto_uuid()
    test_unique_constraint()
    test_100_unique_ids()
    test_chinese_id_field_names()
    test_eval_expression()
    test_list_templates()
    test_get_template_detail()
    test_generate_by_template()
    test_ref_data_with_refs()
    test_order_template_with_refs()
    test_generate_related_data()

    print("=== 所有测试完成 ===")
