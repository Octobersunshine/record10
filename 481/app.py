from flask import Flask, request, jsonify
from mock_generator import generate_mock_data, generate_related_data, get_template, get_all_templates
from jsonschema import validate, ValidationError
import json

app = Flask(__name__)

REQUEST_SCHEMA = {
    "type": "object",
    "required": ["schema"],
    "properties": {
        "schema": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "properties": {"type": "object"}
            },
            "required": ["properties"]
        },
        "count": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100
        },
        "refs": {
            "type": "object"
        }
    }
}

TEMPLATE_GENERATE_SCHEMA = {
    "type": "object",
    "required": ["template"],
    "properties": {
        "template": {
            "type": "string"
        },
        "count": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100
        },
        "refs": {
            "type": "object"
        }
    }
}

RELATED_DATA_SCHEMA = {
    "type": "object",
    "required": ["relations"],
    "properties": {
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "schema"],
                "properties": {
                    "name": {"type": "string"},
                    "schema": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "properties": {"type": "object"}
                        }
                    },
                    "count": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100
                    },
                    "refs": {"type": "object"}
                }
            }
        }
    }
}


@app.route('/api/mock/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "请求体不能为空",
                "message": "请提供JSON格式的请求体"
            }), 400

        validate(instance=data, schema=REQUEST_SCHEMA)

        schema = data['schema']
        count = data.get('count', 1)
        refs = data.get('refs', {})

        mock_data = generate_mock_data(schema, count, refs)

        return jsonify({
            "success": True,
            "data": mock_data,
            "total": len(mock_data),
            "message": f"成功生成 {len(mock_data)} 条模拟数据"
        }), 200

    except ValidationError as e:
        return jsonify({
            "success": False,
            "error": "参数验证失败",
            "message": e.message,
            "path": list(e.absolute_path)
        }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": "服务器内部错误",
            "message": str(e)
        }), 500


@app.route('/api/mock/templates', methods=['GET'])
def list_templates():
    try:
        templates = get_all_templates()
        return jsonify({
            "success": True,
            "data": templates,
            "total": len(templates),
            "message": "获取模板列表成功"
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "服务器内部错误",
            "message": str(e)
        }), 500


@app.route('/api/mock/template/<template_name>', methods=['GET'])
def get_template_detail(template_name):
    try:
        template = get_template(template_name)
        if not template:
            return jsonify({
                "success": False,
                "error": "模板不存在",
                "message": f"未找到名为 {template_name} 的模板"
            }), 404

        return jsonify({
            "success": True,
            "data": template,
            "message": "获取模板详情成功"
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "服务器内部错误",
            "message": str(e)
        }), 500


@app.route('/api/mock/generate/template', methods=['POST'])
def generate_by_template():
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "请求体不能为空",
                "message": "请提供JSON格式的请求体"
            }), 400

        validate(instance=data, schema=TEMPLATE_GENERATE_SCHEMA)

        template_name = data['template']
        count = data.get('count', 10)
        refs = data.get('refs', {})

        template = get_template(template_name)
        if not template:
            return jsonify({
                "success": False,
                "error": "模板不存在",
                "message": f"未找到名为 {template_name} 的模板"
            }), 404

        mock_data = generate_mock_data(template['schema'], count, refs)

        return jsonify({
            "success": True,
            "data": mock_data,
            "template": template_name,
            "total": len(mock_data),
            "message": f"使用模板 {template_name} 成功生成 {len(mock_data)} 条模拟数据"
        }), 200

    except ValidationError as e:
        return jsonify({
            "success": False,
            "error": "参数验证失败",
            "message": e.message,
            "path": list(e.absolute_path)
        }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": "服务器内部错误",
            "message": str(e)
        }), 500


@app.route('/api/mock/generate/related', methods=['POST'])
def generate_related():
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "请求体不能为空",
                "message": "请提供JSON格式的请求体"
            }), 400

        validate(instance=data, schema=RELATED_DATA_SCHEMA)

        relations = data['relations']
        result = generate_related_data(relations)

        totals = {k: len(v) for k, v in result.items()}
        return jsonify({
            "success": True,
            "data": result,
            "totals": totals,
            "message": f"成功生成 {len(relations)} 组关联数据"
        }), 200

    except ValidationError as e:
        return jsonify({
            "success": False,
            "error": "参数验证失败",
            "message": e.message,
            "path": list(e.absolute_path)
        }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": "服务器内部错误",
            "message": str(e)
        }), 500


@app.route('/api/mock/schema', methods=['GET'])
def get_schema_info():
    return jsonify({
        "success": True,
        "data": {
            "endpoints": {
                "POST /api/mock/generate": "基于自定义JSON Schema生成数据",
                "POST /api/mock/generate/template": "使用模板生成数据",
                "POST /api/mock/generate/related": "生成关联数据",
                "GET /api/mock/templates": "获取所有模板列表",
                "GET /api/mock/template/:name": "获取指定模板详情",
                "GET /api/mock/health": "健康检查"
            },
            "request_schema": REQUEST_SCHEMA,
            "supported_types": ["name", "address", "email", "phone", "mobile", "date", "datetime", "string", "integer", "number", "boolean", "uuid"],
            "supported_formats": ["email", "date", "date-time", "uri", "url", "hostname", "ipv4", "ipv6", "uuid"],
            "faker_methods": [
                "name", "address", "email", "phone_number", "date", "date_time",
                "word", "text", "sentence", "paragraph", "url", "domain_name",
                "ipv4", "ipv6", "company", "job", "ssn", "user_name", "password",
                "first_name", "last_name", "city", "country", "street_address",
                "postcode", "latitude", "longitude", "uuid4", "uuid"
            ],
            "id_field_names": ["id", "uuid", "guid", "id号", "编号", "标识"],
            "unique_constraint": {
                "description": "支持 unique 属性，设置为 true 时该字段值在所有生成数据中唯一",
                "auto_unique_for_id": "ID字段自动生成UUID并保证唯一性"
            },
            "eval_expression": {
                "description": "支持 eval 属性，使用Python表达式生成数据",
                "available_vars": ["random", "uuid", "datetime", "timedelta", "fake", "index", "同对象其他字段名"]
            },
            "ref_data": {
                "description": "支持 ref 属性，引用外部数据源生成关联数据",
                "usage": "'ref': 'data_source_name' 或 {'ref': {'name': 'source', 'field': 'id'}}"
            },
            "templates": list(get_all_templates().keys()),
            "count_range": {
                "min": 1,
                "max": 100
            }
        },
        "message": "Mock数据生成API信息"
    }), 200


@app.route('/api/mock/health', methods=['GET'])
def health_check():
    return jsonify({
        "success": True,
        "status": "healthy",
        "message": "Mock数据生成API服务运行正常"
    }), 200


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "success": False,
        "error": "接口不存在",
        "message": "请使用 POST /api/mock/generate 接口"
    }), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({
        "success": False,
        "error": "请求方法不允许",
        "message": "请使用 POST 方法请求 /api/mock/generate 接口"
    }), 405


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
