import requests
import json

BASE_URL = "http://localhost:8000"
RENDER_URL = f"{BASE_URL}/api/render/email"
NAMED_RENDER_URL = f"{BASE_URL}/api/render/named"
TEMPLATES_URL = f"{BASE_URL}/api/templates"

def test_basic_render():
    print("=== 测试1: 基础HTML邮件渲染 ===")
    template = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; }
        .header { background: #f0f0f0; padding: 20px; }
        .content { padding: 20px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>您好, {{ username }}!</h1>
    </div>
    <div class="content">
        <p>感谢您注册我们的服务。</p>
        {% if is_vip %}
            <p><strong>尊贵的VIP会员</strong>，专享9折优惠！</p>
        {% endif %}
        <p>请点击以下链接激活您的账户：</p>
        <a href="{{ activate_url }}">{{ activate_url }}</a>
    </div>
</body>
</html>
    """
    
    variables = {
        "username": "李明",
        "is_vip": True,
        "activate_url": "https://example.com/activate?token=abc123"
    }
    
    response = requests.post(RENDER_URL, json={
        "template": template,
        "variables": variables
    })
    
    result = response.json()
    print(f"状态码: {response.status_code}")
    print(f"成功: {result['success']}")
    print(f"预览: {result['template_preview']}")
    print()


def test_list_render():
    print("=== 测试2: 列表循环渲染 ===")
    template = """
<html>
<body>
    <h2>您的订单清单</h2>
    <table border="1" cellpadding="8">
        <tr><th>商品名称</th><th>单价</th><th>数量</th><th>小计</th></tr>
        {% for item in order_items %}
        <tr>
            <td>{{ item.name }}</td>
            <td>¥{{ item.price }}</td>
            <td>{{ item.quantity }}</td>
            <td>¥{{ item.price * item.quantity }}</td>
        </tr>
        {% endfor %}
    </table>
    <p><strong>总计: ¥{{ total }}</strong></p>
</body>
</html>
    """
    
    variables = {
        "order_items": [
            {"name": "无线耳机", "price": 299, "quantity": 1},
            {"name": "手机壳", "price": 49, "quantity": 2},
            {"name": "充电器", "price": 89, "quantity": 1}
        ],
        "total": 486
    }
    
    response = requests.post(RENDER_URL, json={
        "template": template,
        "variables": variables
    })
    
    result = response.json()
    print(f"状态码: {response.status_code}")
    print(f"成功: {result['success']}")
    print(f"预览: {result['template_preview']}")
    print()


def test_template_error():
    print("=== 测试3: 模板语法错误处理 ===")
    template = """
<p>您好, {{ name }</p>
    """
    
    variables = {"name": "测试"}
    
    response = requests.post(RENDER_URL, json={
        "template": template,
        "variables": variables
    })
    
    print(f"状态码: {response.status_code}")
    print(f"错误信息: {response.json().get('detail')}")
    print()


def test_missing_variable_default_empty():
    print("=== 测试4: 缺失变量 - 默认替换为空字符串 ===")
    template = """
<p>您好, {{ name }}!</p>
<p>订单号: {{ order_id }}</p>
<p>备注: {{ remark }}</p>
    """

    variables = {"name": "王五"}

    response = requests.post(RENDER_URL, json={
        "template": template,
        "variables": variables
    })

    result = response.json()
    print(f"状态码: {response.status_code}")
    print(f"成功: {result['success']}")
    print(f"渲染结果:\n{result['rendered_content']}")
    assert "{{ " not in result['rendered_content'], "缺失变量应被替换为空字符串，不应保留占位符"
    print("验证通过: 缺失变量已替换为空字符串\n")


def test_missing_variable_keep_placeholder():
    print("=== 测试5: 缺失变量 - 保留原始占位符 ===")
    template = """
<p>您好, {{ name }}!</p>
<p>订单号: {{ order_id }}</p>
<p>备注: {{ remark }}</p>
    """

    variables = {"name": "赵六"}

    response = requests.post(RENDER_URL, json={
        "template": template,
        "variables": variables,
        "keep_placeholder": True
    })

    result = response.json()
    print(f"状态码: {response.status_code}")
    print(f"成功: {result['success']}")
    print(f"渲染结果:\n{result['rendered_content']}")
    assert "{{ order_id }}" in result['rendered_content'], "缺失变量应保留原始占位符"
    assert "{{ remark }}" in result['rendered_content'], "缺失变量应保留原始占位符"
    assert "赵六" in result['rendered_content'], "已提供的变量应正常替换"
    print("验证通过: 缺失变量已保留原始占位符\n")


def test_missing_variable_custom_default():
    print("=== 测试6: 缺失变量 - 自定义默认值 ===")
    template = """
<p>您好, {{ name }}!</p>
<p>订单号: {{ order_id }}</p>
<p>备注: {{ remark }}</p>
    """

    variables = {"name": "孙七"}

    response = requests.post(RENDER_URL, json={
        "template": template,
        "variables": variables,
        "keep_placeholder": False,
        "default_for_missing": "[未填写]"
    })

    result = response.json()
    print(f"状态码: {response.status_code}")
    print(f"成功: {result['success']}")
    print(f"渲染结果:\n{result['rendered_content']}")
    assert "[未填写]" in result['rendered_content'], "缺失变量应替换为自定义默认值"
    assert "{{ " not in result['rendered_content'], "不应保留占位符"
    assert "孙七" in result['rendered_content'], "已提供的变量应正常替换"
    print("验证通过: 缺失变量已替换为自定义默认值 '[未填写]'\n")


def test_missing_variable_if_condition():
    print("=== 测试7: 缺失变量 - 条件判断中的缺失变量 ===")
    template = """
<p>您好, {{ name }}!</p>
{% if discount %}
<p>优惠金额: ¥{{ discount }}</p>
{% endif %}
<p>感谢您的购买。</p>
    """

    variables = {"name": "周八"}

    response = requests.post(RENDER_URL, json={
        "template": template,
        "variables": variables
    })

    result = response.json()
    print(f"状态码: {response.status_code}")
    print(f"成功: {result['success']}")
    print(f"渲染结果:\n{result['rendered_content']}")
    assert "优惠金额" not in result['rendered_content'], "缺失变量在if中应为False，条件块不应渲染"
    print("验证通过: 缺失变量在条件判断中正确处理为False\n")


def test_template_list():
    print("=== 测试8: 获取模板列表 ===")
    response = requests.get(TEMPLATES_URL)
    result = response.json()
    print(f"状态码: {response.status_code}")
    print(f"模板总数: {result['total']}")
    for t in result['templates']:
        extends_info = f" (继承: {t['extends']})" if t['extends'] else ""
        print(f"  - {t['name']}: {t['display_name']}{extends_info}")
    assert result['total'] >= 4, "至少应有4个内置模板"
    template_names = [t['name'] for t in result['templates']]
    assert 'welcome.html' in template_names
    assert 'reset_password.html' in template_names
    assert 'order_notification.html' in template_names
    assert 'base.html' in template_names
    print("验证通过: 模板列表正确返回\n")


def test_template_preview():
    print("=== 测试9: 模板预览 ===")
    response = requests.get(f"{BASE_URL}/api/templates/welcome.html/preview")
    result = response.json()
    print(f"状态码: {response.status_code}")
    print(f"模板名: {result['display_name']}")
    print(f"描述: {result['description']}")
    print(f"所需变量: {result['variables']}")
    print(f"继承: {result['extends']}")
    print(f"预览: {result['sample_preview']}")
    assert result['sample_rendered'] is not None, "应有示例渲染结果"
    assert "示例用户" in result['sample_rendered'], "示例渲染应包含用户名"
    print("验证通过: 模板预览功能正常\n")


def test_named_render_welcome():
    print("=== 测试10: 渲染欢迎邮件模板 ===")
    response = requests.post(NAMED_RENDER_URL, json={
        "template_name": "welcome.html",
        "variables": {
            "site_name": "MyApp",
            "user_name": "张三",
            "activate_url": "https://myapp.com/activate?token=xyz789",
            "year": "2026",
            "expire_hours": 48,
            "invite_code": "VIP2026"
        }
    })
    result = response.json()
    print(f"状态码: {response.status_code}")
    print(f"成功: {result['success']}")
    print(f"预览: {result['template_preview']}")
    assert result['success']
    assert "张三" in result['rendered_content'], "应包含用户名"
    assert "MyApp" in result['rendered_content'], "应包含站点名"
    assert "xyz789" in result['rendered_content'], "应包含激活链接"
    assert "VIP2026" in result['rendered_content'], "应包含邀请码"
    assert "48" in result['rendered_content'], "应包含过期时间"
    print("验证通过: 欢迎邮件渲染成功，模板继承生效\n")


def test_named_render_reset_password():
    print("=== 测试11: 渲染密码重置模板 ===")
    response = requests.post(NAMED_RENDER_URL, json={
        "template_name": "reset_password.html",
        "variables": {
            "site_name": "MyApp",
            "user_name": "李四",
            "reset_url": "https://myapp.com/reset?token=reset123",
            "year": "2026"
        }
    })
    result = response.json()
    print(f"状态码: {response.status_code}")
    print(f"成功: {result['success']}")
    print(f"预览: {result['template_preview']}")
    assert result['success']
    assert "李四" in result['rendered_content']
    assert "重置密码" in result['rendered_content']
    assert "reset123" in result['rendered_content']
    print("验证通过: 密码重置邮件渲染成功\n")


def test_named_render_order_notification():
    print("=== 测试12: 渲染订单通知模板 ===")
    response = requests.post(NAMED_RENDER_URL, json={
        "template_name": "order_notification.html",
        "variables": {
            "site_name": "MyApp",
            "user_name": "王五",
            "order_id": "ORD-2026-999",
            "order_items": [
                {"name": "机械键盘", "price": 599, "quantity": 1},
                {"name": "鼠标垫", "price": 39, "quantity": 2}
            ],
            "total_amount": 677,
            "year": "2026",
            "shipping_address": "上海市浦东新区xxx路",
            "tracking_url": "https://myapp.com/track/ORD-2026-999"
        }
    })
    result = response.json()
    print(f"状态码: {response.status_code}")
    print(f"成功: {result['success']}")
    print(f"预览: {result['template_preview']}")
    assert result['success']
    assert "王五" in result['rendered_content']
    assert "ORD-2026-999" in result['rendered_content']
    assert "机械键盘" in result['rendered_content']
    assert "677" in result['rendered_content']
    print("验证通过: 订单通知邮件渲染成功\n")


def test_named_render_not_found():
    print("=== 测试13: 渲染不存在的模板 ===")
    response = requests.post(NAMED_RENDER_URL, json={
        "template_name": "nonexistent.html",
        "variables": {}
    })
    print(f"状态码: {response.status_code}")
    assert response.status_code == 404
    print("验证通过: 不存在的模板返回404\n")


def test_custom_base_template_inheritance():
    print("=== 测试14: 自定义base模板继承 ===")
    custom_base = """<html><head><title>{{ site_name }}</title></head><body><header>Custom Header</header>{% block content %}{% endblock %}<footer>Custom Footer</footer></body></html>"""
    custom_child = """{% extends "custom_base.html" %}{% block content %}<p>Hello {{ user_name }}!</p>{% endblock %}"""
    response = requests.post(NAMED_RENDER_URL, json={
        "template_name": "welcome.html",
        "variables": {
            "site_name": "CustomApp",
            "user_name": "测试用户",
            "activate_url": "https://custom.app/activate",
            "year": "2026"
        },
        "extra_templates": {
            "base.html": custom_base
        }
    })
    result = response.json()
    print(f"状态码: {response.status_code}")
    print(f"成功: {result['success']}")
    assert result['success']
    assert "Custom Header" in result['rendered_content'], "应使用自定义base模板的头部"
    assert "Custom Footer" in result['rendered_content'], "应使用自定义base模板的尾部"
    print("验证通过: 自定义base模板继承成功\n")


if __name__ == "__main__":
    try:
        test_basic_render()
        test_list_render()
        test_template_error()
        test_missing_variable_default_empty()
        test_missing_variable_keep_placeholder()
        test_missing_variable_custom_default()
        test_missing_variable_if_condition()
        test_template_list()
        test_template_preview()
        test_named_render_welcome()
        test_named_render_reset_password()
        test_named_render_order_notification()
        test_named_render_not_found()
        test_custom_base_template_inheritance()
        print("=" * 50)
        print("所有测试通过！")
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到API服务，请先运行 start.bat 启动服务。")
