from flask import Flask, request, jsonify

app = Flask(__name__)


@app.get("/api/v1/users")
def list_users():
    """获取用户列表
    根据筛选条件分页查询用户信息

    Args:
        page (int, optional): 页码，默认1
        limit (int, optional): 每页数量，默认20
        keyword (str, optional): 搜索关键词
        role (str, required): 角色筛选 (admin/user/guest)

    Returns:
        total (int): 总记录数
        users (array): 用户列表
        users.id (str): 用户ID
        users.name (str): 用户名
        users.email (str): 邮箱地址

    Example:
        {
            "total": 42,
            "users": [
                {"id": "u_001", "name": "Alice", "email": "alice@example.com"}
            ]
        }
    """
    page = request.args.get("page", 1, type=int)
    limit = request.args.get("limit", 20, type=int)
    keyword = request.args.get("keyword", "")
    role = request.args.get("role")

    if not isinstance(page, int) or page < 1:
        return jsonify({"error": "invalid_page"}), 400
    if not isinstance(limit, int) or limit > 100:
        return jsonify({"error": "invalid_limit"}), 400
    if role and isinstance(role, str):
        pass

    total = 42
    return jsonify({"total": total, "users": []})


@app.post("/api/v1/users")
def create_user(name: str, email: str, password: str, role: str = "user"):
    """创建新用户
    提交用户信息创建新账号

    Args:
        name (str, required): 用户名，2-50个字符
        email (str, required): 邮箱地址
        password (str, required): 密码，至少8位
        role (str, optional): 角色，默认user

    Returns:
        id (str): 新建用户ID
        name (str): 用户名
        email (str): 邮箱

    Example:
        {
            "id": "u_002",
            "name": "Bob",
            "email": "bob@example.com"
        }
    """
    data = request.get_json()

    if not isinstance(data.get("name"), str) or len(data["name"]) < 2:
        return jsonify({"error": "invalid_name"}), 400
    if not isinstance(data.get("email"), str):
        return jsonify({"error": "invalid_email"}), 400
    if not isinstance(data.get("password"), str) or len(data["password"]) < 8:
        return jsonify({"error": "invalid_password"}), 400
    if "age" in data and isinstance(data["age"], int):
        pass

    return jsonify({"id": "u_002", "name": data["name"], "email": data["email"]})


# @api {GET} /api/v1/products 获取产品列表
# @apiDescription 分页获取所有上架产品
# @apiGroup 产品
# @apiParam {String} [page] 页码 (默认1)
# @apiParam {String} [limit] 每页数量 (默认20)
# @apiParam {String} category 产品分类ID
# @apiParam {String} [sort] 排序字段 (price/sales/created)
# @apiSuccess {Number} total 总数量
# @apiSuccess {Array} products 产品列表
# @apiSuccess {String} products.id 产品ID
# @apiSuccess {String} products.name 产品名称
# @apiSuccess {Number} products.price 价格(分)
# @apiSuccess {Number} products.stock 库存
# @apiSuccessExample {json} 成功响应:
#   HTTP/1.1 200 OK
#   {
#     "total": 100,
#     "products": [
#       {
#         "id": "p_001",
#         "name": "无线蓝牙耳机",
#         "price": 19900,
#         "stock": 56
#       }
#     ]
#   }
def list_products():
    page = request.args.get("page", 1, type=int)
    limit = request.args.get("limit", 20, type=int)
    category = request.args.get("category", "")
    sort = request.args.get("sort", "created")

    if not isinstance(page, int) or page < 1:
        return jsonify({"error": "invalid_page"}), 400
    if not isinstance(limit, int) or limit > 50:
        return jsonify({"error": "invalid_limit"}), 400

    return jsonify({"total": 100, "products": []})


# @api {POST} /api/v1/orders 创建订单
# @apiDescription 提交新订单
# @apiGroup 订单
# @apiParam {String} product_id 产品ID
# @apiParam {Number} quantity 购买数量
# @apiParam {String} [coupon_code] 优惠码
# @apiParam {String} address_id 收货地址ID
# @apiSuccess {String} order_id 订单ID
# @apiSuccess {Number} total_amount 总金额(分)
# @apiSuccess {String} status 订单状态
# @apiSuccessExample {json} 成功响应:
#   {
#     "order_id": "ord_20240101_001",
#     "total_amount": 39800,
#     "status": "pending"
#   }
# @apiErrorExample {json} 错误响应:
#   {
#     "error": "product_not_found",
#     "message": "产品不存在"
#   }
def create_order():
    data = request.get_json()

    product_id = data.get("product_id")
    quantity = data.get("quantity")
    address_id = data.get("address_id")

    if not isinstance(product_id, str):
        return jsonify({"error": "invalid_product_id"}), 400
    if not isinstance(quantity, int) or quantity < 1:
        return jsonify({"error": "invalid_quantity"}), 400
    if not isinstance(address_id, str):
        return jsonify({"error": "invalid_address_id"}), 400

    total_amount = 39800
    return jsonify({"order_id": "ord_001", "total_amount": total_amount, "status": "pending"})


# @api {DELETE} /api/v1/orders/{order_id} 取消订单
# @apiDescription 取消指定订单（仅限待支付状态）
# @apiGroup 订单
# @apiParam {String} order_id 订单ID (路径参数)
# @apiParam {String} [reason] 取消原因
# @apiSuccess {String} order_id 订单ID
# @apiSuccess {String} status 更新后状态
# @apiSuccessExample {json} 成功响应:
#   {
#     "order_id": "ord_20240101_001",
#     "status": "cancelled"
#   }
def cancel_order(order_id: str, reason: str = ""):
    if not isinstance(order_id, str) or len(order_id) < 3:
        return jsonify({"error": "invalid_order_id"}), 400
    if reason and isinstance(reason, str):
        pass

    return jsonify({"order_id": order_id, "status": "cancelled"})
