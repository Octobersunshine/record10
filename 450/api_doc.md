# 电商系统API文档

## 目录

### 产品

- [GET /api/v1/products](#get--api-v1-products) — 获取产品列表

### 订单

- [POST /api/v1/orders](#post--api-v1-orders) — 创建订单
- [DELETE /api/v1/orders/{order_id}](#delete--api-v1-orders-order_id) — 取消订单

### 默认

- [GET /api/v1/users](#get--api-v1-users) — 获取用户列表
- [POST /api/v1/users](#post--api-v1-users) — 创建新用户

## 产品

### GET `/api/v1/products`

**获取产品列表**

分页获取所有上架产品

- **方法**: GET
- **路径**: `/api/v1/products`

#### 参数

| 参数名 | 类型 | 必填 | 位置 | 默认值 | 说明 |
|--------|------|------|------|--------|------|
| `page` | integer ⚙ *(原: String)* | 否 | query | - | 页码 (默认1) |
| `limit` | integer ⚙ *(原: String)* | 否 | query | - | 每页数量 (默认20) |
| `category` | String | 是 | query | - | 产品分类ID |
| `sort` | String | 否 | query | - | 排序字段 (price/sales/created) |

> *标记 ⚙ 的类型为从代码/示例自动推断，可能与原始标注不同*

#### 返回字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `total` | integer ⚙ *(原: Number)* | 总数量 |
| `products` | Array | 产品列表 |
| `products.id` | String | 产品ID |
| `products.name` | String | 产品名称 |
| `products.price` | integer ⚙ *(原: Number)* | 价格(分) |
| `products.stock` | integer ⚙ *(原: Number)* | 库存 |

> *标记 ⚙ 的类型为从代码/示例自动推断，可能与原始标注不同*

#### 成功响应示例

```json
{
  "total": 100,
  "products": [
    {
      "id": "p_001",
      "name": "无线蓝牙耳机",
      "price": 19900,
      "stock": 56
    }
  ]
}
```


## 订单

### POST `/api/v1/orders`

**创建订单**

提交新订单

- **方法**: POST
- **路径**: `/api/v1/orders`

#### 参数

| 参数名 | 类型 | 必填 | 位置 | 默认值 | 说明 |
|--------|------|------|------|--------|------|
| `product_id` | String | 是 | query | - | 产品ID |
| `quantity` | integer ⚙ *(原: Number)* | 是 | query | - | 购买数量 |
| `coupon_code` | String | 否 | query | - | 优惠码 |
| `address_id` | String | 是 | query | - | 收货地址ID |

> *标记 ⚙ 的类型为从代码/示例自动推断，可能与原始标注不同*

#### 返回字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `order_id` | String | 订单ID |
| `total_amount` | integer ⚙ *(原: Number)* | 总金额(分) |
| `status` | String | 订单状态 |

> *标记 ⚙ 的类型为从代码/示例自动推断，可能与原始标注不同*

#### 成功响应示例

```json
{
  "order_id": "ord_20240101_001",
  "total_amount": 39800,
  "status": "pending"
}
```

#### 错误响应示例

```json
{
  "error": "product_not_found",
  "message": "产品不存在"
}
```


### DELETE `/api/v1/orders/{order_id}`

**取消订单**

取消指定订单（仅限待支付状态）

- **方法**: DELETE
- **路径**: `/api/v1/orders/{order_id}`

#### 参数

| 参数名 | 类型 | 必填 | 位置 | 默认值 | 说明 |
|--------|------|------|------|--------|------|
| `order_id` | String | 是 | query | - | 订单ID (路径参数) |
| `reason` | String | 否 | query | - | 取消原因 |

#### 返回字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `order_id` | String | 订单ID |
| `status` | String | 更新后状态 |

#### 成功响应示例

```json
{
  "order_id": "ord_20240101_001",
  "status": "cancelled"
}
```


## 默认

### GET `/api/v1/users`

**获取用户列表**

根据筛选条件分页查询用户信息

- **方法**: GET
- **路径**: `/api/v1/users`

#### 参数

| 参数名 | 类型 | 必填 | 位置 | 默认值 | 说明 |
|--------|------|------|------|--------|------|
| `page` | integer | 否 | query | - | 页码，默认1 |
| `limit` | integer | 否 | query | - | 每页数量，默认20 |
| `keyword` | string | 否 | query | - | 搜索关键词 |
| `role` | string | 是 | query | - | 角色筛选 (admin/user/guest) |

#### 返回字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `total` | integer | 总记录数 |
| `users` | array | 用户列表 |
| `users.id` | string | 用户ID |
| `users.name` | string | 用户名 |
| `users.email` | string | 邮箱地址 |

#### 成功响应示例

```json
{
  "total": 42,
  "users": [
    {
      "id": "u_001",
      "name": "Alice",
      "email": "alice@example.com"
    }
  ]
}
```


### POST `/api/v1/users`

**创建新用户**

提交用户信息创建新账号

- **方法**: POST
- **路径**: `/api/v1/users`

#### 参数

| 参数名 | 类型 | 必填 | 位置 | 默认值 | 说明 |
|--------|------|------|------|--------|------|
| `name` | string | 是 | query | - | 用户名，2-50个字符 |
| `email` | string | 是 | query | - | 邮箱地址 |
| `password` | string | 是 | query | - | 密码，至少8位 |
| `role` | string | 否 | query | - | 角色，默认user |

#### 返回字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 新建用户ID |
| `name` | string | 用户名 |
| `email` | string | 邮箱 |

#### 成功响应示例

```json
{
  "id": "u_002",
  "name": "Bob",
  "email": "bob@example.com"
}
```

