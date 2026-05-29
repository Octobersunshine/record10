# IP地址信息查询API

基于本地GeoLite2数据库的IPv4地址信息查询API，支持单IP和批量查询。

## 功能特性

- IPv4地址验证
- 地理信息查询（国家、省份、城市、经纬度、时区）
- ISP信息查询（ASN、运营商）
- RESTful API接口
- 批量查询支持（最多1000个IP/次）
- 支持中文地名

## 环境要求

- Python 3.8+
- GeoLite2 City 数据库
- GeoLite2 ASN 数据库

## 安装依赖

```bash
pip install -r requirements.txt
```

## 准备IP数据库

1. 从 MaxMind 下载 GeoLite2 数据库：
   - GeoLite2-City.mmdb
   - GeoLite2-ASN.mmdb

2. 将数据库文件放在项目根目录，或通过环境变量指定路径：

```bash
# Windows
set GEOLITE2_CITY_DB=path\to\GeoLite2-City.mmdb
set GEOLITE2_ASN_DB=path\to\GeoLite2-ASN.mmdb

# Linux/Mac
export GEOLITE2_CITY_DB=/path/to/GeoLite2-City.mmdb
export GEOLITE2_ASN_DB=/path/to/GeoLite2-ASN.mmdb
```

## 启动API服务

```bash
# 默认启动 (0.0.0.0:5000)
python app.py

# 自定义端口
set PORT=8080
python app.py

# 开发模式
set DEBUG=True
python app.py
```

## API接口文档

### 1. 健康检查

```
GET /health
```

**响应示例：**
```json
{
  "status": "ok",
  "city_db_loaded": true,
  "asn_db_loaded": true
}
```

### 2. 单IP查询 (GET方式)

```
GET /api/ip/{ip_address}
```

**示例：**
```
GET /api/ip/8.8.8.8
```

**响应示例：**
```json
{
  "ip": "8.8.8.8",
  "success": true,
  "country": {
    "code": "US",
    "name": "美国",
    "name_en": "United States"
  },
  "city": {
    "name": "山景城",
    "name_en": "Mountain View"
  },
  "subdivision": {
    "code": "CA",
    "name": "加利福尼亚州",
    "name_en": "California"
  },
  "location": {
    "latitude": 37.751,
    "longitude": -97.822,
    "time_zone": "America/Chicago",
    "accuracy_radius": 1000
  },
  "isp": {
    "asn": 15169,
    "isp": "Google LLC",
    "organization": "Google LLC"
  }
}
```

### 3. 单IP查询 (POST方式)

```
POST /api/ip
Content-Type: application/json

{
  "ip": "8.8.8.8"
}
```

### 4. 批量IP查询

```
POST /api/ip/batch
Content-Type: application/json

{
  "ips": ["8.8.8.8", "1.1.1.1", "114.114.114.114"]
}
```

**限制：** 单次最多查询1000个IP

**响应示例：**
```json
{
  "success": true,
  "total": 3,
  "results": [
    { ... },
    { ... },
    { ... }
  ]
}
```

## 作为Python模块使用

```python
from ip_lookup import IPLookup

# 方式1: 使用上下文管理器
with IPLookup() as lookup:
    # 单IP查询
    result = lookup.lookup('8.8.8.8')
    print(result['country']['name'])
    
    # 批量查询
    results = lookup.batch_lookup(['8.8.8.8', '1.1.1.1'])
    for r in results:
        print(f"{r['ip']}: {r['country']['name']}")

# 方式2: 手动管理
lookup = IPLookup()
result = lookup.lookup('8.8.8.8')
lookup.close()
```

## 项目结构

```
.
├── app.py              # Flask API服务
├── ip_lookup.py        # IP查询核心模块
├── example.py          # Python模块使用示例
├── test_api.py         # API测试脚本
├── requirements.txt    # 依赖包列表
├── README.md           # 说明文档
├── GeoLite2-City.mmdb  # GeoLite2城市数据库(需自行下载)
└── GeoLite2-ASN.mmdb   # GeoLite2 ASN数据库(需自行下载)
```

## 运行示例

```bash
# 运行Python模块示例
python example.py

# 运行API测试(需先启动服务)
python test_api.py
```

## 错误响应

查询失败时返回：
```json
{
  "ip": "999.999.999.999",
  "success": false,
  "error": "Invalid IPv4 address"
}
```

## 注意事项

1. 请遵守 MaxMind 的使用条款和许可协议
2. 建议定期更新IP数据库以获取最新数据
3. 批量查询建议控制在合理范围内，避免内存占用过高
