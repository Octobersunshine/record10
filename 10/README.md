# Flask K-means Clustering API

一个基于 Python + Flask 的 K-means 聚类 API 服务，支持接收多维数据点进行聚类分析。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动服务

```bash
python app.py
```

服务将在 `http://localhost:5000` 启动。

## API 接口

### 1. 健康检查

**GET** `/health`

响应示例：
```json
{
  "status": "ok"
}
```

### 2. K-means 聚类

**POST** `/kmeans`

请求体：
```json
{
  "points": [[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], ...],
  "k": 2
}
```

参数说明：
- `points`: 二维数组，每个子数组代表一个数据点（支持任意维度）
- `k`: 聚类数量，正整数

**空簇处理**：当某个簇没有数据点时，从当前最大簇中随机选择一个点作为新中心，保持K值不变。

响应示例：
```json
{
  "labels": [0, 0, 1, 1, 0, 1],
  "centers": [[1.1667, 1.4667], [7.3333, 9.0]],
  "k": 2,
  "num_points": 6,
  "dimensions": 2
}
```

### 3. 肘部法则推荐K值

**POST** `/kmeans/recommend-k`

请求体：
```json
{
  "points": [[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], ...],
  "max_k": 10
}
```

参数说明：
- `points`: 二维数组，每个子数组代表一个数据点（支持任意维度）
- `max_k` (可选): 最大尝试的K值，默认为min(10, 点数-1)

**算法原理**：
1. 计算K从1到max_k每个值对应的SSE（误差平方和）
2. 通过二阶导数找到SSE下降曲线的拐点（肘部）
3. 拐点对应的K值即为推荐值

响应示例：
```json
{
  "recommended_k": 3,
  "k_values": [1, 2, 3, 4, 5],
  "sse_values": [500.5, 200.3, 50.2, 40.1, 35.0],
  "num_points": 9,
  "dimensions": 2
}
```

## 测试

```bash
pip install requests
python test_api.py
```

## 使用示例 (curl)

### K-means 聚类
```bash
curl -X POST http://localhost:5000/kmeans \
  -H "Content-Type: application/json" \
  -d '{
    "points": [[1, 2], [2, 1], [10, 10], [11, 11]],
    "k": 2
  }'
```

### 肘部法则推荐K值
```bash
curl -X POST http://localhost:5000/kmeans/recommend-k \
  -H "Content-Type: application/json" \
  -d '{
    "points": [[1, 2], [2, 1], [10, 10], [11, 11], [20, 20], [21, 21]],
    "max_k": 5
  }'
```
