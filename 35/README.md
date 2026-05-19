# 时间序列 ACF/PACF 计算 API

基于 Python + FastAPI 构建的时间序列分析API，用于计算ACF（自相关函数）、PACF（偏自相关函数）以及 Ljung-Box 白噪声检验。

## 功能特性

- 计算ACF（自相关函数）及置信区间
- 计算PACF（偏自相关函数）及置信区间
- **Ljung-Box 白噪声检验** - 判断序列是否为白噪声
- 动态延迟阶数选择 - 适配不同长度的序列
- 支持自定义延迟阶数和显著性水平

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动服务

```bash
python main.py
```

服务将在 `http://localhost:8000` 启动。

## API文档

启动服务后，访问以下地址查看交互式API文档：

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API 使用

### POST /api/correlation

计算时间序列的ACF、PACF和Ljung-Box白噪声检验。

**请求参数:**

```json
{
  "data": [10.2, 11.5, 12.1, ...],
  "nlags": 10,
  "alpha": 0.05
}
```

- `data`: 时间序列数据列表（必需）
- `nlags`: 延迟阶数（可选，默认为动态计算值）
- `alpha`: 显著性水平（可选，默认为0.05，即95%置信区间，同时用于白噪声检验）

**默认延迟阶数计算策略:**
- 短序列 (<30): `nobs // 2`
- 中等序列 (30-100): `max(nobs // 3, 20)`
- 长序列 (>100): `max(nobs // 4, 50)`
- 最大不超过 `nobs - 1`（避免过拟合）

此策略确保对于长周期序列（周期>10）不会丢失重要信息。

**Ljung-Box 白噪声检验说明:**
- 原假设 (H0): 序列是白噪声（无自相关）
- 备择假设 (H1): 序列不是白噪声（存在自相关）
- `is_white_noise = True`: 不能拒绝原假设，认为是白噪声（所有 p值 > alpha）
- `is_white_noise = False`: 拒绝原假设，认为不是白噪声（存在 p值 < alpha）

**响应示例:**

```json
{
  "acf": {
    "lags": [0, 1, 2, ...],
    "acf_values": [1.0, 0.95, 0.89, ...],
    "confidence_interval_lower": [...],
    "confidence_interval_upper": [...]
  },
  "pacf": {
    "lags": [0, 1, 2, ...],
    "pacf_values": [1.0, 0.85, -0.12, ...],
    "confidence_interval_lower": [...],
    "confidence_interval_upper": [...]
  },
  "ljung_box": {
    "lags": [1, 2, 3, ...],
    "lb_statistics": [0.5, 1.2, 2.1, ...],
    "p_values": [0.48, 0.55, 0.55, ...],
    "is_white_noise": true,
    "significance_level": 0.05
  }
}
```

## 测试

```bash
pip install requests
python test_api.py
```

## 技术栈

- **FastAPI**: Web框架
- **statsmodels**: 统计分析库，提供ACF和PACF计算
- **NumPy**: 数值计算
- **Pydantic**: 数据验证