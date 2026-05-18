# 拉普拉斯变换数值反演服务

基于FastAPI开发，使用自适应Talbot方法进行拉普拉斯变换数值反演，支持多精度计算。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行服务

```bash
python main.py
```

服务将在 `http://localhost:8000` 启动。

## API文档

启动服务后，访问：
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API使用

### POST /invert

进行拉普拉斯变换数值反演。

**请求体：**
```json
{
  "F_s": "1/(s+1)",
  "t_values": [0.1, 0.5, 1.0, 2.0, 5.0],
  "precision": "double"
}
```

**参数说明：**
- `F_s`: 拉普拉斯变换象函数F(s)的字符串表达式
- `t_values`: 时间点列表，每个值必须大于0
- `N` (可选): 积分点数，不提供时自动选择
- `precision` (可选): 精度级别，可选值:
  - `double`: 64位双精度浮点数（默认），计算最快
  - `high`: 50位多精度计算（mpmath），精度较高
  - `extreme`: 100位多精度 + 三重外推，最高精度

**响应：**
```json
{
  "t_values": [0.1, 0.5, 1.0, 2.0, 5.0],
  "f_t_values": [0.9048, 0.6065, 0.3679, 0.1353, 0.0067],
  "N_used": [32, 32, 32, 32, 32],
  "precision_used": ["double (64-bit)", "double (64-bit)", ...]
}
```

## 精度级别对比

| 级别 | 精度 | 技术 | 适用场景 |
|------|------|------|----------|
| double | 64位 | 双精度浮点数 + Richardson外推 | 常规计算，速度最快 |
| high | 50位 | mpmath任意精度库 | 高精度需求，大t值计算 |
| extreme | 100位 | mpmath + 三重外推 | 极端精度需求，临界情况 |

## 示例

### 示例1: 指数函数
- 象函数: F(s) = 1/(s+1)
- 原函数: f(t) = e^(-t)

### 示例2: 阶跃函数
- 象函数: F(s) = 1/s
- 原函数: f(t) = 1

### 示例3: 正弦函数
- 象函数: F(s) = 1/(s^2 + 1)
- 原函数: f(t) = sin(t)

### 高精度调用示例
```json
{
  "F_s": "1/(s+1)",
  "t_values": [10.0, 20.0, 50.0],
  "precision": "high"
}
```

## 运行精度测试

```bash
python test_accuracy.py
```

## 注意事项

1. 时间值t必须大于0
2. 表达式使用Python语法，支持numpy和scipy函数
3. 支持的数学函数: sin, cos, exp, log, sqrt等
4. extreme精度级别计算较慢，仅推荐用于特殊情况
