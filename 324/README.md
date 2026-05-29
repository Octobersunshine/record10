# 冒泡排序 API

基于 Flask 实现的冒泡排序 API，支持返回每一轮排序后的中间状态，用于前端可视化展示。

## 功能特性

- 接收无序数组，返回完整的排序过程
- 返回每一轮排序后的中间状态数组
- 返回最终排序结果
- 支持跨域请求（CORS）
- 附带前端可视化页面

## 文件说明

- `bubble_sort_api.py` - Flask API 服务端
- `test_api.py` - API 测试脚本
- `index.html` - 前端可视化页面
- `requirements.txt` - Python 依赖包

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动服务

```bash
python bubble_sort_api.py
```

服务将在 `http://localhost:5000` 启动

## API 接口

### 1. 冒泡排序接口

**URL:** `POST /api/bubble-sort`

**请求体:**
```json
{
  "array": [64, 34, 25, 12, 22, 11, 90]
}
```

**响应示例:**
```json
{
  "original_array": [64, 34, 25, 12, 22, 11, 90],
  "sorting_steps": [
    [34, 25, 12, 22, 11, 64, 90],
    [25, 12, 22, 11, 34, 64, 90],
    [12, 22, 11, 25, 34, 64, 90],
    [12, 11, 22, 25, 34, 64, 90],
    [11, 12, 22, 25, 34, 64, 90]
  ],
  "final_sorted_array": [11, 12, 22, 25, 34, 64, 90],
  "total_rounds": 5
}
```

### 2. 健康检查接口

**URL:** `GET /api/health`

**响应:**
```json
{
  "status": "ok",
  "message": "冒泡排序API运行正常"
}
```

## 前端可视化

直接在浏览器中打开 `index.html` 即可使用可视化界面。

功能包括：
- 自定义输入数组
- 随机生成数组
- 动态调整动画速度
- 柱状图实时展示排序过程
- 显示每一轮的排序结果

## 测试 API

运行测试脚本：

```bash
python test_api.py
```

## 使用 curl 测试

```bash
curl -X POST http://localhost:5000/api/bubble-sort \
  -H "Content-Type: application/json" \
  -d '{"array": [64, 34, 25, 12, 22, 11, 90]}'
```
