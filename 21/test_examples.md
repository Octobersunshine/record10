# 测试示例

## 启动服务
```bash
go run main.go
```

## 积分方法选择

支持两种积分方法：
- `romberg`: 龙贝格积分（默认，理查德森外推加速）
- `newton-cotes` 或 `nc`: 牛顿-柯特斯积分

---

## 一、龙贝格积分示例 (推荐)

### 1. 基本用法（默认参数）
积分 f(x) = x² 从 0 到 1，精确值 = 1/3 ≈ 0.333333333333

```bash
curl -X POST http://localhost:8080/integrate ^
  -H "Content-Type: application/json" ^
  -d "{\"function\":\"x^2\",\"a\":0,\"b\":1,\"method\":\"romberg\"}"
```

预期结果: ≈ 0.333333333333 (精度极高)

### 2. 自定义精度和迭代层数
```bash
curl -X POST http://localhost:8080/integrate ^
  -H "Content-Type: application/json" ^
  -d "{\"function\":\"sin(x)\",\"a\":0,\"b\":3.141592653589793,\"method\":\"romberg\",\"tolerance\":1e-15,\"max_levels\":15}"
```

返回包含龙贝格积分表，可查看收敛过程

### 3. 指数函数积分
```bash
curl -X POST http://localhost:8080/integrate ^
  -H "Content-Type: application/json" ^
  -d "{\"function\":\"exp(x)\",\"a\":0,\"b\":1,\"method\":\"romberg\"}"
```

---

## 二、牛顿-柯特斯积分示例

### 1. 梯形法则 (n=1)
```bash
curl -X POST http://localhost:8080/integrate ^
  -H "Content-Type: application/json" ^
  -d "{\"function\":\"x^2\",\"a\":0,\"b\":1,\"method\":\"nc\",\"n\":1,\"N\":10}"
```

### 2. 辛普森法则 (n=2) - 最常用
```bash
curl -X POST http://localhost:8080/integrate ^
  -H "Content-Type: application/json" ^
  -d "{\"function\":\"x^2\",\"a\":0,\"b\":1,\"method\":\"newton-cotes\",\"n\":2,\"N\":10}"
```

### 3. 辛普森3/8法则 (n=3)
```bash
curl -X POST http://localhost:8080/integrate ^
  -H "Content-Type: application/json" ^
  -d "{\"function\":\"x^2\",\"a\":0,\"b\":1,\"method\":\"nc\",\"n\":3,\"N\":10}"
```

### 4. 布尔法则 (n=4)
```bash
curl -X POST http://localhost:8080/integrate ^
  -H "Content-Type: application/json" ^
  -d "{\"function\":\"x^2\",\"a\":0,\"b\":1,\"method\":\"nc\",\"n\":4,\"N\":10}"
```

### 5. Milne法则 (n=5)
```bash
curl -X POST http://localhost:8080/integrate ^
  -H "Content-Type: application/json" ^
  -d "{\"function\":\"x^2\",\"a\":0,\"b\":1,\"method\":\"nc\",\"n\":5,\"N\":10}"
```

### 6. Weddle法则 (n=6)
```bash
curl -X POST http://localhost:8080/integrate ^
  -H "Content-Type: application/json" ^
  -d "{\"function\":\"x^2\",\"a\":0,\"b\":1,\"method\":\"nc\",\"n\":6,\"N\":10}"
```

---

## 支持的函数和运算符

- 四则运算: `+`, `-`, `*`, `/`, `^` (幂)
- 数学函数: `sin`, `cos`, `tan`, `sqrt`, `log`, `exp`, `asin`, `acos`, `atan`
- 常数: `pi`, `e`
- 变量: `x`

---

## 方法对比

| 方法 | 特点 | 适用场景 |
|------|------|---------|
| 龙贝格 | 高精度，自适应，自动收敛 | 高精度要求，函数光滑 |
| 牛顿-柯特斯(n=2) | 简单，辛普森法则 | 一般精度要求 |
| 牛顿-柯特斯(n=4) | 高阶精度 | 平滑函数高精度 |

龙贝格积分通过理查德森外推技术，通常只需较少的函数求值就能达到非常高的精度。

## 运行验证程序

```bash
go run verify_coefficients.go
```

将显示：伯努利数、牛顿-柯特斯系数验证、龙贝格积分收敛过程和精度对比
