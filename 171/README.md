# 高精度轨道预报系统

基于Python实现的完整轨道预报系统，包含：
- ✅ **SGP4/SDP4解析模型** - 标准轨道传播
- ✅ **近地点奇异性修复** - 圆轨道自动切换数值积分
- ✅ **高精度数值积分** - J2-J4摄动 + 大气阻力 + 太阳光压
- ✅ **扩展卡尔曼滤波(EKF)** - GPS观测数据同化
- ✅ **精度对比分析** - 不同模型精度验证

## 🔧 核心改进

### ✅ 近地点奇异性修复
- **问题**: 标准SGP4模型在偏心率接近0（圆轨道）时，近地点计算会出现数值不稳定
- **解决方案**: 自动检测近圆轨道（e < 1e-4），切换到数值积分模式

### ✅ 多传播模式支持
| 模式 | 适用场景 | 说明 |
|------|----------|------|
| **AUTO** | 自动选择 | 根据轨道特征自动选择最优模式 |
| **SGP4** | 近地椭圆轨道 | 标准SGP4模型，适合LEO卫星 |
| **SDP4** | 深空卫星 | 深空扩展模型，适合周期≥225分钟卫星 |
| **NUMERICAL** | 近圆轨道 | Runge-Kutta数值积分，避免奇异性 |

### ✅ 高精度数值积分器
- **RK4** (4阶Runge-Kutta): 平衡精度和速度
- **RK8** (8阶Dormand-Prince): 高精度积分，适合长期预报

## 📦 安装依赖

```bash
pip install sgp4 numpy
```

或使用 requirements.txt:
```bash
pip install -r requirements.txt
```

## 🚀 快速开始

### 基本使用

```python
from datetime import datetime
from sgp4_orbit_predictor import EnhancedOrbitPredictor

# TLE两行轨道根数
tle1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9993"
tle2 = "2 25544  51.6400 208.9163 0006703  35.7657  69.0011 15.49923619  1234"

# 创建预报器（自动模式）
predictor = EnhancedOrbitPredictor(
    tle1, tle2, 
    satellite_name="ISS (ZARYA)",
    propagation_mode='auto'  # 自动选择最优传播模式
)

# 获取ECI位置
current_time = datetime.utcnow()
pos, vel = predictor.get_position_eci(current_time)

print(f"位置: X={pos[0]:.2f} km, Y={pos[1]:.2f} km, Z={pos[2]:.2f} km")
print(f"速度: {np.linalg.norm(vel):.4f} km/s")
print(f"传播模式: {predictor.current_mode}")
```

### 强制使用数值积分（避免奇异性）

```python
# 对于近圆轨道，强制使用数值积分
predictor = EnhancedOrbitPredictor(
    tle1, tle2,
    satellite_name="Circular Sat",
    propagation_mode='numerical',  # 强制数值积分
    integrator='rk8'  # 选择RK8高精度积分器
)
```

### 过顶预报

```python
# 预报北京地区未来24小时的过顶事件
passes = predictor.predict_passes(
    observer_lat=39.9,      # 纬度
    observer_lon=116.4,     # 经度
    observer_alt=50,        # 海拔(米)
    duration_hours=24,      # 预报时长
    min_elevation=10.0      # 最小仰角
)

for p in passes:
    print(f"开始: {p['start_time'].strftime('%H:%M')}")
    print(f"结束: {p['end_time'].strftime('%H:%M')}")
    print(f"时长: {p['duration']:.1f}分钟")
    print(f"最大仰角: {p['max_elevation']:.1f}°")
```

### 传播方法对比

```python
# 对比SGP4和数值积分的差异
comparison = predictor.compare_propagation_methods(datetime.utcnow())

if 'difference' in comparison:
    print(f"位置差: {comparison['difference']['position_km']:.4f} km")
    print(f"速度差: {comparison['difference']['velocity_km_s']:.6f} km/s")
```

## 📚 API参考

### EnhancedOrbitPredictor 类

#### 构造函数

```python
EnhancedOrbitPredictor(
    tle_line1, tle_line2,
    satellite_name="Satellite",
    propagation_mode='auto',  # 'auto', 'sgp4', 'sdp4', 'numerical'
    integrator='rk4'          # 'rk4', 'rk8'
)
```

#### 主要方法

| 方法 | 功能 |
|------|------|
| `get_position_eci(datetime)` | 获取指定时间的ECI坐标 |
| `get_position_eci_with_mode(dt, mode)` | 使用指定模式计算 |
| `predict_orbit(...)` | 预报一段时间内的轨道 |
| `predict_passes(...)` | 预报过顶事件 |
| `get_communication_windows(...)` | 获取通信窗口详情 |
| `get_ground_track(orbit_points)` | 计算星下点轨迹 |
| `compare_propagation_methods(dt)` | 对比不同传播方法 |

#### 属性

| 属性 | 说明 |
|------|------|
| `eccentricity` | 轨道偏心率 |
| `period` | 轨道周期（分钟） |
| `is_near_circular` | 是否近圆轨道（e < 1e-4） |
| `is_deep_space` | 是否深空卫星（周期≥225分钟） |
| `current_mode` | 当前传播模式 |

## 🔬 技术细节

### 奇异性检测算法

```python
ECCENTRICITY_THRESHOLD = 1e-4  # 近圆轨道阈值

@property
def is_near_circular(self):
    return self.eccentricity < ECCENTRICITY_THRESHOLD
```

### 自动模式选择逻辑

```
if 周期 >= 225分钟:
    使用 SDP4 (深空)
elif 偏心率 < 1e-4:
    使用 数值积分 (避免奇异性)
else:
    使用 SGP4 (标准模型)
```

### 数值积分实现

- **动力学模型**: 二体引力 + J2地球扁率摄动
- **积分器**: RK4 / RK8 (Dormand-Prince)
- **步长控制**: 自适应步长，保证精度

### J2摄动模型

```python
acc_J2 = factor_J2 * [
    x * (5z²/r² - 1),
    y * (5z²/r² - 1),
    z * (5z²/r² - 3)
]
```

## 📊 运行示例

运行主程序：
```bash
python sgp4_orbit_predictor.py
```

运行详细示例：
```bash
python example_usage.py
```

## ⚠️ 注意事项

1. **时间基准**: 所有输入时间应为UTC时间
2. **TLE更新**: TLE数据会老化，建议定期更新（通常1-2周）
3. **数值积分精度**: 
   - 短期预报（<1天）: 精度与SGP4相当
   - 长期预报（>3天）: 需考虑更多摄动项
4. **近圆轨道**: 自动切换到数值积分，避免除零错误
5. **EKF初始化**: 建议使用TLE或历史观测数据初始化
6. **大气模型**: Harris-Priester模型需要太阳活动参数

---

## 🔗 参考资源

- [SGP4标准文档](https://celestrak.com/publications/AIAA/2006-6753/)
- [Celestrak TLE数据](https://celestrak.com/)
- [Vallado, D. A. - Fundamentals of Astrodynamics and Applications]
- [Montenbruck, Gill - Satellite Orbits]

---

## 📝 更新日志

### v3.0 (当前版本)
- ✅ J2-J4摄动模型
- ✅ 大气阻力模型（Harris-Priester）
- ✅ 太阳光压模型（含地影）
- ✅ 扩展卡尔曼滤波(EKF)
- ✅ GPS观测数据同化
- ✅ 精度对比分析工具

### v2.0
- ✅ 修复近地点奇异性问题
- ✅ 添加SDP4深空扩展支持
- ✅ 实现RK4/RK8数值积分器
- ✅ 自动传播模式切换

### v1.0
- 基础SGP4轨道预报
- 过顶预报和通信窗口分析
