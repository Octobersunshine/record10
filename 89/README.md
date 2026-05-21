# RRT* 最优路径规划与平滑

使用 Python 实现的 RRT*（最优快速随机扩展树）算法，支持多种障碍物类型和路径平滑。

## 功能特性

### RRT* 算法核心
- 支持圆形和多边形障碍物
- 渐近最优路径规划
- 可配置的扩展距离、目标采样率、最大迭代次数
- 实时动画显示搜索过程

### 路径平滑方法
1. **B样条平滑** - 使用 scipy 的 splprep/splev 实现，平滑效果好
2. **贝塞尔曲线平滑** - 使用 De Casteljau 算法
3. **梯度下降平滑** - 考虑障碍物排斥力的优化平滑

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 基本使用

```python
from rrt_star import RRTStar
from path_smoothing import PathSmoother

# 定义起点和终点
start = [0.0, 0.0]
goal = [6.0, 6.0]

# 定义障碍物列表
# 圆形障碍物: [x, y, radius]
# 多边形障碍物: [[x1,y1], [x2,y2], ..., [xn,yn]]
obstacle_list = [
    [3.0, 3.0, 1.0],  # 圆形
    [[1.0, 1.0], [1.0, 2.0], [2.0, 2.0], [2.0, 1.0]]  # 矩形
]

# 搜索区域 [min, max]
search_area = [0, 7]

# 创建 RRT* 规划器
rrt_star = RRTStar(
    start=start,
    goal=goal,
    obstacle_list=obstacle_list,
    search_area=search_area,
    expand_dis=0.5,
    goal_sample_rate=10,
    max_iter=500,
    connect_circle_dist=1.5
)

# 执行路径规划
path = rrt_star.planning(animation=True)

# 路径平滑
smoother = PathSmoother(obstacle_list=obstacle_list)
smooth_path = smoother.smooth_path(path, method='b_spline', s=0.3)
```

### 运行示例

```bash
python main.py
```

## 参数说明

### RRTStar 类参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| start | list | 必填 | 起点坐标 [x, y] |
| goal | list | 必填 | 终点坐标 [x, y] |
| obstacle_list | list | 必填 | 障碍物列表 |
| search_area | list | 必填 | 搜索区域 [min, max] |
| expand_dis | float | 0.5 | 每次扩展的距离 |
| goal_sample_rate | int | 5 | 采样目标点的概率 (%) |
| max_iter | int | 500 | 最大迭代次数 |
| connect_circle_dist | float | 1.0 | 重布线连接距离系数 |

### 平滑方法参数

#### B样条平滑 (b_spline)
- `s`: 平滑因子 (默认 0.5)，值越大越平滑
- `k`: B样条阶数 (默认 3)
- `num_points`: 输出点数 (默认 100)

#### 贝塞尔平滑 (bezier)
- `num_points`: 输出点数 (默认 100)

#### 梯度下降平滑 (gradient_descent)
- `alpha`: 平滑项权重 (默认 0.1)
- `beta`: 障碍物排斥项权重 (默认 0.1)
- `iterations`: 迭代次数 (默认 100)

## 障碍物格式

### 圆形障碍物
```python
[x_center, y_center, radius]
```

### 多边形障碍物
```python
[[x1, y1], [x2, y2], [x3, y3], ...]
```

## 文件结构

```
.
├── rrt_star.py          # RRT* 算法核心实现
├── path_smoothing.py    # 路径平滑模块
├── main.py              # 主程序和示例
├── requirements.txt     # 依赖库列表
└── README.md            # 说明文档
```

## 算法原理

### RRT* vs RRT
- **RRT**: 快速找到可行路径，但不一定最优
- **RRT***: 通过 choose_parent 和 rewire 步骤逐步优化路径，渐近最优

### 核心步骤
1. 随机采样节点
2. 找到最近节点
3. 向随机节点方向扩展新节点
4. 在新节点附近寻找最优父节点
5. 对附近节点进行重布线优化
6. 到达目标后回溯生成路径

## 注意事项

1. 增大 `max_iter` 可获得更优路径，但计算时间更长
2. `expand_dis` 过小会导致搜索缓慢，过大可能穿过障碍物
3. 路径平滑可能导致路径进入障碍物，程序会进行碰撞检测修正
4. 对于复杂环境，建议启用动画观察搜索过程
