# 指纹图像方向场计算与鲁棒奇异点检测

基于Python实现的指纹图像方向场计算和奇异点检测，针对低质量指纹（模糊、噪声、湿润、干燥）进行了全面优化，支持传统计算机视觉方法和深度学习（CNN关键点回归网络）方法。

## 功能特性

- **方向场计算**: 基于梯度法计算指纹图像的方向场
- **Gabor滤波增强**: 自适应方向的Gabor滤波器增强指纹脊线
- **多分辨率分析**: 多尺度金字塔检测，提高鲁棒性
- **置信度阈值**: 基于方向一致性的奇异点置信度评估
- **深度学习检测**: CNN关键点回归网络，适用于湿润、干燥手指
- **奇异点检测**: 
  - Poincare指数法: 检测核心点(Core, Poincare指数≈1)和三角点(Delta, Poincare指数≈-1)
  - 复滤波法: 基于复数卷积的奇异点检测
  - 深度学习法: 基于CNN的关键点回归
- **合成指纹生成**: 支持生成4种典型指纹类型，模拟湿润、干燥、模糊、噪声等多种条件
- **可视化**: 方向场可视化、Poincare指数图、奇异点标注及置信度显示

## 安装依赖

```bash
pip install -r requirements.txt
```

依赖包:
- numpy
- opencv-python
- matplotlib
- scikit-image
- scipy
- torch (可选，用于深度学习)
- torchvision (可选，用于深度学习)

## 使用方法

### 1. 运行默认测试（所有方法对比）

```bash
python main.py
```

这将生成4种典型指纹类型并对比多种检测方法，同时测试湿润、干燥、模糊、噪声等条件下的鲁棒性。

### 2. 处理自定义指纹图像

```bash
python main.py path/to/your/fingerprint.png

# 使用训练好的深度学习模型
python main.py path/to/your/fingerprint.png path/to/model.pth
```

### 3. 训练深度学习模型

```bash
# 快速测试数据加载和模型
python train_model.py --quick_test

# 完整训练
python train_model.py --num_train 5000 --num_val 500 --epochs 30 --batch_size 16

# 使用Heatmap-based模型训练
python train_model.py --use_heatmap --epochs 50
```

### 4. 生成数据集样本

```bash
python data_generator.py
```

这将生成各种条件下的指纹样本到 `dataset_samples/` 目录。

## 低质量指纹优化方案

### 1. Gabor滤波增强 (`gabor_enhance.py`)

**核心函数**:
- `fast_enhance_fingerprint(image, theta_field, ridge_frequency)`: 快速Gabor增强
- `multiscale_enhance(image, theta_field, scales)`: 多尺度Gabor增强
- `estimate_ridge_frequency(image, block_size)`: 估计脊线频率
- `estimate_quality_map(image, theta_field, block_size)`: 估计指纹质量图

### 2. 置信度阈值机制 (`singularity.py`)

```
置信度 = Poincare指数准确度 × 方向一致性
```

**核心函数**:
- `calculate_singularity_confidence(theta_field, x, y, poincare_idx, coherence_map)`: 计算奇异点置信度
- `detect_singularities_robust(...)`: 带置信度过滤的鲁棒检测

### 3. 多分辨率分析策略

**算法原理**:
1. 构建图像金字塔 (scales=[1.0, 0.75, 0.5])
2. 在每个尺度上独立检测奇异点
3. 跨尺度聚类验证：只有在至少2个尺度上都检测到的奇异点才被保留

### 4. 深度学习检测 (`deep_learning.py`)

#### 模型架构

**回归网络 (SingularityNet)**:
- 4层CNN特征提取
- 3层全连接层回归
- 输出: N个关键点 × (x, y, confidence)

**热图网络 (SingularityHeatmapNet)**:
- 编码-解码UNet架构
- 输出关键点热图
- 更精确的定位

#### 训练数据增强 (`data_generator.py`)

支持多种指纹变异:
- **湿润效果**: 变暗 + 模糊 + 水纹噪声
- **干燥效果**: 高对比度 + 颗粒噪声 + 中值滤波
- **模糊效果**: 高斯模糊
- **噪声效果**: 高斯噪声
- **弹性形变**: 模拟皮肤变形
- **旋转缩放**: 随机角度旋转和尺度变化

#### 核心函数
- `SingularityDetector.detect(image, confidence_threshold)`: 深度学习检测
- `apply_wet_effect(image, severity)`: 模拟湿润手指
- `apply_dry_effect(image, severity)`: 模拟干燥手指
- `FingerprintDataset`: PyTorch数据集类

## 模块说明

### orientation.py - 方向场计算

**核心函数**:
- `compute_orientation_field(image, block_size, gradient_sigma, orientation_sigma)`: 计算指纹方向场
- `visualize_orientation_field(image, theta_field, step, scale)`: 方向场可视化

### singularity.py - 鲁棒奇异点检测

**核心函数**:
- `calculate_poincare_index(theta_field, x, y, radius)`: 计算指定点的Poincare指数
- `detect_singularities_robust(theta_field, coherence_map, ...)`: 鲁棒奇异点检测
- `multiscale_singularity_detection(image, theta_field_func, ...)`: 多分辨率检测
- `detect_complex_filter(theta_field, block_size)`: 复滤波法检测

### gabor_enhance.py - Gabor滤波增强

**核心函数**:
- `create_gabor_kernel(ksize, sigma, theta, lambd, gamma, psi)`: 创建Gabor核
- `fast_enhance_fingerprint(image, theta_field)`: 快速Gabor增强
- `multiscale_enhance(image, theta_field, scales)`: 多尺度Gabor增强

### deep_learning.py - 深度学习检测

**核心类与函数**:
- `SingularityNet`: CNN回归网络
- `SingularityHeatmapNet`: UNet热图网络
- `SingularityDetector`: 检测器封装类
- `keypoint_loss`: 关键点回归损失
- `heatmap_loss`: 热图损失

### data_generator.py - 数据生成与增强

**核心函数**:
- `generate_synthetic_fingerprint_with_keypoints(...)`: 生成带标注的合成指纹
- `apply_wet_effect`, `apply_dry_effect`, `apply_blur_effect`, `apply_noise_effect`: 各种质量退化
- `augment_fingerprint(...)`: 综合数据增强
- `FingerprintDataset`: PyTorch数据集

### train_model.py - 训练脚本

**功能**:
- 模型训练与验证
- 学习率调度
- 模型保存与加载
- 鲁棒性评估（各种条件下测试）

### main.py - 主程序与对比测试

**核心函数**:
- `compare_all_methods(...)`: 对比所有检测方法
- `compare_wet_dry_conditions(...)`: 测试湿润/干燥等条件
- `process_fingerprint_robust(...)`: 鲁棒处理单张指纹

## 参数调优

### 深度学习训练参数
- `--num_train`: 训练样本数，默认5000
- `--num_val`: 验证样本数，默认500
- `--epochs`: 训练轮数，默认30
- `--batch_size`: 批次大小，默认16
- `--lr`: 学习率，默认0.001
- `--use_heatmap`: 使用热图模型

### Gabor增强参数
- `ksize`: Gabor核大小，默认25-31
- `sigma`: 高斯标准差，默认 `lambd * 0.5`
- `gamma`: 空间长宽比，默认0.5

### 鲁棒检测参数
- `poincare_threshold`: Poincare指数阈值，默认0.6
- `confidence_threshold`: 置信度阈值，默认0.3（减少假阳性的关键参数）
- `coherence_threshold`: 方向一致性阈值，默认0.2
- `min_distance`: 非极大值抑制最小距离，默认20

## 方法对比

| 方法 | 低质量鲁棒性 | 湿润/干燥 | 假阳性率 | 计算速度 |
|------|------------|----------|----------|----------|
| 基础Poincare | 低 | 差 | 高 | 快 |
| Poincare + 置信度 | 中 | 差 | 中 | 快 |
| Gabor增强 + 置信度 | 高 | 一般 | 低 | 中 |
| Gabor + 多分辨率 | 很高 | 较好 | 很低 | 较慢 |
| **深度学习** | **最高** | **优秀** | **极低** | 中 |

## 文件结构

```
.
├── orientation.py      # 方向场计算模块
├── singularity.py      # 鲁棒奇异点检测模块
├── gabor_enhance.py    # Gabor滤波增强模块
├── deep_learning.py    # 深度学习检测模块
├── data_generator.py   # 数据生成与增强
├── train_model.py      # 模型训练脚本
├── main.py             # 主程序与对比测试
├── requirements.txt    # 依赖列表
└── README.md           # 使用说明
```

## 典型输出

### 1. all_methods_*.png - 所有方法对比
3行 × N列（N为方法数）：
- 第1行：预处理后图像
- 第2行：检测结果（红点=Core, 蓝点=Delta）
- 第3行：Poincare指数或方向一致性

### 2. conditions_*.png - 各种条件测试
2行 × 5列：
- 第1行：不同条件的指纹（原始、湿润、干燥、模糊、噪声）
- 第2行：对应的检测结果

### 3. training_output/ - 训练输出
- `checkpoints/best_model.pth`: 最佳模型
- `checkpoints/epoch_*.pth`: 各轮模型
- `training_curves.png`: 训练曲线

## 训练流程示例

```bash
# 1. 安装PyTorch
pip install torch torchvision

# 2. 快速测试
python train_model.py --quick_test

# 3. 训练模型
python train_model.py --num_train 10000 --num_val 1000 --epochs 50 --batch_size 32

# 4. 使用训练好的模型进行检测
python main.py path/to/fingerprint.png training_output/checkpoints/best_model.pth
```

## 引用说明

本实现基于以下经典算法：
1. Poincare指数法用于奇异点检测
2. Gabor滤波用于指纹增强
3. CNN关键点回归用于深度学习检测
4. 多分辨率分析提高鲁棒性
