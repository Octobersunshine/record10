# 基频F0提取工具 - 支持多种算法

## 功能概述

提供三种基频提取方法，从传统到深度学习，适应不同场景需求：

| 方法 | 速度 | 精度 | 抗噪声 | 适用场景 |
|------|------|------|--------|---------|
| **ACF** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | 高信噪比、实时处理 |
| **pYIN** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 通用场景、平衡精度和速度 |
| **CREPE** | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 噪声环境、高精度需求 |

## 新增：深度学习方法

### pYIN（概率YIN算法）
- **原理**：YIN算法的概率版本，使用Viterbi解码
- **优势**：精度和鲁棒性显著优于传统ACF
- **特点**：无需深度学习模型，纯信号处理算法
- **安装**：librosa内置，无需额外安装

### CREPE（深度学习CNN）
- **原理**：基于CNN的深度学习基频提取器
- **优势**：在噪声环境下表现优异，精度最高
- **特点**：首次运行自动下载预训练模型
- **安装**：`pip install crepe tensorflow`

## 修复：基频范围固定问题

**问题**：原始实现假设固定的基频范围（如80-400Hz），对于低音（男声，<80Hz）或高音（女声/儿童，>400Hz）失效。

**解决方案**：
1. **语音类型预设**：提供9种预设，覆盖男声、女声、儿童、不同声部
2. **自适应基频范围**：自动分析信号，估计合适的基频范围
3. **倍频/半频校正**：检测并校正常见的倍频错误
4. **通用范围**：默认使用40-800Hz的宽范围，覆盖绝大多数情况

## 统一接口

所有方法都可以通过统一的 `extract_f0` 函数调用：

```python
from acf_f0_extraction import extract_f0, compare_methods

# 自相关法（快速）
f0, times = extract_f0('audio.wav', method='acf', voice_type='male')

# pYIN（高精度，鲁棒）
f0, times = extract_f0('audio.wav', method='pyin', voice_type='female')

# CREPE（深度学习，抗噪声）
f0, times = extract_f0('audio.wav', method='crepe', model_capacity='large')

# 对比多种方法
compare_methods('audio.wav', methods=['acf', 'pyin', 'crepe'])
```

## 方法详细说明

### 1. 自相关法（ACF）

#### 算法原理
自相关法是基频提取最经典的方法之一，基于以下原理：

1. **语音信号的周期性**：浊音信号具有准周期性，其周期对应基音周期
2. **自相关函数性质**：对于周期信号，其自相关函数在延迟等于周期时取得峰值
3. **公式**：
   ```
   R(τ) = Σ[n=0 to N-1-τ] x[n] * x[n+τ]
   ```
   其中 τ 为延迟量，R(τ) 为自相关值

#### 实现步骤
1. **预加重**：提升高频分量，补偿高频衰减
2. **基频范围确定**：语音类型预设或自适应
3. **分帧加窗**：30ms帧长，15ms帧移，汉明窗
4. **自相关计算**：计算每帧的自相关函数
5. **峰值检测**：在指定范围内寻找峰值
6. **倍频/半频校正**：基于中位数检测错误
7. **后处理**：中值滤波平滑

#### 适用场景
- 高信噪比语音
- 实时处理需求
- 计算资源有限的环境

---

### 2. pYIN（概率YIN算法）

#### 算法原理
pYIN是YIN算法的概率版本，主要改进：
1. **多阈值处理**：使用多个阈值进行基音检测
2. **概率分布**：计算每个可能F0值的后验概率
3. **Viterbi解码**：考虑时间连续性，优化F0轨迹

#### 核心参数
```python
f0, times = extract_f0(
    'audio.wav',
    method='pyin',
    voice_type='universal',      # 语音类型
    frame_duration=0.03,          # 帧长（秒）
    hop_duration=0.015,           # 帧移（秒）
    resolution=0.01,              # F0搜索分辨率
    boltzmann_parameter=2.0,      # Boltzmann分布参数
    max_transition_rate=35.92,    # 最大转换率
    switch_prob=0.01              # 清浊切换概率
)
```

#### 适用场景
- 通用语音处理任务
- 平衡精度和速度的需求
- 不需要深度学习依赖的场景

---

### 3. CREPE（深度学习CNN）

#### 算法原理
CREPE是在大规模数据集上训练的CNN模型，特点：
1. **端到端学习**：直接从波形学习基频特征
2. **鲁棒性强**：对噪声、混响等干扰不敏感
3. **多尺度处理**：适应不同基频范围的语音

#### 模型容量选择
| 容量 | 参数量 | 精度 | 速度 | 推荐场景 |
|------|--------|------|------|---------|
| `tiny` | ~400K | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 实时处理、移动端 |
| `small` | ~800K | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 平衡型 |
| `medium` | ~1.8M | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 精度优先 |
| `large` | ~5M | ⭐⭐⭐⭐⭐ | ⭐⭐ | 研究、高精度需求 |
| `full` | ~10M | ⭐⭐⭐⭐⭐ | ⭐ | 最高精度 |

#### 使用示例
```python
f0, times = extract_f0(
    'noisy_audio.wav',
    method='crepe',
    model_capacity='large',       # 模型容量
    voice_type='universal',        # 基频范围
    conf_threshold=0.5,            # 置信度阈值
    step_size=10                    # 分析步长（毫秒）
)
```

#### 适用场景
- 噪声环境下的语音
- 低质量录音
- 高精度基频提取需求
- 研究和学术用途

## 语音类型预设

| 预设类型 | 基频范围 | 适用场景 |
|---------|---------|---------|
| `male` | 50-250Hz | 普通男声 |
| `female` | 100-500Hz | 普通女声 |
| `child` | 200-800Hz | 儿童语音 |
| `tenor` | 80-350Hz | 男高音 |
| `bass` | 40-200Hz | 男低音 |
| `soprano` | 180-700Hz | 女高音 |
| `alto` | 120-450Hz | 女低音 |
| `universal` | 40-800Hz | 通用，默认选项 |
| `auto` | 自适应 | 配合auto_range参数使用 |

## 方法对比函数

```python
from acf_f0_extraction import compare_methods

# 对比所有可用方法
compare_methods(
    'your_audio.wav',
    methods=['acf', 'pyin', 'crepe'],
    voice_type='universal',
    save_path='f0_comparison.png'
)
```

对比函数会：
1. 输出每种方法的统计信息（平均F0、范围、浊音率）
2. 生成对比图，直观展示不同方法的结果差异
3. 帮助选择最适合特定音频的方法

## 使用方法

### 基本使用（统一接口）
```python
from acf_f0_extraction import extract_f0, plot_f0, extract_f0_statistics

# 使用默认方法（ACF）
f0, times = extract_f0('your_audio.wav')

# 或指定方法
f0, times = extract_f0('your_audio.wav', method='pyin')  # pYIN
f0, times = extract_f0('your_audio.wav', method='crepe')  # CREPE

# 绘制F0轨迹
plot_f0(f0, times)

# 获取统计信息
stats = extract_f0_statistics(f0)
print(stats)
```

### 使用语音类型预设
```python
# 男声（低音优化）
f0, times = extract_f0('male.wav', method='pyin', voice_type='male')

# 女声（中音优化）
f0, times = extract_f0('female.wav', method='pyin', voice_type='female')

# 儿童（高音优化）
f0, times = extract_f0('child.wav', method='pyin', voice_type='child')

# CREPE也支持语音类型
f0, times = extract_f0('singer.wav', method='crepe', voice_type='soprano')
```

### 噪声环境下推荐使用CREPE
```python
# CREPE对噪声最鲁棒
f0, times = extract_f0(
    'noisy_recording.wav',
    method='crepe',
    model_capacity='large',      # 使用大容量模型
    conf_threshold=0.3           # 降低置信度阈值，适应噪声
)
```

### 高精度要求推荐pYIN
```python
# 平衡精度和速度
f0, times = extract_f0(
    'studio_recording.wav',
    method='pyin',
    resolution=0.005,            # 提高搜索分辨率
    switch_prob=0.005            # 降低清浊切换概率
)
```

### 方法选择指南

| 场景 | 推荐方法 | 理由 |
|------|---------|------|
| 高信噪比、实时处理 | ACF | 速度最快 |
| 一般语音处理 | pYIN | 精度速度平衡好 |
| 有背景噪声 | CREPE | 抗噪声能力强 |
| 低质量录音 | CREPE | 对失真不敏感 |
| 研究/学术用途 | CREPE (full) | 最高精度 |
| 嵌入式/移动端 | CREPE (tiny) | 小体积模型 |

## 依赖安装

### 基础依赖（ACF + pYIN）
```bash
pip install -r requirements.txt
```

### 完整依赖（包含CREPE深度学习）
```bash
pip install numpy librosa matplotlib scipy soundfile crepe tensorflow
```

注意：TensorFlow安装可能需要根据系统配置选择合适的版本（CPU/GPU）。

## 输出说明

### 返回值
- `f0`: 基频轨迹数组，单位Hz（清浊音判断为清音时值为0）
- `times`: 对应时间点数组，单位秒

### 统计信息
- `mean_f0`: 平均基频（Hz）
- `std_f0`: 基频标准差
- `min_f0`: 最小基频
- `max_f0`: 最大基频
- `median_f0`: 中位数基频
- `voiced_ratio`: 浊音比例

## 运行演示
```bash
python acf_f0_extraction.py
```

演示程序会：
1. 显示所有可用的提取方法和语音类型预设
2. 生成一个带噪声的测试语音信号（F0变化：80Hz -> 150Hz -> 250Hz -> 350Hz）
3. 运行ACF和pYIN方法进行提取对比
4. 如果安装了CREPE，也会进行对比
5. 计算并显示每种方法的误差（MAE）
6. 生成对比图并保存为 `f0_methods_comparison.png`

## 文件说明
- `acf_f0_extraction.py`: 完整版（支持3种方法、音频读取、可视化）
- `acf_f0_simple.py`: 简化版（直接处理numpy数组，ACF方法）
- `requirements.txt`: 依赖包列表
- `README_ACF_F0.md`: 本文档
- `test_signal.wav`: 运行演示后生成的测试音频
- `f0_methods_comparison.png`: 运行演示后生成的方法对比图

## 选择语音类型的建议

| 情况 | 建议使用 |
|------|---------|
| 确定是普通成年男性语音 | `voice_type='male'` |
| 确定是普通成年女性语音 | `voice_type='female'` |
| 确定是儿童语音 | `voice_type='child'` |
| 语音特别低沉（如男低音歌手） | `voice_type='bass'` |
| 语音特别高亢（如女高音歌手） | `voice_type='soprano'` |
| 不确定说话人或混合语音 | `voice_type='universal'`（默认） |

## 方法详细参数

### ACF方法参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `frame_duration` | 0.03 | 帧长（秒） |
| `hop_duration` | 0.015 | 帧移（秒） |
| `pre_emphasis_coeff` | 0.97 | 预加重系数 |
| `window_type` | 'hamming' | 窗函数类型 |
| `median_filter_order` | 3 | 中值滤波阶数 |
| `harmonic_removal` | True | 是否启用倍频校正 |

### pYIN方法参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `frame_duration` | 0.03 | 帧长（秒） |
| `hop_duration` | 0.015 | 帧移（秒） |
| `resolution` | 0.01 | F0搜索分辨率 |
| `boltzmann_parameter` | 2.0 | Boltzmann分布参数 |
| `max_transition_rate` | 35.92 | 最大转换率（半音/秒） |
| `switch_prob` | 0.01 | 清浊切换概率 |

### CREPE方法参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_capacity` | 'full' | 模型容量: tiny/small/medium/large/full |
| `conf_threshold` | 0.5 | 置信度阈值 |
| `step_size` | 10 | 分析步长（毫秒） |

## 注意事项

1. 音频格式建议使用WAV格式
2. 采样率建议8kHz或16kHz
3. CREPE首次运行会自动下载预训练模型
4. TensorFlow可能需要单独安装（GPU版本性能更好）
5. 对于真实语音，优先选择正确的语音类型预设
6. 噪声环境下强烈推荐使用CREPE方法
7. pYIN方法是平衡精度和速度的好选择
