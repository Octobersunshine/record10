# Parks-McClellan 等波纹 FIR 滤波器使用说明

## 概述

Parks-McClellan 算法是一种最优 FIR 滤波器设计方法，基于 Remez 交换算法实现最小最大误差优化。

**核心特点:**
- ✅ **等波纹特性**: 通带和阻带内的误差均匀分布
- ✅ **最优设计**: 给定阶数下最优的频率响应
- ✅ **线性相位**: 对称抽头实现精确线性相位
- ✅ **固有稳定**: FIR 滤波器永远稳定
- ✅ **多频带支持**: 支持任意多频带设计

## 快速开始

### 基础使用

```python
from fir_parks_mcclellan import ParksMcClellanDesigner, print_filter_info

# 1. 创建设计器
designer = ParksMcClellanDesigner(fs=1000.0, verbose=True)

# 2. 设计低通滤波器
lp_filter = designer.design_lowpass(
    passband_freq=100.0,      # 通带截止频率 (Hz)
    stopband_freq=150.0,      # 阻带截止频率 (Hz)
    passband_ripple_db=1.0,   # 通带最大波纹 (dB)
    stopband_atten_db=40.0      # 阻带最小衰减 (dB)
)

# 3. 查看滤波器信息
print_filter_info(lp_filter)

# 4. 实时处理信号
output_sample = lp_filter.process_sample(input_sample)  # 单样本处理
output_block = lp_filter.process_block(input_block)  # 块处理

# 5. 导出 C 语言头文件
lp_filter.export_c_header('fir_coeffs.h')
```

## 滤波器类型

### 1. 低通滤波器

```python
lp_filter = designer.design_lowpass(
    passband_freq=100.0,
    stopband_freq=150.0,
    passband_ripple_db=1.0,
    stopband_atten_db=40.0,
    order=None  # 自动估计阶数
)
```

### 2. 高通滤波器

```python
hp_filter = designer.design_highpass(
    stopband_freq=50.0,
    passband_freq=100.0,
    passband_ripple_db=1.0,
    stopband_atten_db=40.0
)
```

### 3. 带通滤波器

```python
bp_filter = designer.design_bandpass(
    lower_stopband=60.0,   # 下阻带截止
    lower_passband=80.0,   # 下通带截止
    upper_passband=120.0,  # 上通带截止
    upper_stopband=140.0,  # 上阻带截止
    passband_ripple_db=1.0,
    stopband_atten_db=40.0
)
```

### 4. 带阻滤波器

```python
bs_filter = designer.design_bandstop(
    lower_passband=40.0,
    lower_stopband=48.0,
    upper_stopband=52.0,
    upper_passband=60.0,
    passband_ripple_db=1.0,
    stopband_atten_db=60.0
)
```

## 高级功能

### 多频带滤波器

```python
# 设计三频带滤波器
mb_filter = designer.design_multiband(
    bands=[0, 50, 70, 130, 150, 200, 250, 500],
    desired=[0, 0, 1, 1, 0, 0, 0.5, 0.5],
    weights=[2.0, 1.0, 2.0, 1.0],
    order=80,
    filter_type_name="三频带滤波器"
)
```

**参数说明:**
- `bands`: 频带边缘频率列表 (成对出现)
- `desired`: 各频带的期望幅度
- `weights`: 各频带的权重 (越大越重视)
- `order`: 滤波器阶数 (None 为自动估计)

### 自定义响应滤波器

```python
# 自定义幅度响应函数
def custom_response(f):
    if f < 100:
        return 1.0
    elif f < 200:
        return 1.0 - (f - 100) / 100 * 0.5
    elif f < 300:
        return 0.5
    else:
        return 0.0

# 设计自定义响应滤波器
arb_filter = designer.design_multiband(
    bands=[0, 100, 200, 300, 400, 1000],
    desired=[1.0, 1.0, 0.5, 0.5, 0.0, 0.0],
    order=100
)
```

## Parks-McClellan vs 算法原理

### 最小最大误差优化

Parks-McClellan 算法寻找一组抽头系数 h[n]，使得最大误差最小化：

```
minimize  max |H(e^(jω)) - H_desired(ω)|
         h[n]    ω∈[0,π]
```

### 交替定理

最优等波纹滤波器满足交替定理：
- 存在至少 r+2 个极值频率点
- 误差在这些点上交替取正负号
- 误差绝对值相等

### 波纹与阶数估计

```
N ≈ -20 log10(√(δ_pδ_s)) / (22 * Δf)
```

其中:
- `δ_p`: 通带波纹
- `δ_s`: 阻带波纹
- `Δf`: 过渡带宽度 (归一化)

## 性能对比

| 特性 | FIR (Parks-McClellan) | IIR Butterworth | IIR Chebyshev |
|------|----------------------|-----------------|---------------|
| 线性相位 | ✅ 是 | ❌ 否 | ❌ 否 |
| 固有稳定 | ✅ 是 | ❌ 否 | ❌ 否 |
| 等波纹 | ✅ 是 | ❌ 否 | ✅ 通带 |
| 阶数 | 高 | 低 | 最低 |
| 计算量 | 大 | 小 | 最小 |
| 相位失真 | 无 | 有 | 最大 |

## 使用示例

### 示例 1: 音频均衡器

```python
designer = ParksMcClellanDesigner(fs=44100.0)

# 三频带音频均衡器
eq_filter = designer.design_multiband(
    bands=[0, 200, 300, 2000, 3000, 20000],
    desired=[1.5, 1.5, 1.0, 1.0, 0.8, 0.8],  # 提升低音，降低高音
    weights=[1.0, 1.0, 1.0],
    order=200
)
```

### 示例 2: 多通道分离

```python
designer = ParksMcClellanDesigner(fs=8000.0)

# 电话按键音检测滤波器
dtmf_filter = designer.design_multiband(
    bands=[0, 680, 720, 740, 760, 830, 870, 940, 980, 1209, 1240, 1340, 1380, 1490, 1530, 4000],
    desired=[0,0, 1,1, 0,0, 1,1, 0,0, 1,1, 0,0, 1,1],
    weights=[10, 1, 10, 1, 10, 1, 10, 1],
    order=150
)
```

## C 语言实现

### 导出的头文件格式

```c
// FIR Filter Coefficients (Parks-McClellan)
// Type: Parks-McClellan Lowpass
// Order: 45
// Number of taps: 46
// Sampling Frequency: 1000.0 Hz

#define FIR_NUM_TAPS 46
#define FIR_ORDER 45
#define FIR_FS 1000.0f

const float fir_taps[46] = {
    1.234567890123e-03f,
    // ... 更多系数
};
```

### C 语言实时处理代码

```c
typedef struct {
    const float *taps;
    int num_taps;
    float *state;
    int state_idx;
} FIRFilter;

void fir_init(FIRFilter *f, const float *taps, int num_taps, float *state) {
    f->taps = taps;
    f->num_taps = num_taps;
    f->state = state;
    f->state_idx = 0;
    memset(state, 0, sizeof(float) * (num_taps - 1));
}

float fir_process(FIRFilter *f, float x) {
    float y = 0.0f;
    int i, j;
    
    f->state[f->state_idx] = x;
    
    j = f->state_idx;
    for (i = 0; i < f->num_taps; i++) {
        y += f->taps[i] * f->state[j];
        if (--j < 0) j = f->num_taps - 2;
    }
    
    if (++f->state_idx >= f->num_taps - 1) {
        f->state_idx = 0;
    }
    
    return y;
}
```

## 注意事项

### 1. 阶数选择
- 阶数越高，过渡带越窄
- 但计算量也越大
- 建议从估计阶数开始，根据需要调整

### 2. 过渡带宽度
- 过渡带越窄，阶数越高
- 过渡带宽度应 > fs/100 以上比较合理

### 3. 权重设置
- 权重越大，对应频带的误差越小
- 增加阻带权重可提高阻带衰减
- 增加通带权重可减小通带波纹

### 4. 线性相位
- Parks-McClellan 设计的是线性相位滤波器
- 群延迟 = (N-1)/2 个样本
- 信号会有固定的延迟，但各频率延迟相同

## 文件说明

| 文件 | 说明 |
|------|------|
| `fir_parks_mcclellan.py` | Parks-McClellan 等波纹 FIR 滤波器设计 |
| `fir_vs_iir_comparison.py` | FIR vs IIR 滤波器对比演示 |
| `iir_filter_fixed.py` | 带双线性变换预畸变的 IIR 滤波器 |
| `prewarp_verification.py` | 双线性变换预畸变验证 |

## 运行演示

```bash
# 运行 Parks-McClellan 演示
python fir_parks_mcclellan.py

# 运行 FIR vs IIR 对比
python fir_vs_iir_comparison.py
```

## 选型建议

### 选择 FIR (Parks-McClellan) 当:
- ✅ 需要线性相位 (无相位失真)
- ✅ 需要保证绝对稳定
- ✅ 需要精确的多频带响应
- ✅ 需要等波纹特性
- ✅ 计算资源充足

### 选择 IIR 当:
- ✅ 需要低阶数/少系数
- ✅ 计算资源受限
- ✅ 可以接受非线性相位
- ✅ 对相位要求不高

## 参考资料

- Parks, T. W., & McClellan, J. H. (1972). Chebyshev Approximation for Nonrecursive Digital Filters with Linear Phase.
- Remez Exchange Algorithm
- Digital Signal Processing (Proakis & Manolakis)
