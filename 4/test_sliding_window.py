import numpy as np
import sys
sys.path.insert(0, '.')
from app import OutlierDetector

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def test_case_1_basic_time_series():
    """测试用例1: 基本时间序列异常检测"""
    print_section("测试用例1: 基本时间序列异常检测")
    
    data = [1, 2, 3, 4, 5, 6, 7, 8, 100, 9, 10, 11]
    print(f"数据: {data}")
    print(f"异常点: 索引8, 值=100")
    
    detector = OutlierDetector(data)
    result = detector.detect_sliding_window(window_size=5, k=2.0)
    
    print(f"\n检测结果:")
    print(f"  方法: {result['method']}")
    print(f"  窗口大小: {result['window_size']}")
    print(f"  异常值数量: {result['outlier_count']}")
    print(f"  异常值索引: {result['outlier_indices']}")
    print(f"  异常值: {result['outlier_values']}")
    
    print(f"\n窗口均值: {[round(x, 2) for x in result['window_means']}")
    print(f"偏离程度: {[round(x, 2) for x in result['deviations']}")
    
    assert result['outlier_count'] >= 1, "应该检测到至少1个异常值"
    assert 100 in result['outlier_values'], "100应该被检测为异常值"
    print("\n✅ 测试用例1通过")

def test_case_2_centered_window():
    """测试用例2: 中心窗口模式"""
    print_section("测试用例2: 中心窗口模式")
    
    data = [10, 10, 10, 10, 50, 10, 10, 10, 10]
    print(f"数据: {data}")
    print(f"异常点: 索引4, 值=50")
    
    detector = OutlierDetector(data)
    
    print("\n前向窗口 (center=False):")
    result1 = detector.detect_sliding_window(window_size=5, k=2.0, center=False)
    print(f"  异常值索引: {result1['outlier_indices']}")
    print(f"  偏离程度: {[round(x, 2) for x in result1['deviations']}")
    
    print("\n中心窗口 (center=True):")
    result2 = detector.detect_sliding_window(window_size=5, k=2.0, center=True)
    print(f"  异常值索引: {result2['outlier_indices']}")
    print(f"  偏离程度: {[round(x, 2) for x in result2['deviations']}")
    
    print("\n✅ 测试用例2通过")

def test_case_3_gradual_trend():
    """测试用例3: 渐变趋势中的异常"""
    print_section("测试用例3: 渐变趋势中的异常")
    
    np.random.seed(42)
    base = np.arange(1, 21)  # 1到20的递增序列
    noise = np.random.normal(0, 0.5, 20)
    data = (base + noise).tolist()
    data[10] = 50  # 在中间插入异常值
    data[15] = -10  # 再插入一个异常值
    
    print(f"数据: {[round(x, 1) for x in data]}")
    
    detector = OutlierDetector(data)
    result = detector.detect_sliding_window(window_size=4, k=2.5)
    
    print(f"\n检测结果:")
    print(f"  异常值数量: {result['outlier_count']}")
    print(f"  异常值索引: {result['outlier_indices']}")
    print(f"  异常值: {result['outlier_values']}")
    
    handle_result = detector.handle_outliers('mean')
    print(f"\n均值替换后:")
    print(f"  清洗后数据: {[round(x, 1) for x in handle_result['cleaned_data']}")
    
    print("\n✅ 测试用例3通过")

def test_case_4_small_window():
    """测试用例4: 小窗口测试"""
    print_section("测试用例4: 小窗口测试")
    
    data = [5, 5, 5, 20, 5, 5, 5]
    print(f"数据: {data}")
    
    detector = OutlierDetector(data)
    
    for window_size in [2, 3, 5]:
        result = detector.detect_sliding_window(window_size=window_size, k=1.5)
        print(f"窗口大小={window_size}: 异常值={result['outlier_values']}")
    
    print("\n✅ 测试用例4通过")

def test_case_5_constant_data():
    """测试用例5: 恒定数据"""
    print_section("测试用例5: 恒定数据")
    
    data = [7, 7, 7, 7, 7, 7, 7]
    print(f"数据: {data}")
    
    detector = OutlierDetector(data)
    result = detector.detect_sliding_window(window_size=3, k=2.0)
    
    print(f"异常值数量: {result['outlier_count']}")
    print(f"窗口标准差: {[round(x, 4) for x in result['window_stds']}")
    
    assert result['outlier_count'] == 0, "恒定数据不应有异常值"
    print("\n✅ 测试用例5通过")

def test_case_6_multiple_outliers():
    """测试用例6: 连续多个异常值"""
    print_section("测试用例6: 连续多个异常值")
    
    data = [1, 2, 3, 100, 200, 300, 4, 5, 6]
    print(f"数据: {data}")
    
    detector = OutlierDetector(data)
    result = detector.detect_sliding_window(window_size=3, k=1.5)
    
    print(f"检测结果:")
    print(f"  异常值数量: {result['outlier_count']}")
    print(f"  异常值索引: {result['outlier_indices']}")
    print(f"  异常值: {result['outlier_values']}")
    
    for method in ['remove', 'mean', 'median']:
        handle_result = detector.handle_outliers(method)
        print(f"\n处理方法: {method}")
        print(f"  清洗后数据: {handle_result['cleaned_data']}")
    
    print("\n✅ 测试用例6通过")

def test_case_7_edge_cases():
    """测试用例7: 边界情况"""
    print_section("测试用例7: 边界情况")
    
    print("测试最小数据量 (2个点):")
    data = [1, 100]
    detector = OutlierDetector(data)
    result = detector.detect_sliding_window(window_size=2, k=1.5)
    print(f"  数据: {data}")
    print(f"  异常值: {result['outlier_values']}")
    
    print(f"\n测试窗口大于数据长度:")
    data = [1, 2, 3, 4, 5]
    result = detector = OutlierDetector(data)
    result = detector.detect_sliding_window(window_size=10, k=1.5)
    print(f"  数据: {data}")
    print(f"  异常值: {result['outlier_values']}")
    
    print("\n✅ 测试用例7通过")

def test_case_8_negative_values():
    """测试用例8: 包含负数的时间序列"""
    print_section("测试用例8: 包含负数的时间序列")
    
    data = [-5, -4, -3, -2, -1, -100, 0, 1, 2]
    print(f"数据: {data}")
    
    detector = OutlierDetector(data)
    result = detector.detect_sliding_window(window_size=4, k=2.0)
    
    print(f"异常值数量: {result['outlier_count']}")
    print(f"异常值: {result['outlier_values']}")
    
    print("\n✅ 测试用例8通过")

def test_case_9_seasonal_pattern():
    """测试用例9: 季节性模式中的异常"""
    print_section("测试用例9: 季节性模式中的异常")
    
    t = np.arange(20)
    seasonal = 10 * np.sin(t * 0.5)
    data = seasonal.tolist()
    data[10] = 50  # 异常高峰
    data[15] = -30  # 异常低谷
    
    print(f"数据: {[round(x, 1) for x in data}")
    
    detector = OutlierDetector(data)
    result = detector.detect_sliding_window(window_size=6, k=2.5)
    
    print(f"\n检测结果:")
    print(f"  异常值数量: {result['outlier_count']}")
    print(f"  异常值索引: {result['outlier_indices']}")
    print(f"  异常值: {result['outlier_values']}")
    
    print("\n✅ 测试用例9通过")

if __name__ == "__main__":
    try:
        test_case_1_basic_time_series()
        test_case_2_centered_window()
        test_case_3_gradual_trend()
        test_case_4_small_window()
        test_case_5_constant_data()
        test_case_6_multiple_outliers()
        test_case_7_edge_cases()
        test_case_8_negative_values()
        test_case_9_seasonal_pattern()
        
        print("\n" + "="*70)
        print("  🎉 所有滑动窗口测试全部通过！")
        print("="*70 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
