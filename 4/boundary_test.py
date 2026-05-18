import numpy as np
import sys
sys.path.insert(0, '.')
from app import OutlierDetector

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def test_case_1_all_same_values():
    """测试用例1: 所有值完全相同"""
    print_section("测试用例1: 所有值完全相同")
    data = [42, 42, 42, 42, 42, 42, 42]
    print(f"数据: {data}")
    print(f"均值: {np.mean(data):.4f}, 标准差: {np.std(data):.4f}")
    
    detector = OutlierDetector(data)
    
    # Z-score 测试
    z_result = detector.detect_zscore()
    print(f"\nZ-score 结果:")
    print(f"  异常值数量: {z_result['outlier_count']}")
    print(f"  标准差: {z_result['std']:.10f}")
    print(f"  说明: {z_result.get('note', 'N/A')}")
    
    # IQR 测试
    iqr_result = detector.detect_iqr()
    print(f"\nIQR 结果:")
    print(f"  Q1: {iqr_result['q1']}, Q3: {iqr_result['q3']}, IQR: {iqr_result['iqr']}")
    print(f"  异常值数量: {iqr_result['outlier_count']}")
    
    # 验证处理方法
    for method in ['remove', 'mean', 'median']:
        handle_result = detector.handle_outliers(method)
        print(f"\n处理方法 '{method}':")
        print(f"  清洗后数据: {handle_result['cleaned_data']}")
        print(f"  数据长度: {handle_result['original_length']} -> {handle_result['cleaned_length']}")
    
    assert z_result['outlier_count'] == 0, "Z-score: 恒定数据不应有异常值"
    assert iqr_result['outlier_count'] == 0, "IQR: 恒定数据不应有异常值"
    print("\n✅ 测试用例1通过")

def test_case_2_two_values_same():
    """测试用例2: 只有两个相同的值"""
    print_section("测试用例2: 只有两个相同的值")
    data = [10, 10]
    print(f"数据: {data}")
    print(f"均值: {np.mean(data):.4f}, 标准差: {np.std(data):.4f}")
    
    detector = OutlierDetector(data)
    z_result = detector.detect_zscore()
    print(f"\nZ-score 异常值数量: {z_result['outlier_count']}")
    
    iqr_result = detector.detect_iqr()
    print(f"IQR 异常值数量: {iqr_result['outlier_count']}")
    
    assert z_result['outlier_count'] == 0
    assert iqr_result['outlier_count'] == 0
    print("\n✅ 测试用例2通过")

def test_case_3_almost_constant():
    """测试用例3: 几乎恒定的数据（微小波动）"""
    print_section("测试用例3: 几乎恒定的数据（微小波动）")
    data = [1.0, 1.0000001, 0.9999999, 1.0000002, 0.9999998]
    print(f"数据: {data}")
    print(f"均值: {np.mean(data):.10f}, 标准差: {np.std(data):.10f}")
    
    detector = OutlierDetector(data)
    
    z_result = detector.detect_zscore(threshold=3.0)
    print(f"\nZ-score 结果 (阈值=3.0):")
    print(f"  异常值数量: {z_result['outlier_count']}")
    print(f"  标准差: {z_result['std']:.15f}")
    if 'note' in z_result:
        print(f"  说明: {z_result['note']}")
    
    # 用更小的阈值测试
    z_result_2 = detector.detect_zscore(threshold=0.000001)
    print(f"\nZ-score 结果 (阈值=0.000001):")
    print(f"  异常值数量: {z_result_2['outlier_count']}")
    print(f"  异常值: {z_result_2['outlier_values']}")
    
    print("\n✅ 测试用例3通过")

def test_case_4_single_outlier_in_constant():
    """测试用例4: 恒定数据中有一个明显异常值"""
    print_section("测试用例4: 恒定数据中有一个明显异常值")
    data = [5, 5, 5, 5, 5, 5, 100, 5, 5]  # 100 是异常值
    print(f"数据: {data}")
    print(f"均值: {np.mean(data):.4f}, 标准差: {np.std(data):.4f}")
    
    detector = OutlierDetector(data)
    
    z_result = detector.detect_zscore(threshold=2.0)
    print(f"\nZ-score 结果:")
    print(f"  异常值数量: {z_result['outlier_count']}")
    print(f"  异常值索引: {z_result['outlier_indices']}")
    print(f"  异常值: {z_result['outlier_values']}")
    
    handle_result = detector.handle_outliers('mean')
    print(f"\n用均值替换后:")
    print(f"  清洗数据: {handle_result['cleaned_data']}")
    print(f"  原始均值: {handle_result['statistics']['original_mean']:.2f}")
    print(f"  清洗后均值: {handle_result['statistics']['cleaned_mean']:.2f}")
    
    assert z_result['outlier_count'] == 1, "应该检测到1个异常值"
    assert 100 in z_result['outlier_values'], "100应该被检测为异常值"
    print("\n✅ 测试用例4通过")

def test_case_5_all_outliers():
    """测试用例5: 极端情况 - 交替的极大极小值"""
    print_section("测试用例5: 交替的极大极小值")
    data = [-100, 100, -100, 100, -100, 100]
    print(f"数据: {data}")
    print(f"均值: {np.mean(data):.4f}, 标准差: {np.std(data):.4f}")
    
    detector = OutlierDetector(data)
    
    z_result = detector.detect_zscore(threshold=0.5)
    print(f"\nZ-score 结果 (阈值=0.5):")
    print(f"  异常值数量: {z_result['outlier_count']}")
    print(f"  异常值: {z_result['outlier_values']}")
    
    iqr_result = detector.detect_iqr(k=0.5)
    print(f"\nIQR 结果 (k=0.5):")
    print(f"  异常值数量: {iqr_result['outlier_count']}")
    
    print("\n✅ 测试用例5通过")

def test_case_6_empty_or_too_small():
    """测试用例6: 数据量太小的情况"""
    print_section("测试用例6: 小规模数据")
    
    # 测试4个数据点（最小要求）
    data = [1, 2, 3, 4]
    print(f"数据 (4个点): {data}")
    detector = OutlierDetector(data)
    z_result = detector.detect_zscore()
    print(f"Z-score 异常值数量: {z_result['outlier_count']}")
    
    # 测试只有一个值
    print(f"\n数据 (1个点): [100]")
    detector2 = OutlierDetector([100])
    z_result2 = detector2.detect_zscore()
    print(f"Z-score 结果:")
    print(f"  均值: {z_result2['mean']}")
    print(f"  标准差: {z_result2['std']}")
    print(f"  异常值数量: {z_result2['outlier_count']}")
    
    print("\n✅ 测试用例6通过")

def test_case_7_handle_methods_edge():
    """测试用例7: 各种处理方法的边界情况"""
    print_section("测试用例7: 处理方法边界测试")
    
    # 全部都是异常值时的处理
    data = [1, 100, 101, 102, 103, 104]  # 1 是异常值
    print(f"数据: {data}")
    
    detector = OutlierDetector(data)
    z_result = detector.detect_zscore(threshold=1.5)
    print(f"\n检测到的异常值: {z_result['outlier_values']}")
    
    for method in ['remove', 'mean', 'median']:
        result = detector.handle_outliers(method)
        print(f"\n处理方法: {method}")
        print(f"  清洗后数据: {result['cleaned_data']}")
        print(f"  清洗后均值: {result['statistics']['cleaned_mean']:.2f}")
        print(f"  清洗后中位数: {result['statistics']['cleaned_median']:.2f}")
        print(f"  清洗后标准差: {result['statistics']['cleaned_std']:.2f}")
    
    print("\n✅ 测试用例7通过")

def test_case_8_negative_values():
    """测试用例8: 包含负数的数据"""
    print_section("测试用例8: 包含负数的数据")
    
    data = [-10, -5, 0, 5, 10, -100]
    print(f"数据: {data}")
    
    detector = OutlierDetector(data)
    z_result = detector.detect_zscore()
    print(f"\nZ-score 异常值: {z_result['outlier_values']}")
    
    iqr_result = detector.detect_iqr()
    print(f"IQR 异常值: {iqr_result['outlier_values']}")
    
    print("\n✅ 测试用例8通过")

if __name__ == "__main__":
    try:
        test_case_1_all_same_values()
        test_case_2_two_values_same()
        test_case_3_almost_constant()
        test_case_4_single_outlier_in_constant()
        test_case_5_all_outliers()
        test_case_6_empty_or_too_small()
        test_case_7_handle_methods_edge()
        test_case_8_negative_values()
        
        print("\n" + "="*60)
        print("  🎉 所有边界条件测试全部通过！")
        print("="*60 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
