import numpy as np
import sys
sys.path.insert(0, '.')
from app import OutlierDetector

def test_zscore_zero_std():
    print("=== Unit Test: Z-score with Zero Standard Deviation ===")
    
    data = [5, 5, 5, 5, 5]
    detector = OutlierDetector(data)
    
    result = detector.detect_zscore()
    print(f"Method: {result['method']}")
    print(f"Mean: {result['mean']}")
    print(f"Std: {result['std']}")
    print(f"Outlier count: {result['outlier_count']}")
    print(f"Note: {result.get('note', 'N/A')}")
    
    assert result['outlier_count'] == 0, "Should have zero outliers"
    assert result['std'] == 0, "Standard deviation should be zero"
    print("✓ Test passed!\n")

def test_zscore_near_zero_std():
    print("=== Unit Test: Z-score with Near-Zero Standard Deviation ===")
    
    data = [1.0000000001, 1.0, 1.0000000002, 0.9999999999]
    detector = OutlierDetector(data)
    
    result = detector.detect_zscore()
    print(f"Method: {result['method']}")
    print(f"Mean: {result['mean']}")
    print(f"Std: {result['std']}")
    print(f"Outlier count: {result['outlier_count']}")
    print(f"Note: {result.get('note', 'N/A')}")
    
    assert result['outlier_count'] == 0, "Should have zero outliers"
    print("✓ Test passed!\n")

def test_zscore_normal_case():
    print("=== Unit Test: Z-score Normal Case ===")
    
    data = [1, 2, 3, 4, 5, 100]
    detector = OutlierDetector(data)
    
    result = detector.detect_zscore(threshold=2.0)
    print(f"Method: {result['method']}")
    print(f"Mean: {result['mean']:.2f}")
    print(f"Std: {result['std']:.2f}")
    print(f"Outlier count: {result['outlier_count']}")
    print(f"Outlier values: {result['outlier_values']}")
    
    assert result['outlier_count'] == 1, "Should detect one outlier"
    assert 100 in result['outlier_values'], "100 should be detected as outlier"
    print("✓ Test passed!\n")

def test_handle_outliers_no_outliers():
    print("=== Unit Test: Handle Outliers when No Outliers Detected ===")
    
    data = [5, 5, 5, 5, 5]
    detector = OutlierDetector(data)
    detector.detect_zscore()
    
    for method in ['remove', 'mean', 'median']:
        result = detector.handle_outliers(method)
        print(f"Handling method: {method}")
        print(f"Original length: {result['original_length']}")
        print(f"Cleaned length: {result['cleaned_length']}")
        print(f"Cleaned data: {result['cleaned_data']}")
        
        assert result['original_length'] == result['cleaned_length'], "Length should be same when no outliers"
        assert all(x == 5 for x in result['cleaned_data']), "All values should be 5"
    print("✓ All handling methods tested and passed!\n")

def test_iqr_constant_data():
    print("=== Unit Test: IQR with Constant Data ===")
    
    data = [7, 7, 7, 7, 7, 7]
    detector = OutlierDetector(data)
    
    result = detector.detect_iqr()
    print(f"Method: {result['method']}")
    print(f"Q1: {result['q1']}, Q3: {result['q3']}, IQR: {result['iqr']}")
    print(f"Lower bound: {result['lower_bound']}, Upper bound: {result['upper_bound']}")
    print(f"Outlier count: {result['outlier_count']}")
    
    assert result['outlier_count'] == 0, "Should have zero outliers"
    assert result['iqr'] == 0, "IQR should be zero for constant data"
    print("✓ Test passed!\n")

if __name__ == "__main__":
    try:
        test_zscore_zero_std()
        test_zscore_near_zero_std()
        test_zscore_normal_case()
        test_handle_outliers_no_outliers()
        test_iqr_constant_data()
        print("=" * 50)
        print("All unit tests passed successfully! ✅")
        print("=" * 50)
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
