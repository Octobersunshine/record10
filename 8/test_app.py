import requests
import json
import numpy as np

BASE_URL = 'http://localhost:5000'

def test_health():
    response = requests.get(f'{BASE_URL}/health')
    print('Health check:', response.json())
    return response.status_code == 200

def generate_non_linear_data(n_samples=100, n_features=8):
    np.random.seed(42)
    
    X = np.random.randn(n_samples, n_features)
    
    y_nonlinear = np.sin(X[:, 0]) + np.exp(X[:, 1] / 2) + np.random.randn(n_samples) * 0.1
    
    y_linear = 2 * X[:, 2] + 3 * X[:, 3] + np.random.randn(n_samples) * 0.1
    
    X[:, 4] = np.random.randint(0, 5, n_samples)
    X[:, 5] = np.random.randint(0, 2, n_samples)
    
    X[10:15, 0] = np.nan
    X[30:35, 2] = np.nan
    
    return X, y_nonlinear, y_linear

def test_pearson_vs_spearman():
    print(f'\n{"="*60}')
    print('Testing Pearson vs Spearman on non-linear data')
    print(f'{"="*60}')
    
    X, y_nonlinear, y_linear = generate_non_linear_data()
    
    for name, y in [('Non-linear', y_nonlinear), ('Linear', y_linear)]:
        print(f'\n--- {name} relationship ---')
        
        for method in ['pearson', 'spearman']:
            data = {
                'feature_matrix': X.tolist(),
                'target_vector': y.tolist(),
                'k': 4,
                'method': method,
                'strategy': 'mean'
            }
            
            response = requests.post(
                f'{BASE_URL}/feature-selection',
                json=data,
                headers={'Content-Type': 'application/json'}
            )
            
            result = response.json()
            
            print(f'\n  {method.upper()}:')
            print(f'    Top features: {[f["feature_index"] for f in result["top_k_features"]]}')
            print(f'    Correlations: {[f"{f["correlation"]:.3f}" for f in result["top_k_features"]]}')
            for f in result["top_k_features"]:
                print(f'      Feature {f["feature_index"]} ({f["feature_type"]}): {f["correlation"]:.3f} (p={f["p_value"]:.4f})')

def test_auto_recommendation():
    print(f'\n{"="*60}')
    print('Testing automatic method recommendation')
    print(f'{"="*60}')
    
    X, y_nonlinear, y_linear = generate_non_linear_data()
    
    for name, y in [('Non-linear', y_nonlinear), ('Linear', y_linear)]:
        print(f'\n--- {name} relationship ---')
        
        data = {
            'feature_matrix': X.tolist(),
            'target_vector': y.tolist(),
            'k': 3,
            'method': 'auto'
        }
        
        response = requests.post(
            f'{BASE_URL}/feature-selection',
            json=data,
            headers={'Content-Type': 'application/json'}
        )
        
        result = response.json()
        
        print(f'  Data analysis:')
        print(f'    Target type: {result["data_analysis"]["target_type"]}')
        print(f'    Feature types: {result["data_analysis"]["feature_type_counts"]}')
        print(f'    Recommended method: {result["data_analysis"]["recommended_method"]}')
        print(f'    Reasons: {result["data_analysis"]["recommendation_reason"]}')
        print(f'    Method used: {result["method_used"]}')

def test_p_values():
    print(f'\n{"="*60}')
    print('Testing p-value calculation')
    print(f'{"="*60}')
    
    np.random.seed(42)
    n_samples = 50
    
    X = np.random.randn(n_samples, 3)
    X[:, 0] = X[:, 0] * 2 + np.random.randn(n_samples) * 0.5
    X[:, 1] = np.random.randn(n_samples) * 10
    X[:, 2] = np.random.randn(n_samples) * 0.1
    
    y = X[:, 0] + np.random.randn(n_samples) * 0.5
    
    for method in ['pearson', 'spearman']:
        data = {
            'feature_matrix': X.tolist(),
            'target_vector': y.tolist(),
            'method': method
        }
        
        response = requests.post(
            f'{BASE_URL}/feature-selection',
            json=data,
            headers={'Content-Type': 'application/json'}
        )
        
        result = response.json()
        
        print(f'\n  {method.upper()}:')
        for f in result['top_k_features']:
            sig = '*' if f['p_value'] < 0.05 else ''
            print(f'    Feature {f["feature_index"]}: r={f["correlation"]:.3f}, p={f["p_value"]:.6f} {sig}')

def test_filling_strategies_with_methods():
    print(f'\n{"="*60}')
    print('Testing all filling strategies with both correlation methods')
    print(f'{"="*60}')
    
    X, y_nonlinear, _ = generate_non_linear_data(n_samples=80)
    
    strategies = ['delete', 'mean', 'median']
    methods = ['pearson', 'spearman']
    
    all_passed = True
    
    for strategy in strategies:
        for method in methods:
            data = {
                'feature_matrix': X.tolist(),
                'target_vector': y_nonlinear.tolist(),
                'k': 3,
                'strategy': strategy,
                'method': method
            }
            
            response = requests.post(
                f'{BASE_URL}/feature-selection',
                json=data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code != 200:
                print(f'✗ Strategy {strategy} + Method {method} failed')
                all_passed = False
            else:
                result = response.json()
                has_nan = any(np.isnan(f.get('correlation', 1)) for f in result['top_k_features'])
                if has_nan:
                    print(f'⚠ Strategy {strategy} + Method {method}: NaN in results')
                    all_passed = False
                else:
                    print(f'✓ Strategy {strategy} + Method {method}: OK')
    
    return all_passed

def test_invalid_method():
    print(f'\n{"="*60}')
    print('Testing invalid method handling')
    print(f'{"="*60}')
    
    X = np.random.randn(10, 3)
    y = np.random.randn(10)
    
    data = {
        'feature_matrix': X.tolist(),
        'target_vector': y.tolist(),
        'method': 'invalid_method'
    }
    
    response = requests.post(
        f'{BASE_URL}/feature-selection',
        json=data,
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code == 400:
        print(f'✓ Correctly rejected invalid method')
        return True
    else:
        print(f'✗ Should have rejected invalid method')
        return False

def test_feature_types():
    print(f'\n{"="*60}')
    print('Testing feature type detection')
    print(f'{"="*60}')
    
    np.random.seed(42)
    n_samples = 100
    
    X = np.zeros((n_samples, 4))
    X[:, 0] = np.random.randint(0, 2, n_samples)
    X[:, 1] = np.random.randint(0, 5, n_samples)
    X[:, 2] = np.random.randn(n_samples)
    X[:, 3] = np.random.choice([1, 2, 3, 4, 5], n_samples) + np.random.randn(n_samples) * 0.1
    
    y = X[:, 1] + np.random.randn(n_samples) * 0.5
    
    data = {
        'feature_matrix': X.tolist(),
        'target_vector': y.tolist(),
        'method': 'auto'
    }
    
    response = requests.post(
        f'{BASE_URL}/feature-selection',
        json=data,
        headers={'Content-Type': 'application/json'}
    )
    
    result = response.json()
    
    print(f'  Feature types detected:')
    for i, ft in enumerate(result['data_analysis']['feature_types']):
        print(f'    Feature {i}: {ft}')
    print(f'  Target type: {result["data_analysis"]["target_type"]}')
    print(f'  Recommended method: {result["data_analysis"]["recommended_method"]}')

if __name__ == '__main__':
    print('Testing Advanced Feature Selection API\n')
    
    print('1. Testing health endpoint...')
    if test_health():
        print('✓ Health check passed')
    else:
        print('✗ Health check failed')
    
    print('\n2. Testing Pearson vs Spearman comparison...')
    try:
        test_pearson_vs_spearman()
        print('✓ Pearson vs Spearman test completed')
    except Exception as e:
        print(f'✗ Error: {e}')
        print('Make sure the Flask server is running (python app.py)')
    
    print('\n3. Testing automatic method recommendation...')
    try:
        test_auto_recommendation()
        print('✓ Auto recommendation test completed')
    except Exception as e:
        print(f'✗ Error: {e}')
    
    print('\n4. Testing p-value calculation...')
    try:
        test_p_values()
        print('✓ P-value test completed')
    except Exception as e:
        print(f'✗ Error: {e}')
    
    print('\n5. Testing all strategies with both methods...')
    try:
        if test_filling_strategies_with_methods():
            print('✓ All strategy + method combinations passed')
        else:
            print('✗ Some combinations failed')
    except Exception as e:
        print(f'✗ Error: {e}')
    
    print('\n6. Testing invalid method handling...')
    try:
        if test_invalid_method():
            print('✓ Invalid method handling passed')
        else:
            print('✗ Invalid method handling failed')
    except Exception as e:
        print(f'✗ Error: {e}')
    
    print('\n7. Testing feature type detection...')
    try:
        test_feature_types()
        print('✓ Feature type detection completed')
    except Exception as e:
        print(f'✗ Error: {e}')
    
    print(f'\n{"="*60}')
    print('All tests completed!')
    print(f'{"="*60}')
