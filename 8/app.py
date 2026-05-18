from flask import Flask, request, jsonify
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy import stats

app = Flask(__name__)

VALID_STRATEGIES = ['delete', 'mean', 'median', 'mode', 'constant']
VALID_METHODS = ['pearson', 'spearman', 'auto']

def fill_missing_values(data, strategy='delete', constant_value=0):
    data = np.array(data, dtype=float)
    original_shape = data.shape
    
    nan_mask = np.isnan(data)
    nan_count = np.sum(nan_mask)
    total_count = data.size
    
    result = {
        'original_shape': list(original_shape),
        'nan_count': int(nan_count),
        'nan_percentage': float(nan_count / total_count * 100) if total_count > 0 else 0,
        'strategy': strategy,
        'filled_data': None
    }
    
    if nan_count == 0:
        result['filled_data'] = data
        result['message'] = 'No missing values found'
        return result
    
    if strategy == 'delete':
        result['filled_data'] = data
        result['message'] = 'Using pairwise deletion (no global filling)'
        return result
    
    elif strategy == 'mean':
        if data.ndim == 1:
            mean_val = np.nanmean(data)
            filled = np.where(nan_mask, mean_val, data)
            result['fill_value'] = float(mean_val)
        else:
            col_means = np.nanmean(data, axis=0)
            filled = data.copy()
            for i in range(data.shape[1]):
                filled[:, i] = np.where(np.isnan(filled[:, i]), col_means[i], filled[:, i])
            result['fill_values'] = col_means.tolist()
        result['filled_data'] = filled
        result['message'] = 'Filled with mean values'
    
    elif strategy == 'median':
        if data.ndim == 1:
            median_val = np.nanmedian(data)
            filled = np.where(nan_mask, median_val, data)
            result['fill_value'] = float(median_val)
        else:
            col_medians = np.nanmedian(data, axis=0)
            filled = data.copy()
            for i in range(data.shape[1]):
                filled[:, i] = np.where(np.isnan(filled[:, i]), col_medians[i], filled[:, i])
            result['fill_values'] = col_medians.tolist()
        result['filled_data'] = filled
        result['message'] = 'Filled with median values'
    
    elif strategy == 'mode':
        if data.ndim == 1:
            mode_result = stats.mode(data[~nan_mask], nan_policy='omit')
            mode_val = mode_result.mode[0] if hasattr(mode_result.mode, '__len__') and len(mode_result.mode) > 0 else mode_result.mode
            filled = np.where(nan_mask, mode_val, data)
            result['fill_value'] = float(mode_val)
        else:
            filled = data.copy()
            col_modes = []
            for i in range(data.shape[1]):
                col_data = data[:, i]
                col_nan_mask = np.isnan(col_data)
                if np.all(col_nan_mask):
                    col_mode = 0
                else:
                    mode_result = stats.mode(col_data[~col_nan_mask], nan_policy='omit')
                    col_mode = mode_result.mode[0] if hasattr(mode_result.mode, '__len__') and len(mode_result.mode) > 0 else mode_result.mode
                col_modes.append(float(col_mode))
                filled[:, i] = np.where(col_nan_mask, col_mode, filled[:, i])
            result['fill_values'] = col_modes
        result['filled_data'] = filled
        result['message'] = 'Filled with mode values'
    
    elif strategy == 'constant':
        filled = np.where(nan_mask, constant_value, data)
        result['filled_data'] = filled
        result['fill_value'] = float(constant_value)
        result['message'] = f'Filled with constant value: {constant_value}'
    
    return result

def analyze_data_type(X, y):
    n_samples, n_features = X.shape
    
    data_analysis = {
        'feature_types': [],
        'target_type': None,
        'recommended_method': None,
        'recommendation_reason': []
    }
    
    for i in range(n_features):
        feature = X[:, i]
        valid_data = feature[~np.isnan(feature)]
        
        if len(valid_data) == 0:
            data_analysis['feature_types'].append('empty')
            continue
        
        unique_ratio = len(np.unique(valid_data)) / len(valid_data)
        is_integer = np.allclose(valid_data, valid_data.astype(int))
        is_binary = len(np.unique(valid_data)) <= 2
        
        if is_binary:
            feat_type = 'binary'
        elif is_integer and unique_ratio < 0.1:
            feat_type = 'ordinal'
        elif unique_ratio > 0.9:
            feat_type = 'continuous'
        else:
            feat_type = 'mixed'
        
        data_analysis['feature_types'].append(feat_type)
    
    valid_y = y[~np.isnan(y)]
    if len(valid_y) > 0:
        unique_ratio_y = len(np.unique(valid_y)) / len(valid_y)
        is_integer_y = np.allclose(valid_y, valid_y.astype(int))
        is_binary_y = len(np.unique(valid_y)) <= 2
        
        if is_binary_y:
            data_analysis['target_type'] = 'binary'
        elif is_integer_y and unique_ratio_y < 0.1:
            data_analysis['target_type'] = 'ordinal'
        elif unique_ratio_y > 0.9:
            data_analysis['target_type'] = 'continuous'
        else:
            data_analysis['target_type'] = 'mixed'
    
    has_ordinal = 'ordinal' in data_analysis['feature_types']
    has_binary = 'binary' in data_analysis['feature_types']
    target_type = data_analysis['target_type']
    
    pearson_score = 0
    spearman_score = 0
    
    if target_type == 'continuous':
        pearson_score += 2
    else:
        spearman_score += 2
        data_analysis['recommendation_reason'].append('Target is not strictly continuous')
    
    if has_ordinal or has_binary:
        spearman_score += 1
        data_analysis['recommendation_reason'].append('Contains ordinal/binary features')
    
    if pearson_score > spearman_score:
        data_analysis['recommended_method'] = 'pearson'
        data_analysis['recommendation_reason'].insert(0, 'Linear relationship assumption holds')
    else:
        data_analysis['recommended_method'] = 'spearman'
        data_analysis['recommendation_reason'].insert(0, 'Better for non-linear monotonic relationships')
    
    type_counts = {}
    for ft in data_analysis['feature_types']:
        type_counts[ft] = type_counts.get(ft, 0) + 1
    data_analysis['feature_type_counts'] = type_counts
    
    return data_analysis

def compute_correlation(feature, target, method='pearson'):
    try:
        if method == 'pearson':
            corr, p_value = pearsonr(feature, target)
        elif method == 'spearman':
            corr, p_value = spearmanr(feature, target, nan_policy='omit')
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if np.isnan(corr):
            return 0.0, 1.0, 'Correlation resulted in NaN (constant values?)'
        return float(corr), float(p_value) if not np.isnan(p_value) else 1.0, None
    except Exception as e:
        return 0.0, 1.0, f'Failed to compute correlation: {str(e)}'

@app.route('/feature-selection', methods=['POST'])
def feature_selection():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        feature_matrix = data.get('feature_matrix')
        target_vector = data.get('target_vector')
        k = data.get('k', 5)
        strategy = data.get('strategy', 'delete')
        constant_value = data.get('constant_value', 0)
        method = data.get('method', 'auto')
        
        if feature_matrix is None or target_vector is None:
            return jsonify({'error': 'feature_matrix and target_vector are required'}), 400
        
        if strategy not in VALID_STRATEGIES:
            return jsonify({
                'error': f'Invalid strategy. Must be one of: {VALID_STRATEGIES}'
            }), 400
        
        if method not in VALID_METHODS:
            return jsonify({
                'error': f'Invalid method. Must be one of: {VALID_METHODS}'
            }), 400
        
        X = np.array(feature_matrix, dtype=float)
        y = np.array(target_vector, dtype=float)
        
        if X.ndim != 2:
            return jsonify({'error': 'feature_matrix must be 2-dimensional'}), 400
        
        if y.ndim != 1:
            return jsonify({'error': 'target_vector must be 1-dimensional'}), 400
        
        n_samples, n_features = X.shape
        
        if n_samples != len(y):
            return jsonify({'error': 'Number of samples in feature_matrix and target_vector must match'}), 400
        
        data_analysis = analyze_data_type(X, y)
        
        actual_method = method if method != 'auto' else data_analysis['recommended_method']
        
        filling_summary = {
            'feature_matrix': fill_missing_values(X, strategy, constant_value),
            'target_vector': fill_missing_values(y, strategy, constant_value)
        }
        
        X_filled = filling_summary['feature_matrix']['filled_data']
        y_filled = filling_summary['target_vector']['filled_data']
        
        correlations = []
        for i in range(n_features):
            feature_col = X_filled[:, i]
            target_col = y_filled
            
            if strategy == 'delete':
                valid_mask = ~np.isnan(feature_col) & ~np.isnan(target_col)
                valid_feature = feature_col[valid_mask]
                valid_target = target_col[valid_mask]
                n_valid = len(valid_feature)
            else:
                valid_feature = feature_col
                valid_target = target_col
                n_valid = n_samples
            
            if n_valid < 2:
                corr = 0.0
                p_value = 1.0
                warning = f'Insufficient valid samples (only {n_valid})'
            else:
                corr, p_value, warning = compute_correlation(valid_feature, valid_target, actual_method)
            
            result = {
                'feature_index': i,
                'feature_type': data_analysis['feature_types'][i],
                'correlation': corr,
                'abs_correlation': abs(corr),
                'p_value': p_value,
                'valid_samples': n_valid
            }
            if warning:
                result['warning'] = warning
            correlations.append(result)
        
        correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        top_k = correlations[:k]
        
        for key in ['feature_matrix', 'target_vector']:
            if 'filled_data' in filling_summary[key]:
                del filling_summary[key]['filled_data']
        
        return jsonify({
            'success': True,
            'total_features': n_features,
            'k': k,
            'method_used': actual_method,
            'method_requested': method,
            'filling_strategy_used': strategy,
            'data_analysis': data_analysis,
            'filling_summary': filling_summary,
            'method_comparison': {
                'pearson': 'Measures linear correlation (good for continuous normally distributed data)',
                'spearman': 'Measures monotonic correlation using ranks (robust to outliers, works for ordinal data)'
            },
            'top_k_features': top_k
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
