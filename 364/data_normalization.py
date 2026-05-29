import warnings

import numpy as np


def z_score_normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    safe_std = np.where(std == 0, 1.0, std)
    normalized_data = (data - mean) / safe_std
    normalized_data = np.where(std == 0, 0.0, normalized_data)
    return normalized_data, {'mean': mean, 'std': std}


def min_max_normalize(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    range_val = max_val - min_val
    constant_mask = range_val == 0
    if np.any(constant_mask):
        warnings.warn(
            'Constant feature detected (max == min). '
            'Min-Max normalization is undefined for constant features. '
            'Consider using Z-score normalization instead.',
            stacklevel=2,
        )
        safe_range = np.where(constant_mask, 1.0, range_val)
        normalized_data = (data - min_val) / safe_range
        normalized_data = np.where(constant_mask, 0.0, normalized_data)
    else:
        normalized_data = (data - min_val) / range_val
    return normalized_data, {'min': min_val, 'max': max_val}


def robust_normalize(data):
    median = np.median(data, axis=0)
    q1 = np.percentile(data, 25, axis=0)
    q3 = np.percentile(data, 75, axis=0)
    iqr = q3 - q1
    safe_iqr = np.where(iqr == 0, 1.0, iqr)
    normalized_data = (data - median) / safe_iqr
    normalized_data = np.where(iqr == 0, 0.0, normalized_data)
    return normalized_data, {'median': median, 'iqr': iqr}


def l2_normalize(data):
    norms = np.linalg.norm(data, axis=0)
    safe_norms = np.where(norms == 0, 1.0, norms)
    normalized_data = data / safe_norms
    normalized_data = np.where(norms == 0, 0.0, normalized_data)
    return normalized_data, {'l2_norm': norms}


if __name__ == '__main__':
    data = np.array([[1, 2], [3, 4], [5, 6]])

    print('Original data:')
    print(data)
    print()

    z_data, z_params = z_score_normalize(data)
    print('Z-score normalized:')
    print(z_data)
    print('Parameters - mean:', z_params['mean'], 'std:', z_params['std'])
    print()

    mm_data, mm_params = min_max_normalize(data)
    print('Min-Max normalized:')
    print(mm_data)
    print('Parameters - min:', mm_params['min'], 'max:', mm_params['max'])
    print()

    rb_data, rb_params = robust_normalize(data)
    print('Robust normalized:')
    print(rb_data)
    print('Parameters - median:', rb_params['median'], 'iqr:', rb_params['iqr'])
    print()

    l2_data, l2_params = l2_normalize(data)
    print('L2 normalized:')
    print(l2_data)
    print('Parameters - l2_norm:', l2_params['l2_norm'])
    print()

    constant_data = np.array([[3, 2], [3, 4], [3, 6]])
    print('Constant feature data:')
    print(constant_data)
    print()

    z_data2, z_params2 = z_score_normalize(constant_data)
    print('Z-score normalized (constant feature):')
    print(z_data2)
    print('Parameters - mean:', z_params2['mean'], 'std:', z_params2['std'])
    print()

    mm_data2, mm_params2 = min_max_normalize(constant_data)
    print('Min-Max normalized (constant feature):')
    print(mm_data2)
    print('Parameters - min:', mm_params2['min'], 'max:', mm_params2['max'])
    print()

    print('=' * 60)
    print('Outlier Sensitivity Comparison')
    print('=' * 60)
    clean = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    with_outlier = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])

    print(f'Clean data:      {clean.flatten()}')
    print(f'With outlier:    {with_outlier.flatten()}')
    print()

    methods = [
        ('Z-score', z_score_normalize),
        ('Min-Max', min_max_normalize),
        ('Robust', robust_normalize),
        ('L2', l2_normalize),
    ]

    print(f'{"Method":<10} {"Clean":>30} {"With Outlier":>30} {"Max Shift":>12}')
    print('-' * 84)

    for name, func in methods:
        c_result, _ = func(clean)
        o_result, _ = func(with_outlier)
        shift = np.max(np.abs(o_result[:4] - c_result[:4]))
        print(f'{name:<10} {np.array2string(c_result.flatten(), precision=4):>30} {np.array2string(o_result.flatten(), precision=4):>30} {shift:>12.4f}')
