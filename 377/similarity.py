import math
import warnings
from collections import defaultdict


def is_sparse(vec):
    return isinstance(vec, dict)


def to_dense(vec, length=None):
    if isinstance(vec, list):
        return vec
    if isinstance(vec, dict):
        if length is None:
            if not vec:
                return []
            length = max(vec.keys()) + 1
        dense = [0.0] * length
        for idx, val in vec.items():
            dense[idx] = val
        return dense
    raise TypeError("向量必须是list（稠密）或dict（稀疏）类型")


def to_sparse(vec):
    if isinstance(vec, dict):
        return {k: float(v) for k, v in vec.items() if v != 0}
    if isinstance(vec, list):
        return {i: float(v) for i, v in enumerate(vec) if v != 0}
    raise TypeError("向量必须是list（稠密）或dict（稀疏）类型")


def sparse_mean(vec):
    if isinstance(vec, dict):
        if not vec:
            return 0.0
        return sum(vec.values()) / len(vec)
    if isinstance(vec, list):
        non_zero = [v for v in vec if v != 0]
        if not non_zero:
            return 0.0
        return sum(non_zero) / len(non_zero)
    raise TypeError("向量必须是list（稠密）或dict（稀疏）类型")


def sparse_overlap(vec1, vec2):
    if is_sparse(vec1) and is_sparse(vec2):
        return sorted(set(vec1.keys()) & set(vec2.keys()))
    dense1 = to_dense(vec1)
    dense2 = to_dense(vec2)
    min_len = min(len(dense1), len(dense2))
    return [i for i in range(min_len) if dense1[i] != 0 and dense2[i] != 0]


def get_value(vec, idx):
    if isinstance(vec, dict):
        return vec.get(idx, 0.0)
    if isinstance(vec, list):
        return vec[idx] if idx < len(vec) else 0.0
    raise TypeError("向量必须是list（稠密）或dict（稀疏）类型")


def cosine_similarity(vec1, vec2, default=0.0, return_details=False):
    if is_sparse(vec1) and is_sparse(vec2):
        return _cosine_similarity_sparse(vec1, vec2, default, return_details)
    return _cosine_similarity_dense(vec1, vec2, default, return_details)


def _cosine_similarity_dense(vec1, vec2, default=0.0, return_details=False):
    if len(vec1) != len(vec2):
        raise ValueError("两个向量长度必须相同")
    
    details = {
        'method': 'cosine_similarity',
        'vector_type': 'dense',
        'input_vec1': list(vec1),
        'input_vec2': list(vec2),
        'steps': []
    }
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    
    details['steps'].append({'name': '点积计算', 'value': dot_product, 'formula': 'sum(a*b for a,b in zip(vec1,vec2))'})
    details['steps'].append({'name': 'vec1模长', 'value': norm1, 'formula': 'sqrt(sum(a² for a in vec1))'})
    details['steps'].append({'name': 'vec2模长', 'value': norm2, 'formula': 'sqrt(sum(b² for b in vec2))'})
    
    if norm1 == 0 or norm2 == 0:
        zero_vecs = []
        if norm1 == 0:
            zero_vecs.append("vec1")
        if norm2 == 0:
            zero_vecs.append("vec2")
        warnings.warn(
            f"余弦相似度计算警告: {'和'.join(zero_vecs)} 为全零向量，"
            f"无法计算方向相似性，返回默认值 {default}",
            UserWarning
        )
        details['result'] = default
        details['warning'] = f"{'和'.join(zero_vecs)}为全零向量"
        return (default, details) if return_details else default
    
    result = dot_product / (norm1 * norm2)
    details['steps'].append({'name': '相似度计算', 'value': result, 'formula': '点积 / (vec1模长 × vec2模长)'})
    details['result'] = result
    
    return (result, details) if return_details else result


def _cosine_similarity_sparse(vec1, vec2, default=0.0, return_details=False):
    s1 = to_sparse(vec1)
    s2 = to_sparse(vec2)
    
    details = {
        'method': 'cosine_similarity',
        'vector_type': 'sparse',
        'input_vec1': dict(s1),
        'input_vec2': dict(s2),
        'steps': []
    }
    
    common_indices = sorted(set(s1.keys()) & set(s2.keys()))
    dot_product = sum(s1[i] * s2[i] for i in common_indices)
    norm1 = math.sqrt(sum(v * v for v in s1.values()))
    norm2 = math.sqrt(sum(v * v for v in s2.values()))
    
    details['steps'].append({'name': '公共索引', 'value': common_indices, 'formula': 'set(s1.keys()) & set(s2.keys())'})
    details['steps'].append({'name': '点积计算', 'value': dot_product, 'formula': 'sum(s1[i]*s2[i] for i in common_indices)'})
    details['steps'].append({'name': 'vec1模长', 'value': norm1, 'formula': 'sqrt(sum(v² for v in s1.values()))'})
    details['steps'].append({'name': 'vec2模长', 'value': norm2, 'formula': 'sqrt(sum(v² for v in s2.values()))'})
    
    if norm1 == 0 or norm2 == 0:
        zero_vecs = []
        if norm1 == 0:
            zero_vecs.append("vec1")
        if norm2 == 0:
            zero_vecs.append("vec2")
        warnings.warn(
            f"余弦相似度计算警告: {'和'.join(zero_vecs)} 为全零向量，"
            f"无法计算方向相似性，返回默认值 {default}",
            UserWarning
        )
        details['result'] = default
        details['warning'] = f"{'和'.join(zero_vecs)}为全零向量"
        return (default, details) if return_details else default
    
    result = dot_product / (norm1 * norm2)
    details['steps'].append({'name': '相似度计算', 'value': result, 'formula': '点积 / (vec1模长 × vec2模长)'})
    details['result'] = result
    
    return (result, details) if return_details else result


def adjusted_cosine_similarity(vec1, vec2, default=0.0, return_details=False, mean1=None, mean2=None):
    if is_sparse(vec1) and is_sparse(vec2):
        return _adjusted_cosine_sparse(vec1, vec2, default, return_details, mean1, mean2)
    return _adjusted_cosine_dense(vec1, vec2, default, return_details, mean1, mean2)


def _adjusted_cosine_dense(vec1, vec2, default=0.0, return_details=False, mean1=None, mean2=None):
    if len(vec1) != len(vec2):
        raise ValueError("两个向量长度必须相同")
    
    n = len(vec1)
    if n == 0:
        return (default, {'result': default, 'warning': '空向量'}) if return_details else default
    
    if mean1 is None:
        mean1 = sum(vec1) / n
    if mean2 is None:
        mean2 = sum(vec2) / n
    
    details = {
        'method': 'adjusted_cosine_similarity',
        'vector_type': 'dense',
        'input_vec1': list(vec1),
        'input_vec2': list(vec2),
        'mean1': mean1,
        'mean2': mean2,
        'steps': []
    }
    
    details['steps'].append({'name': 'vec1均值', 'value': mean1, 'formula': 'sum(vec1) / len(vec1)'})
    details['steps'].append({'name': 'vec2均值', 'value': mean2, 'formula': 'sum(vec2) / len(vec2)'})
    
    centered1 = [a - mean1 for a in vec1]
    centered2 = [b - mean2 for b in vec2]
    
    details['steps'].append({'name': 'vec1中心化', 'value': centered1, 'formula': '[a - mean1 for a in vec1]'})
    details['steps'].append({'name': 'vec2中心化', 'value': centered2, 'formula': '[b - mean2 for b in vec2]'})
    
    dot_product = sum(a * b for a, b in zip(centered1, centered2))
    norm1 = math.sqrt(sum(a * a for a in centered1))
    norm2 = math.sqrt(sum(b * b for b in centered2))
    
    details['steps'].append({'name': '中心化点积', 'value': dot_product, 'formula': 'sum(a*b for a,b in zip(centered1,centered2))'})
    details['steps'].append({'name': '中心化vec1模长', 'value': norm1, 'formula': 'sqrt(sum(a² for a in centered1))'})
    details['steps'].append({'name': '中心化vec2模长', 'value': norm2, 'formula': 'sqrt(sum(b² for b in centered2))'})
    
    if norm1 == 0 or norm2 == 0:
        zero_vecs = []
        if norm1 == 0:
            zero_vecs.append("vec1")
        if norm2 == 0:
            zero_vecs.append("vec2")
        warnings.warn(
            f"调整余弦相似度计算警告: {'和'.join(zero_vecs)} 中心化后方差为零，"
            f"无法计算方向相似性，返回默认值 {default}",
            UserWarning
        )
        details['result'] = default
        details['warning'] = f"{'和'.join(zero_vecs)}中心化后方差为零"
        return (default, details) if return_details else default
    
    result = dot_product / (norm1 * norm2)
    details['steps'].append({'name': '相似度计算', 'value': result, 'formula': '中心化点积 / (中心化vec1模长 × 中心化vec2模长)'})
    details['result'] = result
    
    return (result, details) if return_details else result


def _adjusted_cosine_sparse(vec1, vec2, default=0.0, return_details=False, mean1=None, mean2=None):
    s1 = to_sparse(vec1)
    s2 = to_sparse(vec2)
    
    if mean1 is None:
        mean1 = sparse_mean(s1) if s1 else 0.0
    if mean2 is None:
        mean2 = sparse_mean(s2) if s2 else 0.0
    
    details = {
        'method': 'adjusted_cosine_similarity',
        'vector_type': 'sparse',
        'input_vec1': dict(s1),
        'input_vec2': dict(s2),
        'mean1': mean1,
        'mean2': mean2,
        'steps': []
    }
    
    details['steps'].append({'name': 'vec1均值（仅非零元素）', 'value': mean1, 'formula': 'sum(s1.values()) / len(s1)'})
    details['steps'].append({'name': 'vec2均值（仅非零元素）', 'value': mean2, 'formula': 'sum(s2.values()) / len(s2)'})
    
    common_indices = sorted(set(s1.keys()) & set(s2.keys()))
    all_indices = sorted(set(s1.keys()) | set(s2.keys()))
    
    details['steps'].append({'name': '公共评分索引', 'value': common_indices, 'formula': 'set(s1.keys()) & set(s2.keys())'})
    
    centered_product = sum((s1[i] - mean1) * (s2[i] - mean2) for i in common_indices)
    
    s1_norm_sq = sum((v - mean1) ** 2 for v in s1.values()) + (len(all_indices) - len(s1)) * (mean1 ** 2)
    s2_norm_sq = sum((v - mean2) ** 2 for v in s2.values()) + (len(all_indices) - len(s2)) * (mean2 ** 2)
    
    norm1 = math.sqrt(s1_norm_sq)
    norm2 = math.sqrt(s2_norm_sq)
    
    details['steps'].append({'name': '中心化点积（公共项）', 'value': centered_product, 'formula': 'sum((s1[i]-mean1)*(s2[i]-mean2) for i in common_indices)'})
    details['steps'].append({'name': '中心化vec1模长平方', 'value': s1_norm_sq, 'formula': 'sum((v-mean1)² for v in s1.values()) + (zero_count) * mean1²'})
    details['steps'].append({'name': '中心化vec2模长平方', 'value': s2_norm_sq, 'formula': 'sum((v-mean2)² for v in s2.values()) + (zero_count) * mean2²'})
    details['steps'].append({'name': '中心化vec1模长', 'value': norm1, 'formula': 'sqrt(s1_norm_sq)'})
    details['steps'].append({'name': '中心化vec2模长', 'value': norm2, 'formula': 'sqrt(s2_norm_sq)'})
    
    if norm1 == 0 or norm2 == 0:
        zero_vecs = []
        if norm1 == 0:
            zero_vecs.append("vec1")
        if norm2 == 0:
            zero_vecs.append("vec2")
        warnings.warn(
            f"调整余弦相似度计算警告: {'和'.join(zero_vecs)} 中心化后方差为零，"
            f"无法计算方向相似性，返回默认值 {default}",
            UserWarning
        )
        details['result'] = default
        details['warning'] = f"{'和'.join(zero_vecs)}中心化后方差为零"
        return (default, details) if return_details else default
    
    result = centered_product / (norm1 * norm2)
    details['steps'].append({'name': '相似度计算', 'value': result, 'formula': '中心化点积 / (中心化vec1模长 × 中心化vec2模长)'})
    details['result'] = result
    
    return (result, details) if return_details else result


def pearson_correlation(vec1, vec2, default=0.0, return_details=False):
    if is_sparse(vec1) and is_sparse(vec2):
        return _pearson_sparse(vec1, vec2, default, return_details)
    return _pearson_dense(vec1, vec2, default, return_details)


def _pearson_dense(vec1, vec2, default=0.0, return_details=False):
    if len(vec1) != len(vec2):
        raise ValueError("两个向量长度必须相同")
    
    n = len(vec1)
    if n == 0:
        return (default, {'result': default, 'warning': '空向量'}) if return_details else default
    
    mean1 = sum(vec1) / n
    mean2 = sum(vec2) / n
    
    details = {
        'method': 'pearson_correlation',
        'vector_type': 'dense',
        'input_vec1': list(vec1),
        'input_vec2': list(vec2),
        'mean1': mean1,
        'mean2': mean2,
        'steps': []
    }
    
    details['steps'].append({'name': 'vec1均值', 'value': mean1, 'formula': 'sum(vec1) / len(vec1)'})
    details['steps'].append({'name': 'vec2均值', 'value': mean2, 'formula': 'sum(vec2) / len(vec2)'})
    
    covariance = sum((a - mean1) * (b - mean2) for a, b in zip(vec1, vec2))
    variance1 = sum((a - mean1) ** 2 for a in vec1)
    variance2 = sum((b - mean2) ** 2 for b in vec2)
    
    details['steps'].append({'name': '协方差', 'value': covariance, 'formula': 'sum((a-mean1)*(b-mean2) for a,b in zip(vec1,vec2))'})
    details['steps'].append({'name': 'vec1方差', 'value': variance1, 'formula': 'sum((a-mean1)² for a in vec1)'})
    details['steps'].append({'name': 'vec2方差', 'value': variance2, 'formula': 'sum((b-mean2)² for b in vec2)'})
    
    if variance1 == 0 or variance2 == 0:
        zero_var_vecs = []
        if variance1 == 0:
            zero_var_vecs.append("vec1")
        if variance2 == 0:
            zero_var_vecs.append("vec2")
        warnings.warn(
            f"皮尔逊相关系数计算警告: {'和'.join(zero_var_vecs)} 方差为零"
            f"（所有元素相同），无法计算相关性，返回默认值 {default}",
            UserWarning
        )
        details['result'] = default
        details['warning'] = f"{'和'.join(zero_var_vecs)}方差为零"
        return (default, details) if return_details else default
    
    result = covariance / math.sqrt(variance1 * variance2)
    details['steps'].append({'name': '相关系数计算', 'value': result, 'formula': '协方差 / sqrt(vec1方差 × vec2方差)'})
    details['result'] = result
    
    return (result, details) if return_details else result


def _pearson_sparse(vec1, vec2, default=0.0, return_details=False):
    s1 = to_sparse(vec1)
    s2 = to_sparse(vec2)
    
    common_indices = sorted(set(s1.keys()) & set(s2.keys()))
    
    if not common_indices:
        warnings.warn(
            f"皮尔逊相关系数计算警告: 两个稀疏向量没有公共非零元素，"
            f"无法计算相关性，返回默认值 {default}",
            UserWarning
        )
        details = {
            'method': 'pearson_correlation',
            'vector_type': 'sparse',
            'input_vec1': dict(s1),
            'input_vec2': dict(s2),
            'result': default,
            'warning': '无公共非零元素'
        }
        return (default, details) if return_details else default
    
    n = len(common_indices)
    values1 = [s1[i] for i in common_indices]
    values2 = [s2[i] for i in common_indices]
    
    mean1 = sum(values1) / n
    mean2 = sum(values2) / n
    
    details = {
        'method': 'pearson_correlation',
        'vector_type': 'sparse',
        'input_vec1': dict(s1),
        'input_vec2': dict(s2),
        'common_indices': common_indices,
        'mean1': mean1,
        'mean2': mean2,
        'steps': []
    }
    
    details['steps'].append({'name': '公共非零索引', 'value': common_indices, 'formula': 'set(s1.keys()) & set(s2.keys())'})
    details['steps'].append({'name': 'vec1均值（公共项）', 'value': mean1, 'formula': 'sum(values1) / n'})
    details['steps'].append({'name': 'vec2均值（公共项）', 'value': mean2, 'formula': 'sum(values2) / n'})
    
    covariance = sum((a - mean1) * (b - mean2) for a, b in zip(values1, values2))
    variance1 = sum((a - mean1) ** 2 for a in values1)
    variance2 = sum((b - mean2) ** 2 for b in values2)
    
    details['steps'].append({'name': '协方差（公共项）', 'value': covariance, 'formula': 'sum((a-mean1)*(b-mean2))'})
    details['steps'].append({'name': 'vec1方差（公共项）', 'value': variance1, 'formula': 'sum((a-mean1)²)'})
    details['steps'].append({'name': 'vec2方差（公共项）', 'value': variance2, 'formula': 'sum((b-mean2)²)'})
    
    if variance1 == 0 or variance2 == 0:
        zero_var_vecs = []
        if variance1 == 0:
            zero_var_vecs.append("vec1")
        if variance2 == 0:
            zero_var_vecs.append("vec2")
        warnings.warn(
            f"皮尔逊相关系数计算警告: {'和'.join(zero_var_vecs)} 公共项方差为零，"
            f"无法计算相关性，返回默认值 {default}",
            UserWarning
        )
        details['result'] = default
        details['warning'] = f"{'和'.join(zero_var_vecs)}公共项方差为零"
        return (default, details) if return_details else default
    
    result = covariance / math.sqrt(variance1 * variance2)
    details['steps'].append({'name': '相关系数计算', 'value': result, 'formula': '协方差 / sqrt(vec1方差 × vec2方差)'})
    details['result'] = result
    
    return (result, details) if return_details else result


def print_details(details, indent=0):
    prefix = "  " * indent
    print(f"{prefix}方法: {details.get('method', 'N/A')}")
    print(f"{prefix}向量类型: {details.get('vector_type', 'N/A')}")
    print(f"{prefix}输入vec1: {details.get('input_vec1', 'N/A')}")
    print(f"{prefix}输入vec2: {details.get('input_vec2', 'N/A')}")
    
    if 'mean1' in details:
        print(f"{prefix}vec1均值: {details['mean1']:.4f}")
    if 'mean2' in details:
        print(f"{prefix}vec2均值: {details['mean2']:.4f}")
    
    print(f"\n{prefix}计算步骤:")
    for i, step in enumerate(details.get('steps', []), 1):
        print(f"{prefix}  {i}. {step['name']}")
        print(f"{prefix}     公式: {step['formula']}")
        val = step['value']
        if isinstance(val, (int, float)):
            print(f"{prefix}     结果: {val:.6f}")
        else:
            print(f"{prefix}     结果: {val}")
    
    if 'warning' in details:
        print(f"\n{prefix}警告: {details['warning']}")
    
    result = details.get('result', 'N/A')
    if isinstance(result, (int, float)):
        print(f"\n{prefix}最终结果: {result:.6f}")
    else:
        print(f"\n{prefix}最终结果: {result}")


if __name__ == "__main__":
    warnings.filterwarnings("always")
    
    print("=" * 70)
    print("测试1: 稠密向量 - 基础功能")
    print("=" * 70)
    
    dense_test_cases = [
        ([1, 2, 3, 4, 5], [2, 4, 6, 8, 10], "完全正相关"),
        ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], "完全负相关"),
    ]
    
    for vec1, vec2, desc in dense_test_cases:
        print(f"\n场景: {desc}")
        cos_sim, cos_det = cosine_similarity(vec1, vec2, return_details=True)
        adj_sim, adj_det = adjusted_cosine_similarity(vec1, vec2, return_details=True)
        pearson, pea_det = pearson_correlation(vec1, vec2, return_details=True)
        
        print(f"  余弦相似度: {cos_sim:.4f}")
        print(f"  调整余弦: {adj_sim:.4f}")
        print(f"  皮尔逊系数: {pearson:.4f}")
    
    print("\n" + "=" * 70)
    print("测试2: 稀疏向量 - 推荐系统场景（用户评分）")
    print("=" * 70)
    
    user1 = {0: 5.0, 1: 3.0, 3: 4.0, 5: 2.0}
    user2 = {0: 4.5, 1: 2.5, 2: 5.0, 3: 3.5}
    user3 = {6: 1.0, 7: 5.0, 8: 2.0}
    
    print(f"\n用户1评分（稀疏）: {user1}")
    print(f"用户2评分（稀疏）: {user2}")
    print(f"用户3评分（稀疏）: {user3}")
    
    cos_sim, cos_det = cosine_similarity(user1, user2, return_details=True)
    adj_sim, adj_det = adjusted_cosine_similarity(user1, user2, return_details=True)
    pearson, pea_det = pearson_correlation(user1, user2, return_details=True)
    
    print(f"\n用户1 vs 用户2:")
    print(f"  余弦相似度: {cos_sim:.4f}")
    print(f"  调整余弦: {adj_sim:.4f}")
    print(f"  皮尔逊系数: {pearson:.4f}")
    
    cos_sim2, _ = cosine_similarity(user1, user3, return_details=True)
    print(f"\n用户1 vs 用户3（无公共项）:")
    print(f"  余弦相似度: {cos_sim2:.4f}")
    
    print("\n" + "=" * 70)
    print("测试3: 详细步骤输出")
    print("=" * 70)
    
    vec_a = [4.0, 5.0, 3.0, 4.0]
    vec_b = [3.0, 4.0, 2.0, 3.0]
    print(f"\n向量A: {vec_a}")
    print(f"向量B: {vec_b}")
    print("\n--- 调整余弦相似度详细步骤 ---")
    _, det = adjusted_cosine_similarity(vec_a, vec_b, return_details=True)
    print_details(det)
    
    print("\n" + "=" * 70)
    print("测试4: 稀疏向量调整余弦详细步骤")
    print("=" * 70)
    
    s1 = {0: 5.0, 1: 3.0, 3: 4.0}
    s2 = {0: 4.0, 1: 5.0, 2: 2.0, 3: 3.0}
    print(f"\n稀疏向量1: {s1}")
    print(f"稀疏向量2: {s2}")
    print("\n--- 稀疏向量调整余弦相似度详细步骤 ---")
    _, det2 = adjusted_cosine_similarity(s1, s2, return_details=True)
    print_details(det2)
    
    print("\n" + "=" * 70)
    print("测试5: 边界情况 - 零向量和常向量")
    print("=" * 70)
    
    cos_zero, det_zero = cosine_similarity([0, 0, 0], [1, 2, 3], default=-1.0, return_details=True)
    print(f"\n零向量余弦相似度: {cos_zero}")
    print(f"警告信息: {det_zero.get('warning', '无')}")
    
    pearson_const, det_const = pearson_correlation([5, 5, 5], [1, 2, 3], return_details=True)
    print(f"\n常向量皮尔逊系数: {pearson_const}")
    print(f"警告信息: {det_const.get('warning', '无')}")
