import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Optional
from utils import normalize_data


def cosine_similarity_1d(x: np.ndarray, y: np.ndarray) -> float:
    """计算两个1D向量的余弦相似度"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    
    if norm_x < 1e-10 or norm_y < 1e-10:
        return 0.0
    
    return np.dot(x, y) / (norm_x * norm_y)


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """计算皮尔逊相关系数"""
    if len(x) < 2 or len(y) < 2:
        return 0.0
    
    std_x = np.std(x)
    std_y = np.std(y)
    
    if std_x < 1e-10 or std_y < 1e-10:
        return 0.0
    
    corr, _ = pearsonr(x, y)
    return corr if not np.isnan(corr) else 0.0


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """计算斯皮尔曼相关系数"""
    if len(x) < 2 or len(y) < 2:
        return 0.0
    
    std_x = np.std(x)
    std_y = np.std(y)
    
    if std_x < 1e-10 or std_y < 1e-10:
        return 0.0
    
    corr, _ = spearmanr(x, y)
    return corr if not np.isnan(corr) else 0.0


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """计算欧氏距离（转换为相似度）"""
    dist = np.linalg.norm(x - y)
    return 1.0 / (1.0 + dist)


def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """计算曼哈顿距离（转换为相似度）"""
    dist = np.sum(np.abs(x - y))
    return 1.0 / (1.0 + dist)


def compute_similarity(x: np.ndarray, y: np.ndarray,
                       method: str = 'cosine') -> float:
    """
    计算两个光谱的相似度
    
    Args:
        x: 光谱1
        y: 光谱2
        method: 相似度方法，'cosine', 'pearson', 'spearman', 'euclidean', 'manhattan'
    
    Returns:
        相似度值，范围[0, 1]
    """
    if len(x) != len(y):
        raise ValueError(f"Length mismatch: {len(x)} vs {len(y)}")
    
    if method == 'cosine':
        sim = cosine_similarity_1d(x, y)
    elif method == 'pearson':
        sim = pearson_correlation(x, y)
    elif method == 'spearman':
        sim = spearman_correlation(x, y)
    elif method == 'euclidean':
        sim = euclidean_distance(x, y)
    elif method == 'manhattan':
        sim = manhattan_distance(x, y)
    else:
        raise ValueError(f"Unknown similarity method: {method}")
    
    return max(0.0, min(1.0, sim))


def match_spectrum(unknown_intensities: np.ndarray,
                   reference_intensities_list: List[np.ndarray],
                   reference_names: List[str],
                   method: str = 'cosine',
                   normalize: bool = True,
                   normalize_method: str = 'minmax',
                   top_k: int = 5) -> List[Tuple[str, float]]:
    """
    将未知光谱与参考光谱库匹配
    
    Args:
        unknown_intensities: 未知光谱强度
        reference_intensities_list: 参考光谱强度列表
        reference_names: 参考光谱名称列表
        method: 相似度方法
        normalize: 是否归一化
        normalize_method: 归一化方法
        top_k: 返回前K个匹配结果
    
    Returns:
        [(名称, 相似度)] 按相似度降序排列
    """
    if len(reference_intensities_list) != len(reference_names):
        raise ValueError("Reference intensities and names length mismatch")
    
    unknown = unknown_intensities.copy()
    if normalize:
        unknown = normalize_data(unknown, normalize_method)
    
    similarities = []
    for i, ref_intensities in enumerate(reference_intensities_list):
        ref = ref_intensities.copy()
        if normalize:
            ref = normalize_data(ref, normalize_method)
        
        sim = compute_similarity(unknown, ref, method)
        similarities.append((reference_names[i], sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


def peak_based_match(unknown_peak_vector: np.ndarray,
                     reference_peak_vectors: List[np.ndarray],
                     reference_names: List[str],
                     method: str = 'cosine',
                     top_k: int = 5) -> List[Tuple[str, float]]:
    """
    基于峰向量的光谱匹配
    
    Args:
        unknown_peak_vector: 未知光谱的峰向量
        reference_peak_vectors: 参考光谱的峰向量列表
        reference_names: 参考光谱名称列表
        method: 相似度方法
        top_k: 返回前K个匹配结果
    
    Returns:
        [(名称, 相似度)] 按相似度降序排列
    """
    return match_spectrum(
        unknown_peak_vector,
        reference_peak_vectors,
        reference_names,
        method=method,
        normalize=True,
        normalize_method='l2',
        top_k=top_k
    )


def combined_match(unknown_intensities: np.ndarray,
                   unknown_peak_vector: np.ndarray,
                   reference_intensities_list: List[np.ndarray],
                   reference_peak_vectors: List[np.ndarray],
                   reference_names: List[str],
                   intensity_weight: float = 0.5,
                   peak_weight: float = 0.5,
                   method: str = 'cosine',
                   top_k: int = 5) -> List[Tuple[str, float]]:
    """
    组合匹配：结合全谱匹配和峰匹配
    
    Args:
        unknown_intensities: 未知光谱强度
        unknown_peak_vector: 未知光谱的峰向量
        reference_intensities_list: 参考光谱强度列表
        reference_peak_vectors: 参考光谱的峰向量列表
        reference_names: 参考光谱名称列表
        intensity_weight: 全谱匹配权重
        peak_weight: 峰匹配权重
        method: 相似度方法
        top_k: 返回前K个匹配结果
    
    Returns:
        [(名称, 综合相似度)] 按相似度降序排列
    """
    intensity_matches = match_spectrum(
        unknown_intensities,
        reference_intensities_list,
        reference_names,
        method=method,
        top_k=len(reference_names)
    )
    
    peak_matches = peak_based_match(
        unknown_peak_vector,
        reference_peak_vectors,
        reference_names,
        method=method,
        top_k=len(reference_names)
    )
    
    intensity_scores = {name: score for name, score in intensity_matches}
    peak_scores = {name: score for name, score in peak_matches}
    
    combined_scores = []
    for name in reference_names:
        int_score = intensity_scores.get(name, 0.0)
        peak_score = peak_scores.get(name, 0.0)
        combined = intensity_weight * int_score + peak_weight * peak_score
        combined_scores.append((name, combined))
    
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    
    return combined_scores[:top_k]


def identify_material(matches: List[Tuple[str, float]],
                      threshold: float = 0.7) -> Tuple[Optional[str], float, bool]:
    """
    根据匹配结果识别物质
    
    Args:
        matches: 匹配结果列表
        threshold: 识别阈值
    
    Returns:
        (识别的物质名称, 最高相似度, 是否可靠)
    """
    if not matches:
        return None, 0.0, False
    
    best_name, best_score = matches[0]
    
    if len(matches) >= 2:
        second_score = matches[1][1]
        margin = best_score - second_score
    else:
        margin = best_score
    
    reliable = best_score >= threshold and margin >= 0.1
    
    return best_name, best_score, reliable
