import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

BLOSUM62 = {
    ('A', 'A'): 4, ('A', 'R'): -1, ('A', 'N'): -2, ('A', 'D'): -2, ('A', 'C'): 0,
    ('A', 'Q'): -1, ('A', 'E'): -1, ('A', 'G'): 0, ('A', 'H'): -2, ('A', 'I'): -1,
    ('A', 'L'): -1, ('A', 'K'): -1, ('A', 'M'): -1, ('A', 'F'): -2, ('A', 'P'): -1,
    ('A', 'S'): 1, ('A', 'T'): 0, ('A', 'W'): -3, ('A', 'Y'): -2, ('A', 'V'): 0,
    ('R', 'R'): 5, ('R', 'N'): 0, ('R', 'D'): -2, ('R', 'C'): -3, ('R', 'Q'): 1,
    ('R', 'E'): 0, ('R', 'G'): -2, ('R', 'H'): 0, ('R', 'I'): -3, ('R', 'L'): -2,
    ('R', 'K'): 2, ('R', 'M'): -1, ('R', 'F'): -3, ('R', 'P'): -2, ('R', 'S'): -1,
    ('R', 'T'): -1, ('R', 'W'): -3, ('R', 'Y'): -2, ('R', 'V'): -3,
    ('N', 'N'): 6, ('N', 'D'): 1, ('N', 'C'): -3, ('N', 'Q'): 0, ('N', 'E'): 0,
    ('N', 'G'): 0, ('N', 'H'): 1, ('N', 'I'): -3, ('N', 'L'): -3, ('N', 'K'): 0,
    ('N', 'M'): -2, ('N', 'F'): -3, ('N', 'P'): -2, ('N', 'S'): 1, ('N', 'T'): 0,
    ('N', 'W'): -4, ('N', 'Y'): -2, ('N', 'V'): -3,
    ('D', 'D'): 6, ('D', 'C'): -3, ('D', 'Q'): 0, ('D', 'E'): 2, ('D', 'G'): -1,
    ('D', 'H'): -1, ('D', 'I'): -3, ('D', 'L'): -4, ('D', 'K'): -1, ('D', 'M'): -3,
    ('D', 'F'): -3, ('D', 'P'): -1, ('D', 'S'): 0, ('D', 'T'): -1, ('D', 'W'): -4,
    ('D', 'Y'): -3, ('D', 'V'): -3,
    ('C', 'C'): 9, ('C', 'Q'): -3, ('C', 'E'): -4, ('C', 'G'): -3, ('C', 'H'): -3,
    ('C', 'I'): -1, ('C', 'L'): -1, ('C', 'K'): -3, ('C', 'M'): -1, ('C', 'F'): -2,
    ('C', 'P'): -3, ('C', 'S'): -1, ('C', 'T'): -1, ('C', 'W'): -2, ('C', 'Y'): -2,
    ('C', 'V'): -1,
    ('Q', 'Q'): 5, ('Q', 'E'): 2, ('Q', 'G'): -2, ('Q', 'H'): 0, ('Q', 'I'): -3,
    ('Q', 'L'): -2, ('Q', 'K'): 1, ('Q', 'M'): 0, ('Q', 'F'): -3, ('Q', 'P'): -1,
    ('Q', 'S'): 0, ('Q', 'T'): -1, ('Q', 'W'): -2, ('Q', 'Y'): -1, ('Q', 'V'): -2,
    ('E', 'E'): 5, ('E', 'G'): -2, ('E', 'H'): 0, ('E', 'I'): -3, ('E', 'L'): -3,
    ('E', 'K'): 1, ('E', 'M'): -2, ('E', 'F'): -3, ('E', 'P'): -1, ('E', 'S'): 0,
    ('E', 'T'): -1, ('E', 'W'): -3, ('E', 'Y'): -2, ('E', 'V'): -2,
    ('G', 'G'): 6, ('G', 'H'): -2, ('G', 'I'): -4, ('G', 'L'): -4, ('G', 'K'): -2,
    ('G', 'M'): -3, ('G', 'F'): -3, ('G', 'P'): -2, ('G', 'S'): 0, ('G', 'T'): -2,
    ('G', 'W'): -2, ('G', 'Y'): -3, ('G', 'V'): -3,
    ('H', 'H'): 8, ('H', 'I'): -3, ('H', 'L'): -3, ('H', 'K'): -1, ('H', 'M'): -2,
    ('H', 'F'): -1, ('H', 'P'): -2, ('H', 'S'): -1, ('H', 'T'): -2, ('H', 'W'): -2,
    ('H', 'Y'): 2, ('H', 'V'): -3,
    ('I', 'I'): 4, ('I', 'L'): 2, ('I', 'K'): -3, ('I', 'M'): 1, ('I', 'F'): 0,
    ('I', 'P'): -3, ('I', 'S'): -2, ('I', 'T'): -1, ('I', 'W'): -3, ('I', 'Y'): -1,
    ('I', 'V'): 3,
    ('L', 'L'): 4, ('L', 'K'): -2, ('L', 'M'): 2, ('L', 'F'): 0, ('L', 'P'): -3,
    ('L', 'S'): -2, ('L', 'T'): -1, ('L', 'W'): -2, ('L', 'Y'): -1, ('L', 'V'): 1,
    ('K', 'K'): 5, ('K', 'M'): -1, ('K', 'F'): -3, ('K', 'P'): -1, ('K', 'S'): 0,
    ('K', 'T'): -1, ('K', 'W'): -3, ('K', 'Y'): -2, ('K', 'V'): -2,
    ('M', 'M'): 5, ('M', 'F'): 0, ('M', 'P'): -2, ('M', 'S'): -1, ('M', 'T'): -1,
    ('M', 'W'): -1, ('M', 'Y'): -1, ('M', 'V'): 1,
    ('F', 'F'): 6, ('F', 'P'): -4, ('F', 'S'): -2, ('F', 'T'): -2, ('F', 'W'): 1,
    ('F', 'Y'): 3, ('F', 'V'): -1,
    ('P', 'P'): 7, ('P', 'S'): -1, ('P', 'T'): -1, ('P', 'W'): -4, ('P', 'Y'): -3,
    ('P', 'V'): -2,
    ('S', 'S'): 4, ('S', 'T'): 1, ('S', 'W'): -3, ('S', 'Y'): -2, ('S', 'V'): -2,
    ('T', 'T'): 5, ('T', 'W'): -2, ('T', 'Y'): -2, ('T', 'V'): 0,
    ('W', 'W'): 11, ('W', 'Y'): 2, ('W', 'V'): -3,
    ('Y', 'Y'): 7, ('Y', 'V'): -1,
    ('V', 'V'): 4,
}

for (a, b), score in list(BLOSUM62.items()):
    if a != b:
        BLOSUM62[(b, a)] = score


def get_score(a: str, b: str, matrix: Dict[Tuple[str, str], int] = None,
              match: int = 1, mismatch: int = -1) -> int:
    if matrix is not None:
        return matrix.get((a, b), matrix.get((b, a), mismatch))
    return match if a == b else mismatch


def needleman_wunsch(
    seq1: str,
    seq2: str,
    gap_penalty: int = -2,
    substitution_matrix: Dict[Tuple[str, str], int] = None,
    match: int = 1,
    mismatch: int = -1
) -> Tuple[str, str, int, np.ndarray]:
    """
    Needleman-Wunsch 全局比对算法（线性空位罚分）
    
    Args:
        seq1: 第一条序列
        seq2: 第二条序列
        gap_penalty: 空位罚分（应为负数）
        substitution_matrix: 替换矩阵（如BLOSUM62），若为None则使用简单匹配/不匹配
        match: 匹配得分（当substitution_matrix为None时使用）
        mismatch: 不匹配罚分（当substitution_matrix为None时使用）
    
    Returns:
        (aligned_seq1, aligned_seq2, score, dp_matrix)
    """
    n = len(seq1)
    m = len(seq2)
    
    dp = np.zeros((n + 1, m + 1), dtype=int)
    
    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][0] + gap_penalty
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j-1] + gap_penalty
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_score = dp[i-1][j-1] + get_score(seq1[i-1], seq2[j-1], substitution_matrix, match, mismatch)
            gap_seq1 = dp[i-1][j] + gap_penalty
            gap_seq2 = dp[i][j-1] + gap_penalty
            dp[i][j] = max(match_score, gap_seq1, gap_seq2)
    
    align1 = []
    align2 = []
    i, j = n, m
    
    while i > 0 and j > 0:
        current_score = dp[i][j]
        if current_score == dp[i-1][j-1] + get_score(seq1[i-1], seq2[j-1], substitution_matrix, match, mismatch):
            align1.append(seq1[i-1])
            align2.append(seq2[j-1])
            i -= 1
            j -= 1
        elif current_score == dp[i-1][j] + gap_penalty:
            align1.append(seq1[i-1])
            align2.append('-')
            i -= 1
        else:
            align1.append('-')
            align2.append(seq2[j-1])
            j -= 1
    
    while i > 0:
        align1.append(seq1[i-1])
        align2.append('-')
        i -= 1
    while j > 0:
        align1.append('-')
        align2.append(seq2[j-1])
        j -= 1
    
    align1.reverse()
    align2.reverse()
    
    return ''.join(align1), ''.join(align2), int(dp[n][m]), dp


def needleman_wunsch_affine(
    seq1: str,
    seq2: str,
    gap_open: int = -10,
    gap_extend: int = -1,
    substitution_matrix: Dict[Tuple[str, str], int] = None,
    match: int = 1,
    mismatch: int = -1
) -> Tuple[str, str, int]:
    """
    Needleman-Wunsch 全局比对算法（仿射空位罚分，Gotoh算法）
    
    Args:
        seq1: 第一条序列
        seq2: 第二条序列
        gap_open: 空位起始罚分（应为负数）
        gap_extend: 空位延伸罚分（应为负数，通常大于gap_open）
        substitution_matrix: 替换矩阵（如BLOSUM62）
        match: 匹配得分
        mismatch: 不匹配罚分
    
    Returns:
        (aligned_seq1, aligned_seq2, score)
    """
    n = len(seq1)
    m = len(seq2)
    INF = -10**9
    
    M = np.zeros((n + 1, m + 1), dtype=int)
    Ix = np.zeros((n + 1, m + 1), dtype=int)
    Iy = np.zeros((n + 1, m + 1), dtype=int)
    
    for i in range(n + 1):
        for j in range(m + 1):
            M[i][j] = INF
            Ix[i][j] = INF
            Iy[i][j] = INF
    
    M[0][0] = 0
    
    for i in range(1, n + 1):
        Ix[i][0] = gap_open + (i - 1) * gap_extend
    
    for j in range(1, m + 1):
        Iy[0][j] = gap_open + (j - 1) * gap_extend
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score = get_score(seq1[i-1], seq2[j-1], substitution_matrix, match, mismatch)
            M[i][j] = max(M[i-1][j-1], Ix[i-1][j-1], Iy[i-1][j-1]) + score
            Ix[i][j] = max(M[i-1][j] + gap_open, Ix[i-1][j] + gap_extend)
            Iy[i][j] = max(M[i][j-1] + gap_open, Iy[i][j-1] + gap_extend)
    
    align1 = []
    align2 = []
    i, j = n, m
    
    max_score = max(M[i][j], Ix[i][j], Iy[i][j])
    if max_score == M[i][j]:
        current = 'M'
    elif max_score == Ix[i][j]:
        current = 'Ix'
    else:
        current = 'Iy'
    
    while i > 0 or j > 0:
        if current == 'M':
            align1.append(seq1[i-1])
            align2.append(seq2[j-1])
            score = get_score(seq1[i-1], seq2[j-1], substitution_matrix, match, mismatch)
            prev = M[i][j] - score
            if prev == M[i-1][j-1]:
                current = 'M'
            elif prev == Ix[i-1][j-1]:
                current = 'Ix'
            else:
                current = 'Iy'
            i -= 1
            j -= 1
        elif current == 'Ix':
            align1.append(seq1[i-1])
            align2.append('-')
            if Ix[i][j] == M[i-1][j] + gap_open:
                current = 'M'
            else:
                current = 'Ix'
            i -= 1
        else:
            align1.append('-')
            align2.append(seq2[j-1])
            if Iy[i][j] == M[i][j-1] + gap_open:
                current = 'M'
            else:
                current = 'Iy'
            j -= 1
    
    align1.reverse()
    align2.reverse()
    
    final_score = max(M[n][m], Ix[n][m], Iy[n][m])
    
    return ''.join(align1), ''.join(align2), int(final_score)


def print_alignment(align1: str, align2: str, score: int) -> None:
    """美观地打印比对结果"""
    print("序列1:", align1)
    print("序列2:", align2)
    match_str = []
    for a, b in zip(align1, align2):
        if a == b and a != '-':
            match_str.append('|')
        elif a == '-' or b == '-':
            match_str.append(' ')
        else:
            match_str.append('.')
    print("匹配 :", ''.join(match_str))
    print("得分 :", score)
    print()


def calculate_alignment_score(align1: str, align2: str, gap_open: int = -10, 
                              gap_extend: int = -1, 
                              substitution_matrix: Dict[Tuple[str, str], int] = None,
                              match: int = 1, mismatch: int = -1) -> int:
    """手动计算比对结果的得分，用于验证"""
    score = 0
    in_gap1 = False
    in_gap2 = False
    
    for a, b in zip(align1, align2):
        if a == '-' and b == '-':
            continue
        elif a == '-':
            if not in_gap2:
                score += gap_open
                in_gap2 = True
            else:
                score += gap_extend
            in_gap1 = False
        elif b == '-':
            if not in_gap1:
                score += gap_open
                in_gap1 = True
            else:
                score += gap_extend
            in_gap2 = False
        else:
            score += get_score(a, b, substitution_matrix, match, mismatch)
            in_gap1 = False
            in_gap2 = False
    
    return score


def nw_score_linear(seq1: str, seq2: str, gap_penalty: int = -2,
                    substitution_matrix: Dict[Tuple[str, str], int] = None,
                    match: int = 1, mismatch: int = -1) -> np.ndarray:
    """
    线性空间计算Needleman-Wunsch得分，只返回最后一行（长度为len(seq2)+1）
    空间复杂度: O(m)
    """
    n = len(seq1)
    m = len(seq2)
    
    prev = np.zeros(m + 1, dtype=int)
    curr = np.zeros(m + 1, dtype=int)
    
    for j in range(1, m + 1):
        prev[j] = prev[j-1] + gap_penalty
    
    for i in range(1, n + 1):
        curr[0] = prev[0] + gap_penalty
        for j in range(1, m + 1):
            match_score = prev[j-1] + get_score(seq1[i-1], seq2[j-1], substitution_matrix, match, mismatch)
            gap_seq1 = prev[j] + gap_penalty
            gap_seq2 = curr[j-1] + gap_penalty
            curr[j] = max(match_score, gap_seq1, gap_seq2)
        prev, curr = curr, prev
    
    return prev


def hirschberg(seq1: str, seq2: str, gap_penalty: int = -2,
               substitution_matrix: Dict[Tuple[str, str], int] = None,
               match: int = 1, mismatch: int = -1) -> Tuple[str, str, int]:
    """
    Hirschberg算法：线性空间的Needleman-Wunsch全局比对
    空间复杂度: O(min(n, m))，时间复杂度: O(n*m)
    
    Args:
        seq1: 第一条序列
        seq2: 第二条序列
        gap_penalty: 空位罚分（应为负数）
        substitution_matrix: 替换矩阵
        match: 匹配得分
        mismatch: 不匹配罚分
    
    Returns:
        (aligned_seq1, aligned_seq2, score)
    """
    n = len(seq1)
    m = len(seq2)
    
    if n == 0:
        return '-' * m, seq2, m * gap_penalty
    if m == 0:
        return seq1, '-' * n, n * gap_penalty
    if n == 1 or m == 1:
        align1, align2, score, _ = needleman_wunsch(seq1, seq2, gap_penalty, substitution_matrix, match, mismatch)
        return align1, align2, score
    
    mid1 = n // 2
    
    score_left = nw_score_linear(seq1[:mid1], seq2, gap_penalty, substitution_matrix, match, mismatch)
    score_right = nw_score_linear(seq1[mid1:][::-1], seq2[::-1], gap_penalty, substitution_matrix, match, mismatch)[::-1]
    
    total = score_left + score_right
    mid2 = np.argmax(total)
    max_score = total[mid2]
    
    align1_left, align2_left, _ = hirschberg(seq1[:mid1], seq2[:mid2], gap_penalty, substitution_matrix, match, mismatch)
    align1_right, align2_right, _ = hirschberg(seq1[mid1:], seq2[mid2:], gap_penalty, substitution_matrix, match, mismatch)
    
    return align1_left + align1_right, align2_left + align2_right, max_score


def hirschberg_affine(seq1: str, seq2: str, gap_open: int = -10, gap_extend: int = -1,
                      substitution_matrix: Dict[Tuple[str, str], int] = None,
                      match: int = 1, mismatch: int = -1) -> Tuple[str, str, int]:
    """
    仿射空位罚分的线性空间比对（简化版，使用线性罚分近似或小序列用精确算法）
    对于大规模序列，先用Hirschberg快速得到近似骨架，再在关键区域用精确算法
    """
    n = len(seq1)
    m = len(seq2)
    
    if n * m < 1000000:
        return needleman_wunsch_affine(seq1, seq2, gap_open, gap_extend, substitution_matrix, match, mismatch)
    
    approx_gap = gap_open // 2
    align1, align2, _ = hirschberg(seq1, seq2, approx_gap, substitution_matrix, match, mismatch)
    
    score = calculate_alignment_score(align1, align2, gap_open, gap_extend, substitution_matrix, match, mismatch)
    
    return align1, align2, score


def estimate_memory_usage(n: int, m: int, dtype_size: int = 4) -> dict:
    """
    估算不同算法的内存使用量
    Args:
        n, m: 序列长度
        dtype_size: 每个元素字节数（int32=4, int64=8）
    """
    return {
        'standard_dp': f'{(n+1)*(m+1)*dtype_size / 1024 / 1024:.2f} MB',
        'standard_dp_affine': f'{3*(n+1)*(m+1)*dtype_size / 1024 / 1024:.2f} MB',
        'hirschberg': f'{2*min(n, m)*dtype_size / 1024:.2f} KB'
    }


def build_kmer_index(sequence: str, k: int = 11) -> Dict[str, List[int]]:
    """
    构建k-mer索引（BLAST种子匹配核心）
    
    Args:
        sequence: 目标序列（数据库序列）
        k: k-mer长度（DNA通常用11，蛋白质用3）
    
    Returns:
        k-mer到位置列表的映射
    """
    index = defaultdict(list)
    n = len(sequence)
    for i in range(n - k + 1):
        kmer = sequence[i:i+k]
        index[kmer].append(i)
    return index


def find_seeds(query: str, db_index: Dict[str, List[int]], k: int = 11) -> List[Tuple[int, int]]:
    """
    查找查询序列与数据库的k-mer种子匹配
    
    Args:
        query: 查询序列
        db_index: 数据库序列的k-mer索引
        k: k-mer长度
    
    Returns:
        (query_pos, db_pos) 种子匹配位置列表
    """
    seeds = []
    n = len(query)
    for i in range(n - k + 1):
        kmer = query[i:i+k]
        if kmer in db_index:
            for db_pos in db_index[kmer]:
                seeds.append((i, db_pos))
    return seeds


def extend_seed(
    query: str,
    db_seq: str,
    seed_q: int,
    seed_db: int,
    k: int = 11,
    gap_penalty: int = -2,
    substitution_matrix: Dict[Tuple[str, str], int] = None,
    match: int = 1,
    mismatch: int = -1,
    x_dropoff: int = 20
) -> Tuple[str, str, int, Tuple[int, int], Tuple[int, int]]:
    """
    BLAST式种子延伸（X-dropoff算法）
    
    Args:
        query: 查询序列
        db_seq: 数据库序列
        seed_q: 种子在查询中的起始位置
        seed_db: 种子在数据库中的起始位置
        k: 种子长度
        gap_penalty: 空位罚分
        substitution_matrix: 替换矩阵
        match: 匹配得分
        mismatch: 不匹配罚分
        x_dropoff: 允许的最大得分下降值
    
    Returns:
        (aligned_query, aligned_db, score, (q_start, q_end), (db_start, db_end))
    """
    best_score = 0
    best_i, best_j = k, k
    
    current_score = 0
    i, j = k, k
    
    while i > 0 and j > 0:
        qi = seed_q + i - 1
        dj = seed_db + j - 1
        
        if qi >= 0 and dj >= 0:
            s = get_score(query[qi], db_seq[dj], substitution_matrix, match, mismatch)
        else:
            break
        
        diag = current_score + s
        up = current_score + gap_penalty
        left = current_score + gap_penalty
        
        current_score = max(diag, up, left)
        
        if current_score > best_score:
            best_score = current_score
            best_i, best_j = i, j
        
        if best_score - current_score > x_dropoff:
            break
        
        if current_score == diag:
            i -= 1
            j -= 1
        elif current_score == up:
            i -= 1
        else:
            j -= 1
    
    q_start = seed_q + i
    db_start = seed_db + j
    
    current_score = best_score
    i, j = best_i, best_j
    max_i = len(query) - seed_q
    max_j = len(db_seq) - seed_db
    
    end_i, end_j = k, k
    
    while i < max_i and j < max_j:
        qi = seed_q + i
        dj = seed_db + j
        
        s = get_score(query[qi], db_seq[dj], substitution_matrix, match, mismatch)
        
        diag = current_score + s
        up = current_score + gap_penalty
        left = current_score + gap_penalty
        
        current_score = max(diag, up, left)
        
        if current_score > best_score:
            best_score = current_score
            end_i, end_j = i + 1, j + 1
        
        if best_score - current_score > x_dropoff:
            break
        
        if current_score == diag:
            i += 1
            j += 1
        elif current_score == up:
            i += 1
        else:
            j += 1
    
    q_end = seed_q + end_i
    db_end = seed_db + end_j
    
    align_q = query[q_start:q_end]
    align_db = db_seq[db_start:db_end]
    
    return align_q, align_db, best_score, (q_start, q_end), (db_start, db_end)


class BlastLikeAligner:
    """
    简化版BLAST启发式比对器
    基于k-mer种子匹配 + X-dropoff延伸
    """
    
    def __init__(self, k: int = 11, gap_penalty: int = -2,
                 match: int = 1, mismatch: int = -1,
                 substitution_matrix: Dict[Tuple[str, str], int] = None,
                 x_dropoff: int = 20, min_score: int = 20):
        """
        Args:
            k: k-mer种子长度（DNA推荐11，蛋白质推荐3）
            gap_penalty: 空位罚分
            match: 匹配得分
            mismatch: 不匹配罚分
            substitution_matrix: 替换矩阵（如BLOSUM62）
            x_dropoff: X-dropoff阈值
            min_score: 最小报告得分
        """
        self.k = k
        self.gap_penalty = gap_penalty
        self.match = match
        self.mismatch = mismatch
        self.substitution_matrix = substitution_matrix
        self.x_dropoff = x_dropoff
        self.min_score = min_score
    
    def align(self, query: str, subject: str) -> List[Dict]:
        """
        比对查询序列与目标序列
        
        Returns:
            比对结果列表，每个结果包含：
            - score: 比对得分
            - query_start, query_end: 查询序列比对范围
            - subject_start, subject_end: 目标序列比对范围
            - aligned_query, aligned_subject: 比对后的序列
        """
        if len(query) < self.k or len(subject) < self.k:
            return []
        
        db_index = build_kmer_index(subject, self.k)
        seeds = find_seeds(query, db_index, self.k)
        
        results = []
        seen = set()
        
        for seed_q, seed_db in seeds:
            key = (seed_q // 10, seed_db // 10)
            if key in seen:
                continue
            seen.add(key)
            
            align_q, align_s, score, (q_start, q_end), (s_start, s_end) = extend_seed(
                query, subject, seed_q, seed_db, self.k,
                self.gap_penalty, self.substitution_matrix,
                self.match, self.mismatch, self.x_dropoff
            )
            
            if score >= self.min_score:
                results.append({
                    'score': score,
                    'query_start': q_start,
                    'query_end': q_end,
                    'subject_start': s_start,
                    'subject_end': s_end,
                    'aligned_query': align_q,
                    'aligned_subject': align_s,
                    'query_coverage': (q_end - q_start) / len(query),
                    'identity': sum(1 for a, b in zip(align_q, align_s) if a == b) / len(align_q) if align_q else 0
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results


class SequenceDatabase:
    """
    序列数据库，支持批量搜索
    """
    
    def __init__(self, sequences: Dict[str, str] = None):
        """
        Args:
            sequences: {sequence_id: sequence} 字典
        """
        self.sequences = sequences or {}
        self._indexes = {}
        self._lock = threading.Lock()
    
    def add_sequence(self, seq_id: str, sequence: str) -> None:
        """添加序列到数据库"""
        with self._lock:
            self.sequences[seq_id] = sequence
            if seq_id in self._indexes:
                del self._indexes[seq_id]
    
    def add_sequences_batch(self, sequences: Dict[str, str]) -> None:
        """批量添加序列"""
        for seq_id, seq in sequences.items():
            self.add_sequence(seq_id, seq)
    
    def get_index(self, seq_id: str, k: int) -> Dict[str, List[int]]:
        """获取指定序列的k-mer索引（懒加载+缓存）"""
        key = (seq_id, k)
        if key not in self._indexes:
            with self._lock:
                if key not in self._indexes:
                    self._indexes[key] = build_kmer_index(self.sequences[seq_id], k)
        return self._indexes[key]
    
    def search(
        self,
        query: str,
        k: int = 11,
        max_results: int = 10,
        num_threads: int = 4,
        **aligner_kwargs
    ) -> List[Dict]:
        """
        在数据库中搜索查询序列（多线程并行）
        
        Args:
            query: 查询序列
            k: k-mer长度
            max_results: 最大返回结果数
            num_threads: 线程数
            **aligner_kwargs: 传递给BlastLikeAligner的参数
        
        Returns:
            按得分排序的比对结果列表
        """
        aligner = BlastLikeAligner(k=k, **aligner_kwargs)
        seq_ids = list(self.sequences.keys())
        
        results = []
        
        def search_one(seq_id: str) -> Optional[Dict]:
            if seq_id not in self.sequences:
                return None
            seq = self.sequences[seq_id]
            if len(seq) < k:
                return None
            
            try:
                index = self.get_index(seq_id, k)
                seeds = find_seeds(query, index, k)
                
                if not seeds:
                    return None
                
                best_result = None
                seen = set()
                
                for seed_q, seed_db in seeds:
                    key = (seed_q // 10, seed_db // 10)
                    if key in seen:
                        continue
                    seen.add(key)
                    
                    align_q, align_s, score, (q_start, q_end), (s_start, s_end) = extend_seed(
                        query, seq, seed_q, seed_db, k,
                        aligner.gap_penalty, aligner.substitution_matrix,
                        aligner.match, aligner.mismatch, aligner.x_dropoff
                    )
                    
                    if score >= aligner.min_score:
                        current = {
                            'sequence_id': seq_id,
                            'score': score,
                            'query_start': q_start,
                            'query_end': q_end,
                            'subject_start': s_start,
                            'subject_end': s_end,
                            'aligned_query': align_q,
                            'aligned_subject': align_s,
                            'query_coverage': (q_end - q_start) / len(query),
                            'identity': sum(1 for a, b in zip(align_q, align_s) if a == b) / len(align_q) if align_q else 0
                        }
                        if best_result is None or current['score'] > best_result['score']:
                            best_result = current
                
                return best_result
            except Exception as e:
                return None
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_id = {executor.submit(search_one, seq_id): seq_id for seq_id in seq_ids}
            
            for future in as_completed(future_to_id):
                result = future.result()
                if result is not None:
                    results.append(result)
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:max_results]
    
    def search_batch(
        self,
        queries: Dict[str, str],
        num_threads: int = 4,
        **search_kwargs
    ) -> Dict[str, List[Dict]]:
        """
        批量搜索多个查询序列
        
        Args:
            queries: {query_id: query_sequence} 字典
            num_threads: 线程数
            **search_kwargs: 传递给search的参数
        
        Returns:
            {query_id: [results...]} 字典
        """
        results = {}
        
        def process_query(query_id: str, query_seq: str) -> Tuple[str, List[Dict]]:
            res = self.search(query_seq, num_threads=1, **search_kwargs)
            return query_id, res
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_query, qid, qseq) for qid, qseq in queries.items()]
            
            for future in as_completed(futures):
                query_id, res = future.result()
                results[query_id] = res
        
        return results
    
    def __len__(self) -> int:
        return len(self.sequences)


def create_mock_uniprot_database(num_sequences: int = 100, min_length: int = 50, max_length: int = 500) -> SequenceDatabase:
    """
    创建模拟的UniProt蛋白质数据库用于测试
    
    Args:
        num_sequences: 序列数量
        min_length: 最小序列长度
        max_length: 最大序列长度
    
    Returns:
        SequenceDatabase对象
    """
    import random
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    
    db = SequenceDatabase()
    for i in range(num_sequences):
        length = random.randint(min_length, max_length)
        seq = ''.join(random.choice(amino_acids) for _ in range(length))
        db.add_sequence(f'sp|P{i+1:06d}|Protein_{i+1}', seq)
    
    return db


def search_result_to_string(result: Dict, query_seq: str = None, subject_seq: str = None) -> str:
    """格式化搜索结果为字符串"""
    lines = []
    lines.append(f"序列ID: {result.get('sequence_id', 'N/A')}")
    lines.append(f"得分: {result['score']}")
    lines.append(f"同一性: {result['identity']:.1%}")
    lines.append(f"查询覆盖: {result['query_coverage']:.1%}")
    lines.append(f"查询范围: {result['query_start']}-{result['query_end']}")
    lines.append(f"目标范围: {result['subject_start']}-{result['subject_end']}")
    lines.append(f"查询: {result['aligned_query']}")
    lines.append(f"匹配: {''.join('|' if a == b else '.' for a, b in zip(result['aligned_query'], result['aligned_subject']))}")
    lines.append(f"目标: {result['aligned_subject']}")
    return '\n'.join(lines)


if __name__ == "__main__":
    import random
    import time
    import tracemalloc
    
    print("=" * 60)
    print("Needleman-Wunsch + BLAST启发式 + 并行数据库搜索")
    print("=" * 60)
    print()
    
    print("=" * 60)
    print("第一部分: 精确比对算法 (Needleman-Wunsch)")
    print("=" * 60)
    print()
    
    print("示例1: DNA序列比对 (标准DP vs Hirschberg)")
    print("-" * 40)
    seq1_dna = "ATCGATCG"
    seq2_dna = "ATCACG"
    print(f"输入序列1: {seq1_dna}")
    print(f"输入序列2: {seq2_dna}")
    print()
    print("标准DP算法:")
    align1_std, align2_std, score_std, dp = needleman_wunsch(seq1_dna, seq2_dna, gap_penalty=-2, match=2, mismatch=-1)
    print_alignment(align1_std, align2_std, score_std)
    print("Hirschberg线性空间算法:")
    align1_hir, align2_hir, score_hir = hirschberg(seq1_dna, seq2_dna, gap_penalty=-2, match=2, mismatch=-1)
    print_alignment(align1_hir, align2_hir, score_hir)
    print(f"结果一致: {score_std == score_hir}")
    print()
    
    print("示例2: 内存使用对比 (10000bp x 10000bp)")
    print("-" * 40)
    n, m = 10000, 10000
    memory_est = estimate_memory_usage(n, m)
    print(f"序列长度: {n} x {m}")
    print(f"标准DP内存估算: {memory_est['standard_dp']}")
    print(f"标准DP(仿射)内存估算: {memory_est['standard_dp_affine']}")
    print(f"Hirschberg内存估算: {memory_est['hirschberg']}")
    print(f"内存节省: {100 - (2*min(n,m)*4) / ((n+1)*(m+1)*4) * 100:.2f}%")
    print()
    
    print("=" * 60)
    print("第二部分: 启发式比对 (BLAST-like)")
    print("=" * 60)
    print()
    
    print("示例3: BLAST启发式比对 - 查找局部相似区域")
    print("-" * 40)
    random.seed(42)
    bases = ['A', 'T', 'C', 'G']
    
    query = "ATCGATCGATCGATCGATCG" + ''.join(random.choice(bases) for _ in range(100))
    subject = ''.join(random.choice(bases) for _ in range(50)) + "ATCGATCGATCGATCGATCG" + ''.join(random.choice(bases) for _ in range(50))
    
    print(f"查询序列长度: {len(query)}")
    print(f"目标序列长度: {len(subject)}")
    print()
    
    aligner = BlastLikeAligner(k=7, gap_penalty=-2, match=2, mismatch=-1, x_dropoff=10, min_score=10)
    results = aligner.align(query, subject)
    
    print(f"找到 {len(results)} 个比对区域:")
    for i, res in enumerate(results[:3]):
        print(f"  区域 {i+1}: 得分={res['score']}, 同一性={res['identity']:.1%}")
        print(f"    查询: {res['query_start']}-{res['query_end']}")
        print(f"    目标: {res['subject_start']}-{res['subject_end']}")
        print(f"    {res['aligned_query']}")
        print(f"    {''.join('|' if a == b else '.' for a, b in zip(res['aligned_query'], res['aligned_subject']))}")
        print(f"    {res['aligned_subject']}")
    print()
    
    print("示例4: 蛋白质启发式比对 (BLOSUM62)")
    print("-" * 40)
    prot_query = "HEAGAWGHEE" * 3
    prot_subject = "PAWHEAE" * 2 + "HEAGAWGHEE" * 2
    
    prot_aligner = BlastLikeAligner(k=3, gap_penalty=-8, substitution_matrix=BLOSUM62, x_dropoff=15, min_score=20)
    prot_results = prot_aligner.align(prot_query, prot_subject)
    
    print(f"查询序列: {prot_query[:50]}...")
    print(f"目标序列: {prot_subject[:50]}...")
    print(f"找到 {len(prot_results)} 个比对区域")
    if prot_results:
        best = prot_results[0]
        print(f"最佳比对: 得分={best['score']}, 同一性={best['identity']:.1%}")
        print(f"  {best['aligned_query']}")
        print(f"  {''.join('|' if a == b else '.' for a, b in zip(best['aligned_query'], best['aligned_subject']))}")
        print(f"  {best['aligned_subject']}")
    print()
    
    print("=" * 60)
    print("第三部分: 并行数据库搜索")
    print("=" * 60)
    print()
    
    print("示例5: 模拟UniProt数据库搜索")
    print("-" * 40)
    print("创建模拟蛋白质数据库 (200条序列)...")
    db = create_mock_uniprot_database(num_sequences=200, min_length=100, max_length=300)
    print(f"数据库大小: {len(db)} 条序列")
    print()
    
    known_seq = "HEAGAWGHEE" * 5
    db.add_sequence("sp|P00000|KNOWN_PROTEIN", known_seq + ''.join(random.choice('ACDEFGHIKLMNPQRSTVWY') for _ in range(50)))
    
    query_prot = known_seq[:30]
    print(f"查询序列: {query_prot}")
    print(f"查询长度: {len(query_prot)}")
    print()
    
    print("单线程搜索 (4线程)...")
    start_time = time.time()
    search_results = db.search(
        query_prot, k=3, max_results=5, num_threads=4,
        gap_penalty=-8, substitution_matrix=BLOSUM62,
        x_dropoff=20, min_score=30
    )
    search_time = time.time() - start_time
    print(f"搜索用时: {search_time:.2f}s")
    print(f"找到 {len(search_results)} 个匹配:")
    for i, res in enumerate(search_results):
        print(f"  {i+1}. {res['sequence_id']}")
        print(f"     得分: {res['score']}, 同一性: {res['identity']:.1%}, 覆盖: {res['query_coverage']:.1%}")
    print()
    
    print("示例6: 批量查询并行搜索")
    print("-" * 40)
    queries = {
        "Query_1": "HEAGAWGHEE" * 3,
        "Query_2": "PAWHEAE" * 4,
        "Query_3": ''.join(random.choice('ACDEFGHIKLMNPQRSTVWY') for _ in range(50)),
        "Query_4": "ACDEFGHIKLMNPQRSTVWY" * 2,
    }
    print(f"批量查询: {len(queries)} 个查询序列")
    print()
    
    start_time = time.time()
    batch_results = db.search_batch(
        queries, num_threads=4, k=3, max_results=3,
        gap_penalty=-8, substitution_matrix=BLOSUM62,
        x_dropoff=15, min_score=25
    )
    batch_time = time.time() - start_time
    print(f"批量搜索用时: {batch_time:.2f}s")
    for qid, res_list in batch_results.items():
        print(f"  {qid}: 找到 {len(res_list)} 个匹配")
        if res_list:
            print(f"    最佳: {res_list[0]['sequence_id']} (得分: {res_list[0]['score']})")
    print()
    
    print("示例7: 性能对比 - 不同线程数")
    print("-" * 40)
    small_db = create_mock_uniprot_database(num_sequences=50, min_length=100, max_length=200)
    test_query = "HEAGAWGHEE" * 4
    
    for threads in [1, 2, 4]:
        start_time = time.time()
        _ = small_db.search(
            test_query, k=3, max_results=5, num_threads=threads,
            gap_penalty=-8, substitution_matrix=BLOSUM62,
            x_dropoff=15, min_score=25
        )
        elapsed = time.time() - start_time
        print(f"  {threads} 线程: {elapsed:.2f}s")
    print()
    
    print("示例8: 完整工作流演示")
    print("-" * 40)
    print("1. 构建数据库...")
    workflow_db = SequenceDatabase()
    seqs = {
        "Gene_1": "ATCGATCGATCG" * 10,
        "Gene_2": "ATCG" + "GGGG" + "ATCGATCG" * 8,
        "Gene_3": "ATCGATCG" * 5 + "AAAA" + "ATCG" * 5,
        "Gene_4": "TTTT" + "ATCGATCG" * 9,
        "Gene_5": "ATCGATCG" * 10,
    }
    workflow_db.add_sequences_batch(seqs)
    print(f"   已添加 {len(workflow_db)} 条序列")
    
    print("2. 搜索查询序列...")
    workflow_query = "ATCGATCGATCG" * 3
    hits = workflow_db.search(workflow_query, k=8, max_results=3, num_threads=2, gap_penalty=-2, match=2, mismatch=-1, min_score=20)
    
    print("3. 结果展示:")
    for hit in hits:
        print(search_result_to_string(hit))
        print()
    
    print("=" * 60)
    print("所有示例执行完成!")
    print("=" * 60)
