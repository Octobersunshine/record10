import math
import warnings

_LENGTH_WARN_THRESHOLD = 10000


def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)

    if m > _LENGTH_WARN_THRESHOLD or n > _LENGTH_WARN_THRESHOLD:
        warnings.warn(
            f"字符串长度较大(s1={m}, s2={n})，编辑距离计算时间复杂度为O(m×n)，"
            f"可能耗时较长。建议优先考虑Jaccard相似度。",
            RuntimeWarning,
            stacklevel=2,
        )

    if m < n:
        s1, s2 = s2, s1
        m, n = n, m

    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev

    return prev[n]


def levenshtein_similarity(s1, s2):
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1 - distance / max_len


def jaccard_similarity(s1, s2):
    set1 = set(s1)
    set2 = set(s2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    if union == 0:
        return 1.0
    return intersection / union


def _ngram_counts(s, n=2):
    counts = {}
    for i in range(len(s) - n + 1):
        gram = s[i:i + n]
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def cosine_similarity(s1, s2, n=2):
    if len(s1) < n and len(s2) < n:
        return 1.0 if s1 == s2 else 0.0

    counts1 = _ngram_counts(s1, n)
    counts2 = _ngram_counts(s2, n)

    all_grams = set(counts1) | set(counts2)
    dot = sum(counts1.get(g, 0) * counts2.get(g, 0) for g in all_grams)
    norm1 = math.sqrt(sum(v * v for v in counts1.values()))
    norm2 = math.sqrt(sum(v * v for v in counts2.values()))

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def jaro_similarity(s1, s2):
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_dist = max(len1, len2) // 2 - 1
    if match_dist < 0:
        match_dist = 0

    s1_matched = [False] * len1
    s2_matched = [False] * len2

    matches = 0
    transpositions = 0

    for i in range(len1):
        lo = max(0, i - match_dist)
        hi = min(i + match_dist + 1, len2)
        for j in range(lo, hi):
            if s2_matched[j] or s1[i] != s2[j]:
                continue
            s1_matched[i] = True
            s2_matched[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matched[i]:
            continue
        while not s2_matched[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    return (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3


def jaro_winkler_similarity(s1, s2, p=0.1):
    jaro_sim = jaro_similarity(s1, s2)

    prefix_len = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break

    return jaro_sim + prefix_len * p * (1 - jaro_sim)


def batch_compare(source, targets, metric="levenshtein"):
    metric_funcs = {
        "levenshtein": levenshtein_similarity,
        "jaccard": jaccard_similarity,
        "cosine": cosine_similarity,
        "jaro_winkler": jaro_winkler_similarity,
    }
    if metric not in metric_funcs:
        raise ValueError(f"不支持的度量方法: {metric}，可选: {list(metric_funcs.keys())}")

    func = metric_funcs[metric]
    results = []
    for target in targets:
        score = func(source, target)
        results.append((target, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


if __name__ == "__main__":
    test_pairs = [
        ("kitten", "sitting"),
        ("saturday", "sunday"),
        ("python", "pyhton"),
        ("abc", "abc"),
        ("abc", ""),
        ("", ""),
        ("中国", "中华"),
        ("人工智能", "人工智障"),
    ]

    print("=" * 90)
    print("成对字符串相似度对比")
    print("=" * 90)
    header = f"{'s1':<12} {'s2':<12} {'Levenshtein':>12} {'Jaccard':>10} {'Cosine':>10} {'Jaro-Winkler':>14}"
    print(header)
    print("-" * 90)

    for s1, s2 in test_pairs:
        lev = levenshtein_similarity(s1, s2)
        jac = jaccard_similarity(s1, s2)
        cos = cosine_similarity(s1, s2)
        jw = jaro_winkler_similarity(s1, s2)
        print(f"{s1:<12} {s2:<12} {lev:>12.4f} {jac:>10.4f} {cos:>10.4f} {jw:>14.4f}")

    print()
    print("=" * 90)
    print("批量对比：'张三' 与候选姓名列表（Jaro-Winkler）")
    print("=" * 90)
    source = "张三"
    candidates = ["张三", "张山", "章三", "李四", "王五", "赵六", "张思"]
    results = batch_compare(source, candidates, metric="jaro_winkler")
    for target, score in results:
        bar = "█" * int(score * 30)
        print(f"  {source} vs {target:<6}  {score:.4f}  {bar}")

    print()
    print("=" * 90)
    print("批量对比：'machine learning' 与候选短语（Cosine n-gram=2）")
    print("=" * 90)
    source = "machine learning"
    candidates = [
        "machine learning",
        "deep learning",
        "machine earning",
        "machines learn",
        "natural language processing",
        "learning machine",
    ]
    results = batch_compare(source, candidates, metric="cosine")
    for target, score in results:
        bar = "█" * int(score * 30)
        print(f"  {source} vs {target:<35} {score:.4f}  {bar}")
