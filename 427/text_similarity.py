import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


def tokenize_chinese(text):
    return list(jieba.cut(text))


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def levenshtein_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    
    distance = levenshtein_distance(text1, text2)
    max_len = max(len(text1), len(text2))
    
    if max_len == 0:
        return 0.0
    
    return 1 - (distance / max_len)


def longest_common_substring(text1, text2):
    if not text1 or not text2:
        return 0
    
    m = len(text1)
    n = len(text2)
    max_len = 0
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_len = max(max_len, dp[i][j])
            else:
                dp[i][j] = 0
    
    return max_len


def lcs_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    
    lcs_len = longest_common_substring(text1, text2)
    min_len = min(len(text1), len(text2))
    
    if min_len == 0:
        return 0.0
    
    return lcs_len / min_len


_char_embedding_cache = {}


def _get_char_embedding(char):
    if char not in _char_embedding_cache:
        np.random.seed(hash(char) % (2**32))
        _char_embedding_cache[char] = np.random.randn(50)
    return _char_embedding_cache[char]


def get_average_embedding(text):
    tokens = tokenize_chinese(text)
    if not tokens:
        return np.zeros(50)
    
    embeddings = []
    for token in tokens:
        for char in token:
            embeddings.append(_get_char_embedding(char))
    
    if not embeddings:
        return np.zeros(50)
    
    return np.mean(embeddings, axis=0)


def semantic_similarity_avg(text1, text2):
    if not text1 or not text2:
        return 0.0
    
    emb1 = get_average_embedding(text1)
    emb2 = get_average_embedding(text2)
    
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    cos_sim = np.dot(emb1, emb2) / (norm1 * norm2)
    
    return (cos_sim + 1) / 2


_sbert_model = None


def semantic_similarity_sbert(text1, text2, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
    global _sbert_model
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")
    
    if not text1 or not text2:
        return 0.0
    
    if _sbert_model is None:
        _sbert_model = SentenceTransformer(model_name)
    
    embeddings = _sbert_model.encode([text1, text2])
    cos_sim = cosine_similarity(embeddings[0:1], embeddings[1:2])[0][0]
    
    return (cos_sim + 1) / 2


def semantic_similarity(text1, text2, use_sbert=False):
    if use_sbert and SENTENCE_TRANSFORMERS_AVAILABLE:
        return semantic_similarity_sbert(text1, text2)
    else:
        return semantic_similarity_avg(text1, text2)


def jaccard_similarity(text1, text2):
    tokens1 = set(tokenize_chinese(text1))
    tokens2 = set(tokenize_chinese(text2))
    
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    
    return len(intersection) / len(union)


def cosine_similarity_tfidf(text1, text2):
    corpus = [text1, text2]
    
    vectorizer = TfidfVectorizer(tokenizer=tokenize_chinese, token_pattern=None)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    return similarity[0][0]


def calculate_similarity(text1, text2, method='both', short_text_threshold=10, use_sbert=False):
    result = {}
    is_short_text = len(text1) < short_text_threshold or len(text2) < short_text_threshold
    
    if method in ['cosine', 'both']:
        result['cosine'] = cosine_similarity_tfidf(text1, text2)
    
    if method in ['jaccard', 'both']:
        result['jaccard'] = jaccard_similarity(text1, text2)
    
    if method in ['levenshtein', 'both']:
        result['levenshtein'] = levenshtein_similarity(text1, text2)
    
    if method in ['lcs', 'both']:
        result['lcs'] = lcs_similarity(text1, text2)
    
    if method in ['semantic', 'both']:
        result['semantic'] = semantic_similarity(text1, text2, use_sbert=use_sbert)
    
    if is_short_text and method == 'both':
        result['is_short_text'] = True
        result['weighted_avg'] = (
            result['cosine'] * 0.12 +
            result['jaccard'] * 0.12 +
            result['levenshtein'] * 0.28 +
            result['lcs'] * 0.28 +
            result['semantic'] * 0.20
        )
    elif method == 'both':
        result['is_short_text'] = False
        result['weighted_avg'] = (
            result['cosine'] * 0.28 +
            result['jaccard'] * 0.28 +
            result['levenshtein'] * 0.12 +
            result['lcs'] * 0.12 +
            result['semantic'] * 0.20
        )
    
    return result


def calculate_batch_similarity(texts, method='weighted_avg', **kwargs):
    n = len(texts)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 1.0
            elif i < j:
                result = calculate_similarity(texts[i], texts[j], **kwargs)
                score = result.get(method, result.get('weighted_avg', 0.0))
                matrix[i][j] = float(score)
                matrix[j][i] = matrix[i][j]
    
    return matrix


def print_similarity_matrix(matrix, labels=None, title="相似度矩阵"):
    n = matrix.shape[0]
    
    if labels is None:
        labels = [f"文本{i+1}" for i in range(n)]
    
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")
    
    col_width = 12
    header = " " * 10 + "".join(f"{label:^{col_width}}" for label in labels)
    print(header)
    print("-" * len(header))
    
    for i in range(n):
        row = f"{labels[i]:<10}"
        for j in range(n):
            row += f"{matrix[i][j]:^{col_width}.4f}"
        print(row)
    
    print("-" * len(header))


def find_most_similar(texts, target_idx, top_k=3, method='weighted_avg', **kwargs):
    n = len(texts)
    similarities = []
    
    for i in range(n):
        if i != target_idx:
            result = calculate_similarity(texts[target_idx], texts[i], **kwargs)
            score = result.get(method, result.get('weighted_avg', 0.0))
            similarities.append((i, texts[i], float(score)))
    
    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:top_k]


def print_result(label, result):
    print(f"\n【{label}】")
    print(f"  短文本模式: {'是' if result.get('is_short_text') else '否'}")
    print(f"  余弦相似度 (TF-IDF): {result['cosine']:.4f}")
    print(f"  Jaccard相似度: {result['jaccard']:.4f}")
    print(f"  编辑距离相似度: {result['levenshtein']:.4f}")
    print(f"  最长公共子串: {result['lcs']:.4f}")
    print(f"  语义相似度: {result['semantic']:.4f}")
    if 'weighted_avg' in result:
        print(f"  加权平均得分: {result['weighted_avg']:.4f}")


if __name__ == "__main__":
    text1 = "今天天气真好，适合出去散步"
    text2 = "今天天气不错，可以出去走走"
    text3 = "机器学习是人工智能的核心技术"
    text4 = "深度学习是机器学习的重要分支"
    
    short1 = "你好"
    short2 = "您好"
    short3 = "天气"
    short4 = "天气好"
    short5 = "苹果"
    short6 = "香蕉"
    
    print("=" * 60)
    print("文本相似度计算示例")
    print("=" * 60)
    
    print(f"\n=== 长文本测试 (>=10字) ===")
    print(f"文本1: {text1}")
    print(f"文本2: {text2}")
    print(f"文本3: {text3}")
    print(f"文本4: {text4}")
    
    print_result("文本1 vs 文本2 (语义相近)", calculate_similarity(text1, text2))
    print_result("文本1 vs 文本3 (不相关)", calculate_similarity(text1, text3))
    print_result("文本3 vs 文本4 (语义相关)", calculate_similarity(text3, text4))
    
    print(f"\n=== 短文本测试 (<10字) ===")
    print(f"short1: {short1} ({len(short1)}字)")
    print(f"short2: {short2} ({len(short2)}字)")
    print(f"short3: {short3} ({len(short3)}字)")
    print(f"short4: {short4} ({len(short4)}字)")
    print(f"short5: {short5} ({len(short5)}字)")
    print(f"short6: {short6} ({len(short6)}字)")
    
    print_result("你好 vs 您好 (语义相近)", calculate_similarity(short1, short2))
    print_result("天气 vs 天气好 (语义相近)", calculate_similarity(short3, short4))
    print_result("苹果 vs 香蕉 (不相关)", calculate_similarity(short5, short6))
    print_result("你好 vs 苹果 (不相关)", calculate_similarity(short1, short5))
    
    print(f"\n=== 长短文本混合测试 ===")
    print_result("你好 vs 今天天气真好 (混合)", calculate_similarity(short1, text1))
    
    print(f"\n=== 语义相似度测试 ===")
    print(f"Sentence-BERT可用: {SENTENCE_TRANSFORMERS_AVAILABLE}")
    print(f"仅语义相似度 (词向量平均): {calculate_similarity(text3, text4, method='semantic')}")
    
    print(f"\n=== 批量相似度计算 ===")
    batch_texts = [text1, text2, text3, text4]
    labels = ["散步天气", "天气不错", "人工智能", "深度学习"]
    
    sim_matrix = calculate_batch_similarity(batch_texts, method='weighted_avg')
    print_similarity_matrix(sim_matrix, labels=labels, title="批量文本相似度矩阵 (加权平均)")
    
    cos_matrix = calculate_batch_similarity(batch_texts, method='cosine')
    print_similarity_matrix(cos_matrix, labels=labels, title="批量文本相似度矩阵 (余弦相似度)")
    
    semantic_matrix = calculate_batch_similarity(batch_texts, method='semantic')
    print_similarity_matrix(semantic_matrix, labels=labels, title="批量文本相似度矩阵 (语义相似度)")
    
    print(f"\n=== 查找最相似文本 ===")
    target_idx = 2
    print(f"目标文本: {batch_texts[target_idx]}")
    print(f"目标标签: {labels[target_idx]}")
    most_similar = find_most_similar(batch_texts, target_idx, top_k=2)
    for idx, text, score in most_similar:
        print(f"  相似度: {score:.4f} - {labels[idx]}: {text}")
    
    print("\n" + "=" * 60)
    print("单独调用示例:")
    print("=" * 60)
    print(f"仅编辑距离: {calculate_similarity(short1, short2, method='levenshtein')}")
    print(f"仅最长公共子串: {calculate_similarity(short3, short4, method='lcs')}")
    print(f"仅语义相似度: {calculate_similarity(text1, text2, method='semantic')}")
    print(f"自定义阈值(5字): {calculate_similarity('测试', '测试', short_text_threshold=5)}")
