import math
import re
from collections import Counter
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_SBERT_AVAILABLE = False
_SBERT_MODEL = None
try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except ImportError:
    pass


def _get_sbert_model(model_name="all-MiniLM-L6-v2"):
    global _SBERT_MODEL
    if not _SBERT_AVAILABLE:
        return None
    if _SBERT_MODEL is None:
        _SBERT_MODEL = SentenceTransformer(model_name)
    return _SBERT_MODEL


def tokenize(text):
    return re.findall(r'\w+', text.lower())


def levenshtein_similarity(text1, text2):
    s1 = text1.lower()
    s2 = text2.lower()

    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    if len(s1) > len(s2):
        s1, s2 = s2, s1

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            insert = prev[j + 1] + 1
            delete = curr[j] + 1
            replace = prev[j] + (c1 != c2)
            curr.append(min(insert, delete, replace))
        prev = curr

    distance = prev[-1]
    max_len = max(len(s1), len(s2))
    return 1 - (distance / max_len) if max_len > 0 else 1.0


def longest_common_substring_similarity(text1, text2):
    s1 = text1.lower()
    s2 = text2.lower()

    if not s1 or not s2:
        return 0.0

    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_len = max(max_len, dp[i][j])
            else:
                dp[i][j] = 0

    min_len = min(len(s1), len(s2))
    return max_len / min_len if min_len > 0 else 0.0


def cosine_similarity(text1, text2):
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)

    if not tokens1 or not tokens2:
        return 0.0

    all_tokens = set(tokens1) | set(tokens2)

    tf1 = Counter(tokens1)
    tf2 = Counter(tokens2)

    idf = {}
    n = 2
    for token in all_tokens:
        df = (1 if token in tf1 else 0) + (1 if token in tf2 else 0)
        idf[token] = math.log(n / df) + 1

    v1 = {t: tf1.get(t, 0) * idf.get(t, 0) for t in all_tokens}
    v2 = {t: tf2.get(t, 0) * idf.get(t, 0) for t in all_tokens}

    dot = sum(v1[t] * v2[t] for t in all_tokens)
    norm1 = math.sqrt(sum(v ** 2 for v in v1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in v2.values()))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


def _cosine_vec(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


_WORD_VEC_CACHE = {}
_WORD_VEC_DIM = 100


def _char_ngram_vector(word, dim=_WORD_VEC_DIM):
    if word in _WORD_VEC_CACHE:
        return _WORD_VEC_CACHE[word]

    vec = np.zeros(dim, dtype=np.float64)
    ngrams = []
    word_pad = f"^{word}$"
    for n in range(2, 5):
        for i in range(len(word_pad) - n + 1):
            ngrams.append(word_pad[i:i + n])

    for ng in ngrams:
        idx = hash(ng) % dim
        vec[idx] += 1.0

    if np.linalg.norm(vec) > 0:
        vec = vec / np.linalg.norm(vec)

    _WORD_VEC_CACHE[word] = vec
    return vec


def word_embedding_average_similarity(text1, text2):
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)

    if not tokens1 or not tokens2:
        return 0.0

    vec1 = np.mean([_char_ngram_vector(t) for t in tokens1], axis=0)
    vec2 = np.mean([_char_ngram_vector(t) for t in tokens2], axis=0)

    return _cosine_vec(vec1, vec2)


def sbert_similarity(text1, text2, model_name="all-MiniLM-L6-v2"):
    model = _get_sbert_model(model_name)
    if model is None:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Install with: pip install sentence-transformers"
        )

    embeddings = model.encode([text1, text2], convert_to_numpy=True)
    return _cosine_vec(embeddings[0], embeddings[1])


def semantic_similarity(text1, text2, method="embedding_average"):
    if method == "embedding_average":
        return word_embedding_average_similarity(text1, text2)
    elif method == "sbert":
        return sbert_similarity(text1, text2)
    else:
        raise ValueError(
            f"Unknown semantic method: {method}. "
            f"Use 'embedding_average' or 'sbert'."
        )


def jaccard_similarity(text1, text2):
    set1 = set(tokenize(text1))
    set2 = set(tokenize(text2))

    if not set1 and not set2:
        return 1.0

    intersection = set1 & set2
    union = set1 | set2

    return len(intersection) / len(union)


def is_short_text(text1, text2, threshold=10):
    return min(len(text1), len(text2)) < threshold


def text_similarity(text1, text2, method="cosine"):
    if method in ("semantic", "embedding_average"):
        return semantic_similarity(text1, text2, "embedding_average")
    elif method == "sbert":
        return semantic_similarity(text1, text2, "sbert")
    elif method == "levenshtein":
        return levenshtein_similarity(text1, text2)
    elif method == "lcs":
        return longest_common_substring_similarity(text1, text2)

    base_score = 0.0
    if method == "cosine":
        base_score = cosine_similarity(text1, text2)
    elif method == "jaccard":
        base_score = jaccard_similarity(text1, text2)
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Use 'cosine', 'jaccard', 'levenshtein', 'lcs', 'semantic', 'embedding_average', or 'sbert'."
        )

    if is_short_text(text1, text2):
        lev_score = levenshtein_similarity(text1, text2)
        lcs_score = longest_common_substring_similarity(text1, text2)
        final_score = base_score * 0.3 + lev_score * 0.5 + lcs_score * 0.2
        return final_score

    return base_score


def batch_similarity(texts, method="cosine"):
    n = len(texts)
    matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        matrix[i, i] = 1.0
        for j in range(i + 1, n):
            score = text_similarity(texts[i], texts[j], method)
            matrix[i, j] = score
            matrix[j, i] = score

    return matrix


def print_similarity_matrix(matrix, texts, method=""):
    n = len(texts)
    labels = [f"T{i+1}" for i in range(n)]

    if method:
        print(f"\n=== Similarity Matrix ({method}) ===")
    else:
        print("\n=== Similarity Matrix ===")

    for i, text in enumerate(texts):
        print(f"T{i+1}: {text}")

    header = "       " + "  ".join(f"{label:>6}" for label in labels)
    print("\n" + header)

    for i in range(n):
        row = f"{labels[i]:<6}"
        for j in range(n):
            row += f"  {matrix[i, j]:>6.4f}"
        print(row)

    return matrix


def find_most_similar(texts, target_text, method="cosine", top_k=3):
    scores = []
    for i, text in enumerate(texts):
        score = text_similarity(target_text, text, method)
        scores.append((i, text, score))

    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:top_k]


if __name__ == "__main__":
    t1 = "The cat sat on the mat"
    t2 = "The dog sat on the mat"
    t3 = "A completely different sentence about programming"

    print("=== Regular Text (>=10 chars) ===")
    print("=== Cosine Similarity (TF-IDF) ===")
    print(f"'{t1}' vs '{t2}': {text_similarity(t1, t2, 'cosine'):.4f}")
    print(f"'{t1}' vs '{t3}': {text_similarity(t1, t3, 'cosine'):.4f}")

    print("\n=== Jaccard Similarity ===")
    print(f"'{t1}' vs '{t2}': {text_similarity(t1, t2, 'jaccard'):.4f}")
    print(f"'{t1}' vs '{t3}': {text_similarity(t1, t3, 'jaccard'):.4f}")

    s1 = "cat"
    s2 = "cap"
    s3 = "dog"

    print("\n=== Short Text (<10 chars) - With Hybrid Scoring ===")
    print(f"'{s1}' vs '{s2}': {text_similarity(s1, s2, 'cosine'):.4f}")
    print(f"  - Cosine alone: {cosine_similarity(s1, s2):.4f}")
    print(f"  - Levenshtein:   {levenshtein_similarity(s1, s2):.4f}")
    print(f"  - LCS:           {longest_common_substring_similarity(s1, s2):.4f}")

    print(f"\n'{s1}' vs '{s3}': {text_similarity(s1, s3, 'cosine'):.4f}")
    print(f"  - Cosine alone: {cosine_similarity(s1, s3):.4f}")
    print(f"  - Levenshtein:   {levenshtein_similarity(s1, s3):.4f}")
    print(f"  - LCS:           {longest_common_substring_similarity(s1, s3):.4f}")

    print("\n=== Semantic Similarity (Embedding Average) ===")
    se1 = "The quick brown fox jumps over the lazy dog"
    se2 = "A fast brown fox leaps over a sleepy hound"
    se3 = "Machine learning models analyze patterns in data"
    se4 = "Deep learning neural networks detect patterns"

    print(f"'{se1}'")
    print(f"vs '{se2}': {text_similarity(se1, se2, 'semantic'):.4f}")
    print(f"vs '{se3}': {text_similarity(se1, se3, 'semantic'):.4f}")
    print(f"\n'{se3}'")
    print(f"vs '{se4}': {text_similarity(se3, se4, 'semantic'):.4f}")

    print("\n=== Batch Similarity Matrix ===")
    corpus = [
        "The cat sat on the mat",
        "The dog sat on the mat",
        "A feline rested on the rug",
        "Programming with Python is fun",
        "Coding in Python is enjoyable",
        "The weather is nice today",
    ]

    for m in ["cosine", "jaccard", "semantic"]:
        mat = batch_similarity(corpus, method=m)
        print_similarity_matrix(mat, corpus, method=m)

    print("\n=== Find Most Similar ===")
    target = "I love Python programming"
    top = find_most_similar(corpus, target, method="semantic", top_k=3)
    print(f"Target: '{target}'")
    for rank, (idx, text, score) in enumerate(top, 1):
        print(f"  {rank}. Score={score:.4f} | '{text}'")
