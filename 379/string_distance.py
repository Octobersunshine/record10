from dataclasses import dataclass
from typing import List, Tuple, Optional

MAX_STRING_LENGTH = 10000

OP_INSERT = "insert"
OP_DELETE = "delete"
OP_SUBSTITUTE = "substitute"
OP_MATCH = "match"


@dataclass
class EditCosts:
    insert: float = 1.0
    delete: float = 1.0
    substitute: float = 1.0

    def max_cost(self) -> float:
        return max(self.insert, self.delete, self.substitute)


@dataclass
class EditOperation:
    op: str
    char: Optional[str] = None
    target_char: Optional[str] = None
    position: Optional[int] = None

    def __repr__(self) -> str:
        if self.op == OP_MATCH:
            return f"match('{self.char}')"
        elif self.op == OP_INSERT:
            return f"insert('{self.char}' at pos {self.position})"
        elif self.op == OP_DELETE:
            return f"delete('{self.char}' at pos {self.position})"
        elif self.op == OP_SUBSTITUTE:
            return f"substitute('{self.char}' -> '{self.target_char}' at pos {self.position})"
        return self.op


@dataclass
class LevenshteinResult:
    distance: float
    operations: List[EditOperation]
    similarity: float


def hamming_distance(s1: str, s2: str) -> int:
    if len(s1) != len(s2):
        raise ValueError("Strings must be of equal length for Hamming distance")
    if len(s1) > MAX_STRING_LENGTH:
        raise ValueError(f"String length exceeds maximum limit ({MAX_STRING_LENGTH})")
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def levenshtein_distance(s1: str, s2: str, max_length: int = MAX_STRING_LENGTH) -> int:
    if len(s1) > max_length or len(s2) > max_length:
        raise ValueError(
            f"String length exceeds maximum limit ({max_length}). "
            f"Got lengths {len(s1)} and {len(s2)}"
        )

    if len(s1) < len(s2):
        s1, s2 = s2, s1

    if len(s2) == 0:
        return len(s1)

    row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        prev = row[0]
        row[0] = i + 1
        for j, c2 in enumerate(s2):
            temp = row[j + 1]
            if c1 == c2:
                row[j + 1] = prev
            else:
                row[j + 1] = 1 + min(prev, row[j], row[j + 1])
            prev = temp

    return row[-1]


def levenshtein_full(
    s1: str,
    s2: str,
    costs: Optional[EditCosts] = None,
    max_length: int = MAX_STRING_LENGTH,
) -> LevenshteinResult:
    if costs is None:
        costs = EditCosts()

    if len(s1) > max_length or len(s2) > max_length:
        raise ValueError(
            f"String length exceeds maximum limit ({max_length}). "
            f"Got lengths {len(s1)} and {len(s2)}"
        )

    m, n = len(s1), len(s2)

    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i * costs.delete
    for j in range(n + 1):
        dp[0][j] = j * costs.insert

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + costs.delete,
                    dp[i][j - 1] + costs.insert,
                    dp[i - 1][j - 1] + costs.substitute,
                )

    operations = _backtrack_path(dp, s1, s2, costs)
    distance = dp[m][n]
    similarity = normalized_similarity(distance, m, n, costs)

    return LevenshteinResult(
        distance=distance,
        operations=operations,
        similarity=similarity,
    )


def _backtrack_path(
    dp: List[List[float]],
    s1: str,
    s2: str,
    costs: EditCosts,
) -> List[EditOperation]:
    i, j = len(s1), len(s2)
    operations: List[EditOperation] = []

    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i - 1] == s2[j - 1]:
            operations.append(
                EditOperation(op=OP_MATCH, char=s1[i - 1], position=i - 1)
            )
            i -= 1
            j -= 1
        else:
            candidates = []
            if i > 0 and j > 0:
                sub_cost = dp[i - 1][j - 1] + costs.substitute
                candidates.append((sub_cost, "sub", i - 1, j - 1))
            if i > 0:
                del_cost = dp[i - 1][j] + costs.delete
                candidates.append((del_cost, "del", i - 1, j))
            if j > 0:
                ins_cost = dp[i][j - 1] + costs.insert
                candidates.append((ins_cost, "ins", i, j - 1))

            _, op_type, new_i, new_j = min(candidates, key=lambda x: x[0])

            if op_type == "sub":
                operations.append(
                    EditOperation(
                        op=OP_SUBSTITUTE,
                        char=s1[i - 1],
                        target_char=s2[j - 1],
                        position=i - 1,
                    )
                )
            elif op_type == "del":
                operations.append(
                    EditOperation(op=OP_DELETE, char=s1[i - 1], position=i - 1)
                )
            else:
                operations.append(
                    EditOperation(op=OP_INSERT, char=s2[j - 1], position=i)
                )

            i, j = new_i, new_j

    operations.reverse()
    return operations


def normalized_similarity(
    distance: float,
    len1: int,
    len2: int,
    costs: Optional[EditCosts] = None,
) -> float:
    if costs is None:
        costs = EditCosts()

    max_len = max(len1, len2)
    if max_len == 0:
        return 1.0

    max_possible = max_len * costs.max_cost()
    if max_possible == 0:
        return 1.0

    return max(0.0, 1.0 - (distance / max_possible))


if __name__ == "__main__":
    s1 = "kitten"
    s2 = "sitting"

    print(f"字符串1: {s1}")
    print(f"字符串2: {s2}")

    try:
        ham_dist = hamming_distance(s1, s2)
        print(f"汉明距离: {ham_dist}")
    except ValueError as e:
        print(f"汉明距离: 无法计算 - {e}")

    lev_dist = levenshtein_distance(s1, s2)
    print(f"Levenshtein编辑距离: {lev_dist}")

    print("\n--- 测试等长字符串 ---")
    s3 = "abcde"
    s4 = "abxde"
    print(f"字符串1: {s3}")
    print(f"字符串2: {s4}")
    print(f"汉明距离: {hamming_distance(s3, s4)}")
    print(f"Levenshtein编辑距离: {levenshtein_distance(s3, s4)}")

    print("\n--- 测试超长字符串限制 ---")
    long_str = "a" * (MAX_STRING_LENGTH + 1)
    try:
        levenshtein_distance(long_str, "b")
    except ValueError as e:
        print(f"超长字符串保护生效: {e}")

    print("\n--- 测试接近上限的字符串 ---")
    import time
    n = 5000
    a = "a" * n
    b = "b" * n
    t0 = time.perf_counter()
    result = levenshtein_distance(a, b, max_length=n)
    t1 = time.perf_counter()
    print(f"两个 {n} 字符字符串的编辑距离: {result}, 耗时: {t1 - t0:.3f}s")

    print("\n--- 测试完整功能版 (带路径回溯) ---")
    result = levenshtein_full("kitten", "sitting")
    print(f"距离: {result.distance}")
    print(f"相似度: {result.similarity:.4f}")
    print("编辑操作序列:")
    for op in result.operations:
        print(f"  {op}")

    print("\n--- 测试自定义代价 (替换代价更高) ---")
    high_sub_costs = EditCosts(insert=1.0, delete=1.0, substitute=10.0)
    result2 = levenshtein_full("kitten", "sitting", costs=high_sub_costs)
    print(f"距离 (替换代价10): {result2.distance}")
    print(f"相似度: {result2.similarity:.4f}")
    print("编辑操作序列:")
    for op in result2.operations:
        print(f"  {op}")

    print("\n--- 测试完全相同字符串 ---")
    result3 = levenshtein_full("hello", "hello")
    print(f"距离: {result3.distance}")
    print(f"相似度: {result3.similarity:.4f}")

    print("\n--- 测试完全不同字符串 ---")
    result4 = levenshtein_full("abc", "xyz")
    print(f"距离: {result4.distance}")
    print(f"相似度: {result4.similarity:.4f}")
