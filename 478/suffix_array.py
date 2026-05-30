from typing import List, Tuple, Dict

MAX_STRING_LENGTH = 10000


def build_sa(s: str) -> List[int]:
    n = len(s)
    if n == 0:
        return []
    if n == 1:
        return [0]

    if n > MAX_STRING_LENGTH:
        raise ValueError(f"String length {n} exceeds maximum allowed length {MAX_STRING_LENGTH}")

    sa = list(range(n))
    rank = [ord(c) for c in s]
    tmp = [0] * n
    k = 1

    while k < n:
        def get_key(i: int) -> tuple:
            second = rank[i + k] if i + k < n else -1
            return (rank[i], second)

        sa.sort(key=get_key)

        tmp[sa[0]] = 0
        for i in range(1, n):
            tmp[sa[i]] = tmp[sa[i - 1]]
            if get_key(sa[i - 1]) < get_key(sa[i]):
                tmp[sa[i]] += 1

        rank = tmp[:]
        if rank[sa[n - 1]] == n - 1:
            break
        k *= 2

    return sa


def build_rank(sa: List[int]) -> List[int]:
    n = len(sa)
    rank = [0] * n
    for i in range(n):
        rank[sa[i]] = i
    return rank


def build_lcp(s: str, sa: List[int], rank: List[int]) -> List[int]:
    n = len(s)
    if n <= 1:
        return []
    lcp = [0] * (n - 1)
    h = 0
    for i in range(n):
        if rank[i] == n - 1:
            h = 0
            continue
        j = sa[rank[i] + 1]
        while i + h < n and j + h < n and s[i + h] == s[j + h]:
            h += 1
        lcp[rank[i]] = h
        if h > 0:
            h -= 1
    return lcp


def pattern_match(s: str, pattern: str, sa: List[int]) -> Tuple[int, int]:
    n = len(s)
    m = len(pattern)
    if m == 0 or m > n:
        return (-1, -1)

    def compare(pos: int, p: str) -> int:
        k = min(n - pos, len(p))
        for i in range(k):
            if s[pos + i] < p[i]:
                return -1
            elif s[pos + i] > p[i]:
                return 1
        if n - pos < len(p):
            return -1
        return 0

    left, right = 0, n
    while right - left > 1:
        mid = (left + right) // 2
        if compare(sa[mid], pattern) < 0:
            left = mid
        else:
            right = mid
    start = right
    if start >= n or compare(sa[start], pattern) != 0:
        return (-1, -1)

    left, right = 0, n
    while right - left > 1:
        mid = (left + right) // 2
        if compare(sa[mid], pattern) <= 0:
            left = mid
        else:
            right = mid
    end = left + 1

    return (start, end)


def get_match_positions(sa: List[int], match_range: Tuple[int, int]) -> List[int]:
    if match_range[0] == -1:
        return []
    return sorted(sa[match_range[0]:match_range[1]])


def find_repeated_substrings(s: str, sa: List[int], lcp: List[int], min_len: int = 2) -> List[Tuple[str, int, List[int]]]:
    n = len(s)
    if n <= 1 or min_len < 2:
        return []

    visited = [False] * (n - 1)
    results = []

    for i in range(n - 1):
        if visited[i] or lcp[i] < min_len:
            continue

        group = [sa[i], sa[i + 1]]
        current_lcp = lcp[i]
        j = i + 1
        while j < n - 1 and lcp[j] >= min_len:
            current_lcp = min(current_lcp, lcp[j])
            group.append(sa[j + 1])
            visited[j] = True
            j += 1

        max_len = current_lcp
        if max_len >= min_len:
            substr = s[sa[i]:sa[i] + max_len]
            positions = sorted(group)
            results.append((substr, max_len, positions))

    seen = {}
    unique_results = []
    for substr, length, positions in results:
        key = (substr, length)
        if key not in seen:
            seen[key] = set()
        for pos in positions:
            seen[key].add(pos)

    for (substr, length), positions in seen.items():
        unique_results.append((substr, length, sorted(positions)))

    unique_results.sort(key=lambda x: (-x[1], x[0]))
    return unique_results


def longest_repeated_substring(s: str, sa: List[int], lcp: List[int], min_len: int = 2) -> List[Tuple[str, int, List[int]]]:
    n = len(s)
    if n <= 1:
        return []

    max_lcp = max(lcp) if lcp else 0
    if max_lcp < min_len:
        return []

    results = []
    i = 0
    while i < len(lcp):
        if lcp[i] == max_lcp:
            group_positions = set()
            group_positions.add(sa[i])
            group_positions.add(sa[i + 1])

            j = i - 1
            while j >= 0 and lcp[j] >= max_lcp:
                group_positions.add(sa[j])
                j -= 1

            j = i + 1
            while j < len(lcp) and lcp[j] >= max_lcp:
                group_positions.add(sa[j + 1])
                j += 1

            substr = s[sa[i]:sa[i] + max_lcp]
            positions = sorted(group_positions)
            results.append((substr, max_lcp, positions))
            i = j
        else:
            i += 1

    seen = {}
    for substr, length, positions in results:
        key = substr
        if key not in seen:
            seen[key] = set()
        for pos in positions:
            seen[key].add(pos)

    final = []
    for substr, positions in seen.items():
        final.append((substr, max_lcp, sorted(positions)))

    final.sort(key=lambda x: x[0])
    return final


def longest_common_substring(s1: str, s2: str) -> List[Tuple[str, int, List[Tuple[int, int]]]]:
    if not s1 or not s2:
        return []

    total_len = len(s1) + len(s2) + 1
    if total_len > MAX_STRING_LENGTH:
        raise ValueError(f"Combined string length {total_len} exceeds maximum allowed length {MAX_STRING_LENGTH}")

    separator = chr(0)
    combined = s1 + separator + s2
    n = len(combined)
    n1 = len(s1)

    sa = build_sa(combined)
    rank = build_rank(sa)
    lcp = build_lcp(combined, sa, rank)

    max_lcp = 0
    for i in range(len(lcp)):
        left_from_s1 = sa[i] < n1
        right_from_s1 = sa[i + 1] < n1
        if left_from_s1 != right_from_s1:
            if lcp[i] > max_lcp:
                max_lcp = lcp[i]

    if max_lcp == 0:
        return []

    results = []
    visited = [False] * len(lcp)

    for i in range(len(lcp)):
        if visited[i] or lcp[i] != max_lcp:
            continue
        left_from_s1 = sa[i] < n1
        right_from_s1 = sa[i + 1] < n1
        if left_from_s1 == right_from_s1:
            continue

        group_s1 = set()
        group_s2 = set()

        j = i
        while j >= 0 and lcp[j] >= max_lcp:
            visited[j] = True
            if sa[j] < n1:
                group_s1.add(sa[j])
            elif sa[j] > n1:
                group_s2.add(sa[j] - n1 - 1)
            if sa[j + 1] < n1:
                group_s1.add(sa[j + 1])
            elif sa[j + 1] > n1:
                group_s2.add(sa[j + 1] - n1 - 1)
            j -= 1

        j = i + 1
        while j < len(lcp) and lcp[j] >= max_lcp:
            visited[j] = True
            if sa[j] < n1:
                group_s1.add(sa[j])
            elif sa[j] > n1:
                group_s2.add(sa[j] - n1 - 1)
            if sa[j + 1] < n1:
                group_s1.add(sa[j + 1])
            elif sa[j + 1] > n1:
                group_s2.add(sa[j + 1] - n1 - 1)
            j += 1

        substr = combined[sa[i]:sa[i] + max_lcp]
        pairs = []
        for p1 in sorted(group_s1):
            for p2 in sorted(group_s2):
                pairs.append((p1, p2))
        results.append((substr, max_lcp, pairs))

    seen = {}
    for substr, length, pairs in results:
        if substr not in seen:
            seen[substr] = []
        seen[substr].extend(pairs)

    final = []
    for substr, pairs in seen.items():
        unique_pairs = list(dict.fromkeys(pairs))
        final.append((substr, max_lcp, unique_pairs))

    final.sort(key=lambda x: x[0])
    return final


def sa_analysis(s: str) -> Dict:
    sa = build_sa(s)
    rank = build_rank(sa)
    lcp = build_lcp(s, sa, rank)
    return {
        'string': s,
        'sa': sa,
        'rank': rank,
        'lcp': lcp
    }


if __name__ == '__main__':
    test_cases = [
        "banana",
        "mississippi",
        "abcabcab",
        "ababab",
        "hello world",
    ]

    for s in test_cases:
        print(f"\n{'=' * 60}")
        print(f"String: '{s}'")
        print(f"{'=' * 60}")

        result = sa_analysis(s)
        sa = result['sa']
        rank = result['rank']
        lcp = result['lcp']

        print(f"\nSuffix Array (SA): {sa}")
        print(f"Rank Array: {rank}")
        print(f"LCP Array: {lcp}")

        print(f"\nSuffixes in sorted order:")
        for i, idx in enumerate(sa):
            lcp_str = f" (LCP={lcp[i - 1]})" if i > 0 else ""
            print(f"  {i:2d}: {idx:2d} -> '{s[idx:]}'{lcp_str}")

        print(f"\nRank for each position:")
        for i, r in enumerate(rank):
            print(f"  s[{i}]='{s[i:]}' -> rank {r}")

        print(f"\nAdjacent LCP values:")
        for i, v in enumerate(lcp):
            print(f"  LCP[{i}] = {v}: between '{s[sa[i]:]}' and '{s[sa[i + 1]:]}'")

        print(f"\nPattern matching test:")
        patterns = {
            "banana": ["ana", "ban", "na"],
            "mississippi": ["issi", "sip", "miss"],
            "abcabcab": ["abc", "bca", "cab"],
            "ababab": ["aba", "bab", "abab"],
            "hello world": ["lo", "wor", "hello"],
        }
        for p in patterns.get(s, []):
            match_range = pattern_match(s, p, sa)
            positions = get_match_positions(sa, match_range)
            print(f"  Pattern '{p}': range={match_range}, positions={positions}")

        print(f"\nRepeated substrings (min_len=2):")
        repeats = find_repeated_substrings(s, sa, lcp, min_len=2)
        for substr, length, positions in repeats[:5]:
            print(f"  '{substr}' (len={length}): positions {positions}")

        print(f"\nLongest repeated substring:")
        lrs = longest_repeated_substring(s, sa, lcp)
        if lrs:
            for substr, length, positions in lrs:
                print(f"  '{substr}' (len={length}): positions {positions}")
        else:
            print(f"  (none)")

    print(f"\n{'=' * 60}")
    print("Longest common substring test")
    print(f"{'=' * 60}")
    common_tests = [
        ("abcdefg", "xbcdeyz"),
        ("banana", "corona"),
        ("abracadabra", "cadabra"),
        ("abcabc", "xyzabc"),
        ("hello", "world"),
        ("AAAAAA", "AA"),
    ]
    for s1, s2 in common_tests:
        result = longest_common_substring(s1, s2)
        print(f"\n  '{s1}' & '{s2}':")
        if result:
            for substr, length, pairs in result:
                print(f"    '{substr}' (len={length}): s1_pos/s2_pos = {pairs}")
        else:
            print(f"    (no common substring)")

    print(f"\n{'=' * 60}")
    print("Performance test")
    print(f"{'=' * 60}")
    import time

    for length in [1000, 5000, 10000]:
        s_long = "ab" * (length // 2)
        start = time.time()
        sa_long = build_sa(s_long)
        rank_long = build_rank(sa_long)
        lcp_long = build_lcp(s_long, sa_long, rank_long)
        elapsed = time.time() - start
        print(f"  Length {length}: SA+LCP built in {elapsed:.3f}s")

    print(f"\n{'=' * 60}")
    print("Max length limit test")
    print(f"{'=' * 60}")
    try:
        s_too_long = "a" * (MAX_STRING_LENGTH + 1)
        build_sa(s_too_long)
        print("  ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"  OK: {e}")
