def permute(s):
    result = []
    chars = sorted(list(s))
    n = len(chars)
    used = [False] * n
    path = []

    def backtrack():
        if len(path) == n:
            result.append(''.join(path))
            return
        for i in range(n):
            if used[i]:
                continue
            if i > 0 and chars[i] == chars[i - 1] and not used[i - 1]:
                continue
            used[i] = True
            path.append(chars[i])
            backtrack()
            path.pop()
            used[i] = False

    backtrack()
    return result


def kth_permute(s, k):
    chars = sorted(list(s))
    n = len(chars)
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i

    if k < 1 or k > fact[n]:
        raise ValueError(f"k must be between 1 and {fact[n]}")

    k -= 1
    result = []
    available = chars[:]

    for i in range(n, 0, -1):
        idx = k // fact[i - 1]
        k = k % fact[i - 1]
        result.append(available.pop(idx))

    return ''.join(result)


if __name__ == '__main__':
    test1 = "abc"
    print(f"输入: {test1}")
    print(f"全排列: {permute(test1)}")
    test2 = "aab"
    print(f"输入: {test2}")
    print(f"全排列: {permute(test2)}")
    print()
    s3 = "abc"
    for k in range(1, 7):
        print(f"第{k}个排列: {kth_permute(s3, k)}")
    print()
    s4 = "abcdefghijklmno"
    print(f"字符串长度 {len(s4)}，第1个排列: {kth_permute(s4, 1)}")
    print(f"字符串长度 {len(s4)}，最后一个排列: {kth_permute(s4, 1307674368000)}")
