import math


def derangement(n):
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    if n == 0:
        return 1
    if n == 1:
        return 0
    
    a, b = 1, 0
    for i in range(2, n + 1):
        a, b = b, (i - 1) * (a + b)
    
    return b


def derangement_approx(n):
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    if n == 0:
        return 1.0
    return math.factorial(n) / math.e


def partial_derangement(n, k):
    if n < 0 or k < 0 or k > n:
        raise ValueError("n and k must be non-negative integers with k <= n")
    if k == n:
        return 1
    comb = math.comb(n, k)
    return comb * derangement(n - k)


def derangement_probability():
    return 1.0 / math.e


if __name__ == "__main__":
    for n in range(11):
        exact = derangement(n)
        approx = derangement_approx(n)
        print(f"!{n} = {exact} (approx: {approx:.6f})")
    
    print("\nPartial derangements (n=5):")
    for k in range(6):
        print(f"  k={k}: {partial_derangement(5, k)}")
    
    print(f"\nProbability of derangement: {derangement_probability():.6f} (~1/e)")
