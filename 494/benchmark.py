import time
import cmath
import random
from fwht import fwht
from convolution import xor_convolution


def _fft(a, invert=False):
    n = len(a)
    if n == 1:
        return a
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    length = 2
    while length <= n:
        ang = 2 * cmath.pi / length * (-1 if invert else 1)
        wlen = complex(cmath.cos(ang), cmath.sin(ang))
        for i in range(0, n, length):
            w = 1 + 0j
            for j in range(i, i + length // 2):
                u = a[j]
                v = a[j + length // 2] * w
                a[j] = u + v
                a[j + length // 2] = u - v
                w *= wlen
        length <<= 1
    if invert:
        for i in range(n):
            a[i] /= n
    return a


def fft_boolean_convolution(a, b):
    n = 1
    while n < max(len(a), len(b)):
        n <<= 1
    n <<= 1
    fa = [complex(x, 0) for x in a] + [0j] * (n - len(a))
    fb = [complex(x, 0) for x in b] + [0j] * (n - len(b))
    fa = _fft(fa)
    fb = _fft(fb)
    fc = [fa[i] * fb[i] for i in range(n)]
    fc = _fft(fc, invert=True)
    return [fc[i].real for i in range(len(a))]


def benchmark(sizes):
    print("=" * 70)
    print(f"{'Size':>8} | {'FWHT (ms)':>12} | {'FFT (ms)':>12} | {'Speedup':>8}")
    print("-" * 70)

    for size in sizes:
        a = [random.randint(0, 1) for _ in range(size)]
        b = [random.randint(0, 1) for _ in range(size)]

        t0 = time.perf_counter()
        for _ in range(10):
            xor_convolution(a[:], b[:])
        t_fwht = (time.perf_counter() - t0) / 10 * 1000

        t0 = time.perf_counter()
        for _ in range(10):
            fft_boolean_convolution(a[:], b[:])
        t_fft = (time.perf_counter() - t0) / 10 * 1000

        speedup = t_fft / t_fwht if t_fwht > 0 else float("inf")
        print(f"{size:>8} | {t_fwht:>12.3f} | {t_fft:>12.3f} | {speedup:>7.2f}x")

    print("=" * 70)
    print()
    print("Key differences between FFT and FWHT for boolean convolution:")
    print("  1. FFT uses complex arithmetic (cos/sin), FWHT uses only +/- 1")
    print("  2. FFT doubles the array size for cyclic conv, FWHT operates in-place")
    print("  3. FWHT XOR convolution directly computes c[k] = sum(a[i]*b[j], i^j=k)")
    print("  4. FFT computes cyclic conv which requires 2x padding to avoid wrap-around")
    print("  5. FWHT avoids floating-point rounding issues for integer inputs")


if __name__ == "__main__":
    benchmark([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
