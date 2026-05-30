import numpy as np
from math import comb, factorial
from itertools import permutations
from scipy import stats
from collections import Counter


def frequency_test(sequence):
    n = len(sequence)
    unique_values = sorted(set(sequence))

    if len(unique_values) == 2 and set(unique_values) <= {0, 1}:
        n1 = sum(sequence)
        n0 = n - n1
        s = (n1 - n0) ** 2 / n
        p_value = 1 - stats.chi2.cdf(s, df=1)
        return {"statistic": s, "p_value": p_value}

    observed = Counter(sequence)
    expected = n / len(unique_values)
    chi2 = sum((observed.get(v, 0) - expected) ** 2 / expected for v in unique_values)
    df = len(unique_values) - 1
    p_value = 1 - stats.chi2.cdf(chi2, df=df)
    return {"statistic": chi2, "p_value": p_value}


def _runs_exact_prob(r, n0, n1):
    n = n0 + n1
    if r % 2 == 0:
        k = r // 2
        numer = 2 * comb(n1 - 1, k - 1) * comb(n0 - 1, k - 1)
    else:
        k = (r - 1) // 2
        numer = (comb(n1 - 1, k) * comb(n0 - 1, k - 1)
                 + comb(n1 - 1, k - 1) * comb(n0 - 1, k))
    denom = comb(n, n1)
    return numer / denom if denom > 0 else 0.0


def _exact_runs_p_value(n0, n1, r_obs):
    r_min = 2
    r_max = 2 * min(n0, n1) + (1 if n0 != n1 else 0)
    dist = {}
    for r in range(r_min, r_max + 1):
        dist[r] = _runs_exact_prob(r, n0, n1)
    p_obs = dist.get(r_obs, 0.0)
    p_value = sum(p for p in dist.values() if p <= p_obs + 1e-12)
    return min(p_value, 1.0)


def _monte_carlo_runs_p_value(n0, n1, r_obs, n_simulations=10000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    base = np.array([0] * n0 + [1] * n1)
    count = 0
    for _ in range(n_simulations):
        perm = rng.permutation(base)
        runs = 1
        for i in range(1, len(perm)):
            if perm[i] != perm[i - 1]:
                runs += 1
        if runs == r_obs:
            count += 1
        else:
            p_r = _runs_exact_prob(runs, n0, n1)
            p_obs = _runs_exact_prob(r_obs, n0, n1)
            if p_r <= p_obs + 1e-12:
                count += 1
    return min(count / n_simulations, 1.0)


def runs_test(sequence, method="auto", n_simulations=10000):
    n = len(sequence)
    unique_values = sorted(set(sequence))

    if len(unique_values) == 2 and set(unique_values) <= {0, 1}:
        n1 = sum(sequence)
        n0 = n - n1
    else:
        median = np.median(sequence)
        binary = [1 if x >= median else 0 for x in sequence]
        n1 = sum(binary)
        n0 = n - n1
        sequence = binary

    if n1 == 0 or n0 == 0:
        return {"statistic": 0.0, "p_value": 1.0, "method": "degenerate"}

    runs = 1
    for i in range(1, n):
        if sequence[i] != sequence[i - 1]:
            runs += 1

    if method == "auto":
        if n < 20:
            method = "exact"
        else:
            method = "asymptotic"

    if method == "exact":
        p_value = _exact_runs_p_value(n0, n1, runs)
        return {"statistic": runs, "p_value": p_value, "method": "exact"}

    if method == "monte_carlo":
        p_value = _monte_carlo_runs_p_value(n0, n1, runs, n_simulations)
        return {"statistic": runs, "p_value": p_value, "method": "monte_carlo"}

    expected_runs = (2 * n0 * n1) / n + 1
    variance_runs = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n * n * (n - 1))
    z = (runs - expected_runs) / np.sqrt(variance_runs) if variance_runs > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"statistic": z, "p_value": p_value, "method": "asymptotic"}


def serial_correlation_test(sequence, lag=1):
    n = len(sequence)
    if n <= lag + 1:
        return {"statistic": 0.0, "p_value": 1.0}

    x = np.array(sequence[: n - lag], dtype=float)
    y = np.array(sequence[lag:], dtype=float)

    r, p_value = stats.pearsonr(x, y)
    return {"statistic": r, "p_value": p_value}


def birthday_spacing_test(sequence, n_birthday_samples=512, n_days=2 ** 24, n_trials=10):
    observed_repeats = []

    for t in range(n_trials):
        start = t * n_birthday_samples
        end = start + n_birthday_samples
        if end > len(sequence):
            break
        birthdays = sorted(sequence[start:end])
        spacings = []
        for i in range(1, len(birthdays)):
            spacings.append(birthdays[i] - birthdays[i - 1])
        spacings.append(n_days - birthdays[-1] + birthdays[0])
        spacing_counts = Counter(spacings)
        num_repeats = sum(c - 1 for c in spacing_counts.values() if c > 1)
        observed_repeats.append(num_repeats)

    if not observed_repeats:
        return {"statistic": 0.0, "p_value": 1.0, "method": "insufficient_data"}

    lambda_param = n_birthday_samples ** 3 / (4 * n_days)
    expected = {k: len(observed_repeats) * stats.poisson.pmf(k, lambda_param)
                for k in range(max(observed_repeats) + 2)}

    observed_dist = Counter(observed_repeats)
    all_keys = sorted(set(list(observed_dist.keys()) + list(expected.keys())))

    while all_keys and expected.get(all_keys[-1], 0) < 0.5:
        last = all_keys.pop()
        if all_keys:
            expected[all_keys[-1]] = expected.get(all_keys[-1], 0) + expected.get(last, 0)
            observed_dist[all_keys[-1]] = observed_dist.get(all_keys[-1], 0) + observed_dist.get(last, 0)
        else:
            all_keys = [0]

    chi2 = 0.0
    for k in all_keys:
        obs = observed_dist.get(k, 0)
        exp = expected.get(k, 0.001)
        chi2 += (obs - exp) ** 2 / exp

    df = max(len(all_keys) - 1 - 1, 1)
    p_value = 1 - stats.chi2.cdf(chi2, df=df)
    return {"statistic": chi2, "p_value": p_value, "method": "poisson_chi2",
            "observed_repeats": observed_repeats, "lambda": lambda_param}


def overlapping_permutation_test(sequence, t=3, n_bins=None):
    if n_bins is None:
        n_bins = factorial(t)

    perm_list = list(permutations(range(t)))
    perm_index = {p: i for i, p in enumerate(perm_list)}

    n = len(sequence)
    counts = np.zeros(n_bins, dtype=int)

    total_windows = 0
    for i in range(n - t + 1):
        window = sequence[i: i + t]
        rank_tuple = tuple(np.argsort(window).tolist())
        if rank_tuple in perm_index:
            counts[perm_index[rank_tuple]] += 1
        total_windows += 1

    if total_windows == 0:
        return {"statistic": 0.0, "p_value": 1.0, "method": "insufficient_data"}

    expected = total_windows / n_bins
    chi2 = np.sum((counts - expected) ** 2 / expected)
    df = n_bins - 1
    p_value = 1 - stats.chi2.cdf(chi2, df=df)
    return {"statistic": chi2, "p_value": p_value, "method": "chi2",
            "counts": counts.tolist(), "expected": expected}


def spectral_test(sequence, n_bits=1024):
    n = len(sequence)
    if n < n_bits:
        n_bits = n
    if n_bits < 32:
        return {"statistic": 0.0, "p_value": 1.0, "method": "insufficient_data"}

    bits = np.array(sequence[:n_bits], dtype=float)
    bits = 2.0 * bits - 1.0

    fft_result = np.fft.fft(bits)
    magnitudes = np.abs(fft_result[:n_bits // 2])

    T = np.sqrt(n_bits * np.log(1.0 / 0.05))
    n_below = int(np.sum(magnitudes < T))
    n_freq = n_bits // 2
    n_above = n_freq - n_below

    N_0 = 0.95 * n_freq
    d = (n_below - N_0) / np.sqrt(n_freq * 0.95 * 0.05)
    p_value = float(stats.norm.sf(abs(d)) * 2)

    return {"statistic": float(d), "p_value": p_value, "method": "nist_fft",
            "n_peaks": n_above, "expected_peaks": 0.05 * n_freq,
            "threshold": float(T)}


class Xorshift128:
    def __init__(self, seed=None):
        if seed is None:
            seed = np.random.randint(1, 2 ** 32)
        self.state = np.array([
            (seed ^ 0xBAD5EED) & 0xFFFFFFFF,
            (seed * 1103515245 + 12345) & 0xFFFFFFFF,
            (seed * 1664525 + 1013904223) & 0xFFFFFFFF,
            (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFF,
        ], dtype=np.uint64)
        for _ in range(20):
            self._next()

    def _next(self):
        x = self.state[3]
        x ^= (x << 11) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 8)
        x ^= (x << 19) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 58)
        self.state[3] = self.state[2]
        self.state[2] = self.state[1]
        self.state[1] = self.state[0]
        self.state[0] = x
        return int(x & 0xFFFFFFFFFFFFFFFF)

    def random_integer(self):
        return self._next()

    def generate_integers(self, n, low=0, high=None):
        if high is None:
            high = low + 1
        span = high - low
        result = []
        for _ in range(n):
            val = self.random_integer() % span + low
            result.append(val)
        return result

    def generate_uniform(self, n):
        result = []
        for _ in range(n):
            result.append(self.random_integer() / (2 ** 64))
        return result


def create_rng(generator_type="mt19937", seed=42):
    if generator_type == "mt19937":
        return np.random.Generator(np.random.MT19937(seed))
    elif generator_type == "pcg64":
        return np.random.Generator(np.random.PCG64(seed))
    elif generator_type == "xorshift128":
        return Xorshift128(seed)
    else:
        raise ValueError(f"Unknown generator: {generator_type}")


def generate_test_sequences(generator_type="mt19937", seed=42, n_int=20000,
                           n_float=20000, int_high=2 ** 24):
    rng = create_rng(generator_type, seed)

    if isinstance(rng, Xorshift128):
        int_seq = rng.generate_integers(n_int, low=0, high=int_high)
        float_seq = rng.generate_uniform(n_float)
        bit_seq = [x & 1 for x in rng.generate_integers(n_int, low=0, high=2)]
    else:
        int_seq = rng.integers(0, int_high, size=n_int).tolist()
        float_seq = rng.random(n_float).tolist()
        bit_seq = rng.integers(0, 2, size=n_int).tolist()

    return int_seq, float_seq, bit_seq


def test_random_quality(sequence, lag=1, runs_method="auto", n_simulations=10000):
    freq = frequency_test(sequence)
    runs = runs_test(sequence, method=runs_method, n_simulations=n_simulations)
    corr = serial_correlation_test(sequence, lag=lag)

    results = {
        "frequency_test": {
            "statistic": freq["statistic"],
            "p_value": freq["p_value"],
            "passed": freq["p_value"] > 0.05,
        },
        "runs_test": {
            "statistic": runs["statistic"],
            "p_value": runs["p_value"],
            "passed": runs["p_value"] > 0.05,
            "method": runs["method"],
        },
        "serial_correlation_test": {
            "statistic": corr["statistic"],
            "p_value": corr["p_value"],
            "passed": corr["p_value"] > 0.05,
        },
    }
    return results


def test_advanced_quality(int_seq, float_seq, bit_seq):
    results = {}

    results["birthday_spacing_test"] = {
        "statistic": None,
        "p_value": None,
        "passed": None,
    }
    try:
        bday = birthday_spacing_test(int_seq, n_birthday_samples=512,
                                     n_days=2 ** 24, n_trials=10)
        results["birthday_spacing_test"] = {
            "statistic": bday["statistic"],
            "p_value": bday["p_value"],
            "passed": bday["p_value"] > 0.05,
            "method": bday.get("method", ""),
        }
    except Exception as e:
        results["birthday_spacing_test"]["error"] = str(e)

    results["overlapping_permutation_test"] = {
        "statistic": None,
        "p_value": None,
        "passed": None,
    }
    try:
        operm = overlapping_permutation_test(float_seq[:5000], t=3)
        results["overlapping_permutation_test"] = {
            "statistic": operm["statistic"],
            "p_value": operm["p_value"],
            "passed": operm["p_value"] > 0.05,
            "method": operm.get("method", ""),
        }
    except Exception as e:
        results["overlapping_permutation_test"]["error"] = str(e)

    results["spectral_test"] = {
        "statistic": None,
        "p_value": None,
        "passed": None,
    }
    try:
        spec = spectral_test(bit_seq[:1024])
        results["spectral_test"] = {
            "statistic": spec["statistic"],
            "p_value": spec["p_value"],
            "passed": spec["p_value"] > 0.05,
            "method": spec.get("method", ""),
            "n_peaks": spec.get("n_peaks"),
        }
    except Exception as e:
        results["spectral_test"]["error"] = str(e)

    return results


def compare_generators(generators=None, seed=42, n_int=20000, n_float=20000):
    if generators is None:
        generators = ["mt19937", "pcg64", "xorshift128"]

    all_results = {}
    for gen_type in generators:
        int_seq, float_seq, bit_seq = generate_test_sequences(
            gen_type, seed=seed, n_int=n_int, n_float=n_float
        )

        basic = test_random_quality(bit_seq, lag=1)
        advanced = test_advanced_quality(int_seq, float_seq, bit_seq)

        all_results[gen_type] = {
            "basic": basic,
            "advanced": advanced,
        }

    return all_results


def print_comparison_report(all_results):
    test_names_basic = ["frequency_test", "runs_test", "serial_correlation_test"]
    test_names_advanced = ["birthday_spacing_test", "overlapping_permutation_test",
                           "spectral_test"]
    test_names = test_names_basic + test_names_advanced
    generators = list(all_results.keys())

    col_width = 14
    header = f"{'测试项目':<30}"
    for gen in generators:
        header += f" {gen:>{col_width}}"
    separator = "-" * len(header)

    print("\n" + "=" * len(header))
    print("随机数生成器质量对比报告")
    print("=" * len(header))

    print("\n▶ p值汇总 (p > 0.05 为通过)")
    print(separator)
    print(header)
    print(separator)

    for test_name in test_names:
        row = f"{test_name:<30}"
        for gen in generators:
            if test_name in test_names_basic:
                data = all_results[gen]["basic"].get(test_name, {})
            else:
                data = all_results[gen]["advanced"].get(test_name, {})

            p_val = data.get("p_value")
            if p_val is not None:
                row += f" {p_val:>{col_width}.4f}"
            else:
                row += f" {'N/A':>{col_width}}"
        print(row)

    print(separator)

    print("\n▶ 通过/未通过汇总")
    print(separator)
    print(header)
    print(separator)

    for test_name in test_names:
        row = f"{test_name:<30}"
        for gen in generators:
            if test_name in test_names_basic:
                data = all_results[gen]["basic"].get(test_name, {})
            else:
                data = all_results[gen]["advanced"].get(test_name, {})

            passed = data.get("passed")
            if passed is not None and bool(passed):
                row += f" {'PASS':>{col_width}}"
            elif passed is not None and not bool(passed):
                row += f" {'FAIL':>{col_width}}"
            else:
                row += f" {'N/A':>{col_width}}"
        print(row)

    print(separator)

    print("\n▶ 统计量汇总")
    print(separator)
    print(header)
    print(separator)

    for test_name in test_names:
        row = f"{test_name:<30}"
        for gen in generators:
            if test_name in test_names_basic:
                data = all_results[gen]["basic"].get(test_name, {})
            else:
                data = all_results[gen]["advanced"].get(test_name, {})

            stat = data.get("statistic")
            if stat is not None:
                row += f" {stat:>{col_width}.4f}"
            else:
                row += f" {'N/A':>{col_width}}"
        print(row)

    print(separator)

    print("\n▶ 综合评分")
    print(separator)
    for gen in generators:
        total = 0
        passed = 0
        for test_name in test_names:
            if test_name in test_names_basic:
                data = all_results[gen]["basic"].get(test_name, {})
            else:
                data = all_results[gen]["advanced"].get(test_name, {})
            if data.get("passed") is not None:
                total += 1
                if bool(data["passed"]):
                    passed += 1
        score = f"{passed}/{total}" if total > 0 else "N/A"
        bar = "█" * passed + "░" * (total - passed)
        print(f"  {gen:<14} {bar} {score}")

    print(separator)
    print()


if __name__ == "__main__":
    results = compare_generators(
        generators=["mt19937", "pcg64", "xorshift128"],
        seed=42,
        n_int=20000,
        n_float=20000,
    )
    print_comparison_report(results)
