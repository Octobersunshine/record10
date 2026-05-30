from fwht import fwht
from convolution import xor_convolution, or_convolution, and_convolution, subset_convolution


def test_xor_forward():
    a = [1, 2, 3, 4]
    result, orig_len = fwht(a[:], transform="xor")
    assert result == [10, -2, -4, 0], f"XOR forward failed: {result}"
    assert orig_len == 4


def test_xor_inverse():
    a = [1, 2, 3, 4]
    transformed, orig_len = fwht(a[:], transform="xor")
    restored, _ = fwht(transformed[:], inverse=True, original_len=orig_len, transform="xor")
    assert restored == [1.0, 2.0, 3.0, 4.0], f"XOR inverse failed: {restored}"


def test_or_forward():
    a = [1, 2, 3, 4]
    result, _ = fwht(a[:], transform="or")
    assert result == [1, 3, 4, 10], f"OR forward failed: {result}"


def test_or_inverse():
    a = [1, 2, 3, 4]
    transformed, orig_len = fwht(a[:], transform="or")
    restored, _ = fwht(transformed[:], inverse=True, original_len=orig_len, transform="or")
    assert restored == [1.0, 2.0, 3.0, 4.0], f"OR inverse failed: {restored}"


def test_and_forward():
    a = [1, 2, 3, 4]
    result, _ = fwht(a[:], transform="and")
    assert result == [10, 6, 7, 4], f"AND forward failed: {result}"


def test_and_inverse():
    a = [1, 2, 3, 4]
    transformed, orig_len = fwht(a[:], transform="and")
    restored, _ = fwht(transformed[:], inverse=True, original_len=orig_len, transform="and")
    assert restored == [1.0, 2.0, 3.0, 4.0], f"AND inverse failed: {restored}"


def test_invalid_transform():
    try:
        fwht([1, 2], transform="nand")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_xor_convolution():
    a = [1, 0, 0, 0]
    b = [1, 1, 1, 1]
    c = xor_convolution(a, b)
    assert c == [1.0, 1.0, 1.0, 1.0], f"XOR convolution failed: {c}"


def test_or_convolution():
    a = [0, 1, 0, 0]
    b = [0, 1, 0, 0]
    c = or_convolution(a, b)
    assert c == [0.0, 1.0, 0.0, 0.0], f"OR convolution failed: {c}"


def test_and_convolution():
    a = [0, 1, 1, 0]
    b = [0, 1, 1, 0]
    c = and_convolution(a, b)
    assert c == [2.0, 1.0, 1.0, 0.0], f"AND convolution failed: {c}"


def test_non_power_of_two_padding():
    a = [1, 2, 3]
    result, orig_len = fwht(a[:], transform="xor")
    assert len(result) == 4
    assert orig_len == 3


def test_non_power_of_two_roundtrip():
    a = [1, 2, 3, 4, 5]
    transformed, orig_len = fwht(a[:], transform="xor")
    assert len(transformed) == 8
    restored, _ = fwht(transformed[:], inverse=True, original_len=orig_len, transform="xor")
    assert len(restored) == 5
    assert restored == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_or_roundtrip_non_power_of_two():
    a = [1, 2, 3]
    transformed, orig_len = fwht(a[:], transform="or")
    restored, _ = fwht(transformed[:], inverse=True, original_len=orig_len, transform="or")
    assert restored == [1.0, 2.0, 3.0]


def test_and_roundtrip_non_power_of_two():
    a = [1, 2, 3]
    transformed, orig_len = fwht(a[:], transform="and")
    restored, _ = fwht(transformed[:], inverse=True, original_len=orig_len, transform="and")
    assert restored == [1.0, 2.0, 3.0]


def test_subset_convolution_basic():
    f = [1, 1, 1, 0]
    g = [1, 1, 1, 0]
    h = subset_convolution(f, g)
    assert h[0] == 1, f"subset_conv[0] failed: {h[0]}"
    assert h[1] == 2, f"subset_conv[1] failed: {h[1]}"
    assert h[2] == 2, f"subset_conv[2] failed: {h[2]}"
    assert h[3] == 2, f"subset_conv[3] failed: {h[3]}"


def test_subset_convolution_disjoint():
    f = [0, 1, 0, 0]
    g = [0, 0, 1, 0]
    h = subset_convolution(f, g)
    assert h[3] == 1, f"subset_conv disjoint failed: {h[3]}"
    assert h[0] == 0 and h[1] == 0 and h[2] == 0, f"subset_conv non-disjoint should be 0: {h}"


def test_subset_vs_or_convolution():
    f = [1, 1, 1, 1]
    g = [1, 1, 1, 1]
    sc = subset_convolution(f, g)
    oc = or_convolution(f[:], g[:])
    assert sc[3] == 4, f"subset_conv[3] should be 4, got {sc[3]}"
    assert oc[3] == 9.0, f"or_conv[3] should be 9, got {oc[3]}"


if __name__ == "__main__":
    test_xor_forward()
    test_xor_inverse()
    test_or_forward()
    test_or_inverse()
    test_and_forward()
    test_and_inverse()
    test_invalid_transform()
    test_xor_convolution()
    test_or_convolution()
    test_and_convolution()
    test_non_power_of_two_padding()
    test_non_power_of_two_roundtrip()
    test_or_roundtrip_non_power_of_two()
    test_and_roundtrip_non_power_of_two()
    test_subset_convolution_basic()
    test_subset_convolution_disjoint()
    test_subset_vs_or_convolution()
    print("All tests passed.")
