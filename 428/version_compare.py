import re
from typing import Tuple, List, Union, Optional, Dict

SEMVER_PATTERN = re.compile(
    r'^v?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)'
    r'(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?'
    r'(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$'
)


def is_valid_version(version: str) -> bool:
    if not isinstance(version, str):
        return False
    return bool(SEMVER_PATTERN.match(version.strip()))


def _parse_version(version: str) -> Tuple[int, int, int, Union[List, None]]:
    match = SEMVER_PATTERN.match(version.strip())
    if not match:
        raise ValueError(f"Invalid semantic version: {version}")
    major = int(match.group(1))
    minor = int(match.group(2))
    patch = int(match.group(3))
    prerelease = match.group(4)
    prerelease_parts = None
    if prerelease is not None:
        prerelease_parts = []
        for part in prerelease.split('.'):
            if part.isdigit():
                prerelease_parts.append(int(part))
            else:
                prerelease_parts.append(part)
    return (major, minor, patch, prerelease_parts)


def _compare_prerelease(a: List, b: List) -> int:
    for i in range(max(len(a), len(b))):
        if i >= len(a):
            return -1
        if i >= len(b):
            return 1
        part_a, part_b = a[i], b[i]
        if isinstance(part_a, int) and isinstance(part_b, int):
            if part_a < part_b:
                return -1
            if part_a > part_b:
                return 1
        elif isinstance(part_a, int):
            return -1
        elif isinstance(part_b, int):
            return 1
        else:
            if part_a < part_b:
                return -1
            if part_a > part_b:
                return 1
    return 0


def compare_versions(v1: str, v2: str) -> Tuple[int, bool, bool]:
    v1_valid = is_valid_version(v1)
    v2_valid = is_valid_version(v2)
    if not v1_valid or not v2_valid:
        return (0, v1_valid, v2_valid)
    parsed_v1 = _parse_version(v1)
    parsed_v2 = _parse_version(v2)
    core_v1 = (parsed_v1[0], parsed_v1[1], parsed_v1[2])
    core_v2 = (parsed_v2[0], parsed_v2[1], parsed_v2[2])
    if core_v1 < core_v2:
        return (-1, True, True)
    if core_v1 > core_v2:
        return (1, True, True)
    pre_v1, pre_v2 = parsed_v1[3], parsed_v2[3]
    if pre_v1 is None and pre_v2 is None:
        return (0, True, True)
    if pre_v1 is None:
        return (1, True, True)
    if pre_v2 is None:
        return (-1, True, True)
    result = _compare_prerelease(pre_v1, pre_v2)
    return (result, True, True)


def _compare_raw(v1: str, v2: str) -> int:
    result, _, _ = compare_versions(v1, v2)
    return result


def matches_range(version: str, range_expr: str) -> Tuple[bool, str]:
    if not is_valid_version(version):
        return (False, "invalid version")

    range_expr = range_expr.strip()

    if range_expr.startswith('~'):
        base_version = range_expr[1:].strip()
        if not is_valid_version(base_version):
            return (False, "invalid range")
        parsed = _parse_version(base_version)
        major, minor = parsed[0], parsed[1]
        lower = base_version
        upper = f"{major}.{minor + 1}.0"
        cmp_lower = _compare_raw(version, lower)
        cmp_upper = _compare_raw(version, upper)
        return (cmp_lower >= 0 and cmp_upper < 0, "ok")

    elif range_expr.startswith('^'):
        base_version = range_expr[1:].strip()
        if not is_valid_version(base_version):
            return (False, "invalid range")
        parsed = _parse_version(base_version)
        major = parsed[0]
        lower = base_version
        if major > 0:
            upper = f"{major + 1}.0.0"
        else:
            minor = parsed[1]
            if minor > 0:
                upper = f"0.{minor + 1}.0"
            else:
                patch = parsed[2]
                upper = f"0.0.{patch + 1}"
        cmp_lower = _compare_raw(version, lower)
        cmp_upper = _compare_raw(version, upper)
        return (cmp_lower >= 0 and cmp_upper < 0, "ok")

    elif range_expr.startswith('>='):
        target = range_expr[2:].strip()
        if not is_valid_version(target):
            return (False, "invalid range")
        return (_compare_raw(version, target) >= 0, "ok")

    elif range_expr.startswith('<='):
        target = range_expr[2:].strip()
        if not is_valid_version(target):
            return (False, "invalid range")
        return (_compare_raw(version, target) <= 0, "ok")

    elif range_expr.startswith('>'):
        target = range_expr[1:].strip()
        if not is_valid_version(target):
            return (False, "invalid range")
        return (_compare_raw(version, target) > 0, "ok")

    elif range_expr.startswith('<'):
        target = range_expr[1:].strip()
        if not is_valid_version(target):
            return (False, "invalid range")
        return (_compare_raw(version, target) < 0, "ok")

    elif range_expr.startswith('==') or range_expr.startswith('='):
        op_len = 2 if range_expr.startswith('==') else 1
        target = range_expr[op_len:].strip()
        if not is_valid_version(target):
            return (False, "invalid range")
        return (_compare_raw(version, target) == 0, "ok")

    elif ' - ' in range_expr:
        parts = range_expr.split(' - ', 1)
        lower, upper = parts[0].strip(), parts[1].strip()
        if not is_valid_version(lower) or not is_valid_version(upper):
            return (False, "invalid range")
        cmp_lower = _compare_raw(version, lower)
        cmp_upper = _compare_raw(version, upper)
        return (cmp_lower >= 0 and cmp_upper <= 0, "ok")

    elif is_valid_version(range_expr):
        return (_compare_raw(version, range_expr) == 0, "ok")

    else:
        return (False, "invalid range")


def _sort_key(version: str):
    parsed = _parse_version(version)
    major, minor, patch, prerelease = parsed
    if prerelease is None:
        prerelease_key = (1,)
    else:
        key_parts = []
        for part in prerelease:
            if isinstance(part, int):
                key_parts.append((0, part))
            else:
                key_parts.append((1, part))
        prerelease_key = (0, tuple(key_parts))
    return (major, minor, patch, prerelease_key)


def sort_versions(versions: List[str], reverse: bool = False) -> Tuple[List[str], List[str]]:
    valid_versions = []
    invalid_versions = []
    for v in versions:
        if is_valid_version(v):
            valid_versions.append(v)
        else:
            invalid_versions.append(v)
    sorted_versions = sorted(valid_versions, key=_sort_key, reverse=reverse)
    return (sorted_versions, invalid_versions)


def version_diff(v1: str, v2: str) -> Tuple[Optional[Dict[str, int]], Optional[str]]:
    v1_valid = is_valid_version(v1)
    v2_valid = is_valid_version(v2)
    if not v1_valid and not v2_valid:
        return (None, "both versions are invalid")
    if not v1_valid:
        return (None, "v1 is invalid")
    if not v2_valid:
        return (None, "v2 is invalid")

    parsed_v1 = _parse_version(v1)
    parsed_v2 = _parse_version(v2)

    major1, minor1, patch1 = parsed_v1[0], parsed_v1[1], parsed_v1[2]
    major2, minor2, patch2 = parsed_v2[0], parsed_v2[1], parsed_v2[2]

    cmp_result = _compare_raw(v1, v2)

    if cmp_result == 0:
        return ({
            "major_diff": 0,
            "minor_diff": 0,
            "patch_diff": 0,
            "level": "same",
            "direction": 0
        }, None)

    if cmp_result < 0:
        lower_major, lower_minor, lower_patch = major1, minor1, patch1
        upper_major, upper_minor, upper_patch = major2, minor2, patch2
        direction = -1
    else:
        lower_major, lower_minor, lower_patch = major2, minor2, patch2
        upper_major, upper_minor, upper_patch = major1, minor1, patch1
        direction = 1

    major_diff = upper_major - lower_major
    if major_diff > 0:
        minor_diff = upper_minor
        patch_diff = upper_patch
    else:
        minor_diff = upper_minor - lower_minor
        if minor_diff > 0:
            patch_diff = upper_patch
        else:
            patch_diff = upper_patch - lower_patch

    if major_diff != 0:
        level = "major"
    elif minor_diff != 0:
        level = "minor"
    elif patch_diff != 0:
        level = "patch"
    else:
        level = "prerelease"

    return ({
        "major_diff": major_diff,
        "minor_diff": minor_diff,
        "patch_diff": patch_diff,
        "level": level,
        "direction": direction
    }, None)


if __name__ == '__main__':
    test_cases = [
        ("1.0.0", "1.0.0", 0),
        ("1.0.0", "2.0.0", -1),
        ("2.0.0", "1.0.0", 1),
        ("1.0.0", "1.1.0", -1),
        ("1.0.0", "1.0.1", -1),
        ("v1.2.3", "1.2.3", 0),
        ("1.0.0-alpha", "1.0.0", -1),
        ("1.0.0", "1.0.0-alpha", 1),
        ("1.0.0-alpha", "1.0.0-beta", -1),
        ("1.0.0-alpha.1", "1.0.0-alpha", 1),
        ("1.0.0-alpha", "1.0.0-alpha.1", -1),
        ("1.0.0-alpha.1", "1.0.0-alpha.2", -1),
        ("1.0.0-alpha.beta", "1.0.0-beta", -1),
        ("1.0.0-beta", "1.0.0-beta.2", -1),
        ("1.0.0-beta.2", "1.0.0-beta.11", -1),
        ("1.0.0-1", "1.0.0-alpha", -1),
        ("1.0.0+build.1", "1.0.0+build.2", 0),
        ("1.0.0-alpha+001", "1.0.0-alpha+002", 0),
        ("1.2.3-alpha", "1.2.3", -1),
        ("1.2.3", "1.2.3-alpha", 1),
        ("1.2.3-alpha", "1.2.3-beta", -1),
        ("1.2.3-beta", "1.2.3-rc", -1),
        ("1.2.3-rc", "1.2.3", -1),
        ("1.2.3-alpha.1", "1.2.3-alpha.2", -1),
        ("1.2.3-alpha.2", "1.2.3-alpha.1", 1),
        ("1.2.3-alpha.1", "1.2.3-alpha.1", 0),
        ("1.2.3-alpha.1.beta", "1.2.3-alpha.2", -1),
        ("1.2.3-alpha.beta", "1.2.3-alpha.1", 1),
        ("1.2.3-rc.1", "1.2.3-rc.2", -1),
        ("1.2.3-beta.rc", "1.2.3-rc", -1),
        ("1.2.3-alpha.10", "1.2.3-alpha.2", 1),
        ("1.2.3-0", "1.2.3-alpha", -1),
        ("1.2.3-alpha", "1.2.3-alpha.0", -1),
        ("1.2.3-alpha.0", "1.2.3-alpha.1", -1),
    ]

    print("=== 版本比较测试 ===")
    all_passed = True
    for v1, v2, expected in test_cases:
        result, v1_valid, v2_valid = compare_versions(v1, v2)
        passed = result == expected and v1_valid and v2_valid
        all_passed = all_passed and passed
        status = "✓" if passed else "✗"
        print(f"{status} compare('{v1}', '{v2}') = {result} (expected {expected})")

    print("\n=== 版本合法性测试 ===")
    valid_versions = [
        "1.0.0", "v1.0.0", "1.0.0-alpha", "1.0.0-alpha.1",
        "1.0.0-0.3.7", "1.0.0-x.7.z.92", "1.0.0+build.1",
        "1.0.0-beta+exp.sha.5114f85",
    ]
    invalid_versions = [
        "1", "1.0", "1.0.0.0", "01.0.0", "1.01.0", "1.0.01",
        "1.0.0-alpha.", "1.0.0-01", "1.0.0-alpha+", "v",
        "version", "1.0", "1.0.0-beta!",
    ]

    for v in valid_versions:
        result = is_valid_version(v)
        status = "✓" if result else "✗"
        print(f"{status} is_valid('{v}') = {result} (expected True)")
        all_passed = all_passed and result

    for v in invalid_versions:
        result = is_valid_version(v)
        passed = not result
        status = "✓" if passed else "✗"
        print(f"{status} is_valid('{v}') = {result} (expected False)")
        all_passed = all_passed and passed

    print("\n=== 非法版本比较测试 ===")
    result, v1v, v2v = compare_versions("invalid", "1.0.0")
    print(f"compare('invalid', '1.0.0') = ({result}, v1_valid={v1v}, v2_valid={v2v})")

    print("\n=== 版本范围匹配测试 ===")
    range_tests = [
        ("1.2.3", "~1.2.3", True),
        ("1.2.5", "~1.2.3", True),
        ("1.3.0", "~1.2.3", False),
        ("1.2.2", "~1.2.3", False),
        ("1.2.3", "^1.2.3", True),
        ("1.5.0", "^1.2.3", True),
        ("2.0.0", "^1.2.3", False),
        ("0.2.3", "^0.2.3", True),
        ("0.3.0", "^0.2.3", False),
        ("0.0.5", "^0.0.5", True),
        ("0.0.6", "^0.0.5", False),
        ("1.5.0", ">=1.2.3", True),
        ("1.2.3", ">=1.2.3", True),
        ("1.0.0", ">=1.2.3", False),
        ("1.0.0", "<=1.2.3", True),
        ("1.2.3", "<=1.2.3", True),
        ("2.0.0", "<=1.2.3", False),
        ("1.5.0", ">1.2.3", True),
        ("1.2.3", ">1.2.3", False),
        ("1.0.0", "<1.2.3", True),
        ("1.2.3", "<1.2.3", False),
        ("1.2.3", "==1.2.3", True),
        ("1.2.4", "==1.2.3", False),
        ("1.2.3", "1.2.3", True),
        ("1.5.0", "1.2.0 - 2.0.0", True),
        ("2.0.0", "1.2.0 - 2.0.0", True),
        ("2.0.1", "1.2.0 - 2.0.0", False),
        ("1.2.3-alpha", "~1.2.3", False),
        ("1.2.3-alpha", ">=1.2.3-alpha", True),
        ("1.2.3", "^1.2.0", True),
        ("1.2.3-alpha", ">=1.2.0", True),
    ]

    for version, range_expr, expected in range_tests:
        result, status = matches_range(version, range_expr)
        passed = result == expected
        all_passed = all_passed and passed
        status_mark = "✓" if passed else "✗"
        print(f"{status_mark} matches_range('{version}', '{range_expr}') = {result} (expected {expected}, status={status})")

    print("\n=== 版本排序测试 ===")
    unsorted_versions = [
        "2.0.0", "1.0.0", "1.5.0", "1.2.3", "1.2.3-alpha",
        "1.2.3-beta", "1.10.0", "0.9.0", "invalid", "v1.2.4"
    ]
    sorted_asc, invalid = sort_versions(unsorted_versions)
    expected_asc = [
        "0.9.0", "1.0.0", "1.2.3-alpha", "1.2.3-beta",
        "1.2.3", "v1.2.4", "1.5.0", "1.10.0", "2.0.0"
    ]
    passed = sorted_asc == expected_asc and invalid == ["invalid"]
    all_passed = all_passed and passed
    status_mark = "✓" if passed else "✗"
    print(f"{status_mark} sort_versions(ascending)")
    print(f"  输入: {unsorted_versions}")
    print(f"  排序后: {sorted_asc}")
    print(f"  非法版本: {invalid}")

    sorted_desc, invalid = sort_versions(unsorted_versions, reverse=True)
    expected_desc = list(reversed(expected_asc))
    passed = sorted_desc == expected_desc and invalid == ["invalid"]
    all_passed = all_passed and passed
    status_mark = "✓" if passed else "✗"
    print(f"{status_mark} sort_versions(descending)")
    print(f"  排序后: {sorted_desc}")

    print("\n=== 版本差值计算测试 ===")
    diff_tests = [
        ("1.2.3", "1.2.3", {"major_diff": 0, "minor_diff": 0, "patch_diff": 0, "level": "same", "direction": 0}),
        ("1.2.3", "2.0.0", {"major_diff": 1, "minor_diff": 0, "patch_diff": 0, "level": "major", "direction": -1}),
        ("2.0.0", "1.2.3", {"major_diff": 1, "minor_diff": 0, "patch_diff": 0, "level": "major", "direction": 1}),
        ("1.2.3", "1.5.0", {"major_diff": 0, "minor_diff": 3, "patch_diff": 0, "level": "minor", "direction": -1}),
        ("1.5.0", "1.2.3", {"major_diff": 0, "minor_diff": 3, "patch_diff": 0, "level": "minor", "direction": 1}),
        ("1.2.3", "1.2.7", {"major_diff": 0, "minor_diff": 0, "patch_diff": 4, "level": "patch", "direction": -1}),
        ("1.2.7", "1.2.3", {"major_diff": 0, "minor_diff": 0, "patch_diff": 4, "level": "patch", "direction": 1}),
        ("1.2.3-alpha", "1.2.3-beta", {"major_diff": 0, "minor_diff": 0, "patch_diff": 0, "level": "prerelease", "direction": -1}),
        ("1.2.3", "1.2.3-alpha", {"major_diff": 0, "minor_diff": 0, "patch_diff": 0, "level": "prerelease", "direction": 1}),
    ]

    for v1, v2, expected in diff_tests:
        result, error = version_diff(v1, v2)
        passed = error is None and result is not None and all(result.get(k) == v for k, v in expected.items())
        all_passed = all_passed and passed
        status_mark = "✓" if passed else "✗"
        print(f"{status_mark} version_diff('{v1}', '{v2}')")
        print(f"  结果: {result}")
        print(f"  期望: {expected}")
        if error:
            print(f"  错误: {error}")

    print("\n=== 版本差值错误处理测试 ===")
    result, error = version_diff("invalid", "1.0.0")
    passed = result is None and error == "v1 is invalid"
    all_passed = all_passed and passed
    status_mark = "✓" if passed else "✗"
    print(f"{status_mark} version_diff('invalid', '1.0.0'): error='{error}'")

    print(f"\n{'所有测试通过!' if all_passed else '存在测试失败!'}")
