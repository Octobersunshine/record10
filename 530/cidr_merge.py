import ipaddress
from typing import List, Union, Tuple


def cidr_contains(outer: str, inner: str) -> bool:
    try:
        outer_net = ipaddress.ip_network(outer, strict=True)
        inner_net = ipaddress.ip_network(inner, strict=True)
    except (ipaddress.AddressValueError, ipaddress.NetmaskValueError, ValueError) as e:
        raise ValueError(f"无效的CIDR格式 - {str(e)}") from e
    if outer_net.version != inner_net.version:
        return False
    return inner_net.subnet_of(outer_net)


def _cidr_to_range(cidr: str) -> Tuple[int, int, int]:
    net = ipaddress.ip_network(cidr, strict=True)
    version = net.version
    return (int(net.network_address), int(net.broadcast_address), version)


def _int_to_ip(val: int, version: int) -> str:
    if version == 4:
        return str(ipaddress.IPv4Address(val))
    return str(ipaddress.IPv6Address(val))


def _range_to_cidrs(start: int, end: int, version: int) -> List[str]:
    result = []
    total_bits = 32 if version == 4 else 128
    while start <= end:
        trailing_zeros = 0
        if start == 0:
            trailing_zeros = total_bits
        else:
            trailing_zeros = (start & -start).bit_length() - 1
        remaining = end - start + 1
        size_bits = remaining.bit_length() - 1
        if (1 << size_bits) > remaining:
            size_bits -= 1
        host_bits = min(trailing_zeros, size_bits)
        prefix_len = total_bits - host_bits
        block_size = 1 << host_bits
        result.append(f"{_int_to_ip(start, version)}/{prefix_len}")
        start += block_size
    return result


def merge_cidrs(cidrs: List[str]) -> Tuple[List[str], int]:
    if not cidrs:
        return [], 0
    ranges = []
    for cidr in cidrs:
        try:
            ranges.append(_cidr_to_range(cidr))
        except (ipaddress.AddressValueError, ipaddress.NetmaskValueError, ValueError) as e:
            raise ValueError(f"无效的CIDR格式: '{cidr}' - {str(e)}") from e
    versions = {r[2] for r in ranges}
    if len(versions) > 1:
        raise ValueError("不能同时合并IPv4和IPv6的CIDR")
    version = next(iter(versions))
    ranges.sort()
    merged = []
    for start, end, ver in ranges:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end), ver)
        else:
            merged.append((start, end, ver))
    result = []
    total_ips = 0
    for start, end, ver in merged:
        result.extend(_range_to_cidrs(start, end, ver))
        total_ips += end - start + 1
    return result, total_ips


if __name__ == "__main__":
    print("=== 1. 包含关系测试 ===")
    contains_tests = [
        ("192.168.0.0/16", "192.168.1.0/24", True),
        ("10.0.0.0/8", "10.0.0.0/16", True),
        ("192.168.1.0/24", "192.168.2.0/24", False),
        ("172.16.0.0/12", "172.16.5.0/24", True),
        ("2001:db8::/32", "2001:db8::/64", True),
        ("2001:db8::/32", "2002:db8::/32", False),
        ("192.168.0.0/24", "2001:db8::/32", False),
    ]
    for outer, inner, expected in contains_tests:
        try:
            result = cidr_contains(outer, inner)
            status = "✓" if result == expected else "✗"
            print(f"{status} {outer} 包含 {inner}? {result} (期望: {expected})")
        except ValueError as e:
            print(f"✗ 错误: {e}")

    print("\n=== 2. IPv4 合并测试 ===")
    ipv4_tests = [
        (["192.168.0.0/24", "192.168.1.0/24"], ["192.168.0.0/23"], 512),
        (["10.0.0.0/24", "10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"], ["10.0.0.0/22"], 1024),
        (["172.16.0.0/16", "172.17.0.0/16"], ["172.16.0.0/15"], 131072),
        (["192.168.0.0/24", "192.168.0.128/25"], ["192.168.0.0/24"], 256),
        (["10.0.0.0/8", "10.0.0.0/16"], ["10.0.0.0/8"], 16777216),
        (["192.168.1.0/24", "192.168.3.0/24"], ["192.168.1.0/24", "192.168.3.0/24"], 512),
    ]
    for cidrs, expected_list, expected_count in ipv4_tests:
        result_list, result_count = merge_cidrs(cidrs)
        status = "✓" if result_list == expected_list and result_count == expected_count else "✗"
        print(f"{status} 输入: {cidrs}")
        print(f"  期望: {expected_list} (数量: {expected_count})")
        print(f"  实际: {result_list} (数量: {result_count})")

    print("\n=== 3. IPv6 合并测试 ===")
    ipv6_tests = [
        (["2001:db8::/64", "2001:db8:0:1::/64"], ["2001:db8::/63"], 2 ** 65),
        (["2001:db8::/128", "2001:db8::1/128"], ["2001:db8::/127"], 2),
        (["fe80::/10", "fe80:0:0:1::/64"], ["fe80::/10"], 2 ** 118),
        (["2001:db8::/48", "2001:db8:1::/48", "2001:db8:2::/48", "2001:db8:3::/48"], ["2001:db8::/46"], 2 ** 82),
        (["2001:db8::/32", "2001:db8:1::/48"], ["2001:db8::/32"], 2 ** 96),
    ]
    for cidrs, expected_list, expected_count in ipv6_tests:
        result_list, result_count = merge_cidrs(cidrs)
        status = "✓" if result_list == expected_list and result_count == expected_count else "✗"
        print(f"{status} 输入: {cidrs}")
        print(f"  期望: {expected_list} (数量: {expected_count})")
        print(f"  实际: {result_list} (数量: {result_count})")

    print("\n=== 4. 错误处理测试 ===")
    error_tests = [
        (["192.168.1.5/24"], "主机位非零"),
        (["256.0.0.1/24"], "非法IP"),
        (["192.168.0.0/33"], "非法前缀"),
        (["not-a-cidr"], "格式错误"),
        (["192.168.0.0/24", "2001:db8::/32"], "混合IPv4/IPv6"),
    ]
    for cidrs, desc in error_tests:
        try:
            merge_cidrs(cidrs)
            print(f"✗ 未捕获错误 [{desc}]: {cidrs}")
        except ValueError as e:
            print(f"✓ 正确捕获 [{desc}]: {cidrs}")
            print(f"  错误: {e}")

    print("\n=== 5. 混合版本包含关系错误测试 ===")
    try:
        cidr_contains("192.168.1.5/24", "192.168.1.0/24")
        print("✗ 未捕获主机位非零错误")
    except ValueError as e:
        print(f"✓ 正确捕获包含关系中的格式错误: {e}")
