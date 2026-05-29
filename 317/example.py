from ip_lookup import IPLookup


def single_ip_lookup_example():
    print("=== 单IP查询示例 ===")
    with IPLookup() as lookup:
        ip = '8.8.8.8'
        result = lookup.lookup(ip)
        print(f"IP: {ip}")
        print(f"国家: {result['country']['name']} ({result['country']['code']})")
        print(f"省份: {result['subdivision']['name']}")
        print(f"城市: {result['city']['name']}")
        print(f"ISP: {result['isp']['isp']}")
        print(f"经纬度: {result['location']['latitude']}, {result['location']['longitude']}")
        print()


def batch_ip_lookup_example():
    print("=== 批量IP查询示例 ===")
    with IPLookup() as lookup:
        ips = ['8.8.8.8', '1.1.1.1', '208.67.222.222', '114.114.114.114']
        results = lookup.batch_lookup(ips)
        for result in results:
            if result['success']:
                print(f"{result['ip']:15} - {result['country']['name']:10} {result['isp']['isp'][:20]}")
            else:
                print(f"{result['ip']:15} - 查询失败: {result['error']}")
        print()


def invalid_ip_example():
    print("=== 无效IP示例 ===")
    with IPLookup() as lookup:
        invalid_ips = ['999.999.999.999', 'not-an-ip', '192.168.1.999']
        for ip in invalid_ips:
            result = lookup.lookup(ip)
            print(f"{ip:20} - {result['error']}")
        print()


if __name__ == '__main__':
    print("IP地址信息查询工具 - 使用示例")
    print("注意：请确保GeoLite2-City.mmdb和GeoLite2-ASN.mmdb文件在当前目录")
    print("=" * 60)
    print()

    single_ip_lookup_example()
    batch_ip_lookup_example()
    invalid_ip_example()

    print("示例运行完成！")
