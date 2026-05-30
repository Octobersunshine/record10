import dns.resolver
import socket
from typing import List, Dict, Optional


class DNSResolver:
    def __init__(self, nameservers: Optional[List[str]] = None):
        self.resolver = dns.resolver.Resolver()
        if nameservers:
            self.resolver.nameservers = nameservers

    def resolve_a(self, domain: str) -> List[str]:
        try:
            answers = self.resolver.resolve(domain, 'A')
            return [str(rdata) for rdata in answers]
        except Exception as e:
            return [f"Error: {str(e)}"]

    def resolve_aaaa(self, domain: str) -> List[str]:
        try:
            answers = self.resolver.resolve(domain, 'AAAA')
            return [str(rdata) for rdata in answers]
        except Exception as e:
            return [f"Error: {str(e)}"]

    def resolve_all(self, domain: str) -> Dict[str, List[str]]:
        return {
            'A': self.resolve_a(domain),
            'AAAA': self.resolve_aaaa(domain)
        }


def resolve_with_socket(domain: str) -> Dict[str, List[str]]:
    result = {'A': [], 'AAAA': []}
    try:
        addrinfo = socket.getaddrinfo(domain, None)
        for info in addrinfo:
            family = info[0]
            ip = info[4][0]
            if family == socket.AF_INET:
                if ip not in result['A']:
                    result['A'].append(ip)
            elif family == socket.AF_INET6:
                if ip not in result['AAAA']:
                    result['AAAA'].append(ip)
    except Exception as e:
        result = {'A': [f"Error: {str(e)}"], 'AAAA': [f"Error: {str(e)}"]}
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description='DNS解析工具')
    parser.add_argument('domain', help='要解析的域名')
    parser.add_argument('-s', '--server', help='指定DNS服务器（如8.8.8.8）')
    parser.add_argument('-t', '--type', choices=['A', 'AAAA', 'all'], default='all',
                        help='查询类型（A/AAAA/all，默认all）')
    parser.add_argument('--socket', action='store_true',
                        help='使用socket库而不是dnspython')

    args = parser.parse_args()

    if args.socket:
        print(f"使用socket库解析域名: {args.domain}")
        results = resolve_with_socket(args.domain)
    else:
        nameservers = [args.server] if args.server else None
        resolver = DNSResolver(nameservers)
        if nameservers:
            print(f"使用DNS服务器 {nameservers[0]} 解析域名: {args.domain}")
        else:
            print(f"使用系统默认DNS服务器解析域名: {args.domain}")

        if args.type == 'A':
            results = {'A': resolver.resolve_a(args.domain)}
        elif args.type == 'AAAA':
            results = {'AAAA': resolver.resolve_aaaa(args.domain)}
        else:
            results = resolver.resolve_all(args.domain)

    print("\n查询结果:")
    print("=" * 50)
    for record_type, ips in results.items():
        if ips:
            print(f"\n{record_type} 记录:")
            for ip in ips:
                print(f"  {ip}")
        else:
            print(f"\n{record_type} 记录: 无")


if __name__ == '__main__':
    main()
