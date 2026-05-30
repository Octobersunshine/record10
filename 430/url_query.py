from urllib.parse import parse_qs, urlencode, urlparse, urlunparse, unquote_plus, quote_plus, urljoin, urldefrag, urlsplit, urlunsplit
import re
import posixpath


def _parse_nested_key(key, value):
    result = {}
    
    bracket_pattern = re.compile(r'^([^\[\]]+)((?:\[[^\]]*\])+)$')
    match = bracket_pattern.match(key)
    
    if not match:
        return {key: value}
    
    main_key = match.group(1)
    brackets_str = match.group(2)
    parts = [main_key]
    
    bracket_matches = re.findall(r'\[([^\]]*)\]', brackets_str)
    parts.extend(bracket_matches)
    
    current = result
    for i, part in enumerate(parts[:-1]):
        if isinstance(current, dict):
            if part not in current:
                next_part = parts[i + 1] if i + 1 < len(parts) else None
                if next_part == '' or (next_part is not None and next_part.isdigit()):
                    current[part] = []
                else:
                    current[part] = {}
            current = current[part]
        elif isinstance(current, list):
            if part == '':
                current.append({})
                current = current[-1]
            elif part.isdigit():
                idx = int(part)
                while len(current) <= idx:
                    current.append(None)
                if current[idx] is None:
                    next_part = parts[i + 1] if i + 1 < len(parts) else None
                    if next_part == '' or (next_part is not None and next_part.isdigit()):
                        current[idx] = []
                    else:
                        current[idx] = {}
                current = current[idx]
            else:
                current.append({})
                current = current[-1]
    
    last_part = parts[-1]
    if isinstance(current, list):
        if last_part == '':
            if isinstance(value, list):
                current.extend(value)
            else:
                current.append(value)
        elif last_part.isdigit():
            idx = int(last_part)
            while len(current) <= idx:
                current.append(None)
            current[idx] = value
        else:
            if isinstance(value, list):
                current.extend(value)
            else:
                current.append(value)
    else:
        current[last_part] = value
    
    return result


def _merge_nested(dest, src, is_empty_bracket=False):
    for key, value in src.items():
        if key in dest:
            if is_empty_bracket and isinstance(dest[key], list):
                if isinstance(value, list):
                    dest[key].extend(value)
                else:
                    dest[key].append(value)
            elif isinstance(dest[key], dict) and isinstance(value, dict):
                _merge_nested(dest[key], value)
            elif isinstance(dest[key], list) and isinstance(value, list):
                for i, v in enumerate(value):
                    if v is not None:
                        if i < len(dest[key]):
                            if isinstance(dest[key][i], dict) and isinstance(v, dict):
                                _merge_nested(dest[key][i], v)
                            else:
                                dest[key][i] = v
                        else:
                            dest[key].append(v)
            elif not isinstance(dest[key], list):
                dest[key] = [dest[key], value]
            else:
                dest[key].append(value)
        else:
            dest[key] = value
    return dest


def _decode_value(value):
    if isinstance(value, list):
        return [_decode_value(v) for v in value]
    if isinstance(value, str):
        return unquote_plus(value)
    return value


def _encode_value(value):
    if isinstance(value, list):
        return [_encode_value(v) for v in value]
    if isinstance(value, str):
        return quote_plus(value)
    return value


def parse_query_params(url_or_query):
    if "?" in url_or_query:
        query = urlparse(url_or_query).query
    else:
        query = url_or_query.lstrip("?")

    raw = parse_qs(query, keep_blank_values=True)
    flat = {}
    for key, values in raw.items():
        decoded_key = unquote_plus(key)
        if len(values) == 1:
            flat[decoded_key] = _decode_value(values[0])
        else:
            flat[decoded_key] = [_decode_value(v) for v in values]

    result = {}
    for key, value in flat.items():
        if '[' in key and ']' in key:
            if key.endswith('[]'):
                nested = _parse_nested_key(key, value)
                _merge_nested(result, nested, is_empty_bracket=True)
            else:
                nested = _parse_nested_key(key, value)
                _merge_nested(result, nested)
        else:
            if key in result:
                if isinstance(result[key], list):
                    if isinstance(value, list):
                        result[key].extend(value)
                    else:
                        result[key].append(value)
                else:
                    existing = result[key]
                    if isinstance(value, list):
                        result[key] = [existing] + value
                    else:
                        result[key] = [existing, value]
            else:
                result[key] = value

    return result


def _flatten_nested(obj, prefix=''):
    items = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = f"{prefix}[{key}]" if prefix else key
            items.extend(_flatten_nested(value, new_key))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            new_key = f"{prefix}[{i}]"
            items.extend(_flatten_nested(value, new_key))
    else:
        items.append((prefix, obj))
    return items


def dict_to_query_string(params, doseq=False):
    flat_params = []
    has_nested = False

    for key, value in params.items():
        if isinstance(value, dict):
            has_nested = True
            items = _flatten_nested(value, key)
            for k, v in items:
                if isinstance(v, list) and not doseq:
                    flat_params.append((k, ",".join(str(item) for item in v)))
                else:
                    flat_params.append((k, v))
        elif isinstance(value, list):
            if doseq:
                for v in value:
                    flat_params.append((key, v))
            else:
                flat_params.append((key, ",".join(str(v) for v in value)))
        else:
            flat_params.append((key, value))

    if has_nested:
        encoded_parts = []
        for k, v in flat_params:
            if isinstance(v, list):
                for item in v:
                    encoded_key = quote_plus(k)
                    encoded_val = quote_plus(str(item))
                    encoded_parts.append(f"{encoded_key}={encoded_val}")
            else:
                encoded_key = quote_plus(k)
                encoded_val = quote_plus(str(v))
                encoded_parts.append(f"{encoded_key}={encoded_val}")
        query_str = "&".join(encoded_parts)
        return f"?{query_str}" if query_str else ""
    else:
        encoded = urlencode(flat_params, doseq=doseq)
        return f"?{encoded}" if encoded else ""


def parse_url(url):
    parsed = urlparse(url)
    hostname = parsed.hostname
    port = parsed.port
    default_ports = {'http': 80, 'https': 443, 'ftp': 21, 'ftps': 990, 'ssh': 22}
    is_default_port = port is None or (parsed.scheme in default_ports and port == default_ports[parsed.scheme])

    return {
        'scheme': parsed.scheme,
        'protocol': parsed.scheme + '://',
        'username': parsed.username,
        'password': parsed.password,
        'hostname': hostname,
        'domain': hostname,
        'port': port,
        'is_default_port': is_default_port,
        'netloc': parsed.netloc,
        'path': parsed.path or '/',
        'params': parsed.params,
        'query': parsed.query,
        'query_params': parse_query_params(parsed.query) if parsed.query else {},
        'fragment': parsed.fragment,
        'anchor': parsed.fragment,
        'authority': parsed.netloc,
        'url': url,
    }


def join_url(base, relative):
    return urljoin(base, relative)


def normalize_url(url, lowercase_host=True, lowercase_scheme=True,
                  remove_default_port=True, sort_query_params=True,
                  remove_fragment=False, add_trailing_slash=False):
    parsed = urlparse(url)
    scheme = parsed.scheme
    netloc = parsed.netloc
    path = parsed.path
    query = parsed.query
    fragment = parsed.fragment

    if lowercase_scheme:
        scheme = scheme.lower()

    if lowercase_host and '@' in netloc:
        userinfo, host = netloc.rsplit('@', 1)
        netloc = userinfo + '@' + host.lower()
    elif lowercase_host:
        netloc = netloc.lower()

    if remove_default_port:
        default_ports = {'http': 80, 'https': 443, 'ftp': 21, 'ftps': 990, 'ssh': 22}
        if ':' in netloc and '@' not in netloc:
            host_part, port_part = netloc.rsplit(':', 1)
            try:
                port_num = int(port_part)
                if scheme in default_ports and port_num == default_ports[scheme]:
                    netloc = host_part
            except ValueError:
                pass
        elif ':' in netloc and '@' in netloc:
            userinfo, hostport = netloc.rsplit('@', 1)
            if ':' in hostport:
                host_part, port_part = hostport.rsplit(':', 1)
                try:
                    port_num = int(port_part)
                    if scheme in default_ports and port_num == default_ports[scheme]:
                        netloc = userinfo + '@' + host_part
                except ValueError:
                    pass

    if not path:
        path = '/'
    else:
        path = posixpath.normpath(path)
        if add_trailing_slash and not path.endswith('/') and '.' not in path.split('/')[-1]:
            path += '/'

    if sort_query_params and query:
        params = parse_qs(query, keep_blank_values=True)
        sorted_items = sorted(params.items(), key=lambda x: x[0])
        flat = []
        for key, values in sorted_items:
            for val in values:
                flat.append((key, val))
        query = urlencode(flat)

    if remove_fragment:
        fragment = ''

    result = urlunparse((scheme, netloc, path, parsed.params, query, fragment))
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("【测试1】查询参数解析与序列化")
    print("=" * 60)
    test_cases = [
        "https://example.com/path?a=1&b=2",
        "https://example.com/path?a=1&a=2&a=3&b=hello&c=",
        "a=1&b=2&c=3",
        "?x=10&y=20",
        "q=hello+world&name=%E4%B8%AD%E6%96%87",
        "?text=hello%20world&special=!%40%23%24%25%5E%26*()",
        "user[name]=张三&user[age]=25&user[email]=test@example.com",
        "a[b][c]=1&a[b][d]=2",
        "items[0]=apple&items[1]=banana&items[2]=cherry",
        "data[users][0][name]=Alice&data[users][0][id]=1&data[users][1][name]=Bob&data[users][1][id]=2",
        "tags[]=python&tags[]=javascript&tags[]=go",
        "?filter[status]=active&filter[sort]=desc&page=1",
    ]

    for u in test_cases:
        params = parse_query_params(u)
        print(f"输入: {u}")
        print(f"解析: {params}")
        qs = dict_to_query_string(params, doseq=True)
        print(f"序列化: {qs}")
        print()

    print("=" * 60)
    print("【测试2】URL完整解析")
    print("=" * 60)
    url_parse_cases = [
        "https://example.com:8080/path/to/page.html?a=1&b=2#section",
        "http://user:pass@example.com:80/api/users?id=123",
        "ftp://files.example.com/pub/README.txt",
        "https://www.baidu.com/s?wd=python&rsv_spt=1#test",
        "ssh://git@github.com:22/user/repo.git",
    ]

    for url in url_parse_cases:
        result = parse_url(url)
        print(f"URL: {url}")
        print(f"  协议: {result['scheme']}")
        print(f"  主机: {result['hostname']}")
        print(f"  端口: {result['port']} (默认端口: {result['is_default_port']})")
        print(f"  路径: {result['path']}")
        print(f"  用户名: {result['username']}, 密码: {result['password']}")
        print(f"  查询参数: {result['query_params']}")
        print(f"  锚点: {result['anchor']}")
        print()

    print("=" * 60)
    print("【测试3】URL拼接")
    print("=" * 60)
    join_cases = [
        ("https://example.com/path/", "subpage.html"),
        ("https://example.com/path/file.html", "../other/page.html"),
        ("https://example.com/path/?a=1", "?b=2"),
        ("https://example.com/path#top", "newpath"),
        ("https://example.com/", "https://other.com/page.html"),
        ("https://example.com/path/to/file.html", "./relative.html"),
        ("https://example.com/path/to/file.html", "/absolute.html"),
    ]

    for base, rel in join_cases:
        result = join_url(base, rel)
        print(f"基准: {base}")
        print(f"相对: {rel}")
        print(f"结果: {result}")
        print()

    print("=" * 60)
    print("【测试4】URL规范化")
    print("=" * 60)
    norm_cases = [
        "HTTP://EXAMPLE.COM:80/path?z=1&a=2&m=3#top",
        "HTTPS://Example.COM:443/./path/../to/./page",
        "https://example.com/path//to///file",
        "http://example.com:8080/page?b=2&a=1",
        "https://user:pass@EXAMPLE.com/path#fragment",
        "https://example.com/no-slash",
    ]

    for url in norm_cases:
        print(f"原始: {url}")
        print(f"规范: {normalize_url(url)}")
        print(f"去锚点: {normalize_url(url, remove_fragment=True)}")
        print(f"补斜杠: {normalize_url(url, add_trailing_slash=True)}")
        print()

