ESCAPE_CHAR = '\\'


def _encode_single(char: str, count: int) -> str:
    result = []
    if char.isdigit() or char == ESCAPE_CHAR:
        result.append(ESCAPE_CHAR)
    result.append(char)
    result.append(str(count))
    return ''.join(result)


def _encode_segment(char: str, count: int) -> str:
    result = []
    remaining = count
    while remaining > 0:
        segment = min(remaining, 9)
        result.append(_encode_single(char, segment))
        remaining -= segment
    return ''.join(result)


def rle_encode(data: str) -> tuple[str, float]:
    if not data:
        return "", 0.0
    
    result = []
    count = 1
    prev_char = data[0]
    
    for char in data[1:]:
        if char == prev_char:
            count += 1
        else:
            result.append(_encode_segment(prev_char, count))
            prev_char = char
            count = 1
    result.append(_encode_segment(prev_char, count))
    
    encoded = ''.join(result)
    compression_ratio = len(encoded) / len(data)
    
    return encoded, compression_ratio


def rle_decode(encoded: str) -> str:
    if not encoded:
        return ""
    
    result = []
    i = 0
    n = len(encoded)
    
    while i < n:
        is_escaped = False
        if encoded[i] == ESCAPE_CHAR:
            is_escaped = True
            i += 1
        
        char = encoded[i]
        i += 1
        
        while i < n and encoded[i].isdigit():
            count = int(encoded[i])
            result.append(char * count)
            i += 1
    
    return ''.join(result)


def lz77_token_to_string(token: tuple[int, int, str]) -> str:
    offset, length, char = token
    if char == ESCAPE_CHAR:
        char_str = ESCAPE_CHAR + ESCAPE_CHAR
    elif char.isdigit():
        char_str = ESCAPE_CHAR + char
    elif char == '':
        char_str = ESCAPE_CHAR
    else:
        char_str = char
    
    if offset < 10 and length < 10:
        return f"{offset}{length}{char_str}"
    else:
        offset_str = str(offset) if offset < 10 else f"{ESCAPE_CHAR}{offset:02d}"
        length_str = str(length) if length < 10 else f"{ESCAPE_CHAR}{length:02d}"
        return f"{offset_str}{length_str}{char_str}"


def lz77_encode(data: str, window_size: int = 32, lookahead_size: int = 15) -> tuple[list[tuple[int, int, str]], float]:
    if not data:
        return [], 0.0
    
    tokens = []
    i = 0
    n = len(data)
    
    while i < n:
        search_start = max(0, i - window_size)
        search_buffer = data[search_start:i]
        lookahead_end = min(i + lookahead_size, n)
        lookahead_buffer = data[i:lookahead_end]
        
        best_offset = 0
        best_length = 0
        
        for length in range(1, len(lookahead_buffer) + 1):
            substring = lookahead_buffer[:length]
            offset = search_buffer.rfind(substring)
            if offset != -1:
                best_offset = i - (search_start + offset)
                best_length = length
            else:
                break
        
        if best_length > 0:
            next_char = data[i + best_length] if (i + best_length) < n else ''
            tokens.append((best_offset, best_length, next_char))
            i += best_length + 1
        else:
            tokens.append((0, 0, data[i]))
            i += 1
    
    encoded_str = ''.join(lz77_token_to_string(t) for t in tokens)
    encoded_len = len(encoded_str)
    compression_ratio = encoded_len / n
    
    return tokens, compression_ratio


def lz77_decode(tokens: list[tuple[int, int, str]]) -> str:
    if not tokens:
        return ""
    
    result = []
    
    for offset, length, char in tokens:
        if length > 0:
            start = len(result) - offset
            for i in range(length):
                result.append(result[start + i])
        if char:
            result.append(char)
    
    return ''.join(result)


def lz77_tokens_to_string(tokens: list[tuple[int, int, str]]) -> str:
    return ''.join(lz77_token_to_string(t) for t in tokens)


def generate_test_texts() -> dict[str, str]:
    texts = {}
    
    texts['高度重复-连续相同字符'] = 'a' * 200
    
    texts['高度重复-重复模式'] = 'ababababab' * 20
    
    texts['高度重复-长重复序列'] = 'abcdefghij' * 20
    
    texts['中等重复-自然语言风格'] = (
        '今天天气真好，今天天气真好，今天天气真好。'
        '我去公园散步，我去公园散步，我去公园散步。'
        '花儿开了，鸟儿叫了，花儿开了，鸟儿叫了。'
    )
    
    texts['低重复-随机字母'] = 'qwertyuiopasdfghjklzxcvbnm' * 10
    
    texts['低重复-无规律混合'] = 'a1b2c3d4e5f6g7h8i9j0k!l@m#n$o%p^q&r*s(t)u_v+w=x-y/z.'
    
    texts['混合-部分重复'] = 'aaaaabbbbbcccccdddddeeeeefffffghijklmnopqrstuvwxyz'
    
    return texts


if __name__ == "__main__":
    print("=" * 70)
    print("RLE 算法测试")
    print("=" * 70)
    
    rle_test_cases = [
        "aaabbcc",
        "aaaaaaaaaaaa",
        "aaaaa",
        "abcd",
        "aabbaabbcc",
        "",
        "abababab",
        "WWWWWWWWWWWWBWWWWWWWWWWWWBBBWWWWWWWWWWWWWWWWWWWWWWWWBWWWWWWWWWWWWWW",
        "1233344444",
        "a1b2c3",
        "\\\\\\\\",
        "111111111111",
        "a\\\\111bbb"
    ]
    
    for original in rle_test_cases:
        encoded, ratio = rle_encode(original)
        decoded = rle_decode(encoded)
        
        print(f"原始字符串: '{original}'")
        print(f"压缩后: '{encoded}'")
        print(f"压缩率: {ratio:.2%}")
        print(f"解码验证: {'通过' if decoded == original else '失败'}")
        if decoded != original:
            print(f"  解码结果: '{decoded}'")
        print("-" * 50)
    
    print("\n" + "=" * 70)
    print("LZ77 算法基础测试")
    print("=" * 70)
    
    lz77_test_cases = [
        "aabbaabbaabb",
        "abcabcabcabc",
        "abracadabra",
        "aaaaabbbbbccccc",
        "the quick brown fox jumps over the lazy dog",
        ""
    ]
    
    for original in lz77_test_cases:
        tokens, ratio = lz77_encode(original)
        decoded = lz77_decode(tokens)
        
        print(f"原始字符串: '{original}'")
        print(f"压缩后: {lz77_tokens_to_string(tokens)}")
        print(f"压缩率: {ratio:.2%}")
        print(f"解码验证: {'通过' if decoded == original else '失败'}")
        if decoded != original:
            print(f"  解码结果: '{decoded}'")
        print("-" * 50)
    
    print("\n" + "=" * 70)
    print("RLE vs LZ77 压缩率对比测试")
    print("=" * 70)
    
    test_texts = generate_test_texts()
    
    print(f"{'文本类型':<30} {'原始长度':<10} {'RLE压缩率':<12} {'LZ77压缩率':<12} {'优胜算法'}")
    print("-" * 76)
    
    for text_type, text in test_texts.items():
        rle_encoded, rle_ratio = rle_encode(text)
        lz77_tokens, lz77_ratio = lz77_encode(text)
        
        lz77_decoded = lz77_decode(lz77_tokens)
        rle_decoded = rle_decode(rle_encoded)
        
        lz77_ok = lz77_decoded == text
        rle_ok = rle_decoded == text
        
        if rle_ratio < lz77_ratio:
            winner = "RLE"
        elif lz77_ratio < rle_ratio:
            winner = "LZ77"
        else:
            winner = "平局"
        
        status = ""
        if not rle_ok:
            status += " [RLE解码失败]"
        if not lz77_ok:
            status += " [LZ77解码失败]"
        
        print(f"{text_type:<30} {len(text):<10} {rle_ratio:<12.2%} {lz77_ratio:<12.2%} {winner}{status}")
    
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print("RLE优势: 连续重复字符极多的文本（如'aaaaa...'）")
    print("LZ77优势: 存在重复模式但非连续相同字符的文本（如'ababab...'）")
    print("随机文本: 两种算法都会膨胀，不适合压缩")
