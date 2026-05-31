import heapq
import json
import os


class HuffmanNode:
    __slots__ = ('char', 'freq', 'left', 'right')

    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

    def is_leaf(self):
        return self.left is None and self.right is None


def build_huffman_tree(freq_map):
    heap = [HuffmanNode(char=ch, freq=f) for ch, f in freq_map.items()]
    heapq.heapify(heap)

    if len(heap) == 0:
        return None
    if len(heap) == 1:
        only = heapq.heappop(heap)
        dummy = HuffmanNode(freq=0)
        return HuffmanNode(freq=only.freq, left=only, right=dummy)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)

    return heap[0]


def _generate_codes(node, prefix, code_map):
    if node is None:
        return
    if node.is_leaf():
        if node.char is not None:
            code_map[node.char] = prefix
        return
    _generate_codes(node.left, prefix + '0', code_map)
    _generate_codes(node.right, prefix + '1', code_map)


def huffman_encode(freq_map):
    root = build_huffman_tree(freq_map)
    code_map = {}
    _generate_codes(root, '', code_map)

    total_chars = sum(freq_map.values())
    weighted_sum = sum(freq_map[ch] * len(code_map[ch]) for ch in freq_map)
    avg_len = weighted_sum / total_chars if total_chars else 0

    encoded = ''.join(code_map[ch] for ch in sorted(freq_map.keys()))
    return code_map, encoded, avg_len, root


def decode(encoded_str, root):
    if root is None or not encoded_str:
        return ''
    if root.is_leaf():
        return root.char * len(encoded_str)
    result = []
    node = root
    for bit in encoded_str:
        node = node.left if bit == '0' else node.right
        if node.is_leaf():
            result.append(node.char)
            node = root
    return ''.join(result)


def compute_freq(text):
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    return freq


def bits_to_bytes(bit_str):
    padding = (8 - len(bit_str) % 8) % 8
    padded = bit_str + '0' * padding
    result = bytearray()
    for i in range(0, len(padded), 8):
        result.append(int(padded[i:i + 8], 2))
    return bytes(result), padding


def bytes_to_bits(data, padding):
    bit_str = ''.join(f'{b:08b}' for b in data)
    if padding > 0:
        bit_str = bit_str[:-padding]
    return bit_str


def compress_file(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    if not text:
        print("文件为空，无需压缩。")
        return

    freq_map = compute_freq(text)
    code_map, _, avg_len, root = huffman_encode(freq_map)

    encoded_bits = ''.join(code_map[ch] for ch in text)
    compressed_bytes, padding = bits_to_bytes(encoded_bits)

    num_unique = len(freq_map)
    ascii_bits_per_char = max(1, (num_unique - 1).bit_length())
    fixed_bits_per_char = 8

    original_size_bits = len(text) * fixed_bits_per_char
    compressed_size_bits = len(encoded_bits)
    compressed_size_bytes = len(compressed_bytes)

    code_table_path = os.path.join(output_dir, 'code_table.json')
    with open(code_table_path, 'w', encoding='utf-8') as f:
        json.dump({
            'code_map': {repr(k): v for k, v in code_map.items()},
            'padding': padding,
            'original_length': len(text),
        }, f, ensure_ascii=False, indent=2)

    compressed_path = os.path.join(output_dir, 'compressed.bin')
    with open(compressed_path, 'wb') as f:
        f.write(compressed_bytes)

    compression_ratio = compressed_size_bits / original_size_bits * 100
    saving_ratio = (1 - compressed_size_bits / original_size_bits) * 100

    print(f"{'=' * 50}")
    print(f"文件压缩报告: {os.path.basename(input_path)}")
    print(f"{'=' * 50}")
    print(f"原始文本长度:         {len(text)} 字符")
    print(f"不同字符数:           {num_unique}")
    print(f"{'-' * 50}")
    print(f"ASCII 固定长度编码:   {fixed_bits_per_char} 位/字符")
    print(f"  → 总位数:           {original_size_bits} 位")
    print(f"  → 总字节数:         {original_size_bits // 8} 字节")
    print(f"定长编码(最小位数):   {ascii_bits_per_char} 位/字符")
    print(f"  → 总位数:           {len(text) * ascii_bits_per_char} 位")
    print(f"哈夫曼变长编码:       {avg_len:.2f} 位/字符 (平均)")
    print(f"  → 总位数:           {compressed_size_bits} 位")
    print(f"  → 总字节数:         {compressed_size_bytes} 字节 (含 {padding} 位填充)")
    print(f"{'-' * 50}")
    print(f"对比 ASCII (8位):     压缩率 {compression_ratio:.1f}%  节省 {saving_ratio:.1f}%")
    fixed_min_ratio = len(text) * ascii_bits_per_char / original_size_bits * 100
    fixed_min_saving = (1 - len(text) * ascii_bits_per_char / original_size_bits) * 100
    print(f"对比定长({ascii_bits_per_char}位):     压缩率 {fixed_min_ratio:.1f}%  节省 {fixed_min_saving:.1f}%")
    huff_vs_fixed = compressed_size_bits / (len(text) * ascii_bits_per_char) * 100
    huff_vs_fixed_saving = (1 - compressed_size_bits / (len(text) * ascii_bits_per_char)) * 100
    print(f"哈夫曼 vs 定长({ascii_bits_per_char}位): 压缩率 {huff_vs_fixed:.1f}%  额外节省 {huff_vs_fixed_saving:.1f}%")
    print(f"{'=' * 50}")
    print(f"码表已保存:           {code_table_path}")
    print(f"压缩文件已保存:       {compressed_path}")

    return compressed_path, code_table_path


def decompress_file(compressed_path, code_table_path, output_path):
    with open(code_table_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    raw_map = meta['code_map']
    code_map = {}
    for k, v in raw_map.items():
        if k.startswith("'") and k.endswith("'"):
            ch = eval(k)
        else:
            ch = k
        code_map[ch] = v

    padding = meta['padding']
    original_length = meta['original_length']

    with open(compressed_path, 'rb') as f:
        compressed_bytes = f.read()

    bit_str = bytes_to_bits(compressed_bytes, padding)

    reverse_map = {v: k for k, v in code_map.items()}
    result = []
    buf = ''
    for bit in bit_str:
        buf += bit
        if buf in reverse_map:
            result.append(reverse_map[buf])
            buf = ''

    decoded = ''.join(result)[:original_length]

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(decoded)

    print(f"\n解压完成: {output_path}")
    print(f"还原文本长度: {len(decoded)} 字符")
    return decoded


if __name__ == '__main__':
    freq_map = {
        'a': 5,
        'b': 9,
        'c': 12,
        'd': 13,
        'e': 16,
        'f': 45,
    }

    code_map, encoded, avg_len, root = huffman_encode(freq_map)

    print(f"{'字符':^6}{'频率':^8}{'编码':^12}{'编码长度':^10}")
    print('-' * 40)
    for ch in sorted(freq_map, key=lambda c: freq_map[c], reverse=True):
        print(f"{ch:^6}{freq_map[ch]:^8}{code_map[ch]:^12}{len(code_map[ch]):^10}")

    print(f"\n平均编码长度: {avg_len:.2f}")
    print(f"拼接编码串:   {encoded}")

    message = 'fac'
    binary = ''.join(code_map[ch] for ch in message)
    print(f"\n原文 '{message}' → 二进制串: {binary}")
    print(f"二进制串 '{binary}' → 解码: {decode(binary, root)}")

    print("\n" + "=" * 40)
    print("单字符测试:")
    single_map = {'x': 10}
    code_map2, _, _, root2 = huffman_encode(single_map)
    print(f"字符 'x' 编码: {code_map2['x']}")
    binary2 = code_map2['x'] * 3
    print(f"编码 'xxx' → 二进制串: {binary2}")
    print(f"二进制串 '{binary2}' → 解码: {decode(binary2, root2)}")

    sample_text = (
        "this is an example of a huffman tree. "
        "the huffman coding algorithm is a lossless data compression method. "
        "it assigns variable-length codes to input characters based on their frequencies. "
        "characters that appear more frequently get shorter codes, while rarer characters get longer codes. "
        "this ensures that the overall encoded message is as compact as possible."
    )

    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample')
    os.makedirs(sample_dir, exist_ok=True)
    sample_path = os.path.join(sample_dir, 'input.txt')
    with open(sample_path, 'w', encoding='utf-8') as f:
        f.write(sample_text)

    output_dir = os.path.join(sample_dir, 'output')
    comp_path, table_path = compress_file(sample_path, output_dir)

    restored_path = os.path.join(output_dir, 'restored.txt')
    decompress_file(comp_path, table_path, restored_path)

    with open(restored_path, 'r', encoding='utf-8') as f:
        restored_text = f.read()
    match = restored_text == sample_text
    print(f"\n还原验证: {'通过 ✓' if match else '失败 ✗'}")
    if not match:
        print(f"  原文长度: {len(sample_text)}, 还原长度: {len(restored_text)}")
