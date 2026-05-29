from language_detector import (
    detect_language,
    detect_language_details,
    detect_language_mixed,
    detect_language_comprehensive
)

print("=" * 70)
print("语言检测工具完整示例")
print("=" * 70)

print("\n" + "=" * 70)
print("1. 多语言混合文本检测（返回各语言比例）")
print("=" * 70)

mixed_texts = [
    "Hello 你好 Bonjour 今日はいい天気です",
    "Привет! 今天的 meeting 在 3 点召开。",
    "The 人工智能 AI 技术 is very powerful.",
    "안녕하세요, 我是 John, nice to meet you!",
]

for text in mixed_texts:
    result = detect_language_mixed(text)
    print(f"\n文本: {text}")
    print(f"语言比例: {result}")

print("\n" + "=" * 70)
print("2. 繁体/简体中文自动识别")
print("=" * 70)

chinese_texts = [
    ("简体中文", "这是简体中文的范例，使用了「么」「这」「里」等简体字。"),
    ("繁体中文", "這是繁體中文的範例，使用了「麼」「這」「裡」等繁體字。"),
    ("混合文本", "这个项目使用了「技術」与「研究」两方面的内容。"),
]

for name, text in chinese_texts:
    result = detect_language_comprehensive(text)
    variant = result.get('chinese_variant', {})
    print(f"\n[{name}] {text}")
    print(f"  检测结果: {variant}")

print("\n" + "=" * 70)
print("3. 综合检测（包含所有信息）")
print("=" * 70)

text = "Привет! 今天的 meeting 在 3 点召开，讨论 AI project 的进展。這裡有繁體字。"
result = detect_language_comprehensive(text)
print(f"\n文本: {text}")
print(f"\n主语言检测:")
print(f"  ISO代码: {result['primary']['iso_code']}")
print(f"  语言名称: {result['primary']['language_name']}")
print(f"  置信度: {result['primary']['confidence']}")
print(f"\n各语言比例:")
for lang, ratio in result['mixed_languages'].items():
    print(f"  {lang}: {ratio:.2%}")
if 'chinese_variant' in result:
    print(f"\n中文变体: {result['chinese_variant']['variant']} ({result['chinese_variant']['confidence']:.2%})")
print(f"\n有效字符数: {result['total_effective_chars']}")

print("\n" + "=" * 70)
print("4. 便捷函数对比")
print("=" * 70)

text = "Hello world, 你好世界！"
print(f"\n文本: {text}")
print(f"\ndetect_language (简单): {detect_language(text)}")
print(f"\ndetect_language_details (详细): {detect_language_details(text)}")
print(f"\ndetect_language_mixed (混合): {detect_language_mixed(text)}")
print(f"\ndetect_language_comprehensive (综合): 见上方示例")
