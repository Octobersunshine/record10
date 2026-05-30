from sentiment_analyzer import analyze_sentiment

test_cases = [
    '不是很好',
    '不是很差',
    '不不是很好',
    '不不是很差',
    '不太喜欢',
    '不太满意',
    '不是很满意',
    '不是不满意',
    '没有不满意',
    '不可能不好',
]

print("=== 当前否定词处理测试 ===")
for text in test_cases:
    result = analyze_sentiment(text)
    print(f'{text:15} -> {result["sentiment"]:8} ({result["score"]:.4f}) 否定词: {result["negations_found"]}')
