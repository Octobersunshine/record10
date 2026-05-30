from sentiment_analyzer import analyze_sentiment, analyze_batch


def test_positive():
    print("=== 正面情感测试 ===")
    test_cases = [
        "这部电影非常好看，很精彩！",
        "今天的天气真好，心情特别开心！",
        "这家餐厅的菜很美味，服务也很周到",
        "超级棒的产品，非常满意，五星好评！",
        "这个方案太棒了，我很喜欢",
        "物美价廉，性价比很高，推荐购买",
    ]
    for text in test_cases:
        result = analyze_sentiment(text)
        print(f"文本: {text}")
        print(f"  情感: {result['sentiment']}, 得分: {result['score']}")
        print(f"  正面词: {result['positive_words']}")
        print()


def test_negative():
    print("=== 负面情感测试 ===")
    test_cases = [
        "这部电影太差了，非常难看",
        "今天的天气真糟糕，心情很差",
        "这家餐厅的菜很难吃，服务也很差",
        "产品质量很差，非常失望，一星差评！",
        "这个方案太烂了，我很讨厌",
        "价格贵，质量差，不推荐购买",
    ]
    for text in test_cases:
        result = analyze_sentiment(text)
        print(f"文本: {text}")
        print(f"  情感: {result['sentiment']}, 得分: {result['score']}")
        print(f"  负面词: {result['negative_words']}")
        print()


def test_negation():
    print("=== 否定词处理测试 ===")
    test_cases = [
        ("这部电影好看", "这部电影不好看"),
        ("这个产品很满意", "这个产品不满意"),
        ("他很开心", "他不开心"),
        ("这东西好吃", "这东西不好吃"),
        ("服务周到", "服务不周到"),
    ]
    for positive, negative in test_cases:
        r1 = analyze_sentiment(positive)
        r2 = analyze_sentiment(negative)
        print(f"肯定: {positive} -> {r1['sentiment']} ({r1['score']})")
        print(f"否定: {negative} -> {r2['sentiment']} ({r2['score']})")
        print(f"否定词: {r2['negations_found']}")
        print()


def test_intensifier():
    print("=== 程度副词测试 ===")
    test_cases = [
        "好",
        "很好",
        "非常好",
        "特别好",
        "极其好",
        "超级棒",
    ]
    for text in test_cases:
        result = analyze_sentiment(text)
        print(f"文本: {text:10} -> {result['sentiment']:8} 得分: {result['score']:.4f}")
    print()


def test_exclamation():
    print("=== 感叹号增强测试 ===")
    test_cases = [
        "太好了",
        "太好了!",
        "太好了!!",
        "太好了!!!",
    ]
    for text in test_cases:
        result = analyze_sentiment(text)
        print(f"文本: {text:12} -> {result['sentiment']:8} 得分: {result['score']:.4f}")
    print()


def test_neutral():
    print("=== 中性情感测试 ===")
    test_cases = [
        "今天是星期三",
        "我吃饭了",
        "这个产品是红色的",
        "会议下午三点开始",
        "",
        "   ",
    ]
    for text in test_cases:
        result = analyze_sentiment(text)
        print(f"文本: '{text}'")
        print(f"  情感: {result['sentiment']}, 得分: {result['score']}")
        print()


def test_batch():
    print("=== 批量分析测试 ===")
    texts = [
        "这个产品非常好，很满意！",
        "质量很差，非常失望",
        "今天天气不错",
        "不太喜欢这个设计",
        "超级棒，强烈推荐！！！",
    ]
    results = analyze_batch(texts)
    for i, r in enumerate(results):
        print(f"[{i+1}] {r['text']}")
        print(f"    情感: {r['sentiment']}, 得分: {r['score']}")
    print()


def test_mixed():
    print("=== 混合情感测试 ===")
    test_cases = [
        "虽然价格贵了点，但是质量很好，我很满意",
        "产品不错，但是发货太慢了",
        "环境很好，服务很差",
        "不好不坏，一般般",
    ]
    for text in test_cases:
        result = analyze_sentiment(text)
        print(f"文本: {text}")
        print(f"  情感: {result['sentiment']}, 得分: {result['score']}")
        print(f"  正面词: {result['positive_words']}")
        print(f"  负面词: {result['negative_words']}")
        print()


def test_english():
    print("=== 英文测试 ===")
    test_cases = [
        "very good",
        "not good",
        "this is very bad",
        "I don't like it",
        "extremely happy",
    ]
    for text in test_cases:
        result = analyze_sentiment(text)
        print(f"文本: {text}")
        print(f"  情感: {result['sentiment']}, 得分: {result['score']}")
        print()


if __name__ == '__main__':
    test_positive()
    test_negative()
    test_negation()
    test_intensifier()
    test_exclamation()
    test_neutral()
    test_batch()
    test_mixed()
    test_english()
    print("=== 所有测试完成 ===")
