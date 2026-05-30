from sentiment_analyzer import (
    analyze_sentiment_with_aspects,
    analyze_sentiment,
    get_available_domains,
)
from domain_lexicon import DOMAIN_NAMES


def test_domain_list():
    print("=== 可用领域列表 ===")
    domains = get_available_domains()
    for d in domains:
        print(f"  {d}: {DOMAIN_NAMES.get(d, d)}")
    print()


def test_aspect_electronics():
    print("=== 电子产品方面级分析 ===")
    test_cases = [
        "屏幕很清晰，但电池不耐用",
        "摄像头拍照很棒，就是系统有点卡顿",
        "音质不错，散热也很好，就是价格贵了点",
        "这部手机屏幕很好，但电池太差了",
        "续航持久，显示效果也很细腻",
    ]

    for text in test_cases:
        result = analyze_sentiment_with_aspects(text, domain="electronics")
        print(f"文本: {text}")
        print(f"  整体情感: {result['sentiment']} ({result['score']})")
        print(f"  方面分析:")
        for aspect, data in result['aspects'].items():
            print(f"    - {aspect}: {data['sentiment']} ({data['score']})")
            print(f"      提到的词: {data['mentions']}")
            if data['positive_words']:
                print(f"      正面词: {data['positive_words']}")
            if data['negative_words']:
                print(f"      负面词: {data['negative_words']}")
        print()


def test_aspect_restaurant():
    print("=== 餐饮方面级分析 ===")
    test_cases = [
        "菜品味道不错，但环境太吵了",
        "服务态度很好，上菜也快，就是菜量太少了",
        "食材很新鲜，味道也正宗，就是价格有点贵",
        "这家店菜不好吃，服务也差，环境还脏",
        "味道一般般，不过分量足，性价比高",
    ]

    for text in test_cases:
        result = analyze_sentiment_with_aspects(text, domain="restaurant")
        print(f"文本: {text}")
        print(f"  整体情感: {result['sentiment']} ({result['score']})")
        print(f"  方面分析:")
        for aspect, data in result['aspects'].items():
            print(f"    - {aspect}: {data['sentiment']} ({data['score']})")
            print(f"      提到的词: {data['mentions']}")
            if data['positive_words']:
                print(f"      正面词: {data['positive_words']}")
            if data['negative_words']:
                print(f"      负面词: {data['negative_words']}")
        print()


def test_aspect_finance():
    print("=== 金融理财方面级分析 ===")
    test_cases = [
        "这只基金收益不错，但波动太大了",
        "收益稳定，风险也低，就是手续费有点高",
        "最近涨得很好，就是赎回到账太慢了",
        "这个产品太坑了，一直跌，客服也不专业",
        "安全性好，流动性也强，靠谱",
    ]

    for text in test_cases:
        result = analyze_sentiment_with_aspects(text, domain="finance")
        print(f"文本: {text}")
        print(f"  整体情感: {result['sentiment']} ({result['score']})")
        print(f"  方面分析:")
        for aspect, data in result['aspects'].items():
            print(f"    - {aspect}: {data['sentiment']} ({data['score']})")
            print(f"      提到的词: {data['mentions']}")
            if data['positive_words']:
                print(f"      正面词: {data['positive_words']}")
            if data['negative_words']:
                print(f"      负面词: {data['negative_words']}")
        print()


def test_aspect_clothing():
    print("=== 服装方面级分析 ===")
    test_cases = [
        "面料很舒服，就是做工有点粗糙",
        "款式很好看，但码数偏小",
        "颜色和图片色差太大，不过版型不错",
        "这件衣服性价比高，做工也好，就是容易起球",
        "穿上很显瘦，材质也高级，推荐！",
    ]

    for text in test_cases:
        result = analyze_sentiment_with_aspects(text, domain="clothing")
        print(f"文本: {text}")
        print(f"  整体情感: {result['sentiment']} ({result['score']})")
        print(f"  方面分析:")
        for aspect, data in result['aspects'].items():
            print(f"    - {aspect}: {data['sentiment']} ({data['score']})")
            print(f"      提到的词: {data['mentions']}")
            if data['positive_words']:
                print(f"      正面词: {data['positive_words']}")
            if data['negative_words']:
                print(f"      负面词: {data['negative_words']}")
        print()


def test_domain_specific_words():
    print("=== 领域专用词汇测试 ===")

    print("--- 电子产品领域 ---")
    test_words_electronics = [
        ("这个手机屏幕很清晰", "electronics"),
        ("电池很耐用", "electronics"),
        ("玩游戏很流畅不卡顿", "electronics"),
    ]
    for text, domain in test_words_electronics:
        result_general = analyze_sentiment(text, domain="general")
        result_domain = analyze_sentiment(text, domain=domain)
        print(f"文本: {text}")
        print(f"  通用领域: {result_general['sentiment']} ({result_general['score']})")
        print(f"  电子领域: {result_domain['sentiment']} ({result_domain['score']})")
    print()

    print("--- 餐饮领域 ---")
    test_words_restaurant = [
        ("这道菜很入味", "restaurant"),
        ("肉质很鲜嫩", "restaurant"),
        ("环境很雅致", "restaurant"),
    ]
    for text, domain in test_words_restaurant:
        result_general = analyze_sentiment(text, domain="general")
        result_domain = analyze_sentiment(text, domain=domain)
        print(f"文本: {text}")
        print(f"  通用领域: {result_general['sentiment']} ({result_general['score']})")
        print(f"  餐饮领域: {result_domain['sentiment']} ({result_domain['score']})")
    print()

    print("--- 金融领域 ---")
    test_words_finance = [
        ("这只基金最近涨得不错", "finance"),
        ("收益很稳定", "finance"),
        ("这个平台很靠谱", "finance"),
    ]
    for text, domain in test_words_finance:
        result_general = analyze_sentiment(text, domain="general")
        result_domain = analyze_sentiment(text, domain=domain)
        print(f"文本: {text}")
        print(f"  通用领域: {result_general['sentiment']} ({result_general['score']})")
        print(f"  金融领域: {result_domain['sentiment']} ({result_domain['score']})")
    print()


def test_general_aspects():
    print("=== 通用方面级分析（跨领域） ===")
    test_cases = [
        "质量很好，就是价格有点贵",
        "服务态度不错，物流也很快",
        "外观设计很漂亮，做工也精致",
        "功能一般，但是性价比高",
        "包装完好，品质也不错，值得购买",
    ]

    for text in test_cases:
        result = analyze_sentiment_with_aspects(text, domain="general")
        print(f"文本: {text}")
        print(f"  整体情感: {result['sentiment']} ({result['score']})")
        print(f"  方面分析:")
        for aspect, data in result['aspects'].items():
            print(f"    - {aspect}: {data['sentiment']} ({data['score']})")
        print()


if __name__ == '__main__':
    test_domain_list()
    test_aspect_electronics()
    test_aspect_restaurant()
    test_aspect_finance()
    test_aspect_clothing()
    test_domain_specific_words()
    test_general_aspects()
    print("=== 所有测试完成 ===")
