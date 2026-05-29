import re
from collections import OrderedDict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="情感分析API", version="1.0.0")

POSITIVE_WORDS = {
    "开心": 0.9, "高兴": 0.9, "快乐": 0.9, "愉快": 0.8, "幸福": 0.95,
    "满意": 0.7, "喜欢": 0.8, "爱": 0.9, "美好": 0.8, "优秀": 0.8,
    "出色": 0.8, "棒": 0.8, "好": 0.6, "不错": 0.6, "赞": 0.8,
    "感谢": 0.7, "感激": 0.7, "感动": 0.75, "温暖": 0.7, "温馨": 0.7,
    "希望": 0.5, "期待": 0.6, "欣赏": 0.7, "赞赏": 0.8, "敬佩": 0.8,
    "成功": 0.8, "胜利": 0.8, "进步": 0.7, "改善": 0.6, "提升": 0.6,
    "漂亮": 0.7, "美丽": 0.8, "可爱": 0.7, "精彩": 0.8, "完美": 0.95,
    "善良": 0.8, "友好": 0.7, "真诚": 0.7, "诚实": 0.7, "勇敢": 0.8,
    "乐观": 0.7, "积极": 0.7, "热情": 0.7, "活力": 0.6, "阳光": 0.7,
    "舒适": 0.6, "安心": 0.6, "放心": 0.5, "信任": 0.6, "尊重": 0.7,
    "惊喜": 0.8, "庆幸": 0.7, "自豪": 0.8, "骄傲": 0.7, "得意": 0.6,
    "甜蜜": 0.8, "浪漫": 0.7, "和谐": 0.7, "繁荣": 0.7, "丰富": 0.5,
    "高效": 0.6, "便捷": 0.6, "实用": 0.5, "方便": 0.5, "划算": 0.6,
    "值得": 0.6, "推荐": 0.6, "受益": 0.6, "帮助": 0.4,
    "快": 0.6, "便宜": 0.6, "持久": 0.7, "强劲": 0.7, "正宗": 0.7,
    "新鲜": 0.7, "美味": 0.8, "清晰": 0.7, "细腻": 0.7, "稳健": 0.7,
    "好看": 0.6, "好听": 0.6, "好吃": 0.7,
    "good": 0.7, "great": 0.8, "excellent": 0.9, "amazing": 0.9,
    "wonderful": 0.9, "fantastic": 0.9, "happy": 0.8, "love": 0.9,
    "like": 0.6, "best": 0.9, "perfect": 0.95, "beautiful": 0.8,
    "nice": 0.7, "awesome": 0.85, "brilliant": 0.85, "fast": 0.6,
    "cheap": 0.5, "delicious": 0.8, "fresh": 0.7,
}

NEGATIVE_WORDS = {
    "悲伤": -0.9, "难过": -0.8, "痛苦": -0.9, "伤心": -0.85, "失望": -0.7,
    "讨厌": -0.8, "恨": -0.9, "愤怒": -0.9, "生气": -0.8, "烦躁": -0.7,
    "糟糕": -0.8, "差": -0.6, "坏": -0.7, "烂": -0.8, "恶心": -0.85,
    "可怕": -0.8, "恐惧": -0.9, "害怕": -0.7, "担心": -0.5, "焦虑": -0.7,
    "无聊": -0.5, "郁闷": -0.7, "沮丧": -0.8, "绝望": -0.95, "崩溃": -0.9,
    "失败": -0.8, "挫折": -0.7, "困难": -0.5, "麻烦": -0.5, "问题": -0.3,
    "错误": -0.6, "缺陷": -0.6, "不足": -0.5, "缺乏": -0.5, "缺失": -0.6,
    "危险": -0.7, "威胁": -0.7, "危害": -0.8, "损害": -0.7, "破坏": -0.8,
    "消极": -0.6, "悲观": -0.7, "冷漠": -0.6, "孤独": -0.6, "寂寞": -0.6,
    "后悔": -0.7, "遗憾": -0.5, "抱歉": -0.5, "内疚": -0.6, "羞愧": -0.7,
    "紧张": -0.5, "压力": -0.6, "疲惫": -0.6, "劳累": -0.5, "厌倦": -0.6,
    "丑陋": -0.7, "恶劣": -0.8, "肮脏": -0.7, "混乱": -0.6,
    "欺骗": -0.8, "背叛": -0.9, "虚伪": -0.8, "自私": -0.7, "贪婪": -0.7,
    "慢": -0.6, "贵": -0.5, "久": -0.5, "昂贵": -0.6, "难吃": -0.8,
    "卡顿": -0.7, "闪退": -0.8, "波动": -0.5, "难看": -0.6, "难听": -0.6,
    "bad": -0.7, "terrible": -0.9, "horrible": -0.9, "awful": -0.85,
    "hate": -0.9, "angry": -0.8, "sad": -0.8, "worst": -0.9,
    "ugly": -0.7, "disgusting": -0.85, "disappointing": -0.7,
    "poor": -0.6, "fail": -0.7, "boring": -0.5, "annoying": -0.7,
    "slow": -0.6, "expensive": -0.5, "bad": -0.7,
}

NEGATION_WORDS = {
    "不", "没", "没有", "无", "非", "别", "未", "莫", "勿", "否",
    "毫不", "绝不", "从不", "从未", "并非", "不是", "不能", "不会",
    "不再", "不够", "不怎么", "不太", "不大",
    "not", "no", "never", "neither", "nor", "nobody", "nothing",
    "nowhere", "hardly", "barely", "scarcely", "don't", "doesn't",
    "didn't", "won't", "wouldn't", "couldn't", "shouldn't",
    "isn't", "aren't", "wasn't", "weren't", "can't", "cannot",
}

DEGREE_WORDS = OrderedDict([
    ("极其|极度|极为|万分", 2.0),
    ("非常|特别|十分|异常|相当|格外|太", 1.75),
    ("很|挺|颇|甚|殊|巨", 1.5),
    ("较|比较|较为|有些|有点儿|有点", 1.25),
    ("稍微|略微|稍|稍稍|微", 0.75),
    ("一点点|一丝", 0.5),
])


ASPECT_WORDS_GENERAL = [
    "质量", "品质", "性价比", "价格", "外观", "设计", "功能", "性能",
    "服务", "售后", "客服", "物流", "包装", "效果", "体验", "感受",
]

ASPECT_WORDS_ECOMMERCE = [
    "屏幕", "电池", "续航", "摄像头", "拍照", "处理器", "内存", "存储",
    "系统", "流畅度", "发热", "信号", "音质", "手感", "重量", "厚度",
    "边框", "分辨率", "刷新率", "快充", "充电", "指纹", "面部识别",
    "防水", "耐用", "做工", "材质", "品牌", "口碑",
] + ASPECT_WORDS_GENERAL

ASPECT_WORDS_RESTAURANT = [
    "口味", "味道", "菜品", "菜量", "食材", "新鲜", "摆盘", "创意",
    "环境", "装修", "氛围", "座位", "排队", "停车", "噪音", "卫生",
    "服务", "服务员", "上菜速度", "态度", "热情", "专业",
    "价格", "性价比", "人均", "优惠", "团购", "代金券",
    "推荐", "招牌", "特色", "回头客",
] + ASPECT_WORDS_GENERAL

ASPECT_WORDS_FINANCE = [
    "收益", "回报率", "利率", "利息", "分红", "涨幅", "盈利", "增值",
    "风险", "波动", "亏损", "回撤", "爆仓", "清盘", "违约", "逾期",
    "手续费", "佣金", "管理费", "托管费", "税费", "成本",
    "服务", "客户经理", "客服", "响应速度", "专业度",
    "安全", "资金安全", "隐私", "合规", "监管", "牌照",
    "产品", "基金", "股票", "债券", "保险", "理财", "信托",
    "流动性", "提现", "赎回", "到账", "门槛", "期限",
] + ASPECT_WORDS_GENERAL

DOMAIN_LEXICONS = {
    "general": {
        "positive": POSITIVE_WORDS,
        "negative": NEGATIVE_WORDS,
        "aspects": ASPECT_WORDS_GENERAL,
        "positive_domain": {},
        "negative_domain": {},
    },
    "ecommerce": {
        "positive": {
            **POSITIVE_WORDS,
            "清晰": 0.7, "细腻": 0.7, "鲜艳": 0.7, "耐用": 0.6,
            "省电": 0.6, "静音": 0.5, "轻薄": 0.5, "智能": 0.6,
        },
        "negative": {
            **NEGATIVE_WORDS,
            "卡顿": -0.7, "掉帧": -0.6, "闪退": -0.8, "死机": -0.9,
            "耗电": -0.7, "发烫": -0.7, "漏光": -0.6, "色差": -0.5,
            "模糊": -0.6, "虚假宣传": -0.8, "翻新": -0.8,
        },
        "aspects": ASPECT_WORDS_ECOMMERCE,
        "positive_domain": {"清晰": 0.7, "细腻": 0.7, "鲜艳": 0.7},
        "negative_domain": {"卡顿": -0.7, "掉帧": -0.6, "闪退": -0.8},
    },
    "restaurant": {
        "positive": {
            **POSITIVE_WORDS,
            "美味": 0.8, "鲜美": 0.8, "酥脆": 0.6, "嫩滑": 0.7,
            "入味": 0.6, "正宗": 0.7, "地道": 0.7, "新鲜": 0.7,
            "干净": 0.7, "整洁": 0.6, "雅致": 0.6, "幽静": 0.5,
            "实惠": 0.6, "超值": 0.7,
        },
        "negative": {
            **NEGATIVE_WORDS,
            "难吃": -0.8, "寡淡": -0.6, "油腻": -0.6, "咸": -0.4,
            "淡": -0.4, "不新鲜": -0.7, "变质": -0.8, "异味": -0.7,
            "慢": -0.5, "慢": -0.5, "插队": -0.6, "拒单": -0.8,
            "不卫生": -0.8, "脏乱": -0.7, "拥挤": -0.5,
        },
        "aspects": ASPECT_WORDS_RESTAURANT,
        "positive_domain": {"美味": 0.8, "鲜美": 0.8, "正宗": 0.7},
        "negative_domain": {"难吃": -0.8, "不新鲜": -0.7, "变质": -0.8},
    },
    "finance": {
        "positive": {
            **POSITIVE_WORDS,
            "稳健": 0.7, "稳定": 0.6, "安全": 0.7, "可靠": 0.6,
            "高效": 0.6, "便捷": 0.6, "灵活": 0.5, "透明": 0.6,
            "合规": 0.6, "专业": 0.7, "严谨": 0.6, "省心": 0.6,
            "高收益": 0.8, "低风险": 0.7, "保本": 0.6,
            "收益高": 0.8, "风险低": 0.7, "回报高": 0.8,
        },
        "negative": {
            **NEGATIVE_WORDS,
            "亏损": -0.8, "暴跌": -0.9, "爆雷": -0.95, "跑路": -0.95,
            "诈骗": -0.95, "非法集资": -0.95, "违规": -0.7,
            "高风险": -0.8, "不透明": -0.7, "套路": -0.7,
            "冻结": -0.8, "限制": -0.6, "强制平仓": -0.9,
            "风险高": -0.8, "手续费高": -0.7, "费用高": -0.7,
            "利息高": -0.6, "波动大": -0.6, "风险大": -0.8,
        },
        "aspects": ASPECT_WORDS_FINANCE,
        "positive_domain": {"稳健": 0.7, "高收益": 0.8, "低风险": 0.7, "收益高": 0.8, "风险低": 0.7},
        "negative_domain": {"亏损": -0.8, "爆雷": -0.95, "跑路": -0.95, "高风险": -0.8, "风险高": -0.8, "手续费高": -0.7},
    },
}

AVAILABLE_DOMAINS = list(DOMAIN_LEXICONS.keys())


def _build_degree_map():
    result = {}
    for pattern, weight in DEGREE_WORDS.items():
        for word in pattern.split("|"):
            result[word] = weight
    return result


DEGREE_MAP = _build_degree_map()

NEGATION_WINDOW = 4
SCORE_THRESHOLD = 0.05
NEGATION_STRENGTH = 0.6
ASPECT_WINDOW = 5

CONJUNCTION_WORDS = {
    "但", "但是", "可是", "可", "却", "然而", "不过", "只是",
    "反倒", "反而", "虽然", "尽管", "即便", "即使", "而", "but", "however", "yet", "though", "although",
}
CONJUNCTION_BREAK = True


def _get_domain_lexicon(domain: str = "general") -> dict:
    if domain not in DOMAIN_LEXICONS:
        domain = "general"
    return DOMAIN_LEXICONS[domain]


def _build_token_set(domain: str = "general") -> set:
    lex = _get_domain_lexicon(domain)
    positive = lex["positive"]
    negative = lex["negative"]
    aspect_set = set(lex["aspects"])
    tokens = set(positive.keys()) | set(negative.keys()) | NEGATION_WORDS | set(DEGREE_MAP.keys()) | aspect_set
    return tokens


def _simple_tokenize(text: str, domain: str = "general") -> list[dict]:
    lex = _get_domain_lexicon(domain)
    positive = lex["positive"]
    negative = lex["negative"]
    aspects = set(lex["aspects"])
    all_token_set = set(positive.keys()) | set(negative.keys()) | NEGATION_WORDS | set(DEGREE_MAP.keys()) | aspects

    tokens = []
    i = 0
    n = len(text)
    while i < n:
        matched = False
        for length in range(min(6, n - i), 0, -1):
            word = text[i : i + length]
            if word in all_token_set:
                token_type = "aspect" if word in aspects else (
                    "positive" if word in positive else (
                        "negative" if word in negative else (
                            "negation" if word in NEGATION_WORDS else "degree"
                        )
                    )
                )
                tokens.append({
                    "text": word,
                    "type": token_type,
                    "start": i,
                    "end": i + length,
                })
                i += length
                matched = True
                break
        if not matched:
            if text[i].strip():
                tokens.append({
                    "text": text[i],
                    "type": "other",
                    "start": i,
                    "end": i + 1,
                })
            i += 1
    return tokens


def _compute_sentiment_score(
    sentiment_token: dict,
    tokens: list[dict],
    idx: int,
    positive_words: dict,
    negative_words: dict,
    sentiment_words: dict,
) -> dict:
    base_word = sentiment_token["text"]
    base_score = positive_words.get(base_word, negative_words.get(base_word, 0.0))

    negation_count = 0
    degree_word = None
    negation_words_found = []

    window_start = max(0, idx - NEGATION_WINDOW)
    for j in range(idx - 1, window_start - 1, -1):
        prev = tokens[j]
        prev_text = prev["text"]
        if prev_text in NEGATION_WORDS:
            negation_count += 1
            negation_words_found.append(prev_text)
        elif prev_text in sentiment_words:
            break
        elif prev_text in DEGREE_MAP and degree_word is None:
            degree_word = prev_text

    if degree_word is not None:
        base_score = base_score * DEGREE_MAP[degree_word]

    if negation_count > 0:
        if negation_count % 2 == 1:
            base_score = -base_score * NEGATION_STRENGTH

    final_score = max(-1.0, min(1.0, base_score))

    return {
        "word": base_word,
        "token_index": idx,
        "base_score": round(positive_words.get(base_word, negative_words.get(base_word, 0.0)), 4),
        "degree_word": degree_word,
        "negation_words": negation_words_found,
        "negation_count": negation_count,
        "final_score": round(final_score, 4),
    }


def _analyze_context_sentiment(
    tokens: list[dict],
    positive_words: dict,
    negative_words: dict,
    sentiment_words: dict,
    domain: str,
) -> dict:
    context_results = {}

    if domain == "finance":
        positive_aspects = {"收益", "盈利", "涨幅", "回报", "利息", "分红", "利率"}
        negative_aspects = {"手续费", "费用", "成本", "回撤", "亏损", "逾期"}
        special_aspects = {"风险", "波动"}
        positive_indicators = {"高", "大", "多", "快", "强", "低", "小", "少"}
        negative_indicators = {"高", "大", "多", "快", "差"}
        special_negative_indicators = {"高", "大", "多", "快"}
        special_positive_indicators = {"低", "小", "少", "稳"}

        for idx, token in enumerate(tokens):
            if token["text"] in positive_aspects:
                for j in range(idx + 1, min(len(tokens), idx + 4)):
                    next_tok = tokens[j]
                    if next_tok["text"] in positive_indicators and next_tok["type"] == "other":
                        base_score = 0.8
                        degree_word = None
                        for k in range(idx + 1, j):
                            if tokens[k]["text"] in DEGREE_MAP:
                                degree_word = tokens[k]["text"]
                                base_score = base_score * DEGREE_MAP[degree_word]
                                break
                        final_score = max(-1.0, min(1.0, base_score))
                        context_results[j] = {
                            "word": next_tok["text"],
                            "token_index": j,
                            "base_score": 0.8,
                            "degree_word": degree_word,
                            "negation_words": [],
                            "negation_count": 0,
                            "final_score": round(final_score, 4),
                            "aspect": token["text"],
                            "context_source": "finance_positive",
                        }
                        break
                    elif next_tok["text"] in sentiment_words or next_tok["text"] in NEGATION_WORDS:
                        break

            elif token["text"] in negative_aspects:
                for j in range(idx + 1, min(len(tokens), idx + 4)):
                    next_tok = tokens[j]
                    if next_tok["text"] in negative_indicators and next_tok["type"] == "other":
                        base_score = -0.7
                        degree_word = None
                        for k in range(idx + 1, j):
                            if tokens[k]["text"] in DEGREE_MAP:
                                degree_word = tokens[k]["text"]
                                base_score = base_score * DEGREE_MAP[degree_word]
                                break
                        final_score = max(-1.0, min(1.0, base_score))
                        context_results[j] = {
                            "word": next_tok["text"],
                            "token_index": j,
                            "base_score": -0.7,
                            "degree_word": degree_word,
                            "negation_words": [],
                            "negation_count": 0,
                            "final_score": round(final_score, 4),
                            "aspect": token["text"],
                            "context_source": "finance_negative",
                        }
                        break
                    elif next_tok["text"] in sentiment_words or next_tok["text"] in NEGATION_WORDS:
                        break

            elif token["text"] in special_aspects:
                for j in range(idx + 1, min(len(tokens), idx + 4)):
                    next_tok = tokens[j]
                    if next_tok["type"] == "other":
                        base_score = None
                        if next_tok["text"] in special_positive_indicators:
                            base_score = 0.7
                        elif next_tok["text"] in special_negative_indicators:
                            base_score = -0.7
                        if base_score is not None:
                            degree_word = None
                            for k in range(idx + 1, j):
                                if tokens[k]["text"] in DEGREE_MAP:
                                    degree_word = tokens[k]["text"]
                                    base_score = base_score * DEGREE_MAP[degree_word]
                                    break
                            final_score = max(-1.0, min(1.0, base_score))
                            context_results[j] = {
                                "word": next_tok["text"],
                                "token_index": j,
                                "base_score": base_score,
                                "degree_word": degree_word,
                                "negation_words": [],
                                "negation_count": 0,
                                "final_score": round(final_score, 4),
                                "aspect": token["text"],
                                "context_source": "finance_special",
                            }
                            break
                    elif next_tok["text"] in sentiment_words or next_tok["text"] in NEGATION_WORDS:
                        break

    return context_results


def analyze_sentiment(text: str, domain: str = "general") -> dict:
    if not text or not text.strip():
        return {"label": "中性", "score": 0.0, "details": {}, "aspects": [], "domain": domain}

    lex = _get_domain_lexicon(domain)
    positive_words = lex["positive"]
    negative_words = lex["negative"]
    sentiment_words = {**positive_words, **negative_words}

    tokens = _simple_tokenize(text, domain)
    total_score = 0.0
    matched_count = 0
    pos_count = 0
    neg_count = 0
    hits = []

    sentiment_results = {}

    context_hits = _analyze_context_sentiment(tokens, positive_words, negative_words, sentiment_words, domain)
    for idx, result in context_hits.items():
        sentiment_results[idx] = result
        hits.append(result)
        total_score += result["final_score"]
        matched_count += 1
        if result["final_score"] > 0:
            pos_count += 1
        elif result["final_score"] < 0:
            neg_count += 1

    for idx, token in enumerate(tokens):
        if idx in sentiment_results:
            continue
        if token["type"] in ("positive", "negative"):
            result = _compute_sentiment_score(token, tokens, idx, positive_words, negative_words, sentiment_words)
            sentiment_results[idx] = result
            hits.append(result)
            total_score += result["final_score"]
            matched_count += 1
            if result["final_score"] > 0:
                pos_count += 1
            elif result["final_score"] < 0:
                neg_count += 1

    if matched_count > 0:
        raw_avg = total_score / matched_count
        confidence = min(matched_count / 5.0, 1.0)
        score = raw_avg * confidence
    else:
        score = 0.0

    score = round(max(-1.0, min(1.0, score)), 4)

    if score > SCORE_THRESHOLD:
        label = "正面"
    elif score < -SCORE_THRESHOLD:
        label = "负面"
    else:
        label = "中性"

    aspect_results = []
    for idx, token in enumerate(tokens):
        if token["type"] != "aspect":
            continue

        aspect_word = token["text"]
        window_start = max(0, idx - ASPECT_WINDOW)
        window_end = min(len(tokens), idx + ASPECT_WINDOW + 1)

        related_sentiment = []
        total_aspect_score = 0.0

        left_bound = window_start
        for j in range(idx - 1, window_start - 1, -1):
            if tokens[j]["text"] in CONJUNCTION_WORDS:
                left_bound = j + 1
                break

        right_bound = window_end
        for j in range(idx + 1, window_end):
            if tokens[j]["text"] in CONJUNCTION_WORDS:
                right_bound = j
                break

        for j in range(left_bound, right_bound):
            if j == idx:
                continue
            if j in sentiment_results:
                sent = sentiment_results[j]
                distance = abs(j - idx)
                weight = 1.0 / (1.0 + distance * 0.1)
                weighted_score = sent["final_score"] * weight
                related_sentiment.append({
                    "sentiment_word": sent["word"],
                    "original_score": sent["final_score"],
                    "distance": distance,
                    "weight": round(weight, 4),
                    "weighted_score": round(weighted_score, 4),
                })
                total_aspect_score += weighted_score

        if related_sentiment:
            avg_aspect_score = total_aspect_score / len(related_sentiment)
            avg_aspect_score = round(max(-1.0, min(1.0, avg_aspect_score)), 4)
        else:
            avg_aspect_score = 0.0

        if avg_aspect_score > SCORE_THRESHOLD:
            aspect_label = "正面"
        elif avg_aspect_score < -SCORE_THRESHOLD:
            aspect_label = "负面"
        else:
            aspect_label = "中性"

        aspect_results.append({
            "aspect": aspect_word,
            "token_index": idx,
            "label": aspect_label,
            "score": avg_aspect_score,
            "related_sentiment": related_sentiment,
        })

    token_texts = [t["text"] for t in tokens]

    return {
        "label": label,
        "score": score,
        "domain": domain,
        "details": {
            "tokens": token_texts,
            "token_details": tokens,
            "sentiment_hits": hits,
            "positive_count": pos_count,
            "negative_count": neg_count,
        },
        "aspects": aspect_results,
    }


class SentimentRequest(BaseModel):
    text: str
    domain: str = "general"


class SentimentResponse(BaseModel):
    label: str
    score: float
    domain: str
    details: dict
    aspects: list


@app.get("/domains")
async def list_domains():
    result = []
    for domain_name, lex in DOMAIN_LEXICONS.items():
        result.append({
            "domain": domain_name,
            "aspects": lex["aspects"][:20],
            "positive_domain_words": list(lex.get("positive_domain", {}).keys()),
            "negative_domain_words": list(lex.get("negative_domain", {}).keys()),
            "aspect_count": len(lex["aspects"]),
            "positive_count": len(lex["positive"]),
            "negative_count": len(lex["negative"]),
        })
    return {"domains": result, "available": AVAILABLE_DOMAINS}


@app.post("/analyze", response_model=SentimentResponse)
async def analyze_endpoint(req: SentimentRequest):
    if req.domain not in AVAILABLE_DOMAINS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid domain '{req.domain}'. Available domains: {AVAILABLE_DOMAINS}"
        )
    result = analyze_sentiment(req.text, req.domain)
    return result


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
