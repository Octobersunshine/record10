from flask import Flask, request, jsonify
from pypinyin import pinyin, Style
from opencc import OpenCC
import jieba

app = Flask(__name__)

cc_s2t = OpenCC("s2t")
cc_t2s = OpenCC("t2s")

COMMON_CHARS = [
    "啊阿呵嗄腌",
    "八吧把爸拔霸罢扒叭疤笆粑靶捌跋鲅",
    "才菜财猜材采彩踩睬",
    "大答打达代带袋待戴贷歹",
    "二而尔儿耳饵贰",
    "发法罚乏伐筏阀",
    "个各格哥歌隔革阁葛",
    "和喝河何合禾盒贺核赫",
    "一以已依医衣益意艺亿",
    "几及急级即集记计季寄",
    "可克科课颗壳渴刻咳",
    "了乐勒雷类泪累垒",
    "吗妈马麻骂码嘛抹",
    "你呢泥拟逆匿溺",
    "哦噢喔呕偶藕",
    "怕帕爬琶帕葩",
    "七起其气期齐妻奇",
    "日入",
    "三散伞叁",
    "是时事市十石实识史",
    "他她它塔踏塌挞",
    "五无伍午武舞务物",
    "西洗系细喜戏吸希",
    "一以已亿意益艺易",
    "在再载灾载栽",
    "你理里李礼丽历力立",
]

PINYIN_TO_CHARS = {}

def _init_pinyin_map():
    from pypinyin import lazy_pinyin, Style
    for group in COMMON_CHARS:
        for char in group:
            pys = lazy_pinyin(char, style=Style.NORMAL)
            if pys:
                py = pys[0]
                if py not in PINYIN_TO_CHARS:
                    PINYIN_TO_CHARS[py] = []
                if char not in PINYIN_TO_CHARS[py]:
                    PINYIN_TO_CHARS[py].append(char)

_init_pinyin_map()

def detect_script_type(text):
    simplified = set()
    traditional = set()
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            s = cc_t2s.convert(char)
            t = cc_s2t.convert(char)
            if char == s and char != t:
                simplified.add(char)
            elif char == t and char != s:
                traditional.add(char)
    has_s = len(simplified) > 0
    has_t = len(traditional) > 0
    if has_s and has_t:
        return "mixed"
    elif has_s:
        return "simplified"
    elif has_t:
        return "traditional"
    else:
        return "unknown"

def unify_chinese(text, target):
    if target == "simplified":
        return cc_t2s.convert(text)
    elif target == "traditional":
        return cc_s2t.convert(text)
    return text

BUILTIN_POLYPHONE = {
    "长尺子": ["cháng", "chǐ", "zi"],
    "长尺": ["cháng", "chǐ"],
    "好还": ["hǎo", "huán"],
    "还看": ["hái", "kàn"],
    "还书": ["huán", "shū"],
    "还钱": ["huán", "qián"],
    "还在": ["hái", "zài"],
    "还是": ["hái", "shì"],
    "还好": ["hái", "hǎo"],
    "还有": ["hái", "yǒu"],
    "还要": ["hái", "yào"],
    "还差": ["hái", "chà"],
    "还没": ["hái", "méi"],
    "借还": ["jiè", "huán"],
    "偿还": ["cháng", "huán"],
    "归还": ["guī", "huán"],
    "送还": ["sòng", "huán"],
    "退还": ["tuì", "huán"],
    "交还": ["jiāo", "huán"],
    "还击": ["huán", "jī"],
    "还价": ["huán", "jià"],
    "还口": ["huán", "kǒu"],
    "还礼": ["huán", "lǐ"],
    "还清": ["huán", "qīng"],
    "还手": ["huán", "shǒu"],
    "还愿": ["huán", "yuàn"],
    "还债": ["huán", "zhài"],
    "还账": ["huán", "zhàng"],
    "还嘴": ["huán", "zuǐ"],
    "以牙还牙": ["yǐ", "yá", "huán", "yá"],
    "买椟还珠": ["mǎi", "dú", "huán", "zhū"],
    "讨价还价": ["tǎo", "jià", "huán", "jià"],
}

custom_polyphone = {}

for phrase in BUILTIN_POLYPHONE:
    jieba.add_word(phrase)


def style_to_pinyin_style(fmt):
    if fmt == "tone":
        return Style.TONE
    elif fmt == "none":
        return Style.NORMAL
    elif fmt == "capital":
        return Style.FIRST_LETTER
    return None


def apply_format(pinyin_list, fmt):
    if fmt == "tone":
        return " ".join(pinyin_list)
    elif fmt == "none":
        from pypinyin import lazy_pinyin
        return " ".join(lazy_pinyin("".join([Style.TONE2.convert(item) if hasattr(Style, 'TONE2') else item for item in pinyin_list]), style=Style.NORMAL))
    elif fmt == "capital":
        return "".join([item[0].upper() for item in pinyin_list])
    return None


def pinyin_to_format(orig_text, fmt):
    style = style_to_pinyin_style(fmt)
    if style is None:
        return None
    result = pinyin(orig_text, style=style)
    items = [item[0] for item in result]
    if fmt == "capital":
        return "".join(items)
    return " ".join(items)


def lookup_dict(phrase, dicts):
    for d in dicts:
        if phrase in d:
            return d[phrase]
    return None


def max_forward_match(text, all_dict):
    results = []
    i = 0
    n = len(text)
    max_len = max((len(k) for k in all_dict.keys()), default=1)

    while i < n:
        matched = False
        for l in range(min(max_len, n - i), 0, -1):
            substr = text[i:i + l]
            if substr in all_dict:
                source, pinyin_list = all_dict[substr]
                results.append({"segment": substr, "source": source, "pinyin": pinyin_list, "matched": True})
                i += l
                matched = True
                break
        if not matched:
            results.append({"segment": text[i], "source": "none", "pinyin": None, "matched": False})
            i += 1

    merged = []
    current_unmatched = ""
    for item in results:
        if item["matched"]:
            if current_unmatched:
                merged.append({"segment": current_unmatched, "source": "unmatched", "pinyin": None, "matched": False})
                current_unmatched = ""
            merged.append(item)
        else:
            current_unmatched += item["segment"]
    if current_unmatched:
        merged.append({"segment": current_unmatched, "source": "unmatched", "pinyin": None, "matched": False})

    return merged


def convert_pinyin_smart(text, fmt):
    style = style_to_pinyin_style(fmt)
    if style is None:
        return None, None

    all_dict = {}
    for phrase, py in custom_polyphone.items():
        all_dict[phrase] = ("custom", py)
    for phrase, py in BUILTIN_POLYPHONE.items():
        if phrase not in all_dict:
            all_dict[phrase] = ("builtin", py)

    matched_segments = max_forward_match(text, all_dict)

    final_pinyin = []
    debug_info = []

    for seg_info in matched_segments:
        seg = seg_info["segment"]
        if seg_info["matched"]:
            debug_info.append({
                "segment": seg,
                "source": seg_info["source"],
                "pinyin": seg_info["pinyin"]
            })
            final_pinyin.extend(seg_info["pinyin"])
            continue

        sub_segs = list(jieba.cut(seg))
        for sub_seg in sub_segs:
            result = pinyin(sub_seg, style=Style.TONE)
            seg_pinyin = [item[0] for item in result]
            debug_info.append({
                "segment": sub_seg,
                "source": "pypinyin",
                "pinyin": seg_pinyin
            })
            final_pinyin.extend(seg_pinyin)

    if fmt == "tone":
        formatted = " ".join(final_pinyin)
    elif fmt == "none":
        from pypinyin import lazy_pinyin
        formatted = " ".join(lazy_pinyin(text, style=Style.NORMAL))
    elif fmt == "capital":
        formatted = "".join([item[0].upper() for item in final_pinyin])
    else:
        return None, None

    return formatted, debug_info


def convert_chinese(text, direction):
    if direction == "s2t":
        return cc_s2t.convert(text)
    elif direction == "t2s":
        return cc_t2s.convert(text)
    else:
        return None


@app.route("/pinyin", methods=["POST"])
def pinyin_api():
    data = request.get_json(force=True)
    text = data.get("text", "")
    texts = data.get("texts", [])
    fmt = data.get("format", "tone")
    show_debug = data.get("debug", False)

    valid_formats = ("tone", "none", "capital")
    if fmt not in valid_formats:
        return jsonify({"error": f"format must be one of {valid_formats}"}), 400

    if texts:
        results = []
        for item in texts:
            result, debug_info = convert_pinyin_smart(item, fmt)
            item_result = {"text": item, "pinyin": result}
            if show_debug:
                item_result["debug"] = debug_info
            results.append(item_result)
        return jsonify({"results": results})

    if not text:
        return jsonify({"error": "text or texts field is required"}), 400

    result, debug_info = convert_pinyin_smart(text, fmt)
    if result is None:
        return jsonify({"error": "pinyin conversion failed"}), 500

    resp = {"text": text, "format": fmt, "pinyin": result}
    if show_debug:
        resp["debug"] = debug_info
    return jsonify(resp)


@app.route("/convert", methods=["POST"])
def convert_api():
    data = request.get_json(force=True)
    text = data.get("text", "")
    texts = data.get("texts", [])
    direction = data.get("direction", "s2t")

    valid_dirs = ("s2t", "t2s")
    if direction not in valid_dirs:
        return jsonify({"error": f"direction must be one of {valid_dirs}"}), 400

    if texts:
        results = []
        for item in texts:
            result = convert_chinese(item, direction)
            results.append({"text": item, "result": result})
        return jsonify({"results": results})

    if not text:
        return jsonify({"error": "text or texts field is required"}), 400

    result = convert_chinese(text, direction)
    if result is None:
        return jsonify({"error": "conversion failed"}), 500

    label = "simplified_to_traditional" if direction == "s2t" else "traditional_to_simplified"
    return jsonify({"text": text, "direction": label, "result": result})


@app.route("/convert/unify", methods=["POST"])
def unify_api():
    data = request.get_json(force=True)
    text = data.get("text", "")
    texts = data.get("texts", [])
    target = data.get("target", "simplified")

    valid_targets = ("simplified", "traditional")
    if target not in valid_targets:
        return jsonify({"error": f"target must be one of {valid_targets}"}), 400

    if texts:
        results = []
        for item in texts:
            script_type = detect_script_type(item)
            unified = unify_chinese(item, target)
            results.append({
                "text": item,
                "detected_type": script_type,
                "result": unified
            })
        return jsonify({"results": results})

    if not text:
        return jsonify({"error": "text or texts field is required"}), 400

    script_type = detect_script_type(text)
    unified = unify_chinese(text, target)

    return jsonify({
        "text": text,
        "detected_type": script_type,
        "target": target,
        "result": unified
    })


@app.route("/polyphone/builtin", methods=["GET"])
def list_builtin():
    return jsonify({
        "count": len(BUILTIN_POLYPHONE),
        "data": BUILTIN_POLYPHONE
    })


@app.route("/polyphone/custom", methods=["GET"])
def list_custom():
    return jsonify({
        "count": len(custom_polyphone),
        "data": custom_polyphone
    })


@app.route("/polyphone/custom", methods=["POST"])
def add_custom():
    data = request.get_json(force=True)
    phrase = data.get("phrase", "")
    pinyin_list = data.get("pinyin", [])

    if not phrase:
        return jsonify({"error": "phrase field is required"}), 400
    if not isinstance(pinyin_list, list) or len(pinyin_list) == 0:
        return jsonify({"error": "pinyin must be a non-empty list"}), 400
    if len(pinyin_list) != len(phrase):
        return jsonify({"error": f"pinyin list length ({len(pinyin_list)}) must match phrase length ({len(phrase)})"}), 400

    custom_polyphone[phrase] = pinyin_list
    jieba.add_word(phrase)
    return jsonify({
        "status": "added",
        "phrase": phrase,
        "pinyin": pinyin_list
    })


@app.route("/polyphone/custom/<phrase>", methods=["DELETE"])
def delete_custom(phrase):
    if phrase not in custom_polyphone:
        return jsonify({"error": f"phrase '{phrase}' not found in custom dictionary"}), 404
    removed = custom_polyphone.pop(phrase)
    return jsonify({
        "status": "deleted",
        "phrase": phrase,
        "pinyin": removed
    })


@app.route("/pinyin/match", methods=["POST"])
def pinyin_match():
    data = request.get_json(force=True)
    pinyin_query = data.get("pinyin", "")
    fuzzy = data.get("fuzzy", True)

    if not pinyin_query:
        return jsonify({"error": "pinyin field is required"}), 400

    pinyin_query = pinyin_query.lower()
    matched = []

    for py, chars in PINYIN_TO_CHARS.items():
        if fuzzy:
            if py.startswith(pinyin_query) or pinyin_query in py:
                matched.append({"pinyin": py, "chars": chars})
        else:
            if py == pinyin_query:
                matched.append({"pinyin": py, "chars": chars})

    matched.sort(key=lambda x: x["pinyin"])
    return jsonify({
        "query": pinyin_query,
        "fuzzy": fuzzy,
        "count": len(matched),
        "results": matched
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
