import json
import re
import sys
import os
from difflib import SequenceMatcher


_CUSTOM_BROWSER_RULES = []
_CUSTOM_OS_RULES = []


BUILTIN_BROWSER_RULES = [
    (r"micromessenger/([\d.]+)", "WeChat"),
    (r"qqbrowser/([\d.]+)", "QQ Browser"),
    (r"ucbrowser/([\d.]+)", "UC Browser"),
    (r"ucweb/([\d.]+)", "UC Browser"),
    (r"uc ?browser/([\d.]+)", "UC Browser"),
    (r"quark/([\d.]+)", "Quark"),
    (r"samsungbrowser/([\d.]+)", "Samsung Browser"),
    (r"miuibrowser/([\d.]+)", "MIUI Browser"),
    (r"vivobrowser/([\d.]+)", "Vivo Browser"),
    (r"oppobrowser/([\d.]+)", "OPPO Browser"),
    (r"huaweibrowser/([\d.]+)", "Huawei Browser"),
    (r"hwbrowser/([\d.]+)", "Huawei Browser"),
    (r"heytapbrowser/([\d.]+)", "HeyTap Browser"),
    (r"meizubrowser/([\d.]+)", "Meizu Browser"),
    (r"lenovobrowser/([\d.]+)", "Lenovo Browser"),
    (r"bdhd/([\d.]+)", "Baidu Browser"),
    (r"baiduboxapp/([\d.]+)", "Baidu App"),
    (r"baidubrowser/([\d.]+)", "Baidu Browser"),
    (r"bdbrowser/([\d.]+)", "Baidu Browser"),
    (r"sogoumobilebrowser/([\d.]+)", "Sogou Browser"),
    (r"sogousearch/([\d.]+)", "Sogou Search"),
    (r"2345explorer/([\d.]+)", "2345 Explorer"),
    (r"2345chrome/([\d.]+)", "2345 Chrome"),
    (r"mb2345browser/([\d.]+)", "2345 Browser"),
    (r"qihu 360se/([\d.]+)", "360 Safe Browser"),
    (r"360se/([\d.]+)", "360 Safe Browser"),
    (r"360ee/([\d.]+)", "360 Extreme Explorer"),
    (r"360 aphone browser ([\d.]+)", "360 Browser"),
    (r"huohoubrowser/([\d.]+)", "Huohou Browser"),
    (r"maxthon/([\d.]+)", "Maxthon"),
    (r"theworld/([\d.]+)", "TheWorld"),
    (r"avant browser", "Avant Browser"),
    (r"greenbrowser/([\d.]+)", "GreenBrowser"),
    (r"coolnovo/([\d.]+)", "CoolNovo"),
    (r"yabrowser/([\d.]+)", "Yandex Browser"),
    (r"brave/([\d.]+)", "Brave"),
    (r"vivaldi/([\d.]+)", "Vivaldi"),
    (r"palemoon/([\d.]+)", "Pale Moon"),
    (r"waterfox/([\d.]+)", "Waterfox"),
    (r"seamonkey/([\d.]+)", "SeaMonkey"),
    (r"fxios/([\d.]+)", "Firefox iOS"),
    (r"opt/([\d.]+)", "Opera Touch"),
    (r"coast/([\d.]+)", "Opera Coast"),
    (r"silk/([\d.]+)", "Amazon Silk"),
    (r"dolphin/([\d.]+)", "Dolphin"),
    (r"qq/([\d.]+)", "QQ"),
    (r"edg/([\d.]+)", "Edge"),
    (r"opera/([\d.]+)", "Opera"),
    (r"opr/([\d.]+)", "Opera"),
    (r"chrome/([\d.]+)", "Chrome"),
    (r"firefox/([\d.]+)", "Firefox"),
    (r"msie ([\d.]+)", "Internet Explorer"),
    (r"trident/.*rv:([\d.]+)", "Internet Explorer"),
    (r"safari/([\d.]+)", "Safari"),
]

BUILTIN_OS_RULES = [
    (r"windows nt ([\d.]+)", "Windows"),
    (r"mac os x ([\d_]+)", "macOS"),
    (r"iphone os ([\d_]+)", "iOS"),
    (r"ipad; cpu os ([\d_]+)", "iOS"),
    (r"cpu os ([\d_]+).*like mac os x", "iOS"),
    (r"ios ([\d.]+)", "iOS"),
    (r"android ([\d.]+)", "Android"),
    (r"android", "Android"),
    (r"harmonyos ([\d.]+)", "HarmonyOS"),
    (r"harmonyos", "HarmonyOS"),
    (r"linux", "Linux"),
]


def add_custom_rule(pattern, name, category="browser"):
    try:
        re.compile(pattern)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern '{pattern}': {e}")

    rule = (pattern, name)
    if category == "browser":
        _CUSTOM_BROWSER_RULES.insert(0, rule)
    elif category == "os":
        _CUSTOM_OS_RULES.insert(0, rule)
    else:
        raise ValueError(f"Unknown category '{category}', must be 'browser' or 'os'")


def load_custom_rules(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Rules file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Rules file must be a JSON object with 'browser' and/or 'os' keys")

    loaded = 0
    for category, rules in data.items():
        if category not in ("browser", "os"):
            continue
        if not isinstance(rules, list):
            raise ValueError(f"'{category}' must be a list of rule objects")

        for rule in rules:
            if not isinstance(rule, dict) or "pattern" not in rule or "name" not in rule:
                raise ValueError("Each rule must be an object with 'pattern' and 'name' fields")
            add_custom_rule(rule["pattern"], rule["name"], category)
            loaded += 1

    return loaded


def load_custom_rules_from_string(json_string):
    data = json.loads(json_string)
    if not isinstance(data, dict):
        raise ValueError("Rules JSON must be an object with 'browser' and/or 'os' keys")

    loaded = 0
    for category, rules in data.items():
        if category not in ("browser", "os"):
            continue
        if not isinstance(rules, list):
            raise ValueError(f"'{category}' must be a list of rule objects")

        for rule in rules:
            if not isinstance(rule, dict) or "pattern" not in rule or "name" not in rule:
                raise ValueError("Each rule must be an object with 'pattern' and 'name' fields")
            add_custom_rule(rule["pattern"], rule["name"], category)
            loaded += 1

    return loaded


def clear_custom_rules(category=None):
    global _CUSTOM_BROWSER_RULES, _CUSTOM_OS_RULES
    if category is None:
        _CUSTOM_BROWSER_RULES.clear()
        _CUSTOM_OS_RULES.clear()
    elif category == "browser":
        _CUSTOM_BROWSER_RULES.clear()
    elif category == "os":
        _CUSTOM_OS_RULES.clear()
    else:
        raise ValueError(f"Unknown category '{category}', must be 'browser', 'os', or None")


def get_custom_rules(category=None):
    result = {}
    if category is None or category == "browser":
        result["browser"] = [{"pattern": p, "name": n} for p, n in _CUSTOM_BROWSER_RULES]
    if category is None or category == "os":
        result["os"] = [{"pattern": p, "name": n} for p, n in _CUSTOM_OS_RULES]
    return result


def parse_ua_with_library(ua_string):
    try:
        from user_agents import parse
        ua = parse(ua_string)

        device_type = "desktop"
        if ua.is_mobile:
            device_type = "mobile"
        elif ua.is_tablet:
            device_type = "tablet"

        browser_version = ua.browser.version_string.split(".")[0] if ua.browser.version_string else ""

        return {
            "browser": ua.browser.family or "",
            "browser_version": browser_version,
            "os": ua.os.family or "",
            "os_version": ua.os.version_string or "",
            "device": device_type,
            "device_brand": ua.device.brand or "",
            "device_model": ua.device.model or ""
        }
    except ImportError:
        return None


def _match_browser(ua_string, rules):
    ua_lower = ua_string.lower()
    for pattern, name in rules:
        match = re.search(pattern, ua_lower)
        if match:
            version = ""
            if match.lastindex and match.lastindex >= 1:
                version = match.group(1).split(".")[0] if match.group(1) else ""
            return name, version
    return "", ""


def _match_os(ua_lower, rules):
    for pattern, name in rules:
        match = re.search(pattern, ua_lower)
        if match:
            os_version = ""
            if match.lastindex and match.lastindex >= 1:
                os_version = match.group(1).replace("_", ".")
            return name, os_version
    return "", ""


def parse_ua_with_regex(ua_string):
    ua_lower = ua_string.lower()

    browser = ""
    browser_version = ""
    os_name = ""
    os_version = ""
    device_type = "desktop"

    if "mobile" in ua_lower:
        if "tablet" in ua_lower or "ipad" in ua_lower or ("android" in ua_lower and "mobile" not in ua_lower):
            device_type = "tablet"
        else:
            device_type = "mobile"
    elif "tablet" in ua_lower or "ipad" in ua_lower:
        device_type = "tablet"

    all_browser_rules = list(_CUSTOM_BROWSER_RULES) + BUILTIN_BROWSER_RULES
    browser, browser_version = _match_browser(ua_string, all_browser_rules)

    if not browser and "safari" in ua_lower:
        if "chrome" not in ua_lower and "chromium" not in ua_lower and "edg" not in ua_lower:
            browser = "Safari"
            version_match = re.search(r"version/([\d.]+)", ua_lower)
            if version_match:
                browser_version = version_match.group(1).split(".")[0]
            else:
                safari_match = re.search(r"safari/([\d.]+)", ua_lower)
                if safari_match:
                    browser_version = safari_match.group(1).split(".")[0]

    all_os_rules = list(_CUSTOM_OS_RULES) + BUILTIN_OS_RULES
    os_name, os_version = _match_os(ua_lower, all_os_rules)

    if not os_name:
        if "windows" in ua_lower:
            os_name = "Windows"
        elif "mac" in ua_lower:
            os_name = "macOS"
        elif "linux" in ua_lower:
            os_name = "Linux"
        elif "android" in ua_lower:
            os_name = "Android"
        elif "iphone" in ua_lower or "ipad" in ua_lower:
            os_name = "iOS"

    return {
        "browser": browser,
        "browser_version": browser_version,
        "os": os_name,
        "os_version": os_version,
        "device": device_type,
        "device_brand": "",
        "device_model": ""
    }


def parse_user_agent(ua_string, use_library=True):
    if not ua_string:
        return json.dumps({"error": "Empty User-Agent string"}, ensure_ascii=False, indent=2)

    result = None
    parser_used = "regex"

    if use_library:
        result = parse_ua_with_library(ua_string)
        if result:
            parser_used = "user-agents library"

    if result is None:
        result = parse_ua_with_regex(ua_string)

    result["parser"] = parser_used

    return json.dumps(result, ensure_ascii=False, indent=2)


def _extract_fingerprint(ua_string):
    ua_lower = ua_string.lower()
    parsed = parse_ua_with_regex(ua_string)

    chrome_ver = ""
    m = re.search(r"chrome/([\d.]+)", ua_lower)
    if m:
        chrome_ver = m.group(1)

    safari_ver = ""
    m = re.search(r"safari/([\d.]+)", ua_lower)
    if m:
        safari_ver = m.group(1)

    gecko_ver = ""
    m = re.search(r"gecko/([\d.]+)", ua_lower)
    if m:
        gecko_ver = m.group(1)

    applewebkit_ver = ""
    m = re.search(r"applewebkit/([\d.]+)", ua_lower)
    if m:
        applewebkit_ver = m.group(1)

    build_id = ""
    m = re.search(r"build/([a-zA-Z0-9.]+)", ua_lower)
    if m:
        build_id = m.group(1)

    device_model = ""
    m = re.search(r"; ([a-zA-Z0-9_]+(?:[- ][a-zA-Z0-9_]+)*)\s+build", ua_lower)
    if not m:
        m = re.search(r"; ([a-zA-Z0-9_]+(?:[- ][a-zA-Z0-9_]+)*)\)\s+applewebkit", ua_lower)
    if m:
        device_model = m.group(1)

    full_browser_ver = ""
    all_browser_rules = list(_CUSTOM_BROWSER_RULES) + BUILTIN_BROWSER_RULES
    for pattern, name in all_browser_rules:
        match = re.search(pattern, ua_lower)
        if match:
            if match.lastindex and match.lastindex >= 1:
                full_browser_ver = match.group(1)
            break

    full_os_ver = ""
    all_os_rules = list(_CUSTOM_OS_RULES) + BUILTIN_OS_RULES
    for pattern, name in all_os_rules:
        match = re.search(pattern, ua_lower)
        if match:
            if match.lastindex and match.lastindex >= 1:
                full_os_ver = match.group(1).replace("_", ".")
            break

    engine = ""
    if applewebkit_ver:
        engine = "WebKit"
    if gecko_ver:
        engine = "Gecko"

    has_mobile = "mobile" in ua_lower
    has_tablet = "tablet" in ua_lower or "ipad" in ua_lower

    return {
        "browser": parsed["browser"],
        "browser_version": parsed["browser_version"],
        "browser_version_full": full_browser_ver,
        "os": parsed["os"],
        "os_version": parsed["os_version"],
        "os_version_full": full_os_ver,
        "device_type": parsed["device"],
        "engine": engine,
        "engine_chrome_version": chrome_ver,
        "engine_safari_version": safari_ver,
        "engine_webkit_version": applewebkit_ver,
        "engine_gecko_version": gecko_ver,
        "build_id": build_id,
        "device_model": device_model,
        "has_mobile_flag": has_mobile,
        "has_tablet_flag": has_tablet,
    }


def compare_user_agents(ua1, ua2):
    fp1 = _extract_fingerprint(ua1)
    fp2 = _extract_fingerprint(ua2)

    weights = {
        "browser": 0.20,
        "browser_version": 0.15,
        "os": 0.15,
        "os_version": 0.10,
        "device_type": 0.10,
        "engine": 0.05,
        "engine_chrome_version": 0.05,
        "engine_safari_version": 0.05,
        "build_id": 0.05,
        "device_model": 0.05,
        "string_similarity": 0.05,
    }

    scores = {}

    scores["browser"] = 1.0 if fp1["browser"] and fp1["browser"] == fp2["browser"] else 0.0

    if fp1["browser_version"] and fp2["browser_version"]:
        if fp1["browser_version"] == fp2["browser_version"]:
            scores["browser_version"] = 1.0
        else:
            v1_parts = fp1["browser_version_full"].split(".")
            v2_parts = fp2["browser_version_full"].split(".")
            common = 0
            for a, b in zip(v1_parts, v2_parts):
                if a == b:
                    common += 1
                else:
                    break
            scores["browser_version"] = min(common / max(len(v1_parts), len(v2_parts), 1), 0.7)
    else:
        scores["browser_version"] = 0.0

    scores["os"] = 1.0 if fp1["os"] and fp1["os"] == fp2["os"] else 0.0

    if fp1["os_version"] and fp2["os_version"]:
        if fp1["os_version"] == fp2["os_version"]:
            scores["os_version"] = 1.0
        else:
            ov1_parts = fp1["os_version_full"].split(".")
            ov2_parts = fp2["os_version_full"].split(".")
            common = 0
            for a, b in zip(ov1_parts, ov2_parts):
                if a == b:
                    common += 1
                else:
                    break
            scores["os_version"] = min(common / max(len(ov1_parts), len(ov2_parts), 1), 0.7)
    else:
        scores["os_version"] = 0.0

    scores["device_type"] = 1.0 if fp1["device_type"] == fp2["device_type"] else 0.0

    scores["engine"] = 1.0 if fp1["engine"] == fp2["engine"] else 0.0

    scores["engine_chrome_version"] = (
        1.0 if fp1["engine_chrome_version"] and fp1["engine_chrome_version"] == fp2["engine_chrome_version"]
        else (0.5 if fp1["engine_chrome_version"] and fp2["engine_chrome_version"] else 0.0)
    )

    scores["engine_safari_version"] = (
        1.0 if fp1["engine_safari_version"] and fp1["engine_safari_version"] == fp2["engine_safari_version"]
        else (0.5 if fp1["engine_safari_version"] and fp2["engine_safari_version"] else 0.0)
    )

    scores["build_id"] = (
        1.0 if fp1["build_id"] and fp1["build_id"] == fp2["build_id"]
        else (0.3 if fp1["build_id"] and fp2["build_id"] else 0.0)
    )

    scores["device_model"] = (
        1.0 if fp1["device_model"] and fp1["device_model"] == fp2["device_model"]
        else (0.3 if fp1["device_model"] and fp2["device_model"] else 0.0)
    )

    scores["string_similarity"] = SequenceMatcher(None, ua1.lower(), ua2.lower()).ratio()

    total_score = sum(weights[k] * scores[k] for k in weights)
    total_score = round(total_score, 4)

    if total_score >= 0.85:
        verdict = "same_device"
    elif total_score >= 0.60:
        verdict = "likely_same_device"
    elif total_score >= 0.35:
        verdict = "different_device_same_profile"
    else:
        verdict = "different_device"

    detail = {k: round(v, 4) for k, v in scores.items()}

    return json.dumps({
        "similarity_score": total_score,
        "verdict": verdict,
        "detail": detail,
        "fingerprint_1": fp1,
        "fingerprint_2": fp2,
    }, ensure_ascii=False, indent=2)


def detect_anomaly(ua_string):
    fp = _extract_fingerprint(ua_string)
    ua_lower = ua_string.lower()
    anomalies = []
    risk_score = 0.0

    known_crawler_patterns = [
        (r"bot", "crawler"),
        (r"crawl", "crawler"),
        (r"spider", "crawler"),
        (r"scraper", "crawler"),
        (r"scrapy", "crawler"),
        (r"headless", "headless_browser"),
        (r"phantomjs", "headless_browser"),
        (r"selenium", "automation"),
        (r"puppeteer", "automation"),
        (r"playwright", "automation"),
        (r"webdriver", "automation"),
        (r"python-requests", "script"),
        (r"python-urllib", "script"),
        (r"requests/", "script"),
        (r"axios/", "script"),
        (r"java/", "script"),
        (r"curl/", "script"),
        (r"wget/", "script"),
        (r"go-http-client", "script"),
        (r"apache-httpclient", "script"),
    ]

    for pattern, category in known_crawler_patterns:
        if re.search(pattern, ua_lower):
            anomalies.append({"type": category, "detail": f"Matched known pattern: {pattern}"})
            if category == "crawler":
                risk_score += 0.50
            elif category == "headless_browser":
                risk_score += 0.60
            elif category == "automation":
                risk_score += 0.55
            elif category == "script":
                risk_score += 0.70

    if fp["engine"] == "WebKit":
        if not fp["engine_webkit_version"]:
            anomalies.append({"type": "missing_engine_version", "detail": "WebKit engine declared but no AppleWebKit version found"})
            risk_score += 0.20

    if fp["browser"] == "Chrome":
        if not fp["engine_chrome_version"]:
            anomalies.append({"type": "missing_browser_version", "detail": "Chrome browser identified but no Chrome/ version in UA"})
            risk_score += 0.25

    if fp["browser"] == "Safari":
        if not fp["engine_safari_version"]:
            anomalies.append({"type": "missing_browser_version", "detail": "Safari browser identified but no Safari/ version in UA"})
            risk_score += 0.15

    if fp["browser"] == "Firefox":
        if not fp["engine_gecko_version"]:
            anomalies.append({"type": "missing_engine_version", "detail": "Firefox browser identified but no Gecko version in UA"})
            risk_score += 0.20

    if fp["os"] == "Windows" and fp["device_type"] == "mobile":
        anomalies.append({"type": "os_device_mismatch", "detail": "Windows OS with mobile device type is suspicious"})
        risk_score += 0.30

    if fp["os"] == "Android":
        if fp["device_type"] == "desktop":
            anomalies.append({"type": "os_device_mismatch", "detail": "Android OS with desktop device type is suspicious"})
            risk_score += 0.30
        if not fp["has_mobile_flag"] and fp["device_type"] == "mobile":
            anomalies.append({"type": "missing_mobile_flag", "detail": "Android mobile device without Mobile flag in UA"})
            risk_score += 0.15

    if fp["os"] == "iOS":
        if fp["device_type"] == "desktop":
            anomalies.append({"type": "os_device_mismatch", "detail": "iOS with desktop device type is suspicious"})
            risk_score += 0.30
        if fp["browser"] not in ("Safari", "Firefox iOS", ""):
            if not any(k in ua_lower for k in ["chrome", "edg", "opera", "brave", "vivaldi"]):
                anomalies.append({"type": "ios_browser_mismatch", "detail": f"iOS does not typically run {fp['browser']} natively"})
                risk_score += 0.20

    if fp["browser"] == "Chrome" and fp["os"] == "macOS":
        if not fp["engine_safari_version"]:
            anomalies.append({"type": "macos_chrome_missing_safari", "detail": "Chrome on macOS should contain Safari/ version token"})
            risk_score += 0.10

    if fp["browser"] == "Firefox" and fp["os"] in ("Windows", "Linux", "macOS"):
        if fp["engine_chrome_version"]:
            anomalies.append({"type": "firefox_with_chrome_token", "detail": "Firefox UA should not contain Chrome/ version token"})
            risk_score += 0.25

    if fp["engine"] == "Gecko" and fp["browser"] in ("Chrome", "Edge", "Opera", "Brave", "Vivaldi", "Samsung Browser"):
        anomalies.append({"type": "engine_browser_mismatch", "detail": f"{fp['browser']} uses WebKit/Blink engine, but Gecko was detected — possible UA spoofing"})
        risk_score += 0.40

    if fp["engine"] == "WebKit" and fp["browser"] == "Firefox":
        if fp["os"] in ("Windows", "Linux") and "fxios" not in ua_lower:
            anomalies.append({"type": "engine_browser_mismatch", "detail": "Firefox on desktop uses Gecko engine, but WebKit was detected — possible UA spoofing"})
            risk_score += 0.30

    if fp["browser"] == "Edge" and fp["os"] not in ("Windows", ""):
        if fp["os"] == "Linux":
            anomalies.append({"type": "edge_on_linux", "detail": "Edge on Linux is uncommon (possible spoofing)"})
            risk_score += 0.10

    mozilla_match = re.match(r"^mozilla/([\d.]+)", ua_lower)
    if mozilla_match:
        mozilla_ver = mozilla_match.group(1)
        if mozilla_ver != "5.0" and fp["browser"] in ("Chrome", "Firefox", "Safari", "Edge"):
            anomalies.append({"type": "non_standard_mozilla_version", "detail": f"Mozilla/{mozilla_ver} is unusual for {fp['browser']}, expected 5.0"})
            risk_score += 0.15

    generic_tokens = ["mozilla/5.0", "applewebkit/537.36", "khtml, like gecko", "safari/537.36"]
    stripped = ua_lower
    for token in generic_tokens:
        stripped = stripped.replace(token, "")
    stripped = stripped.strip(" ()")
    if len(stripped) < 5:
        anomalies.append({"type": "overly_generic_ua", "detail": "UA string contains only generic tokens with no identifying information"})
        risk_score += 0.25

    risk_score = min(risk_score, 1.0)

    if risk_score >= 0.70:
        risk_level = "high"
    elif risk_score >= 0.35:
        risk_level = "medium"
    elif risk_score >= 0.15:
        risk_level = "low"
    else:
        risk_level = "safe"

    return json.dumps({
        "risk_score": round(risk_score, 4),
        "risk_level": risk_level,
        "anomalies": anomalies,
        "fingerprint": fp,
    }, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    test_uas = [
        ("Windows + Chrome", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
        ("macOS + Safari", "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15"),
        ("iPhone + Safari", "Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1"),
        ("iPad + Safari", "Mozilla/5.0 (iPad; CPU OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1"),
        ("Android + Chrome", "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36"),
        ("Linux + Firefox", "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0"),
        ("Windows + Firefox", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0"),
        ("Windows + Edge", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"),
        ("Android + UC Browser", "Mozilla/5.0 (Linux; U; Android 13; zh-CN; MI 8 Build/SKQ1.220303.001) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/100.0.4896.127 UCBrowser/15.5.6.1156 Mobile Safari/537.36"),
        ("Android + Quark", "Mozilla/5.0 (Linux; U; Android 13; zh-CN; M2004J7AC Build/SKQ1.220303.001) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/100.0.4896.127 Quark/6.8.2.358 Mobile Safari/537.36"),
        ("Android + Baidu Browser", "Mozilla/5.0 (Linux; Android 13; Pixel 6 Build/SKQ1.220303.001) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 BDBrowser/13.5.0.0 Mobile Safari/537.36"),
        ("Android + Sogou Browser", "Mozilla/5.0 (Linux; Android 13; SM-G990B Build/SKQ1.220303.001) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/100.0.4896.127 SogouMobileBrowser/6.3.5 Mobile Safari/537.36"),
        ("Windows + 360 Safe", "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 360SE/13.5.1026.0 Safari/537.36"),
        ("Android + QQ Browser", "Mozilla/5.0 (Linux; Android 13; M2004J7AC Build/SKQ1.220303.001) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/100.0.4896.127 MQQBrowser/13.5 Mobile Safari/537.36"),
        ("Android + Huawei Browser", "Mozilla/5.0 (Linux; Android 13; NOH-AN00 Build/SKQ1.220303.001) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.88 HuaweiBrowser/13.0.5.300 Mobile Safari/537.36"),
        ("Android + MIUI Browser", "Mozilla/5.0 (Linux; U; Android 13; zh-cn; MI 8 Build/SKQ1.220303.001) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/100.0.4896.127 Mobile Safari/537.36 XiaoMi/MiuiBrowser/17.5.5"),
        ("HarmonyOS + Huawei Browser", "Mozilla/5.0 (Linux; HarmonyOS 4.0.0; NOH-AN00 Build/SKQ1.220303.001) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Mobile Safari/537.36 HwBrowser/14.0.5.300"),
        ("Android + WeChat", "Mozilla/5.0 (Linux; Android 13; M2004J7AC Build/SKQ1.220303.001) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/111.0.5563.116 Mobile Safari/537.36 MicroMessenger/8.0.38.2400"),
    ]

    compare_cases = [
        ("Same device, Chrome updated", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"),
        ("Different OS, same browser family", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36", "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
        ("Completely different browsers", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0", "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15"),
        ("Same Android device, browser upgrade", "Mozilla/5.0 (Linux; Android 14; Pixel 8 Build/SKQ1.220303.001) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36", "Mozilla/5.0 (Linux; Android 14; Pixel 8 Build/SKQ1.220303.001) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Mobile Safari/537.36"),
        ("Identical UA", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    ]

    anomaly_cases = [
        ("Normal Chrome", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
        ("Google Bot", "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"),
        ("Python Requests", "python-requests/2.31.0"),
        ("Headless Chrome", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/120.0.0.0 Safari/537.36"),
        ("PhantomJS", "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) PhantomJS/2.1.1 Safari/537.36"),
        ("curl", "curl/8.4.0"),
        ("Firefox with Chrome token (spoofed)", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0 Chrome/120.0.0.0"),
        ("Overly generic UA", "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko) Safari/537.36"),
        ("Android desktop mismatch", "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
        ("Scrapy crawler", "Scrapy/2.11.0 (+https://scrapy.org)"),
    ]

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--load-rules" and len(sys.argv) > 2:
            count = load_custom_rules(sys.argv[2])
            print(f"Loaded {count} custom rules from {sys.argv[2]}")
            if len(sys.argv) > 3:
                ua = sys.argv[3]
                print(parse_user_agent(ua))
        elif arg == "--compare" and len(sys.argv) > 3:
            print(compare_user_agents(sys.argv[2], sys.argv[3]))
        elif arg == "--detect" and len(sys.argv) > 2:
            print(detect_anomaly(sys.argv[2]))
        elif arg == "--demo-custom":
            print("=== Demo: Custom Rules Extension ===\n")
            add_custom_rule(r"mybrowser/([\d.]+)", "My Custom Browser", "browser")
            add_custom_rule(r"myos ([\d.]+)", "My Custom OS", "os")

            demo_ua = "Mozilla/5.0 (MyOS 2.0) AppleWebKit/537.36 (KHTML, like Gecko) MyBrowser/1.0.0 Safari/537.36"
            print(f"UA: {demo_ua}")
            print("Result (with custom rules):")
            print(parse_user_agent(demo_ua))

            print("\nCustom rules in effect:")
            print(json.dumps(get_custom_rules(), ensure_ascii=False, indent=2))

            clear_custom_rules()
            print("\nResult (after clearing custom rules):")
            print(parse_user_agent(demo_ua))
        elif arg == "--demo-compare":
            print("=== Demo: UA Comparison ===\n")
            for label, ua1, ua2 in compare_cases:
                result = json.loads(compare_user_agents(ua1, ua2))
                print(f"{label}:")
                print(f"  Score: {result['similarity_score']} | Verdict: {result['verdict']}")
                print(f"  Detail: {json.dumps(result['detail'], ensure_ascii=False)}")
                print()
        elif arg == "--demo-detect":
            print("=== Demo: Anomaly Detection ===\n")
            for label, ua in anomaly_cases:
                result = json.loads(detect_anomaly(ua))
                print(f"{label}:")
                print(f"  Risk: {result['risk_score']} | Level: {result['risk_level']}")
                if result["anomalies"]:
                    for a in result["anomalies"]:
                        print(f"  - [{a['type']}] {a['detail']}")
                else:
                    print("  - No anomalies detected")
                print()
        else:
            print(parse_user_agent(arg))
    else:
        print("=== User-Agent Parser Test ===\n")
        print("Usage:")
        print("  python ua_parser.py \"<UA string>\"           # Parse a single UA")
        print("  python ua_parser.py --compare \"<UA1>\" \"<UA2>\"  # Compare two UAs")
        print("  python ua_parser.py --detect \"<UA>\"            # Detect anomalies")
        print("  python ua_parser.py --demo-compare             # Run comparison demos")
        print("  python ua_parser.py --demo-detect              # Run anomaly detection demos")
        print("  python ua_parser.py --load-rules <file> [UA]   # Load custom rules")
        print("  python ua_parser.py --demo-custom              # Demo custom rules")
        print()
        for i, (label, ua) in enumerate(test_uas, 1):
            print(f"Test {i}: {label}")
            print(f"UA: {ua[:90]}...")
            print("Result:")
            print(parse_user_agent(ua))
            print("-" * 60)
