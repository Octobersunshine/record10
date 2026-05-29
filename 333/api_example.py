"""
语言检测 API 调用示例
包含模拟的 REST API 接口和客户端调用示例
"""

import json
from typing import Dict, Any
from language_detector import (
    detect_language,
    detect_language_details,
    detect_language_mixed,
    detect_language_comprehensive
)


class LanguageDetectionAPI:
    """模拟的 REST API 服务类"""

    @staticmethod
    def detect_endpoint(request: Dict[str, Any]) -> Dict[str, Any]:
        """
        POST /api/detect
        简单语言检测接口
        """
        text = request.get('text', '')
        iso_code, confidence = detect_language(text)
        return {
            'status': 'success',
            'data': {
                'iso_code': iso_code,
                'confidence': confidence
            }
        }

    @staticmethod
    def detect_details_endpoint(request: Dict[str, Any]) -> Dict[str, Any]:
        """
        POST /api/detect/details
        详细语言检测接口
        """
        text = request.get('text', '')
        result = detect_language_details(text)
        return {
            'status': 'success',
            'data': result
        }

    @staticmethod
    def detect_mixed_endpoint(request: Dict[str, Any]) -> Dict[str, Any]:
        """
        POST /api/detect/mixed
        多语言混合检测接口（返回各语言比例）
        """
        text = request.get('text', '')
        result = detect_language_mixed(text)
        return {
            'status': 'success',
            'data': {
                'languages': result
            }
        }

    @staticmethod
    def detect_comprehensive_endpoint(request: Dict[str, Any]) -> Dict[str, Any]:
        """
        POST /api/detect/comprehensive
        综合检测接口（包含所有信息）
        """
        text = request.get('text', '')
        result = detect_language_comprehensive(text)
        return {
            'status': 'success',
            'data': result
        }


class APIClient:
    """模拟的 API 客户端"""

    def __init__(self):
        self.api = LanguageDetectionAPI()

    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """模拟 POST 请求"""
        if endpoint == '/api/detect':
            return self.api.detect_endpoint(data)
        elif endpoint == '/api/detect/details':
            return self.api.detect_details_endpoint(data)
        elif endpoint == '/api/detect/mixed':
            return self.api.detect_mixed_endpoint(data)
        elif endpoint == '/api/detect/comprehensive':
            return self.api.detect_comprehensive_endpoint(data)
        else:
            return {'status': 'error', 'message': 'Endpoint not found'}


def demo_api_calls():
    """演示各种 API 调用"""
    client = APIClient()

    print("=" * 70)
    print("语言检测 API 调用示例")
    print("=" * 70)

    test_texts = [
        "Hello world! This is a test.",
        "今天天气真好，适合出去散步。",
        "這是繁體中文的範例文字。",
        "Hello 你好 Bonjour 今日はいい天気です",
        "Привет! 今天的 meeting 在 3 点召开。",
    ]

    print("\n" + "=" * 70)
    print("1. 简单检测接口 - POST /api/detect")
    print("=" * 70)
    for text in test_texts[:2]:
        response = client.post('/api/detect', {'text': text})
        print(f"\n请求文本: {text!r}")
        print(f"响应: {json.dumps(response, ensure_ascii=False, indent=2)}")

    print("\n" + "=" * 70)
    print("2. 详细检测接口 - POST /api/detect/details")
    print("=" * 70)
    text = "今天天气真好，适合出去散步。"
    response = client.post('/api/detect/details', {'text': text})
    print(f"\n请求文本: {text!r}")
    print(f"响应: {json.dumps(response, ensure_ascii=False, indent=2)}")

    print("\n" + "=" * 70)
    print("3. 多语言混合检测接口 - POST /api/detect/mixed")
    print("=" * 70)
    text = "Hello 你好 Bonjour 今日はいい天気です"
    response = client.post('/api/detect/mixed', {'text': text})
    print(f"\n请求文本: {text!r}")
    print(f"响应: {json.dumps(response, ensure_ascii=False, indent=2)}")

    print("\n" + "=" * 70)
    print("4. 繁简识别演示")
    print("=" * 70)
    traditional_text = "這是繁體中文的範例，使用了「麼」「這」「裡」等繁體字。"
    simplified_text = "这是简体中文的范例，使用了「么」「这」「里」等简体字。"

    for text in [traditional_text, simplified_text]:
        response = client.post('/api/detect/comprehensive', {'text': text})
        print(f"\n文本: {text!r}")
        print(f"中文变体: {json.dumps(response['data'].get('chinese_variant', 'N/A'), ensure_ascii=False)}")

    print("\n" + "=" * 70)
    print("5. 综合检测接口 - POST /api/detect/comprehensive")
    print("=" * 70)
    text = "Привет! 今天的 meeting 在 3 点召开，讨论 AI project 的进展。"
    response = client.post('/api/detect/comprehensive', {'text': text})
    print(f"\n请求文本: {text!r}")
    print(f"响应: {json.dumps(response, ensure_ascii=False, indent=2)}")

    print("\n" + "=" * 70)
    print("API 文档总结")
    print("=" * 70)
    print("""
可用端点:

  POST /api/detect
    输入: {"text": "要检测的文本"}
    输出: {"iso_code": "en", "confidence": 0.95}
    用途: 快速检测主要语言

  POST /api/detect/details
    输入: {"text": "要检测的文本"}
    输出: 包含iso代码、语言名称、置信度、所有语言得分、提示
    用途: 详细检测结果

  POST /api/detect/mixed
    输入: {"text": "要检测的文本"}
    输出: {"languages": {"zh": 0.6, "en": 0.3, "ja": 0.1}}
    用途: 多语言混合文本比例分析

  POST /api/detect/comprehensive
    输入: {"text": "要检测的文本"}
    输出: 包含primary主语言检测、mixed_languages各语言比例、
          chinese_variant中文繁简识别（如适用）
    用途: 完整的语言分析
""")


if __name__ == "__main__":
    demo_api_calls()
