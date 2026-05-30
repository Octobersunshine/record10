import requests
import json
import time
import sys
import subprocess
import threading


BASE_URL = "http://localhost:5000"


def test_health():
    print("=== 测试健康检查接口 ===")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        print("✅ 健康检查测试通过\n")
        return True
    except Exception as e:
        print(f"❌ 健康检查测试失败: {e}\n")
        return False


def test_sentiment_positive():
    print("=== 测试正面情感分析 ===")
    data = {"text": "这部电影非常精彩，我很喜欢！"}
    try:
        response = requests.post(f"{BASE_URL}/api/sentiment", json=data)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        assert response.status_code == 200
        assert result["sentiment"] == "positive"
        assert result["score"] > 0
        print("✅ 正面情感测试通过\n")
        return True
    except Exception as e:
        print(f"❌ 正面情感测试失败: {e}\n")
        return False


def test_sentiment_negative():
    print("=== 测试负面情感分析 ===")
    data = {"text": "这部电影太差了，非常难看，很失望"}
    try:
        response = requests.post(f"{BASE_URL}/api/sentiment", json=data)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        assert response.status_code == 200
        assert result["sentiment"] == "negative"
        assert result["score"] < 0
        print("✅ 负面情感测试通过\n")
        return True
    except Exception as e:
        print(f"❌ 负面情感测试失败: {e}\n")
        return False


def test_sentiment_negation():
    print("=== 测试否定词处理 ===")
    data = {"text": "这个产品不好看，我不喜欢"}
    try:
        response = requests.post(f"{BASE_URL}/api/sentiment", json=data)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        assert response.status_code == 200
        assert result["sentiment"] == "negative"
        assert len(result["negations_found"]) > 0
        print("✅ 否定词处理测试通过\n")
        return True
    except Exception as e:
        print(f"❌ 否定词处理测试失败: {e}\n")
        return False


def test_sentiment_empty():
    print("=== 测试空文本输入 ===")
    data = {"text": ""}
    try:
        response = requests.post(f"{BASE_URL}/api/sentiment", json=data)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        assert response.status_code == 400
        print("✅ 空文本测试通过\n")
        return True
    except Exception as e:
        print(f"❌ 空文本测试失败: {e}\n")
        return False


def test_sentiment_batch():
    print("=== 测试批量分析接口 ===")
    data = {
        "texts": [
            "这个产品非常好，很满意！",
            "质量很差，非常失望",
            "今天天气不错",
            "不太喜欢这个设计",
            "I don't like this product, it's very bad"
        ]
    }
    try:
        response = requests.post(f"{BASE_URL}/api/sentiment/batch", json=data)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        assert response.status_code == 200
        assert result["count"] == 5
        assert len(result["results"]) == 5
        print("✅ 批量分析测试通过\n")
        return True
    except Exception as e:
        print(f"❌ 批量分析测试失败: {e}\n")
        return False


def test_sentiment_english():
    print("=== 测试英文情感分析 ===")
    data = {"text": "This is an amazing product, I really love it!"}
    try:
        response = requests.post(f"{BASE_URL}/api/sentiment", json=data)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        assert response.status_code == 200
        assert result["sentiment"] == "positive"
        print("✅ 英文情感测试通过\n")
        return True
    except Exception as e:
        print(f"❌ 英文情感测试失败: {e}\n")
        return False


def test_invalid_json():
    print("=== 测试无效JSON输入 ===")
    try:
        response = requests.post(
            f"{BASE_URL}/api/sentiment",
            data="not json",
            headers={"Content-Type": "application/json"}
        )
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        assert response.status_code == 400
        print("✅ 无效JSON测试通过\n")
        return True
    except Exception as e:
        print(f"❌ 无效JSON测试失败: {e}\n")
        return False


def test_404():
    print("=== 测试404错误 ===")
    try:
        response = requests.get(f"{BASE_URL}/invalid/endpoint")
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        assert response.status_code == 404
        print("✅ 404测试通过\n")
        return True
    except Exception as e:
        print(f"❌ 404测试失败: {e}\n")
        return False


def wait_for_server(timeout=30):
    print("等待服务器启动...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✅ 服务器已启动\n")
                return True
        except:
            time.sleep(1)
    print("❌ 服务器启动超时\n")
    return False


def run_all_tests():
    if not wait_for_server():
        return False

    tests = [
        test_health,
        test_sentiment_positive,
        test_sentiment_negative,
        test_sentiment_negation,
        test_sentiment_empty,
        test_sentiment_batch,
        test_sentiment_english,
        test_invalid_json,
        test_404,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ 测试 {test.__name__} 异常: {e}\n")
            failed += 1

    print(f"=== 测试总结 ===")
    print(f"通过: {passed}/{len(tests)}")
    print(f"失败: {failed}/{len(tests)}")

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
