import requests
import json
import time
import sys


BASE_URL = "http://localhost:5000"


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


def test_list_domains():
    print("=== 测试获取领域列表 ===")
    try:
        response = requests.get(f"{BASE_URL}/api/domains")
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        assert response.status_code == 200
        assert result["count"] > 0
        print("✅ 领域列表测试通过\n")
        return True
    except Exception as e:
        print(f"❌ 领域列表测试失败: {e}\n")
        return False


def test_aspect_analysis():
    print("=== 测试方面级情感分析 ===")
    data = {
        "text": "屏幕很清晰，但电池不耐用，摄像头拍照很棒",
        "domain": "electronics"
    }
    try:
        response = requests.post(f"{BASE_URL}/api/sentiment/aspects", json=data)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        assert response.status_code == 200
        assert "aspects" in result
        assert len(result["aspects"]) > 0
        print("✅ 方面级分析测试通过\n")
        return True
    except Exception as e:
        print(f"❌ 方面级分析测试失败: {e}\n")
        return False


def test_aspect_restaurant():
    print("=== 测试餐饮领域方面级分析 ===")
    data = {
        "text": "菜品味道不错，食材很新鲜，但环境太吵了，服务也慢",
        "domain": "restaurant"
    }
    try:
        response = requests.post(f"{BASE_URL}/api/sentiment/aspects", json=data)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        assert response.status_code == 200
        assert "aspects" in result
        print("✅ 餐饮领域方面级测试通过\n")
        return True
    except Exception as e:
        print(f"❌ 餐饮领域方面级测试失败: {e}\n")
        return False


def test_domain_effect():
    print("=== 测试领域词典效果 ===")
    print("--- 通用领域 vs 电子产品领域 ---")
    text = "这个手机屏幕很清晰，电池也耐用"

    data_general = {"text": text, "domain": "general"}
    data_electronics = {"text": text, "domain": "electronics"}

    try:
        r1 = requests.post(f"{BASE_URL}/api/sentiment", json=data_general)
        r2 = requests.post(f"{BASE_URL}/api/sentiment", json=data_electronics)

        result_general = r1.json()
        result_electronics = r2.json()

        print(f"文本: {text}")
        print(f"通用领域: {result_general['sentiment']} ({result_general['score']})")
        print(f"电子领域: {result_electronics['sentiment']} ({result_electronics['score']})")
        print(f"电子领域正面词: {result_electronics['positive_words']}")

        assert result_electronics["score"] > result_general["score"]
        print("✅ 领域词典效果测试通过（电子领域得分更高）\n")
        return True
    except Exception as e:
        print(f"❌ 领域词典效果测试失败: {e}\n")
        return False


def test_batch_with_domain():
    print("=== 测试批量分析（带领域） ===")
    data = {
        "texts": [
            "屏幕很清晰",
            "电池不耐用",
            "拍照效果很棒"
        ],
        "domain": "electronics"
    }
    try:
        response = requests.post(f"{BASE_URL}/api/sentiment/batch", json=data)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        assert response.status_code == 200
        assert result["domain"] == "electronics"
        assert result["count"] == 3
        print("✅ 批量分析（带领域）测试通过\n")
        return True
    except Exception as e:
        print(f"❌ 批量分析（带领域）测试失败: {e}\n")
        return False


def test_aspect_finance():
    print("=== 测试金融领域方面级分析 ===")
    data = {
        "text": "这只基金收益不错，但波动太大，客服也不专业",
        "domain": "finance"
    }
    try:
        response = requests.post(f"{BASE_URL}/api/sentiment/aspects", json=data)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        assert response.status_code == 200
        assert "aspects" in result
        print("✅ 金融领域方面级测试通过\n")
        return True
    except Exception as e:
        print(f"❌ 金融领域方面级测试失败: {e}\n")
        return False


def test_invalid_domain():
    print("=== 测试无效领域（应使用通用领域） ===")
    data = {
        "text": "这个产品很好",
        "domain": "invalid_domain"
    }
    try:
        response = requests.post(f"{BASE_URL}/api/sentiment", json=data)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        assert response.status_code == 200
        print("✅ 无效领域测试通过（正常返回）\n")
        return True
    except Exception as e:
        print(f"❌ 无效领域测试失败: {e}\n")
        return False


def test_aspect_without_domain():
    print("=== 测试不指定领域的方面级分析 ===")
    data = {
        "text": "质量很好，服务不错，就是价格有点贵"
    }
    try:
        response = requests.post(f"{BASE_URL}/api/sentiment/aspects", json=data)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        assert response.status_code == 200
        assert "aspects" in result
        print("✅ 不指定领域的方面级分析测试通过\n")
        return True
    except Exception as e:
        print(f"❌ 不指定领域的方面级分析测试失败: {e}\n")
        return False


def run_all_tests():
    if not wait_for_server():
        return False

    tests = [
        test_list_domains,
        test_aspect_analysis,
        test_aspect_restaurant,
        test_domain_effect,
        test_batch_with_domain,
        test_aspect_finance,
        test_invalid_domain,
        test_aspect_without_domain,
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
