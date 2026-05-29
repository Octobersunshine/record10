import requests
import json

BASE_URL = "http://127.0.0.1:5000/api/sort"

def send_request(payload):
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(BASE_URL, data=json.dumps(payload), headers=headers)
        return response.status_code, response.json()
    except requests.exceptions.ConnectionError:
        print("连接失败，请确保API服务已启动")
        return None, None
    except Exception as e:
        print(f"发生错误: {e}")
        return None, None

def test_bubble_sort():
    print("\n" + "=" * 70)
    print("测试1: 冒泡排序 (algorithm=bubble)")
    print("=" * 70)
    
    test_array = [64, 34, 25, 12, 22, 11, 90]
    status, result = send_request({"array": test_array, "algorithm": "bubble"})
    
    if status == 200:
        print(f"算法: {result['algorithm_name']}")
        print(f"总轮数: {result['total_rounds']}")
        print(f"交换次数: {result['swap_count']}")
        print(f"返回步数: {result['returned_steps_count']}")
        print("排序过程:")
        for step in result['sorting_steps'][:3]:
            print(f"  第 {step['round']} 轮: 交换累计={step['swaps_so_far']}")
        print(f"最终结果: {result['final_sorted_array']}")
    else:
        print(f"失败 [{status}]: {result}")

def test_cocktail_sort():
    print("\n" + "=" * 70)
    print("测试2: 鸡尾酒排序 (algorithm=cocktail)")
    print("=" * 70)
    
    test_array = [64, 34, 25, 12, 22, 11, 90]
    status, result = send_request({"array": test_array, "algorithm": "cocktail"})
    
    if status == 200:
        print(f"算法: {result['algorithm_name']}")
        print(f"总轮数: {result['total_rounds']}")
        print(f"交换次数: {result['swap_count']}")
        print(f"返回步数: {result['returned_steps_count']}")
        print("排序过程:")
        for step in result['sorting_steps']:
            dir_str = "→右→" if step['direction'] == '右' else "←左←"
            print(f"  第 {step['round']} 轮 [{dir_str}]: 交换累计={step['swaps_so_far']}")
        print(f"最终结果: {result['final_sorted_array']}")
    else:
        print(f"失败 [{status}]: {result}")

def test_compare_random():
    print("\n" + "=" * 70)
    print("测试3: 两种算法对比 (algorithm=compare) - 随机数组")
    print("=" * 70)
    
    import random
    random.seed(42)
    test_array = [random.randint(1, 100) for _ in range(20)]
    
    status, result = send_request({"array": test_array, "algorithm": "compare"})
    
    if status == 200:
        comp = result['comparison']
        print(f"原始数组: {test_array}")
        print(f"\n冒泡排序:   {comp['bubble_sort']['total_rounds']} 轮, {comp['bubble_sort']['swap_count']} 次交换")
        print(f"鸡尾酒排序: {comp['cocktail_sort']['total_rounds']} 轮, {comp['cocktail_sort']['swap_count']} 次交换")
        print(f"\n轮数减少:   {comp['round_difference']['absolute']} 轮 ({comp['round_difference']['percent']}%)")
        print(f"交换减少:   {comp['swap_difference']['absolute']} 次 ({comp['swap_difference']['percent']}%)")
        print(f"\n胜出算法:   {comp['winner']}")
    else:
        print(f"失败 [{status}]: {result}")

def test_compare_small_values_at_end():
    print("\n" + "=" * 70)
    print("测试4: 鸡尾酒优势场景 - 小值在末尾")
    print("=" * 70)
    
    test_array = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    
    status, result = send_request({"array": test_array, "algorithm": "compare"})
    
    if status == 200:
        comp = result['comparison']
        print(f"原始数组: {test_array}")
        print(f"\n冒泡排序:   {comp['bubble_sort']['total_rounds']} 轮, {comp['bubble_sort']['swap_count']} 次交换")
        print(f"鸡尾酒排序: {comp['cocktail_sort']['total_rounds']} 轮, {comp['cocktail_sort']['swap_count']} 次交换")
        print(f"\n轮数减少:   {comp['round_difference']['absolute']} 轮 ({comp['round_difference']['percent']}%)")
        print(f"交换减少:   {comp['swap_difference']['absolute']} 次 ({comp['swap_difference']['percent']}%)")
        print(f"\n胜出算法:   {comp['winner']}")
    else:
        print(f"失败 [{status}]: {result}")

def test_compare_large_values_at_start():
    print("\n" + "=" * 70)
    print("测试5: 鸡尾酒优势场景 - 大值在开头")
    print("=" * 70)
    
    test_array = [100, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    status, result = send_request({"array": test_array, "algorithm": "compare"})
    
    if status == 200:
        comp = result['comparison']
        print(f"原始数组: {test_array}")
        print(f"\n冒泡排序:   {comp['bubble_sort']['total_rounds']} 轮, {comp['bubble_sort']['swap_count']} 次交换")
        print(f"鸡尾酒排序: {comp['cocktail_sort']['total_rounds']} 轮, {comp['cocktail_sort']['swap_count']} 次交换")
        print(f"\n轮数减少:   {comp['round_difference']['absolute']} 轮 ({comp['round_difference']['percent']}%)")
        print(f"交换减少:   {comp['swap_difference']['absolute']} 次 ({comp['swap_difference']['percent']}%)")
        print(f"\n胜出算法:   {comp['winner']}")
    else:
        print(f"失败 [{status}]: {result}")

def test_large_array_compare():
    print("\n" + "=" * 70)
    print("测试6: 大数组对比 (500元素)")
    print("=" * 70)
    
    import random
    random.seed(123)
    test_array = list(range(500, 0, -1))
    
    status, result = send_request({"array": test_array, "algorithm": "compare"})
    
    if status == 200:
        comp = result['comparison']
        print(f"数组长度: 500 (完全逆序)")
        print(f"\n冒泡排序:   {comp['bubble_sort']['total_rounds']} 轮, {comp['bubble_sort']['swap_count']} 次交换")
        print(f"鸡尾酒排序: {comp['cocktail_sort']['total_rounds']} 轮, {comp['cocktail_sort']['swap_count']} 次交换")
        print(f"\n轮数减少:   {comp['round_difference']['absolute']} 轮 ({comp['round_difference']['percent']}%)")
        print(f"交换减少:   {comp['swap_difference']['absolute']} 次 ({comp['swap_difference']['percent']}%)")
        print(f"\n胜出算法:   {comp['winner']}")
    else:
        print(f"失败 [{status}]: {result}")

if __name__ == "__main__":
    test_bubble_sort()
    test_cocktail_sort()
    test_compare_random()
    test_compare_small_values_at_end()
    test_compare_large_values_at_start()
    test_large_array_compare()
