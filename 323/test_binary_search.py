from binary_search import binary_search, binary_search_range, binary_search_rotated

def test_ascending_array():
    arr = [1, 3, 5, 7, 9, 11, 13, 15]
    assert binary_search(arr, 7) == 3
    assert binary_search(arr, 1) == 0
    assert binary_search(arr, 15) == 7
    assert binary_search(arr, 2) == -1
    assert binary_search(arr, 0) == -1
    assert binary_search(arr, 16) == -1
    print("升序数组测试通过!")

def test_descending_array():
    arr = [15, 13, 11, 9, 7, 5, 3, 1]
    assert binary_search(arr, 7) == 4
    assert binary_search(arr, 15) == 0
    assert binary_search(arr, 1) == 7
    assert binary_search(arr, 2) == -1
    assert binary_search(arr, 0) == -1
    assert binary_search(arr, 16) == -1
    print("降序数组测试通过!")

def test_edge_cases():
    assert binary_search([], 5) == -1
    assert binary_search([5], 5) == 0
    assert binary_search([5], 3) == -1
    assert binary_search([2, 2, 2], 2) == 0
    print("边界情况测试通过!")

def test_duplicate_elements():
    arr_asc = [1, 2, 2, 2, 3, 4, 4, 5, 5, 5, 5]
    assert binary_search(arr_asc, 2) == 1
    assert binary_search(arr_asc, 4) == 5
    assert binary_search(arr_asc, 5) == 7
    assert binary_search(arr_asc, 1) == 0
    
    arr_desc = [5, 5, 5, 5, 4, 4, 3, 2, 2, 2, 1]
    assert binary_search(arr_desc, 5) == 0
    assert binary_search(arr_desc, 4) == 4
    assert binary_search(arr_desc, 2) == 7
    assert binary_search(arr_desc, 1) == 10
    
    arr_all_same = [3, 3, 3, 3, 3]
    assert binary_search(arr_all_same, 3) == 0
    assert binary_search(arr_all_same, 4) == -1
    
    arr_leading_dup = [1, 1, 1, 2, 3, 4]
    assert binary_search(arr_leading_dup, 1) == 0
    
    arr_trailing_dup = [1, 2, 3, 4, 4, 4]
    assert binary_search(arr_trailing_dup, 4) == 3
    
    print("重复元素测试通过!")

def test_string_array():
    arr_asc = ['apple', 'banana', 'cherry', 'date', 'grape']
    assert binary_search(arr_asc, 'cherry') == 2
    assert binary_search(arr_asc, 'orange') == -1
    
    arr_desc = ['grape', 'date', 'cherry', 'banana', 'apple']
    assert binary_search(arr_desc, 'cherry') == 2
    
    arr_dup_str = ['apple', 'apple', 'banana', 'banana', 'cherry']
    assert binary_search(arr_dup_str, 'apple') == 0
    assert binary_search(arr_dup_str, 'banana') == 2
    
    print("字符串数组测试通过!")

def test_search_range():
    arr_asc = [1, 2, 2, 2, 3, 4, 4, 5, 5, 5, 5]
    assert binary_search_range(arr_asc, 2) == [1, 3]
    assert binary_search_range(arr_asc, 4) == [5, 6]
    assert binary_search_range(arr_asc, 5) == [7, 10]
    assert binary_search_range(arr_asc, 1) == [0, 0]
    assert binary_search_range(arr_asc, 3) == [4, 4]
    assert binary_search_range(arr_asc, 0) == [-1, -1]
    assert binary_search_range(arr_asc, 6) == [-1, -1]
    
    arr_desc = [5, 5, 5, 5, 4, 4, 3, 2, 2, 2, 1]
    assert binary_search_range(arr_desc, 5) == [0, 3]
    assert binary_search_range(arr_desc, 4) == [4, 5]
    assert binary_search_range(arr_desc, 2) == [7, 9]
    assert binary_search_range(arr_desc, 1) == [10, 10]
    
    arr_all_same = [3, 3, 3, 3, 3]
    assert binary_search_range(arr_all_same, 3) == [0, 4]
    assert binary_search_range(arr_all_same, 4) == [-1, -1]
    
    assert binary_search_range([], 5) == [-1, -1]
    assert binary_search_range([5], 5) == [0, 0]
    assert binary_search_range([5], 3) == [-1, -1]
    
    arr_dup_str = ['apple', 'apple', 'banana', 'banana', 'cherry']
    assert binary_search_range(arr_dup_str, 'apple') == [0, 1]
    assert binary_search_range(arr_dup_str, 'banana') == [2, 3]
    assert binary_search_range(arr_dup_str, 'cherry') == [4, 4]
    
    arr_asc_no_dup = [1, 3, 5, 7, 9]
    assert binary_search_range(arr_asc_no_dup, 5) == [2, 2]
    assert binary_search_range(arr_asc_no_dup, 6) == [-1, -1]
    
    print("区间查找测试通过!")

def test_search_rotated():
    arr1 = [4, 5, 6, 1, 2, 3]
    assert binary_search_rotated(arr1, 4) == 0
    assert binary_search_rotated(arr1, 5) == 1
    assert binary_search_rotated(arr1, 6) == 2
    assert binary_search_rotated(arr1, 1) == 3
    assert binary_search_rotated(arr1, 2) == 4
    assert binary_search_rotated(arr1, 3) == 5
    assert binary_search_rotated(arr1, 0) == -1
    assert binary_search_rotated(arr1, 7) == -1
    
    arr2 = [5, 1, 3]
    assert binary_search_rotated(arr2, 5) == 0
    assert binary_search_rotated(arr2, 1) == 1
    assert binary_search_rotated(arr2, 3) == 2
    assert binary_search_rotated(arr2, 0) == -1
    
    arr3 = [1, 2, 3, 4, 5]
    assert binary_search_rotated(arr3, 3) == 2
    assert binary_search_rotated(arr3, 1) == 0
    assert binary_search_rotated(arr3, 5) == 4
    assert binary_search_rotated(arr3, 0) == -1
    
    arr4 = [2, 1]
    assert binary_search_rotated(arr4, 2) == 0
    assert binary_search_rotated(arr4, 1) == 1
    assert binary_search_rotated(arr4, 3) == -1
    
    arr5 = [3, 4, 5, 6, 7, 8, 1, 2]
    assert binary_search_rotated(arr5, 7) == 4
    assert binary_search_rotated(arr5, 8) == 5
    assert binary_search_rotated(arr5, 1) == 6
    assert binary_search_rotated(arr5, 2) == 7
    assert binary_search_rotated(arr5, 3) == 0
    assert binary_search_rotated(arr5, 6) == 3
    
    assert binary_search_rotated([], 5) == -1
    assert binary_search_rotated([5], 5) == 0
    assert binary_search_rotated([5], 3) == -1
    
    print("循环有序数组测试通过!")

if __name__ == "__main__":
    test_ascending_array()
    test_descending_array()
    test_edge_cases()
    test_duplicate_elements()
    test_string_array()
    test_search_range()
    test_search_rotated()
    print("\n所有测试通过!")
