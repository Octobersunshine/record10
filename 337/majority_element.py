def majority_element(nums):
    count = 0
    candidate = None
    for num in nums:
        if count == 0:
            candidate = num
        count += 1 if num == candidate else -1

    verify_count = sum(1 for num in nums if num == candidate)
    if verify_count <= len(nums) // 2:
        return None
    return candidate


def majority_element_n3(nums):
    candidate1, candidate2 = None, None
    count1, count2 = 0, 0

    for num in nums:
        if candidate1 is not None and num == candidate1:
            count1 += 1
        elif candidate2 is not None and num == candidate2:
            count2 += 1
        elif count1 == 0:
            candidate1, count1 = num, 1
        elif count2 == 0:
            candidate2, count2 = num, 1
        else:
            count1 -= 1
            count2 -= 1

    result = []
    for c in (candidate1, candidate2):
        if c is not None and sum(1 for num in nums if num == c) > len(nums) // 3:
            result.append(c)
    return result


if __name__ == "__main__":
    assert majority_element([3, 2, 3]) == 3
    assert majority_element([2, 2, 1, 1, 1, 2, 2]) == 2
    assert majority_element([1]) == 1
    assert majority_element([6, 5, 5]) == 5
    assert majority_element([1, 2, 3]) is None
    assert majority_element([1, 2, 3, 4]) is None
    assert majority_element([1, 1, 2, 2]) is None

    assert majority_element_n3([3, 2, 3]) == [3]
    assert majority_element_n3([1, 1, 1, 3, 3, 2, 2, 2]) in ([1, 2], [2, 1])
    assert majority_element_n3([1, 2, 3, 4]) == []
    assert majority_element_n3([1]) == [1]
    assert set(majority_element_n3([1, 2])) == {1, 2}
    assert majority_element_n3([2, 2, 2, 2, 3, 3, 3, 3, 1]) in ([2, 3], [3, 2])
    assert majority_element_n3([1, 1, 1, 2, 2, 2, 3, 3, 3]) == []
    print("All tests passed.")
