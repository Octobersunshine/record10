def binary_search(arr, target):
    if not arr:
        return -1
    
    is_ascending = arr[0] <= arr[-1]
    left, right = 0, len(arr)
    
    while left < right:
        mid = (left + right) // 2
        
        if is_ascending:
            if arr[mid] >= target:
                right = mid
            else:
                left = mid + 1
        else:
            if arr[mid] <= target:
                right = mid
            else:
                left = mid + 1
    
    if left < len(arr) and arr[left] == target:
        return left
    
    return -1


def binary_search_range(arr, target):
    if not arr:
        return [-1, -1]
    
    is_ascending = arr[0] <= arr[-1]
    
    def find_left_bound():
        left, right = 0, len(arr)
        while left < right:
            mid = (left + right) // 2
            if is_ascending:
                if arr[mid] >= target:
                    right = mid
                else:
                    left = mid + 1
            else:
                if arr[mid] <= target:
                    right = mid
                else:
                    left = mid + 1
        return left
    
    def find_right_bound():
        left, right = 0, len(arr)
        while left < right:
            mid = (left + right) // 2
            if is_ascending:
                if arr[mid] > target:
                    right = mid
                else:
                    left = mid + 1
            else:
                if arr[mid] < target:
                    right = mid
                else:
                    left = mid + 1
        return left - 1
    
    left_idx = find_left_bound()
    
    if left_idx >= len(arr) or arr[left_idx] != target:
        return [-1, -1]
    
    right_idx = find_right_bound()
    return [left_idx, right_idx]


def binary_search_rotated(arr, target):
    if not arr:
        return -1
    
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        
        if arr[left] <= arr[mid]:
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1
