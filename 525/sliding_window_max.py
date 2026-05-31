from collections import deque
import bisect
from typing import List, Dict, Tuple


class SlidingWindow:
    def __init__(self, nums: List[float]):
        if not isinstance(nums, list):
            raise TypeError("nums must be a list")
        self.nums = nums
        self.n = len(nums)
        self.left = 0
        self.right = -1
        self._sorted_window: List[Tuple[float, int]] = []
        self._window_sum = 0.0

    def _push(self, idx: int):
        bisect.insort(self._sorted_window, (self.nums[idx], idx))
        self._window_sum += self.nums[idx]

    def _remove_idx(self, idx: int):
        pos = bisect.bisect_left(self._sorted_window, (self.nums[idx], idx))
        if pos < len(self._sorted_window) and self._sorted_window[pos] == (self.nums[idx], idx):
            self._sorted_window.pop(pos)
            self._window_sum -= self.nums[idx]

    def set_window(self, left: int, right: int) -> Dict:
        if not isinstance(left, int) or not isinstance(right, int):
            raise TypeError("left and right must be integers")
        if left < 0 or right >= self.n:
            raise ValueError(f"window bounds out of range: left={left}, right={right}, n={self.n}")
        if left > right:
            raise ValueError(f"left ({left}) must be <= right ({right})")

        while self.right < right:
            self.right += 1
            self._push(self.right)

        while self.left > left:
            self.left -= 1
            self._push(self.left)

        while self.left < left:
            self._remove_idx(self.left)
            self.left += 1

        while self.right > right:
            self._remove_idx(self.right)
            self.right -= 1

        window_size = self.right - self.left + 1
        min_val, min_idx = self._sorted_window[0]
        max_val, max_idx = self._sorted_window[-1]

        mid = window_size // 2
        if window_size % 2 == 1:
            median = float(self._sorted_window[mid][0])
        else:
            median = float(self._sorted_window[mid - 1][0] + self._sorted_window[mid][0]) / 2.0

        return {
            "max": max_val,
            "max_index": max_idx,
            "min": min_val,
            "min_index": min_idx,
            "median": median,
            "avg": self._window_sum / window_size,
            "sum": self._window_sum,
            "size": window_size,
            "left": self.left,
            "right": self.right,
            "elements": self.nums[self.left:self.right + 1]
        }


def sliding_window_stats(nums: List[float], k: int) -> List[Dict]:
    if not isinstance(nums, list):
        raise TypeError("nums must be a list")
    if not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k <= 0:
        return []
    if not nums:
        return []
    if k > len(nums):
        sw = SlidingWindow(nums)
        return [sw.set_window(0, len(nums) - 1)]

    sw = SlidingWindow(nums)
    results = []
    for right in range(k - 1, len(nums)):
        left = right - k + 1
        results.append(sw.set_window(left, right))
    return results


def max_sliding_window(nums, k):
    if not isinstance(nums, list):
        raise TypeError("nums must be a list")
    if not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k <= 0:
        return []
    if not nums:
        return []
    if k > len(nums):
        return [max(nums)]
    dq = deque()
    result = []
    for i in range(len(nums)):
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result


if __name__ == "__main__":
    print("=== Original max_sliding_window ===")
    print(max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3))
    print(max_sliding_window([1], 1))
    print(max_sliding_window([9, 11], 2))
    print(max_sliding_window([4, -2], 2))
    print(max_sliding_window([1, 2, 3], 5))
    print(max_sliding_window([1, 2, 3], 0))
    print(max_sliding_window([], 3))

    print("\n=== sliding_window_stats with fixed k ===")
    results = sliding_window_stats([1, 3, -1, -3, 5, 3, 6, 7], 3)
    for r in results:
        print(f"window={r['elements']}, max={r['max']}@{r['max_index']}, "
              f"min={r['min']}@{r['min_index']}, median={r['median']}, avg={r['avg']:.2f}")

    print("\n=== Dynamic window adjustment ===")
    sw = SlidingWindow([1, 3, -1, -3, 5, 3, 6, 7])
    print("Window [0,2]:", sw.set_window(0, 2))
    print("Window [2,5]:", sw.set_window(2, 5))
    print("Window [4,7]:", sw.set_window(4, 7))
    print("Window [1,6]:", sw.set_window(1, 6))

    print("\n=== Median test ===")
    sw2 = SlidingWindow([1, 2, 3, 4, 5])
    print("Window [0,4] (odd):", sw2.set_window(0, 4)["median"])
    print("Window [0,3] (even):", sw2.set_window(0, 3)["median"])
