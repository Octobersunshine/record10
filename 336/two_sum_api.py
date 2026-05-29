from http.server import HTTPServer, BaseHTTPRequestHandler
import json


def two_sum(nums, target):
    lookup = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in lookup:
            return [lookup[complement][0], i]
        if num not in lookup:
            lookup[num] = []
        lookup[num].append(i)
    return None


def three_sum(nums, target=0):
    nums = sorted(nums)
    result = []
    n = len(nums)

    for i in range(n - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        left, right = i + 1, n - 1
        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]
            if current_sum == target:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif current_sum < target:
                left += 1
            else:
                right -= 1

    return result


def four_sum(nums, target):
    nums = sorted(nums)
    result = []
    n = len(nums)

    for i in range(n - 3):
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        for j in range(i + 1, n - 2):
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue

            left, right = j + 1, n - 1
            while left < right:
                current_sum = nums[i] + nums[j] + nums[left] + nums[right]
                if current_sum == target:
                    result.append([nums[i], nums[j], nums[left], nums[right]])
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif current_sum < target:
                    left += 1
                else:
                    right -= 1

    return result


class SumHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/two-sum":
            self.handle_two_sum()
        elif self.path == "/three-sum":
            self.handle_three_sum()
        elif self.path == "/four-sum":
            self.handle_four_sum()
        else:
            self.send_response(404)
            self.end_headers()

    def parse_request(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
            nums = data["nums"]
            target = data.get("target", 0)
            return nums, target, None
        except (json.JSONDecodeError, KeyError) as e:
            return None, None, str(e)

    def validate_nums(self, nums):
        return isinstance(nums, list) and all(isinstance(x, int) for x in nums)

    def send_json_response(self, status_code, data):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def handle_two_sum(self):
        nums, target, error = self.parse_request()
        if error:
            self.send_json_response(400, {"error": f"Invalid request: {error}"})
            return

        if not self.validate_nums(nums):
            self.send_json_response(400, {"error": "'nums' must be a list of integers"})
            return

        if not isinstance(target, int):
            self.send_json_response(400, {"error": "'target' must be an integer"})
            return

        result = two_sum(nums, target)
        if result is None:
            self.send_json_response(404, {"error": "No two sum solution found"})
            return

        self.send_json_response(200, {"indices": result})

    def handle_three_sum(self):
        nums, target, error = self.parse_request()
        if error:
            self.send_json_response(400, {"error": f"Invalid request: {error}"})
            return

        if not self.validate_nums(nums):
            self.send_json_response(400, {"error": "'nums' must be a list of integers"})
            return

        if not isinstance(target, int):
            self.send_json_response(400, {"error": "'target' must be an integer"})
            return

        result = three_sum(nums, target)
        self.send_json_response(200, {"triplets": result, "count": len(result)})

    def handle_four_sum(self):
        nums, target, error = self.parse_request()
        if error:
            self.send_json_response(400, {"error": f"Invalid request: {error}"})
            return

        if not self.validate_nums(nums):
            self.send_json_response(400, {"error": "'nums' must be a list of integers"})
            return

        if not isinstance(target, int):
            self.send_json_response(400, {"error": "'target' must be an integer"})
            return

        result = four_sum(nums, target)
        self.send_json_response(200, {"quadruplets": result, "count": len(result)})

    def log_message(self, format, *args):
        print(f"[SumAPI] {args[0]}")


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8000), SumHandler)
    print("Sum API running on http://0.0.0.0:8000")
    print('POST /two-sum    body: {"nums": [2,7,11,15], "target": 9}')
    print('POST /three-sum  body: {"nums": [-1,0,1,2,-1,-4], "target": 0}')
    print('POST /four-sum   body: {"nums": [1,0,-1,0,-2,2], "target": 0}')
    server.serve_forever()
