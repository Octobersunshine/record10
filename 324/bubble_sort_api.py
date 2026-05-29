from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def bubble_sort_with_steps(arr, step_interval=1, final_only=False):
    arr = arr.copy()
    n = len(arr)
    steps = []
    total_rounds = 0
    swap_count = 0
    
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
                swap_count += 1
        total_rounds += 1
        
        if final_only:
            if not swapped:
                break
        else:
            if step_interval <= 1 or total_rounds % step_interval == 0:
                steps.append({
                    "round": total_rounds,
                    "array": arr.copy(),
                    "swaps_so_far": swap_count
                })
            if not swapped:
                break
    
    if final_only:
        return {
            "steps": [],
            "final_result": arr,
            "total_rounds": total_rounds,
            "swap_count": swap_count
        }
    
    if steps and steps[-1]["round"] != total_rounds:
        steps.append({
            "round": total_rounds,
            "array": arr.copy(),
            "swaps_so_far": swap_count
        })
    
    return {
        "steps": steps,
        "final_result": arr,
        "total_rounds": total_rounds,
        "swap_count": swap_count
    }

def cocktail_sort_with_steps(arr, step_interval=1, final_only=False):
    arr = arr.copy()
    n = len(arr)
    steps = []
    total_rounds = 0
    swap_count = 0
    start = 0
    end = n - 1
    swapped = True
    
    while swapped:
        swapped = False
        
        for i in range(start, end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True
                swap_count += 1
        
        total_rounds += 1
        direction = "右"
        
        if not swapped:
            if not final_only:
                if step_interval <= 1 or total_rounds % step_interval == 0:
                    steps.append({
                        "round": total_rounds,
                        "direction": direction,
                        "array": arr.copy(),
                        "swaps_so_far": swap_count
                    })
            break
        
        if not final_only:
            if step_interval <= 1 or total_rounds % step_interval == 0:
                steps.append({
                    "round": total_rounds,
                    "direction": direction,
                    "array": arr.copy(),
                    "swaps_so_far": swap_count
                })
        
        end -= 1
        swapped = False
        
        for i in range(end - 1, start - 1, -1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True
                swap_count += 1
        
        total_rounds += 1
        direction = "左"
        
        if not final_only:
            if step_interval <= 1 or total_rounds % step_interval == 0:
                steps.append({
                    "round": total_rounds,
                    "direction": direction,
                    "array": arr.copy(),
                    "swaps_so_far": swap_count
                })
        
        start += 1
    
    if final_only:
        return {
            "steps": [],
            "final_result": arr,
            "total_rounds": total_rounds,
            "swap_count": swap_count
        }
    
    if steps and steps[-1]["round"] != total_rounds:
        steps.append({
            "round": total_rounds,
            "direction": direction,
            "array": arr.copy(),
            "swaps_so_far": swap_count
        })
    
    return {
        "steps": steps,
        "final_result": arr,
        "total_rounds": total_rounds,
        "swap_count": swap_count
    }

@app.route('/api/sort', methods=['POST'])
def sort_api():
    try:
        data = request.get_json()
        
        if not data or 'array' not in data:
            return jsonify({"error": "请提供array参数"}), 400
        
        unsorted_array = data['array']
        
        if not isinstance(unsorted_array, list):
            return jsonify({"error": "array必须是列表类型"}), 400
        
        if len(unsorted_array) == 0:
            return jsonify({"error": "array不能为空"}), 400
        
        algorithm = data.get('algorithm', 'bubble')
        final_only = data.get('final_only', False)
        step_interval = data.get('step_interval', 1)
        
        if algorithm not in ['bubble', 'cocktail', 'compare']:
            return jsonify({"error": "algorithm必须是bubble、cocktail或compare"}), 400
        
        if not isinstance(final_only, bool):
            return jsonify({"error": "final_only必须是布尔值"}), 400
        
        if not isinstance(step_interval, int) or step_interval < 1:
            return jsonify({"error": "step_interval必须是大于0的整数"}), 400
        
        if len(unsorted_array) > 1000 and step_interval <= 1 and not final_only and algorithm != 'compare':
            return jsonify({
                "error": "数组长度超过1000时，请设置step_interval>1或final_only=true以减小响应体积",
                "array_length": len(unsorted_array),
                "suggested_step_interval": max(1, len(unsorted_array) // 50)
            }), 400
        
        if algorithm == 'compare':
            bubble_result = bubble_sort_with_steps(
                unsorted_array,
                step_interval=step_interval,
                final_only=True
            )
            
            cocktail_result = cocktail_sort_with_steps(
                unsorted_array,
                step_interval=step_interval,
                final_only=True
            )
            
            round_reduction = bubble_result['total_rounds'] - cocktail_result['total_rounds']
            round_reduction_percent = (round_reduction / bubble_result['total_rounds'] * 100) if bubble_result['total_rounds'] > 0 else 0
            
            swap_reduction = bubble_result['swap_count'] - cocktail_result['swap_count']
            swap_reduction_percent = (swap_reduction / bubble_result['swap_count'] * 100) if bubble_result['swap_count'] > 0 else 0
            
            comparison = {
                "bubble_sort": {
                    "total_rounds": bubble_result['total_rounds'],
                    "swap_count": bubble_result['swap_count']
                },
                "cocktail_sort": {
                    "total_rounds": cocktail_result['total_rounds'],
                    "swap_count": cocktail_result['swap_count']
                },
                "round_difference": {
                    "absolute": round_reduction,
                    "percent": round(round_reduction_percent, 2)
                },
                "swap_difference": {
                    "absolute": swap_reduction,
                    "percent": round(swap_reduction_percent, 2)
                },
                "winner": "cocktail" if cocktail_result['total_rounds'] < bubble_result['total_rounds'] else "bubble" if cocktail_result['total_rounds'] > bubble_result['total_rounds'] else "tie"
            }
            
            return jsonify({
                "original_array": unsorted_array,
                "final_sorted_array": cocktail_result["final_result"],
                "algorithm": "compare",
                "comparison": comparison
            })
        
        if algorithm == 'bubble':
            result = bubble_sort_with_steps(
                unsorted_array,
                step_interval=step_interval,
                final_only=final_only
            )
            algorithm_name = "冒泡排序"
        else:
            result = cocktail_sort_with_steps(
                unsorted_array,
                step_interval=step_interval,
                final_only=final_only
            )
            algorithm_name = "鸡尾酒排序"
        
        response = {
            "original_array": unsorted_array,
            "final_sorted_array": result["final_result"],
            "total_rounds": result["total_rounds"],
            "swap_count": result["swap_count"],
            "algorithm": algorithm,
            "algorithm_name": algorithm_name,
            "final_only": final_only,
            "step_interval": step_interval
        }
        
        if not final_only:
            response["sorting_steps"] = result["steps"]
            response["returned_steps_count"] = len(result["steps"])
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/bubble-sort', methods=['POST'])
def bubble_sort_api():
    try:
        data = request.get_json()
        
        if not data or 'array' not in data:
            return jsonify({"error": "请提供array参数"}), 400
        
        unsorted_array = data['array']
        
        if not isinstance(unsorted_array, list):
            return jsonify({"error": "array必须是列表类型"}), 400
        
        if len(unsorted_array) == 0:
            return jsonify({"error": "array不能为空"}), 400
        
        final_only = data.get('final_only', False)
        step_interval = data.get('step_interval', 1)
        
        if not isinstance(final_only, bool):
            return jsonify({"error": "final_only必须是布尔值"}), 400
        
        if not isinstance(step_interval, int) or step_interval < 1:
            return jsonify({"error": "step_interval必须是大于0的整数"}), 400
        
        if len(unsorted_array) > 1000 and step_interval <= 1 and not final_only:
            return jsonify({
                "error": "数组长度超过1000时，请设置step_interval>1或final_only=true以减小响应体积",
                "array_length": len(unsorted_array),
                "suggested_step_interval": max(1, len(unsorted_array) // 50)
            }), 400
        
        result = bubble_sort_with_steps(
            unsorted_array,
            step_interval=step_interval,
            final_only=final_only
        )
        
        response = {
            "original_array": unsorted_array,
            "final_sorted_array": result["final_result"],
            "total_rounds": result["total_rounds"],
            "swap_count": result["swap_count"],
            "algorithm": "bubble",
            "algorithm_name": "冒泡排序",
            "final_only": final_only,
            "step_interval": step_interval
        }
        
        if not final_only:
            response["sorting_steps"] = result["steps"]
            response["returned_steps_count"] = len(result["steps"])
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "冒泡排序API运行正常"})

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)
