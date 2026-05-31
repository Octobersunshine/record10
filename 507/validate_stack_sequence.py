from itertools import permutations
import math

def catalan_number(n):
    if n < 0:
        return 0
    return math.comb(2 * n, n) // (n + 1)

def generate_all_valid_pop_sequences(push_order):
    n = len(push_order)
    result = []
    
    def backtrack(stack, current_pop, push_index):
        if len(current_pop) == n:
            result.append(current_pop.copy())
            return
        
        if push_index < n:
            stack.append(push_order[push_index])
            backtrack(stack, current_pop, push_index + 1)
            stack.pop()
        
        if stack:
            popped = stack.pop()
            current_pop.append(popped)
            backtrack(stack, current_pop, push_index)
            current_pop.pop()
            stack.append(popped)
    
    backtrack([], [], 0)
    return result

def brute_force_valid_sequences(push_order):
    n = len(push_order)
    valid_sequences = []
    
    for perm in permutations(push_order):
        is_valid, _ = validate_stack_sequence(push_order, list(perm))
        if is_valid:
            valid_sequences.append(list(perm))
    
    return valid_sequences

def validate_stack_sequence(push_order, pop_order):
    if not push_order and not pop_order:
        return True, []
    
    if len(push_order) != len(pop_order):
        return False, []
    
    if sorted(push_order) != sorted(pop_order):
        return False, []
    
    stack = []
    push_index = 0
    pop_index = 0
    process = []
    
    while pop_index < len(pop_order):
        if stack and stack[-1] == pop_order[pop_index]:
            popped = stack.pop()
            process.append(f"弹出 {popped}")
            pop_index += 1
        else:
            if push_index < len(push_order):
                stack.append(push_order[push_index])
                process.append(f"压入 {push_order[push_index]}")
                push_index += 1
            else:
                break
    
    is_valid = not stack and pop_index == len(pop_order)
    return is_valid, process

def get_push_order_from_user():
    print("\n请选择入栈序列输入方式:")
    print("  1. 自动生成 1~n (默认)")
    print("  2. 自定义入栈序列")
    choice = input("请输入选项 (1 或 2，直接回车默认 1): ").strip()
    
    if choice == "2":
        push_input = input("请输入入栈序列 (用空格分隔): ")
        push_order = list(map(int, push_input.split()))
        if not push_order:
            print("错误: 入栈序列不能为空")
            return None
        n = len(push_order)
    else:
        n_input = input("请输入 n 的值: ").strip()
        if not n_input.isdigit() or int(n_input) <= 0:
            print("错误: n 必须是正整数")
            return None
        n = int(n_input)
        push_order = list(range(1, n + 1))
    
    return push_order

def main():
    print("=" * 50)
    print("         栈序列验证与生成工具")
    print("=" * 50)
    print("\n请选择功能:")
    print("  1. 验证单个出栈序列是否合法")
    print("  2. 生成所有合法出栈序列 (回溯法)")
    print("  3. 暴力枚举所有排列并验证 (对比卡特兰数)")
    print("  4. 仅计算卡特兰数 C_n")
    
    func_choice = input("\n请输入选项 (1-4，默认 1): ").strip() or "1"
    
    if func_choice == "4":
        n_input = input("请输入 n 的值: ").strip()
        if not n_input.isdigit() or int(n_input) < 0:
            print("错误: n 必须是非负整数")
            return
        n = int(n_input)
        catalan = catalan_number(n)
        print("\n" + "=" * 50)
        print(f"卡特兰数 C({n}) = {catalan}")
        print("=" * 50)
        return
    
    push_order = get_push_order_from_user()
    if push_order is None:
        return
    
    n = len(push_order)
    
    if func_choice == "1":
        pop_input = input(f"请输入出栈序列 (用空格分隔 {n} 个数字): ")
        pop_order = list(map(int, pop_input.split()))
        
        if len(pop_order) != n:
            print(f"错误: 出栈序列长度 ({len(pop_order)}) 与入栈序列长度 ({n}) 不匹配")
            return
        
        push_set = set(push_order)
        pop_set = set(pop_order)
        if push_set != pop_set:
            extra = pop_set - push_set
            missing = push_set - pop_set
            print("错误: 入栈序列和出栈序列元素不匹配")
            if extra:
                print(f"  出栈序列包含额外元素: {sorted(extra)}")
            if missing:
                print(f"  出栈序列缺少元素: {sorted(missing)}")
            return
        
        is_valid, process = validate_stack_sequence(push_order, pop_order)
        
        print("\n" + "=" * 50)
        print(f"入栈顺序: {push_order}")
        print(f"出栈序列: {pop_order}")
        print("=" * 50)
        print("\n模拟过程:")
        if not process:
            print("  (无操作)")
        else:
            for step, action in enumerate(process, 1):
                print(f"  步骤 {step}: {action}")
        
        print("\n" + "=" * 50)
        if is_valid:
            print("✓ 该出栈序列是合法的!")
        else:
            print("✗ 该出栈序列是不合法的!")
        print("=" * 50)
    
    elif func_choice == "2":
        import time
        start_time = time.time()
        sequences = generate_all_valid_pop_sequences(push_order)
        elapsed = time.time() - start_time
        
        catalan = catalan_number(n)
        
        print("\n" + "=" * 50)
        print(f"入栈顺序: {push_order}")
        print(f"n = {n}")
        print("=" * 50)
        print(f"\n卡特兰数 C({n}) = {catalan}")
        print(f"回溯法生成的合法出栈序列数: {len(sequences)}")
        print(f"是否一致: {'✓ 是' if len(sequences) == catalan else '✗ 否'}")
        print(f"耗时: {elapsed:.4f} 秒")
        
        if sequences:
            print(f"\n所有 {len(sequences)} 个合法出栈序列:")
            for i, seq in enumerate(sequences, 1):
                print(f"  {i:2d}. {seq}")
        print("=" * 50)
    
    elif func_choice == "3":
        import time
        
        print("\n" + "=" * 50)
        print(f"入栈顺序: {push_order}")
        print(f"n = {n}")
        print("=" * 50)
        
        total_permutations = math.factorial(n)
        print(f"\n总排列数: {total_permutations}")
        print(f"卡特兰数 C({n}) = {catalan_number(n)}")
        
        if n > 10:
            print(f"\n警告: n={n} 时排列数达 {total_permutations}，暴力枚举可能很慢!")
            confirm = input("是否继续? (y/n): ").strip().lower()
            if confirm != 'y':
                print("已取消")
                return
        
        print("\n正在执行暴力枚举...")
        start_time = time.time()
        brute_sequences = brute_force_valid_sequences(push_order)
        brute_time = time.time() - start_time
        
        print("\n正在执行回溯法...")
        start_time = time.time()
        backtrack_sequences = generate_all_valid_pop_sequences(push_order)
        backtrack_time = time.time() - start_time
        
        catalan = catalan_number(n)
        
        print("\n" + "=" * 50)
        print("                结果对比")
        print("=" * 50)
        print(f"{'方法':<20} {'序列数':<12} {'耗时(秒)':<12}")
        print("-" * 50)
        print(f"{'卡特兰数公式':<20} {catalan:<12} {'-':<12}")
        print(f"{'回溯法':<20} {len(backtrack_sequences):<12} {backtrack_time:<12.4f}")
        print(f"{'暴力枚举':<20} {len(brute_sequences):<12} {brute_time:<12.4f}")
        print("=" * 50)
        
        all_match = (len(backtrack_sequences) == catalan and 
                     len(brute_sequences) == catalan)
        
        sorted_backtrack = sorted(backtrack_sequences)
        sorted_brute = sorted(brute_sequences)
        sequences_match = sorted_backtrack == sorted_brute
        
        print(f"\n数量一致: {'✓ 是' if all_match else '✗ 否'}")
        print(f"序列内容一致: {'✓ 是' if sequences_match else '✗ 否'}")
        
        if all_match and sequences_match:
            print("\n✓ 所有方法结果完全一致，验证通过!")
        else:
            print("\n✗ 结果不一致，存在错误!")
        
        if n <= 6:
            print(f"\n所有 {catalan} 个合法出栈序列:")
            for i, seq in enumerate(sorted_backtrack, 1):
                print(f"  {i:2d}. {seq}")
        
        print("=" * 50)
    
    else:
        print("错误: 无效的选项")

if __name__ == "__main__":
    main()
