def is_valid(board, row, col, num):
    for i in range(9):
        if board[row][i] == num:
            return False

    for i in range(9):
        if board[i][col] == num:
            return False

    start_row = (row // 3) * 3
    start_col = (col // 3) * 3
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == num:
                return False

    return True


def find_empty(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j)
    return None


def _popcount(x):
    return bin(x).count('1')


def _get_numbers_from_mask(mask):
    numbers = []
    while mask:
        lsb = mask & -mask
        num = (lsb.bit_length())
        numbers.append(num)
        mask ^= lsb
    return numbers


def solve_sudoku(board, optimized=True):
    if not isinstance(board, list) or len(board) != 9:
        raise ValueError("数独板必须是9x9的二维列表")

    for row in board:
        if not isinstance(row, list) or len(row) != 9:
            raise ValueError("数独板必须是9x9的二维列表")
        for cell in row:
            if not isinstance(cell, int) or cell < 0 or cell > 9:
                raise ValueError("数独板只能包含0-9的整数")

    board_copy = [row[:] for row in board]

    if not optimized:
        def backtrack_basic():
            empty = find_empty(board_copy)
            if not empty:
                return True
            row, col = empty

            for num in range(1, 10):
                if is_valid(board_copy, row, col, num):
                    board_copy[row][col] = num
                    if backtrack_basic():
                        return True
                    board_copy[row][col] = 0

            return False

        if backtrack_basic():
            return board_copy
        else:
            return None

    candidates = [[0x1FF for _ in range(9)] for _ in range(9)]
    empty_cells = []

    for i in range(9):
        for j in range(9):
            num = board_copy[i][j]
            if num != 0:
                candidates[i][j] = 0
                bit = ~(1 << (num - 1))
                for k in range(9):
                    candidates[i][k] &= bit
                    candidates[k][j] &= bit
                start_row = (i // 3) * 3
                start_col = (j // 3) * 3
                for di in range(3):
                    for dj in range(3):
                        candidates[start_row + di][start_col + dj] &= bit
            else:
                empty_cells.append((i, j))

    def find_best_empty_fast():
        best_count = 10
        best_pos = None
        best_mask = 0
        best_idx = -1

        for idx, (i, j) in enumerate(empty_cells):
            mask = candidates[i][j]
            if mask != 0:
                count = _popcount(mask)
                if count < best_count:
                    best_count = count
                    best_pos = (i, j)
                    best_mask = mask
                    best_idx = idx
                    if count == 1:
                        return best_pos, best_mask, best_idx

        return best_pos, best_mask, best_idx

    def update_candidates_fast(row, col, num):
        bit = ~(1 << (num - 1))
        affected = []

        for k in range(9):
            if k != col:
                if candidates[row][k] & (1 << (num - 1)):
                    candidates[row][k] &= bit
                    affected.append((row, k))
            if k != row:
                if candidates[k][col] & (1 << (num - 1)):
                    candidates[k][col] &= bit
                    affected.append((k, col))

        start_row = (row // 3) * 3
        start_col = (col // 3) * 3
        for di in range(3):
            for dj in range(3):
                ri, cj = start_row + di, start_col + dj
                if ri != row and cj != col:
                    if candidates[ri][cj] & (1 << (num - 1)):
                        candidates[ri][cj] &= bit
                        affected.append((ri, cj))

        return affected

    def restore_candidates_fast(affected, num):
        bit = 1 << (num - 1)
        for r, c in affected:
            candidates[r][c] |= bit

    def backtrack_optimized_fast():
        if not empty_cells:
            return True

        best_pos, best_mask, best_idx = find_best_empty_fast()
        if best_pos is None:
            return True
        if best_mask == 0:
            return False

        row, col = best_pos
        candidates[row][col] = 0
        empty_cells.pop(best_idx)

        numbers = _get_numbers_from_mask(best_mask)

        for num in numbers:
            affected = update_candidates_fast(row, col, num)
            board_copy[row][col] = num

            valid = True
            for r, c in affected:
                if board_copy[r][c] == 0 and candidates[r][c] == 0:
                    valid = False
                    break

            if valid and backtrack_optimized_fast():
                return True

            board_copy[row][col] = 0
            restore_candidates_fast(affected, num)

        empty_cells.insert(best_idx, (row, col))
        candidates[row][col] = best_mask
        return False

    if backtrack_optimized_fast():
        return board_copy
    else:
        return None


def solve_sudoku_basic(board):
    return solve_sudoku(board, optimized=False)


def solve_sudoku_optimized(board):
    return solve_sudoku(board, optimized=True)


def print_board(board):
    if board is None:
        print("无解")
        return

    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - ")
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")
            if j == 8:
                print(board[i][j])
            else:
                print(str(board[i][j]) + " ", end="")


def validate_sudoku(board, allow_empty=True):
    if not isinstance(board, list) or len(board) != 9:
        return False, "数独板必须是9x9的二维列表"

    for row in board:
        if not isinstance(row, list) or len(row) != 9:
            return False, "数独板必须是9x9的二维列表"
        for cell in row:
            if not isinstance(cell, int) or cell < 0 or cell > 9:
                return False, "数独板只能包含0-9的整数"

    for i in range(9):
        row_nums = set()
        for j in range(9):
            num = board[i][j]
            if num != 0:
                if num in row_nums:
                    return False, f"第 {i+1} 行存在重复数字 {num}"
                row_nums.add(num)

    for j in range(9):
        col_nums = set()
        for i in range(9):
            num = board[i][j]
            if num != 0:
                if num in col_nums:
                    return False, f"第 {j+1} 列存在重复数字 {num}"
                col_nums.add(num)

    for box_row in range(3):
        for box_col in range(3):
            box_nums = set()
            for i in range(3):
                for j in range(3):
                    num = board[box_row*3 + i][box_col*3 + j]
                    if num != 0:
                        if num in box_nums:
                            return False, f"第 {box_row*3+1}-{box_row*3+3} 行, 第 {box_col*3+1}-{box_col*3+3} 列的3x3宫格存在重复数字 {num}"
                        box_nums.add(num)

    if not allow_empty:
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    return False, "数独板存在空格"

    return True, "数独有效"


def count_solutions(board, limit=2):
    board_copy = [row[:] for row in board]
    count = [0]

    candidates = [[0x1FF for _ in range(9)] for _ in range(9)]
    empty_cells = []

    for i in range(9):
        for j in range(9):
            num = board_copy[i][j]
            if num != 0:
                candidates[i][j] = 0
                bit = ~(1 << (num - 1))
                for k in range(9):
                    candidates[i][k] &= bit
                    candidates[k][j] &= bit
                start_row = (i // 3) * 3
                start_col = (j // 3) * 3
                for di in range(3):
                    for dj in range(3):
                        candidates[start_row + di][start_col + dj] &= bit
            else:
                empty_cells.append((i, j))

    def find_best_empty_fast():
        best_count = 10
        best_pos = None
        best_mask = 0
        best_idx = -1

        for idx, (i, j) in enumerate(empty_cells):
            mask = candidates[i][j]
            if mask != 0:
                popcnt = bin(mask).count('1')
                if popcnt < best_count:
                    best_count = popcnt
                    best_pos = (i, j)
                    best_mask = mask
                    best_idx = idx
                    if popcnt == 1:
                        return best_pos, best_mask, best_idx

        return best_pos, best_mask, best_idx

    def backtrack():
        if count[0] >= limit:
            return

        best_pos, best_mask, best_idx = find_best_empty_fast()
        if best_pos is None:
            count[0] += 1
            return
        if best_mask == 0:
            return

        row, col = best_pos
        candidates[row][col] = 0
        empty_cells.pop(best_idx)

        numbers = []
        m = best_mask
        while m:
            lsb = m & -m
            num = lsb.bit_length()
            numbers.append(num)
            m ^= lsb

        for num in numbers:
            bit = ~(1 << (num - 1))
            affected = []

            for k in range(9):
                if k != col:
                    if candidates[row][k] & (1 << (num - 1)):
                        candidates[row][k] &= bit
                        affected.append((row, k))
                if k != row:
                    if candidates[k][col] & (1 << (num - 1)):
                        candidates[k][col] &= bit
                        affected.append((k, col))

            start_row = (row // 3) * 3
            start_col = (col // 3) * 3
            for di in range(3):
                for dj in range(3):
                    ri, cj = start_row + di, start_col + dj
                    if ri != row and cj != col:
                        if candidates[ri][cj] & (1 << (num - 1)):
                            candidates[ri][cj] &= bit
                            affected.append((ri, cj))

            board_copy[row][col] = num

            valid = True
            for r, c in affected:
                if board_copy[r][c] == 0 and candidates[r][c] == 0:
                    valid = False
                    break

            if valid:
                backtrack()

            board_copy[row][col] = 0

            restore_bit = 1 << (num - 1)
            for r, c in affected:
                candidates[r][c] |= restore_bit

        empty_cells.insert(best_idx, (row, col))
        candidates[row][col] = best_mask

    backtrack()
    return count[0]


def has_unique_solution(board):
    return count_solutions(board, limit=2) == 1


import random

DIFFICULTY_LEVELS = {
    'easy': {'empty_cells': (30, 35), 'name': '简单'},
    'medium': {'empty_cells': (36, 45), 'name': '中等'},
    'hard': {'empty_cells': (46, 52), 'name': '困难'},
    'expert': {'empty_cells': (53, 58), 'name': '专家'}
}


def _generate_full_solution():
    board = [[0 for _ in range(9)] for _ in range(9)]
    nums = list(range(1, 10))

    def fill_board():
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    random.shuffle(nums)
                    for num in nums:
                        if is_valid(board, i, j, num):
                            board[i][j] = num
                            if fill_board():
                                return True
                            board[i][j] = 0
                    return False
        return True

    fill_board()
    return board


def generate_sudoku(difficulty='medium', ensure_unique=True, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)

    if difficulty not in DIFFICULTY_LEVELS:
        raise ValueError(f"难度级别必须是: {', '.join(DIFFICULTY_LEVELS.keys())}")

    min_empty, max_empty = DIFFICULTY_LEVELS[difficulty]['empty_cells']
    target_empty = random.randint(min_empty, max_empty)

    solution = _generate_full_solution()
    puzzle = [row[:] for row in solution]

    positions = [(i, j) for i in range(9) for j in range(9)]
    random.shuffle(positions)

    empty_count = 0
    for i, j in positions:
        if empty_count >= target_empty:
            break

        original_value = puzzle[i][j]
        puzzle[i][j] = 0

        if ensure_unique:
            if not has_unique_solution(puzzle):
                puzzle[i][j] = original_value
                continue

        empty_count += 1

    return puzzle, solution


def get_difficulty_info():
    return {k: {
        'name': v['name'],
        'empty_cells_range': v['empty_cells'],
        'clues_range': (81 - v['empty_cells'][1], 81 - v['empty_cells'][0])
    } for k, v in DIFFICULTY_LEVELS.items()}


if __name__ == "__main__":
    board = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]

    print("原始数独:")
    print_board(board)
    print("\n求解结果:")
    solution = solve_sudoku(board)
    print_board(solution)

    print("\n" + "=" * 50)
    print("数独生成器演示")
    print("=" * 50)

    for diff in ['easy', 'medium', 'hard', 'expert']:
        print(f"\n生成 {DIFFICULTY_LEVELS[diff]['name']} 难度数独:")
        puzzle, sol = generate_sudoku(difficulty=diff, random_seed=42)
        empty_count = sum(1 for i in range(9) for j in range(9) if puzzle[i][j] == 0)
        print(f"空格数: {empty_count}, 提示数: {81 - empty_count}")
        print("\n谜题:")
        print_board(puzzle)
        print(f"\n唯一解: {has_unique_solution(puzzle)}")
        break
