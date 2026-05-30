import unittest
import time
from sudoku_solver import (
    solve_sudoku, solve_sudoku_basic, solve_sudoku_optimized,
    is_valid, find_empty, print_board,
    validate_sudoku, count_solutions, has_unique_solution,
    generate_sudoku, get_difficulty_info, DIFFICULTY_LEVELS
)


def is_valid_solution(board):
    if board is None:
        return False

    for i in range(9):
        if set(board[i]) != {1, 2, 3, 4, 5, 6, 7, 8, 9}:
            return False
        if set([board[j][i] for j in range(9)]) != {1, 2, 3, 4, 5, 6, 7, 8, 9}:
            return False

    for box_row in range(3):
        for box_col in range(3):
            box_nums = []
            for i in range(3):
                for j in range(3):
                    box_nums.append(board[box_row*3 + i][box_col*3 + j])
            if set(box_nums) != {1, 2, 3, 4, 5, 6, 7, 8, 9}:
                return False

    return True


HARD_SUDOKU_BOARDS = [
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 0, 8, 5],
        [0, 0, 1, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 5, 0, 7, 0, 0, 0],
        [0, 0, 4, 0, 0, 0, 1, 0, 0],
        [0, 9, 0, 0, 0, 0, 0, 0, 0],
        [5, 0, 0, 0, 0, 0, 0, 7, 3],
        [0, 0, 2, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 0, 0, 0, 9]
    ],
    [
        [8, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 6, 0, 0, 0, 0, 0],
        [0, 7, 0, 0, 9, 0, 2, 0, 0],
        [0, 5, 0, 0, 0, 7, 0, 0, 0],
        [0, 0, 0, 0, 4, 5, 7, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 3, 0],
        [0, 0, 1, 0, 0, 0, 0, 6, 8],
        [0, 0, 8, 5, 0, 0, 0, 1, 0],
        [0, 9, 0, 0, 0, 0, 4, 0, 0]
    ],
    [
        [0, 0, 5, 3, 0, 0, 0, 0, 0],
        [8, 0, 0, 0, 0, 0, 0, 2, 0],
        [0, 7, 0, 0, 1, 0, 5, 0, 0],
        [4, 0, 0, 0, 0, 5, 3, 0, 0],
        [0, 1, 0, 0, 7, 0, 0, 0, 6],
        [0, 0, 3, 2, 0, 0, 0, 8, 0],
        [0, 6, 0, 5, 0, 0, 0, 0, 9],
        [0, 0, 4, 0, 0, 0, 0, 3, 0],
        [0, 0, 0, 0, 0, 9, 7, 0, 0]
    ]
]


class TestSudokuSolver(unittest.TestCase):

    def test_valid_sudoku(self):
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
        solution = solve_sudoku(board)
        self.assertIsNotNone(solution)
        self.assertTrue(is_valid_solution(solution))

    def test_unsolvable_sudoku(self):
        board = [
            [5, 3, 5, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9]
        ]
        solution = solve_sudoku(board)
        self.assertIsNone(solution)

    def test_empty_board(self):
        board = [[0 for _ in range(9)] for _ in range(9)]
        solution = solve_sudoku(board)
        self.assertIsNotNone(solution)
        self.assertTrue(is_valid_solution(solution))

    def test_already_solved_board(self):
        board = [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9]
        ]
        solution = solve_sudoku(board)
        self.assertIsNotNone(solution)
        self.assertEqual(solution, board)

    def test_invalid_board_size(self):
        board = [[0 for _ in range(8)] for _ in range(9)]
        with self.assertRaises(ValueError) as context:
            solve_sudoku(board)
        self.assertIn("9x9", str(context.exception))

    def test_invalid_cell_value(self):
        board = [[0 for _ in range(9)] for _ in range(9)]
        board[0][0] = 10
        with self.assertRaises(ValueError) as context:
            solve_sudoku(board)
        self.assertIn("0-9", str(context.exception))

    def test_is_valid_function(self):
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
        self.assertTrue(is_valid(board, 0, 2, 4))
        self.assertFalse(is_valid(board, 0, 2, 3))
        self.assertFalse(is_valid(board, 0, 2, 6))
        self.assertFalse(is_valid(board, 0, 2, 8))

    def test_find_empty_function(self):
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
        self.assertEqual(find_empty(board), (0, 2))
        full_board = [[1 for _ in range(9)] for _ in range(9)]
        self.assertIsNone(find_empty(full_board))

    def test_optimized_vs_basic_consistency(self):
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
        solution_basic = solve_sudoku_basic(board)
        solution_optimized = solve_sudoku_optimized(board)
        self.assertIsNotNone(solution_basic)
        self.assertIsNotNone(solution_optimized)
        self.assertEqual(solution_basic, solution_optimized)
        self.assertTrue(is_valid_solution(solution_basic))

    def test_unsolvable_consistency(self):
        board = [
            [5, 3, 5, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9]
        ]
        solution_basic = solve_sudoku_basic(board)
        solution_optimized = solve_sudoku_optimized(board)
        self.assertIsNone(solution_basic)
        self.assertIsNone(solution_optimized)

    def test_hard_sudoku_optimized(self):
        for board in HARD_SUDOKU_BOARDS:
            solution = solve_sudoku_optimized(board)
            self.assertIsNotNone(solution)
            self.assertTrue(is_valid_solution(solution))

    def test_performance_10x_speedup(self):
        print("\n" + "=" * 60)
        print("性能测试：验证启发式搜索加速效果")
        print("=" * 60)

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

        iterations = 20
        basic_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            sol_basic = solve_sudoku_basic(board)
            basic_times.append(time.perf_counter() - start)
        avg_basic = sum(basic_times) / len(basic_times)
        print(f"基础版 ({iterations}次平均): {avg_basic*1000:.2f} 毫秒")

        opt_iterations = 100
        opt_times = []
        for _ in range(opt_iterations):
            start = time.perf_counter()
            sol_opt = solve_sudoku_optimized(board)
            opt_times.append(time.perf_counter() - start)
        avg_opt = sum(opt_times) / len(opt_times)
        print(f"优化版 ({opt_iterations}次平均): {avg_opt*1000:.2f} 毫秒")

        speedup = avg_basic / avg_opt if avg_opt > 0 else float('inf')
        print(f"加速比: {speedup:.1f}x")

        self.assertEqual(sol_basic, sol_opt)
        self.assertGreater(speedup, 10,
                          f"优化效果未达到10倍加速，当前仅 {speedup:.1f} 倍")

        print(f"✓ 性能提升目标达成！加速比 {speedup:.1f}x >= 10x")
        print("=" * 60 + "\n")


class TestSudokuValidator(unittest.TestCase):

    def test_validate_valid_puzzle(self):
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
        valid, msg = validate_sudoku(board)
        self.assertTrue(valid, msg)

    def test_validate_valid_solution(self):
        board = [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9]
        ]
        valid, msg = validate_sudoku(board, allow_empty=False)
        self.assertTrue(valid, msg)

    def test_validate_row_duplicate(self):
        board = [[0 for _ in range(9)] for _ in range(9)]
        board[0][0] = 5
        board[0][5] = 5
        valid, msg = validate_sudoku(board)
        self.assertFalse(valid)
        self.assertIn("行", msg)

    def test_validate_col_duplicate(self):
        board = [[0 for _ in range(9)] for _ in range(9)]
        board[0][0] = 5
        board[5][0] = 5
        valid, msg = validate_sudoku(board)
        self.assertFalse(valid)
        self.assertIn("列", msg)

    def test_validate_box_duplicate(self):
        board = [[0 for _ in range(9)] for _ in range(9)]
        board[0][0] = 5
        board[1][1] = 5
        valid, msg = validate_sudoku(board)
        self.assertFalse(valid)
        self.assertIn("宫格", msg)

    def test_validate_empty_not_allowed(self):
        board = [[0 for _ in range(9)] for _ in range(9)]
        valid, msg = validate_sudoku(board, allow_empty=False)
        self.assertFalse(valid)
        self.assertIn("空格", msg)

    def test_validate_invalid_size(self):
        board = [[0 for _ in range(8)] for _ in range(9)]
        valid, msg = validate_sudoku(board)
        self.assertFalse(valid)

    def test_validate_invalid_value(self):
        board = [[0 for _ in range(9)] for _ in range(9)]
        board[0][0] = 10
        valid, msg = validate_sudoku(board)
        self.assertFalse(valid)


class TestSudokuSolutionCounter(unittest.TestCase):

    def test_count_unique_solution(self):
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
        count = count_solutions(board, limit=2)
        self.assertEqual(count, 1)
        self.assertTrue(has_unique_solution(board))

    def test_count_multiple_solutions(self):
        board = [[0 for _ in range(9)] for _ in range(9)]
        count = count_solutions(board, limit=3)
        self.assertGreater(count, 1)
        self.assertFalse(has_unique_solution(board))

    def test_count_no_solution(self):
        board = [
            [5, 3, 5, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9]
        ]
        count = count_solutions(board)
        self.assertEqual(count, 0)
        self.assertFalse(has_unique_solution(board))

    def test_count_full_board(self):
        board = [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9]
        ]
        count = count_solutions(board)
        self.assertEqual(count, 1)


class TestSudokuGenerator(unittest.TestCase):

    def test_generate_easy(self):
        puzzle, solution = generate_sudoku('easy', random_seed=42)
        valid, msg = validate_sudoku(puzzle)
        self.assertTrue(valid, msg)
        empty_count = sum(1 for i in range(9) for j in range(9) if puzzle[i][j] == 0)
        self.assertGreaterEqual(empty_count, 30)
        self.assertLessEqual(empty_count, 35)
        self.assertTrue(has_unique_solution(puzzle))
        sol = solve_sudoku(puzzle)
        self.assertEqual(sol, solution)

    def test_generate_medium(self):
        puzzle, solution = generate_sudoku('medium', random_seed=42)
        valid, msg = validate_sudoku(puzzle)
        self.assertTrue(valid, msg)
        empty_count = sum(1 for i in range(9) for j in range(9) if puzzle[i][j] == 0)
        self.assertGreaterEqual(empty_count, 36)
        self.assertLessEqual(empty_count, 45)
        self.assertTrue(has_unique_solution(puzzle))

    def test_generate_hard(self):
        puzzle, solution = generate_sudoku('hard', ensure_unique=False, random_seed=42)
        valid, msg = validate_sudoku(puzzle)
        self.assertTrue(valid, msg)
        empty_count = sum(1 for i in range(9) for j in range(9) if puzzle[i][j] == 0)
        self.assertGreaterEqual(empty_count, 46)
        self.assertLessEqual(empty_count, 52)

    def test_generate_expert(self):
        puzzle, solution = generate_sudoku('expert', ensure_unique=False, random_seed=42)
        valid, msg = validate_sudoku(puzzle)
        self.assertTrue(valid, msg)
        empty_count = sum(1 for i in range(9) for j in range(9) if puzzle[i][j] == 0)
        self.assertGreaterEqual(empty_count, 53)
        self.assertLessEqual(empty_count, 58)

    def test_generate_invalid_difficulty(self):
        with self.assertRaises(ValueError) as context:
            generate_sudoku('invalid')
        self.assertIn("难度级别", str(context.exception))

    def test_difficulty_info(self):
        info = get_difficulty_info()
        self.assertIn('easy', info)
        self.assertIn('medium', info)
        self.assertIn('hard', info)
        self.assertIn('expert', info)
        for diff in info:
            self.assertIn('name', info[diff])
            self.assertIn('empty_cells_range', info[diff])
            self.assertIn('clues_range', info[diff])

    def test_solution_is_valid(self):
        puzzle, solution = generate_sudoku('medium', random_seed=42)
        valid, msg = validate_sudoku(solution, allow_empty=False)
        self.assertTrue(valid, msg)

    def test_generate_multiple_different(self):
        puzzle1, _ = generate_sudoku('easy', random_seed=1)
        puzzle2, _ = generate_sudoku('easy', random_seed=2)
        self.assertNotEqual(puzzle1, puzzle2)

    def test_generate_same_seed_same_result(self):
        puzzle1, sol1 = generate_sudoku('easy', random_seed=42)
        puzzle2, sol2 = generate_sudoku('easy', random_seed=42)
        self.assertEqual(puzzle1, puzzle2)
        self.assertEqual(sol1, sol2)


if __name__ == "__main__":
    unittest.main()
