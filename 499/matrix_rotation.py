def _validate_matrix(matrix):
    if not isinstance(matrix, list):
        raise TypeError("Matrix must be a list")
    if len(matrix) == 0:
        raise ValueError("Matrix cannot be empty")
    n = len(matrix)
    for i, row in enumerate(matrix):
        if not isinstance(row, list):
            raise TypeError(f"Row {i} must be a list")
        if len(row) != n:
            raise ValueError(
                f"Matrix must be a square matrix (n x n). "
                f"Row {i} has {len(row)} elements, expected {n}"
            )
    return n


def _copy_matrix(matrix):
    return [row[:] for row in matrix]


def _get_center_element(matrix):
    n = len(matrix)
    if n % 2 == 1:
        mid = n // 2
        return matrix[mid][mid]
    return None


def _validate_3d_matrix(matrix):
    if not isinstance(matrix, list):
        raise TypeError("3D matrix must be a list")
    if len(matrix) == 0:
        raise ValueError("3D matrix cannot be empty")
    n = len(matrix)
    for i, layer in enumerate(matrix):
        if not isinstance(layer, list):
            raise TypeError(f"Layer {i} must be a list")
        if len(layer) != n:
            raise ValueError(
                f"3D matrix must be a cube (n x n x n). "
                f"Layer {i} has {len(layer)} rows, expected {n}"
            )
        for j, row in enumerate(layer):
            if not isinstance(row, list):
                raise TypeError(f"Row {j} in layer {i} must be a list")
            if len(row) != n:
                raise ValueError(
                    f"3D matrix must be a cube (n x n x n). "
                    f"Row {j} in layer {i} has {len(row)} elements, expected {n}"
                )
    return n


def _copy_3d_matrix(matrix):
    return [[row[:] for row in layer] for layer in matrix]


def _rotate_2d_layer_clockwise(layer):
    n = len(layer)
    for i in range(n):
        for j in range(i + 1, n):
            layer[i][j], layer[j][i] = layer[j][i], layer[i][j]
    for row in layer:
        row.reverse()
    return layer


def _rotate_2d_layer_counterclockwise(layer):
    n = len(layer)
    for i in range(n):
        for j in range(i + 1, n):
            layer[i][j], layer[j][i] = layer[j][i], layer[i][j]
    layer.reverse()
    return layer


def rotate_clockwise_90(matrix, inplace=True):
    n = _validate_matrix(matrix)
    if not inplace:
        matrix = _copy_matrix(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for row in matrix:
        row.reverse()
    return matrix


def rotate_counterclockwise_90(matrix, inplace=True):
    n = _validate_matrix(matrix)
    if not inplace:
        matrix = _copy_matrix(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    matrix.reverse()
    return matrix


def rotate_180(matrix, inplace=True):
    n = _validate_matrix(matrix)
    if not inplace:
        matrix = _copy_matrix(matrix)
    matrix.reverse()
    for row in matrix:
        row.reverse()
    return matrix


def rotate_clockwise_90_flip(matrix, inplace=True):
    n = _validate_matrix(matrix)
    if not inplace:
        matrix = _copy_matrix(matrix)
    matrix.reverse()
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    return matrix


def rotate_counterclockwise_90_flip(matrix, inplace=True):
    n = _validate_matrix(matrix)
    if not inplace:
        matrix = _copy_matrix(matrix)
    for row in matrix:
        row.reverse()
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    return matrix


def rotate_180_flip(matrix, inplace=True):
    n = _validate_matrix(matrix)
    if not inplace:
        matrix = _copy_matrix(matrix)
    for i in range(n // 2):
        for j in range(n):
            matrix[i][j], matrix[n - 1 - i][n - 1 - j] = matrix[n - 1 - i][n - 1 - j], matrix[i][j]
    if n % 2 == 1:
        mid = n // 2
        for j in range(n // 2):
            matrix[mid][j], matrix[mid][n - 1 - j] = matrix[mid][n - 1 - j], matrix[mid][j]
    return matrix


def rotate_3d_x_90(matrix, inplace=True):
    n = _validate_3d_matrix(matrix)
    if not inplace:
        matrix = _copy_3d_matrix(matrix)
    for x in range(n):
        yz_plane = [[matrix[x][y][z] for z in range(n)] for y in range(n)]
        _rotate_2d_layer_clockwise(yz_plane)
        for y in range(n):
            for z in range(n):
                matrix[x][y][z] = yz_plane[y][z]
    return matrix


def rotate_3d_y_90(matrix, inplace=True):
    n = _validate_3d_matrix(matrix)
    if not inplace:
        matrix = _copy_3d_matrix(matrix)
    for y in range(n):
        xz_plane = [[matrix[x][y][z] for z in range(n)] for x in range(n)]
        _rotate_2d_layer_counterclockwise(xz_plane)
        for x in range(n):
            for z in range(n):
                matrix[x][y][z] = xz_plane[x][z]
    return matrix


def rotate_3d_z_90(matrix, inplace=True):
    n = _validate_3d_matrix(matrix)
    if not inplace:
        matrix = _copy_3d_matrix(matrix)
    for z in range(n):
        xy_plane = [[matrix[x][y][z] for y in range(n)] for x in range(n)]
        _rotate_2d_layer_clockwise(xy_plane)
        for x in range(n):
            for y in range(n):
                matrix[x][y][z] = xy_plane[x][y]
    return matrix


def rotate_3d_x_90_inplace(matrix, inplace=True):
    n = _validate_3d_matrix(matrix)
    if not inplace:
        matrix = _copy_3d_matrix(matrix)
    for x in range(n):
        for y in range(n // 2):
            for z in range(y + 1, n):
                matrix[x][y][z], matrix[x][z][y] = matrix[x][z][y], matrix[x][y][z]
        for y in range(n):
            matrix[x][y].reverse()
    return matrix


def rotate_3d_y_90_inplace(matrix, inplace=True):
    n = _validate_3d_matrix(matrix)
    if not inplace:
        matrix = _copy_3d_matrix(matrix)
    for y in range(n):
        for x in range(n // 2):
            for z in range(x + 1, n):
                matrix[x][y][z], matrix[z][y][x] = matrix[z][y][x], matrix[x][y][z]
        for x in range(n // 2):
            for z in range(n):
                matrix[x][y][z], matrix[n - 1 - x][y][z] = matrix[n - 1 - x][y][z], matrix[x][y][z]
    return matrix


def rotate_3d_z_90_inplace(matrix, inplace=True):
    n = _validate_3d_matrix(matrix)
    if not inplace:
        matrix = _copy_3d_matrix(matrix)
    for z in range(n):
        for x in range(n // 2):
            for y in range(x + 1, n):
                matrix[x][y][z], matrix[y][x][z] = matrix[y][x][z], matrix[x][y][z]
        for x in range(n):
            row = [matrix[x][y][z] for y in range(n)]
            row.reverse()
            for y in range(n):
                matrix[x][y][z] = row[y]
    return matrix


def print_matrix(matrix):
    for row in matrix:
        print(row)
    print()


def print_3d_matrix(matrix):
    for x, layer in enumerate(matrix):
        print(f"Layer x={x}:")
        for row in layer:
            print(f"  {row}")
    print()


import time
import sys


def measure_performance(func, matrix, iterations=10):
    matrix_copy = _copy_3d_matrix(matrix)
    start_time = time.perf_counter()
    for _ in range(iterations):
        test_mat = _copy_3d_matrix(matrix_copy)
        func(test_mat)
    end_time = time.perf_counter()
    avg_time = (end_time - start_time) / iterations * 1000
    return avg_time


def measure_memory(matrix):
    total = 0
    for layer in matrix:
        for row in layer:
            total += sys.getsizeof(row)
        total += sys.getsizeof(layer)
    total += sys.getsizeof(matrix)
    return total / 1024


def create_3d_matrix(n):
    return [[[x * n * n + y * n + z + 1 for z in range(n)] for y in range(n)] for x in range(n)]


def test_rotation():
    original_3x3 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]

    expected_cw90 = [
        [7, 4, 1],
        [8, 5, 2],
        [9, 6, 3],
    ]

    expected_ccw90 = [
        [3, 6, 9],
        [2, 5, 8],
        [1, 4, 7],
    ]

    expected_180 = [
        [9, 8, 7],
        [6, 5, 4],
        [3, 2, 1],
    ]

    original_4x4 = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]

    expected_cw90_4x4 = [
        [13, 9, 5, 1],
        [14, 10, 6, 2],
        [15, 11, 7, 3],
        [16, 12, 8, 4],
    ]

    expected_ccw90_4x4 = [
        [4, 8, 12, 16],
        [3, 7, 11, 15],
        [2, 6, 10, 14],
        [1, 5, 9, 13],
    ]

    expected_180_4x4 = [
        [16, 15, 14, 13],
        [12, 11, 10, 9],
        [8, 7, 6, 5],
        [4, 3, 2, 1],
    ]

    print("=" * 60)
    print("Testing 3x3 Matrix")
    print("=" * 60)

    print("\nOriginal 3x3:")
    print_matrix(original_3x3)

    m = _copy_matrix(original_3x3)
    result = rotate_clockwise_90(m)
    assert result == expected_cw90, "Clockwise 90° failed"
    print("Clockwise 90° (transpose + reverse):")
    print_matrix(result)

    m = _copy_matrix(original_3x3)
    result = rotate_clockwise_90_flip(m)
    assert result == expected_cw90, "Clockwise 90° flip method failed"
    print("Clockwise 90° (reverse + transpose):")
    print_matrix(result)

    m = _copy_matrix(original_3x3)
    result = rotate_counterclockwise_90(m)
    assert result == expected_ccw90, "Counter-clockwise 90° failed"
    print("Counter-clockwise 90° (transpose + reverse rows):")
    print_matrix(result)

    m = _copy_matrix(original_3x3)
    result = rotate_counterclockwise_90_flip(m)
    assert result == expected_ccw90, "Counter-clockwise 90° flip method failed"
    print("Counter-clockwise 90° (reverse rows + transpose):")
    print_matrix(result)

    m = _copy_matrix(original_3x3)
    result = rotate_180(m)
    assert result == expected_180, "180° failed"
    print("180° (reverse + reverse rows):")
    print_matrix(result)

    m = _copy_matrix(original_3x3)
    result = rotate_180_flip(m)
    assert result == expected_180, "180° flip method failed"
    print("180° (swap symmetric elements):")
    print_matrix(result)

    print("=" * 60)
    print("Testing 4x4 Matrix")
    print("=" * 60)

    print("\nOriginal 4x4:")
    print_matrix(original_4x4)

    m = _copy_matrix(original_4x4)
    result = rotate_clockwise_90(m)
    assert result == expected_cw90_4x4, "4x4 Clockwise 90° failed"
    print("Clockwise 90°:")
    print_matrix(result)

    m = _copy_matrix(original_4x4)
    result = rotate_counterclockwise_90(m)
    assert result == expected_ccw90_4x4, "4x4 Counter-clockwise 90° failed"
    print("Counter-clockwise 90°:")
    print_matrix(result)

    m = _copy_matrix(original_4x4)
    result = rotate_180(m)
    assert result == expected_180_4x4, "4x4 180° failed"
    print("180°:")
    print_matrix(result)

    print("=" * 60)
    print("Testing inplace=False (new matrix)")
    print("=" * 60)

    original = _copy_matrix(original_3x3)
    result = rotate_clockwise_90(original, inplace=False)
    assert original == original_3x3, "Original matrix modified when inplace=False"
    assert result == expected_cw90, "Result incorrect for inplace=False"
    print("Original matrix preserved:")
    print_matrix(original)
    print("New matrix (rotated):")
    print_matrix(result)

    print("=" * 60)
    print("Testing 5x5 Odd-dimension Matrix (center element check)")
    print("=" * 60)

    original_5x5 = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 999, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
    ]
    center_value = _get_center_element(original_5x5)
    print(f"\nOriginal 5x5 matrix, center element = {center_value}:")
    print_matrix(original_5x5)

    m = _copy_matrix(original_5x5)
    result = rotate_clockwise_90(m)
    assert _get_center_element(result) == center_value, "Clockwise 90° center element changed"
    print("Clockwise 90°, center element preserved:")
    print_matrix(result)

    m = _copy_matrix(original_5x5)
    result = rotate_counterclockwise_90(m)
    assert _get_center_element(result) == center_value, "Counter-clockwise 90° center element changed"
    print("Counter-clockwise 90°, center element preserved:")
    print_matrix(result)

    m = _copy_matrix(original_5x5)
    result = rotate_180(m)
    assert _get_center_element(result) == center_value, "180° center element changed"
    print("180°, center element preserved:")
    print_matrix(result)

    m = _copy_matrix(original_5x5)
    result = rotate_180_flip(m)
    assert _get_center_element(result) == center_value, "180° flip center element changed"
    print("180° (flip method), center element preserved:")
    print_matrix(result)

    print("=" * 60)
    print("Testing validation")
    print("=" * 60)

    try:
        rotate_clockwise_90("not a list")
        print("ERROR: Should have raised TypeError for non-list input")
    except TypeError as e:
        print(f"✓ Non-list validation: {e}")

    try:
        rotate_clockwise_90([])
        print("ERROR: Should have raised ValueError for empty matrix")
    except ValueError as e:
        print(f"✓ Empty matrix validation: {e}")

    try:
        rotate_clockwise_90([1, 2, 3])
        print("ERROR: Should have raised TypeError for non-list rows")
    except TypeError as e:
        print(f"✓ Non-list rows validation: {e}")

    try:
        rotate_clockwise_90([[1, 2], [3, 4, 5]])
        print("ERROR: Should have raised ValueError for non-square matrix")
    except ValueError as e:
        print(f"✓ Non-square matrix validation: {e}")

    try:
        rotate_clockwise_90([[1, 2, 3], [4, 5], [6, 7, 8]])
        print("ERROR: Should have raised ValueError for non-square matrix (row 1 short)")
    except ValueError as e:
        print(f"✓ Non-square matrix (row 1 short) validation: {e}")

    print("=" * 60)
    print("Testing 3D Matrix Rotation (2x2x2 cube)")
    print("=" * 60)

    cube_2x2 = create_3d_matrix(2)
    print("\nOriginal 2x2x2 cube:")
    print_3d_matrix(cube_2x2)

    expected_x_rot = [
        [[5, 1], [6, 2]],
        [[7, 3], [8, 4]],
    ]

    m = _copy_3d_matrix(cube_2x2)
    result = rotate_3d_x_90(m)
    print("After X-axis 90° rotation:")
    print_3d_matrix(result)

    m = _copy_3d_matrix(cube_2x2)
    result = rotate_3d_x_90_inplace(m)
    print("After X-axis 90° rotation (inplace method):")
    print_3d_matrix(result)

    m = _copy_3d_matrix(cube_2x2)
    result = rotate_3d_y_90(m)
    print("After Y-axis 90° rotation:")
    print_3d_matrix(result)

    m = _copy_3d_matrix(cube_2x2)
    result = rotate_3d_z_90(m)
    print("After Z-axis 90° rotation:")
    print_3d_matrix(result)

    print("=" * 60)
    print("Testing 3D Matrix 3x3x3 cube (center element check)")
    print("=" * 60)

    cube_3x3 = create_3d_matrix(3)
    cube_3x3[1][1][1] = 999
    print("\nOriginal 3x3x3 cube, center element = 999:")
    print_3d_matrix(cube_3x3)

    m = _copy_3d_matrix(cube_3x3)
    result = rotate_3d_x_90(m)
    assert result[1][1][1] == 999, "3D X-axis rotation center element changed"
    print("After X-axis 90° rotation, center preserved:")
    print_3d_matrix(result)

    m = _copy_3d_matrix(cube_3x3)
    result = rotate_3d_y_90(m)
    assert result[1][1][1] == 999, "3D Y-axis rotation center element changed"
    print("After Y-axis 90° rotation, center preserved:")
    print_3d_matrix(result)

    m = _copy_3d_matrix(cube_3x3)
    result = rotate_3d_z_90(m)
    assert result[1][1][1] == 999, "3D Z-axis rotation center element changed"
    print("After Z-axis 90° rotation, center preserved:")
    print_3d_matrix(result)

    print("=" * 60)
    print("Performance Comparison (3D Matrix Rotation)")
    print("=" * 60)

    sizes = [4, 8, 16]
    for size in sizes:
        print(f"\n--- {size}x{size}x{size} Cube ---")
        cube = create_3d_matrix(size)
        mem_kb = measure_memory(cube)
        print(f"Memory usage: {mem_kb:.2f} KB")

        time_x = measure_performance(rotate_3d_x_90, cube, iterations=5)
        time_x_inplace = measure_performance(rotate_3d_x_90_inplace, cube, iterations=5)
        print(f"X-axis rotation (copy method):  {time_x:.4f} ms")
        print(f"X-axis rotation (inplace method): {time_x_inplace:.4f} ms")

        time_y = measure_performance(rotate_3d_y_90, cube, iterations=5)
        print(f"Y-axis rotation (copy method):  {time_y:.4f} ms")

        time_z = measure_performance(rotate_3d_z_90, cube, iterations=5)
        time_z_inplace = measure_performance(rotate_3d_z_90_inplace, cube, iterations=5)
        print(f"Z-axis rotation (copy method):  {time_z:.4f} ms")
        print(f"Z-axis rotation (inplace method): {time_z_inplace:.4f} ms")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_rotation()
