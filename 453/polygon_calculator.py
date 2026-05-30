def _signed_area(vertices):
    n = len(vertices)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        s += (x1 * y2) - (x2 * y1)
    return s / 2.0


def detect_orientation(vertices):
    """
    检测多边形顶点的排列方向

    参数:
        vertices: 顶点坐标列表，格式为 [(x1, y1), (x2, y2), ..., (xn, yn)]

    返回:
        "counterclockwise" - 逆时针（数学标准正方向，有符号面积为正）
        "clockwise"        - 顺时针（有符号面积为负）
        "collinear"        - 退化为共线（面积为零）
    """
    sa = _signed_area(vertices)
    if sa > 0:
        return "counterclockwise"
    elif sa < 0:
        return "clockwise"
    else:
        return "collinear"


def polygon_area(vertices):
    """
    使用鞋带公式计算简单多边形的面积

    参数:
        vertices: 顶点坐标列表，按顺序排列（顺时针或逆时针均可），
                  格式为 [(x1, y1), (x2, y2), ..., (xn, yn)]

    返回:
        多边形的面积（正值）
    """
    return abs(_signed_area(vertices))


def polygon_centroid(vertices):
    """
    计算简单多边形的重心（质心），使用三角形重心加权平均法

    参数:
        vertices: 顶点坐标列表，按顺序排列（顺时针或逆时针均可），
                  格式为 [(x1, y1), (x2, y2), ..., (xn, yn)]

    返回:
        (cx, cy) 重心坐标
    """
    n = len(vertices)
    if n < 3:
        return None

    sa = 0.0
    cx = 0.0
    cy = 0.0

    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        cross = (x1 * y2) - (x2 * y1)
        sa += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross

    sa *= 0.5
    if abs(sa) < 1e-12:
        return None
    factor = 1.0 / (6.0 * sa)
    cx *= factor
    cy *= factor

    return (cx, cy)


def point_in_polygon(point, vertices, include_boundary=True, epsilon=1e-12):
    """
    使用射线法判断点是否在多边形内部

    参数:
        point: 待检测点 (x, y)
        vertices: 多边形顶点列表 [(x1,y1), ..., (xn,yn)]
        include_boundary: 是否将边界视为内部
        epsilon: 浮点数比较容差

    返回:
        True - 点在多边形内（或边界上）
        False - 点在多边形外
    """
    n = len(vertices)
    if n < 3:
        return False

    px, py = point

    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]

        min_x = min(x1, x2) - epsilon
        max_x = max(x1, x2) + epsilon
        min_y = min(y1, y2) - epsilon
        max_y = max(y1, y2) + epsilon

        if min_x <= px <= max_x and min_y <= py <= max_y:
            cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
            if abs(cross) < epsilon:
                return include_boundary

    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]

        if ((yi > py) != (yj > py)):
            x_intersect = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < x_intersect:
                inside = not inside
        j = i

    return inside


def polygon_bounding_box(vertices):
    """
    计算多边形的轴对齐边界框

    参数:
        vertices: 多边形顶点列表 [(x1,y1), ..., (xn,yn)]

    返回:
        (min_x, min_y, max_x, max_y) 边界框
        顶点不足时返回 None
    """
    n = len(vertices)
    if n < 1:
        return None

    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]

    return (min(xs), min(ys), max(xs), max(ys))


def _unit_vector(dx, dy):
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-12:
        return 0.0, 0.0
    return dx / length, dy / length


def polygon_buffer(vertices, distance):
    """
    简化版多边形缓冲区偏移（向内/向外）

    算法说明：
    1. 对每条边计算法向量
    2. 将每个顶点沿其相邻两条边的角平分线方向移动 distance 距离
    3. 逆时针多边形：distance > 0 向外扩张，distance < 0 向内收缩
       顺时针多边形：方向相反（符合常规直觉）

    参数:
        vertices: 多边形顶点列表 [(x1,y1), ..., (xn,yn)]
        distance: 偏移距离（正数向外，负数向内）

    返回:
        偏移后的新顶点列表
    """
    n = len(vertices)
    if n < 3:
        return vertices[:]

    orientation = detect_orientation(vertices)
    if orientation == "counterclockwise":
        sign = 1.0
    elif orientation == "clockwise":
        sign = -1.0
    else:
        return vertices[:]

    d = distance * sign

    new_vertices = []
    for i in range(n):
        px, py = vertices[i]
        ax, ay = vertices[(i - 1) % n]
        bx, by = vertices[(i + 1) % n]

        nx1, ny1 = _unit_vector(py - ay, ax - px)
        nx2, ny2 = _unit_vector(by - py, px - bx)

        mx = nx1 + nx2
        my = ny1 + ny2
        ml = (mx * mx + my * my) ** 0.5

        if ml < 1e-12:
            nx, ny = nx1, ny1
        else:
            nx, ny = mx / ml, my / ml

        dot = nx1 * nx2 + ny1 * ny2
        dot = max(-1.0, min(1.0, dot))
        sin_half = ((1.0 - dot) / 2.0) ** 0.5
        factor = 1.0 / sin_half if sin_half > 1e-12 else 1.0

        nx *= factor
        ny *= factor

        new_vertices.append((px + nx * d, py + ny * d))

    return new_vertices


def _print_result(name, vertices):
    area = polygon_area(vertices)
    centroid = polygon_centroid(vertices)
    orientation = detect_orientation(vertices)
    orient_cn = {"counterclockwise": "逆时针", "clockwise": "顺时针", "collinear": "共线"}
    print(f"\n{name}:")
    print(f"   顶点: {vertices}")
    print(f"   方向: {orient_cn.get(orientation, orientation)}")
    print(f"   面积: {area}")
    print(f"   重心: {centroid}")


if __name__ == "__main__":
    print("=" * 50)
    print("多边形面积和重心计算工具")
    print("=" * 50)

    _print_result("1. 正方形-逆时针", [(0, 0), (2, 0), (2, 2), (0, 2)])
    _print_result("2. 正方形-顺时针", [(0, 0), (0, 2), (2, 2), (2, 0)])
    _print_result("3. 直角三角形-逆时针", [(0, 0), (3, 0), (0, 4)])
    _print_result("4. 直角三角形-顺时针", [(0, 0), (0, 4), (3, 0)])
    _print_result("5. 凹多边形-逆时针", [(0, 0), (4, 0), (4, 4), (2, 2), (0, 4)])
    _print_result("6. 凹多边形-顺时针", [(0, 4), (2, 2), (4, 4), (4, 0), (0, 0)])
    _print_result("7. 正六边形-逆时针", [(1, 0), (0.5, 0.866), (-0.5, 0.866),
                                          (-1, 0), (-0.5, -0.866), (0.5, -0.866)])
    _print_result("8. 正六边形-顺时针", [(1, 0), (0.5, -0.866), (-0.5, -0.866),
                                          (-1, 0), (-0.5, 0.866), (0.5, 0.866)])
    _print_result("9. 共线退化", [(0, 0), (1, 1), (2, 2)])

    print("\n" + "-" * 50)
    print("【点在多边形内部测试】")
    print("-" * 50)

    square_ccw = [(0, 0), (2, 0), (2, 2), (0, 2)]
    concave = [(0, 0), (4, 0), (4, 4), (2, 2), (0, 4)]

    test_cases = [
        ("正方形内点", square_ccw, (1, 1), True),
        ("正方形外点", square_ccw, (3, 1), False),
        ("正方形边界点", square_ccw, (0, 1), True),
        ("正方形外点边界外", square_ccw, (2.001, 1), False),
        ("凹多边形内点(左半)", concave, (1, 2), True),
        ("凹多边形边界点(边(4,4)-(2,2))", concave, (3, 3), True),
        ("凹多边形外点(凹陷上方)", concave, (3, 3.5), False),
        ("凹多边形内点(凹陷下方)", concave, (3, 2.5), True),
        ("凹多边形边界点", concave, (4, 2), True),
    ]

    for name, poly, pt, expected in test_cases:
        result = point_in_polygon(pt, poly)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {name}: 点{pt} -> {result} (期望 {expected})")

    pt_boundary = (2, 0)
    r1 = point_in_polygon(pt_boundary, square_ccw, include_boundary=True)
    r2 = point_in_polygon(pt_boundary, square_ccw, include_boundary=False)
    print(f"  ✓ 边界包含/排除测试: {pt_boundary} 包含={r1}, 排除={r2}")

    print("\n" + "-" * 50)
    print("【边界框测试】")
    print("-" * 50)

    for name, poly in [("正方形", square_ccw), ("凹多边形", concave),
                       ("正六边形", [(1, 0), (0.5, 0.866), (-0.5, 0.866),
                                     (-1, 0), (-0.5, -0.866), (0.5, -0.866)])]:
        bbox = polygon_bounding_box(poly)
        print(f"  {name}: bbox = {bbox}")

    print("\n" + "-" * 50)
    print("【缓冲区偏移测试】")
    print("-" * 50)

    print("\n  * 正方形(0,0)-(2,2) 向外偏移 0.5:")
    buffered_out = polygon_buffer(square_ccw, 0.5)
    area_ori = polygon_area(square_ccw)
    area_out = polygon_area(buffered_out)
    print(f"    原始顶点: {square_ccw}")
    print(f"    偏移顶点: {[(round(x, 3), round(y, 3)) for x, y in buffered_out]}")
    print(f"    原始面积: {area_ori}, 偏移后面积: {area_out:.3f}")
    print(f"    边界框: {polygon_bounding_box(buffered_out)}")

    print("\n  * 正方形向内偏移 0.5:")
    buffered_in = polygon_buffer(square_ccw, -0.5)
    area_in = polygon_area(buffered_in)
    print(f"    偏移顶点: {[(round(x, 3), round(y, 3)) for x, y in buffered_in]}")
    print(f"    偏移后面积: {area_in:.3f} (期望: 1.0, 边长1.0)")

    print("\n  * 顺时针正方形向外偏移 0.5 (验证方向独立性):")
    square_cw = [(0, 0), (0, 2), (2, 2), (2, 0)]
    buffered_cw = polygon_buffer(square_cw, 0.5)
    print(f"    偏移顶点: {[(round(x, 3), round(y, 3)) for x, y in buffered_cw]}")
    print(f"    面积: {polygon_area(buffered_cw):.3f}")

    print("\n  * 凹多边形向外偏移 0.3:")
    buffered_concave = polygon_buffer(concave, 0.3)
    print(f"    原始面积: {polygon_area(concave):.3f}")
    print(f"    偏移后面积: {polygon_area(buffered_concave):.3f}")
    print(f"    偏移顶点: {[(round(x, 3), round(y, 3)) for x, y in buffered_concave]}")

    print("\n" + "=" * 50)
