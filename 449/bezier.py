from __future__ import annotations

import math
from typing import List, Tuple


Point = Tuple[float, float]
WeightedPoint = Tuple[float, float, float]
BoundingBox = Tuple[Point, Point]


def _validate_points(points: List[Point], min_count: int = 2) -> None:
    if len(points) < min_count:
        raise ValueError(f"控制点至少需要{min_count}个，当前只有{len(points)}个")
    if len(points) > 4:
        raise ValueError(f"仅支持一次(2点)、二次(3点)和三次(4点)贝塞尔曲线，当前提供了{len(points)}个控制点")


def _validate_t(t: float) -> None:
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"参数t必须在[0, 1]范围内，当前值为{t}")


def de_casteljau(points: List[Point], t: float) -> Point:
    _validate_points(points)
    _validate_t(t)

    prev = list(points)
    for _ in range(len(prev) - 1):
        curr = []
        for i in range(len(prev) - 1):
            x = (1 - t) * prev[i][0] + t * prev[i + 1][0]
            y = (1 - t) * prev[i][1] + t * prev[i + 1][1]
            curr.append((x, y))
        prev = curr
    return prev[0]


def bezier_curve(points: List[Point], steps: int) -> List[Point]:
    if steps < 1:
        raise ValueError(f"步数必须 >= 1，当前值为{steps}")
    _validate_points(points)

    curve: List[Point] = []
    for i in range(steps + 1):
        t = i / steps
        curve.append(de_casteljau(points, t))
    return curve


def rational_de_casteljau(wpoints: List[WeightedPoint], t: float) -> Point:
    if len(wpoints) < 2:
        raise ValueError(f"控制点至少需要2个，当前只有{len(wpoints)}个")
    if len(wpoints) > 4:
        raise ValueError(f"仅支持一次(2点)、二次(3点)和三次(4点)有理贝塞尔曲线，当前提供了{len(wpoints)}个控制点")
    _validate_t(t)

    for idx, wp in enumerate(wpoints):
        if wp[2] <= 0:
            raise ValueError(f"权重必须为正数，第{idx}个控制点的权重为{wp[2]}")

    prev: List[Tuple[float, float, float]] = [(wp[0] * wp[2], wp[1] * wp[2], wp[2]) for wp in wpoints]

    for _ in range(len(prev) - 1):
        curr: List[Tuple[float, float, float]] = []
        for i in range(len(prev) - 1):
            wx = (1 - t) * prev[i][0] + t * prev[i + 1][0]
            wy = (1 - t) * prev[i][1] + t * prev[i + 1][1]
            w = (1 - t) * prev[i][2] + t * prev[i + 1][2]
            curr.append((wx, wy, w))
        prev = curr

    return (prev[0][0] / prev[0][2], prev[0][1] / prev[0][2])


def rational_bezier_curve(wpoints: List[WeightedPoint], steps: int) -> List[Point]:
    if steps < 1:
        raise ValueError(f"步数必须 >= 1，当前值为{steps}")
    if len(wpoints) < 2:
        raise ValueError(f"控制点至少需要2个，当前只有{len(wpoints)}个")
    if len(wpoints) > 4:
        raise ValueError(f"仅支持一次(2点)、二次(3点)和三次(4点)有理贝塞尔曲线，当前提供了{len(wpoints)}个控制点")

    curve: List[Point] = []
    for i in range(steps + 1):
        t = i / steps
        curve.append(rational_de_casteljau(wpoints, t))
    return curve


def subdivide_bezier(points: List[Point], t: float) -> Tuple[List[Point], List[Point]]:
    _validate_points(points)
    _validate_t(t)

    n = len(points)
    levels: List[List[Point]] = [list(points)]

    prev = list(points)
    for _ in range(n - 1):
        curr = []
        for i in range(len(prev) - 1):
            x = (1 - t) * prev[i][0] + t * prev[i + 1][0]
            y = (1 - t) * prev[i][1] + t * prev[i + 1][1]
            curr.append((x, y))
        levels.append(curr)
        prev = curr

    left = [level[0] for level in levels]
    right = [level[-1] for level in reversed(levels)]

    return left, right


def bounding_box(curve_points: List[Point]) -> BoundingBox:
    if not curve_points:
        raise ValueError("曲线点列表不能为空")

    min_x = min(p[0] for p in curve_points)
    min_y = min(p[1] for p in curve_points)
    max_x = max(p[0] for p in curve_points)
    max_y = max(p[1] for p in curve_points)
    return ((min_x, min_y), (max_x, max_y))


def arc_length(curve_points: List[Point]) -> float:
    if not curve_points:
        raise ValueError("曲线点列表不能为空")

    total = 0.0
    for i in range(len(curve_points) - 1):
        dx = curve_points[i + 1][0] - curve_points[i][0]
        dy = curve_points[i + 1][1] - curve_points[i][1]
        total += math.sqrt(dx * dx + dy * dy)
    return total


if __name__ == "__main__":
    print("=== 一次贝塞尔曲线 (线性插值) ===")
    linear_pts = [(0.0, 0.0), (10.0, 10.0)]
    for p in bezier_curve(linear_pts, 5):
        print(f"  ({p[0]:.2f}, {p[1]:.2f})")

    print("\n=== 二次贝塞尔曲线 ===")
    quad_pts = [(0.0, 0.0), (5.0, 10.0), (10.0, 0.0)]
    for p in bezier_curve(quad_pts, 10):
        print(f"  ({p[0]:.2f}, {p[1]:.2f})")

    print("\n=== 三次贝塞尔曲线 ===")
    cubic_pts = [(0.0, 0.0), (2.0, 8.0), (8.0, 8.0), (10.0, 0.0)]
    curve = bezier_curve(cubic_pts, 20)
    for p in curve:
        print(f"  ({p[0]:.2f}, {p[1]:.2f})")

    print("\n=== 有理贝塞尔曲线 (圆弧近似) ===")
    wpts = [(0.0, 0.0, 1.0), (5.0, 10.0, 2.0), (10.0, 0.0, 1.0)]
    for p in rational_bezier_curve(wpts, 10):
        print(f"  ({p[0]:.2f}, {p[1]:.2f})")

    print("\n=== 曲线分割 (t=0.5) ===")
    left, right = subdivide_bezier(cubic_pts, 0.5)
    print(f"  左半控制点: {['({:.2f}, {:.2f})'.format(*p) for p in left]}")
    print(f"  右半控制点: {['({:.2f}, {:.2f})'.format(*p) for p in right]}")

    print("\n=== 包围盒 ===")
    bb = bounding_box(curve)
    print(f"  最小角: ({bb[0][0]:.2f}, {bb[0][1]:.2f})")
    print(f"  最大角: ({bb[1][0]:.2f}, {bb[1][1]:.2f})")

    print("\n=== 弧长近似 ===")
    length = arc_length(curve)
    print(f"  弧长 ≈ {length:.4f}")
