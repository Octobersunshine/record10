import functools
import math
from typing import List, Tuple, NamedTuple

Point = Tuple[float, float]


class BoundingBox(NamedTuple):
    min_x: float
    min_y: float
    max_x: float
    max_y: float


def cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def dist2(a: Point, b: Point) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _same_direction(anchor: Point, a: Point, b: Point) -> bool:
    if cross(anchor, a, b) != 0:
        return False
    return (a[0] - anchor[0]) * (b[0] - anchor[0]) + (a[1] - anchor[1]) * (b[1] - anchor[1]) > 0


def _polar_cmp(anchor: Point, a: Point, b: Point) -> int:
    if a == anchor:
        return -1 if b != anchor else 0
    if b == anchor:
        return 1

    c = cross(anchor, a, b)
    if c > 0:
        return -1
    if c < 0:
        return 1

    da = dist2(anchor, a)
    db = dist2(anchor, b)
    if da < db:
        return -1
    if da > db:
        return 1
    return 0


def _all_collinear(sorted_pts: List[Point], anchor: Point) -> bool:
    if len(sorted_pts) <= 2:
        return True
    for i in range(2, len(sorted_pts)):
        if cross(anchor, sorted_pts[1], sorted_pts[i]) != 0:
            return False
    return True


def _reverse_last_direction_group(sorted_pts: List[Point], anchor: Point) -> List[Point]:
    if len(sorted_pts) <= 2:
        return list(sorted_pts)

    i = len(sorted_pts) - 2
    while i > 0 and _same_direction(anchor, sorted_pts[-1], sorted_pts[i]):
        i -= 1

    return sorted_pts[: i + 1] + sorted_pts[i + 1 :][::-1]


def graham_scan(
    points: List[Point],
    collinear: str = "keep_endpoints",
) -> List[Point]:
    if collinear not in ("keep_endpoints", "keep_all"):
        raise ValueError("collinear must be 'keep_endpoints' or 'keep_all'")

    if not points:
        return []

    points = sorted(set(points))

    if len(points) <= 1:
        return list(points)

    if len(points) == 2:
        return list(points)

    anchor = min(points, key=lambda p: (p[1], p[0]))

    sorted_pts = sorted(
        points,
        key=functools.cmp_to_key(lambda a, b: _polar_cmp(anchor, a, b)),
    )

    if _all_collinear(sorted_pts, anchor):
        if collinear == "keep_endpoints":
            return [sorted_pts[0], sorted_pts[-1]]
        return list(sorted_pts)

    if collinear == "keep_endpoints":
        return _scan_strict(sorted_pts)
    return _scan_relaxed(sorted_pts, anchor)


def _scan_strict(sorted_pts: List[Point]) -> List[Point]:
    stack: List[Point] = [sorted_pts[0], sorted_pts[1]]
    for i in range(2, len(sorted_pts)):
        while len(stack) >= 2 and cross(stack[-2], stack[-1], sorted_pts[i]) <= 0:
            stack.pop()
        stack.append(sorted_pts[i])
    return stack


def _scan_relaxed(sorted_pts: List[Point], anchor: Point) -> List[Point]:
    sorted_pts = _reverse_last_direction_group(sorted_pts, anchor)

    stack: List[Point] = [sorted_pts[0], sorted_pts[1]]
    for i in range(2, len(sorted_pts)):
        while len(stack) >= 2 and cross(stack[-2], stack[-1], sorted_pts[i]) < 0:
            stack.pop()
        stack.append(sorted_pts[i])
    return stack


def andrew_monotone_chain(
    points: List[Point],
    collinear: str = "keep_endpoints",
) -> List[Point]:
    if collinear not in ("keep_endpoints", "keep_all"):
        raise ValueError("collinear must be 'keep_endpoints' or 'keep_all'")

    if not points:
        return []

    pts = sorted(set(points))

    if len(pts) <= 1:
        return list(pts)

    if len(pts) == 2:
        return list(pts)

    strict = collinear == "keep_endpoints"

    lower: List[Point] = []
    for p in pts:
        if strict:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
        else:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) < 0:
                lower.pop()
        lower.append(p)

    upper: List[Point] = []
    for p in reversed(pts):
        if strict:
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
        else:
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) < 0:
                upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]

    if len(hull) < 3 and strict:
        if len(pts) >= 2:
            return [pts[0], pts[-1]]
        return list(pts)

    return hull


def hull_area(hull: List[Point]) -> float:
    if len(hull) < 3:
        return 0.0

    n = len(hull)
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += hull[i][0] * hull[j][1]
        s -= hull[j][0] * hull[i][1]
    return abs(s) / 2.0


def hull_perimeter(hull: List[Point]) -> float:
    if len(hull) < 2:
        return 0.0

    if len(hull) == 2:
        return math.sqrt(dist2(hull[0], hull[1]))

    n = len(hull)
    p = 0.0
    for i in range(n):
        j = (i + 1) % n
        p += math.sqrt(dist2(hull[i], hull[j]))
    return p


def hull_bounding_box(hull: List[Point]) -> BoundingBox:
    if not hull:
        raise ValueError("hull must not be empty")

    xs = [p[0] for p in hull]
    ys = [p[1] for p in hull]
    return BoundingBox(min(xs), min(ys), max(xs), max(ys))


def main():
    pts: List[Point] = [
        (0, 0), (1, 0), (2, 0),
        (2, 2), (1, 1), (0, 2),
        (1, 2), (1, -1),
    ]

    print("=" * 60)
    print("Points:", pts)
    print("=" * 60)
    print()

    print("--- Graham Scan ---")
    g_ep = graham_scan(pts, "keep_endpoints")
    g_all = graham_scan(pts, "keep_all")
    print(f"  keep_endpoints: {g_ep}")
    print(f"  keep_all:       {g_all}")
    print()

    print("--- Andrew Monotone Chain ---")
    a_ep = andrew_monotone_chain(pts, "keep_endpoints")
    a_all = andrew_monotone_chain(pts, "keep_all")
    print(f"  keep_endpoints: {a_ep}")
    print(f"  keep_all:       {a_all}")
    print()

    print("--- Metrics (using keep_endpoints hull) ---")
    for name, hull in [("Graham", g_ep), ("Andrew", a_ep)]:
        area = hull_area(hull)
        peri = hull_perimeter(hull)
        bb = hull_bounding_box(hull)
        print(f"  {name}:")
        print(f"    Area      = {area}")
        print(f"    Perimeter = {peri}")
        print(f"    BBox      = ({bb.min_x}, {bb.min_y}) ~ ({bb.max_x}, {bb.max_y})")
    print()

    print("=" * 60)
    print("Degenerate cases")
    print("=" * 60)
    print()

    collinear_pts: List[Point] = [(0, 0), (1, 1), (2, 2), (3, 3)]
    print(f"All collinear: {collinear_pts}")
    for name, algo in [("Graham", graham_scan), ("Andrew", andrew_monotone_chain)]:
        ep = algo(collinear_pts, "keep_endpoints")
        al = algo(collinear_pts, "keep_all")
        print(f"  {name} keep_endpoints: {ep}  area={hull_area(ep)}  peri={hull_perimeter(ep)}")
        print(f"  {name} keep_all:       {al}  area={hull_area(al)}  peri={hull_perimeter(al)}")
    print()

    single: List[Point] = [(5, 5)]
    print(f"Single point: {single}")
    for name, algo in [("Graham", graham_scan), ("Andrew", andrew_monotone_chain)]:
        h = algo(single, "keep_endpoints")
        print(f"  {name}: {h}  area={hull_area(h)}  peri={hull_perimeter(h)}")
    print()

    two: List[Point] = [(0, 0), (3, 4)]
    print(f"Two points: {two}")
    for name, algo in [("Graham", graham_scan), ("Andrew", andrew_monotone_chain)]:
        h = algo(two, "keep_endpoints")
        print(f"  {name}: {h}  area={hull_area(h)}  peri={hull_perimeter(h)}")
    print()

    square: List[Point] = [(0, 0), (1, 0), (1, 1), (0, 1)]
    print(f"Square: {square}")
    for name, algo in [("Graham", graham_scan), ("Andrew", andrew_monotone_chain)]:
        h = algo(square, "keep_endpoints")
        print(f"  {name}: {h}  area={hull_area(h)}  peri={hull_perimeter(h)}  bbox={hull_bounding_box(h)}")


if __name__ == "__main__":
    main()
