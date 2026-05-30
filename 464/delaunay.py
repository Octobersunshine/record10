import math
import random
from typing import List, Tuple, Optional, Dict, Set

EPS = 1e-10
PERTURBATION = 1e-7
INF = 1e10


def _circumcircle(points, i, j, k):
    ax, ay = points[i]
    bx, by = points[j]
    cx, cy = points[k]
    D = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < EPS:
        return None
    A2 = ax * ax + ay * ay
    B2 = bx * bx + by * by
    C2 = cx * cx + cy * cy
    ux = (A2 * (by - cy) + B2 * (cy - ay) + C2 * (ay - by)) / D
    uy = (A2 * (cx - bx) + B2 * (ax - cx) + C2 * (bx - ax)) / D
    r_sq = (ax - ux) ** 2 + (ay - uy) ** 2
    return ux, uy, r_sq


def _in_circumcircle(points, v0, v1, v2, p):
    cc = _circumcircle(points, v0, v1, v2)
    if cc is None:
        return False
    ux, uy, r_sq = cc
    px, py = points[p]
    return (px - ux) ** 2 + (py - uy) ** 2 < r_sq - EPS


def _on_circumcircle(points, v0, v1, v2, p):
    cc = _circumcircle(points, v0, v1, v2)
    if cc is None:
        return False
    ux, uy, r_sq = cc
    px, py = points[p]
    dist_sq = (px - ux) ** 2 + (py - uy) ** 2
    return abs(dist_sq - r_sq) < EPS * 100


def _make_counter_clockwise(points, v0, v1, v2):
    ax, ay = points[v0]
    bx, by = points[v1]
    cx, cy = points[v2]
    cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if cross < 0:
        return (v2, v1, v0)
    return (v0, v1, v2)


def _triangle_key(v0, v1, v2):
    return tuple(sorted((v0, v1, v2)))


def _edge_key(a, b):
    return (min(a, b), max(a, b))


def _is_degenerate_triangle(tri):
    return len(set(tri)) < 3


def _perturb_points(points: List[Tuple[float, float]], seed: int = 42) -> List[Tuple[float, float]]:
    rng = random.Random(seed)
    result = []
    for x, y in points:
        dx = rng.uniform(-PERTURBATION, PERTURBATION)
        dy = rng.uniform(-PERTURBATION, PERTURBATION)
        result.append((x + dx, y + dy))
    return result


def _polygon_area(polygon: List[Tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    area = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _clip_polygon_to_bbox(polygon: List[Tuple[float, float]],
                         bbox: Tuple[float, float, float, float]) -> List[Tuple[float, float]]:
    min_x, min_y, max_x, max_y = bbox

    def _clip_edge(pts, edge):
        if not pts:
            return []
        result = []
        n = len(pts)
        for i in range(n):
            curr = pts[i]
            prev = pts[(i - 1) % n]
            curr_inside = edge(curr)
            prev_inside = edge(prev)
            if curr_inside:
                if not prev_inside:
                    intersect = _intersect(prev, curr, edge)
                    if intersect is not None:
                        result.append(intersect)
                result.append(curr)
            elif prev_inside:
                intersect = _intersect(prev, curr, edge)
                if intersect is not None:
                    result.append(intersect)
        return result

    def _intersect(p1, p2, edge):
        (x1, y1), (x2, y2) = p1, p2
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) < EPS and abs(dy) < EPS:
            return None
        t = 0.0
        if edge == _left:
            t = (min_x - x1) / dx if abs(dx) > EPS else 1e10
        elif edge == _right:
            t = (max_x - x1) / dx if abs(dx) > EPS else 1e10
        elif edge == _bottom:
            t = (min_y - y1) / dy if abs(dy) > EPS else 1e10
        elif edge == _top:
            t = (max_y - y1) / dy if abs(dy) > EPS else 1e10
        if 0 <= t <= 1:
            return (x1 + t * dx, y1 + t * dy)
        return None

    def _left(p):
        return p[0] >= min_x

    def _right(p):
        return p[0] <= max_x

    def _bottom(p):
        return p[1] >= min_y

    def _top(p):
        return p[1] <= max_y

    result = list(polygon)
    for edge in [_left, _right, _bottom, _top]:
        result = _clip_edge(result, edge)
        if not result:
            break
    return result


class DelaunayTriangulation:
    def __init__(self, points: List[Tuple[float, float]], perturb: bool = True):
        self._original_points = list(points)
        self._perturb = perturb
        self.points = list(points)
        self.triangles: List[Tuple[int, int, int]] = []
        self._super_indices: List[int] = []
        self._built = False
        if len(self.points) >= 3:
            self._build()

    def _build(self):
        pts = self.points
        n = len(self._original_points)

        if n < 3:
            return

        if self._perturb:
            self._apply_perturbation_if_needed()
            pts = self.points

        xs = [p[0] for p in self._original_points]
        ys = [p[1] for p in self._original_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        dx = max_x - min_x
        dy = max_y - min_y
        delta = max(dx, dy, 1e-6)
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0

        super_scale = 100.0
        s0, s1, s2 = n, n + 1, n + 2
        self._super_indices = [s0, s1, s2]

        while len(pts) < n + 3:
            pts.append((0.0, 0.0))
        pts[s0] = (cx - super_scale * delta, cy - super_scale * delta)
        pts[s1] = (cx + super_scale * delta, cy - super_scale * delta)
        pts[s2] = (cx, cy + super_scale * delta)

        self.triangles = [_make_counter_clockwise(pts, s0, s1, s2)]

        for p_idx in range(n):
            self._insert_point(p_idx)

        self._remove_super()
        self._deduplicate_and_clean()
        self._built = True

    def _apply_perturbation_if_needed(self):
        n = len(self._original_points)
        if n < 4:
            return

        has_cocircular = False
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    cc = _circumcircle(self._original_points, i, j, k)
                    if cc is None:
                        continue
                    for m in range(k + 1, n):
                        if _on_circumcircle(self._original_points, i, j, k, m):
                            has_cocircular = True
                            break
                    if has_cocircular:
                        break
                if has_cocircular:
                    break
            if has_cocircular:
                break

        if has_cocircular:
            self.points = _perturb_points(self._original_points)
            while len(self.points) < len(self._original_points) + 3:
                self.points.append((0.0, 0.0))

    def _insert_point(self, p_idx):
        pts = self.points
        bad = []
        for tri in self.triangles:
            if _in_circumcircle(pts, tri[0], tri[1], tri[2], p_idx):
                bad.append(tri)

        if not bad:
            return

        edge_count: Dict[Tuple[int, int], int] = {}
        edge_to_directed: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for tri in bad:
            for ea, eb in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                ek = _edge_key(ea, eb)
                edge_count[ek] = edge_count.get(ek, 0) + 1
                if ek not in edge_to_directed:
                    edge_to_directed[ek] = (ea, eb)

        polygon = []
        for ek, count in edge_count.items():
            if count == 1:
                ea, eb = edge_to_directed[ek]
                polygon.append((ea, eb))

        bad_set = set(bad)
        self.triangles = [t for t in self.triangles if t not in bad_set]

        for ea, eb in polygon:
            new_tri = _make_counter_clockwise(self.points, p_idx, ea, eb)
            if not _is_degenerate_triangle(new_tri):
                self.triangles.append(new_tri)

    def _remove_super(self):
        super_set = set(self._super_indices)
        self.triangles = [
            t for t in self.triangles
            if not (t[0] in super_set or t[1] in super_set or t[2] in super_set)
        ]

    def _deduplicate_and_clean(self):
        seen: Set[Tuple[int, int, int]] = set()
        cleaned = []
        for t in self.triangles:
            if _is_degenerate_triangle(t):
                continue
            key = _triangle_key(*t)
            if key not in seen:
                seen.add(key)
                cleaned.append(t)
        self.triangles = cleaned

    def add_point(self, point: Tuple[float, float]) -> int:
        p_idx = len(self._original_points)
        self._original_points.append(point)

        if self._perturb:
            rng = random.Random(p_idx + 42)
            dx = rng.uniform(-PERTURBATION, PERTURBATION)
            dy = rng.uniform(-PERTURBATION, PERTURBATION)
            perturbed_point = (point[0] + dx, point[1] + dy)
        else:
            perturbed_point = point

        while len(self.points) <= p_idx:
            self.points.append((0.0, 0.0))
        self.points[p_idx] = perturbed_point

        if not self._built:
            if len(self._original_points) >= 3:
                self._build()
            return p_idx

        self._insert_point(p_idx)
        self._deduplicate_and_clean()

        return p_idx

    def get_triangles(self) -> List[Tuple[int, int, int]]:
        return list(self.triangles)

    def get_adjacency(self) -> List[List[Optional[int]]]:
        n_tri = len(self.triangles)
        edge_to_tri: Dict[Tuple[int, int], List[int]] = {}
        for ti in range(n_tri):
            t = self.triangles[ti]
            for ea, eb in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]:
                ek = _edge_key(ea, eb)
                if ek not in edge_to_tri:
                    edge_to_tri[ek] = []
                edge_to_tri[ek].append(ti)

        adjacency = [[None, None, None] for _ in range(n_tri)]
        for ti in range(n_tri):
            t = self.triangles[ti]
            for ei, (ea, eb) in enumerate([(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]):
                ek = _edge_key(ea, eb)
                tris = edge_to_tri.get(ek, [])
                for tj in tris:
                    if tj != ti:
                        adjacency[ti][ei] = tj
                        break
        return adjacency

    def get_edges(self) -> List[Tuple[int, int]]:
        edge_set = set()
        for t in self.triangles:
            for ea, eb in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]:
                edge_set.add(_edge_key(ea, eb))
        return list(edge_set)

    def get_point_triangles(self, point_idx: int) -> List[int]:
        result = []
        for ti, t in enumerate(self.triangles):
            if point_idx in t:
                result.append(ti)
        return result

    def get_point_neighbors(self, point_idx: int) -> Set[int]:
        neighbors = set()
        for t in self.triangles:
            if point_idx in t:
                for v in t:
                    if v != point_idx and v < len(self._original_points):
                        neighbors.add(v)
        return neighbors


class VoronoiCell:
    def __init__(self, point_idx: int):
        self.point_idx = point_idx
        self.vertices: List[Tuple[float, float]] = []
        self.is_finite: bool = True
        self.ray_directions: List[Tuple[float, float]] = []
        self._area: Optional[float] = None

    def area(self, bbox: Optional[Tuple[float, float, float, float]] = None) -> float:
        if not self.vertices:
            return 0.0

        if len(self.vertices) < 3 and self.is_finite:
            return 0.0

        if bbox is None:
            xs = [v[0] for v in self.vertices]
            ys = [v[1] for v in self.vertices]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            dx = max(max_x - min_x, 1e-6)
            dy = max(max_y - min_y, 1e-6)
            scale = 10.0
            bbox = (min_x - scale * dx, min_y - scale * dy,
                    max_x + scale * dx, max_y + scale * dy)

        if self.is_finite and len(self.vertices) >= 3:
            clipped = _clip_polygon_to_bbox(self.vertices, bbox)
            return _polygon_area(clipped)

        polygon = []
        n = len(self.vertices)
        for i in range(n):
            vx, vy = self.vertices[i]
            polygon.append((vx, vy))
            
            if i < len(self.ray_directions) and self.ray_directions[i] is not None:
                dx, dy = self.ray_directions[i]
                ray_len = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 10
                ext_x = vx + dx * ray_len
                ext_y = vy + dy * ray_len
                polygon.append((ext_x, ext_y))

        if len(polygon) < 3:
            return 0.0

        clipped = _clip_polygon_to_bbox(polygon, bbox)
        return _polygon_area(clipped)


class VoronoiDiagram:
    def __init__(self, dt: DelaunayTriangulation):
        self.dt = dt
        self.vertices: List[Tuple[float, float]] = []
        self.edges: List[Tuple[int, int]] = []
        self.ray_edges: List[Tuple[int, Tuple[float, float]]] = []
        self.cells: List[VoronoiCell] = []
        self._ccs: List[Optional[Tuple[float, float]]] = []
        self._cc_idx: Dict[int, int] = {}
        self._build()

    def _build(self):
        tris = self.dt.triangles
        n_tri = len(tris)
        n_pts = len(self.dt._original_points)

        if n_tri == 0:
            self.cells = [VoronoiCell(i) for i in range(n_pts)]
            return

        adj = self.dt.get_adjacency()

        self._ccs = []
        for t in tris:
            cc = _circumcircle(self.dt.points, t[0], t[1], t[2])
            self._ccs.append((cc[0], cc[1]) if cc else None)

        self.vertices = [cc for cc in self._ccs if cc is not None]
        self._cc_idx = {}
        vi = 0
        for i, cc in enumerate(self._ccs):
            if cc is not None:
                self._cc_idx[i] = vi
                vi += 1

        seen = set()
        for ti in range(n_tri):
            if self._ccs[ti] is None:
                continue
            for ei in range(3):
                tj = adj[ti][ei]
                if tj is not None and self._ccs[tj] is not None:
                    ek = (min(ti, tj), max(ti, tj))
                    if ek not in seen:
                        seen.add(ek)
                        self.edges.append((self._cc_idx[ti], self._cc_idx[tj]))
                elif tj is None and self._ccs[ti] is not None:
                    t = tris[ti]
                    ev0 = t[(ei + 1) % 3]
                    ev1 = t[(ei + 2) % 3]
                    if ev0 >= n_pts or ev1 >= n_pts:
                        continue
                    direction = self._outward_direction(ev0, ev1, self._ccs[ti])
                    self.ray_edges.append((self._cc_idx[ti], direction))

        self._build_cells()

    def _build_cells(self):
        tris = self.dt.triangles
        n_pts = len(self.dt._original_points)
        n_tri = len(tris)
        adj = self.dt.get_adjacency()

        point_to_tris: Dict[int, List[Tuple[int, int]]] = {}
        for ti in range(n_tri):
            t = tris[ti]
            for vi, v in enumerate(t):
                if v < n_pts:
                    if v not in point_to_tris:
                        point_to_tris[v] = []
                    point_to_tris[v].append((ti, vi))

        edge_to_tri: Dict[Tuple[int, int], List[int]] = {}
        for ti in range(n_tri):
            t = tris[ti]
            for ei in range(3):
                ek = _edge_key(t[ei], t[(ei + 1) % 3])
                if ek not in edge_to_tri:
                    edge_to_tri[ek] = []
                edge_to_tri[ek].append(ti)

        boundary_edges: Set[Tuple[int, int]] = set()
        for ek, tri_list in edge_to_tri.items():
            if len(tri_list) == 1:
                boundary_edges.add(ek)

        self.cells = []
        for p_idx in range(n_pts):
            cell = VoronoiCell(p_idx)
            if p_idx not in point_to_tris:
                self.cells.append(cell)
                continue

            tri_list = point_to_tris[p_idx]
            ordered = self._order_triangles_around_point(p_idx, tri_list, adj, tris)

            if len(ordered) == 0:
                self.cells.append(cell)
                continue

            for ti, vi in ordered:
                if ti in self._cc_idx and self._ccs[ti] is not None:
                    cell.vertices.append(self._ccs[ti])

            k = len(ordered)
            cell.ray_directions = [None] * k

            p_boundary_edges = []
            for ti, vi in ordered:
                t = tris[ti]
                for ei in range(3):
                    v0 = t[ei]
                    v1 = t[(ei + 1) % 3]
                    if v0 == p_idx or v1 == p_idx:
                        ek = _edge_key(v0, v1)
                        if ek in boundary_edges:
                            other = v1 if v0 == p_idx else v0
                            p_boundary_edges.append((ti, other))

            if len(p_boundary_edges) > 0:
                cell.is_finite = False
                for ti, other in p_boundary_edges:
                    for i, (ti_ordered, vi_ordered) in enumerate(ordered):
                        if ti_ordered == ti:
                            if ti in self._cc_idx and self._ccs[ti] is not None:
                                direction = self._outward_direction(p_idx, other, self._ccs[ti])
                                cell.ray_directions[i] = direction
                            break
            else:
                cell.is_finite = True

            self.cells.append(cell)

    def _order_triangles_around_point(self, p_idx, tri_list, adj, tris):
        if not tri_list:
            return []

        point = self.dt.points[p_idx]
        with_angle = []
        for ti, vi in tri_list:
            t = tris[ti]
            prev_v = t[(vi - 1) % 3]
            next_v = t[(vi + 1) % 3]
            if prev_v < len(self.dt.points) and next_v < len(self.dt.points):
                pv = self.dt.points[prev_v]
                nv = self.dt.points[next_v]
                angle1 = math.atan2(pv[1] - point[1], pv[0] - point[0])
                angle2 = math.atan2(nv[1] - point[1], nv[0] - point[0])
                mid_angle = (angle1 + angle2) / 2
                with_angle.append((mid_angle, ti, vi))

        with_angle.sort(key=lambda x: x[0])
        return [(ti, vi) for _, ti, vi in with_angle]

    def _outward_direction(self, v0, v1, cc):
        p0 = self.dt.points[v0]
        p1 = self.dt.points[v1]
        ex = p1[0] - p0[0]
        ey = p1[1] - p0[1]
        nx, ny = -ey, ex
        length = math.sqrt(nx * nx + ny * ny)
        if length < EPS:
            return (1.0, 0.0)
        nx /= length
        ny /= length
        mid_x = (p0[0] + p1[0]) / 2.0
        mid_y = (p0[1] + p1[1]) / 2.0
        if nx * (cc[0] - mid_x) + ny * (cc[1] - mid_y) > 0:
            nx, ny = -nx, -ny
        return (nx, ny)

    def rebuild(self):
        self.vertices = []
        self.edges = []
        self.ray_edges = []
        self.cells = []
        self._ccs = []
        self._cc_idx = {}
        self._build()

    def get_cells(self) -> List[VoronoiCell]:
        return list(self.cells)

    def get_cell_areas(self, bbox: Optional[Tuple[float, float, float, float]] = None) -> List[float]:
        return [cell.area(bbox) for cell in self.cells]

    def get_bbox(self, margin: float = 0.1) -> Tuple[float, float, float, float]:
        pts = self.dt._original_points
        if not pts:
            return (0, 0, 1, 1)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        dx = max(max_x - min_x, 1e-6)
        dy = max(max_y - min_y, 1e-6)
        return (min_x - margin * dx, min_y - margin * dy,
                max_x + margin * dx, max_y + margin * dy)


def delaunay(points: List[Tuple[float, float]], perturb: bool = True) -> Tuple[List[Tuple[int, int, int]], List[List[Optional[int]]]]:
    dt = DelaunayTriangulation(points, perturb=perturb)
    return dt.get_triangles(), dt.get_adjacency()


def voronoi(points: List[Tuple[float, float]], perturb: bool = True, bbox: Optional[Tuple[float, float, float, float]] = None) -> Tuple[List[VoronoiCell], List[float]]:
    dt = DelaunayTriangulation(points, perturb=perturb)
    vd = VoronoiDiagram(dt)
    cells = vd.get_cells()
    if bbox is None:
        bbox = vd.get_bbox()
    areas = vd.get_cell_areas(bbox)
    return cells, areas


def voronoi_full(points: List[Tuple[float, float]], perturb: bool = True) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]], List[Tuple[int, Tuple[float, float]]], List[VoronoiCell], List[float]]:
    dt = DelaunayTriangulation(points, perturb=perturb)
    vd = VoronoiDiagram(dt)
    bbox = vd.get_bbox()
    cells = vd.get_cells()
    areas = vd.get_cell_areas(bbox)
    return vd.vertices, vd.edges, vd.ray_edges, cells, areas


if __name__ == "__main__":
    import random

    random.seed(42)
    pts = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(10)]

    print("=" * 60)
    print("Initial Delaunay Triangulation")
    print("=" * 60)
    dt = DelaunayTriangulation(pts)
    triangles = dt.get_triangles()
    adjacency = dt.get_adjacency()

    print(f"Points: {len(pts)}")
    print(f"Triangles: {len(triangles)}")

    print("\n" + "=" * 60)
    print("Voronoi Diagram with Cells")
    print("=" * 60)
    vd = VoronoiDiagram(dt)
    cells = vd.get_cells()
    bbox = vd.get_bbox()
    areas = vd.get_cell_areas(bbox)

    print(f"Voronoi vertices: {len(vd.vertices)}")
    print(f"Voronoi edges: {len(vd.edges)}")
    print(f"Voronoi rays: {len(vd.ray_edges)}")
    print(f"Voronoi cells: {len(cells)}")

    print(f"\nBounding box: {bbox}")
    print(f"\nCell details:")
    total_area = 0.0
    for i, cell in enumerate(cells):
        area = areas[i]
        total_area += area
        finite_str = "finite" if cell.is_finite else "infinite"
        print(f"  Cell {i}: {len(cell.vertices)} vertices, {finite_str}, area={area:.4f}")
    print(f"\nTotal area of all cells: {total_area:.4f}")
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    print(f"Bounding box area: {bbox_area:.4f}")

    print("\n" + "=" * 60)
    print("Incremental Point Addition")
    print("=" * 60)
    new_point = (5.0, 5.0)
    new_idx = dt.add_point(new_point)
    print(f"Added point {new_idx}: {new_point}")
    print(f"New triangle count: {len(dt.get_triangles())}")

    vd.rebuild()
    cells = vd.get_cells()
    areas = vd.get_cell_areas(bbox)
    print(f"New cell count: {len(cells)}")
    print(f"New cell {new_idx}: {len(cells[new_idx].vertices)} vertices, area={areas[new_idx]:.4f}")

    try:
        import matplotlib.pyplot as plt
        import matplotlib.collections as mc
        from matplotlib.patches import Polygon

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        ax1.set_title("Delaunay Triangulation")
        ax1.set_aspect('equal')
        for i, (x, y) in enumerate(pts):
            ax1.plot(x, y, 'ko', markersize=4)
            ax1.annotate(str(i), (x, y), textcoords="offset points", xytext=(3, 3), fontsize=7)
        ax1.plot(new_point[0], new_point[1], 'ro', markersize=6, label='New point')
        lines = []
        for t in triangles:
            for a, b in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]:
                lines.append([pts[a], pts[b]])
        lc = mc.LineCollection(lines, colors='blue', linewidths=0.8)
        ax1.add_collection(lc)
        ax1.set_xlim(-1, 11)
        ax1.set_ylim(-1, 11)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.set_title("Voronoi Diagram")
        ax2.set_aspect('equal')
        for x, y in pts:
            ax2.plot(x, y, 'ko', markersize=4)
        ax2.plot(new_point[0], new_point[1], 'ro', markersize=6)
        lines_v = []
        for i, j in vd.edges:
            lines_v.append([vd.vertices[i], vd.vertices[j]])
        lc_v = mc.LineCollection(lines_v, colors='red', linewidths=0.8)
        ax2.add_collection(lc_v)
        ray_lines = []
        for vi_idx, (nx, ny) in vd.ray_edges:
            sx, sy = vd.vertices[vi_idx]
            ray_lines.append([(sx, sy), (sx + nx * 20, sy + ny * 20)])
        if ray_lines:
            lc_rays = mc.LineCollection(ray_lines, colors='red', linewidths=0.8, linestyles='dashed')
            ax2.add_collection(lc_rays)
        ax2.set_xlim(-1, 11)
        ax2.set_ylim(-1, 11)
        ax2.grid(True, alpha=0.3)

        ax3.set_title("Voronoi Cells (colored by area)")
        ax3.set_aspect('equal')
        cmap = plt.cm.get_cmap('viridis')
        max_area = max(areas) if areas else 1.0
        for i, cell in enumerate(cells):
            if len(cell.vertices) >= 3:
                poly_pts = list(cell.vertices)
                if not cell.is_finite:
                    for ri, (vx, vy) in enumerate(cell.vertices):
                        if ri < len(cell.ray_directions) and cell.ray_directions[ri] is not None:
                            dx, dy = cell.ray_directions[ri]
                            ext_x = vx + dx * 100
                            ext_y = vy + dy * 100
                            next_ri = (ri + 1) % len(cell.vertices)
                            poly_pts.insert(next_ri, (ext_x, ext_y))
                color = cmap(areas[i] / max_area)
                poly = Polygon(poly_pts, facecolor=color, alpha=0.6, edgecolor='black', linewidth=0.5)
                ax3.add_patch(poly)
        for x, y in pts:
            ax3.plot(x, y, 'ko', markersize=4)
        ax3.plot(new_point[0], new_point[1], 'ro', markersize=6)
        ax3.set_xlim(-1, 11)
        ax3.set_ylim(-1, 11)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("voronoi_cells.png", dpi=150)
        print("\nFigure saved to voronoi_cells.png")

    except ImportError:
        print("\n(matplotlib not available, skipping visualization)")
