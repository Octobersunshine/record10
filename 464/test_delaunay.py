import random
import math
from delaunay import DelaunayTriangulation, delaunay, voronoi, voronoi_full, _circumcircle, _in_circumcircle


def test_delaunay_properties():
    print("=" * 60)
    print("Testing Delaunay Triangulation Properties")
    print("=" * 60)

    random.seed(123)
    n_points = 30
    pts = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n_points)]

    dt = DelaunayTriangulation(pts)
    triangles = dt.get_triangles()
    adjacency = dt.get_adjacency()

    print(f"\nNumber of points: {n_points}")
    print(f"Number of triangles: {len(triangles)}")
    print(f"Expected range: [{2 * n_points - 5}, {2 * n_points - 2}]")
    assert 2 * n_points - 10 <= len(triangles) <= 2 * n_points + 5, \
        f"Triangle count {len(triangles)} out of expected range"

    print("\n1. Checking all points are used...")
    point_usage = set()
    for t in triangles:
        point_usage.update(t)
    all_original_points = set(range(n_points))
    unused = all_original_points - point_usage
    assert len(unused) == 0, f"Unused points: {unused}"
    print(f"   ✓ All {n_points} points are used in triangulation")

    print("\n2. Checking empty circumcircle property...")
    violations = 0
    for t in triangles:
        v0, v1, v2 = t
        cc = _circumcircle(pts, v0, v1, v2)
        if cc is None:
            continue
        cx, cy, r_sq = cc
        for p_idx in range(n_points):
            if p_idx in t:
                continue
            px, py = pts[p_idx]
            dist_sq = (px - cx) ** 2 + (py - cy) ** 2
            if dist_sq < r_sq - 1e-8:
                violations += 1
                print(f"   Violation: point {p_idx} inside triangle {t} circumcircle")
    assert violations == 0, f"Found {violations} empty circle violations"
    print(f"   ✓ Empty circumcircle property satisfied (0 violations)")

    print("\n3. Checking adjacency consistency...")
    adj_violations = 0
    for ti, t in enumerate(triangles):
        for ei, neighbor in enumerate(adjacency[ti]):
            if neighbor is None:
                continue
            if neighbor < 0 or neighbor >= len(triangles):
                adj_violations += 1
                print(f"   Invalid neighbor index: T{ti}.adj[{ei}] = {neighbor}")
                continue
            found = False
            for nei, nbr in enumerate(adjacency[neighbor]):
                if nbr == ti:
                    found = True
                    break
            if not found:
                adj_violations += 1
                print(f"   Asymmetric adjacency: T{ti} → T{neighbor}, but T{neighbor} ↛ T{ti}")
    assert adj_violations == 0, f"Found {adj_violations} adjacency violations"
    print(f"   ✓ Adjacency relations are consistent")

    print("\n4. Checking edge orientation (counter-clockwise)...")
    ccw_violations = 0
    for t in triangles:
        v0, v1, v2 = t
        ax, ay = pts[v0]
        bx, by = pts[v1]
        cx, cy = pts[v2]
        cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if cross <= 0:
            ccw_violations += 1
            print(f"   Triangle {t} is not CCW (cross = {cross})")
    assert ccw_violations == 0, f"Found {ccw_violations} orientation violations"
    print(f"   ✓ All triangles are counter-clockwise oriented")

    print("\n5. Testing convenience function delaunay()...")
    tris, adj = delaunay(pts)
    assert tris == triangles, "delaunay() triangles mismatch"
    assert adj == adjacency, "delaunay() adjacency mismatch"
    print(f"   ✓ delaunay() returns correct triangles and adjacency")

    print("\n6. Testing Voronoi diagram generation...")
    cells, areas = voronoi(pts)
    print(f"   Voronoi cells: {len(cells)}")
    print(f"   Areas computed: {len(areas)}")
    assert len(cells) == n_points, f"Voronoi cells count mismatch: {len(cells)} vs {n_points}"
    assert len(areas) == n_points, f"Areas count mismatch: {len(areas)} vs {n_points}"
    print(f"   ✓ Voronoi diagram with cells generated successfully")

    print("\n7. Testing voronoi_full() for complete data...")
    v_verts, v_edges, v_rays, v_cells, v_areas = voronoi_full(pts)
    print(f"   Voronoi vertices: {len(v_verts)}")
    print(f"   Voronoi finite edges: {len(v_edges)}")
    print(f"   Voronoi infinite rays: {len(v_rays)}")
    print(f"   Voronoi cells: {len(v_cells)}")
    assert len(v_verts) == len(triangles), f"Voronoi vertices count mismatch: {len(v_verts)} vs {len(triangles)}"
    print(f"   ✓ Full Voronoi diagram generated successfully")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

    return triangles, adjacency


def test_special_cases():
    print("\n" + "=" * 60)
    print("Testing Special Cases")
    print("=" * 60)

    print("\n1. Collinear points...")
    collinear_pts = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    dt = DelaunayTriangulation(collinear_pts)
    tris = dt.get_triangles()
    print(f"   Input: 5 collinear points")
    print(f"   Triangles: {len(tris)} (may be 0 for degenerate case)")

    print("\n2. Minimum points (3 points)...")
    tri_pts = [(0, 0), (1, 0), (0.5, 1)]
    dt = DelaunayTriangulation(tri_pts)
    tris = dt.get_triangles()
    adj = dt.get_adjacency()
    print(f"   Input: 3 points forming triangle")
    print(f"   Triangles: {len(tris)}")
    print(f"   Triangle: {tris[0]}")
    print(f"   Adjacency: {adj[0]} (all should be None for boundary)")
    assert len(tris) == 1, f"Expected 1 triangle, got {len(tris)}"
    assert all(a is None for a in adj[0]), "Expected all None adjacency"

    print("\n3. Square with center point...")
    square_pts = [(0, 0), (2, 0), (2, 2), (0, 2), (1, 1)]
    dt = DelaunayTriangulation(square_pts)
    tris = dt.get_triangles()
    adj = dt.get_adjacency()
    print(f"   Input: 5 points (square + center)")
    print(f"   Triangles: {len(tris)} (expected 4)")
    for i, t in enumerate(tris):
        print(f"   T{i}: {t}, neighbors: {adj[i]}")
    assert len(tris) == 4, f"Expected 4 triangles, got {len(tris)}"

    print("\n" + "=" * 60)
    print("Special case tests passed! ✓")
    print("=" * 60)


def test_usage_example():
    print("\n" + "=" * 60)
    print("Usage Example: How to Use the API")
    print("=" * 60)

    pts = [(0, 0), (2, 0), (2, 2), (0, 2), (1, 1), (1, 0.5)]

    triangles, adjacency = delaunay(pts)

    print("\nInput points:")
    for i, p in enumerate(pts):
        print(f"  P{i}: {p}")

    print("\nTriangles (vertex indices):")
    for i, t in enumerate(triangles):
        print(f"  T{i}: {t}")

    print("\nAdjacency (neighbor triangle indices for each edge):")
    for i, neighbors in enumerate(adjacency):
        edges_desc = []
        for ei, nbr in enumerate(neighbors):
            if nbr is None:
                edges_desc.append("boundary")
            else:
                edges_desc.append(f"T{nbr}")
        print(f"  T{i}: edge0→{edges_desc[0]}, edge1→{edges_desc[1]}, edge2→{edges_desc[2]}")

    print("\nGenerating Voronoi diagram...")
    v_cells, v_areas = voronoi(pts)
    print(f"Voronoi cells: {len(v_cells)}")
    for i, cell in enumerate(v_cells):
        finite_str = "finite" if cell.is_finite else "infinite"
        print(f"  Cell {i}: {len(cell.vertices)} vertices, {finite_str}, area={v_areas[i]:.3f}")


if __name__ == "__main__":
    test_delaunay_properties()
    test_special_cases()
    test_usage_example()
