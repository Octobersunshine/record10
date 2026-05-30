import random
from delaunay import DelaunayTriangulation, VoronoiDiagram

random.seed(42)
pts = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(10)]

print("Points:")
for i, p in enumerate(pts):
    print(f"  {i}: {p}")

dt = DelaunayTriangulation(pts)
triangles = dt.get_triangles()

print("\nTriangles:")
for i, t in enumerate(triangles):
    print(f"  T{i}: {t}")

print("\nPoint to triangles mapping:")
n_pts = len(pts)
point_to_tris = {}
for ti, t in enumerate(triangles):
    for vi, v in enumerate(t):
        if v < n_pts:
            if v not in point_to_tris:
                point_to_tris[v] = []
            point_to_tris[v].append((ti, vi))

for p_idx in range(n_pts):
    if p_idx in point_to_tris:
        tris_list = point_to_tris[p_idx]
        print(f"  Point {p_idx}: triangles = {[ti for ti, _ in tris_list]}")
    else:
        print(f"  Point {p_idx}: NO TRIANGLES!")

vd = VoronoiDiagram(dt)
cells = vd.get_cells()

print("\nCells:")
for i, cell in enumerate(cells):
    print(f"  Cell {i}: {len(cell.vertices)} vertices, "
          f"{'finite' if cell.is_finite else 'infinite'}")
    if len(cell.vertices) == 2:
        print(f"    WARNING: Only 2 vertices!")
        print(f"    Point {i} triangles: {point_to_tris.get(i, [])}")
        tris_list = point_to_tris.get(i, [])
        for ti, vi in tris_list:
            adj = dt.get_adjacency()
            print(f"    T{ti}: adj = {adj[ti]}, tri = {triangles[ti]}")
