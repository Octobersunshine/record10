import random
from delaunay import DelaunayTriangulation, _circumcircle, _in_circumcircle


def test_four_cocircular_points():
    print("Testing four cocircular points (degenerate case)...")
    
    pts = [(0, 0), (2, 0), (2, 2), (0, 2)]
    
    dt = DelaunayTriangulation(pts)
    triangles = dt.get_triangles()
    adjacency = dt.get_adjacency()
    
    print(f"Points: {pts}")
    print(f"Number of triangles: {len(triangles)}")
    
    for i, t in enumerate(triangles):
        print(f"  T{i}: {t}")
        cc = _circumcircle(pts, t[0], t[1], t[2])
        if cc:
            print(f"    circumcircle: center=({cc[0]:.3f}, {cc[1]:.3f}), r²={cc[2]:.3f}")
    
    triangle_set = set()
    duplicates = 0
    for t in triangles:
        key = tuple(sorted(t))
        if key in triangle_set:
            duplicates += 1
            print(f"  DUPLICATE: {t}")
        triangle_set.add(key)
    
    if duplicates > 0:
        print(f"  Found {duplicates} duplicate triangles!")
    else:
        print("  No duplicate triangles")
    
    edges = set()
    invalid_edges = 0
    for t in triangles:
        for a, b in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]:
            if a == b:
                invalid_edges += 1
                print(f"  INVALID EDGE (same vertex): {a}-{b} in triangle {t}")
    
    if invalid_edges > 0:
        print(f"  Found {invalid_edges} invalid edges!")
    else:
        print("  No invalid edges")
    
    return triangles


def test_boundary_super_triangle():
    print("\nTesting super triangle boundary removal...")
    
    pts = [(0, 0), (10, 0), (10, 10), (0, 10), (5, 5)]
    
    dt = DelaunayTriangulation(pts)
    triangles = dt.get_triangles()
    
    print(f"Number of triangles: {len(triangles)}")
    
    has_super = False
    for t in triangles:
        if any(v >= len(pts) for v in t):
            has_super = True
            print(f"  Triangle contains super vertex: {t}")
    
    if has_super:
        print("  ERROR: Super vertices not properly removed!")
    else:
        print("  Super vertices properly removed")
    
    return triangles


def test_many_cocircular():
    print("\nTesting many cocircular points...")
    
    import math
    n = 8
    pts = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        pts.append((math.cos(angle), math.sin(angle)))
    
    dt = DelaunayTriangulation(pts)
    triangles = dt.get_triangles()
    
    print(f"Number of points on circle: {n}")
    print(f"Number of triangles: {len(triangles)}")
    
    triangle_set = set()
    duplicates = 0
    for t in triangles:
        key = tuple(sorted(t))
        if key in triangle_set:
            duplicates += 1
        triangle_set.add(key)
    
    print(f"Duplicate triangles: {duplicates}")
    
    for i, t in enumerate(triangles[:10]):
        print(f"  T{i}: {t}")
    
    return triangles


if __name__ == "__main__":
    test_four_cocircular_points()
    test_boundary_super_triangle()
    test_many_cocircular()
