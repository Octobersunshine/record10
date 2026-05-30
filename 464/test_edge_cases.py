import random
import math
from delaunay import DelaunayTriangulation, _circumcircle, _in_circumcircle


def test_points_on_circle_strict():
    print("Testing strict cocircular points with various configurations...")
    
    results = []
    for n in [4, 6, 8, 12]:
        pts = []
        for i in range(n):
            angle = 2 * math.pi * i / n
            pts.append((math.cos(angle), math.sin(angle)))
        
        dt = DelaunayTriangulation(pts)
        triangles = dt.get_triangles()
        
        triangle_set = set()
        duplicates = 0
        for t in triangles:
            key = tuple(sorted(t))
            if key in triangle_set:
                duplicates += 1
            triangle_set.add(key)
        
        degenerate = 0
        for t in triangles:
            if len(set(t)) < 3:
                degenerate += 1
        
        print(f"  n={n}: {len(triangles)} triangles, {duplicates} duplicates, {degenerate} degenerate")
        
        violations = 0
        for t in triangles:
            for p_idx in range(n):
                if p_idx in t:
                    continue
                if _in_circumcircle(pts, t[0], t[1], t[2], p_idx):
                        violations += 1
                        print(f"    Violation: point {p_idx} in triangle {t}")
        
        results.append((n, len(triangles), duplicates, degenerate, violations))
    
    return results


def test_super_triangle_size():
    print("\nTesting super triangle size issues...")
    
    pts = [(-100, -100), (100, -100), (100, 100), (-100, 100), (0, 0)]
    
    dt = DelaunayTriangulation(pts)
    triangles = dt.get_triangles()
    
    print(f"  Points: {len(pts)}")
    print(f"  Triangles: {len(triangles)}")
    
    for i, t in enumerate(triangles):
        print(f"    T{i}: {t}")
    
    all_pts_used = set()
    for t in triangles:
        all_pts_used.update(t)
    print(f"  Points used: {sorted(all_pts_used)}")
    
    return triangles


def test_edge_cases():
    print("\nTesting edge cases...")
    
    print("  1. Points forming a regular grid...")
    pts = []
    for i in range(4):
        for j in range(4):
            pts.append((i, j))
    
    dt = DelaunayTriangulation(pts)
    triangles = dt.get_triangles()
    print(f"     16 grid points → {len(triangles)} triangles")
    
    triangle_set = set()
    duplicates = 0
    for t in triangles:
        key = tuple(sorted(t))
        if key in triangle_set:
            duplicates += 1
        triangle_set.add(key)
    print(f"     Duplicates: {duplicates}")
    
    print("  2. Very close points...")
    pts = [(0, 0), (1, 0), (0, 1), (0.5, 0.5), (0.5000001, 0.5)]
    dt = DelaunayTriangulation(pts)
    triangles = dt.get_triangles()
    print(f"     5 close points → {len(triangles)} triangles")
    
    print("  3. Points in a line (degenerate)...")
    pts = list(zip([i * 0.1 for i in range(10)], [0 for _ in range(10)]))
    dt = DelaunayTriangulation(pts)
    triangles = dt.get_triangles()
    print(f"     10 collinear points → {len(triangles)} triangles (expected 0)")


if __name__ == "__main__":
    test_points_on_circle_strict()
    test_super_triangle_size()
    test_edge_cases()
