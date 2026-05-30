import math
import random
from delaunay import DelaunayTriangulation, _in_circumcircle, _is_degenerate_triangle


def test_stress_cocircular():
    print("=" * 60)
    print("Stress Test: Cocircular Points")
    print("=" * 60)
    
    for n_points in [4, 8, 16, 32]:
        pts = []
        for i in range(n_points):
            angle = 2 * math.pi * i / n_points
            pts.append((math.cos(angle), math.sin(angle)))
        
        dt = DelaunayTriangulation(pts)
        triangles = dt.get_triangles()
        
        triangle_set = set()
        duplicates = 0
        degenerate = 0
        for t in triangles:
            if _is_degenerate_triangle(t):
                degenerate += 1
            key = tuple(sorted(t))
            if key in triangle_set:
                duplicates += 1
            triangle_set.add(key)
        
        violations = 0
        for t in triangles:
            for p_idx in range(n_points):
                if p_idx in t:
                    continue
                if _in_circumcircle(pts, t[0], t[1], t[2], p_idx):
                    violations += 1
        
        print(f"\nn={n_points} points on circle:")
        print(f"  Triangles: {len(triangles)}")
        print(f"  Duplicates: {duplicates}")
        print(f"  Degenerate: {degenerate}")
        print(f"  Empty circle violations: {violations}")
        
        assert duplicates == 0, f"Found {duplicates} duplicates!"
        assert degenerate == 0, f"Found {degenerate} degenerate triangles!"
        assert violations == 0, f"Found {violations} empty circle violations!"
    
    print("\n✓ All cocircular stress tests passed!")


def test_stress_random():
    print("\n" + "=" * 60)
    print("Stress Test: Random Points")
    print("=" * 60)
    
    random.seed(42)
    
    for n_points in [10, 50, 100]:
        pts = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n_points)]
        
        dt = DelaunayTriangulation(pts)
        triangles = dt.get_triangles()
        adjacency = dt.get_adjacency()
        
        triangle_set = set()
        duplicates = 0
        degenerate = 0
        for t in triangles:
            if _is_degenerate_triangle(t):
                degenerate += 1
            key = tuple(sorted(t))
            if key in triangle_set:
                duplicates += 1
            triangle_set.add(key)
        
        violations = 0
        for t in triangles:
            for p_idx in range(n_points):
                if p_idx in t:
                    continue
                if _in_circumcircle(pts, t[0], t[1], t[2], p_idx):
                    violations += 1
        
        point_usage = set()
        for t in triangles:
            point_usage.update(t)
        unused = len(set(range(n_points)) - point_usage)
        
        adj_violations = 0
        for ti in range(len(triangles)):
            for ei, neighbor in enumerate(adjacency[ti]):
                if neighbor is None:
                    continue
                found = False
                for nei, nbr in enumerate(adjacency[neighbor]):
                    if nbr == ti:
                        found = True
                        break
                if not found:
                    adj_violations += 1
        
        print(f"\nn={n_points} random points:")
        print(f"  Triangles: {len(triangles)} (expected ~{2*n_points})")
        print(f"  Duplicates: {duplicates}")
        print(f"  Degenerate: {degenerate}")
        print(f"  Unused points: {unused}")
        print(f"  Empty circle violations: {violations}")
        print(f"  Adjacency violations: {adj_violations}")
        
        assert duplicates == 0, f"Found {duplicates} duplicates!"
        assert degenerate == 0, f"Found {degenerate} degenerate triangles!"
        assert violations == 0, f"Found {violations} empty circle violations!"
        assert unused == 0, f"Found {unused} unused points!"
        assert adj_violations == 0, f"Found {adj_violations} adjacency violations!"
    
    print("\n✓ All random stress tests passed!")


def test_grid_points():
    print("\n" + "=" * 60)
    print("Test: Grid Points (many cocircular)")
    print("=" * 60)
    
    for grid_size in [3, 4, 5]:
        pts = []
        for i in range(grid_size):
            for j in range(grid_size):
                pts.append((i, j))
        
        n_points = len(pts)
        dt = DelaunayTriangulation(pts)
        triangles = dt.get_triangles()
        
        triangle_set = set()
        duplicates = 0
        degenerate = 0
        for t in triangles:
            if _is_degenerate_triangle(t):
                degenerate += 1
            key = tuple(sorted(t))
            if key in triangle_set:
                duplicates += 1
            triangle_set.add(key)
        
        violations = 0
        for t in triangles:
            for p_idx in range(n_points):
                if p_idx in t:
                    continue
                if _in_circumcircle(pts, t[0], t[1], t[2], p_idx):
                    violations += 1
        
        print(f"\n{grid_size}x{grid_size} grid ({n_points} points):")
        print(f"  Triangles: {len(triangles)}")
        print(f"  Duplicates: {duplicates}")
        print(f"  Degenerate: {degenerate}")
        print(f"  Empty circle violations: {violations}")
        
        assert duplicates == 0, f"Found {duplicates} duplicates!"
        assert degenerate == 0, f"Found {degenerate} degenerate triangles!"
        assert violations == 0, f"Found {violations} empty circle violations!"
    
    print("\n✓ All grid tests passed!")


def test_perturbation_toggle():
    print("\n" + "=" * 60)
    print("Test: Perturbation Toggle")
    print("=" * 60)
    
    pts = [(0, 0), (2, 0), (2, 2), (0, 2)]
    
    print("\nWith perturbation (default):")
    dt1 = DelaunayTriangulation(pts, perturb=True)
    print(f"  Triangles: {len(dt1.get_triangles())}")
    
    print("\nWithout perturbation:")
    dt2 = DelaunayTriangulation(pts, perturb=False)
    print(f"  Triangles: {len(dt2.get_triangles())}")
    
    print("\n✓ Perturbation toggle works!")


if __name__ == "__main__":
    test_stress_cocircular()
    test_stress_random()
    test_grid_points()
    test_perturbation_toggle()
    
    print("\n" + "=" * 60)
    print("ALL STRESS TESTS PASSED! ✓")
    print("=" * 60)
