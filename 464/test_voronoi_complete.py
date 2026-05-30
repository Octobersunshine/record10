import random
import math
from delaunay import (
    DelaunayTriangulation, VoronoiDiagram, VoronoiCell,
    delaunay, voronoi, voronoi_full,
    _polygon_area, _clip_polygon_to_bbox
)


def test_voronoi_cells_basic():
    print("=" * 60)
    print("Test 1: Basic Voronoi Cell Generation")
    print("=" * 60)
    
    random.seed(42)
    pts = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(10)]
    
    dt = DelaunayTriangulation(pts)
    vd = VoronoiDiagram(dt)
    cells = vd.get_cells()
    bbox = vd.get_bbox()
    areas = vd.get_cell_areas(bbox)
    
    print(f"\nNumber of points: {len(pts)}")
    print(f"Number of cells: {len(cells)}")
    print(f"Bounding box: {bbox}")
    
    assert len(cells) == len(pts), f"Cell count mismatch: {len(cells)} vs {len(pts)}"
    
    valid_cells = 0
    finite_cells = 0
    infinite_cells = 0
    for i, cell in enumerate(cells):
        assert cell.point_idx == i, f"Cell point index mismatch: {cell.point_idx} vs {i}"
        if len(cell.vertices) >= 3:
            valid_cells += 1
        if cell.is_finite:
            finite_cells += 1
        else:
            infinite_cells += 1
        
        if len(cell.vertices) >= 3:
            area = cell.area(bbox)
            assert area >= 0, f"Negative area for cell {i}: {area}"
            print(f"  Cell {i}: {len(cell.vertices)} vertices, "
                  f"{'finite' if cell.is_finite else 'infinite'}, "
                  f"area={area:.4f}")
        else:
            print(f"  Cell {i}: {len(cell.vertices)} vertices (insufficient)")
    
    print(f"\nValid cells (≥3 vertices): {valid_cells}/{len(cells)}")
    print(f"Finite cells: {finite_cells}")
    print(f"Infinite cells: {infinite_cells}")
    
    total_area = sum(areas)
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    print(f"\nTotal cell area: {total_area:.4f}")
    print(f"Bounding box area: {bbox_area:.4f}")
    print(f"Ratio: {total_area / bbox_area:.4f}")
    
    print("\n✓ Basic Voronoi cell test passed!")


def test_incremental_addition():
    print("\n" + "=" * 60)
    print("Test 2: Incremental Point Addition")
    print("=" * 60)
    
    random.seed(123)
    initial_pts = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(5)]
    
    dt = DelaunayTriangulation(initial_pts)
    vd = VoronoiDiagram(dt)
    
    print(f"\nInitial state:")
    print(f"  Points: {len(initial_pts)}")
    print(f"  Triangles: {len(dt.get_triangles())}")
    print(f"  Voronoi cells: {len(vd.get_cells())}")
    
    initial_tri_count = len(dt.get_triangles())
    
    for i in range(5):
        new_point = (random.uniform(0, 10), random.uniform(0, 10))
        new_idx = dt.add_point(new_point)
        
        assert new_idx == len(initial_pts) + i, f"Index mismatch: {new_idx}"
        
        vd.rebuild()
        
        new_tri_count = len(dt.get_triangles())
        new_cell_count = len(vd.get_cells())
        
        print(f"\n  Added point {new_idx}: {new_point}")
        print(f"    Triangles: {initial_tri_count} → {new_tri_count}")
        print(f"    Cells: {new_cell_count}")
        
        assert new_tri_count > initial_tri_count, "Triangle count should increase"
        assert new_cell_count == len(initial_pts) + i + 1, "Cell count mismatch"
        
        initial_tri_count = new_tri_count
        
        new_cell = vd.get_cells()[new_idx]
        print(f"    New cell: {len(new_cell.vertices)} vertices, "
              f"{'finite' if new_cell.is_finite else 'infinite'}")
    
    print("\n✓ Incremental addition test passed!")


def test_cell_areas_consistency():
    print("\n" + "=" * 60)
    print("Test 3: Cell Area Consistency")
    print("=" * 60)
    
    pts = [(0, 0), (2, 0), (2, 2), (0, 2), (1, 1)]
    
    dt = DelaunayTriangulation(pts)
    vd = VoronoiDiagram(dt)
    
    bbox_small = vd.get_bbox(margin=0.1)
    bbox_large = vd.get_bbox(margin=10.0)
    
    areas_small = vd.get_cell_areas(bbox_small)
    areas_large = vd.get_cell_areas(bbox_large)
    
    print(f"\nSmall bbox: {bbox_small}")
    print(f"Large bbox: {bbox_large}")
    
    print("\nCell areas with different bboxes:")
    for i in range(len(pts)):
        cell = vd.get_cells()[i]
        print(f"  Cell {i}: "
              f"small_bbox={areas_small[i]:.4f}, "
              f"large_bbox={areas_large[i]:.4f}, "
              f"{'finite' if cell.is_finite else 'infinite'}")
        
        if cell.is_finite:
            assert abs(areas_small[i] - areas_large[i]) < 1e-6, \
                f"Finite cell area should be independent of bbox"
    
    total_small = sum(areas_small)
    total_large = sum(areas_large)
    bbox_area_small = (bbox_small[2] - bbox_small[0]) * (bbox_small[3] - bbox_small[1])
    bbox_area_large = (bbox_large[2] - bbox_large[0]) * (bbox_large[3] - bbox_large[1])
    
    print(f"\nTotal area (small bbox): {total_small:.4f} / {bbox_area_small:.4f} = {total_small / bbox_area_small:.4f}")
    print(f"Total area (large bbox): {total_large:.4f} / {bbox_area_large:.4f} = {total_large / bbox_area_large:.4f}")
    
    print("\n✓ Cell area consistency test passed!")


def test_convenience_functions():
    print("\n" + "=" * 60)
    print("Test 4: Convenience Functions")
    print("=" * 60)
    
    pts = [(0, 0), (1, 0), (0.5, 1), (0.2, 0.5), (0.8, 0.5)]
    
    print("\nTesting delaunay():")
    tris, adj = delaunay(pts)
    print(f"  Triangles: {len(tris)}")
    print(f"  Adjacency: {len(adj)} entries")
    assert len(tris) > 0
    assert len(adj) == len(tris)
    
    print("\nTesting voronoi():")
    cells, areas = voronoi(pts)
    print(f"  Cells: {len(cells)}")
    print(f"  Areas: {len(areas)}")
    assert len(cells) == len(pts)
    assert len(areas) == len(pts)
    
    print("\nTesting voronoi_full():")
    v_verts, v_edges, v_rays, v_cells, v_areas = voronoi_full(pts)
    print(f"  Vertices: {len(v_verts)}")
    print(f"  Edges: {len(v_edges)}")
    print(f"  Rays: {len(v_rays)}")
    print(f"  Cells: {len(v_cells)}")
    print(f"  Areas: {len(v_areas)}")
    assert len(v_verts) == len(tris)
    assert len(v_cells) == len(pts)
    
    print("\n✓ Convenience functions test passed!")


def test_polygon_clipping():
    print("\n" + "=" * 60)
    print("Test 5: Polygon Clipping")
    print("=" * 60)
    
    square = [(0, 0), (2, 0), (2, 2), (0, 2)]
    bbox = (-1, -1, 1, 1)
    
    clipped = _clip_polygon_to_bbox(square, bbox)
    area_full = _polygon_area(square)
    area_clipped = _polygon_area(clipped)
    
    print(f"\nOriginal square area: {area_full:.4f}")
    print(f"Clipped polygon vertices: {len(clipped)}")
    print(f"Clipped area: {area_clipped:.4f}")
    print(f"Expected clipped area: 1.0 (intersection with unit bbox)")
    
    assert abs(area_clipped - 1.0) < 1e-6, f"Clipped area mismatch: {area_clipped} vs 1.0"
    
    triangle = [(0, 0), (3, 0), (0, 3)]
    clipped_tri = _clip_polygon_to_bbox(triangle, bbox)
    area_tri = _polygon_area(clipped_tri)
    print(f"\nClipped triangle vertices: {clipped_tri}")
    print(f"Clipped triangle area: {area_tri:.4f}")
    print(f"Expected: 1.0 (intersection is a unit square)")
    assert abs(area_tri - 1.0) < 1e-6, f"Clipped triangle area mismatch: {area_tri} vs 1.0"
    
    print("\n✓ Polygon clipping test passed!")


def test_voronoi_cell_class():
    print("\n" + "=" * 60)
    print("Test 6: VoronoiCell Class")
    print("=" * 60)
    
    cell = VoronoiCell(0)
    
    print(f"\nEmpty cell:")
    print(f"  point_idx: {cell.point_idx}")
    print(f"  vertices: {cell.vertices}")
    print(f"  is_finite: {cell.is_finite}")
    print(f"  area: {cell.area()}")
    
    assert cell.area() == 0.0, "Empty cell should have area 0"
    
    cell.vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
    cell.is_finite = True
    area = cell.area()
    print(f"\nUnit square cell:")
    print(f"  area: {area:.4f} (expected 1.0)")
    assert abs(area - 1.0) < 1e-6, f"Area mismatch: {area} vs 1.0"
    
    print("\n✓ VoronoiCell class test passed!")


def test_incremental_from_empty():
    print("\n" + "=" * 60)
    print("Test 7: Incremental Building From Empty")
    print("=" * 60)
    
    dt = DelaunayTriangulation([])
    print(f"After empty init: triangles={len(dt.get_triangles())}")
    
    idx1 = dt.add_point((0, 0))
    print(f"After 1 point: triangles={len(dt.get_triangles())}, idx={idx1}")
    assert idx1 == 0
    
    idx2 = dt.add_point((1, 0))
    print(f"After 2 points: triangles={len(dt.get_triangles())}, idx={idx2}")
    assert idx2 == 1
    
    idx3 = dt.add_point((0.5, 1))
    print(f"After 3 points: triangles={len(dt.get_triangles())}, idx={idx3}")
    assert idx3 == 2
    assert len(dt.get_triangles()) == 1
    
    vd = VoronoiDiagram(dt)
    print(f"Voronoi cells: {len(vd.get_cells())}")
    assert len(vd.get_cells()) == 3
    
    for i in range(10):
        idx = dt.add_point((random.uniform(-1, 2), random.uniform(-1, 2)))
        vd.rebuild()
    
    print(f"\nAfter adding 13 points total:")
    print(f"  Triangles: {len(dt.get_triangles())}")
    print(f"  Voronoi cells: {len(vd.get_cells())}")
    
    assert len(vd.get_cells()) == 13
    
    print("\n✓ Incremental from empty test passed!")


if __name__ == "__main__":
    test_voronoi_cells_basic()
    test_incremental_addition()
    test_cell_areas_consistency()
    test_convenience_functions()
    test_polygon_clipping()
    test_voronoi_cell_class()
    test_incremental_from_empty()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)
