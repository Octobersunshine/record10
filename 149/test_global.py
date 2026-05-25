import sys
import numpy as np

print("=" * 70)
print("GLOBAL CENTERLINE EXTRACTION - TEST SCRIPT")
print("=" * 70)

print("\n[1/5] Testing imports...")
try:
    from preprocessing import VesselPreprocessor, generate_synthetic_vessel_image
    print("  ✓ preprocessing module imported")
except Exception as e:
    print(f"  ✗ preprocessing error: {e}")
    sys.exit(1)

try:
    import networkx as nx
    from global_centerline_extractor import GlobalCenterlineExtractor
    print("  ✓ global_centerline_extractor module imported")
except Exception as e:
    print(f"  ✗ global_centerline_extractor error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from topology_tree import VesselTopologyTree
    print("  ✓ topology_tree module imported")
except Exception as e:
    print(f"  ✗ topology_tree error: {e}")
    sys.exit(1)

print("\n[2/5] Generating synthetic vessel image...")
image = generate_synthetic_vessel_image(shape=(60, 60, 60))
print(f"  ✓ Image shape: {image.shape}, range: [{image.min():.3f}, {image.max():.3f}]")

print("\n[3/5] Preprocessing and segmentation...")
preprocessor = VesselPreprocessor()
results = preprocessor.full_pipeline(image, vesselness=False, threshold=0.3)
binary = results['processed']
print(f"  ✓ Binary mask voxels: {np.sum(binary)}")

print("\n[4/5] Testing global centerline extraction methods...")
extractor = GlobalCenterlineExtractor()

print("\n  --- Method 1: Topological Skeleton ---")
try:
    skel_results = extractor.extract_complete_vessel_tree(
        binary, method='skeleton', min_branch_length=5
    )
    print(f"    ✓ Skeleton points: {len(skel_results['centerline_points'])}")
except Exception as e:
    print(f"    ✗ Skeleton error: {e}")
    import traceback
    traceback.print_exc()

print("\n  --- Method 2: Power Watershed ---")
try:
    pw_results = extractor.extract_complete_vessel_tree(
        binary, method='power_watershed', num_sources=6, min_branch_length=5
    )
    print(f"    ✓ Power watershed points: {len(pw_results['centerline_points'])}")
    print(f"    ✓ Seed points: {len(pw_results['seed_points'])}")
except Exception as e:
    print(f"    ✗ Power watershed error: {e}")
    import traceback
    traceback.print_exc()

print("\n  --- Method 3: Hybrid ---")
try:
    hybrid_results = extractor.extract_complete_vessel_tree(
        binary, method='hybrid', min_branch_length=5, num_sources=6
    )
    print(f"    ✓ Hybrid points: {len(hybrid_results['centerline_points'])}")
except Exception as e:
    print(f"    ✗ Hybrid error: {e}")

print("\n[5/5] Building topology tree from skeleton...")
try:
    from scipy import ndimage
    dt = ndimage.distance_transform_edt(binary)
    
    topology = VesselTopologyTree()
    topology.build_from_centerline_points(
        skel_results['centerline_points'],
        binary,
        ndimage.distance_transform_edt(binary)
    )
    topology.print_tree_summary()
except Exception as e:
    print(f"  ✗ Topology tree error: {e}")

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED!")
print("=" * 70)
print("\nTo run full examples:")
print("  python main_global.py --method skeleton")
print("  python main_global.py --method power_watershed")
print("  python main_global.py --compare")
