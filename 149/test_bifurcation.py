import sys
import numpy as np

print("=" * 70)
print("VESSEL BIFURCATION ANALYSIS - TEST SCRIPT")
print("=" * 70)

print("\n[1/6] Testing imports...")
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

try:
    from bifurcation_analyzer import BifurcationAnalyzer
    print("  ✓ bifurcation_analyzer module imported")
except Exception as e:
    print(f"  ✗ bifurcation_analyzer error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[2/6] Generating synthetic vessel image...")
image = generate_synthetic_vessel_image(shape=(70, 70, 70))
print(f"  ✓ Image shape: {image.shape}")

print("\n[3/6] Preprocessing and segmentation...")
preprocessor = VesselPreprocessor()
results = preprocessor.full_pipeline(image, vesselness=False, threshold=0.3)
binary = results['processed']
print(f"  ✓ Binary mask voxels: {np.sum(binary)}")

print("\n[4/6] Extracting vessel tree skeleton...")
extractor = GlobalCenterlineExtractor()
skel_results = extractor.extract_complete_vessel_tree(
    binary, method='skeleton', min_branch_length=5
)
print(f"  ✓ Skeleton points: {len(skel_results['centerline_points'])}")

print("\n[5/6] Building topology tree...")
from scipy import ndimage
dt = ndimage.distance_transform_edt(binary)

topology = VesselTopologyTree()
topology.build_from_centerline_points(
    skel_results['centerline_points'],
    binary,
    dt
)
topology.print_tree_summary()

print("\n[6/6] Running bifurcation analysis...")
analyzer = BifurcationAnalyzer()
bifurcations = analyzer.comprehensive_bifurcation_analysis(topology)

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED!")
print("=" * 70)

print("\n" + "-" * 70)
print("BIFURCATION ANALYSIS SUMMARY")
print("-" * 70)
print(f"Total bifurcations detected: {len(bifurcations)}")

risk_levels = {'low': 0, 'medium': 0, 'high': 0}
for b in bifurcations:
    risk_levels[b.get('risk_level', 'unknown')] += 1

print(f"Risk distribution:")
print(f"  - Low risk:    {risk_levels['low']}")
print(f"  - Medium risk: {risk_levels['medium']}")
print(f"  - High risk:   {risk_levels['high']}")

print("\n" + "-" * 70)
print("To run full analysis with outputs:")
print("  python main_global.py --method skeleton")
print("")
print("Output files will be saved to output/ directory:")
print("  - bifurcations_3d.png     : 3D visualization with risk coloring")
print("  - bifurcation_report.txt  : Clinical planning report")
print("  - bifurcations.csv        : Bifurcation data spreadsheet")
print("  - centerline.vtk          : VTK file for ParaView")
print("" + "=" * 70)
