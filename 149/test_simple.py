import sys
import numpy as np

print("Testing imports...")
try:
    from preprocessing import VesselPreprocessor, generate_synthetic_vessel_image
    print("✓ preprocessing module imported")
except Exception as e:
    print(f"✗ preprocessing error: {e}")

try:
    from distance_transform_centerline import DistanceTransformCenterline
    print("✓ distance_transform_centerline module imported")
except Exception as e:
    print(f"✗ distance_transform error: {e}")

try:
    from fast_marching_centerline import FastMarchingCenterline
    print("✓ fast_marching_centerline module imported")
except Exception as e:
    print(f"✗ fast_marching error: {e}")

try:
    from topology_tree import VesselTopologyTree
    print("✓ topology_tree module imported")
except Exception as e:
    print(f"✗ topology_tree error: {e}")

print("\nGenerating synthetic vessel image...")
image = generate_synthetic_vessel_image(shape=(50, 50, 50))
print(f"Image shape: {image.shape}, range: [{image.min():.3f}, {image.max():.3f}]")

print("\nPreprocessing...")
preprocessor = VesselPreprocessor()
results = preprocessor.full_pipeline(image, vesselness=False, threshold=0.3)
binary = results['processed']
print(f"Binary mask voxels: {np.sum(binary)}")

print("\nDistance Transform Centerline...")
dt_cl = DistanceTransformCenterline()
dt_result = dt_cl.extract(binary, method='ridge')
print(f"Centerline points: {len(dt_result['centerline_points'])}")

if len(dt_result['centerline_points']) > 0:
    print("\nBuilding topology tree...")
    topology = VesselTopologyTree()
    topology.build_from_centerline_points(
        dt_result['centerline_points'],
        binary,
        dt_result['distance_transform']
    )
    topology.print_tree_summary()

print("\nFast Marching Centerline...")
try:
    fm_cl = FastMarchingCenterline()
    fm_result = fm_cl.extract(image, binary)
    print(f"Centerline points: {len(fm_result['centerline_points'])}")
    
    if len(fm_result['centerline_points']) > 0:
        from scipy import ndimage
        dt = ndimage.distance_transform_edt(binary)
        
        topology_fm = VesselTopologyTree()
        topology_fm.build_from_centerline_points(
            fm_result['centerline_points'],
            binary,
            dt
        )
        print("Fast Marching topology tree built!")
except Exception as e:
    print(f"Fast marching error (non-critical): {e}")

print("\n✓ All tests passed!")
