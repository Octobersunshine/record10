import numpy as np
import argparse
import os
import sys
from preprocessing import VesselPreprocessor, generate_synthetic_vessel_image
from distance_transform_centerline import DistanceTransformCenterline
from fast_marching_centerline import FastMarchingCenterline
from topology_tree import VesselTopologyTree


def run_distance_transform_example():
    print("\n" + "="*60)
    print("Running Distance Transform Centerline Extraction")
    print("="*60)
    
    print("Generating synthetic vessel image...")
    image = generate_synthetic_vessel_image(shape=(80, 80, 80))
    print(f"Image shape: {image.shape}")
    
    print("\nPreprocessing and segmentation...")
    preprocessor = VesselPreprocessor(spacing=(1.0, 1.0, 1.0))
    results = preprocessor.full_pipeline(image, vesselness=False, threshold=0.3)
    binary_mask = results['processed']
    print(f"Voxel count: {np.sum(binary_mask)}")
    
    print("\nExtracting centerline using distance transform (ridge method)...")
    dt_centerline = DistanceTransformCenterline(spacing=(1.0, 1.0, 1.0))
    cl_results = dt_centerline.extract(binary_mask, method='ridge')
    
    centerline_points = cl_results['centerline_points']
    print(f"Centerline points: {len(centerline_points)}")
    
    if len(centerline_points) > 0:
        print("\nBuilding topology tree...")
        topology = VesselTopologyTree(spacing=(1.0, 1.0, 1.0))
        topology.build_from_centerline_points(
            centerline_points, 
            binary_mask, 
            cl_results['distance_transform']
        )
        topology.print_tree_summary()
        
        print("\nSaving visualization...")
        os.makedirs('output', exist_ok=True)
        topology.visualize_3d(save_path='output/dt_centerline_3d.png')
        print("Saved output/dt_centerline_3d.png")
    
    return topology


def run_fast_marching_example():
    print("\n" + "="*60)
    print("Running Fast Marching (Minimal Path) Centerline Extraction")
    print("="*60)
    
    print("Generating synthetic vessel image...")
    image = generate_synthetic_vessel_image(shape=(80, 80, 80))
    print(f"Image shape: {image.shape}")
    
    print("\nPreprocessing and segmentation...")
    preprocessor = VesselPreprocessor(spacing=(1.0, 1.0, 1.0))
    results = preprocessor.full_pipeline(image, vesselness=False, threshold=0.3)
    binary_mask = results['processed']
    print(f"Voxel count: {np.sum(binary_mask)}")
    
    print("\nExtracting centerline using fast marching method...")
    fm_centerline = FastMarchingCenterline(spacing=(1.0, 1.0, 1.0))
    fm_results = fm_centerline.extract(image, binary_mask)
    
    centerline_points = fm_results['centerline_points']
    print(f"Centerline points: {len(centerline_points)}")
    
    if len(centerline_points) > 0:
        print("\nBuilding topology tree...")
        from scipy import ndimage
        dt = ndimage.distance_transform_edt(binary_mask)
        
        topology = VesselTopologyTree(spacing=(1.0, 1.0, 1.0))
        topology.build_from_centerline_points(centerline_points, binary_mask, dt)
        topology.print_tree_summary()
        
        print("\nSaving visualization...")
        os.makedirs('output', exist_ok=True)
        topology.visualize_3d(save_path='output/fm_centerline_3d.png')
        print("Saved output/fm_centerline_3d.png")
        
        topology.export_to_vtk('output/centerline.vtk')
    
    return topology


def run_full_pipeline(image_path=None, method='both'):
    if image_path is None:
        print("No image path provided, using synthetic data.")
        image = generate_synthetic_vessel_image(shape=(100, 100, 100))
        spacing = (1.0, 1.0, 1.0)
    else:
        print(f"Loading image from {image_path}...")
        import SimpleITK as sitk
        sitk_image = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(sitk_image)
        spacing = sitk_image.GetSpacing()[::-1]
        print(f"Image shape: {image.shape}, spacing: {spacing}")
    
    print("\nStep 1: Preprocessing and Vessel Segmentation")
    preprocessor = VesselPreprocessor(spacing=spacing)
    results = preprocessor.full_pipeline(image, vesselness=False, threshold=0.3)
    binary_mask = results['processed']
    print(f"Segmented vessel voxels: {np.sum(binary_mask)}")
    
    all_topologies = {}
    
    if method in ['distance', 'both']:
        print("\nStep 2a: Centerline Extraction - Distance Transform Method")
        dt_centerline = DistanceTransformCenterline(spacing=spacing)
        dt_results = dt_centerline.extract(binary_mask, method='ridge')
        
        if len(dt_results['centerline_points']) > 0:
            topology_dt = VesselTopologyTree(spacing=spacing)
            topology_dt.build_from_centerline_points(
                dt_results['centerline_points'],
                binary_mask,
                dt_results['distance_transform']
            )
            all_topologies['distance_transform'] = topology_dt
            print("Distance transform topology tree built.")
    
    if method in ['fastmarching', 'both']:
        print("\nStep 2b: Centerline Extraction - Fast Marching Method")
        fm_centerline = FastMarchingCenterline(spacing=spacing)
        fm_results = fm_centerline.extract(image, binary_mask)
        
        if len(fm_results['centerline_points']) > 0:
            from scipy import ndimage
            dt = ndimage.distance_transform_edt(binary_mask, sampling=spacing)
            
            topology_fm = VesselTopologyTree(spacing=spacing)
            topology_fm.build_from_centerline_points(
                fm_results['centerline_points'],
                binary_mask,
                dt
            )
            all_topologies['fast_marching'] = topology_fm
            print("Fast marching topology tree built.")
    
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    
    for name, topology in all_topologies.items():
        print(f"\n--- {name} Results ---")
        topology.print_tree_summary()
    
    return all_topologies


def main():
    parser = argparse.ArgumentParser(
        description='Vessel Centerline Extraction and Topology Tree Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --example distance
  python main.py --example fastmarching
  python main.py --image path/to/image.nii.gz --method both
        """
    )
    
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image (NIfTI, DICOM, etc.)')
    parser.add_argument('--method', type=str, default='both',
                        choices=['distance', 'fastmarching', 'both'],
                        help='Centerline extraction method')
    parser.add_argument('--example', type=str, default=None,
                        choices=['distance', 'fastmarching', 'all'],
                        help='Run example with synthetic data')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory')
    
    args = parser.parse_args()
    
    if args.example:
        if args.example == 'distance':
            run_distance_transform_example()
        elif args.example == 'fastmarching':
            run_fast_marching_example()
        elif args.example == 'all':
            run_distance_transform_example()
            run_fast_marching_example()
    else:
        run_full_pipeline(args.image, args.method)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
