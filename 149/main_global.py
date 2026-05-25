import numpy as np
import argparse
import os
import sys
from preprocessing import VesselPreprocessor, generate_synthetic_vessel_image
from global_centerline_extractor import GlobalCenterlineExtractor
from topology_tree import VesselTopologyTree
from bifurcation_analyzer import BifurcationAnalyzer


def run_skeleton_method_example():
    print("\n" + "="*70)
    print("GLOBAL CENTERLINE EXTRACTION - Topological Skeleton Method")
    print("="*70)
    
    print("\nGenerating synthetic vessel image...")
    image = generate_synthetic_vessel_image(shape=(80, 80, 80))
    print(f"Image shape: {image.shape}")
    
    print("\nPreprocessing and segmentation...")
    preprocessor = VesselPreprocessor(spacing=(1.0, 1.0, 1.0))
    results = preprocessor.full_pipeline(image, vesselness=False, threshold=0.3)
    binary_mask = results['processed']
    print(f"Vessel voxel count: {np.sum(binary_mask)}")
    
    print("\nExtracting complete vessel tree using topological skeleton...")
    extractor = GlobalCenterlineExtractor(spacing=(1.0, 1.0, 1.0))
    cl_results = extractor.extract_complete_vessel_tree(
        binary_mask, 
        method='skeleton', 
        min_branch_length=8
    )
    
    centerline_points = cl_results['centerline_points']
    print(f"Centerline points: {len(centerline_points)}")
    
    if len(centerline_points) > 0:
        print("\nBuilding topology tree...")
        from scipy import ndimage
        dt = ndimage.distance_transform_edt(binary_mask)
        
        topology = VesselTopologyTree(spacing=(1.0, 1.0, 1.0))
        topology.build_from_centerline_points(
            centerline_points, 
            binary_mask, 
            dt
        )
        topology.print_tree_summary()
        
        print("\nSaving visualization...")
        os.makedirs('output', exist_ok=True)
        topology.visualize_3d(
            binary_image=binary_mask,
            save_path='output/skeleton_centerline_3d.png'
        )
        print("Saved output/skeleton_centerline_3d.png")
        
        topology.visualize_tree_graph(
            save_path='output/skeleton_tree_graph.png'
        )
        print("Saved output/skeleton_tree_graph.png")
        
        topology.export_to_vtk('output/skeleton_centerline.vtk')
        topology.export_to_graphml('output/skeleton_tree.graphml')
        
        print("\nStep 5: Bifurcation Analysis")
        analyzer = BifurcationAnalyzer(spacing=(1.0, 1.0, 1.0))
        analyzer.comprehensive_bifurcation_analysis(topology)
        
        analyzer.visualize_bifurcations_3d(
            topology,
            save_path='output/bifurcations_3d.png'
        )
        analyzer.generate_bifurcation_report(
            topology,
            save_path='output/bifurcation_report.txt'
        )
        analyzer.export_bifurcations_to_csv('output/bifurcations.csv')
    
    return topology, cl_results


def run_power_watershed_example():
    print("\n" + "="*70)
    print("GLOBAL CENTERLINE EXTRACTION - Power Watershed Method")
    print("="*70)
    
    print("\nGenerating synthetic vessel image...")
    image = generate_synthetic_vessel_image(shape=(80, 80, 80))
    print(f"Image shape: {image.shape}")
    
    print("\nPreprocessing and segmentation...")
    preprocessor = VesselPreprocessor(spacing=(1.0, 1.0, 1.0))
    results = preprocessor.full_pipeline(image, vesselness=False, threshold=0.3)
    binary_mask = results['processed']
    print(f"Vessel voxel count: {np.sum(binary_mask)}")
    
    print("\nExtracting complete vessel tree using power watershed...")
    extractor = GlobalCenterlineExtractor(spacing=(1.0, 1.0, 1.0))
    cl_results = extractor.extract_complete_vessel_tree(
        binary_mask, 
        method='power_watershed',
        num_sources=8,
        min_branch_length=5
    )
    
    centerline_points = cl_results['centerline_points']
    print(f"Centerline points: {len(centerline_points)}")
    print(f"Seed points used: {len(cl_results['seed_points'])}")
    
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
        topology.visualize_3d(
            binary_image=binary_mask,
            save_path='output/pw_centerline_3d.png'
        )
        print("Saved output/pw_centerline_3d.png")
        
        topology.export_to_vtk('output/pw_centerline.vtk')
    
    return topology, cl_results


def run_hybrid_example():
    print("\n" + "="*70)
    print("GLOBAL CENTERLINE EXTRACTION - Hybrid Method")
    print("="*70)
    
    print("\nGenerating synthetic vessel image...")
    image = generate_synthetic_vessel_image(shape=(80, 80, 80))
    print(f"Image shape: {image.shape}")
    
    print("\nPreprocessing and segmentation...")
    preprocessor = VesselPreprocessor(spacing=(1.0, 1.0, 1.0))
    results = preprocessor.full_pipeline(image, vesselness=False, threshold=0.3)
    binary_mask = results['processed']
    print(f"Vessel voxel count: {np.sum(binary_mask)}")
    
    print("\nExtracting complete vessel tree using hybrid method...")
    extractor = GlobalCenterlineExtractor(spacing=(1.0, 1.0, 1.0))
    cl_results = extractor.extract_complete_vessel_tree(
        binary_mask, 
        method='hybrid',
        min_branch_length=8,
        num_sources=8
    )
    
    centerline_points = cl_results['centerline_points']
    print(f"Centerline points: {len(centerline_points)}")
    
    if len(centerline_points) > 0:
        print("\nBuilding topology tree...")
        from scipy import ndimage
        dt = ndimage.distance_transform_edt(binary_mask)
        
        topology = VesselTopologyTree(spacing=(1.0, 1.0, 1.0))
        topology.build_from_centerline_points(
            centerline_points, 
            binary_mask, 
            dt
        )
        topology.print_tree_summary()
        
        print("\nSaving visualization...")
        os.makedirs('output', exist_ok=True)
        topology.visualize_3d(
            binary_image=binary_mask,
            save_path='output/hybrid_centerline_3d.png'
        )
        print("Saved output/hybrid_centerline_3d.png")
        
        topology.export_to_vtk('output/hybrid_centerline.vtk')
    
    return topology, cl_results


def compare_methods():
    print("\n" + "="*70)
    print("METHOD COMPARISON")
    print("="*70)
    
    print("\nGenerating synthetic vessel image...")
    image = generate_synthetic_vessel_image(shape=(80, 80, 80))
    
    preprocessor = VesselPreprocessor(spacing=(1.0, 1.0, 1.0))
    results = preprocessor.full_pipeline(image, vesselness=False, threshold=0.3)
    binary_mask = results['processed']
    
    extractor = GlobalCenterlineExtractor(spacing=(1.0, 1.0, 1.0))
    methods = ['skeleton', 'power_watershed', 'hybrid']
    all_results = {}
    
    for method in methods:
        print(f"\n--- Running {method} method ---")
        cl_results = extractor.extract_complete_vessel_tree(
            binary_mask, 
            method=method,
            min_branch_length=8,
            num_sources=8
        )
        all_results[method] = cl_results
        
        from scipy import ndimage
        dt = ndimage.distance_transform_edt(binary_mask)
        
        topology = VesselTopologyTree(spacing=(1.0, 1.0, 1.0))
        if len(cl_results['centerline_points']) > 0:
            topology.build_from_centerline_points(
                cl_results['centerline_points'], 
                binary_mask, 
                dt
            )
            stats = topology.get_tree_statistics()
            print(f"  Points: {len(cl_results['centerline_points'])}")
            print(f"  Junctions: {stats['num_junctions']}")
            print(f"  Endpoints: {stats['num_endpoints']}")
            print(f"  Segments: {stats['num_segments']}")
            print(f"  Total length: {stats['total_vessel_length']:.2f}")
    
    return all_results


def run_full_pipeline(image_path=None, method='skeleton'):
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
    
    print(f"\nStep 2: Global Centerline Extraction ({method})")
    extractor = GlobalCenterlineExtractor(spacing=spacing)
    cl_results = extractor.extract_complete_vessel_tree(
        binary_mask, 
        method=method,
        min_branch_length=8,
        num_sources=10
    )
    
    centerline_points = cl_results['centerline_points']
    print(f"Centerline points extracted: {len(centerline_points)}")
    
    if len(centerline_points) > 0:
        print("\nStep 3: Building Topology Tree")
        from scipy import ndimage
        dt = ndimage.distance_transform_edt(binary_mask, sampling=spacing)
        
        topology = VesselTopologyTree(spacing=spacing)
        topology.build_from_centerline_points(
            centerline_points, 
            binary_mask, 
            dt
        )
        topology.print_tree_summary()
        
        print("\nStep 4: Bifurcation Analysis")
        analyzer = BifurcationAnalyzer(spacing=spacing)
        analyzer.comprehensive_bifurcation_analysis(topology)
        
        print("\nStep 5: Exporting Results")
        os.makedirs('output', exist_ok=True)
        topology.visualize_3d(
            binary_image=binary_mask,
            save_path='output/centerline_3d.png'
        )
        topology.visualize_tree_graph(
            save_path='output/tree_graph.png'
        )
        topology.export_to_vtk('output/centerline.vtk')
        topology.export_to_graphml('output/topology_tree.graphml')
        
        analyzer.visualize_bifurcations_3d(
            topology,
            save_path='output/bifurcations_3d.png'
        )
        analyzer.generate_bifurcation_report(
            topology,
            save_path='output/bifurcation_report.txt'
        )
        analyzer.export_bifurcations_to_csv('output/bifurcations.csv')
        
        return topology, cl_results, analyzer
    
    return None, cl_results


def main():
    parser = argparse.ArgumentParser(
        description='Global Vessel Centerline Extraction and Topology Tree',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_global.py --method skeleton          # Topological skeleton (recommended)
  python main_global.py --method power_watershed   # Multi-source FMM gradient tree
  python main_global.py --method hybrid            # Combined method
  python main_global.py --compare                  # Compare all methods
  python main_global.py --image path/to/image.nii.gz --method skeleton
        """
    )
    
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image (NIfTI, DICOM, etc.)')
    parser.add_argument('--method', type=str, default='skeleton',
                        choices=['skeleton', 'power_watershed', 'hybrid'],
                        help='Global centerline extraction method')
    parser.add_argument('--compare', action='store_true',
                        help='Compare all extraction methods')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_methods()
    else:
        run_full_pipeline(args.image, args.method)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
