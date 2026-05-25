import numpy as np
import cv2
from dense_optical_flow import OpticalFlowCalculator, flow_to_color


def create_synthetic_motion_with_edge():
    print("Creating synthetic motion test with sharp edges...")
    
    size = (120, 120)
    img1 = np.zeros(size, dtype=np.uint8)
    
    cv2.rectangle(img1, (20, 20), (55, 100), 255, -1)
    cv2.circle(img1, (85, 60), 25, 200, -1)
    cv2.rectangle(img1, (40, 40), (80, 80), 150, -1)
    
    dx, dy = 6, 4
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img2 = cv2.warpAffine(img1, M, size[::-1])
    
    return img1, img2, dx, dy


def test_horn_schunck_basic():
    print("\n=== Testing Original Horn-Schunck (Isotropic) ===")
    
    img1, img2, true_dx, true_dy = create_synthetic_motion_with_edge()
    
    calculator = OpticalFlowCalculator(
        method='horn_schunck', 
        alpha=1.0, 
        iterations=200,
        anisotropic=False
    )
    
    u, v = calculator.calculate(img1, img2)
    
    edge_mask = cv2.Canny(img1, 50, 150) > 0
    inner_mask = (img1 > 128) & (~edge_mask)
    
    avg_u_edge = np.mean(u[edge_mask])
    avg_v_edge = np.mean(v[edge_mask])
    avg_u_inner = np.mean(u[inner_mask])
    avg_v_inner = np.mean(v[inner_mask])
    
    print(f"True motion: dx={true_dx}, dy={true_dy}")
    print(f"Estimated (edge regions):  avg_u={avg_u_edge:.3f}, avg_v={avg_v_edge:.3f}")
    print(f"Estimated (inner regions): avg_u={avg_u_inner:.3f}, avg_v={avg_v_inner:.3f}")
    print(f"Edge error: {np.abs(avg_u_edge - true_dx):.3f}, {np.abs(avg_v_edge - true_dy):.3f}")
    
    color_flow = flow_to_color(u, v)
    cv2.imwrite('test_flow_hs_isotropic.png', color_flow)
    cv2.imwrite('test_edge_mask.png', edge_mask.astype(np.uint8) * 255)
    print("Saved: test_flow_hs_isotropic.png, test_edge_mask.png")
    
    return u, v, edge_mask


def test_horn_schunck_anisotropic():
    print("\n=== Testing Improved Horn-Schunck (Anisotropic) ===")
    
    img1, img2, true_dx, true_dy = create_synthetic_motion_with_edge()
    
    calculator = OpticalFlowCalculator(
        method='horn_schunck', 
        alpha=1.0, 
        iterations=200,
        anisotropic=True,
        edge_threshold=0.05,
        edge_sensitivity=2.0
    )
    
    u, v = calculator.calculate(img1, img2)
    
    edge_mask = cv2.Canny(img1, 50, 150) > 0
    inner_mask = (img1 > 128) & (~edge_mask)
    
    avg_u_edge = np.mean(u[edge_mask])
    avg_v_edge = np.mean(v[edge_mask])
    avg_u_inner = np.mean(u[inner_mask])
    avg_v_inner = np.mean(v[inner_mask])
    
    print(f"True motion: dx={true_dx}, dy={true_dy}")
    print(f"Estimated (edge regions):  avg_u={avg_u_edge:.3f}, avg_v={avg_v_edge:.3f}")
    print(f"Estimated (inner regions): avg_u={avg_u_inner:.3f}, avg_v={avg_v_inner:.3f}")
    print(f"Edge error: {np.abs(avg_u_edge - true_dx):.3f}, {np.abs(avg_v_edge - true_dy):.3f}")
    
    color_flow = flow_to_color(u, v)
    cv2.imwrite('test_flow_hs_anisotropic.png', color_flow)
    
    edge_weights = calculator._compute_edge_weights(
        img1.astype(np.float32) / 255.0, 
        img2.astype(np.float32) / 255.0
    )
    cv2.imwrite('test_edge_weights.png', (edge_weights * 255).astype(np.uint8))
    print("Saved: test_flow_hs_anisotropic.png, test_edge_weights.png")
    
    return u, v, edge_mask


def compare_methods():
    print("\n" + "=" * 60)
    print("COMPARISON: Isotropic vs Anisotropic Smoothing")
    print("=" * 60)
    
    img1, img2, true_dx, true_dy = create_synthetic_motion_with_edge()
    edge_mask = cv2.Canny(img1, 50, 150) > 0
    
    calc_iso = OpticalFlowCalculator(method='horn_schunck', alpha=1.0, iterations=200, anisotropic=False)
    u_iso, v_iso = calc_iso.calculate(img1, img2)
    
    calc_aniso = OpticalFlowCalculator(method='horn_schunck', alpha=1.0, iterations=200, anisotropic=True)
    u_aniso, v_aniso = calc_aniso.calculate(img1, img2)
    
    u_error_iso = np.abs(u_iso[edge_mask] - true_dx)
    v_error_iso = np.abs(v_iso[edge_mask] - true_dy)
    u_error_aniso = np.abs(u_aniso[edge_mask] - true_dx)
    v_error_aniso = np.abs(v_aniso[edge_mask] - true_dy)
    
    print("\nEdge Region Error Analysis:")
    print(f"  Isotropic:   u_error_mean={np.mean(u_error_iso):.4f}, u_error_max={np.max(u_error_iso):.4f}")
    print(f"               v_error_mean={np.mean(v_error_iso):.4f}, v_error_max={np.max(v_error_iso):.4f}")
    print(f"  Anisotropic: u_error_mean={np.mean(u_error_aniso):.4f}, u_error_max={np.max(u_error_aniso):.4f}")
    print(f"               v_error_mean={np.mean(v_error_aniso):.4f}, v_error_max={np.max(v_error_aniso):.4f}")
    
    u_improvement = (np.mean(u_error_iso) - np.mean(u_error_aniso)) / np.mean(u_error_iso) * 100
    v_improvement = (np.mean(v_error_iso) - np.mean(v_error_aniso)) / np.mean(v_error_iso) * 100
    
    print(f"\nImprovement at edges: u={u_improvement:.1f}%, v={v_improvement:.1f}%")
    
    flow_diff_u = np.abs(u_iso - u_aniso)
    flow_diff_v = np.abs(v_iso - v_aniso)
    cv2.imwrite('test_flow_diff_u.png', cv2.normalize(flow_diff_u, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    cv2.imwrite('test_flow_diff_v.png', cv2.normalize(flow_diff_v, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    
    print("\nKey differences between methods:")
    print("  - Isotropic:  Uniform smoothing everywhere, blurs motion at edges")
    print("  - Anisotropic: Edge-aware smoothing, preserves motion boundaries")
    print("  - Edge weights reduce smoothing strength near image gradients")
    
    return u_iso, v_iso, u_aniso, v_aniso


def test_lucas_kanade():
    print("\n=== Testing Lucas-Kanade Method ===")
    
    img1, img2, true_dx, true_dy = create_synthetic_motion_with_edge()
    
    calculator = OpticalFlowCalculator(
        method='lucas_kanade', 
        window_size=15
    )
    
    u, v = calculator.calculate(img1, img2)
    
    mask = img1 > 128
    avg_u = np.mean(u[mask])
    avg_v = np.mean(v[mask])
    
    print(f"True motion: dx={true_dx}, dy={true_dy}")
    print(f"Estimated motion (on objects): avg_u={avg_u:.3f}, avg_v={avg_v:.3f}")
    
    color_flow = flow_to_color(u, v)
    cv2.imwrite('test_flow_lk.png', color_flow)
    print("Saved: test_flow_lk.png")
    
    return u, v


def main():
    print("Optical Flow Implementation Tests")
    print("=" * 60)
    print("Testing improved Horn-Schunck with anisotropic smoothing")
    print()
    
    try:
        cv2.imwrite('test_frame1.png', create_synthetic_motion_with_edge()[0])
        cv2.imwrite('test_frame2.png', create_synthetic_motion_with_edge()[1])
        print("Saved test frames: test_frame1.png, test_frame2.png")
        
        test_horn_schunck_basic()
        test_horn_schunck_anisotropic()
        compare_methods()
        test_lucas_kanade()
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("\nGenerated images:")
        print("  - test_frame1.png, test_frame2.png: Input frames")
        print("  - test_flow_hs_isotropic.png: Original Horn-Schunck")
        print("  - test_flow_hs_anisotropic.png: Improved (edge-preserving)")
        print("  - test_edge_weights.png: Edge detection weights")
        print("  - test_flow_lk.png: Lucas-Kanade method")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
