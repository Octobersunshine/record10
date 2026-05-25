import numpy as np
import cv2
import time
from deep_optical_flow import DeepOpticalFlow, SubpixelRefinement, flow_to_color


def create_synthetic_motion_subpixel():
    print("Creating synthetic motion test with subpixel precision...")
    
    size = (200, 200)
    img1 = np.zeros(size, dtype=np.uint8)
    
    cv2.rectangle(img1, (30, 30), (80, 170), 255, -1)
    cv2.circle(img1, (140, 100), 40, 200, -1)
    cv2.line(img1, (20, 20), (180, 180), 150, 3)
    
    dx, dy = 4.7, 3.2
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img2 = cv2.warpAffine(img1, M, size[::-1])
    
    return img1, img2, dx, dy


def create_color_test_frames():
    print("Creating color test frames...")
    
    size = (256, 256)
    img1 = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    
    cv2.rectangle(img1, (40, 40), (100, 200), (255, 100, 50), -1)
    cv2.circle(img1, (180, 120), 50, (50, 200, 255), -1)
    cv2.rectangle(img1, (100, 80), (160, 140), (100, 255, 100), -1)
    
    dx, dy = 5.5, -3.8
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img2 = cv2.warpAffine(img1, M, size[::-1])
    
    return img1, img2, dx, dy


def test_deep_flow_raft():
    print("\n" + "=" * 60)
    print("Testing Deep Optical Flow - RAFT model")
    print("=" * 60)
    
    img1, img2, true_dx, true_dy = create_color_test_frames()
    
    calculator = DeepOpticalFlow(model_name='raft', device='auto')
    
    start_time = time.time()
    u, v = calculator.calculate(img1, img2)
    elapsed = time.time() - start_time
    
    mask = np.any(img1 > 50, axis=2) if len(img1.shape) == 3 else img1 > 50
    
    avg_u = np.mean(u[mask])
    avg_v = np.mean(v[mask])
    std_u = np.std(u[mask])
    std_v = np.std(v[mask])
    
    print(f"\nTrue motion: dx={true_dx:.3f}, dy={true_dy:.3f}")
    print(f"Estimated:   avg_u={avg_u:.3f}, avg_v={avg_v:.3f}")
    print(f"Error:       {np.abs(avg_u - true_dx):.4f}, {np.abs(avg_v - true_dy):.3f}")
    print(f"Std dev:     {std_u:.4f}, {std_v:.4f}")
    print(f"Time:        {elapsed:.3f} seconds")
    print(f"Flow shape:  {u.shape}")
    
    color_flow = flow_to_color(u, v)
    cv2.imwrite('test_deep_flow_raft.png', color_flow)
    cv2.imwrite('test_deep_frame1.png', img1)
    cv2.imwrite('test_deep_frame2.png', img2)
    print("Saved: test_deep_flow_raft.png, test_deep_frame1.png, test_deep_frame2.png")
    
    return u, v


def test_subpixel_refinement():
    print("\n" + "=" * 60)
    print("Testing Subpixel Refinement")
    print("=" * 60)
    
    img1, img2, true_dx, true_dy = create_synthetic_motion_subpixel()
    
    calculator = DeepOpticalFlow(model_name='raft', device='auto')
    u_initial, v_initial = calculator.calculate(img1, img2)
    
    start_time = time.time()
    u_refined, v_refined = SubpixelRefinement.refine_flow(
        u_initial, v_initial, img1, img2,
        iterations=3, window_size=5
    )
    elapsed = time.time() - start_time
    
    mask = img1 > 50
    
    print(f"\nTrue motion: dx={true_dx:.3f}, dy={true_dy:.3f}")
    
    print(f"\nBefore refinement:")
    print(f"  avg_u={np.mean(u_initial[mask]):.4f}, avg_v={np.mean(v_initial[mask]):.4f}")
    print(f"  error={np.abs(np.mean(u_initial[mask]) - true_dx):.4f}, {np.abs(np.mean(v_initial[mask]) - true_dy):.4f}")
    
    print(f"\nAfter refinement ({elapsed:.3f}s):")
    print(f"  avg_u={np.mean(u_refined[mask]):.4f}, avg_v={np.mean(v_refined[mask]):.4f}")
    print(f"  error={np.abs(np.mean(u_refined[mask]) - true_dx):.4f}, {np.abs(np.mean(v_refined[mask]) - true_dy):.4f}")
    
    improvement_u = (np.mean(np.abs(u_initial[mask] - true_dx)) - np.mean(np.abs(u_refined[mask] - true_dx))) / np.mean(np.abs(u_initial[mask] - true_dx)) * 100
    improvement_v = (np.mean(np.abs(v_initial[mask] - true_dy)) - np.mean(np.abs(v_refined[mask] - true_dy))) / np.mean(np.abs(v_initial[mask] - true_dy)) * 100
    print(f"\nImprovement: u={improvement_u:.1f}%, v={improvement_v:.1f}%")
    
    color_flow_initial = flow_to_color(u_initial, v_initial)
    color_flow_refined = flow_to_color(u_refined, v_refined)
    
    cv2.imwrite('test_flow_initial.png', color_flow_initial)
    cv2.imwrite('test_flow_refined.png', color_flow_refined)
    cv2.imwrite('test_subpixel_frame1.png', img1)
    print("Saved: test_flow_initial.png, test_flow_refined.png, test_subpixel_frame1.png")
    
    return u_initial, v_initial, u_refined, v_refined


def test_batch_processing():
    print("\n" + "=" * 60)
    print("Testing Batch Processing")
    print("=" * 60)
    
    batch_size = 3
    frame_pairs = []
    
    for i in range(batch_size):
        img1, img2, dx, dy = create_color_test_frames()
        frame_pairs.append((img1, img2))
        print(f"Pair {i+1}: motion dx={dx:.1f}, dy={dy:.1f}")
    
    calculator = DeepOpticalFlow(model_name='raft', device='auto')
    
    start_time = time.time()
    results = calculator.calculate_batch(frame_pairs)
    elapsed = time.time() - start_time
    
    print(f"\nBatch completed in {elapsed:.3f}s ({elapsed/batch_size:.3f}s per pair)")
    
    for i, (u, v) in enumerate(results):
        mask = np.any(frame_pairs[i][0] > 50, axis=2)
        print(f"Pair {i+1}: avg_u={np.mean(u[mask]):.3f}, avg_v={np.mean(v[mask]):.3f}")
    
    return results


def compare_with_traditional():
    print("\n" + "=" * 60)
    print("Comparison: Deep Learning vs Traditional Methods")
    print("=" * 60)
    
    img1, img2, true_dx, true_dy = create_color_test_frames()
    
    print("\nRunning Deep Flow (RAFT)...")
    deep_calc = DeepOpticalFlow(model_name='raft', device='auto')
    start = time.time()
    u_deep, v_deep = deep_calc.calculate(img1, img2)
    time_deep = time.time() - start
    
    print("Running Traditional (Farneback)...")
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    start = time.time()
    flow_trad = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 0.5, 5, 15, 5, 5, 1.2, 0
    )
    u_trad, v_trad = flow_trad[..., 0], flow_trad[..., 1]
    time_trad = time.time() - start
    
    mask = np.any(img1 > 50, axis=2)
    
    print(f"\n{'Method':<15} {'avg_u':>10} {'avg_v':>10} {'error_u':>10} {'error_v':>10} {'time':>10}")
    print("-" * 65)
    
    err_deep_u = np.abs(np.mean(u_deep[mask]) - true_dx)
    err_deep_v = np.abs(np.mean(v_deep[mask]) - true_dy)
    err_trad_u = np.abs(np.mean(u_trad[mask]) - true_dx)
    err_trad_v = np.abs(np.mean(v_trad[mask]) - true_dy)
    
    print(f"{'Deep (RAFT)':<15} {np.mean(u_deep[mask]):>10.3f} {np.mean(v_deep[mask]):>10.3f} {err_deep_u:>10.4f} {err_deep_v:>10.4f} {time_deep:>10.3f}s")
    print(f"{'Traditional':<15} {np.mean(u_trad[mask]):>10.3f} {np.mean(v_trad[mask]):>10.3f} {err_trad_u:>10.4f} {err_trad_v:>10.4f} {time_trad:>10.3f}s")
    
    color_deep = flow_to_color(u_deep, v_deep)
    color_trad = flow_to_color(u_trad, v_trad)
    
    cv2.imwrite('test_compare_deep.png', color_deep)
    cv2.imwrite('test_compare_traditional.png', color_trad)
    print("\nSaved: test_compare_deep.png, test_compare_traditional.png")
    
    return (u_deep, v_deep), (u_trad, v_trad)


def main():
    print("Deep Optical Flow Test Suite")
    print("=" * 60)
    print("Testing RAFT-based deep optical flow with:")
    print("  - Subpixel accuracy")
    print("  - GPU acceleration (if available)")
    print("  - Subpixel refinement module")
    print()
    
    try:
        test_deep_flow_raft()
        test_subpixel_refinement()
        test_batch_processing()
        compare_with_traditional()
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("\nGenerated output images:")
        print("  - test_deep_flow_raft.png: RAFT optical flow")
        print("  - test_flow_initial.png: Before refinement")
        print("  - test_flow_refined.png: After subpixel refinement")
        print("  - test_compare_deep.png: Deep learning result")
        print("  - test_compare_traditional.png: Traditional result")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
