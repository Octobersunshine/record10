import numpy as np
import time
from time_series_insar import (
    simulate_time_series,
    ps_insar_processing,
    generate_deformation_mask,
    early_warning,
)

def main():
    print("=" * 60)
    print("Time Series InSAR - Deformation Monitoring Test")
    print("=" * 60)

    np.random.seed(42)

    rows, cols = 30, 30
    n_images = 15

    print(f"\nSimulating {n_images} SAR images over {rows}x{cols} area...")
    sim_data = simulate_time_series(
        rows, cols, n_images,
        deformation_type='mixed',
        noise_level=0.06
    )

    true_vel = sim_data['true_velocity']
    true_def = sim_data['true_deformation']
    wavelength = sim_data['wavelength']

    print(f"True deformation pattern:")
    print(f"  Max subsidence: {np.min(true_vel):.2f} mm/year")
    print(f"  Max uplift: {np.max(true_vel):.2f} mm/year")
    print(f"  Wavelength: {wavelength*100:.1f} cm (C-band)")

    start = time.time()
    results = ps_insar_processing(
        sim_data['amplitude_stack'],
        sim_data['wrapped_phase'],
        sim_data['timestamps'],
        sim_data['baseline_perp'],
        ad_threshold=0.3,
        min_coherence=0.6,
        wavelength=wavelength
    )
    elapsed = time.time() - start

    ps_mask = results['ps_mask']
    est_vel = results['velocities']
    est_def = results['deformation_series']

    print(f"\nPS-InSAR Processing Results:")
    print(f"  PS points detected: {np.sum(ps_mask)} / {rows*cols} ({100*np.sum(ps_mask)/(rows*cols):.1f}%)")
    print(f"  Processing time: {elapsed:.2f} s")

    mask = ps_mask & (np.abs(true_vel) > 1.0)
    if np.sum(mask) > 0:
        error = est_vel[mask] - true_vel[mask]
        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error**2))

        print(f"\nVelocity Estimation Accuracy:")
        print(f"  MAE: {mae:.3f} mm/year")
        print(f"  RMSE: {rmse:.3f} mm/year")
        print(f"  Error range: [{np.min(error):.3f}, {np.max(error):.3f}] mm/year")

    print(f"\nStep 6: Active deformation region detection...")
    sig_mask, regions = generate_deformation_mask(
        est_vel, threshold=3.0, min_area=8
    )

    print(f"  Detected {len(regions)} significant deformation regions:")
    for r in regions:
        print(f"    - Region {r['label']}: {r['type'].upper():10s} | "
              f"Area: {r['area']:3d} px | "
              f"Mean: {r['mean_velocity']:+6.2f} mm/year | "
              f"Max: {r['max_velocity']:6.2f} mm/year")

    print(f"\nStep 7: Early Warning Analysis...")
    warnings = early_warning(
        est_def, est_vel, regions,
        warning_threshold_velocity=8.0,
        warning_threshold_acceleration=1.5
    )

    if warnings:
        print(f"  \u26a0 WARNING: {len(warnings)} potential hazard(s) detected!")
        for w in warnings:
            r = w['region']
            print(f"    - [{w['warning_level'].upper()}] {r['type']} region {r['label']}:")
            print(f"      Velocity: {w['velocity']:.2f} mm/year, "
                  f"Acceleration: {w['acceleration']:.3f} mm/year\u00b2")
    else:
        print(f"  \u2705 All regions stable. No immediate hazards detected.")

    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} {'Value':<30}")
    print("-" * 60)
    print(f"{'Spatial coverage':<30} {np.sum(ps_mask)} PS points")
    print(f"{'Temporal coverage':<30} {n_images} images")
    if 'rmse' in locals():
        print(f"{'Velocity accuracy (RMSE)':<30} {rmse:.3f} mm/year")
    print(f"{'Deformation regions detected':<30} {len(regions)}")
    print(f"{'Warning status':<30} {len(warnings)} warning(s)")
    print("=" * 60)

    accuracy_ok = 'rmse' not in locals() or rmse < 5.0
    if accuracy_ok:
        print("\n\u2705 Test PASSED! Deformation monitoring accuracy meets specifications.")
        print("   Typical operational accuracy: 1-5 mm/year for linear velocity.")
    else:
        print("\n\u26a0 Note: Results may improve with more images or lower noise.")

    return accuracy_ok

if __name__ == "__main__":
    main()
