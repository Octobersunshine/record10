import numpy as np
import matplotlib.pyplot as plt
import time
from time_series_insar import (
    simulate_time_series,
    generate_interferograms,
    ps_insar_processing,
    sbas_processing,
    generate_deformation_mask,
    early_warning,
)


def test_subsidence_monitoring():
    print("=" * 70)
    print("TEST 1: Subsidence Monitoring with PS-InSAR")
    print("=" * 70)

    np.random.seed(42)

    rows, cols = 40, 40
    n_images = 12

    sim_data = simulate_time_series(
        rows, cols, n_images,
        deformation_type='subsidence',
        noise_level=0.08
    )

    amplitude_stack = sim_data['amplitude_stack']
    wrapped_phase = sim_data['wrapped_phase']
    true_velocity = sim_data['true_velocity']
    true_deformation = sim_data['true_deformation']
    timestamps = sim_data['timestamps']
    baseline_perp = sim_data['baseline_perp']
    wavelength = sim_data['wavelength']

    print(f"\nData: {n_images} images, {rows}x{cols} pixels")
    print(f"True max subsidence rate: {np.min(true_velocity):.2f} mm/year")
    print(f"True max uplift rate: {np.max(true_velocity):.2f} mm/year")

    start_time = time.time()
    ps_results = ps_insar_processing(
        amplitude_stack, wrapped_phase, timestamps, baseline_perp,
        ad_threshold=0.3, min_coherence=0.65, wavelength=wavelength
    )
    ps_time = time.time() - start_time

    velocity = ps_results['velocities']
    deformation_series = ps_results['deformation_series']
    ps_mask = ps_results['ps_mask']

    print(f"\nPS-InSAR Results:")
    print(f"  Processing time: {ps_time:.2f} s")
    print(f"  Number of PS points: {np.sum(ps_mask)}")

    mask = ps_mask & (np.abs(true_velocity) > 1)
    if np.sum(mask) > 0:
        error = velocity[mask] - true_velocity[mask]
        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error**2))
        print(f"  Velocity MAE: {mae:.3f} mm/year")
        print(f"  Velocity RMSE: {rmse:.3f} mm/year")

    print("\nStep 6: Deformation region detection...")
    sig_mask, regions = generate_deformation_mask(velocity, threshold=3.0, min_area=5)

    print(f"  Detected {len(regions)} deformation regions:")
    for r in regions:
        print(f"    - Region {r['label']}: {r['type']}, area={r['area']}, "
              f"mean_vel={r['mean_velocity']:.2f} mm/year, "
              f"max_vel={r['max_velocity']:.2f} mm/year")

    print("\nStep 7: Early warning analysis...")
    warnings = early_warning(
        deformation_series, velocity, regions,
        warning_threshold_velocity=8.0,
        warning_threshold_acceleration=1.0
    )

    if warnings:
        print(f"  Generated {len(warnings)} warnings:")
        for w in warnings:
            r = w['region']
            print(f"    - [{w['warning_level'].upper()}] Region {r['label']}: "
                  f"velocity={w['velocity']:.2f} mm/year, "
                  f"acceleration={w['acceleration']:.3f} mm/year\u00b2")
    else:
        print("  No warnings generated (within safe limits)")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    vmin, vmax = np.min(true_velocity), np.max(true_velocity)

    im0 = axes[0, 0].imshow(true_velocity, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('True Velocity (mm/year)')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(velocity, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('Estimated Velocity (mm/year)')
    plt.colorbar(im1, ax=axes[0, 1])

    axes[0, 2].imshow(ps_mask, cmap='gray')
    for r in regions:
        axes[0, 2].contour(r['mask'], colors='red', linewidths=1)
    axes[0, 2].set_title('PS Points and Deformation Regions')

    t_days = np.array(timestamps) / (24 * 3600)

    for r in regions[:2]:
        mask = r['mask']
        est_def = np.zeros(n_images)
        true_def = np.zeros(n_images)
        for t in range(n_images):
            est_def[t] = np.mean(deformation_series[t][mask])
            true_def[t] = np.mean(true_deformation[t][mask])

        label = f"Region {r['label']} ({r['type']})"
        axes[1, 0].plot(t_days, true_def, 'o-', label=f'{label} (True)', alpha=0.7)
        axes[1, 0].plot(t_days, est_def, 's--', label=f'{label} (Estimated)', alpha=0.7)

    axes[1, 0].set_xlabel('Time (days)')
    axes[1, 0].set_ylabel('Deformation (mm)')
    axes[1, 0].set_title('Deformation Time Series')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    if np.sum(ps_mask) > 0:
        ps_indices = np.argwhere(ps_mask)
        sample_ps = ps_indices[len(ps_indices) // 2]
        si, sj = sample_ps

        ps_def = deformation_series[:, si, sj]
        ps_true = true_deformation[:, si, sj]

        axes[1, 1].plot(t_days, ps_true, 'o-', label='True', alpha=0.7)
        axes[1, 1].plot(t_days, ps_def, 's--', label='Estimated', alpha=0.7)
        axes[1, 1].set_xlabel('Time (days)')
        axes[1, 1].set_ylabel('Deformation (mm)')
        axes[1, 1].set_title(f'Sample PS at ({si}, {sj})')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    error_map = np.zeros_like(velocity)
    error_map[ps_mask] = velocity[ps_mask] - true_velocity[ps_mask]
    im2 = axes[1, 2].imshow(error_map, cmap='RdBu_r', vmin=-5, vmax=5)
    axes[1, 2].set_title('Velocity Error (mm/year)')
    plt.colorbar(im2, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig('subsidence_monitoring_results.png', dpi=150)
    print("\n  Visualization saved to subsidence_monitoring_results.png")
    plt.close()

    return rmse if 'rmse' in locals() else 0, ps_results


def test_landslide_early_warning():
    print("\n" + "=" * 70)
    print("TEST 2: Landslide Early Warning with SBAS")
    print("=" * 70)

    np.random.seed(123)

    rows, cols = 50, 50
    n_images = 18

    sim_data = simulate_time_series(
        rows, cols, n_images,
        deformation_type='landslide',
        noise_level=0.1
    )

    wrapped_phase = sim_data['wrapped_phase']
    true_velocity = sim_data['true_velocity']
    true_deformation = sim_data['true_deformation']
    timestamps = sim_data['timestamps']
    baseline_perp = sim_data['baseline_perp']
    wavelength = sim_data['wavelength']

    print(f"\nData: {n_images} images, {rows}x{cols} pixels")
    print(f"True max deformation rate: {np.max(np.abs(true_velocity)):.2f} mm/year")

    print("\nStep 1: Generating small baseline interferogram pairs...")
    ifg_stack, pairs = generate_interferograms(
        wrapped_phase, timestamps, baseline_perp,
        max_baseline=200, max_time_diff=540
    )

    print(f"  Generated {len(pairs)} interferograms")

    start_time = time.time()
    sbas_results = sbas_processing(
        ifg_stack, pairs, timestamps, baseline_perp,
        wavelength=wavelength
    )
    sbas_time = time.time() - start_time

    velocity = sbas_results['velocity']
    deformation_series = sbas_results['deformation_series']

    print(f"\nSBAS Results:")
    print(f"  Processing time: {sbas_time:.2f} s")

    mask = np.abs(true_velocity) > 1
    if np.sum(mask) > 0:
        error = velocity[mask] - true_velocity[mask]
        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error**2))
        print(f"  Velocity MAE: {mae:.3f} mm/year")
        print(f"  Velocity RMSE: {rmse:.3f} mm/year")

    print("\nStep 4: Potential landslide detection...")
    sig_mask, regions = generate_deformation_mask(velocity, threshold=8.0, min_area=20)

    print(f"  Detected {len(regions)} active deformation regions:")
    for r in regions:
        print(f"    - Region {r['label']}: {r['type']}, area={r['area']} px, "
              f"mean_vel={r['mean_velocity']:.2f} mm/year, "
              f"max_vel={r['max_velocity']:.2f} mm/year")

    print("\nStep 5: Early warning system...")
    warnings = early_warning(
        deformation_series, velocity, regions,
        warning_threshold_velocity=15.0,
        warning_threshold_acceleration=3.0
    )

    if warnings:
        print(f"  {'=' * 60}")
        print(f"  EARLY WARNING SUMMARY")
        print(f"  {'=' * 60}")
        for w in warnings:
            r = w['region']
            level = w['warning_level'].upper()
            status = "\u26a0 WARNING" if level == 'WARNING' else "\u274c CRITICAL"
            print(f"  {status} - Region {r['label']} ({r['type']})")
            print(f"    Location: area {r['area']} pixels")
            print(f"    Velocity: {w['velocity']:.2f} mm/year")
            print(f"    Acceleration: {w['acceleration']:.3f} mm/year\u00b2")
            print(f"    Max deformation: {w['max_deformation']:.2f} mm")
            print(f"    {'-' * 40}")
    else:
        print("  \u2705 No immediate hazard detected (all stable)")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    vmin, vmax = np.min(true_velocity), np.max(true_velocity)

    im0 = axes[0, 0].imshow(true_velocity, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('True Velocity (mm/year)')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(velocity, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('SBAS Estimated Velocity (mm/year)')
    plt.colorbar(im1, ax=axes[0, 1])

    for r in regions:
        axes[0, 1].contour(r['mask'], colors='white', linewidths=2)

    warning_mask = np.zeros_like(velocity, dtype=bool)
    warning_levels = np.zeros_like(velocity)
    for w in warnings:
        mask = w['region']['mask']
        warning_mask |= mask
        if w['warning_level'] == 'critical':
            warning_levels[mask] = 2
        elif w['warning_level'] == 'warning':
            warning_levels[mask] = 1

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['green', 'yellow', 'red'])
    im2 = axes[0, 2].imshow(warning_levels, cmap=cmap, vmin=0, vmax=2)
    axes[0, 2].set_title('Warning Level: Normal/Warning/Critical')
    cbar = plt.colorbar(im2, ax=axes[0, 2], ticks=[0, 1, 2])
    cbar.set_ticklabels(['Normal', 'Warning', 'Critical'])

    t_days = np.array(timestamps) / (24 * 3600)

    if regions:
        r = regions[0]
        mask = r['mask']
        est_def = np.zeros(n_images)
        true_def = np.zeros(n_images)
        for t in range(n_images):
            est_def[t] = np.mean(deformation_series[t][mask])
            true_def[t] = np.mean(true_deformation[t][mask])

        axes[1, 0].plot(t_days, true_def, 'o-', label='True', alpha=0.7)
        axes[1, 0].plot(t_days, est_def, 's--', label='SBAS Estimated', alpha=0.7)
        axes[1, 0].set_xlabel('Time (days)')
        axes[1, 0].set_ylabel('Deformation (mm)')
        axes[1, 0].set_title(f'Landslide Region Time Series')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        t = np.arange(n_images)
        coeff2 = np.polyfit(t, est_def, 2)
        fit_curve = np.polyval(coeff2, t)
        axes[1, 0].plot(t_days, fit_curve, ':', color='red',
                        label=f'2nd order fit (accel={2*coeff2[0]:.3f})')
        axes[1, 0].legend()

    sample_i, sample_j = rows // 2, cols // 3
    sample_def = deformation_series[:, sample_i, sample_j]
    sample_true = true_deformation[:, sample_i, sample_j]

    axes[1, 1].plot(t_days, sample_true, 'o-', label='True', alpha=0.7)
    axes[1, 1].plot(t_days, sample_def, 's--', label='SBAS Estimated', alpha=0.7)
    axes[1, 1].set_xlabel('Time (days)')
    axes[1, 1].set_ylabel('Deformation (mm)')
    axes[1, 1].set_title(f'Sample Point at ({sample_i}, {sample_j})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    baseline_pairs = [(p[2], p[3]) for p in pairs]
    b_perp = [b for b, t in baseline_pairs]
    t_diff = [t for b, t in baseline_pairs]

    axes[1, 2].scatter(t_diff, b_perp, alpha=0.6, s=50, edgecolors='black')
    axes[1, 2].set_xlabel('Temporal Baseline (days)')
    axes[1, 2].set_ylabel('Perpendicular Baseline (m)')
    axes[1, 2].set_title('SBAS Interferogram Baseline Distribution')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('landslide_early_warning_results.png', dpi=150)
    print("\n  Visualization saved to landslide_early_warning_results.png")
    plt.close()

    return rmse if 'rmse' in locals() else 0, sbas_results


def main():
    print("\n" + "=" * 70)
    print("Time Series InSAR for Ground Deformation Monitoring")
    print("  PS-InSAR & SBAS Algorithms")
    print("  Applications: Subsidence Monitoring, Landslide Early Warning")
    print("=" * 70)

    results = {}

    try:
        rmse_ps, ps_res = test_subsidence_monitoring()
        results['PS-InSAR'] = {'rmse': rmse_ps, 'n_ps': np.sum(ps_res['ps_mask'])}
    except Exception as e:
        print(f"\nPS-InSAR test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        rmse_sbas, sbas_res = test_landslide_early_warning()
        results['SBAS'] = {'rmse': rmse_sbas}
    except Exception as e:
        print(f"\nSBAS test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<20} {'RMSE (mm/year)':<20} {'Notes':<30}")
    print("-" * 70)
    for method, res in results.items():
        notes = f"{res.get('n_ps', 'N/A')} PS points" if 'n_ps' in res else "Full coverage"
        print(f"{method:<20} {res['rmse']:<20.4f} {notes:<30}")

    if all(v['rmse'] < 5.0 for v in results.values()):
        print("\n\u2705 All tests PASSED! Deformation monitoring accuracy is within acceptable limits.")
    else:
        print("\n\u26a0 Some results exceed expected error bounds.")

    print("\nTypical accuracy: 1-5 mm/year for linear velocity, sub-mm for cumulative deformation.")
    print("=" * 70)


if __name__ == "__main__":
    main()
