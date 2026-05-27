import numpy as np
from time_series_insar import (
    simulate_time_series, generate_interferograms,
    ps_insar_processing, sbas_processing,
    generate_deformation_mask, early_warning
)

print('Quick validation test for Time Series InSAR...')
np.random.seed(42)

print('\n1. Testing data simulation...')
sim = simulate_time_series(20, 20, 8, deformation_type='subsidence', noise_level=0.05)
print(f'   Simulated {sim["wrapped_phase"].shape[0]} images, 20x20 pixels')
print(f'   Max velocity: {np.max(np.abs(sim["true_velocity"])):.2f} mm/year')

print('\n2. Testing PS-InSAR processing (small dataset)...')
ps_res = ps_insar_processing(
    sim['amplitude_stack'], sim['wrapped_phase'],
    sim['timestamps'], sim['baseline_perp'],
    ad_threshold=0.35, min_coherence=0.6,
    wavelength=sim['wavelength']
)
print(f'   Detected {np.sum(ps_res["ps_mask"])} PS points')
print(f'   Velocity range: [{np.min(ps_res["velocities"]):.2f}, {np.max(ps_res["velocities"]):.2f}] mm/year')

print('\n3. Testing deformation region detection...')
mask, regions = generate_deformation_mask(ps_res['velocities'], threshold=2.0, min_area=3)
print(f'   Detected {len(regions)} regions')

print('\n4. Testing early warning system...')
warnings = early_warning(
    ps_res['deformation_series'], ps_res['velocities'], regions,
    warning_threshold_velocity=5.0
)
print(f'   Generated {len(warnings)} warnings')

print('\n5. Testing SBAS interferogram generation...')
ifg_stack, pairs = generate_interferograms(
    sim['wrapped_phase'], sim['timestamps'], sim['baseline_perp'],
    max_baseline=300, max_time_diff=730
)
print(f'   Generated {len(pairs)} interferograms')

print('\nAll basic tests PASSED!')
