import numpy as np
from scipy import linalg
from scipy.signal import periodogram
from scipy.ndimage import uniform_filter, gaussian_filter
from collections import defaultdict
import warnings

from phase_unwrapping import wrap, least_squares_unwrap, estimate_coherence


class PermanentScatterer:
    def __init__(self, i, j, amplitude, phase_series):
        self.i = i
        self.j = j
        self.amplitude = amplitude
        self.phase_series = phase_series
        self.amplitude_dispersion = None
        self.coherence = None
        self.linear_velocity = None
        self.nonlinear_component = None
        self.elevation_error = None
        self.deformation_series = None
        self.is_ps = False


def detect_ps(amplitude_stack, phase_stack, ad_threshold=0.25, min_coherence=0.7):
    n_images, rows, cols = amplitude_stack.shape

    amplitude_mean = np.mean(amplitude_stack, axis=0)
    amplitude_std = np.std(amplitude_stack, axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        amplitude_dispersion = amplitude_std / (amplitude_mean + 1e-10)

    coherence = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            coherence[i, j] = np.abs(np.mean(np.exp(1j * phase_stack[:, i, j])))

    ps_mask = (amplitude_dispersion < ad_threshold) & (coherence > min_coherence)

    ps_points = []
    for i in range(rows):
        for j in range(cols):
            if ps_mask[i, j]:
                ps = PermanentScatterer(
                    i=i, j=j,
                    amplitude=amplitude_mean[i, j],
                    phase_series=phase_stack[:, i, j].copy()
                )
                ps.amplitude_dispersion = amplitude_dispersion[i, j]
                ps.coherence = coherence[i, j]
                ps.is_ps = True
                ps_points.append(ps)

    return ps_points, ps_mask, amplitude_dispersion, coherence


def generate_delaunay_triangles(ps_points, rows, cols, max_distance=50):
    n_ps = len(ps_points)
    if n_ps < 3:
        return []

    coords = np.array([(ps.i, ps.j) for ps in ps_points])

    edges = []
    for i in range(n_ps):
        distances = np.sqrt(np.sum((coords - coords[i])**2, axis=1))
        neighbors = np.argsort(distances)[1:8]
        for j in neighbors:
            if distances[j] < max_distance and i < j:
                edges.append((i, j))

    triangles = []
    edge_set = set(edges)

    for i in range(n_ps):
        for j in range(i + 1, n_ps):
            if (i, j) not in edge_set:
                continue
            for k in range(j + 1, n_ps):
                if (i, k) in edge_set and (j, k) in edge_set:
                    triangles.append((i, j, k))

    return triangles if triangles else [(i, i + 1, i + 2) for i in range(0, n_ps - 2, 3)]


def spatial_temporal_unwrap(ps_points, triangles, phase_stack):
    n_ps = len(ps_points)
    n_images = phase_stack.shape[0]

    if n_ps < 2:
        return np.zeros((n_images, n_ps))

    reference_idx = 0
    ref_ps = ps_points[reference_idx]

    unwrapped_phase = np.zeros((n_images, n_ps))
    unwrapped_phase[:, reference_idx] = phase_stack[:, ref_ps.i, ref_ps.j]

    connected = {reference_idx}
    remaining = set(range(n_ps)) - connected

    while remaining:
        best_edge = None
        best_cost = np.inf

        for i in connected:
            pi = ps_points[i]
            for j in remaining:
                pj = ps_points[j]

                dist = np.sqrt((pi.i - pj.i)**2 + (pi.j - pj.j)**2)
                coh = (pi.coherence + pj.coherence) / 2
                cost = dist / (coh + 0.01)

                if cost < best_cost:
                    best_cost = cost
                    best_edge = (i, j)

        if best_edge is None:
            break

        i, j = best_edge
        pi = ps_points[i]
        pj = ps_points[j]

        for t in range(n_images):
            diff = wrap(phase_stack[t, pj.i, pj.j] - phase_stack[t, pi.i, pi.j])
            unwrapped_phase[t, j] = unwrapped_phase[t, i] + diff

        connected.add(j)
        remaining.remove(j)

    for t in range(n_images):
        ref_val = unwrapped_phase[t, reference_idx]
        k = np.round((phase_stack[t, ref_ps.i, ref_ps.j] - wrap(ref_val)) / (2 * np.pi))
        unwrapped_phase[t, :] += 2 * np.pi * k

    return unwrapped_phase


def build_design_matrix(timestamps, baseline_perp, wavelength=0.056):
    n_images = len(timestamps)

    t = np.array(timestamps) - timestamps[0]
    t_years = t / (365.25 * 24 * 3600)

    phase_to_mm = (wavelength / (4 * np.pi)) * 1000

    G = np.zeros((n_images, 3))
    G[:, 0] = t_years
    G[:, 1] = baseline_perp
    G[:, 2] = 1

    return G, t_years, phase_to_mm


def estimate_ps_parameters(unwrapped_phase, timestamps, baseline_perp, wavelength=0.056):
    n_images, n_ps = unwrapped_phase.shape

    G, t_years, phase_to_mm = build_design_matrix(timestamps, baseline_perp, wavelength)

    velocities = np.zeros(n_ps)
    elevation_errors = np.zeros(n_ps)
    residuals = np.zeros((n_images, n_ps))

    for i in range(n_ps):
        phi = unwrapped_phase[:, i]

        try:
            x, res_i, rank, s = linalg.lstsq(G, phi)

            v_rad_per_year = x[0]
            z_error_rad = x[1]

            velocities[i] = v_rad_per_year * (wavelength / (4 * np.pi)) * 1000 * 2
            elevation_errors[i] = z_error_rad * (wavelength / (4 * np.pi))

            residuals[:, i] = phi - G @ x
        except:
            velocities[i] = 0
            elevation_errors[i] = 0
            residuals[:, i] = 0

    return velocities, elevation_errors, residuals, t_years


def estimate_nonlinear_deformation(residuals, t_normalized, ps_points,
                                   window_size=5, alpha=0.01, wavelength=0.056):
    n_images, n_ps = residuals.shape

    nonlinear_deformation = np.zeros_like(residuals)
    phase_to_mm = (wavelength / (4 * np.pi)) * 1000

    for i in range(n_ps):
        r = residuals[:, i]

        if window_size > 1 and n_images >= window_size:
            r_smooth = np.convolve(r, np.ones(window_size) / window_size, mode='same')
        else:
            r_smooth = np.zeros_like(r)

        nonlinear_deformation[:, i] = r_smooth * phase_to_mm

    return nonlinear_deformation


def ps_insar_processing(amplitude_stack, phase_stack, timestamps, baseline_perp,
                        ad_threshold=0.25, min_coherence=0.7, wavelength=0.056):
    n_images, rows, cols = amplitude_stack.shape

    print(f"Processing {n_images} images, {rows}x{cols} pixels...")
    print("Step 1: Detecting Permanent Scatterers...")

    ps_points, ps_mask, amp_disp, coh = detect_ps(
        amplitude_stack, phase_stack, ad_threshold, min_coherence
    )

    print(f"  Detected {len(ps_points)} PS points")

    if len(ps_points) < 2:
        print("WARNING: Too few PS points for time series analysis")
        return {
            'ps_points': ps_points,
            'ps_mask': ps_mask,
            'amplitude_dispersion': amp_disp,
            'coherence': coh,
            'velocities': np.zeros((rows, cols)),
            'deformation_series': np.zeros((n_images, rows, cols)),
            'elevation_error': np.zeros((rows, cols)),
        }

    print("Step 2: Generating Delaunay triangles...")
    triangles = generate_delaunay_triangles(ps_points, rows, cols)
    print(f"  Generated {len(triangles)} triangles")

    print("Step 3: Spatio-temporal phase unwrapping...")
    unwrapped_phase = spatial_temporal_unwrap(ps_points, triangles, phase_stack)

    print("Step 4: Estimating linear deformation parameters...")
    velocities, elevation_errors, residuals, t_norm = estimate_ps_parameters(
        unwrapped_phase, timestamps, baseline_perp, wavelength
    )

    print("Step 5: Estimating non-linear deformation...")
    nonlinear = estimate_nonlinear_deformation(residuals, t_norm, ps_points, wavelength=wavelength)

    n_ps = len(ps_points)
    deformation_series = np.zeros((n_images, n_ps))
    phase_to_mm = (wavelength / (4 * np.pi)) * 1000
    for i in range(n_ps):
        linear_component = velocities[i] * t_norm
        deformation_series[:, i] = linear_component + nonlinear[:, i]

    velocity_map = np.zeros((rows, cols))
    deformation_map = np.zeros((n_images, rows, cols))
    elevation_map = np.zeros((rows, cols))

    for idx, ps in enumerate(ps_points):
        velocity_map[ps.i, ps.j] = velocities[idx]
        deformation_map[:, ps.i, ps.j] = deformation_series[:, idx]
        elevation_map[ps.i, ps.j] = elevation_errors[idx]

        ps.linear_velocity = velocities[idx]
        ps.nonlinear_component = nonlinear[:, idx]
        ps.elevation_error = elevation_errors[idx]
        ps.deformation_series = deformation_series[:, idx]

    print("Processing complete!")

    return {
        'ps_points': ps_points,
        'ps_mask': ps_mask,
        'amplitude_dispersion': amp_disp,
        'coherence': coh,
        'velocities': velocity_map,
        'deformation_series': deformation_map,
        'elevation_error': elevation_map,
        'unwrapped_phase': unwrapped_phase,
        'residuals': residuals,
        't_normalized': t_norm,
        'triangles': triangles,
    }


def select_sbas_pairs(baseline_perp, timestamps, max_baseline=150, max_time_diff=365):
    n_images = len(baseline_perp)
    pairs = []

    for i in range(n_images):
        for j in range(i + 1, n_images):
            b_perp = abs(baseline_perp[j] - baseline_perp[i])
            t_diff = abs(timestamps[j] - timestamps[i]) / (24 * 3600)

            if b_perp < max_baseline and t_diff < max_time_diff:
                pairs.append((i, j, b_perp, t_diff))

    return pairs


def sbas_inversion(ifg_stack, pairs, timestamps, wavelength=0.056,
                   ref_pixel=None):
    n_images = len(timestamps)
    n_ifgs = len(pairs)
    rows, cols = ifg_stack.shape[1:]

    if ref_pixel is None:
        ref_pixel = (rows // 2, cols // 2)

    ri, rj = ref_pixel

    B = np.zeros((n_ifgs, n_images - 1))

    for idx, (i, j, _, _) in enumerate(pairs):
        if i > 0:
            B[idx, i - 1] = -1
        if j > 0:
            B[idx, j - 1] = 1

    phase_diff = np.zeros((n_ifgs, rows, cols))
    for idx, (i, j, _, _) in enumerate(pairs):
        phase_diff[idx] = ifg_stack[idx]

    phase_diff_ref = phase_diff[:, ri, rj:rj + 1, rj:rj + 1]

    deformation = np.zeros((n_images, rows, cols))

    for i in range(rows):
        for j in range(cols):
            y = phase_diff[:, i, j] - phase_diff_ref[:, 0, 0]

            try:
                x, residuals, rank, s = linalg.lstsq(B, y)
                deformation[1:, i, j] = x
            except:
                deformation[1:, i, j] = 0

    deformation_mm = deformation * (wavelength / (4 * np.pi)) * 1000

    velocity = np.zeros((rows, cols))
    t = np.array(timestamps) - timestamps[0]
    t_days = t / (24 * 3600)

    for i in range(rows):
        for j in range(cols):
            if np.std(deformation_mm[:, i, j]) > 1e-6:
                coeff = np.polyfit(t_days, deformation_mm[:, i, j], 1)
                velocity[i, j] = coeff[0] * 365

    return deformation_mm, velocity


def sbas_processing(ifg_stack, pairs, timestamps, baseline_perp,
                    wavelength=0.056, ref_pixel=None):
    n_ifgs, rows, cols = ifg_stack.shape
    n_images = len(timestamps)

    print(f"SBAS Processing: {n_images} images, {len(pairs)} interferograms...")
    print("Step 1: Quality-guided phase unwrapping for each interferogram...")

    unwrapped_ifgs = np.zeros_like(ifg_stack, dtype=np.float64)
    for i in range(n_ifgs):
        coh = estimate_coherence(ifg_stack[i])
        unwrapped_ifgs[i] = least_squares_unwrap(ifg_stack[i])
        print(f"  Unwrapped interferogram {i + 1}/{n_ifgs}")

    print("Step 2: Singular Value Decomposition (SVD) inversion...")
    deformation, velocity = sbas_inversion(
        unwrapped_ifgs, pairs, timestamps, wavelength, ref_pixel
    )

    print("Step 3: Time series smoothing...")
    deformation_smoothed = np.zeros_like(deformation)
    for i in range(rows):
        for j in range(cols):
            deformation_smoothed[:, i, j] = gaussian_filter1d(
                deformation[:, i, j], sigma=1.0
            )

    print("SBAS processing complete!")

    return {
        'deformation_series': deformation_smoothed,
        'velocity': velocity,
        'unwrapped_ifgs': unwrapped_ifgs,
        'pairs': pairs,
    }


def gaussian_filter1d(data, sigma=1.0):
    from scipy.ndimage import gaussian_filter1d as gf
    return gf(data, sigma=sigma)


def generate_deformation_mask(velocity, threshold=5.0, min_area=10):
    from scipy.ndimage import label, binary_closing, binary_dilation

    significant_deformation = np.abs(velocity) > threshold

    if np.sum(significant_deformation) == 0:
        return significant_deformation, []

    labeled, n_regions = label(significant_deformation)

    regions = []
    for i in range(1, n_regions + 1):
        mask = labeled == i
        area = np.sum(mask)
        if area >= min_area:
            max_vel = np.max(np.abs(velocity[mask]))
            mean_vel = np.mean(velocity[mask])
            regions.append({
                'label': i,
                'mask': mask,
                'area': area,
                'max_velocity': max_vel,
                'mean_velocity': mean_vel,
                'type': 'subsidence' if mean_vel < 0 else 'uplift',
            })

    regions.sort(key=lambda x: x['max_velocity'], reverse=True)

    return significant_deformation, regions


def early_warning(deformation_series, velocity, regions,
                  warning_threshold_velocity=10.0,
                  warning_threshold_acceleration=2.0,
                  timestamps=None):
    warnings = []

    n_images = deformation_series.shape[0]

    for region in regions:
        mask = region['mask']

        mean_deformation = np.zeros(n_images)
        for t in range(n_images):
            mean_deformation[t] = np.mean(deformation_series[t][mask])

        if len(mean_deformation) >= 3:
            t = np.arange(n_images)
            coeff1 = np.polyfit(t, mean_deformation, 1)
            velocity_lin = coeff1[0]

            if len(mean_deformation) >= 5:
                coeff2 = np.polyfit(t, mean_deformation, 2)
                acceleration = 2 * coeff2[0]
            else:
                acceleration = 0

            warning_level = 'normal'

            if abs(velocity_lin) > warning_threshold_velocity * 2:
                warning_level = 'critical'
            elif abs(velocity_lin) > warning_threshold_velocity:
                warning_level = 'warning'
            elif abs(acceleration) > warning_threshold_acceleration:
                warning_level = 'warning'

            if warning_level != 'normal':
                warnings.append({
                    'region': region,
                    'warning_level': warning_level,
                    'velocity': velocity_lin,
                    'acceleration': acceleration,
                    'max_deformation': np.max(np.abs(mean_deformation)),
                })

    warnings.sort(key=lambda x: abs(x['velocity']), reverse=True)

    return warnings


def simulate_time_series(rows, cols, n_images=15,
                         deformation_type='subsidence',
                         noise_level=0.1):
    x = np.linspace(-5, 5, cols)
    y = np.linspace(-5, 5, rows)
    X, Y = np.meshgrid(x, y)

    timestamps = []
    for i in range(n_images):
        timestamps.append(i * 30 * 24 * 3600)

    baseline_perp = np.random.normal(0, 50, n_images)

    amplitude_stack = np.random.rand(n_images, rows, cols) * 100 + 50

    true_velocity = np.zeros((rows, cols))

    if deformation_type == 'subsidence':
        center1 = (int(rows * 0.3), int(cols * 0.3))
        center2 = (int(rows * 0.7), int(cols * 0.6))

        r1 = np.sqrt((X - X[center1])**2 + (Y - Y[center1])**2)
        r2 = np.sqrt((X - X[center2])**2 + (Y - Y[center2])**2)

        true_velocity = -15 * np.exp(-r1**2 / 2) - 8 * np.exp(-r2**2 / 4)

    elif deformation_type == 'landslide':
        center = (int(rows * 0.5), int(cols * 0.3))
        r = np.sqrt((X - X[center])**2 + (Y - Y[center])**2)

        direction = np.arctan2(Y - Y[center], X - X[center])
        true_velocity = -20 * np.exp(-r**2 / 3) * np.cos(direction)

    elif deformation_type == 'mixed':
        center1 = (int(rows * 0.3), int(cols * 0.3))
        center2 = (int(rows * 0.7), int(cols * 0.7))

        r1 = np.sqrt((X - X[center1])**2 + (Y - Y[center1])**2)
        r2 = np.sqrt((X - X[center2])**2 + (Y - Y[center2])**2)

        true_velocity = -12 * np.exp(-r1**2 / 2) + 5 * np.exp(-r2**2 / 3)

    true_deformation = np.zeros((n_images, rows, cols))
    for t in range(n_images):
        true_deformation[t] = true_velocity * (t / (n_images - 1))

    wavelength = 0.056
    phase_to_mm = wavelength / (4 * np.pi) * 1000

    phase_stack = np.zeros((n_images, rows, cols))
    for t in range(n_images):
        topo_phase = baseline_perp[t] * (X**2 + Y**2) * 0.01
        phase_stack[t] = true_deformation[t] / phase_to_mm + topo_phase
        phase_stack[t] += np.random.normal(0, noise_level, (rows, cols))

    wrapped_phase = wrap(phase_stack)

    return {
        'amplitude_stack': amplitude_stack,
        'wrapped_phase': wrapped_phase,
        'true_velocity': true_velocity,
        'true_deformation': true_deformation,
        'timestamps': timestamps,
        'baseline_perp': baseline_perp,
        'wavelength': wavelength,
    }


def generate_interferograms(wrapped_phase, timestamps, baseline_perp,
                            max_baseline=150, max_time_diff=365):
    n_images = wrapped_phase.shape[0]
    rows, cols = wrapped_phase.shape[1:]

    pairs = select_sbas_pairs(baseline_perp, timestamps, max_baseline, max_time_diff)

    ifg_stack = np.zeros((len(pairs), rows, cols))
    for idx, (i, j, _, _) in enumerate(pairs):
        ifg_stack[idx] = wrap(wrapped_phase[j] - wrapped_phase[i])

    return ifg_stack, pairs
