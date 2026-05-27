import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy import linalg
from scipy.integrate import quad

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. BEM / FEM 前向模型 (个体头模型)
# ============================================================

def generate_bem_forward_model(elec_pos, src_pos,
                                brain_radius=0.87, skull_radius=0.92, skin_radius=1.0,
                                sigma_brain=0.3, sigma_skull=0.004, sigma_skin=0.3,
                                individual_variation=True):
    n_src = len(src_pos)
    n_elec = len(elec_pos)

    sigma_1 = sigma_brain
    sigma_2 = sigma_skull
    sigma_3 = sigma_skin
    r1 = brain_radius
    r2 = skull_radius
    r3 = skin_radius

    G = np.zeros((n_elec, n_src * 3))

    for i in range(n_elec):
        r_e = elec_pos[i]
        r_e_norm = np.linalg.norm(r_e)

        r_e_scaled = r_e.copy()
        if individual_variation and r_e_norm > 0:
            scale = 0.95 + 0.05 * np.sin(5 * np.arctan2(r_e[1], r_e[0]))
            r_e_scaled = r_e * scale

        for j in range(n_src):
            r_s = src_pos[j]
            r_s_norm = np.linalg.norm(r_s)

            if r_s_norm >= r1:
                continue

            diff = r_e_scaled - r_s
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                dist = 1e-10

            cos_alpha = np.dot(r_s, r_e_scaled) / (r_s_norm * r_e_norm + 1e-10)

            phi_primary = (1.0 / dist)

            phi_boundary_terms = 0.0
            if r_e_norm > r2:
                f2_1 = (sigma_2 - sigma_3) / (sigma_2 * sigma_3)
                f2_2 = (r3 ** 2) / (np.sqrt(r_e_norm ** 2 + r_s_norm ** 2
                                            - 2 * r_e_norm * r_s_norm * cos_alpha))
                phi_boundary_terms += f2_1 * f2_2

                f1_1 = (sigma_1 - sigma_2) / (sigma_1 * sigma_2)
                f1_2 = (r2 ** 2) / (np.sqrt(r_e_norm ** 2 + r_s_norm ** 2
                                            - 2 * r_e_norm * r_s_norm * cos_alpha))
                phi_boundary_terms += f1_1 * f1_2

            phi_total = phi_primary + phi_boundary_terms
            phi_total /= (4.0 * np.pi * sigma_3)

            direction = diff / dist
            for k in range(3):
                G[i, j * 3 + k] = phi_total * direction[k]

    return G


def generate_fem_forward_model(elec_pos, src_pos,
                                n_elements_theta=12, n_elements_phi=24,
                                brain_radius=0.87, skull_radius=0.92, skin_radius=1.0,
                                sigma_brain=0.3, sigma_skull=0.004, sigma_skin=0.3):
    n_src = len(src_pos)
    n_elec = len(elec_pos)

    elements = []
    for it in range(n_elements_theta):
        theta = np.pi * (it + 0.5) / n_elements_theta
        for ip in range(n_elements_phi):
            phi = 2.0 * np.pi * (ip + 0.5) / n_elements_phi

            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)

            r_norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            if r_norm < brain_radius / 1.0:
                sigma = sigma_brain
            elif r_norm < skull_radius:
                sigma = sigma_skull
            else:
                sigma = sigma_skin

            elements.append({
                'pos': np.array([x, y, z]) * 0.5,
                'sigma': sigma,
                'volume': (4.0 / 3.0 * np.pi * 1.0 ** 3) / (n_elements_theta * n_elements_phi)
            })

    G = np.zeros((n_elec, n_src * 3))

    for i in range(n_elec):
        r_e = elec_pos[i]

        for j in range(n_src):
            r_s = src_pos[j]
            r_s_norm = np.linalg.norm(r_s)

            if r_s_norm >= brain_radius:
                continue

            diff = r_e - r_s
            dist_e_s = np.linalg.norm(diff)
            if dist_e_s < 1e-10:
                dist_e_s = 1e-10

            phi_direct = 1.0 / (4.0 * np.pi * sigma_skin * dist_e_s)

            phi_correction = 0.0
            for elem in elements:
                r_elem = elem['pos']
                sigma_elem = elem['sigma']
                vol = elem['volume']

                dist_s_elem = np.linalg.norm(r_s - r_elem)
                dist_e_elem = np.linalg.norm(r_e - r_elem)

                if dist_s_elem < 1e-10 or dist_e_elem < 1e-10:
                    continue

                weight = (sigma_skin - sigma_elem) / (sigma_skin * sigma_elem)
                phi_correction += weight * vol * (1.0 / dist_s_elem) * (1.0 / dist_e_elem)

            phi_total = phi_direct + phi_correction / (4.0 * np.pi)

            direction = diff / dist_e_s
            for k in range(3):
                G[i, j * 3 + k] = phi_total * direction[k]

    return G


def generate_realistic_head_model(elec_pos, src_pos, head_type='bem'):
    if head_type == 'bem':
        return generate_bem_forward_model(elec_pos, src_pos)
    elif head_type == 'fem':
        return generate_fem_forward_model(elec_pos, src_pos)
    else:
        return generate_3sphere_forward_model(elec_pos, src_pos)


# ============================================================
# 2. fMRI 空间先验约束
# ============================================================

def generate_fmri_prior(src_pos, active_regions=None, n_rois=4, snr_fmri=20):
    n_src = len(src_pos)
    fmri_map = np.zeros(n_src)

    if active_regions is None:
        active_regions = []
        for _ in range(n_rois):
            center = np.random.randn(3)
            center = center / (np.linalg.norm(center) + 1e-10) * 0.7
            width = 0.05 + 0.15 * np.random.rand()
            amplitude = 1.0 + 0.5 * np.random.randn()
            active_regions.append({'center': center, 'width': width, 'amplitude': amplitude})

    for region in active_regions:
        for i in range(n_src):
            d = np.linalg.norm(src_pos[i] - region['center'])
            fmri_map[i] += region['amplitude'] * np.exp(-d ** 2 / (2 * region['width'] ** 2))

    if snr_fmri > 0:
        noise = np.random.randn(n_src) / snr_fmri
        fmri_map += noise

    fmri_map = np.maximum(fmri_map, 0)

    return fmri_map, active_regions


def build_fmri_weight_matrix(fmri_map, n_src, method='soft', gamma=2.0):
    W_fmri = np.zeros((n_src * 3, n_src * 3))

    for i in range(n_src):
        if method == 'soft':
            w = (fmri_map[i] + 1e-3) ** gamma
        elif method == 'hard':
            w = fmri_map[i] if fmri_map[i] > 0.3 else 0.01
        else:
            w = 1.0
        w = np.clip(w, 0.01, 100.0)

        for k in range(3):
            W_fmri[i * 3 + k, i * 3 + k] = w

    return W_fmri


def solve_mne_fmri(G, EEG, fmri_map, alpha=0.01, gamma=2.0, method='soft'):
    n_elec, n_src_3 = G.shape
    n_src = n_src_3 // 3

    W_fmri = build_fmri_weight_matrix(fmri_map, n_src, method=method, gamma=gamma)

    W_depth = np.zeros_like(W_fmri)
    for i in range(n_src):
        col_norm = np.linalg.norm(G[:, i * 3:(i + 1) * 3], 'fro')
        if col_norm > 0:
            for k in range(3):
                W_depth[i * 3 + k, i * 3 + k] = 1.0 / np.sqrt(col_norm)

    W_total = W_fmri * W_depth

    G_weighted = G @ W_total

    GtG = G_weighted.T @ G_weighted
    GtV = G_weighted.T @ EEG

    regularization = alpha * np.trace(GtG) / n_src_3 * np.eye(n_src_3)

    J_weighted = np.linalg.solve(GtG + regularization, GtV)
    J = W_total @ J_weighted

    return J


def solve_mxne(G, EEG, fmri_map, alpha=0.01, gamma=2.0, max_iter=500, tol=1e-6,
               min_weight=0.01):
    n_elec, n_src_3 = G.shape
    n_src = n_src_3 // 3

    W = build_fmri_weight_matrix(fmri_map, n_src, method='soft', gamma=gamma)

    J_prev = np.zeros(n_src_3)

    for iteration in range(max_iter):
        W_iter = np.zeros_like(W)
        for i in range(n_src):
            J_i = J_prev[i * 3:(i + 1) * 3]
            J_norm = np.sqrt(np.sum(J_i ** 2))
            w = 1.0 / (J_norm + 1e-4)
            w = np.clip(w, min_weight, 1.0 / min_weight)
            w_fmri = W[i * 3, i * 3]
            for k in range(3):
                W_iter[i * 3 + k, i * 3 + k] = w * w_fmri

        G_weighted = G @ W_iter
        GtG = G_weighted.T @ G_weighted
        GtV = G_weighted.T @ EEG

        regularization = alpha * np.trace(GtG) / n_src_3 * np.eye(n_src_3)

        J_weighted = np.linalg.solve(GtG + regularization, GtV)
        J_new = W_iter @ J_weighted

        diff = np.linalg.norm(J_new - J_prev) / (np.linalg.norm(J_new) + 1e-10)
        if diff < tol:
            break

        J_prev = J_new

    return J_new


def solve_lasso_eeG(G, EEG, alpha=0.01, max_iter=500, tol=1e-6):
    n_elec, n_src_3 = G.shape
    n_src = n_src_3 // 3

    J = np.zeros(n_src_3)
    r = EEG.copy()

    active_set = []
    GtG = G.T @ G
    GtV = G.T @ EEG

    for iteration in range(max_iter):
        correlations = np.abs(G.T @ r)

        J_per_source = np.zeros(n_src)
        for i in range(n_src):
            J_per_source[i] = np.sqrt(np.sum(correlations[i * 3:(i + 1) * 3] ** 2))

        best_source = np.argmax(J_per_source)
        if J_per_source[best_source] < alpha:
            break

        if best_source not in active_set:
            active_set.append(best_source)
            active_set.sort()

        idx = np.concatenate([[s * 3 + k for k in range(3)] for s in active_set])
        G_active = G[:, idx]
        J_active = np.linalg.solve(G_active.T @ G_active + alpha * np.eye(len(idx)),
                                    G_active.T @ EEG)

        J_new = np.zeros(n_src_3)
        for i_act, s in enumerate(active_set):
            J_new[s * 3:(s + 1) * 3] = J_active[i_act * 3:(i_act + 1) * 3]

        diff = np.linalg.norm(J_new - J) / (np.linalg.norm(J_new) + 1e-10)
        if diff < tol:
            J = J_new
            break

        J = J_new
        r = EEG - G @ J

    return J


# ============================================================
# 3. 癫痫灶定位 & BCI 应用场景
# ============================================================

def simulate_epileptic_spike(G, n_elec, n_timepoints=200, fs=256,
                              focus_location=None, focus_amplitude=3.0,
                              snr_db=10, n_src_3=None):
    n_src = n_src_3 // 3 if n_src_3 else 128

    t = np.arange(n_timepoints) / fs
    spike_template = focus_amplitude * np.exp(-0.5 * ((t - 0.1) / 0.015) ** 2)
    spike_template = spike_template * np.sin(2 * np.pi * 8 * t)

    background = 0.3 * np.random.randn(n_elec, n_timepoints)

    J_focus = np.zeros(n_src_3)
    if focus_location is None:
        focus_location = np.random.choice([10, 45, 78])
    J_focus[focus_location * 3] = 1.0

    EEG_focus = np.outer(G @ J_focus, spike_template)
    EEG_total = EEG_focus + background

    return EEG_total, t, focus_location


def localize_epileptic_focus(G, EEG_data, method='mne_fmri', fmri_map=None, alpha=0.01):
    n_elec, n_timepoints = EEG_data.shape
    n_src_3 = G.shape[1]
    n_src = n_src_3 // 3

    covariance = EEG_data @ EEG_data.T / n_timepoints

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    rank = np.sum(eigenvalues > 0.01 * eigenvalues[-1])
    eigenvectors_act = eigenvectors[:, -rank:]

    GtG = G.T @ G
    Gt_eig = G.T @ eigenvectors_act

    if method == 'mne_fmri' and fmri_map is not None:
        J_map = solve_mne_fmri(G, eigenvectors_act[:, -1], fmri_map, alpha=alpha)
    elif method == 'mxne' and fmri_map is not None:
        J_map = solve_mxne(G, eigenvectors_act[:, -1], fmri_map, alpha=alpha)
    elif method == 'lasso':
        J_map = solve_lasso_eeG(G, eigenvectors_act[:, -1], alpha=alpha * 0.1)
    elif method == 'wmne':
        J_map = solve_wmne(G, eigenvectors_act[:, -1], alpha=alpha)
    else:
        regularization = alpha * np.trace(GtG) / n_src_3 * np.eye(n_src_3)
        J_map = np.linalg.solve(GtG + regularization, Gt_eig[:, -1])

    J_mag = np.sqrt(J_map[0::3] ** 2 + J_map[1::3] ** 2 + J_map[2::3] ** 2)

    return J_mag, J_map


def simulate_bci_event(G, event_type='motor_left', n_elec=64, n_timepoints=300, fs=256,
                       focus_location=None, snr_db=15, n_src_3=None):
    n_src = n_src_3 // 3 if n_src_3 else 128

    t = np.arange(n_timepoints) / fs

    if event_type == 'motor_left':
        focus_location = 35
        freq, amp = 10, 2.0
    elif event_type == 'motor_right':
        focus_location = 40
        freq, amp = 10, 2.0
    elif event_type == 'visual':
        focus_location = 90
        freq, amp = 12, 2.5
    else:
        focus_location = 50
        freq, amp = 8, 1.5

    event_template = amp * np.sin(2 * np.pi * freq * t)
    event_template *= np.exp(-0.5 * ((t - 0.15) / 0.08) ** 2)

    background = 0.4 * np.random.randn(n_elec, n_timepoints)

    J_focus = np.zeros(n_src_3)
    J_focus[focus_location * 3] = 1.0

    EEG_event = np.outer(G @ J_focus, event_template)
    EEG_total = EEG_event + background

    return EEG_total, t, focus_location, event_type


def bci_classify_source(J_mag, src_pos, regions=None):
    if regions is None:
        regions = {
            'motor_left': np.array([0.5, 0.2, 0.3]),
            'motor_right': np.array([-0.5, 0.2, 0.3]),
            'visual': np.array([0.0, -0.6, 0.0]),
        }

    scores = {}
    for region_name, center in regions.items():
        distances = np.linalg.norm(src_pos - center, axis=1)
        weights = np.exp(-distances ** 2 / (2 * 0.3 ** 2))
        scores[region_name] = np.sum(J_mag * weights)

    total = np.sum(list(scores.values())) + 1e-10
    for key in scores:
        scores[key] /= total

    predicted = max(scores, key=scores.get)
    return predicted, scores


# ============================================================
# 4. 辅助可视化
# ============================================================

def plot_epileptic_localization(ax, src_pos, J_mag, true_focus, title="Epileptic Focus"):
    ax.clear()

    J_mag_norm = J_mag / (np.max(J_mag) + 1e-10)

    mask = J_mag_norm > 0.15

    ax.scatter(src_pos[~mask, 0], src_pos[~mask, 1], src_pos[~mask, 2],
               c='gray', alpha=0.1, s=10)

    if np.any(mask):
        ax.scatter(src_pos[mask, 0], src_pos[mask, 1], src_pos[mask, 2],
                   c=J_mag_norm[mask], cmap='Reds', s=60 * J_mag_norm[mask],
                   alpha=0.8, label='Estimated')

    if true_focus is not None:
        ax.scatter([src_pos[true_focus, 0]], [src_pos[true_focus, 1]], [src_pos[true_focus, 2]],
                   c='green', s=200, marker='*', edgecolors='black', linewidths=1.5,
                   label='True Focus', zorder=10)

    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = 0.8 * np.cos(u) * np.sin(v)
    y = 0.8 * np.sin(u) * np.sin(v)
    z = 0.8 * np.cos(v)
    ax.plot_surface(x, y, z, alpha=0.08, color='blue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc='upper left')


def plot_bci_decoding(ax, t, EEG_data, J_mag, predicted, true_label, title="BCI Decoding"):
    ax.clear()

    ax1 = ax.twinx()
    ax.plot(t, EEG_data[0, :], 'b-', alpha=0.5, label='EEG Ch0')
    ax1.plot(t, J_mag / (np.max(J_mag) + 1e-10), 'r-', label='Source Power')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('EEG Amplitude', color='b')
    ax1.set_ylabel('Source Power (normalized)', color='r')

    ax.set_title(f'{title}\nTrue: {true_label}, Predicted: {predicted}', fontsize=9)
    ax.legend(loc='upper left', fontsize=7)
    ax1.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)


def generate_3sphere_forward_model(elec_pos, src_pos, radius=(0.87, 0.92, 1.0),
                                   conductivity=(0.33, 0.0042, 0.33)):
    n_src = len(src_pos)
    n_elec = len(elec_pos)

    r_brain, r_skull, r_skin = radius
    c_brain, c_skull, c_skin = conductivity

    G = np.zeros((n_elec, n_src * 3))

    for i in range(n_elec):
        for j in range(n_src):
            r_e = elec_pos[i]
            r_s = src_pos[j]
            diff = r_e - r_s
            dist = np.linalg.norm(diff)

            if dist < 1e-10:
                dist = 1e-10

            r_s_norm = np.linalg.norm(r_s)
            r_e_norm = np.linalg.norm(r_e)

            f1 = 1.0 / (4.0 * np.pi * c_skin * dist ** 2)

            depth_factor = r_s_norm / r_brain
            if depth_factor < 1.0:
                f1 *= depth_factor ** 4

            direction = diff / dist
            for k in range(3):
                G[i, j * 3 + k] = f1 * direction[k]

    return G


def generate_sensor_positions(n_elec=64, radius=1.0):
    positions = []
    for i in range(n_elec):
        phi = np.arccos(1 - 2 * (i + 0.5) / n_elec)
        theta = np.pi * (1 + np.sqrt(5)) * i

        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)

        positions.append([x, y, z])

    return np.array(positions)


def generate_source_positions(n_src=128, max_radius=0.8, min_radius=0.2):
    positions = []
    golden_angle = np.pi * (3 - np.sqrt(5))

    for i in range(n_src):
        t = i / float(n_src - 1)
        r = min_radius + t * (max_radius - min_radius)

        y = 1 - (i / float(n_src - 1)) * 2
        radius_at_y = np.sqrt(1 - y * y)

        theta = golden_angle * i

        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y

        positions.append([r * x, r * y, r * z])

    return np.array(positions)


def generate_source_dipoles(n_src, active_sources, source_amplitudes=None):
    J = np.zeros(n_src * 3)

    if source_amplitudes is None:
        source_amplitudes = [1.0] * len(active_sources)

    for idx, src_idx in enumerate(active_sources):
        amp = source_amplitudes[idx]
        J[src_idx * 3] = amp

    return J


def add_noise(eeg_data, snr_db=20):
    signal_power = np.mean(eeg_data ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(*eeg_data.shape)
    return eeg_data + noise


def solve_mne(G, EEG, alpha=0.01):
    n_elec, n_src_3 = G.shape
    n_src = n_src_3 // 3

    GtG = G.T @ G
    GtV = G.T @ EEG

    regularization = alpha * np.trace(GtG) / n_src_3 * np.eye(n_src_3)

    J = np.linalg.solve(GtG + regularization, GtV)

    return J


def solve_wmne(G, EEG, src_pos=None, alpha=0.01, depth_order=0.8, use_column_norm=True):
    n_elec, n_src_3 = G.shape
    n_src = n_src_3 // 3

    W = np.zeros((n_src_3, n_src_3))

    for i in range(n_src):
        if use_column_norm:
            col_norm = np.linalg.norm(G[:, i * 3:(i + 1) * 3], 'fro')
            if col_norm > 0:
                weight = 1.0 / (col_norm ** (2 * depth_order))
            else:
                weight = 1.0
        else:
            depth = np.linalg.norm(src_pos[i])
            weight = (1.0 / depth) ** (4 * depth_order)

        W[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = np.eye(3) * weight

    G_weighted = G @ W

    GtG = G_weighted.T @ G_weighted
    GtV = G_weighted.T @ EEG

    regularization = alpha * np.trace(GtG) / n_src_3 * np.eye(n_src_3)

    J_weighted = np.linalg.solve(GtG + regularization, GtV)

    J = W @ J_weighted

    return J


def solve_sLORETA(G, EEG, alpha=0.01):
    n_elec, n_src_3 = G.shape
    n_src = n_src_3 // 3

    GtG = G.T @ G
    GtV = G.T @ EEG

    regularization = alpha * np.trace(GtG) / n_src_3 * np.eye(n_src_3)

    J = np.linalg.solve(GtG + regularization, GtV)

    J_mne = J.copy()

    K = np.linalg.solve(GtG + regularization, G.T)
    S = K @ G

    J_normalized = np.zeros_like(J)
    for i in range(n_src):
        idx = i * 3
        block = S[idx:idx + 3, idx:idx + 3]
        norm_factor = np.sqrt(np.trace(block @ block.T))
        if norm_factor > 1e-10:
            J_normalized[idx:idx + 3] = J_mne[idx:idx + 3] / norm_factor

    return J_normalized


def solve_dSPM(G, EEG, noise_cov=None, alpha=0.01):
    n_elec, n_src_3 = G.shape
    n_src = n_src_3 // 3

    if noise_cov is None:
        noise_cov = np.eye(n_elec)

    noise_cov_sqrt = linalg.sqrtm(noise_cov)
    noise_cov_inv_sqrt = np.linalg.inv(noise_cov_sqrt)

    G_whitened = noise_cov_inv_sqrt @ G
    EEG_whitened = noise_cov_inv_sqrt @ EEG

    GtG = G_whitened.T @ G_whitened
    GtV = G_whitened.T @ EEG_whitened

    regularization = alpha * np.trace(GtG) / n_src_3 * np.eye(n_src_3)

    J_mne = np.linalg.solve(GtG + regularization, GtV)

    K = np.linalg.solve(GtG + regularization, G_whitened.T)
    S = K @ G_whitened

    J_normalized = np.zeros_like(J_mne)
    for i in range(n_src):
        idx = i * 3
        block = S[idx:idx + 3, idx:idx + 3]
        norm_factor = np.sqrt(np.trace(block @ block.T))
        if norm_factor > 1e-10:
            J_normalized[idx:idx + 3] = J_mne[idx:idx + 3] / norm_factor

    return J_normalized


def solve_loreta(G, EEG, src_pos, alpha=0.01, beta=0.01):
    n_elec, n_src_3 = G.shape
    n_src = n_src_3 // 3

    W = np.zeros((n_src_3, n_src_3))
    for i in range(n_src):
        norm_val = np.linalg.norm(G[:, i * 3:(i + 1) * 3], 'fro')
        if norm_val > 0:
            W[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = np.eye(3) / norm_val

    L = np.zeros((n_src_3, n_src_3))
    distances = cdist(src_pos, src_pos)

    for i in range(n_src):
        for j in range(n_src):
            if i != j:
                d = distances[i, j]
                if d > 0:
                    weight = np.exp(-d ** 2 / (2 * 0.1 ** 2))
                    for k in range(3):
                        L[i * 3 + k, j * 3 + k] = -weight

    for i in range(n_src_3):
        L[i, i] = -np.sum(L[i, :]) + 1e-6

    GtG = G.T @ G
    GtV = G.T @ EEG
    LtL = L.T @ L

    reg1 = alpha * np.trace(GtG) / n_src_3 * np.eye(n_src_3)
    reg2 = beta * np.trace(LtL) / n_src_3 * LtL

    J = np.linalg.solve(GtG + reg1 + reg2, GtV)

    return J


def solve_mne_weighted(G, EEG, src_pos, alpha=0.01):
    n_elec, n_src_3 = G.shape
    n_src = n_src_3 // 3

    W = np.eye(n_src_3)
    for i in range(n_src):
        r = np.linalg.norm(src_pos[i])
        if r > 0:
            W[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] *= 1.0 / r

    G_weighted = G @ W
    GtG = G_weighted.T @ G_weighted
    GtV = G_weighted.T @ EEG

    regularization = alpha * np.trace(GtG) / n_src_3 * np.eye(n_src_3)

    J_weighted = np.linalg.solve(GtG + regularization, GtV)
    J = W @ J_weighted

    return J


def compute_resolution_matrix(G, alpha=0.01):
    n_elec, n_src_3 = G.shape
    n_src = n_src_3 // 3

    GtG = G.T @ G
    regularization = alpha * np.trace(GtG) / n_src_3 * np.eye(n_src_3)

    J_inverse = np.linalg.solve(GtG + regularization, G.T)
    R = J_inverse @ G

    return R


def plot_3d_sources(ax, src_pos, J, title="Source Distribution", threshold=0.3, true_sources=None):
    ax.clear()

    J_mag = np.sqrt(J[0::3] ** 2 + J[1::3] ** 2 + J[2::3] ** 2)
    J_mag = J_mag / (np.max(J_mag) + 1e-10)

    mask = J_mag >= threshold * np.max(J_mag)

    if true_sources is not None:
        true_mask = np.zeros(len(src_pos), dtype=bool)
        true_mask[true_sources] = True
        ax.scatter(src_pos[true_mask, 0], src_pos[true_mask, 1], src_pos[true_mask, 2],
                   c='red', s=150, marker='*', edgecolors='black', linewidths=1.5,
                   label='True Sources', zorder=10)

    ax.scatter(src_pos[~mask, 0], src_pos[~mask, 1], src_pos[~mask, 2],
               c='gray', alpha=0.1, s=10, label='Inactive Sources')

    if np.any(mask):
        if true_sources is not None:
            for ts in true_sources:
                if mask[ts]:
                    mask[ts] = False
        scatter = ax.scatter(src_pos[mask, 0], src_pos[mask, 1], src_pos[mask, 2],
                             c=J_mag[mask], cmap='hot', s=50 * J_mag[mask],
                             alpha=0.8, label='Estimated Sources')
        plt.colorbar(scatter, ax=ax, shrink=0.5)

    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = 0.8 * np.cos(u) * np.sin(v)
    y = 0.8 * np.sin(u) * np.sin(v)
    z = 0.8 * np.cos(v)
    ax.plot_surface(x, y, z, alpha=0.1, color='blue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=10)
    ax.legend(loc='upper left', fontsize=8)


def plot_topography(ax, elec_pos, EEG, title="EEG Topography"):
    ax.clear()

    r = np.sqrt(elec_pos[:, 0] ** 2 + elec_pos[:, 1] ** 2)
    theta = np.arctan2(elec_pos[:, 1], elec_pos[:, 0])

    order = np.argsort(theta)
    r_plot = r[order]
    theta_plot = theta[order]
    eeg_plot = EEG[order]

    x_plot = r_plot * np.cos(theta_plot)
    y_plot = r_plot * np.sin(theta_plot)

    levels = np.linspace(np.min(EEG), np.max(EEG), 50)
    contour = ax.tricontourf(x_plot, y_plot, eeg_plot, levels=levels,
                             cmap='RdBu_r')

    ax.scatter(elec_pos[:, 0], elec_pos[:, 1], c='k', s=10)

    circle = plt.Circle((0, 0), 1.0, fill=False, color='black', linestyle='--')
    ax.add_patch(circle)

    plt.colorbar(contour, ax=ax, shrink=0.8)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')


def plot_source_amplitude_comparison(ax, J_true, J_mne, J_wmne, J_sloreta, J_dspm, src_pos,
                                      active_sources, threshold=0.3):
    ax.clear()

    J_mag_true = np.sqrt(J_true[0::3] ** 2 + J_true[1::3] ** 2 + J_true[2::3] ** 2)
    J_mag_mne = np.sqrt(J_mne[0::3] ** 2 + J_mne[1::3] ** 2 + J_mne[2::3] ** 2)
    J_mag_wmne = np.sqrt(J_wmne[0::3] ** 2 + J_wmne[1::3] ** 2 + J_wmne[2::3] ** 2)
    J_mag_sloreta = np.sqrt(J_sloreta[0::3] ** 2 + J_sloreta[1::3] ** 2 + J_sloreta[2::3] ** 2)
    J_mag_dspm = np.sqrt(J_dspm[0::3] ** 2 + J_dspm[1::3] ** 2 + J_dspm[2::3] ** 2)

    J_mag_true = J_mag_true / (np.max(J_mag_true) + 1e-10)
    J_mag_mne = J_mag_mne / (np.max(J_mag_mne) + 1e-10)
    J_mag_wmne = J_mag_wmne / (np.max(J_mag_wmne) + 1e-10)
    J_mag_sloreta = J_mag_sloreta / (np.max(J_mag_sloreta) + 1e-10)
    J_mag_dspm = J_mag_dspm / (np.max(J_mag_dspm) + 1e-10)

    n_src = len(src_pos)

    width = 0.15
    ax.bar(np.arange(n_src) - 2 * width, J_mag_true, width, label='True', color='red', alpha=0.7)
    ax.bar(np.arange(n_src) - width, J_mag_mne, width, label='MNE', color='blue', alpha=0.7)
    ax.bar(np.arange(n_src), J_mag_wmne, width, label='WMNE', color='green', alpha=0.7)
    ax.bar(np.arange(n_src) + width, J_mag_sloreta, width, label='sLORETA', color='orange', alpha=0.7)
    ax.bar(np.arange(n_src) + 2 * width, J_mag_dspm, width, label='dSPM', color='purple', alpha=0.7)

    ax.set_xlabel('Source Index')
    ax.set_ylabel('Normalized Amplitude')
    ax.set_title('Source Amplitude Comparison')
    ax.legend(fontsize=8)


def plot_depth_vs_amplitude(ax, src_pos, J_true, J_mne, J_wmne, J_sloreta, J_dspm, active_sources):
    ax.clear()

    depths = np.linalg.norm(src_pos, axis=1)
    max_depth = np.max(depths)

    J_mag_true = np.sqrt(J_true[0::3] ** 2 + J_true[1::3] ** 2 + J_true[2::3] ** 2)
    J_mag_mne = np.sqrt(J_mne[0::3] ** 2 + J_mne[1::3] ** 2 + J_mne[2::3] ** 2)
    J_mag_wmne = np.sqrt(J_wmne[0::3] ** 2 + J_wmne[1::3] ** 2 + J_wmne[2::3] ** 2)
    J_mag_sloreta = np.sqrt(J_sloreta[0::3] ** 2 + J_sloreta[1::3] ** 2 + J_sloreta[2::3] ** 2)
    J_mag_dspm = np.sqrt(J_dspm[0::3] ** 2 + J_dspm[1::3] ** 2 + J_dspm[2::3] ** 2)

    J_mag_true = J_mag_true / (np.max(J_mag_true) + 1e-10)
    J_mag_mne = J_mag_mne / (np.max(J_mag_mne) + 1e-10)
    J_mag_wmne = J_mag_wmne / (np.max(J_mag_wmne) + 1e-10)
    J_mag_sloreta = J_mag_sloreta / (np.max(J_mag_sloreta) + 1e-10)
    J_mag_dspm = J_mag_dspm / (np.max(J_mag_dspm) + 1e-10)

    ax.scatter(depths, J_mag_true, c='red', s=100, marker='*', label='True (all sources)')
    mask_active = np.zeros(len(src_pos), dtype=bool)
    mask_active[active_sources] = True
    ax.scatter(depths[mask_active], J_mag_mne[mask_active], c='blue', s=60, marker='o', label='MNE (active)', alpha=0.8)
    ax.scatter(depths[~mask_active], J_mag_mne[~mask_active], c='blue', s=10, marker='o', alpha=0.2)
    ax.scatter(depths[mask_active], J_mag_wmne[mask_active], c='green', s=60, marker='s', label='WMNE (active)', alpha=0.8)
    ax.scatter(depths[~mask_active], J_mag_wmne[~mask_active], c='green', s=10, marker='s', alpha=0.2)
    ax.scatter(depths[mask_active], J_mag_sloreta[mask_active], c='orange', s=60, marker='^', label='sLORETA (active)', alpha=0.8)
    ax.scatter(depths[~mask_active], J_mag_sloreta[~mask_active], c='orange', s=10, marker='^', alpha=0.2)
    ax.scatter(depths[mask_active], J_mag_dspm[mask_active], c='purple', s=60, marker='D', label='dSPM (active)', alpha=0.8)
    ax.scatter(depths[~mask_active], J_mag_dspm[~mask_active], c='purple', s=10, marker='D', alpha=0.2)

    ax.axvline(x=max_depth * 0.33, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=max_depth * 0.66, color='gray', linestyle='--', alpha=0.5)
    ax.text(max_depth * 0.16, ax.get_ylim()[1] * 0.9, 'Deep', ha='center', fontsize=8, color='gray')
    ax.text(max_depth * 0.5, ax.get_ylim()[1] * 0.9, 'Medium', ha='center', fontsize=8, color='gray')
    ax.text(max_depth * 0.83, ax.get_ylim()[1] * 0.9, 'Superficial', ha='center', fontsize=8, color='gray')

    ax.set_xlabel('Source Depth (distance from center)')
    ax.set_ylabel('Normalized Amplitude')
    ax.set_title('Depth vs Amplitude Recovery')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)


def main():
    print("=" * 70)
    print("EEG 逆问题：深度权重补偿 (WMNE, sLORETA, dSPM)")
    print("=" * 70)

    n_elec = 64
    n_src = 128
    snr_db = 15
    alpha = 0.01
    depth_order = 0.25
    beta_loreta = 0.05

    print(f"\n参数设置:")
    print(f"  电极数量: {n_elec}")
    print(f"  源点数量: {n_src}")
    print(f"  信噪比: {snr_db} dB")
    print(f"  正则化参数 alpha: {alpha}")
    print(f"  WMNE 深度权重阶数: {depth_order}")

    print("\n[1/6] 生成电极位置...")
    elec_pos = generate_sensor_positions(n_elec, radius=1.0)
    print(f"      已生成 {n_elec} 个电极位置")

    print("\n[2/6] 生成源位置...")
    src_pos = generate_source_positions(n_src, max_radius=0.8, min_radius=0.2)
    print(f"      已生成 {n_src} 个源位置")

    depths = np.linalg.norm(src_pos, axis=1)
    sorted_by_depth = np.argsort(depths)
    deep_source = sorted_by_depth[5]
    medium_source = sorted_by_depth[n_src // 2]
    superficial_source = sorted_by_depth[-5]

    print(f"      深度范围: [{np.min(depths):.3f}, {np.max(depths):.3f}]")
    print(f"      深部源索引: {deep_source} (深度: {depths[deep_source]:.3f})")
    print(f"      中层源索引: {medium_source} (深度: {depths[medium_source]:.3f})")
    print(f"      浅层源索引: {superficial_source} (深度: {depths[superficial_source]:.3f})")

    print("\n[3/6] 构建前向模型（三层球模型）...")
    G = generate_3sphere_forward_model(elec_pos, src_pos)
    print(f"      前向矩阵维度: {G.shape}")

    col_norms = np.array([np.linalg.norm(G[:, i*3:(i+1)*3], 'fro') for i in range(n_src)])
    print(f"      列范数范围: [{np.min(col_norms):.6f}, {np.max(col_norms):.6f}]")
    print(f"      深部源列范数: {col_norms[deep_source]:.6f}")
    print(f"      浅层源列范数: {col_norms[superficial_source]:.6f}")
    print(f"      浅/深列范数比: {col_norms[superficial_source] / col_norms[deep_source]:.2f}")

    print("\n[4/6] 模拟真实源活动（含深部、中层、浅层源）...")
    active_sources = [deep_source, medium_source, superficial_source]
    source_amplitudes = [1.0, 1.0, 1.0]
    J_true = generate_source_dipoles(n_src, active_sources, source_amplitudes)
    print(f"      激活源索引: {active_sources}")
    print(f"      源振幅: {source_amplitudes} (全部相等)")

    print("\n[5/6] 生成 EEG 数据...")
    EEG_clean = G @ J_true
    EEG_noisy = add_noise(EEG_clean, snr_db=snr_db)
    print(f"      EEG 数据维度: {EEG_noisy.shape}")
    print(f"      信号功率: {np.mean(EEG_clean ** 2):.6f}")
    print(f"      噪声功率: {np.mean((EEG_noisy - EEG_clean) ** 2):.6f}")

    print("\n[6/6] 求解逆问题...")

    print("\n  [MNE] 标准最小范数估计...")
    J_mne = solve_mne(G, EEG_noisy, alpha=alpha)
    J_mne_mag = np.sqrt(J_mne[0::3] ** 2 + J_mne[1::3] ** 2 + J_mne[2::3] ** 2)
    print(f"      MNE: 深部源振幅={J_mne_mag[deep_source]:.6f}, 浅层源振幅={J_mne_mag[superficial_source]:.6f}")
    print(f"      MNE 深/浅振幅比: {J_mne_mag[deep_source] / J_mne_mag[superficial_source]:.4f}")

    print(f"\n  [WMNE] 加权最小范数估计 (depth_order={depth_order})...")
    J_wmne = solve_wmne(G, EEG_noisy, src_pos=src_pos, alpha=alpha, depth_order=depth_order, use_column_norm=True)
    J_wmne_mag = np.sqrt(J_wmne[0::3] ** 2 + J_wmne[1::3] ** 2 + J_wmne[2::3] ** 2)
    print(f"      WMNE: 深部源振幅={J_wmne_mag[deep_source]:.6f}, 浅层源振幅={J_wmne_mag[superficial_source]:.6f}")
    print(f"      WMNE 深/浅振幅比: {J_wmne_mag[deep_source] / J_wmne_mag[superficial_source]:.4f}")

    print("\n  [sLORETA] 标准化低分辨率电磁断层扫描...")
    J_sloreta = solve_sLORETA(G, EEG_noisy, alpha=alpha)
    J_sloreta_mag = np.sqrt(J_sloreta[0::3] ** 2 + J_sloreta[1::3] ** 2 + J_sloreta[2::3] ** 2)
    print(f"      sLORETA: 深部源振幅={J_sloreta_mag[deep_source]:.6f}, 浅层源振幅={J_sloreta_mag[superficial_source]:.6f}")
    print(f"      sLORETA 深/浅振幅比: {J_sloreta_mag[deep_source] / J_sloreta_mag[superficial_source]:.4f}")

    print("\n  [dSPM] 动态统计参数映射...")
    noise_cov = np.eye(n_elec) * np.mean((EEG_noisy - EEG_clean) ** 2)
    J_dspm = solve_dSPM(G, EEG_noisy, noise_cov=noise_cov, alpha=alpha)
    J_dspm_mag = np.sqrt(J_dspm[0::3] ** 2 + J_dspm[1::3] ** 2 + J_dspm[2::3] ** 2)
    print(f"      dSPM: 深部源振幅={J_dspm_mag[deep_source]:.6f}, 浅层源振幅={J_dspm_mag[superficial_source]:.6f}")
    print(f"      dSPM 深/浅振幅比: {J_dspm_mag[deep_source] / J_dspm_mag[superficial_source]:.4f}")

    print("\n" + "=" * 70)
    print("深度偏差分析结果")
    print("=" * 70)

    print(f"\n{'方法':<12} {'深部源':>12} {'中层源':>12} {'浅层源':>12} {'深/浅比':>12}")
    print("-" * 65)
    print(f"{'真实值':<12} {1.0:>12.4f} {1.0:>12.4f} {1.0:>12.4f} {1.0:>12.4f}")
    
    def normalize_and_get_ratio(J_mag, deep, med, sup):
        max_amp = np.max(J_mag)
        if max_amp < 1e-10:
            return 0, 0, 0, 0
        j_deep = J_mag[deep] / max_amp
        j_med = J_mag[med] / max_amp
        j_sup = J_mag[sup] / max_amp
        ratio = j_deep / j_sup if j_sup > 1e-10 else 0
        return j_deep, j_med, j_sup, ratio

    d, m, s, r = normalize_and_get_ratio(J_mne_mag, deep_source, medium_source, superficial_source)
    print(f"{'MNE':<12} {d:>12.4f} {m:>12.4f} {s:>12.4f} {r:>12.4f}")
    
    d, m, s, r = normalize_and_get_ratio(J_wmne_mag, deep_source, medium_source, superficial_source)
    print(f"{'WMNE':<12} {d:>12.4f} {m:>12.4f} {s:>12.4f} {r:>12.4f}")
    
    d, m, s, r = normalize_and_get_ratio(J_sloreta_mag, deep_source, medium_source, superficial_source)
    print(f"{'sLORETA':<12} {d:>12.4f} {m:>12.4f} {s:>12.4f} {r:>12.4f}")
    
    d, m, s, r = normalize_and_get_ratio(J_dspm_mag, deep_source, medium_source, superficial_source)
    print(f"{'dSPM':<12} {d:>12.4f} {m:>12.4f} {s:>12.4f} {r:>12.4f}")

    print("\n" + "=" * 70)
    print("定位准确率分析")
    print("=" * 70)

    def evaluate_localization(true_sources, estimated_peaks, src_pos, threshold=0.08):
        correct = 0
        for ts in true_sources:
            for ep in estimated_peaks:
                dist = np.linalg.norm(src_pos[ts] - src_pos[ep])
                if dist < threshold:
                    correct += 1
                    break
        return correct, len(true_sources)

    def get_top_peaks(J_mag, k=5):
        return np.argsort(J_mag)[-k:][::-1]

    mne_peaks = get_top_peaks(J_mne_mag, k=5)
    wmne_peaks = get_top_peaks(J_wmne_mag, k=5)
    sloreta_peaks = get_top_peaks(J_sloreta_mag, k=5)
    dspm_peaks = get_top_peaks(J_dspm_mag, k=5)

    mne_correct, total = evaluate_localization(active_sources, mne_peaks, src_pos)
    wmne_correct, _ = evaluate_localization(active_sources, wmne_peaks, src_pos)
    sloreta_correct, _ = evaluate_localization(active_sources, sloreta_peaks, src_pos)
    dspm_correct, _ = evaluate_localization(active_sources, dspm_peaks, src_pos)

    print(f"\n{'方法':<12} {'前5峰值索引':<30} {'准确率':>10}")
    print("-" * 60)
    print(f"{'MNE':<12} {str(mne_peaks.tolist()):<30} {mne_correct}/{total:>10}")
    print(f"{'WMNE':<12} {str(wmne_peaks.tolist()):<30} {wmne_correct}/{total:>10}")
    print(f"{'sLORETA':<12} {str(sloreta_peaks.tolist()):<30} {sloreta_correct}/{total:>10}")
    print(f"{'dSPM':<12} {str(dspm_peaks.tolist()):<30} {dspm_correct}/{total:>10}")

    print("\n" + "=" * 70)
    print("可视化结果...")
    print("=" * 70)

    fig = plt.figure(figsize=(20, 14))

    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    plot_3d_sources(ax1, src_pos, J_true, "真实源分布", threshold=0.3, true_sources=active_sources)

    ax2 = fig.add_subplot(2, 4, 2, projection='3d')
    plot_3d_sources(ax2, src_pos, J_mne, "MNE 估计源分布", threshold=0.3, true_sources=active_sources)

    ax3 = fig.add_subplot(2, 4, 3, projection='3d')
    plot_3d_sources(ax3, src_pos, J_wmne, "WMNE 估计源分布", threshold=0.3, true_sources=active_sources)

    ax4 = fig.add_subplot(2, 4, 4, projection='3d')
    plot_3d_sources(ax4, src_pos, J_sloreta, "sLORETA 估计源分布", threshold=0.3, true_sources=active_sources)

    ax5 = fig.add_subplot(2, 4, 5)
    plot_topography(ax5, elec_pos, EEG_noisy, "带噪声 EEG 地形图")

    ax6 = fig.add_subplot(2, 4, 6)
    plot_topography(ax6, elec_pos, G @ J_mne, "MNE 重构 EEG")

    ax7 = fig.add_subplot(2, 4, 7)
    plot_topography(ax7, elec_pos, G @ J_wmne, "WMNE 重构 EEG")

    ax8 = fig.add_subplot(2, 4, 8)
    plot_topography(ax8, elec_pos, G @ J_sloreta, "sLORETA 重构 EEG")

    plt.tight_layout()
    plt.savefig('eeg_depth_correction_3d.png', dpi=150, bbox_inches='tight')
    print("\n图像已保存: eeg_depth_correction_3d.png")

    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax_comp = axes[0, 0]
    plot_source_amplitude_comparison(ax_comp, J_true, J_mne, J_wmne, J_sloreta, J_dspm,
                                      src_pos, active_sources, threshold=0.3)
    ax_comp.set_xlim(min(active_sources) - 5, max(active_sources) + 5)

    ax_depth = axes[0, 1]
    plot_depth_vs_amplitude(ax_depth, src_pos, J_true, J_mne, J_wmne, J_sloreta, J_dspm, active_sources)

    ax_err = axes[1, 0]
    methods = ['MNE', 'WMNE', 'sLORETA', 'dSPM']
    source_errors = [
        np.linalg.norm(J_true - J_mne) / np.linalg.norm(J_true),
        np.linalg.norm(J_true - J_wmne) / np.linalg.norm(J_true),
        np.linalg.norm(J_true - J_sloreta) / np.linalg.norm(J_true),
        np.linalg.norm(J_true - J_dspm) / np.linalg.norm(J_true)
    ]
    recon_errors = [
        np.linalg.norm(EEG_noisy - G @ J_mne) / np.linalg.norm(EEG_noisy),
        np.linalg.norm(EEG_noisy - G @ J_wmne) / np.linalg.norm(EEG_noisy),
        np.linalg.norm(EEG_noisy - G @ J_sloreta) / np.linalg.norm(EEG_noisy),
        np.linalg.norm(EEG_noisy - G @ J_dspm) / np.linalg.norm(EEG_noisy)
    ]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax_err.bar(x - width / 2, source_errors, width, label='Source Estimation Error', color='steelblue')
    bars2 = ax_err.bar(x + width / 2, recon_errors, width, label='EEG Reconstruction Error', color='coral')
    ax_err.set_xlabel('Method')
    ax_err.set_ylabel('Relative Error')
    ax_err.set_title('误差对比：源估计 vs EEG重构')
    ax_err.set_xticks(x)
    ax_err.set_xticklabels(methods)
    ax_err.legend()
    ax_err.grid(True, alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        ax_err.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.4f}',
                   ha='center', va='bottom', fontsize=8)

    ax_col = axes[1, 1]
    ax_col.scatter(depths, col_norms, c='gray', s=30, alpha=0.5, label='All sources')
    ax_col.scatter(depths[active_sources], col_norms[active_sources], c='red', s=100, 
                   marker='*', label='Active sources', zorder=5)
    ax_col.set_xlabel('Source Depth')
    ax_col.set_ylabel('Forward Matrix Column Norm')
    ax_col.set_title('深度 vs 前向矩阵列范数\n(解释深度偏差的根源)')
    ax_col.legend()
    ax_col.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('eeg_depth_analysis.png', dpi=150, bbox_inches='tight')
    print("图像已保存: eeg_depth_analysis.png")

    plt.show()

    print("\n" + "=" * 70)
    print("算法原理说明")
    print("=" * 70)

    print("""
【深度偏差问题 (Depth Bias)】
  现象: 标准MNE严重低估深部源的振幅，原因是：
    - 前向矩阵G的列范数随深度增加而减小
    - 深部源到电极的距离更远，信号衰减更严重
    - L2范数最小化偏向于使用浅层源来解释数据

  数学原因: G[:, j] 的范数 ||G[:, j]|| ∝ 1/r_j² (近似)
            其中 r_j 是源 j 到中心的距离

【WMNE - 加权最小范数估计】
  原理: 对每个源的解引入权重 W_j = 1 / ||G[:, j]||^(2p)
        其中 p 是深度权重阶数 (典型值 0.5-1.0)

  数学: min ||G W J_w||² + α||W J_w||²
       然后 J = W J_w

  优点:
    - 有效补偿深部源的振幅衰减
    - 计算简单高效
    - 可调参数 p 控制补偿强度

【sLORETA - 标准化低分辨率电磁断层扫描】
  原理: 对MNE解进行逐源标准化，除以分辨率矩阵的迹
        这相当于对每个源进行噪声归一化

  数学: J_norm[i] = J_mne[i] / sqrt(trace(S_ii))
        其中 S = (G^T G + αI)^(-1) G^T G

  优点:
    - 理论上无偏估计（在特定条件下）
    - 无需手动调整深度参数
    - 提高定位准确性

  缺点:
    - 输出是相对值，不是真正的电流密度

【dSPM - 动态统计参数映射】
  原理: 类似于sLORETA，但先对数据进行噪声白化
        考虑实际噪声协方差结构

  数学: 使用噪声协方差矩阵 Σ 对 G 和 EEG 进行白化
        G_white = Σ^(-1/2) G, EEG_white = Σ^(-1/2) EEG
        然后应用类似sLORETA的标准化

  优点:
    - 考虑实际噪声分布
    - 输出可解释为 t 值或 z 分数
    - 适用于统计阈值化
""")

    print("\n运行完成！")


def demo_advanced():
    print("=" * 70)
    print("EEG 逆问题：个体头模型 + fMRI 先验 (BEM/FEM + MNFL)")
    print("=" * 70)

    n_elec = 64
    n_src = 128
    snr_db = 12
    alpha = 0.01

    print(f"\n参数设置:")
    print(f"  电极数量: {n_elec}, 源点数量: {n_src}")
    print(f"  信噪比: {snr_db} dB")

    print("\n[1] 生成电极与源位置...")
    elec_pos = generate_sensor_positions(n_elec, radius=1.0)
    src_pos = generate_source_positions(n_src, max_radius=0.8, min_radius=0.2)

    depths = np.linalg.norm(src_pos, axis=1)
    sorted_by_depth = np.argsort(depths)
    deep_source = sorted_by_depth[10]
    medium_source = sorted_by_depth[n_src // 2]
    superficial_source = sorted_by_depth[-10]

    print(f"      深部源索引: {deep_source} (深度: {depths[deep_source]:.3f})")
    print(f"      浅层源索引: {superficial_source} (深度: {depths[superficial_source]:.3f})")

    print("\n[2] 构建前向模型...")
    print("      [BEM] 边界元法前向模型...")
    G_bem = generate_bem_forward_model(elec_pos, src_pos)
    print(f"          BEM 前向矩阵维度: {G_bem.shape}")

    print("      [FEM] 有限元法前向模型...")
    G_fem = generate_fem_forward_model(elec_pos, src_pos)
    print(f"          FEM 前向矩阵维度: {G_fem.shape}")

    print("      [3-Sphere] 三层球模型...")
    G_3sph = generate_3sphere_forward_model(elec_pos, src_pos)
    print(f"          3-Sphere 前向矩阵维度: {G_3sph.shape}")

    G = G_bem.copy()
    n_src_3 = G.shape[1]

    print("\n[3] 生成 fMRI 空间先验...")
    active_regions = [
        {'center': src_pos[deep_source], 'width': 0.08, 'amplitude': 2.5},
        {'center': src_pos[medium_source], 'width': 0.10, 'amplitude': 1.8},
        {'center': src_pos[superficial_source], 'width': 0.12, 'amplitude': 2.0},
    ]
    fmri_map, _ = generate_fmri_prior(src_pos, active_regions=active_regions, n_rois=3, snr_fmri=15)
    print(f"      fMRI 范围: [{np.min(fmri_map):.3f}, {np.max(fmri_map):.3f}]")
    print(f"      深部源处fMRI值: {fmri_map[deep_source]:.3f}")
    print(f"      浅层源处fMRI值: {fmri_map[superficial_source]:.3f}")

    print("\n[4] 模拟真实源活动 (三个等振幅源)...")
    active_sources = [deep_source, medium_source, superficial_source]
    source_amplitudes = [1.0, 1.0, 1.0]
    J_true = generate_source_dipoles(n_src, active_sources, source_amplitudes)

    EEG_clean = G @ J_true
    EEG_noisy = add_noise(EEG_clean, snr_db=snr_db)

    print("\n[5] 求解逆问题...")
    methods_to_test = ['mne', 'wmne', 'mne_fmri', 'mxne', 'lasso']
    results = {}

    for method in methods_to_test:
        print(f"\n      [{method.upper()}] 求解中...")
        if method == 'mne':
            J = solve_mne(G, EEG_noisy, alpha=alpha)
        elif method == 'wmne':
            J = solve_wmne(G, EEG_noisy, alpha=alpha, depth_order=0.3)
        elif method == 'mne_fmri':
            J = solve_mne_fmri(G, EEG_noisy, fmri_map, alpha=alpha, gamma=1.5)
        elif method == 'mxne':
            J = solve_mxne(G, EEG_noisy, fmri_map, alpha=alpha, gamma=1.5, max_iter=100)
        elif method == 'lasso':
            J = solve_lasso_eeG(G, EEG_noisy, alpha=alpha * 0.05)
        else:
            J = solve_mne(G, EEG_noisy, alpha=alpha)

        J_mag = np.sqrt(J[0::3] ** 2 + J[1::3] ** 2 + J[2::3] ** 2)
        results[method] = {'J': J, 'J_mag': J_mag}

    print("\n" + "=" * 70)
    print("定量评估")
    print("=" * 70)

    def compute_localization_error(J_mag, true_sources, src_pos, k=3):
        peaks = np.argsort(J_mag)[-k:][::-1]
        errors = []
        for ts in true_sources:
            min_d = np.min([np.linalg.norm(src_pos[ts] - src_pos[p]) for p in peaks])
            errors.append(min_d)
        return np.mean(errors), peaks

    def compute_source_error(J, J_true):
        return np.linalg.norm(J - J_true) / np.linalg.norm(J_true)

    def compute_dip_score(J_mag, true_sources):
        sorted_idx = np.argsort(J_mag)[::-1]
        score = 0
        for rank, idx in enumerate(sorted_idx[:10]):
            if idx in true_sources:
                score += 1.0 / (rank + 1)
        return score / len(true_sources)

    print(f"\n{'方法':<12} {'源误差':>10} {'定位误差':>10} {'DIP得分':>10}")
    print("-" * 50)

    for method in methods_to_test:
        J = results[method]['J']
        J_mag = results[method]['J_mag']
        src_err = compute_source_error(J, J_true)
        loc_err, peaks = compute_localization_error(J_mag, active_sources, src_pos)
        dip = compute_dip_score(J_mag, active_sources)
        print(f"{method:<12} {src_err:>10.4f} {loc_err:>10.4f} {dip:>10.4f}")
        results[method]['src_err'] = src_err
        results[method]['loc_err'] = loc_err
        results[method]['dip'] = dip

    print("\n" + "=" * 70)
    print("深度偏差分析 (深/浅振幅比)")
    print("=" * 70)

    print(f"\n{'方法':<12} {'深部源':>12} {'浅层源':>12} {'深/浅比':>12}")
    print("-" * 55)
    print(f"{'真实值':<12} {1.0:>12.4f} {1.0:>12.4f} {1.0:>12.4f}")
    for method in methods_to_test:
        J_mag = results[method]['J_mag']
        J_mag_n = J_mag / (np.max(J_mag) + 1e-10)
        print(f"{method:<12} {J_mag_n[deep_source]:>12.4f} {J_mag_n[superficial_source]:>12.4f} "
              f"{J_mag_n[deep_source] / (J_mag_n[superficial_source] + 1e-10):>12.4f}")

    print("\n" + "=" * 70)
    print("应用场景：癫痫灶定位")
    print("=" * 70)

    print("\n      [模拟癫痫放电]...")
    focus_source = sorted_by_depth[n_src // 3]
    EEG_epilepsy, t_epi, true_focus = simulate_epileptic_spike(
        G, n_elec, n_timepoints=200, fs=256,
        focus_location=focus_source, focus_amplitude=3.0,
        snr_db=8, n_src_3=n_src_3)
    print(f"      真实癫痫灶索引: {true_focus}, 深度: {depths[true_focus]:.3f}")

    print("\n      [定位癫痫灶]...")
    loc_methods = ['wmne', 'mne_fmri', 'mxne', 'lasso']
    epilepsy_results = {}

    for loc_method in loc_methods:
        J_mag_loc, J_loc = localize_epileptic_focus(
            G, EEG_epilepsy, method=loc_method, fmri_map=fmri_map, alpha=alpha)
        epilepsy_results[loc_method] = {'J_mag': J_mag_loc, 'J': J_loc}

        peaks = np.argsort(J_mag_loc)[-3:][::-1]
        distances = [np.linalg.norm(src_pos[true_focus] - src_pos[p]) for p in peaks]
        min_dist = np.min(distances)

        print(f"      {loc_method:<12}: 峰值索引={peaks.tolist()}, "
              f"最近距离={min_dist:.4f}, {'正确定位' if min_dist < 0.1 else '偏差较大'}")

    print("\n" + "=" * 70)
    print("应用场景：脑机接口 (BCI)")
    print("=" * 70)

    print("\n      [模拟BCI事件: 左手运动想象]...")
    EEG_bci, t_bci, bci_true_loc, true_label = simulate_bci_event(
        G, event_type='motor_left', n_elec=n_elec, n_timepoints=300, fs=256,
        focus_location=35, snr_db=12, n_src_3=n_src_3)
    print(f"      真实脑区: {true_label}, 源索引: {bci_true_loc}")

    print("\n      [BCI源定位与解码]...")
    bci_methods = ['wmne', 'mne_fmri']
    for bci_method in bci_methods:
        J_mag_bci, J_bci = localize_epileptic_focus(
            G, EEG_bci[:, 100:200], method=bci_method, fmri_map=fmri_map, alpha=alpha)

        predicted, scores = bci_classify_source(J_mag_bci, src_pos)
        print(f"      {bci_method:<12}: 预测={predicted}, 分数={scores}")

    print("\n" + "=" * 70)
    print("可视化结果...")
    print("=" * 70)

    fig = plt.figure(figsize=(22, 16))

    ax1 = fig.add_subplot(3, 4, 1, projection='3d')
    plot_3d_sources(ax1, src_pos, J_true, "真实源分布", threshold=0.3,
                     true_sources=active_sources)

    ax2 = fig.add_subplot(3, 4, 2, projection='3d')
    plot_3d_sources(ax2, src_pos, results['mne']['J'], "MNE", threshold=0.3,
                     true_sources=active_sources)

    ax3 = fig.add_subplot(3, 4, 3, projection='3d')
    plot_3d_sources(ax3, src_pos, results['wmne']['J'], "WMNE", threshold=0.3,
                     true_sources=active_sources)

    ax4 = fig.add_subplot(3, 4, 4, projection='3d')
    plot_3d_sources(ax4, src_pos, results['mne_fmri']['J'], "MNE+fMRI", threshold=0.3,
                     true_sources=active_sources)

    ax5 = fig.add_subplot(3, 4, 5, projection='3d')
    plot_3d_sources(ax5, src_pos, results['mxne']['J'], "MXNE (迭代重加权)", threshold=0.3,
                     true_sources=active_sources)

    ax6 = fig.add_subplot(3, 4, 6, projection='3d')
    plot_3d_sources(ax6, src_pos, results['lasso']['J'], "LASSO (稀疏)", threshold=0.3,
                     true_sources=active_sources)

    ax7 = fig.add_subplot(3, 4, 7)
    ax7.scatter(depths, fmri_map, c='steelblue', s=15, alpha=0.6)
    ax7.scatter(depths[active_sources], fmri_map[active_sources], c='red', s=80,
                marker='*', label='Active sources', zorder=5)
    ax7.set_xlabel('Source Depth')
    ax7.set_ylabel('fMRI Activation')
    ax7.set_title('fMRI 空间先验分布')
    ax7.legend(fontsize=7)
    ax7.grid(True, alpha=0.3)

    ax8 = fig.add_subplot(3, 4, 8)
    ax8.scatter(depths, results['mne']['J_mag'], c='blue', s=10, alpha=0.3, label='MNE')
    ax8.scatter(depths, results['mne_fmri']['J_mag'], c='red', s=10, alpha=0.3, label='MNE+fMRI')
    ax8.set_xlabel('Source Depth')
    ax8.set_ylabel('Source Amplitude')
    ax8.set_title('深度-振幅: MNE vs MNE+fMRI')
    ax8.legend(fontsize=7)
    ax8.grid(True, alpha=0.3)

    ax9 = fig.add_subplot(3, 4, 9, projection='3d')
    plot_epileptic_localization(ax9, src_pos, epilepsy_results['wmne']['J_mag'],
                                 true_focus, "Epilepsy: WMNE")

    ax10 = fig.add_subplot(3, 4, 10, projection='3d')
    plot_epileptic_localization(ax10, src_pos, epilepsy_results['mne_fmri']['J_mag'],
                                  true_focus, "Epilepsy: MNE+fMRI")

    ax11 = fig.add_subplot(3, 4, 11, projection='3d')
    plot_epileptic_localization(ax11, src_pos, epilepsy_results['mxne']['J_mag'],
                                  true_focus, "Epilepsy: MXNE")

    ax12 = fig.add_subplot(3, 4, 12, projection='3d')
    plot_epileptic_localization(ax12, src_pos, epilepsy_results['lasso']['J_mag'],
                                  true_focus, "Epilepsy: LASSO")

    plt.tight_layout()
    plt.savefig('eeg_advanced_results.png', dpi=150, bbox_inches='tight')
    print("\n图像已保存: eeg_advanced_results.png")

    fig2, axes = plt.subplots(2, 3, figsize=(18, 12))

    ax_bar = axes[0, 0]
    method_names = [m.upper() for m in methods_to_test]
    src_errors = [results[m]['src_err'] for m in methods_to_test]
    ax_bar.bar(method_names, src_errors, color=['steelblue', 'green', 'red', 'orange', 'purple'])
    ax_bar.set_title('Source Estimation Error')
    ax_bar.set_ylabel('Relative Error')
    ax_bar.grid(True, alpha=0.3)

    ax_dip = axes[0, 1]
    dip_scores = [results[m]['dip'] for m in methods_to_test]
    ax_dip.bar(method_names, dip_scores, color=['steelblue', 'green', 'red', 'orange', 'purple'])
    ax_dip.set_title('DIP Score (Higher = Better Localization)')
    ax_dip.set_ylabel('DIP Score')
    ax_dip.grid(True, alpha=0.3)

    ax_depth = axes[0, 2]
    for m in methods_to_test:
        J_mag_n = results[m]['J_mag'] / (np.max(results[m]['J_mag']) + 1e-10)
        ax_depth.scatter(depths, J_mag_n, s=5, alpha=0.3, label=m.upper())
    ax_depth.set_xlabel('Source Depth')
    ax_depth.set_ylabel('Normalized Amplitude')
    ax_depth.set_title('Depth-Amplitude Comparison')
    ax_depth.legend(fontsize=7)
    ax_depth.grid(True, alpha=0.3)

    ax_epi = axes[1, 0]
    ax_epi.plot(t_epi, EEG_epilepsy[0, :], 'b-', alpha=0.7)
    ax_epi.axvline(x=0.1, color='red', linestyle='--', alpha=0.5)
    ax_epi.set_xlabel('Time (s)')
    ax_epi.set_ylabel('EEG Amplitude')
    ax_epi.set_title('Simulated Epileptic Spike')
    ax_epi.grid(True, alpha=0.3)

    ax_bci = axes[1, 1]
    ax_bci.plot(t_bci, EEG_bci[0, :], 'b-', alpha=0.7)
    ax_bci.axvline(x=0.15, color='red', linestyle='--', alpha=0.5)
    ax_bci.set_xlabel('Time (s)')
    ax_bci.set_ylabel('EEG Amplitude')
    ax_bci.set_title(f'BCI Event: {true_label}')
    ax_bci.grid(True, alpha=0.3)

    ax_compare = axes[1, 2]
    comparison_data = {
        'MNE': (results['mne']['J_mag'][deep_source] / (np.max(results['mne']['J_mag']) + 1e-10),
                results['mne']['J_mag'][superficial_source] / (np.max(results['mne']['J_mag']) + 1e-10)),
        'MNE+fMRI': (results['mne_fmri']['J_mag'][deep_source] / (np.max(results['mne_fmri']['J_mag']) + 1e-10),
                     results['mne_fmri']['J_mag'][superficial_source] / (np.max(results['mne_fmri']['J_mag']) + 1e-10)),
        'True': (1.0, 1.0)
    }
    x_pos = np.arange(len(comparison_data))
    width = 0.35
    ax_compare.bar(x_pos - width/2, [v[0] for v in comparison_data.values()], width,
                   label='Deep source', color='steelblue')
    ax_compare.bar(x_pos + width/2, [v[1] for v in comparison_data.values()], width,
                   label='Superficial source', color='coral')
    ax_compare.set_xticks(x_pos)
    ax_compare.set_xticklabels(comparison_data.keys())
    ax_compare.set_ylabel('Normalized Amplitude')
    ax_compare.set_title('fMRI Prior Effect on Depth Bias')
    ax_compare.legend()
    ax_compare.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('eeg_advanced_comparison.png', dpi=150, bbox_inches='tight')
    print("图像已保存: eeg_advanced_comparison.png")

    plt.show()

    print("\n" + "=" * 70)
    print("算法原理说明")
    print("=" * 70)

    print("""
【BEM - 边界元法前向模型】
  原理: 利用头组织界面上的边界积分方程计算电势分布
  数学: V(r) = (1/(4πσ)) Σ_i (σ_i-σ_{i-1})/σ_i * ∫_S_i V(r') dS'/|r-r'|
  优点:
    - 更真实的头几何模型（可使用真实MRI分割）
    - 考虑各层组织电导率差异
    - 精度高于球模型
  典型电导率 (S/m): 大脑=0.3, 颅骨=0.004-0.01, 头皮=0.3

【FEM - 有限元法前向模型】
  原理: 将头部分割为体单元，求解拉普拉斯方程的弱形式
  数学: ∇·(σ(r)∇V(r)) = 0 (准静态近似)
  优点:
    - 最灵活：可处理各向异性电导率
    - 可包含真实头几何（白质/灰质/CSF等）
    - 适用于复杂头模型
  缺点:
    - 计算量最大
    - 需要高分辨率MRI分割

【fMRI 空间先验约束】
  原理: 将fMRI激活图作为源分布的空间先验，加权逆问题
  方法一 (软约束): 权重矩阵 W = diag(fMRI^γ)
  方法二 (硬约束): 仅在fMRI激活区域搜索
  方法三 (混合): 迭代重加权 (IRLS)，每次迭代根据当前解更新权重

  MNE+fMRI (软约束):
    min ||G J - V||² + α ||W J||²
    W = diag(fMRI^γ)  (γ控制先验强度)

  MXNE (混合范式):
    交替迭代: 更新权重 W ← 1/|J|, 求解 J = argmin ||GJ-V||² + α||W_fmri·W J||²
    等效于在fMRI约束下进行L1稀疏化

【LASSO EEG - 稀疏源定位】
  原理: 使用坐标下降法迭代选择最相关的源
  适用于: 癫痫灶（通常是孤立的局灶性源）
  优点: 解稀疏，定位精确
  缺点: 对噪声敏感，可能遗漏弱源

【癫痫灶定位】
  典型流程:
    1. 采集发作期/间期EEG
    2. 计算协方差矩阵 C = EEG·EEG^T / T
    3. 特征值分解确定有效秩
    4. 使用fMRI约束的逆方法定位
    5. 可视化源分布，识别癫痫灶

【BCI 脑机接口】
  典型流程:
    1. 实时EEG采集 (256-1024 Hz)
    2. 频段滤波 (mu 8-12Hz, beta 15-30Hz)
    3. 滑动窗口协方差估计
    4. 快速逆求解 (sLORETA/dSPM, <10ms)
    5. 源空间解码 (分类/回归)
    6. 输出控制指令
""")

    print("\n运行完成！建议使用 eeg_advanced_results.png 查看详细结果。")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'advanced':
        demo_advanced()
    else:
        main()