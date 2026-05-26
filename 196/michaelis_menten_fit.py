import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def michaelis_menten(S, Vmax, Km):
    """Michaelis-Menten equation: v = Vmax * S / (Km + S)"""
    return Vmax * S / (Km + S)


def competitive_inhibition(S, Vmax, Km, Ki, I):
    """
    Competitive inhibition: inhibitor competes with substrate for active site.
    v = Vmax * S / (Km * (1 + I/Ki) + S)
    Effect: Km increases, Vmax unchanged.
    """
    Km_app = Km * (1 + I / Ki)
    return Vmax * S / (Km_app + S)


def noncompetitive_inhibition(S, Vmax, Km, Ki, I):
    """
    Non-competitive inhibition: inhibitor binds E and ES equally well.
    v = (Vmax / (1 + I/Ki)) * S / (Km + S)
    Effect: Vmax decreases, Km unchanged.
    """
    Vmax_app = Vmax / (1 + I / Ki)
    return Vmax_app * S / (Km + S)


def uncompetitive_inhibition(S, Vmax, Km, Ki, I):
    """
    Uncompetitive inhibition: inhibitor binds only to ES complex.
    v = (Vmax / (1 + I/Ki)) * S / (Km / (1 + I/Ki) + S)
    Effect: Both Vmax and Km decrease by the same factor.
    """
    factor = 1 + I / Ki
    Vmax_app = Vmax / factor
    Km_app = Km / factor
    return Vmax_app * S / (Km_app + S)


def fit_michaelis_menten(S_data, v_data, V0=None, Km0=None,
                        sigma=None, weights=None, absolute_sigma=False):
    """
    Fit Michaelis-Menten equation using Levenberg-Marquardt (via scipy curve_fit)
    on the RAW (non-linearized) data, optionally with weights.

    Parameters
    ----------
    S_data : array-like
        Substrate concentrations.
    v_data : array-like
        Initial reaction velocities.
    V0 : float, optional
        Initial guess for Vmax. If None, estimated from max(v_data).
    Km0 : float, optional
        Initial guess for Km. If None, estimated from half-max point.
    sigma : array-like, optional
        Per-point standard deviations. If given, curve_fit minimizes
        sum(((v - model) / sigma)**2). Mutually exclusive with `weights`.
    weights : array-like, optional
        Per-point weights w_i. Internally converted to sigma_i = 1/sqrt(w_i).
        Mutually exclusive with `sigma`.
    absolute_sigma : bool, default False
        If True, `pcov` reflects the absolute errors encoded in sigma/weights.

    Returns
    -------
    popt : ndarray, shape (2,)
        Fitted parameters [Vmax, Km].
    pcov : ndarray, shape (2, 2)
        Covariance matrix of the fitted parameters.
    perr : ndarray, shape (2,)
        Standard deviations (1-sigma) of [Vmax, Km].
    """
    if sigma is not None and weights is not None:
        raise ValueError('Provide either `sigma` or `weights`, not both.')

    S_data = np.asarray(S_data, dtype=float)
    v_data = np.asarray(v_data, dtype=float)

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if np.any(weights <= 0):
            raise ValueError('All weights must be > 0.')
        sigma = 1.0 / np.sqrt(weights)

    if V0 is None:
        V0 = float(np.max(v_data)) * 1.1
    if Km0 is None:
        half_v = np.max(v_data) / 2.0
        idx = np.argmin(np.abs(v_data - half_v))
        Km0 = float(S_data[idx])

    # method='lm' -> Levenberg-Marquardt
    if sigma is not None:
        popt, pcov = curve_fit(
            michaelis_menten,
            S_data,
            v_data,
            p0=[V0, Km0],
            sigma=np.asarray(sigma, dtype=float),
            absolute_sigma=absolute_sigma,
            method='lm',
        )
    else:
        # Uniform (equal) weighting: all points contribute equally
        popt, pcov = curve_fit(
            michaelis_menten,
            S_data,
            v_data,
            p0=[V0, Km0],
            method='lm',
        )
    perr = np.sqrt(np.diag(pcov))
    return popt, pcov, perr


def variance_weights(v_data, kind='constant'):
    """
    Build per-point weights from common velocity variance models.

    Parameters
    ----------
    v_data : array-like
        Measured initial velocities.
    kind : str
        - 'constant'  : Var(v) = sigma^2          -> uniform weights
        - 'proportional': Var(v) ∝ v              -> w_i ∝ 1/v (Poisson-like)
        - 'power'      : Var(v) ∝ v^2            -> w_i ∝ 1/v^2 (relative error)

    Returns
    -------
    weights : ndarray
    """
    v = np.asarray(v_data, dtype=float)
    if kind == 'constant':
        return np.ones_like(v)
    if kind == 'proportional':
        if np.any(v <= 0):
            raise ValueError('All velocities must be > 0 for variance weighting.')
        return 1.0 / v
    if kind == 'power':
        if np.any(v <= 0):
            raise ValueError('All velocities must be > 0 for variance weighting.')
        return 1.0 / v ** 2
    raise ValueError(f"Unknown weight kind: {kind!r}")


def identify_inhibition_type(ctrl_popt, ctrl_perr, inh_popt, inh_perr,
                             alpha=0.05, min_rel_change=0.15):
    """
    Identify enzyme inhibition type by comparing Km and Vmax between
    control (no inhibitor) and inhibitor-treated conditions.

    Uses two criteria for a "meaningful" change:
    1. Statistical significance: |p1 - p2| > z * sqrt(err1^2 + err2^2)
    2. Biological relevance: relative change > min_rel_change (default 15%)

    This avoids classifying tiny, statistically significant fluctuations
    as biologically meaningful inhibition.

    Parameters
    ----------
    ctrl_popt : (Vmax_ctrl, Km_ctrl)
    ctrl_perr : (Vmax_err_ctrl, Km_err_ctrl)
    inh_popt : (Vmax_inh, Km_inh)
    inh_perr : (Vmax_err_inh, Km_err_inh)
    alpha : float, default 0.05
        Significance level (0.05 = 95% confidence).
    min_rel_change : float, default 0.15
        Minimum relative change (|p_new - p_old| / |p_old|) required
        to consider a parameter change biologically meaningful.

    Returns
    -------
    result : dict
        Contains 'type' (str), 'description' (str), 'Km_changed' (bool),
        'Vmax_changed' (bool), 'km_ratio' (Km_inh/Km_ctrl),
        'vmax_ratio' (Vmax_inh/Vmax_ctrl).
    """
    from scipy.stats import norm

    Vmax_ctrl, Km_ctrl = ctrl_popt
    Vmax_inh, Km_inh = inh_popt
    Vmax_err_ctrl, Km_err_ctrl = ctrl_perr
    Vmax_err_inh, Km_err_inh = inh_perr

    z = norm.ppf(1 - alpha / 2)  # ~1.96 for alpha=0.05

    def significant(p1, p2, e1, e2):
        stat_sig = abs(p1 - p2) > z * np.sqrt(e1 ** 2 + e2 ** 2)
        rel_change = abs(p1 - p2) / abs(p2)
        return stat_sig and (rel_change > min_rel_change)

    Km_changed = significant(Km_inh, Km_ctrl, Km_err_inh, Km_err_ctrl)
    Vmax_changed = significant(Vmax_inh, Vmax_ctrl, Vmax_err_inh, Vmax_err_ctrl)

    km_ratio = Km_inh / Km_ctrl
    vmax_ratio = Vmax_inh / Vmax_ctrl

    if Km_changed and not Vmax_changed and km_ratio > 1:
        inhibition_type = 'competitive'
        description = ('Competitive inhibition: Km increased significantly '
                       f'({km_ratio:.2f}x), Vmax unchanged. Inhibitor '
                       'competes with substrate for the active site.')
    elif Vmax_changed and not Km_changed and vmax_ratio < 1:
        inhibition_type = 'noncompetitive'
        description = ('Non-competitive inhibition: Vmax decreased significantly '
                       f'({vmax_ratio:.2f}x), Km unchanged. Inhibitor binds '
                       'both free enzyme and ES complex with equal affinity.')
    elif Km_changed and Vmax_changed and km_ratio < 1 and vmax_ratio < 1:
        ratio_diff = abs(km_ratio - vmax_ratio)
        if ratio_diff < 0.15:
            inhibition_type = 'uncompetitive'
            description = ('Uncompetitive inhibition: both Km and Vmax decreased '
                           f'by similar factors (Km: {km_ratio:.2f}x, '
                           f'Vmax: {vmax_ratio:.2f}x). Inhibitor binds only '
                           'the ES complex.')
        else:
            inhibition_type = 'mixed'
            description = ('Mixed inhibition: both Km and Vmax changed '
                           f'(Km: {km_ratio:.2f}x, Vmax: {vmax_ratio:.2f}x). '
                           'Inhibitor binds both E and ES with different affinities.')
    elif not Km_changed and not Vmax_changed:
        inhibition_type = 'none'
        description = 'No significant inhibition detected under these conditions.'
    else:
        inhibition_type = 'ambiguous'
        description = (f'Ambiguous pattern: Km changed={Km_changed}, '
                       f'Vmax changed={Vmax_changed}. Replicate measurements '
                       'or additional [I] concentrations may help.')

    return {
        'type': inhibition_type,
        'description': description,
        'Km_changed': Km_changed,
        'Vmax_changed': Vmax_changed,
        'km_ratio': km_ratio,
        'vmax_ratio': vmax_ratio,
        'z_critical': z,
        'min_rel_change': min_rel_change,
    }


def fit_inhibition_model(S_data, v_data, ctrl_popt, I, model='competitive'):
    """
    Fit a specific inhibition model to estimate Ki, using control parameters
    as constraints or starting values.

    Parameters
    ----------
    S_data, v_data : array-like
        Substrate concentration and velocity data WITH inhibitor.
    ctrl_popt : (Vmax, Km)
        Fitted parameters from control (no inhibitor) condition.
    I : float
        Inhibitor concentration.
    model : str
        'competitive', 'noncompetitive', or 'uncompetitive'.

    Returns
    -------
    Ki : float
        Estimated inhibition constant.
    Ki_err : float
        Standard error of Ki.
    popt : ndarray
        Fitted parameters (Vmax, Km, Ki) — note Vmax/Km may be constrained.
    """
    Vmax_ctrl, Km_ctrl = ctrl_popt
    if I <= 0:
        raise ValueError('Inhibitor concentration I must be > 0')

    if model == 'competitive':
        def model_func(S, Ki):
            return competitive_inhibition(S, Vmax_ctrl, Km_ctrl, Ki, I)
        p0 = [Km_ctrl]
    elif model == 'noncompetitive':
        def model_func(S, Ki):
            return noncompetitive_inhibition(S, Vmax_ctrl, Km_ctrl, Ki, I)
        p0 = [Km_ctrl]
    elif model == 'uncompetitive':
        def model_func(S, Ki):
            return uncompetitive_inhibition(S, Vmax_ctrl, Km_ctrl, Ki, I)
        p0 = [Km_ctrl]
    else:
        raise ValueError(f"Unknown model: {model!r}")

    popt, pcov = curve_fit(model_func, S_data, v_data, p0=p0, method='lm')
    perr = np.sqrt(np.diag(pcov))
    Ki = popt[0]
    Ki_err = perr[0]
    return Ki, Ki_err, popt


def plot_lineweaver_burk(conditions_dict, title='Lineweaver-Burk Plot'):
    """
    Plot Lineweaver-Burk (1/v vs 1/S) for multiple conditions to visualize
    inhibition patterns. Note: used for visualization only, not for fitting.

    Parameters
    ----------
    conditions_dict : dict
        Keys are condition labels (e.g. 'Control', '+Inhibitor'), values are
        tuples (S_data, v_data, popt).
    title : str
        Plot title.
    """
    colors = ['black', 'red', 'blue', 'green', 'orange']
    markers = ['o', 's', '^', 'D', 'v']

    fig, ax = plt.subplots(figsize=(7, 6))

    for i, (label, (S, v, popt)) in enumerate(conditions_dict.items()):
        inv_S = 1.0 / S
        inv_v = 1.0 / v
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        ax.scatter(inv_S, inv_v, color=color, marker=marker,
                   label=label, zorder=3, s=60)

        inv_S_fit = np.linspace(0, np.max(inv_S) * 1.1, 100)
        Vmax, Km = popt
        inv_v_fit = (Km / Vmax) * inv_S_fit + 1.0 / Vmax
        ax.plot(inv_S_fit, inv_v_fit, color=color, linestyle='--', alpha=0.7)

    ax.set_xlabel('1/[S]')
    ax.set_ylabel('1/v')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    fig.tight_layout()
    return fig


def plot_fit(S_data, v_data, popt, perr):
    """Plot the experimental data and the fitted Michaelis-Menten curve."""
    S_fit = np.linspace(0, np.max(S_data) * 1.1, 200)
    v_fit = michaelis_menten(S_fit, *popt)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(S_data, v_data, color='red', label='Experimental data', zorder=3)
    ax.plot(S_fit, v_fit, color='blue', label=f'Fitted: Vmax={popt[0]:.3f}, Km={popt[1]:.3f}')
    ax.set_xlabel('Substrate concentration [S]')
    ax.set_ylabel('Initial velocity v')
    ax.set_title('Michaelis-Menten fit (Levenberg-Marquardt)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    return fig


def print_result(label, popt, perr, S_data, v_data):
    Vmax, Km = popt
    Vmax_err, Km_err = perr
    v_pred = michaelis_menten(S_data, Vmax, Km)
    ss_res = np.sum((v_data - v_pred) ** 2)
    ss_tot = np.sum((v_data - np.mean(v_data)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    print(f'[{label}]')
    print(f'  Vmax = {Vmax:.4f} +/- {Vmax_err:.4f}')
    print(f'  Km   = {Km:.4f} +/- {Km_err:.4f}')
    print(f'  R^2  = {r_squared:.6f}')


def run_inhibition_analysis(S_data, v_ctrl, v_inh, I, label):
    """Run full inhibition analysis workflow and print results."""
    popt_ctrl, pcov_ctrl, perr_ctrl = fit_michaelis_menten(S_data, v_ctrl)
    popt_inh, pcov_inh, perr_inh = fit_michaelis_menten(S_data, v_inh)

    print(f'\n{"="*60}')
    print(f'  INHIBITION ANALYSIS: {label}')
    print(f'{"="*60}')

    print_result('Control (no inhibitor)', popt_ctrl, perr_ctrl, S_data, v_ctrl)
    print()
    print_result(f'+ Inhibitor ([I] = {I})', popt_inh, perr_inh, S_data, v_inh)

    result = identify_inhibition_type(
        popt_ctrl, perr_ctrl, popt_inh, perr_inh, alpha=0.05
    )

    print(f'\n  Inhibition type: {result["type"].upper()}')
    print(f'  {result["description"]}')
    print(f'  Km ratio (inh/ctrl): {result["km_ratio"]:.3f}')
    print(f'  Vmax ratio (inh/ctrl): {result["vmax_ratio"]:.3f}')

    if result['type'] in ('competitive', 'noncompetitive', 'uncompetitive'):
        try:
            Ki, Ki_err, _ = fit_inhibition_model(
                S_data, v_inh, popt_ctrl, I=I, model=result['type']
            )
            print(f'  Estimated Ki = {Ki:.3f} +/- {Ki_err:.3f}')
        except Exception as e:
            print(f'  Ki estimation failed: {e}')

    conditions = {
        'Control': (S_data, v_ctrl, popt_ctrl),
        f'+ Inhibitor ([I]={I})': (S_data, v_inh, popt_inh),
    }
    safe_label = label.lower().replace(' ', '_')
    plot_lineweaver_burk(conditions, title=f'Lineweaver-Burk: {label}')
    plt.savefig(f'lb_plot_{safe_label}.png', dpi=120)
    print(f'\n  Lineweaver-Burk plot saved to: lb_plot_{safe_label}.png')

    return result


def main():
    np.random.seed(42)

    # Substrate concentrations
    S_data = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])

    # True enzyme parameters
    Vmax_true = 50.0
    Km_true = 2.0
    Ki_true = 5.0
    I_conc = 10.0
    noise = 0.8  # Gaussian noise SD

    print('=' * 70)
    print('  ENZYME INHIBITION TYPE IDENTIFICATION')
    print('  (Nonlinear regression on raw data + significance testing)')
    print('=' * 70)
    print(f'\nTrue parameters: Vmax={Vmax_true}, Km={Km_true}, Ki={Ki_true}')
    print(f'Inhibitor concentration [I] = {I_conc}')

    # --- 1. Control data (no inhibitor) ---
    v_ctrl = michaelis_menten(S_data, Vmax_true, Km_true)
    v_ctrl += np.random.normal(0, noise, size=v_ctrl.shape)

    # --- 2. Simulate Competitive Inhibition ---
    Km_app_comp = Km_true * (1 + I_conc / Ki_true)
    v_comp = michaelis_menten(S_data, Vmax_true, Km_app_comp)
    v_comp += np.random.normal(0, noise, size=v_comp.shape)

    # --- 3. Simulate Non-competitive Inhibition ---
    Vmax_app_non = Vmax_true / (1 + I_conc / Ki_true)
    v_non = michaelis_menten(S_data, Vmax_app_non, Km_true)
    v_non += np.random.normal(0, noise, size=v_non.shape)

    # --- 4. Simulate Uncompetitive Inhibition ---
    factor = 1 + I_conc / Ki_true
    Vmax_app_un = Vmax_true / factor
    Km_app_un = Km_true / factor
    v_un = michaelis_menten(S_data, Vmax_app_un, Km_app_un)
    v_un += np.random.normal(0, noise, size=v_un.shape)

    # Run analysis for each inhibition type
    run_inhibition_analysis(S_data, v_ctrl, v_comp, I_conc, 'Competitive Inhibition')
    run_inhibition_analysis(S_data, v_ctrl, v_non, I_conc, 'Non-competitive Inhibition')
    run_inhibition_analysis(S_data, v_ctrl, v_un, I_conc, 'Uncompetitive Inhibition')

    # Also run basic analysis from before on the control data
    print(f'\n{"="*60}')
    print('  BASIC MICHAELIS-MENTEN ANALYSIS (Control Data)')
    print(f'{"="*60}\n')

    popt_u, pcov_u, perr_u = fit_michaelis_menten(S_data, v_ctrl)
    print_result('Uniform (equal) weights', popt_u, perr_u, S_data, v_ctrl)
    print()

    w_prop = variance_weights(v_ctrl, kind='proportional')
    popt_p, pcov_p, perr_p = fit_michaelis_menten(
        S_data, v_ctrl, weights=w_prop, absolute_sigma=False,
    )
    print_result('Var(v) ∝ v (proportional)', popt_p, perr_p, S_data, v_ctrl)
    print()

    w_pow = variance_weights(v_ctrl, kind='power')
    popt_v, pcov_v, perr_v = fit_michaelis_menten(
        S_data, v_ctrl, weights=w_pow, absolute_sigma=False,
    )
    print_result('Var(v) ∝ v^2 (relative)', popt_v, perr_v, S_data, v_ctrl)

    inv_S = 1.0 / S_data
    inv_v = 1.0 / v_ctrl
    slope, intercept = np.polyfit(inv_S, inv_v, 1)
    Km_lb = slope / intercept
    Vmax_lb = 1.0 / intercept
    print(f'\n[Lineweaver-Burk (reference only, overweight low-[S] points)]')
    print(f'  Vmax (LB) = {Vmax_lb:.4f}, Km (LB) = {Km_lb:.4f}')

    plot_fit(S_data, v_ctrl, popt_u, perr_u)
    plt.savefig('michaelis_menten_fit.png', dpi=120)
    print('\nPlot (uniform weights) saved to: michaelis_menten_fit.png')


if __name__ == '__main__':
    main()
