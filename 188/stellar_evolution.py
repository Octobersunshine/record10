"""
Massive Star Evolution with Mass Loss and Metallicity Effects
=============================================================

This script computes evolutionary tracks on the Hertzsprung-Russell (HR)
diagram for stars of arbitrary mass (1-120 M_sun), including:

  (1) Metallicity (Z) dependence:
        - Opacity scaling:  kappa ~ Z^0.6 (empirical fit for stellar envelopes)
        - CNO reaction rate enhancement with Z
        - Mass loss rate scaling:  dM/dt ~ (Z/Z_sun)^0.5

  (2) Mass loss (stellar winds):
        - Reimers' law for cool giants / red supergiants (RSG)
        - Vink et al. (2001) prescription for hot OB stars
        - Smooth interpolation between regimes

  (3) Nuclear burning:
        - pp chain (low-mass stars, T_c < 2e7 K)
        - CNO cycle (massive stars, T_c > 2e7 K, rate ~ T^15)

  (4) Evolutionary phases for M > 8 M_sun:
        - Main sequence (H core burning)
        - Blue supergiant (BSG) / red supergiant (RSG) transition
        - He core burning (Wolf-Rayet phase if envelope stripped)
        - Termination at supernova progenitor (core mass > Chandrasekhar limit)

Usage
-----
    python stellar_evolution_massive.py
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (cgs internally)
# ---------------------------------------------------------------------------
c = 2.9979e10         # cm s^-1
G = 6.674e-8          # cm^3 g^-1 s^-2
M_sun = 1.989e33      # g
R_sun = 6.957e10      # cm
L_sun = 3.828e33      # erg s^-1
sigma_SB = 5.6704e-5  # erg cm^-2 s^-1 K^-4
k_B = 1.3806e-16      # erg K^-1
m_u = 1.6726e-24      # g
Q_H = 6.3e18          # erg g^-1  (hydrogen burning)
Q_He = 6.0e17         # erg g^-1  (helium burning)
M_Ch = 1.44 * M_sun   # Chandrasekhar mass

# Solar composition
X_sun = 0.70
Y_sun = 0.28
Z_sun = 0.02

# ---------------------------------------------------------------------------
# Convection: Schwarzschild + MLT (from previous version, now Z-aware)
# ---------------------------------------------------------------------------
alpha_MLT = 1.7
gamma_ad = 5.0 / 3.0
nabla_ad = (gamma_ad - 1.0) / gamma_ad
SCHWARZSCHILD_WIDTH = 0.02


def schwarzschild_transition(nabla_rad):
    x = (nabla_rad - nabla_ad) / SCHWARZSCHILD_WIDTH
    f_conv = 1.0 / (1.0 + np.exp(-x))
    return 1.0 - f_conv, f_conv


def mlt_nabla(nabla_rad):
    f_rad, f_conv = schwarzschild_transition(nabla_rad)
    A_MLT = 0.15
    delta_nabla = nabla_ad * (f_conv * A_MLT) ** 2
    nabla = nabla_ad + delta_nabla
    nabla = min(nabla, max(nabla_rad, nabla_ad))
    return nabla, f_rad, f_conv


# ---------------------------------------------------------------------------
# Mass loss prescriptions (wind)
# ---------------------------------------------------------------------------
def mass_loss_rate(M, L, R, T_eff, Z):
    """
    Stellar mass loss rate in g s^-1.

    Two regimes with smooth interpolation:

      * Hot stars (T_eff > 10^4 K): Vink et al. (2001) OB-star winds
            log(dM/dt) ~ log(L/L_sun) + 0.5 * log(Z/Z_sun) - ...
            (calibrated to ~1e-6 M_sun/yr for O7 V, Z=Z_sun)

      * Cool stars (T_eff < 4000 K): Reimers' law
            dM/dt = 4e-13 * eta * (L / L_sun) * (R / R_sun) / (M / M_sun)
            with eta = 0.5 for RGB, eta = 3 for RSG.

    In between, we interpolate linearly in log(dM/dt).
    """
    Z_ratio = Z / Z_sun if Z > 0 else 0.0
    Z_ratio = max(Z_ratio, 0.001)

    # --- Vink prescription (hot) ---
    if T_eff > 1.0e4:
        log_L_5 = np.log10(L / (1e5 * L_sun))
        log_M_30 = np.log10(M / (30 * M_sun))
        # Vink et al. (2001), normalized to typical O stars
        # For L=1e5 L_sun, M=30 M_sun: dM/dt ~ 1e-6 M_sun/yr
        log_dMdt = -6.3 + 1.8 * log_L_5 - 1.2 * log_M_30 + 0.5 * np.log10(Z_ratio)
        # clamp to reasonable range
        log_dMdt = min(max(log_dMdt, -9.0), -4.0)
        dMdt_Vink = 10.0 ** log_dMdt * M_sun / (3.1558e7)  # M_sun/yr -> g/s
    else:
        dMdt_Vink = None

    # --- Reimers prescription (cool) ---
    if T_eff < 6.0e3:
        eta = 0.5
        # boost for red supergiants
        if T_eff < 4000 and L / L_sun > 1000:
            eta = 1.5
        dMdt_Reimers = 4e-13 * eta * (L / L_sun) * (R / R_sun) / (M / M_sun)
        dMdt_Reimers = dMdt_Reimers * M_sun / (3.1558e7)
        dMdt_Reimers *= Z_ratio ** 0.5
    else:
        dMdt_Reimers = None

    # --- interpolation weights ---
    T_hot = 12000.0
    T_cool = 4000.0
    if T_eff >= T_hot:
        return max(dMdt_Vink, 1e-20)
    elif T_eff <= T_cool:
        return max(dMdt_Reimers, 1e-20)
    else:
        w = (T_eff - T_cool) / (T_hot - T_cool)
        w = min(max(w, 0.0), 1.0)
        # use log interpolation
        logV = np.log10(max(dMdt_Vink, 1e-25)) if dMdt_Vink is not None else -25
        logR = np.log10(max(dMdt_Reimers, 1e-25)) if dMdt_Reimers is not None else -25
        return 10.0 ** (w * logV + (1 - w) * logR)


# ---------------------------------------------------------------------------
# Metallicity-dependent opacity
# ---------------------------------------------------------------------------
def effective_opacity(T, rho, X, Z):
    """
    Approximate Rosseland mean opacity in cm^2 g^-1.

    For T > 1e6 K: dominated by electron scattering  kappa_es = 0.2*(1+X).
    For T < 1e6 K: H- opacity, roughly scales as  rho^0.5 T^9  and ~ Z^0.6.
    We take the harmonic maximum of these components plus a low-T floor.
    """
    kappa_es = 0.2 * (1.0 + X)
    Z_factor = (Z / Z_sun) ** 0.6 if Z > 0 else 0.0
    # H- opacity approximation (empirical fit)
    if T > 0 and rho > 0:
        kappa_Hminus = 1e-25 * Z_factor * rho ** 0.5 * (T / 3000.0) ** 9
    else:
        kappa_Hminus = 0.0
    # blend smoothly
    return kappa_es + kappa_Hminus


# ---------------------------------------------------------------------------
# Nuclear energy generation (pp chain + CNO cycle)
# ---------------------------------------------------------------------------
def nuclear_energy_generation(rho, T, X, Z, Z_CNO=None):
    """
    Total specific energy generation rate in erg g^-1 s^-1 from:
      * pp chain:   epsilon_pp ~ rho X^2 T^4   (dominant for M <~ 2 M_sun)
      * CNO cycle:  epsilon_CNO ~ rho X Z_CNO T^15   (dominant for M >~ 2 M_sun)

    Z_CNO is the mass fraction of C+N+O; if not given we assume Z_CNO ~ 0.7 * Z.
    """
    if T <= 0 or rho <= 0:
        return 0.0, 0.0, 0.0
    T7 = T / 1e7

    # pp chain (calibrated to solar L from pp ~ 98% of 1 L_sun)
    eps_pp = 0.11 * rho * (X ** 2) * T7 ** 4

    # CNO cycle (strong T^15 dependence, dominant for M > 2 M_sun)
    if Z_CNO is None:
        Z_CNO = 0.7 * Z
    eps_CNO = 3.5e-6 * rho * X * Z_CNO * T7 ** 15.0

    eps_total = eps_pp + eps_CNO
    return eps_total, eps_pp, eps_CNO


# ---------------------------------------------------------------------------
# Stellar structure (Z-aware, massive-star capable)
# ---------------------------------------------------------------------------
def stellar_structure(M, Xc, Z=Z_sun, X=X_sun, Y=Y_sun):
    """
    Compute R, L, T_eff for a star of current mass M and core H fraction Xc.

    This is a homology / parametric model calibrated to reproduce:
        * Solar model at M=M_sun, Xc=0.7, Z=Z_sun
        * M-L relation for OB stars:  L ~ M^3.5 (Z=Z_sun)
        * RSG expansion: R ~ 1000 R_sun for 20 M_sun post-MS

    Returns (R_cm, L_erg_s, T_eff_K, extra_info_dict)
    """
    Yc = Y + (X - Xc) if Xc <= X else Y
    denom = 2 * Xc + 0.75 * Yc + 0.5 * Z
    if denom <= 0:
        denom = 2 * X + 0.75 * Y + 0.5 * Z
    mu = 1.0 / denom
    mu_sun = 1.0 / (2 * X + 0.75 * Y + 0.5 * Z)
    mu = max(mu, 0.5)

    # --- core temperature (virial) ---
    T_c = 1.5e7 * (mu / mu_sun) * (M / M_sun) ** (1.0 / 3.0)
    # boost for massive stars (shallower radiative envelopes => more central concentration)
    if M > 2 * M_sun:
        T_c *= (M / (2 * M_sun)) ** 0.15

    # --- main-sequence radius (homology) ---
    R_MS = R_sun * (M / M_sun) ** 0.75 * (mu_sun / mu) ** 0.5
    # Z-dependence: higher Z => more opacity => larger radius
    Z_factor_R = (Z / Z_sun) ** 0.1
    R_MS *= Z_factor_R

    rho_c_MS = 3 * M / (4 * np.pi * R_MS ** 3) * 10.0  # central concentration

    # --- nuclear luminosity ---
    f_core = min(0.3 * (M / M_sun) ** 0.1, 0.35)
    eps_total, eps_pp, eps_CNO = nuclear_energy_generation(rho_c_MS, T_c, Xc, Z)
    L_nuc = eps_total * f_core * M

    # --- post-MS: red supergiant expansion ---
    f_burned = max(0.0, (X - Xc) / X)
    M_core = f_core * f_burned * M
    M_core = max(M_core, 1e-4 * M_sun)

    # For massive stars, after H exhaustion, envelope swells drastically
    if Xc < 0.35 and M > 8 * M_sun:
        # RSG expansion factor
        expand = 1.0 + (0.35 - Xc) / 0.35 * 100.0
        expand = min(expand, 120.0)
        R_RSG = R_MS * expand
        # L ~ const for RSG (Hayashi track), just slightly increase with M_core
        L_RSG = L_nuc * (1.0 + (0.35 - Xc) / 0.35 * 2.0)
        L_RSG = min(L_RSG, 2.0e6 * L_sun)
    else:
        R_RSG = R_MS
        L_RSG = L_nuc

    # --- smooth cross-over from MS to RSG ---
    X_hi = 0.35 if M > 8 * M_sun else 0.15
    X_lo = 0.05
    if Xc >= X_hi:
        L_erg_s = L_nuc
        R_cm = R_MS
    elif Xc <= X_lo:
        L_erg_s = L_RSG
        R_cm = R_RSG
    else:
        w = (Xc - X_lo) / (X_hi - X_lo)
        L_erg_s = w * L_nuc + (1 - w) * L_RSG
        R_cm = w * R_MS + (1 - w) * R_RSG

    L_erg_s = max(L_erg_s, 1.0)
    R_cm = min(max(R_cm, 0.3 * R_sun), 2000 * R_sun)

    # --- T_eff with MLT correction ---
    F_surface = L_erg_s / (4 * np.pi * R_cm ** 2)
    T_guess = (F_surface / sigma_SB) ** 0.25

    # For RSG phase (T_eff < 5000 K), enforce Hayashi track floor
    # Cool stars can't go below ~3000-3500 K (convective envelope)
    if Xc < 0.3 and M > 8 * M_sun:
        # Cool down to RSG temperatures
        T_guess = min(T_guess, 4000.0)
        T_guess = max(T_guess, 3200.0)

    P_surface = G * M / (R_cm ** 2) * max(R_cm / (R_sun * 1e8), 1.0)
    kappa = effective_opacity(T_guess, F_surface / (c * 3.0), X, Z) if 'c' in globals() else 0.34
    # estimate nabla_rad
    F_sun = L_sun / (4 * np.pi * R_sun ** 2)
    P_sun = G * M_sun / R_sun ** 2
    nabla_rad = 0.4 * (F_surface / F_sun) / (P_surface / P_sun + 1e-6)
    nabla_rad = min(max(nabla_rad, 0.10), 2.0)

    nabla_actual, f_rad, f_conv = mlt_nabla(nabla_rad)
    T_eff_K = T_guess * (nabla_actual / 0.25) ** 0.18

    # metallicity effect on T_eff: higher Z => cooler for same L,R
    T_eff_K *= (Z / Z_sun) ** (-0.03)

    extra = {
        "T_c": T_c,
        "M_core": M_core,
        "f_conv": f_conv,
        "eps_pp": eps_pp,
        "eps_CNO": eps_CNO,
        "Z": Z,
    }
    return R_cm, L_erg_s, T_eff_K, extra


# ---------------------------------------------------------------------------
# Evolution ODEs: dXc/dt (nucleosynthesis) + dM/dt (mass loss)
# ---------------------------------------------------------------------------
def compute_derivs(t, state, Z=Z_sun):
    M_curr, Xc_curr = state
    if M_curr <= 0 or Xc_curr <= 0:
        return [0.0, 0.0]
    X_env = X_sun  # envelope H (kept fixed in this simple model)
    R, L, T_eff, extra = stellar_structure(M_curr, Xc_curr, Z=Z, X=X_env)

    # mass loss
    dMdt = -mass_loss_rate(M_curr, L, R, T_eff, Z)
    dMdt = max(dMdt, -M_curr / (1e5 * 3.1558e7))  # cap at M-loss timescale > 1e5 yr

    # nuclear burning of core H
    f_core = 0.12
    if M_curr > 2 * M_sun:
        f_core = 0.12 * (M_curr / (2 * M_sun)) ** 0.3
    eps_core = L / (f_core * M_curr) if f_core * M_curr > 0 else 0.0
    dXc_dt = -eps_core / Q_H if Xc_curr > 0 else 0.0

    return [dMdt, dXc_dt]


# ---------------------------------------------------------------------------
# Evolution integrator (with RK4 and adaptive sub-stepping)
# ---------------------------------------------------------------------------
def evolve_massive(M_init, Z=Z_sun, t_end_yr=2e7, n_steps=600):
    """
    Integrate the evolution of a massive star from ZAMS until it reaches
    the supernova progenitor stage (core mass > M_Ch - tiny, or H exhausted).

    M_init:  initial total mass [g]
    Z:       metallicity
    t_end_yr: maximum integration time (years)
    """
    t_arr = np.logspace(5, np.log10(t_end_yr), n_steps)  # years
    t_arr_s = t_arr * 3.1558e7

    M_arr = np.zeros_like(t_arr_s)
    Xc_arr = np.zeros_like(t_arr_s)
    L_arr = np.zeros_like(t_arr_s)
    Teff_arr = np.zeros_like(t_arr_s)
    R_arr = np.zeros_like(t_arr_s)
    M_core_arr = np.zeros_like(t_arr_s)

    state = np.array([M_init, X_sun], dtype=float)
    M_arr[0] = M_init
    Xc_arr[0] = X_sun
    R0, L0, T0, extra0 = stellar_structure(M_init, X_sun, Z=Z)
    L_arr[0] = L0
    Teff_arr[0] = T0
    R_arr[0] = R0
    M_core_arr[0] = extra0["M_core"]

    terminated = False
    term_reason = ""

    for i in range(1, len(t_arr_s)):
        if terminated:
            M_arr[i] = M_arr[i - 1]
            Xc_arr[i] = Xc_arr[i - 1]
            L_arr[i] = L_arr[i - 1]
            Teff_arr[i] = Teff_arr[i - 1]
            R_arr[i] = R_arr[i - 1]
            M_core_arr[i] = M_core_arr[i - 1]
            continue

        dt = t_arr_s[i] - t_arr_s[i - 1]
        n_sub = 50
        h = dt / n_sub

        for _ in range(n_sub):
            # RK4
            k1 = np.array(compute_derivs(t_arr_s[i], state, Z))
            k2 = np.array(compute_derivs(t_arr_s[i] + 0.5 * h, state + 0.5 * h * k1, Z))
            k3 = np.array(compute_derivs(t_arr_s[i] + 0.5 * h, state + 0.5 * h * k2, Z))
            k4 = np.array(compute_derivs(t_arr_s[i] + h, state + h * k3, Z))
            state = state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            state[0] = max(state[0], 0.1 * M_sun)
            state[1] = max(state[1], 0.0)

            # check termination conditions
            R, L, T, extra = stellar_structure(state[0], state[1], Z=Z)
            if extra["M_core"] >= 1.37 * M_sun and state[1] < 0.02:
                terminated = True
                term_reason = f"Core collapse SN progenitor (M_core={extra['M_core']/M_sun:.2f} M_sun)"
                break
            if state[1] <= 0.001:
                terminated = True
                term_reason = "Core H exhausted (He ignition)"
                break
            if state[0] < 2.0 * M_sun and state[1] < 0.1:
                terminated = True
                term_reason = "Envelope nearly stripped (Wolf-Rayet?)"
                break

        M_arr[i] = state[0]
        Xc_arr[i] = state[1]
        R, L, T, extra = stellar_structure(state[0], state[1], Z=Z)
        L_arr[i] = L
        Teff_arr[i] = T
        R_arr[i] = R
        M_core_arr[i] = extra["M_core"]

        if terminated:
            # trim arrays to actual length
            M_arr = M_arr[:i + 1]
            Xc_arr = Xc_arr[:i + 1]
            L_arr = L_arr[:i + 1]
            Teff_arr = Teff_arr[:i + 1]
            R_arr = R_arr[:i + 1]
            M_core_arr = M_core_arr[:i + 1]
            t_arr = t_arr[:i + 1] / 1e6
            break

    if not terminated:
        t_arr = t_arr / 1e6
        term_reason = "Reached maximum integration time"

    return {
        "age_Myr": t_arr,
        "M_sun": M_arr / M_sun,
        "L_sun": L_arr / L_sun,
        "T_eff_K": Teff_arr,
        "R_sun": R_arr / R_sun,
        "Xc": Xc_arr,
        "M_core_sun": M_core_arr / M_sun,
        "termination_reason": term_reason,
        "Z": Z,
    }


# ---------------------------------------------------------------------------
# Main: compute track for a 20 M_sun star
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 78)
    print("Massive Star Evolution (M_init = 20 M_sun, Z = Z_sun)")
    print("=" * 78)
    track = evolve_massive(20 * M_sun, Z=Z_sun, t_end_yr=1e8, n_steps=600)

    idx_show = np.linspace(0, len(track["age_Myr"]) - 1, min(12, len(track["age_Myr"])), dtype=int)
    print(f"{'Age [Myr]':>11s} {'M [M_sun]':>11s} {'L [L_sun]':>12s} "
          f"{'T_eff [K]':>11s} {'R [R_sun]':>11s} {'Xc':>7s} {'M_core':>8s}")
    print("-" * 78)
    for i in idx_show:
        print(f"{track['age_Myr'][i]:11.3f} {track['M_sun'][i]:11.3f} "
              f"{track['L_sun'][i]:12.3f} {track['T_eff_K'][i]:11.0f} "
              f"{track['R_sun'][i]:11.2f} {track['Xc'][i]:7.4f} "
              f"{track['M_core_sun'][i]:8.3f}")
    print("=" * 78)
    print(f"Termination: {track['termination_reason']}")

    # Save
    np.savez("massive_star_track.npz",
             age_Myr=track["age_Myr"],
             M_sun=track["M_sun"],
             log_L=np.log10(track["L_sun"]),
             log_Teff=np.log10(track["T_eff_K"]),
             R_sun=track["R_sun"],
             Xc=track["Xc"],
             M_core_sun=track["M_core_sun"],
             Z=track["Z"])
    print("\nTrack saved to massive_star_track.npz")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(np.log10(track["T_eff_K"]),
                        np.log10(track["L_sun"]),
                        c=track["age_Myr"], cmap="plasma", s=12, edgecolors="none")
        ax.invert_xaxis()
        ax.set_xlabel(r"$\log T_{\mathrm{eff}}$ [K]")
        ax.set_ylabel(r"$\log (L / L_\odot)$")
        ax.set_title(r"20 $M_\odot$ Evolution Track  (Z = $Z_\odot$)")
        cbar = plt.colorbar(sc)
        cbar.set_label("Age [Myr]")

        # Annotate key phases
        ax.text(4.8, 4.6, "ZAMS", fontsize=9)
        if track["R_sun"][-1] > 500:
            ax.text(3.5, 5.5, "RSG", fontsize=9, color="red")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("HR_massive_20Msun.png", dpi=150)
        print("HR diagram saved to HR_massive_20Msun.png")
    except ImportError:
        print("matplotlib not available; skipping plot.")
