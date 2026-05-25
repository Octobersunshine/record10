import numpy as np
import matplotlib.pyplot as plt
from eis_fitting_advanced import (
    load_eis_from_csv,
    fit_eis,
    randles_impedance,
    plot_nyquist,
    plot_bode,
    plot_residuals,
    calculate_statistics
)


def main():
    print("=" * 70)
    print("FITTING YOUR EIS DATA")
    print("=" * 70)
    
    f, Z = load_eis_from_csv(
        'sample_eis_data.csv',
        delimiter=',',
        f_col=0,
        z_real_col=1,
        z_imag_col=2,
        imag_sign='negative'
    )
    
    print(f"\nLoaded {len(f)} data points")
    print(f"Frequency range: {f.min():.2e} Hz to {f.max():.2e} Hz")
    
    print(f"\nFitting with Randles model (Rs + Rct || Cdl) using Modulus weighting (1/|Z|²)...")
    result = fit_eis(f, Z, model='randles', weighting='modulus')
    
    Rs_fit, Rct_fit, Cdl_fit = result.x
    Z_fit = randles_impedance(f, Rs_fit, Rct_fit, Cdl_fit)
    
    print(f"\n{'=' * 70}")
    print("FITTING RESULTS")
    print(f"{'=' * 70}")
    print(f"\nWeighting method: {result.weighting}")
    print(f"\nFitted Parameters:")
    print(f"  Rs  = {Rs_fit:.4f} Ω (Solution resistance)")
    print(f"  Rct = {Rct_fit:.4f} Ω (Charge transfer resistance)")
    print(f"  Cdl = {Cdl_fit:.4e} F (Double layer capacitance)")
    
    chi_sq, chi_sq_red, w_chi_sq, w_chi_sq_red, rmse, dof = calculate_statistics(
        Z, Z_fit, 3, result.weights
    )
    print(f"\nGoodness of Fit:")
    print(f"  Chi-squared (χ²)      = {chi_sq:.6f}")
    print(f"  Reduced χ²            = {chi_sq_red:.6f}")
    print(f"  Weighted χ²           = {w_chi_sq:.6f}")
    print(f"  Weighted Reduced χ²   = {w_chi_sq_red:.6f}")
    print(f"  RMSE                  = {rmse:.6f}")
    print(f"  Degrees of freedom    = {dof}")
    
    print(f"\nOptimization: {result.message}")
    
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    plot_nyquist(Z, Z_fit, title='Nyquist Plot - Your Data (Modulus Weighting)', ax=ax1)
    fig1.savefig('your_data_nyquist.png', dpi=300, bbox_inches='tight')
    
    fig2 = plot_bode(f, Z, Z_fit, title='Bode Plot - Your Data (Modulus Weighting)')
    fig2.savefig('your_data_bode.png', dpi=300, bbox_inches='tight')
    
    fig3 = plot_residuals(f, Z, Z_fit, title='Residuals - Your Data', weights=result.weights)
    fig3.savefig('your_data_residuals.png', dpi=300, bbox_inches='tight')
    
    print(f"\nPlots saved:")
    print(f"  - your_data_nyquist.png")
    print(f"  - your_data_bode.png")
    print(f"  - your_data_residuals.png")
    
    np.savetxt(
        'fitting_results.csv',
        np.column_stack([f, Z.real, Z.imag, Z_fit.real, Z_fit.imag]),
        delimiter=',',
        header='frequency_hz,Z_exp_real,Z_exp_imag,Z_fit_real,Z_fit_imag',
        comments=''
    )
    print(f"\nFitted data saved to fitting_results.csv")
    
    with open('fitted_parameters.txt', 'w') as f_out:
        f_out.write("EIS Fitting Results - Randles Circuit\n")
        f_out.write("=" * 50 + "\n\n")
        f_out.write("Fitted Parameters:\n")
        f_out.write(f"  Rs  = {Rs_fit:.6f} Ω\n")
        f_out.write(f"  Rct = {Rct_fit:.6f} Ω\n")
        f_out.write(f"  Cdl = {Cdl_fit:.6e} F\n\n")
        f_out.write("Goodness of Fit:\n")
        f_out.write(f"  Chi-squared = {chi_sq:.6f}\n")
        f_out.write(f"  Reduced χ²  = {chi_sq_red:.6f}\n")
        f_out.write(f"  RMSE        = {rmse:.6f}\n")
    print(f"Parameters saved to fitted_parameters.txt")
    
    print(f"\n{'=' * 70}")
    print("COMPLETE!")
    print(f"{'=' * 70}")
    
    plt.close('all')


if __name__ == "__main__":
    main()
