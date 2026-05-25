from afm_fitting import AFMCurveFitter, load_afm_data
import numpy as np


def fit_from_csv(csv_file, model='MD', radius=10e-9, nu=0.5, z0=0.3e-9):
    """
    Fit AFM data from a CSV file using Maugis-Dugdale model (default)
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file with columns: indentation (m), force (N)
    model : str
        'MD' (recommended), 'DMT', or 'JKR'
    radius : float
        Tip radius in meters
    nu : float
        Poisson's ratio
    z0 : float
        Interatomic distance (for MD model only)
    """
    print(f"Loading data from: {csv_file}")
    delta, force = load_afm_data(csv_file)
    
    if delta is None:
        print("Failed to load data!")
        return
    
    print(f"Loaded {len(delta)} data points")
    print(f"Indentation range: {delta.min()*1e9:.2f} to {delta.max()*1e9:.2f} nm")
    print(f"Force range: {force.min()*1e9:.2f} to {force.max()*1e9:.2f} nN")
    print()
    
    fitter = AFMCurveFitter(radius=radius, nu=nu, z0=z0)
    
    print(f"Fitting with {model} model...")
    popt, pcov = fitter.fit(delta, force, model=model)
    
    if popt is None:
        print("Fitting failed!")
        return
    
    fitter.print_results()
    
    r2 = fitter.calculate_goodness_of_fit(delta, force)
    print(f"Goodness of fit (R²): {r2:.4f}")
    
    output_file = csv_file.replace('.csv', f'_{model}_fit.png')
    fitter.plot_results(delta, force, save_path=output_file)
    print(f"Plot saved to: {output_file}")
    
    return fitter


def compare_models_on_data(csv_file, radius=10e-9, nu=0.5, z0=0.3e-9):
    """
    Compare all three models on the same dataset
    """
    print("=" * 60)
    print("Comparing DMT, JKR, and MD models")
    print("=" * 60)
    
    delta, force = load_afm_data(csv_file)
    if delta is None:
        return
    
    models = ['DMT', 'JKR', 'MD']
    results = {}
    
    for model in models:
        print(f"\nFitting with {model} model...")
        fitter = AFMCurveFitter(radius=radius, nu=nu, z0=z0)
        fitter.fit(delta, force, model=model)
        r2 = fitter.calculate_goodness_of_fit(delta, force)
        results[model] = {
            'E': fitter.E,
            'F_ad': fitter.F_ad,
            'R2': r2,
            'fitter': fitter
        }
        print(f"  E: {fitter.E:.2e} Pa")
        print(f"  F_ad: {fitter.F_ad:.2e} N")
        print(f"  R²: {r2:.6f}")
    
    best_model = max(results, key=lambda x: results[x]['R2'])
    print(f"\nBest model based on R²: {best_model}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    print("AFM Force-Distance Curve Fitting with Maugis-Dugdale Model")
    print("=" * 60)
    
    fitter_md = fit_from_csv('sample_afm_data.csv', model='MD', 
                             radius=10e-9, nu=0.5, z0=0.3e-9)
    
    print("\n" + "=" * 60 + "\n")
    
    results = compare_models_on_data('sample_afm_data.csv', 
                                     radius=10e-9, nu=0.5, z0=0.3e-9)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
