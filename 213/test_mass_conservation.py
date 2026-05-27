"""
Test mass conservation: Original IMPES vs IMPES with Correction
"""
import numpy as np
import matplotlib.pyplot as plt
from black_oil_impes import (
    Grid, RockProperties, BlackOilPVT, RelativePermeability,
    CapillaryPressure, Well, IMPESSolver
)


def run_simulation(use_correction: bool, t_days: int = 30) -> dict:
    """Run simulation with or without mass conservation correction."""
    nx, ny = 12, 12
    grid = Grid(nx=nx, ny=ny, dx=50, dy=50, dz=10)
    
    permx = np.full(grid.ncells, 500.0)
    permy = np.full(grid.ncells, 500.0)
    porosity = np.full(grid.ncells, 0.22)
    
    np.random.seed(42)
    for idx in range(grid.ncells):
        noise = np.random.uniform(0.8, 1.2)
        permx[idx] *= noise
        permy[idx] *= noise
    
    rock = RockProperties(grid, permx=permx, permy=permy, porosity=porosity)
    pvt = BlackOilPVT()
    relperm = RelativePermeability()
    cap_pres = CapillaryPressure()
    
    wells = [
        Well('PROD', i=9, j=6, well_type='producer', control_mode='bhp', target_bhp=150e5),
        Well('INJ', i=2, j=6, well_type='injector', control_mode='rate', target_rate=300, phases=['water'])
    ]
    
    solver = IMPESSolver(
        grid=grid, rock=rock, pvt=pvt, relperm=relperm,
        cap_pres=cap_pres, wells=wells,
        p_init=200e5, sw_init=0.15, sg_init=0.0,
        dt=1.0, t_max=t_days,
        use_mass_correction=use_correction
    )
    
    results = solver.run()
    results['use_correction'] = use_correction
    results['solver'] = solver
    
    return results


def analyze_results(results: dict) -> dict:
    """Analyze mass balance errors."""
    solver = results['solver']
    mb_errors = solver.mass_balance_errors
    
    if len(mb_errors) == 0:
        return {}
    
    if isinstance(mb_errors[0], dict):
        water_errs = [e['water'] for e in mb_errors]
        oil_errs = [e['oil'] for e in mb_errors]
    else:
        water_errs = [np.mean(np.abs(e)) for e in mb_errors]
        oil_errs = water_errs
    
    return {
        'water_mean': np.mean(water_errs),
        'water_max': np.max(water_errs),
        'water_final': water_errs[-1],
        'oil_mean': np.mean(oil_errs),
        'oil_max': np.max(oil_errs),
        'errors': water_errs
    }


def main():
    print("=" * 70)
    print("  Mass Conservation Test")
    print("=" * 70)
    
    t_days = 30
    
    print(f"\nRunning {t_days} day simulation...")
    
    print("\n" + "-" * 70)
    print("  Running: Original IMPES (no correction)")
    print("-" * 70)
    results_orig = run_simulation(use_correction=False, t_days=t_days)
    analysis_orig = analyze_results(results_orig)
    
    print("\n" + "-" * 70)
    print("  Running: IMPES with Mass Conservation Correction")
    print("-" * 70)
    results_corr = run_simulation(use_correction=True, t_days=t_days)
    analysis_corr = analyze_results(results_corr)
    
    print("\n" + "=" * 70)
    print("  Mass Balance Comparison")
    print("=" * 70)
    
    print(f"\n  {'Metric':<25} {'Original IMPES':>18} {'Corrected IMPES':>18}")
    print("  " + "-" * 61)
    print(f"  {'Water error (mean)':<25} {analysis_orig['water_mean']:>18.2e} {analysis_corr['water_mean']:>18.2e}")
    print(f"  {'Water error (max)':<25} {analysis_orig['water_max']:>18.2e} {analysis_corr['water_max']:>18.2e}")
    print(f"  {'Water error (final)':<25} {analysis_orig['water_final']:>18.2e} {analysis_corr['water_final']:>18.2e}")
    
    improvement = (analysis_orig['water_mean'] - analysis_corr['water_mean']) / analysis_orig['water_mean'] * 100
    print(f"\n  Mass conservation improvement: {improvement:.1f}%")
    
    print("\n" + "=" * 70)
    print("  Production Comparison")
    print("=" * 70)
    
    for name, results in [('Original IMPES', results_orig), ('Corrected IMPES', results_corr)]:
        production = results['production']
        print(f"\n  {name}:")
        for well_name, prod_data in production.items():
            if len(prod_data) > 0:
                t_arr = np.array([d['time'] for d in prod_data])
                dt_arr = np.diff(t_arr, prepend=t_arr[0])
                cum_oil = np.sum(np.array([d['oil_rate'] for d in prod_data]) * dt_arr)
                cum_water = np.sum(np.array([d['water_rate'] for d in prod_data]) * dt_arr)
                print(f"    {well_name}: Oil={cum_oil:.0f} m³, Water={cum_water:.0f} m³")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if 'errors' in analysis_orig and 'errors' in analysis_corr:
        axes[0].semilogy(analysis_orig['errors'], 'r-', label='Original IMPES', alpha=0.7)
        axes[0].semilogy(analysis_corr['errors'], 'b-', label='Corrected IMPES', alpha=0.7)
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Water Mass Balance Error')
        axes[0].set_title('Mass Balance Error Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    prod_orig = results_orig['production']['PROD']
    prod_corr = results_corr['production']['PROD']
    
    if len(prod_orig) > 0:
        t_orig = [d['time'] for d in prod_orig]
        oil_orig = [d['oil_rate'] for d in prod_orig]
        wc_orig = [d['water_cut'] for d in prod_orig]
        
        t_corr = [d['time'] for d in prod_corr]
        oil_corr = [d['oil_rate'] for d in prod_corr]
        wc_corr = [d['water_cut'] for d in prod_corr]
        
        ax2 = axes[1].twinx()
        axes[1].plot(t_orig, oil_orig, 'r-', label='Oil (Original)', alpha=0.5)
        axes[1].plot(t_corr, oil_corr, 'b-', label='Oil (Corrected)', alpha=0.7)
        ax2.plot(t_orig, wc_orig, 'r--', label='WC (Original)', alpha=0.5)
        ax2.plot(t_corr, wc_corr, 'b--', label='WC (Corrected)', alpha=0.7)
        axes[1].set_xlabel('Time (days)')
        axes[1].set_ylabel('Oil Rate (m³/day)')
        ax2.set_ylabel('Water Cut')
        axes[1].set_title('Production Comparison')
        axes[1].legend(loc='lower left')
        ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('mass_conservation_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n  Comparison plot saved to mass_conservation_comparison.png")
    
    print("\n" + "=" * 70)
    print("  Test complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
