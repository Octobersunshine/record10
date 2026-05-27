"""
Comparison script: IMPES vs FIM mass conservation
"""
import numpy as np
import matplotlib.pyplot as plt
from black_oil_impes import (
    Grid, RockProperties, BlackOilPVT, RelativePermeability,
    CapillaryPressure, Well, IMPESSolver, FIMSolver, visualize_results
)


def setup_case(nx=10, ny=10, t_days=30, dt=1.0):
    """Setup common simulation case."""
    dx, dy, dz = 50.0, 50.0, 10.0
    grid = Grid(nx=nx, ny=ny, dx=dx, dy=dy, dz=dz)

    permx = np.full(grid.ncells, 500.0)
    permy = np.full(grid.ncells, 500.0)
    porosity = np.full(grid.ncells, 0.22)

    np.random.seed(42)
    for idx in range(grid.ncells):
        noise = np.random.uniform(0.8, 1.2)
        permx[idx] *= noise
        permy[idx] *= noise

    rock = RockProperties(grid, permx=permx, permy=permy, porosity=porosity)

    pvt = BlackOilPVT(
        p_ref=200.0e5, bo_ref=1.3, co=1.5e-9, mu_o_ref=0.8,
        mu_g=0.015, mu_w=0.4, bw=1.0, rs_max=150.0, pb=180.0e5,
    )

    relperm = RelativePermeability(
        swc=0.10, sor=0.10, sgc=0.05,
        kro_max=1.0, krw_max=0.8, krg_max=0.9,
        no=2.0, nw=2.0, ng=2.0,
    )

    cap_pres = CapillaryPressure(p_entry_water=3000.0, p_entry_gas=2000.0)

    wells = []
    prod_well = Well(
        name='PROD-1', i=nx-2, j=ny//2,
        well_type='producer', control_mode='bhp', target_bhp=150.0e5,
    )
    wells.append(prod_well)

    inj_well = Well(
        name='INJ-1', i=1, j=ny//2,
        well_type='injector', control_mode='rate',
        target_rate=300.0, phases=['water'],
    )
    wells.append(inj_well)

    return grid, rock, pvt, relperm, cap_pres, wells


def run_impes(nx=10, ny=10, t_days=30, dt=1.0):
    """Run IMPES simulation."""
    grid, rock, pvt, relperm, cap_pres, wells = setup_case(nx, ny, t_days, dt)

    solver = IMPESSolver(
        grid=grid, rock=rock, pvt=pvt, relperm=relperm,
        cap_pres=cap_pres, wells=wells,
        p_init=200.0e5, sw_init=0.15, sg_init=0.0,
        dt=dt, t_max=t_days,
    )

    results = solver.run()
    results['solver_type'] = 'IMPES'
    return results, solver


def run_fim(nx=10, ny=10, t_days=30, dt=1.0):
    """Run FIM simulation."""
    grid, rock, pvt, relperm, cap_pres, wells = setup_case(nx, ny, t_days, dt)

    solver = FIMSolver(
        grid=grid, rock=rock, pvt=pvt, relperm=relperm,
        cap_pres=cap_pres, wells=wells,
        p_init=200.0e5, sw_init=0.15, sg_init=0.0,
        dt=dt, t_max=t_days,
        max_newton_iter=20, tol_residual=1e-4,
    )

    results = solver.run()
    results['solver_type'] = 'FIM'
    return results, solver


def analyze_mass_balance(results, solver):
    """Analyze mass balance errors."""
    mb_errors = results.get('mass_balance_errors', [])
    
    if len(mb_errors) == 0:
        return None
    
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
    }


def main():
    print("=" * 70)
    print("  Mass Conservation Comparison: IMPES vs FIM")
    print("=" * 70)
    
    t_days = 30
    dt = 1.0
    
    print(f"\nRunning {t_days} day simulation with dt={dt} day...")
    
    print("\n" + "-" * 70)
    print("  Running IMPES...")
    print("-" * 70)
    results_impes, solver_impes = run_impes(t_days=t_days, dt=dt)
    
    print("\n" + "-" * 70)
    print("  Running FIM...")
    print("-" * 70)
    results_fim, solver_fim = run_fim(t_days=t_days, dt=dt)
    
    print("\n" + "=" * 70)
    print("  Mass Balance Analysis")
    print("=" * 70)
    
    mb_impes = analyze_mass_balance(results_impes, solver_impes)
    mb_fim = analyze_mass_balance(results_fim, solver_fim)
    
    if mb_impes and mb_fim:
        print(f"\n  {'Metric':<25} {'IMPES':>15} {'FIM':>15}")
        print("  " + "-" * 55)
        print(f"  {'Water error (mean)':<25} {mb_impes['water_mean']:>15.2e} {mb_fim['water_mean']:>15.2e}")
        print(f"  {'Water error (max)':<25} {mb_impes['water_max']:>15.2e} {mb_fim['water_max']:>15.2e}")
        print(f"  {'Water error (final)':<25} {mb_impes['water_final']:>15.2e} {mb_fim['water_final']:>15.2e}")
        
        improvement = (mb_impes['water_mean'] - mb_fim['water_mean']) / mb_impes['water_mean'] * 100
        print(f"\n  FIM improvement in water mass conservation: {improvement:.1f}%")
    
    print("\n" + "=" * 70)
    print("  Production Comparison")
    print("=" * 70)
    
    for name, results in [('IMPES', results_impes), ('FIM', results_fim)]:
        production = results['production']
        print(f"\n  {name}:")
        for well_name, prod_data in production.items():
            if len(prod_data) > 0:
                final = prod_data[-1]
                t_arr = np.array([d['time'] for d in prod_data])
                dt_arr = np.diff(t_arr, prepend=t_arr[0])
                cum_oil = np.sum(np.array([d['oil_rate'] for d in prod_data]) * dt_arr)
                cum_water = np.sum(np.array([d['water_rate'] for d in prod_data]) * dt_arr)
                print(f"    {well_name}: Oil={cum_oil:.0f} m³, Water={cum_water:.0f} m³")
    
    print("\n" + "=" * 70)
    
    visualize_results(results_impes, output_dir=".")
    plt.savefig('impes_results.png', dpi=150, bbox_inches='tight')
    print("  IMPES visualization saved to impes_results.png")
    
    visualize_results(results_fim, output_dir=".")
    plt.savefig('fim_results.png', dpi=150, bbox_inches='tight')
    print("  FIM visualization saved to fim_results.png")
    
    print("\n  Comparison complete!")


if __name__ == '__main__':
    main()
