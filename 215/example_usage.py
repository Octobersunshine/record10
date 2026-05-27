from cyclone_separator import CycloneSeparator, CycloneOptimizer


def example_1_pressure_drop_comparison():
    print("=" * 70)
    print("Example 1: Pressure Drop Model Comparison")
    print("=" * 70)

    cyclone = CycloneSeparator(
        diameter=0.1,
        particle_density=2000.0,
        fluid_density=1.2,
        fluid_viscosity=1.81e-5
    )

    inlet_velocity = 15.0

    print(f"\nCyclone diameter: {cyclone.D} m")
    print(f"Inlet velocity: {inlet_velocity} m/s")
    print(f"\nGeometry ratios:")
    ratios = cyclone.get_geometry_ratios()
    for key, value in ratios.items():
        print(f"  {key}: {value:.4f}")

    print(f"\n{'Pressure Drop Model':<40} {'Delta P (Pa)':<20}")
    print("-" * 65)

    models = ['simple', 'shepherd_lapple', 'stairmand', 'dirgo', 'casals']
    model_names = {
        'simple': 'Simple K-factor (K=16*ab/De^2)',
        'shepherd_lapple': 'Shepherd-Lapple (geometry-adjusted K)',
        'stairmand': 'Stairmand (K=32*ab/De^2)',
        'dirgo': 'Dirgo (K=4.72*D^2/ab)',
        'casals': 'Casals (K=20*(a/D)*(b/D)^0.5/(De/D)^2)'
    }

    for model in models:
        dp = cyclone.calculate_pressure_drop(inlet_velocity, model)
        print(f"{model_names[model]:<35} {dp:<20.1f}")

    print("\nKey notes:")
    print("  * Shepherd-Lapple is the most widely used industrial standard")
    print("  * K_SL = 11.3 for standard Stairmand HE design")
    print("  * Simple model overpredicts due to fixed K-factor")
    print()


def example_2_geometry_parameterization():
    print("=" * 70)
    print("Example 2: Geometry Parameterization (Cone Angle & De/D)")
    print("=" * 70)

    base_cyclone = CycloneSeparator(
        diameter=0.1,
        cone_angle=15.0,
        vortex_finder_ratio=0.5
    )

    print(f"\nBase geometry (cone_angle=15 deg, De/D=0.5):")
    print(f"  Cone height Hc: {base_cyclone.Hc:.4f} m ({base_cyclone.Hc/base_cyclone.D:.2f} x D)")
    print(f"  Vortex finder De: {base_cyclone.De:.4f} m")

    print(f"\n{'Cone Angle (deg)':<20} {'Hc/D':<15} {'Hc (m)':<15}")
    print("-" * 55)
    for angle in [5, 10, 15, 20, 30]:
        cyc = CycloneSeparator(diameter=0.1, cone_angle=angle)
        print(f"{angle:<20} {cyc.Hc/cyc.D:<15.2f} {cyc.Hc:<15.4f}")

    print(f"\n{'De/D Ratio':<20} {'De (m)':<15} {'Pressure Drop (Pa)':<20}")
    print("-" * 60)
    for De_ratio in [0.3, 0.4, 0.5, 0.6, 0.7]:
        cyc = CycloneSeparator(diameter=0.1, vortex_finder_ratio=De_ratio)
        dp = cyc.calculate_pressure_drop(15.0)
        print(f"{De_ratio:<20.2f} {cyc.De:<15.4f} {dp:<20.1f}")

    print("\nKey observations:")
    print("  * Smaller cone angle = taller cone = longer residence time")
    print("  * Smaller De/D = higher velocity gradient = higher pressure drop")
    print()


def example_3_single_parameter_sensitivity():
    print("=" * 70)
    print("Example 3: Single Parameter Sensitivity Analysis")
    print("=" * 70)

    optimizer = CycloneOptimizer(
        base_diameter=0.1,
        fluid_density=1.2,
        fluid_viscosity=1.81e-5,
        particle_density=2000.0
    )

    base_params = {
        'a_ratio': 0.5,
        'b_ratio': 0.25,
        'De_ratio': 0.5,
        'cone_angle': 15.0,
        'B_ratio': 0.25
    }

    inlet_velocity = 15.0

    param_ranges = {
        'De_ratio': (0.35, 0.65, 7),
        'cone_angle': (10.0, 25.0, 7),
        'b_ratio': (0.15, 0.35, 7)
    }

    for param_name, prange in param_ranges.items():
        results = optimizer.single_parameter_sensitivity(
            param_name=param_name,
            param_range=prange,
            base_params=base_params,
            inlet_velocity=inlet_velocity
        )

        print(f"\nSensitivity to {param_name}:")
        print(f"{'Value':<15} {'d50 (um)':<15} {'Pressure (Pa)':<20}")
        print("-" * 55)
        for r in results:
            print(f"{r['param_value']:<15.3f} {r['d50_um']:<15.2f} {r['pressure_drop']:<20.1f}")

    print()


def example_4_response_surface_optimization():
    print("=" * 70)
    print("Example 4: Response Surface Optimization")
    print("=" * 70)
    print("Optimizing 5 geometric parameters across 4 levels each")
    print("Total designs evaluated: 4^5 = 1024")
    print("=" * 70)

    optimizer = CycloneOptimizer(
        base_diameter=0.1,
        fluid_density=1.2,
        fluid_viscosity=1.81e-5,
        particle_density=2000.0
    )

    inlet_velocity = 15.0

    print("\n[Design 1: Minimize d50 (high efficiency)]")
    opt_eff = optimizer.response_surface_optimization(
        inlet_velocity=inlet_velocity,
        target_d50=None,
        max_pressure_drop=None
    )
    eff_opt = opt_eff['optimal_design']
    print(f"  Designs evaluated: {opt_eff['n_designs_evaluated']}")
    print(f"  Optimal geometry:")
    for key, value in eff_opt['params'].items():
        print(f"    {key}: {value:.4f}")
    print(f"  Performance:")
    print(f"    d50: {eff_opt['d50_um']:.2f} um")
    print(f"    Pressure drop: {eff_opt['pressure_drop']:.1f} Pa")

    print("\n[Design 2: Constraint-based (max pressure = 800 Pa)]")
    try:
        opt_constraint = optimizer.response_surface_optimization(
            inlet_velocity=inlet_velocity,
            target_d50=None,
            max_pressure_drop=800.0
        )
        const_opt = opt_constraint['optimal_design']
        print(f"  Designs evaluated: {opt_constraint['n_designs_evaluated']}")
        print(f"  Optimal geometry:")
        for key, value in const_opt['params'].items():
            print(f"    {key}: {value:.4f}")
        print(f"  Performance:")
        print(f"    d50: {const_opt['d50_um']:.2f} um")
        print(f"    Pressure drop: {const_opt['pressure_drop']:.1f} Pa")
    except ValueError as e:
        print(f"  {e}")

    print()


def example_5_multi_objective_pareto():
    print("=" * 70)
    print("Example 5: Multi-Objective Optimization (Pareto Front)")
    print("=" * 70)
    print("Objective: Minimize both d50 and pressure drop")
    print("Trade-off: Efficiency vs Energy Consumption")
    print("=" * 70)

    optimizer = CycloneOptimizer(
        base_diameter=0.1,
        fluid_density=1.2,
        fluid_viscosity=1.81e-5,
        particle_density=2000.0
    )

    inlet_velocity = 15.0

    result = optimizer.multi_objective_optimization(
        inlet_velocity=inlet_velocity,
        d50_weight=0.6,
        pressure_weight=0.4
    )

    print(f"\nTotal designs evaluated: {len(result['all_designs'])}")
    print(f"Pareto optimal designs: {len(result['pareto_front'])}")

    print(f"\nPareto Front (d50 vs Pressure Drop):")
    print(f"{'Design':<10} {'d50 (um)':<15} {'Pressure (Pa)':<20}")
    print("-" * 50)
    for i, design in enumerate(result['pareto_front'][:10]):
        print(f"{i+1:<10} {design['d50_um']:<15.2f} {design['pressure_drop']:<20.1f}")

    print(f"\n... (showing first 10 of {len(result['pareto_front'])} designs)")

    optimal = result['optimal_design']
    print(f"\nWeighted Optimal (d50_weight=0.6, pressure_weight=0.4):")
    print(f"  Geometry:")
    for key, value in optimal['params'].items():
        print(f"    {key}: {value:.4f}")
    print(f"  Performance:")
    print(f"    d50: {optimal['d50_um']:.2f} um")
    print(f"    Pressure drop: {optimal['pressure_drop']:.1f} Pa")

    print(f"\nRange of designs explored:")
    print(f"  d50: {result['min_d50']*1e6:.2f} - {result['max_d50']*1e6:.2f} um")
    print(f"  Pressure: {result['min_pressure']:.1f} - {result['max_pressure']:.1f} Pa")
    print()


def example_6_liquid_solid_optimization():
    print("=" * 70)
    print("Example 6: Liquid-Solid Cyclone Optimization (Water-Sand)")
    print("=" * 70)

    optimizer = CycloneOptimizer(
        base_diameter=0.15,
        fluid_density=1000.0,
        fluid_viscosity=0.001,
        particle_density=2650.0
    )

    inlet_velocity = 10.0

    print(f"\nBase condition: Water-Sand separation")
    print(f"  Cyclone diameter: 0.15 m")
    print(f"  Inlet velocity: {inlet_velocity} m/s")
    print(f"  Particle density: 2650 kg/m^3 (sand)")
    print(f"  Fluid density: 1000 kg/m^3 (water)")

    result = optimizer.multi_objective_optimization(
        inlet_velocity=inlet_velocity,
        d50_weight=0.7,
        pressure_weight=0.3
    )

    optimal = result['optimal_design']
    print(f"\nOptimal Hydrocyclone Design:")
    print(f"  Geometry ratios:")
    for key, value in optimal['params'].items():
        print(f"    {key}: {value:.4f}")
    print(f"  Performance:")
    print(f"    d50: {optimal['d50_um']:.2f} um")
    print(f"    Pressure drop: {optimal['pressure_drop']/1000:.1f} kPa")

    print(f"\nPareto front has {len(result['pareto_front'])} optimal designs")
    print()


if __name__ == "__main__":
    example_1_pressure_drop_comparison()
    example_2_geometry_parameterization()
    example_3_single_parameter_sensitivity()
    example_4_response_surface_optimization()
    example_5_multi_objective_pareto()
    example_6_liquid_solid_optimization()

    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
