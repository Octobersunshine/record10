import sys

def validate_imports():
    print("Validating imports...")
    
    try:
        import numpy as np
        print("  ✓ numpy")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False
    
    try:
        from cosmology import Cosmology
        print("  ✓ cosmology")
    except ImportError as e:
        print(f"  ✗ cosmology: {e}")
        return False
    
    try:
        from pm_solver import PMSolver
        print("  ✓ pm_solver")
    except ImportError as e:
        print(f"  ✗ pm_solver: {e}")
        return False
    
    try:
        from p3m_solver import P3MSolver
        print("  ✓ p3m_solver")
    except ImportError as e:
        print(f"  ✗ p3m_solver: {e}")
        return False
    
    try:
        from simulation import NBodySimulation
        print("  ✓ simulation")
    except ImportError as e:
        print(f"  ✗ simulation: {e}")
        return False
    
    try:
        from initial_conditions import InitialConditions
        print("  ✓ initial_conditions")
    except ImportError as e:
        print(f"  ✗ initial_conditions: {e}")
        return False
    
    try:
        from halo_finder import FriendsOfFriends
        print("  ✓ halo_finder")
    except ImportError as e:
        print(f"  ✗ halo_finder: {e}")
        return False
    
    return True

def validate_classes():
    print("\nValidating class instantiation...")
    
    import numpy as np
    from cosmology import Cosmology
    from p3m_solver import P3MSolver
    
    try:
        cosmo = Cosmology()
        print("  ✓ Cosmology")
    except Exception as e:
        print(f"  ✗ Cosmology: {e}")
        return False
    
    try:
        p3m = P3MSolver(ngrid=16, box_size=100.0, cosmo=cosmo)
        print("  ✓ P3MSolver")
        print(f"    - r_switch: {p3m.r_switch:.2f}")
        print(f"    - r_soft: {p3m.r_soft:.2f}")
        print(f"    - Wk shape: {p3m.Wk.shape}")
    except Exception as e:
        print(f"  ✗ P3MSolver: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        pos = np.random.rand(100, 3) * 100.0
        acc_short = p3m.compute_short_range_accelerations(pos, a=1.0)
        print("  ✓ Short range force calculation")
        print(f"    - acc_short shape: {acc_short.shape}")
        print(f"    - acc_short mean: {np.mean(np.abs(acc_short)):.2e}")
    except Exception as e:
        print(f"  ✗ Short range force: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    print("=" * 60)
    print("P³M Implementation Validation")
    print("=" * 60)
    
    if not validate_imports():
        print("\nImport validation failed!")
        sys.exit(1)
    
    if not validate_classes():
        print("\nClass validation failed!")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("All validations passed!")
    print("=" * 60)
    print("\nP³M implementation summary:")
    print("  - Force separation using Gaussian screening function")
    print("  - Long-range force: PM with smoothed potential")
    print("  - Short-range force: Direct PP with domain decomposition")
    print("  - Softening length to prevent force divergence")
    print("  - Periodic boundary conditions support")

if __name__ == "__main__":
    main()
