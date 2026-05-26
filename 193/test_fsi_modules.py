import numpy as np
from structural_model import SimpleSpringModel, EulerBernoulliBeam


def test_spring_model():
    print("Testing Spring Model")
    print("=" * 50)
    
    model = SimpleSpringModel(
        n_points=20,
        chord=1.0,
        stiffness=1000.0,
        damping=10.0,
        mass_per_length=1.0
    )
    
    forces = np.sin(np.pi * model.x) * 100.0
    
    for _ in range(10):
        model.compute_deflection(forces)
        
    print(f"Max deflection: {np.max(np.abs(model.w)):.6f} m")
    print(f"Tip deflection: {model.w[-1]:.6f} m")
    print("Spring model test PASSED\n")
    
    return True


def test_beam_model():
    print("Testing Euler-Bernoulli Beam Model")
    print("=" * 50)
    
    beam = EulerBernoulliBeam(
        n_points=20,
        chord=1.0,
        thickness=0.12,
        E=70e9,
        rho=2700
    )
    
    beam.assemble_stiffness_matrix()
    
    forces = np.sin(np.pi * beam.x) * 1000.0
    
    beam.compute_deflection(forces)
    
    print(f"Max deflection: {np.max(np.abs(beam.w)):.6f} m")
    print(f"Tip deflection: {beam.w[-1]:.6f} m")
    print("Beam model test PASSED\n")
    
    return True


def test_interpolation():
    print("Testing Interpolation Functions")
    print("=" * 50)
    
    x_struct = np.linspace(0, 1, 20)
    x_aero = np.linspace(0, 1, 10)
    
    w_struct = np.sin(np.pi * x_struct) * 0.1
    
    w_aero = np.interp(x_aero, x_struct, w_struct)
    
    print(f"Structural points: {len(x_struct)}")
    print(f"Aerodynamic points: {len(x_aero)}")
    print(f"Max w_struct: {np.max(w_struct):.4f}")
    print(f"Max w_aero: {np.max(w_aero):.4f}")
    print("Interpolation test PASSED\n")
    
    return True


def main():
    print("\n" + "#" * 60)
    print("#  FSI Module Quick Verification Tests")
    print("#" * 60)
    
    tests = [
        test_spring_model,
        test_beam_model,
        test_interpolation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"Test FAILED: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
