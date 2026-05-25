import sys
sys.path.insert(0, r'e:\temp\record10\156')

print("Testing imports...")

try:
    import numpy as np
    print("✓ numpy imported successfully")
except Exception as e:
    print(f"✗ numpy import failed: {e}")
    sys.exit(1)

try:
    from surface_code import SurfaceCode
    print("✓ SurfaceCode imported successfully")
except Exception as e:
    print(f"✗ SurfaceCode import failed: {e}")
    sys.exit(1)

try:
    from mwpm_decoder import MWPMDecoder
    print("✓ MWPMDecoder imported successfully")
except Exception as e:
    print(f"✗ MWPMDecoder import failed: {e}")
    sys.exit(1)

try:
    from error_model import ErrorModel
    print("✓ ErrorModel imported successfully")
except Exception as e:
    print(f"✗ ErrorModel import failed: {e}")
    sys.exit(1)

print("\nTesting SurfaceCode initialization...")
try:
    sc = SurfaceCode(3)
    print(f"✓ SurfaceCode(3) created: {sc.n_data} data qubits")
except Exception as e:
    print(f"✗ SurfaceCode creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTesting basic operations...")
try:
    sc.apply_bit_flip(4)
    print(f"✓ Applied bit flip to qubit 4")
    
    x_stabs, z_stabs = sc.measure_stabilizers()
    print(f"✓ Measured stabilizers")
    print(f"  X defects: {np.where(x_stabs)[0]}")
    print(f"  Z defects: {np.where(z_stabs)[0]}")
except Exception as e:
    print(f"✗ Basic operations failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTesting decoder...")
try:
    decoder = MWPMDecoder(sc)
    x_matching = decoder.decode('x')
    print(f"✓ Decoded X-type defects: {x_matching}")
except Exception as e:
    print(f"✗ Decoder failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed!")
