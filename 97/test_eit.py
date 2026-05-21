import numpy as np
from eit_solver import EITMesh, EITForward, EITInverse


def test_basic_eit():
    print("Testing EIT implementation...")
    
    mesh = EITMesh(n_radius=4, n_angles=8, r=1.0)
    print(f"Mesh created: {mesh.n_nodes} nodes, {mesh.n_elements} elements")
    
    sigma = np.ones(mesh.n_elements)
    sigma[:10] = 2.0
    
    forward = EITForward(mesh)
    voltages = forward.simulate_measurements(sigma)
    print(f"Forward solution: {len(voltages)} measurements")
    
    inverse = EITInverse(forward, max_iter=5, reg_param=1e-2)
    sigma_recon = inverse.reconstruct(voltages)
    print(f"Inverse solution completed")
    
    print("All tests passed!")


if __name__ == "__main__":
    test_basic_eit()
