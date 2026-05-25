import numpy as np
import sys
sys.path.insert(0, '.')
from kdv_internal_wave import KdVSolver


def example_single_soliton():
    print("=" * 60)
    print("Example 1: Single Soliton Propagation (Flat Terrain)")
    print("=" * 60)
    
    solver = KdVSolver(L=80, N=512, dt=0.001, T_max=3)
    
    u0 = solver.soliton_solution(solver.x, x0=-25, c=1.5, A=1)
    
    u_history, h = solver.solve(u0, terrain_type='flat')
    
    solver.plot_evolution(u_history, h, 'ex1_flat_terrain.png')
    solver.plot_spacetime(u_history, 'ex1_flat_spacetime.png')
    
    print("Saved: ex1_flat_terrain.png, ex1_flat_spacetime.png")
    print()


def example_two_solitons():
    print("=" * 60)
    print("Example 2: Two Solitons Interaction")
    print("=" * 60)
    
    solver = KdVSolver(L=100, N=512, dt=0.0005, T_max=6)
    
    u1 = solver.soliton_solution(solver.x, x0=-30, c=3, A=1)
    u2 = solver.soliton_solution(solver.x, x0=-10, c=1, A=0.5)
    u0 = u1 + u2
    
    u_history, h = solver.solve(u0, terrain_type='flat')
    
    solver.plot_evolution(u_history, h, 'ex2_two_solitons.png')
    solver.plot_spacetime(u_history, 'ex2_two_solitons_spacetime.png')
    
    print("Saved: ex2_two_solitons.png, ex2_two_solitons_spacetime.png")
    print()


def example_continental_shelf():
    print("=" * 60)
    print("Example 3: Wave Interaction with Continental Shelf")
    print("=" * 60)
    
    solver = KdVSolver(L=120, N=1024, dt=0.001, T_max=8)
    
    u0 = solver.soliton_solution(solver.x, x0=-40, c=2, A=1)
    
    u_history, h = solver.solve(
        u0,
        terrain_type='shelf',
        h1=1.0,
        h2=0.3,
        x_trans=10,
        width=4
    )
    
    solver.plot_evolution(u_history, h, 'ex3_continental_shelf.png')
    solver.plot_spacetime(u_history, 'ex3_continental_shelf_spacetime.png')
    
    step = max(1, len(u_history) // 80)
    solver.create_animation(u_history[::step], h, 'ex3_shelf_animation.gif', fps=20)
    
    print("Saved: ex3_continental_shelf.png, ex3_continental_shelf_spacetime.png")
    print("Saved: ex3_shelf_animation.gif")
    print()


def example_submarine_ridge():
    print("=" * 60)
    print("Example 4: Wave Passing Over Submarine Ridge")
    print("=" * 60)
    
    solver = KdVSolver(L=100, N=1024, dt=0.001, T_max=6)
    
    u0 = solver.soliton_solution(solver.x, x0=-35, c=2, A=1)
    
    u_history, h = solver.solve(
        u0,
        terrain_type='ridge',
        h0=1.0,
        height=0.6,
        x0=0,
        width=6
    )
    
    solver.plot_evolution(u_history, h, 'ex4_submarine_ridge.png')
    solver.plot_spacetime(u_history, 'ex4_submarine_ridge_spacetime.png')
    
    print("Saved: ex4_submarine_ridge.png, ex4_submarine_ridge_spacetime.png")
    print()


def example_gaussian_wave():
    print("=" * 60)
    print("Example 5: Gaussian Wave Packet Evolution")
    print("=" * 60)
    
    solver = KdVSolver(L=100, N=1024, dt=0.0005, T_max=4)
    
    u0 = solver.gaussian_wave(solver.x, x0=-30, sigma=4, amp=2)
    
    u_history, h = solver.solve(u0, terrain_type='flat')
    
    solver.plot_evolution(u_history, h, 'ex5_gaussian_wave.png')
    solver.plot_spacetime(u_history, 'ex5_gaussian_wave_spacetime.png')
    
    step = max(1, len(u_history) // 80)
    solver.create_animation(u_history[::step], h, 'ex5_gaussian_animation.gif', fps=20)
    
    print("Saved: ex5_gaussian_wave.png, ex5_gaussian_wave_spacetime.png")
    print("Saved: ex5_gaussian_animation.gif")
    print()


def example_complex_terrain():
    print("=" * 60)
    print("Example 6: Complex Combined Terrain")
    print("=" * 60)
    
    solver = KdVSolver(L=150, N=1024, dt=0.001, T_max=10)
    
    u0 = solver.soliton_solution(solver.x, x0=-55, c=1.8, A=1)
    
    def custom_terrain(x):
        h = np.ones_like(x)
        h -= 0.3 * np.exp(-((x + 10) ** 2) / (2 * 4 ** 2))
        h += 0.2 * np.exp(-((x - 20) ** 2) / (2 * 3 ** 2))
        return h
    
    h_x = custom_terrain(solver.x)
    
    solver.terrain_function = lambda *args, **kwargs: h_x
    
    u_history, h = solver.solve(u0, terrain_type='flat')
    
    solver.plot_evolution(u_history, h, 'ex6_complex_terrain.png')
    solver.plot_spacetime(u_history, 'ex6_complex_terrain_spacetime.png')
    
    print("Saved: ex6_complex_terrain.png, ex6_complex_terrain_spacetime.png")
    print()


if __name__ == "__main__":
    print("\nKdV Equation Solver - Internal Wave Simulation Examples\n")
    
    example_single_soliton()
    example_two_solitons()
    example_continental_shelf()
    example_submarine_ridge()
    example_gaussian_wave()
    example_complex_terrain()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
