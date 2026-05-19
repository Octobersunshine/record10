#!/usr/bin/env python
"""
Grad-Shafranov方程求解器使用示例
"""

from grad_shafranov_solver import GradShafranovSolver
import numpy as np


def example_basic():
    """基本示例"""
    print("="*60)
    print("示例1: 基本求解")
    print("="*60)
    
    solver = GradShafranovSolver(
        Rmin=0.4, Rmax=1.6,
        Zmin=-0.5, Zmax=0.5,
        nr=65, nz=65
    )
    
    psi = solver.solve(
        p0=2e5, alpha=1.5,
        F0=1.2, beta=0.1,
        max_iter=120, tol=1e-6
    )
    
    print(f"\nψ 最大值: {np.max(psi):.6f}")
    print(f"ψ 最小值: {np.min(psi):.6f}")
    
    solver.plot_psi(levels=20)
    return solver


def example_different_profiles():
    """不同压力剖面的比较"""
    print("\n" + "="*60)
    print("示例2: 不同压力剖面")
    print("="*60)
    
    profiles = [
        (1.0, "Flat profile (α=1)"),
        (2.0, "Parabolic profile (α=2)"),
        (3.0, "Peaked profile (α=3)"),
    ]
    
    for alpha, desc in profiles:
        print(f"\n{desc}")
        solver = GradShafranovSolver(nr=33, nz=33)
        psi = solver.solve(p0=1e5, alpha=alpha, max_iter=80)
        print(f"  ψ_max = {np.max(psi):.6f}")


def example_magnetic_field():
    """磁场计算示例"""
    print("\n" + "="*60)
    print("示例3: 磁场计算")
    print("="*60)
    
    solver = GradShafranovSolver(nr=65, nz=65)
    solver.solve(max_iter=100)
    
    B_R, B_Z = solver.get_magnetic_field()
    
    print(f"\nB_R 范围: [{np.min(B_R):.4f}, {np.max(B_R):.4f}]")
    print(f"B_Z 范围: [{np.min(B_Z):.4f}, {np.max(B_Z):.4f}]")
    
    B_total = np.sqrt(B_R**2 + B_Z**2)
    print(f"总磁场最大值: {np.max(B_total):.4f}")
    
    solver.plot_magnetic_field()


def main():
    print("\nGrad-Shafranov 求解器示例程序\n")
    
    example_basic()
    example_different_profiles()
    example_magnetic_field()
    
    print("\n" + "="*60)
    print("所有示例完成!")
    print("="*60)


if __name__ == "__main__":
    main()
