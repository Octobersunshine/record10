import numpy as np
from analytical_ik import Analytical7DOF, AnalyticalIKWrapper


def demo_arm_angle_parameterization():
    """演示臂型角参数化 - 同一目标位姿的不同解"""
    print("=" * 70)
    print("演示1: 臂型角参数化 - 同一目标位姿的不同解")
    print("=" * 70)
    
    solver = Analytical7DOF()
    
    target_pos = np.array([0.55, 0.0, 0.45])
    target_rot = np.eye(3)
    
    print(f"\n目标位置: {target_pos}")
    print("-" * 70)
    
    # 采样不同臂型角
    psi_samples = [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]
    
    print("\n不同臂型角对应的肘部位置:")
    valid_solutions = []
    
    for psi in psi_samples:
        elbow_pos = solver.get_elbow_position_from_psi(target_pos, target_rot, psi)
        q = solver.inverse_kinematics_single_psi(target_pos, target_rot, psi)
        
        if elbow_pos is not None:
            valid = q is not None and solver._check_joint_limits(q)
            valid_solutions.append((psi, elbow_pos, q, valid))
            
            status = "✓" if valid else "✗"
            print(f"  psi = {psi:6.3f} rad: 肘部位置 = [{elbow_pos[0]:.3f}, {elbow_pos[1]:.3f}, {elbow_pos[2]:.3f}]  {status}")
    
    if len(valid_solutions) >= 2:
        print(f"\n  肘部Z坐标变化范围: {min([s[1][2] for s in valid_solutions]):.4f} - {max([s[1][2] for s in valid_solutions]):.4f} m")
    
    return solver


def demo_solution_space_analysis():
    """分析解空间特性"""
    print("\n" + "=" * 70)
    print("演示2: 解空间特性分析")
    print("=" * 70)
    
    solver = Analytical7DOF()
    
    test_positions = [
        ("工作空间中心", np.array([0.5, 0.0, 0.5])),
        ("工作空间边缘", np.array([0.85, 0.0, 0.3])),
        ("侧向位置", np.array([0.4, 0.3, 0.5])),
        ("高位位置", np.array([0.3, 0.0, 0.7])),
    ]
    
    for name, target_pos in test_positions:
        print(f"\n目标: {name}, 位置: {target_pos}")
        print("-" * 50)
        
        psi_values, q_solutions, valid_mask = solver.inverse_kinematics_all_solutions(
            target_pos, num_psi=50
        )
        
        valid_count = np.sum(valid_mask)
        valid_psis = psi_values[valid_mask]
        
        print(f"  可行解数量: {valid_count}/{len(psi_values)}")
        
        if valid_count > 0:
            psi_range = valid_psis.max() - valid_psis.min()
            print(f"  可行臂型角范围: [{valid_psis.min():.4f}, {valid_psis.max():.4f}] rad")
            print(f"  可行区间长度: {psi_range:.4f} rad ({np.degrees(psi_range):.1f}°)")
            
            # 分析肘部高度变化
            elbow_heights = []
            for psi in valid_psis:
                elbow_pos = solver.get_elbow_position_from_psi(target_pos, np.eye(3), psi)
                if elbow_pos is not None:
                    elbow_heights.append(elbow_pos[2])
            
            if len(elbow_heights) > 0:
                print(f"  肘部Z坐标范围: [{min(elbow_heights):.4f}, {max(elbow_heights):.4f}] m")
                print(f"  肘部Z坐标变化量: {max(elbow_heights) - min(elbow_heights):.4f} m")
        else:
            print(f"  警告: 该位置超出机械臂工作空间或无可行解!")


def demo_objective_optimization():
    """演示不同优化目标的效果"""
    print("\n" + "=" * 70)
    print("演示3: 不同优化目标的对比")
    print("=" * 70)
    
    solver = Analytical7DOF()
    
    target_pos = np.array([0.5, 0.15, 0.5])
    target_rot = np.eye(3)
    
    print(f"\n目标位置: {target_pos}")
    print("-" * 50)
    
    objectives = ['elbow_up', 'elbow_down', 'center', 'manipulability']
    results = {}
    
    for obj in objectives:
        q_opt, psi_opt, success = solver.inverse_kinematics_optimize_psi(
            target_pos, target_rot, objective=obj
        )
        
        if success:
            elbow_pos = solver.get_elbow_position_from_psi(target_pos, target_rot, psi_opt)
            
            pos_fk, _ = solver.forward_kinematics(q_opt)
            pos_error = np.linalg.norm(target_pos - pos_fk)
            
            results[obj] = {
                'psi': psi_opt,
                'q': q_opt,
                'elbow_z': elbow_pos[2] if elbow_pos is not None else None,
                'error': pos_error
            }
            
            print(f"\n  目标 '{obj}':")
            print(f"    最优臂型角: {psi_opt:.4f} rad ({np.degrees(psi_opt):.1f}°)")
            print(f"    肘部Z坐标: {elbow_pos[2]:.4f} m" if elbow_pos is not None else "    肘部Z坐标: N/A")
            print(f"    位置误差: {pos_error:.2e}")
            print(f"    肘关节角 (q4): {q_opt[3]:.4f} rad")
    
    # 对比结果
    if len(results) >= 2:
        print("\n" + "-" * 50)
        print("结果对比:")
        z_values = [r['elbow_z'] for r in results.values() if r['elbow_z'] is not None]
        if len(z_values) >= 2:
            print(f"  肘部Z坐标最大差异: {max(z_values) - min(z_values):.4f} m")
        
        psi_values = [r['psi'] for r in results.values()]
        print(f"  臂型角范围: [{min(psi_values):.4f}, {max(psi_values):.4f}] rad")


def demo_manipulability_vs_psi():
    """演示可操作度随臂型角的变化"""
    print("\n" + "=" * 70)
    print("演示4: 可操作度随臂型角的变化")
    print("=" * 70)
    
    solver = Analytical7DOF()
    
    target_pos = np.array([0.5, 0.0, 0.5])
    
    print(f"\n目标位置: {target_pos}")
    print("-" * 50)
    
    psi_values = np.linspace(-np.pi/2, np.pi/2, 20)
    manip_values = []
    valid_psi = []
    
    for psi in psi_values:
        q = solver.inverse_kinematics_single_psi(target_pos, np.eye(3), psi)
        if q is not None and solver._check_joint_limits(q):
            manip = solver.compute_manipulability_ellipsoid(q)
            manip_values.append(manip)
            valid_psi.append(psi)
    
    if len(manip_values) > 0:
        print(f"  可操作度范围: [{min(manip_values):.4f}, {max(manip_values):.4f}]")
        print(f"  最佳可操作度臂型角: {valid_psi[np.argmax(manip_values)]:.4f} rad")
        print(f"  最差可操作度臂型角: {valid_psi[np.argmin(manip_values)]:.4f} rad")


def demo_unified_interface():
    """演示统一接口的使用"""
    print("\n" + "=" * 70)
    print("演示5: 统一接口使用")
    print("=" * 70)
    
    solver = AnalyticalIKWrapper()
    
    target_pos = np.array([0.5, 0.0, 0.5])
    
    print(f"\n目标位置: {target_pos}")
    print("-" * 50)
    
    # 不同求解方法
    methods = [
        ('auto', '自动模式'),
        ('optimize', '优化模式 (肘部向上)'),
    ]
    
    for method, desc in methods:
        print(f"\n方法: {desc} (method='{method}')")
        
        if method == 'optimize':
            q, success, info = solver.solve(
                target_pos, method=method, objective='elbow_up'
            )
        else:
            q, success, info = solver.solve(target_pos, method=method)
        
        if success:
            print(f"  成功! 使用方法: {info.get('method', 'N/A')}")
            print(f"  臂型角: {info.get('psi', 'N/A')}")
            print(f"  关节角: {q.round(4)}")
        else:
            print(f"  求解失败: {info.get('warning', '未知错误')}")
    
    # 解空间采样
    print("\n" + "-" * 50)
    print("解空间采样:")
    valid_psis, valid_qs, valid_mask = solver.sample_solution_space(
        target_pos, num_samples=30
    )
    print(f"  采样数量: 30, 可行解: {len(valid_psis)}")
    
    if len(valid_psis) > 0:
        print(f"  可行臂型角范围: [{min(valid_psis):.4f}, {max(valid_psis):.4f}] rad")


def demo_psi_trajectory():
    """演示通过改变臂型角实现肘部运动（末端保持不动）"""
    print("\n" + "=" * 70)
    print("演示6: 冗余自由度利用 - 末端不动，肘部做圆周运动")
    print("=" * 70)
    
    solver = Analytical7DOF()
    
    target_pos = np.array([0.5, 0.0, 0.5])
    target_rot = np.eye(3)
    
    print(f"\n目标位置: {target_pos} (保持不变)")
    print("-" * 50)
    
    # 生成臂型角轨迹
    psi_trajectory = np.linspace(-np.pi/3, np.pi/3, 10)
    elbow_trajectory = []
    valid_count = 0
    
    for i, psi in enumerate(psi_trajectory):
        elbow_pos = solver.get_elbow_position_from_psi(target_pos, target_rot, psi)
        q = solver.inverse_kinematics_single_psi(target_pos, target_rot, psi)
        
        valid = q is not None and solver._check_joint_limits(q)
        if valid:
            valid_count += 1
            elbow_trajectory.append(elbow_pos)
        
        if i % 3 == 0:
            status = "✓" if valid else "✗"
            print(f"  点{i}: psi={psi:.3f}, 肘部={elbow_pos if elbow_pos is not None else 'N/A'} {status}")
    
    if len(elbow_trajectory) >= 2:
        elbow_array = np.array(elbow_trajectory)
        print(f"\n  肘部运动范围:")
        print(f"    X方向: [{elbow_array[:,0].min():.4f}, {elbow_array[:,0].max():.4f}] m")
        print(f"    Y方向: [{elbow_array[:,1].min():.4f}, {elbow_array[:,1].max():.4f}] m")
        print(f"    Z方向: [{elbow_array[:,2].min():.4f}, {elbow_array[:,2].max():.4f}] m")
        print(f"  肘部轨迹长度 (近似): {np.sum(np.linalg.norm(np.diff(elbow_array, axis=0), axis=1)):.4f} m")
    
    print(f"\n  总结: 末端保持不动的同时，肘部可以在一定范围内自由运动")
    print(f"        这就是7DOF机械臂的冗余自由度特性!")


def demo_joint_limit_effects():
    """演示关节限位对解空间的影响"""
    print("\n" + "=" * 70)
    print("演示7: 关节限位对解空间的影响")
    print("=" * 70)
    
    solver = Analytical7DOF()
    
    target_pos = np.array([0.5, 0.2, 0.45])
    
    print(f"\n目标位置: {target_pos}")
    print("-" * 50)
    
    # 原始关节限位
    print("原始关节限位下的解:")
    psi_values, q_solutions, valid_mask = solver.inverse_kinematics_all_solutions(
        target_pos, num_psi=50
    )
    
    valid_original = psi_values[valid_mask]
    print(f"  可行解数量: {len(valid_original)}")
    
    if len(valid_original) > 0:
        print(f"  臂型角范围: [{valid_original.min():.4f}, {valid_original.max():.4f}] rad")
        
        # 分析哪些关节限位被触发
        limit_triggered = np.zeros(7)
        for q in [q_solutions[i] for i in range(len(q_solutions)) if valid_mask[i]]:
            for j in range(7):
                if abs(q[j] - solver.joint_limits[j, 0]) < 0.01 or abs(q[j] - solver.joint_limits[j, 1]) < 0.01:
                    limit_triggered[j] += 1
        
        print("\n  关节限位触发统计:")
        for j in range(7):
            if limit_triggered[j] > 0:
                percentage = limit_triggered[j] / len(valid_original) * 100
                print(f"    关节{j+1}: {int(limit_triggered[j])}次 ({percentage:.1f}%)")


def main():
    print("\n")
    demo_arm_angle_parameterization()
    demo_solution_space_analysis()
    demo_objective_optimization()
    demo_manipulability_vs_psi()
    demo_unified_interface()
    demo_psi_trajectory()
    demo_joint_limit_effects()
    
    print("\n" + "=" * 70)
    print("所有解析IK演示完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
