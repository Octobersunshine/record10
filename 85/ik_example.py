import numpy as np
from ik_solver import SevenDOFArm


def demo_problem_statement():
    print("=" * 60)
    print("问题演示: 多解和解跳变")
    print("=" * 60)
    
    arm = SevenDOFArm()
    target_pos = np.array([0.5, 0.0, 0.5])
    
    print("\n同一目标，不同初始猜测 -> 不同的解:")
    print("-" * 50)
    
    initial_guesses = [
        ("肘部向上猜测", np.array([0, 0, 0, -1.0, 0, 0, 0])),
        ("肘部向下猜测", np.array([0, 0, 0, -2.5, 0, 0, 0])),
        ("随机猜测1", np.array([0.5, -0.3, 0.2, -1.5, 0.1, 0.4, -0.2])),
        ("随机猜测2", np.array([-0.2, 0.4, -0.5, -2.0, 0.3, -0.1, 0.5])),
    ]
    
    solutions = []
    for name, guess in initial_guesses:
        arm.reset_previous_solution()
        sol, success = arm.inverse_kinematics_damped(
            target_pos,
            initial_guess=guess,
            max_iterations=3000,
            tolerance=1e-4,
            step_size=0.05
        )
        solutions.append(sol)
        elbow_angle = sol[3]
        print(f"{name}:")
        print(f"  肘部关节: {elbow_angle:.4f} rad ({np.degrees(elbow_angle):.1f}°)")
        print(f"  完整解: {np.round(sol, 4)}")
        pos_verify, _, _ = arm.forward_kinematics(sol)
        print(f"  位置误差: {np.linalg.norm(target_pos - pos_verify):.2e}")
    
    print("\n解之间的差异（证明多解存在）:")
    for i in range(len(solutions)):
        for j in range(i+1, len(solutions)):
            diff = np.linalg.norm(solutions[i] - solutions[j])
            print(f"  解{i+1} vs 解{j+1}: {diff:.4f}")


def demo_elbow_preference():
    print("\n" + "=" * 60)
    print("解决方案1: 肘部姿态偏好（零空间投影）")
    print("=" * 60)
    
    arm = SevenDOFArm()
    target_pos = np.array([0.5, 0.0, 0.5])
    
    print("\n使用不同的肘部偏好权重:")
    print("-" * 50)
    
    configs = [
        ("无偏好 (原算法)", 0.0),
        ("弱肘部向上偏好", 0.5),
        ("强肘部向上偏好", 2.0),
        ("极强肘部向上偏好", 10.0),
    ]
    
    for name, elbow_weight in configs:
        arm.reset_previous_solution()
        results = []
        for _ in range(5):
            sol, success = arm.inverse_kinematics_stable(
                target_pos,
                initial_guess=np.random.uniform(-1.0, 1.0, 7),
                max_iterations=3000,
                tolerance=1e-4,
                step_size=0.05,
                elbow_preference_weight=elbow_weight,
                center_weight=0.05
            )
            results.append(sol)
        
        results = np.array(results)
        elbow_angles = results[:, 3]
        std_dev = np.std(results, axis=0)
        
        print(f"\n{name}:")
        print(f"  肘部角度范围: [{np.min(elbow_angles):.4f}, {np.max(elbow_angles):.4f}] rad")
        print(f"  肘部角度均值: {np.mean(elbow_angles):.4f} rad")
        print(f"  解的平均标准差: {np.mean(std_dev):.4f}")
        print(f"  肘部角度标准差: {np.std(elbow_angles):.4f}")


def demo_continuity():
    print("\n" + "=" * 60)
    print("解决方案2: 连续性约束（避免跳变）")
    print("=" * 60)
    
    arm = SevenDOFArm()
    
    num_points = 20
    trajectory = []
    for t in np.linspace(0, 1, num_points):
        x = 0.5 + 0.3 * np.cos(t * np.pi)
        y = 0.2 * np.sin(t * np.pi)
        z = 0.5 + 0.1 * t
        trajectory.append(np.array([x, y, z]))
    
    print("\n对比: 无连续性约束 vs 有连续性约束")
    print("-" * 50)
    
    configs = [
        ("无连续性约束", 0.0, 0.0),
        ("弱连续性约束", 0.5, 0.0),
        ("强连续性约束", 2.0, 0.5),
    ]
    
    for name, continuity_weight, elbow_weight in configs:
        arm.reset_previous_solution()
        solutions, successes = arm.inverse_kinematics_track(
            trajectory,
            max_iterations=2000,
            tolerance=1e-4,
            step_size=0.05,
            continuity_weight=continuity_weight,
            elbow_preference_weight=elbow_weight,
            center_weight=0.02
        )
        
        jumps = []
        for i in range(1, len(solutions)):
            jumps.append(np.max(np.abs(solutions[i] - solutions[i-1])))
        
        avg_jump = np.mean(jumps)
        max_jump = np.max(jumps)
        
        pos_errors = []
        for i, sol in enumerate(solutions):
            pos, _, _ = arm.forward_kinematics(sol)
            pos_errors.append(np.linalg.norm(trajectory[i] - pos))
        
        print(f"\n{name}:")
        print(f"  平均关节跳变: {avg_jump:.4f} rad")
        print(f"  最大关节跳变: {max_jump:.4f} rad")
        print(f"  平均位置误差: {np.mean(pos_errors):.2e}")
        print(f"  成功率: {np.mean(successes)*100:.1f}%")
        
        elbow_angles = [sol[3] for sol in solutions]
        print(f"  肘部角度范围: [{np.min(elbow_angles):.4f}, {np.max(elbow_angles):.4f}]")


def demo_center_regularization():
    print("\n" + "=" * 60)
    print("解决方案3: 关节中心正则化（避免极端位形）")
    print("=" * 60)
    
    arm = SevenDOFArm()
    target_pos = np.array([0.45, 0.15, 0.55])
    
    print("\n对比不同中心正则化权重:")
    print("-" * 50)
    
    configs = [
        ("无正则化", 0.0),
        ("弱正则化", 0.01),
        ("中等正则化", 0.1),
        ("强调则化", 1.0),
    ]
    
    for name, center_weight in configs:
        arm.reset_previous_solution()
        results = []
        for _ in range(5):
            sol, success = arm.inverse_kinematics_stable(
                target_pos,
                initial_guess=np.random.uniform(-1.5, 1.5, 7),
                max_iterations=3000,
                tolerance=1e-4,
                step_size=0.05,
                center_weight=center_weight,
                elbow_preference_weight=0.5
            )
            results.append(sol)
        
        results = np.array(results)
        std_dev = np.std(results, axis=0)
        
        dist_to_center = np.mean(np.linalg.norm(results - arm.joint_centers, axis=1))
        
        joint_range_utilization = []
        for sol in results:
            normalized = (sol - arm.joint_limits[:, 0]) / (arm.joint_limits[:, 1] - arm.joint_limits[:, 0])
            joint_range_utilization.append(np.mean(normalized))
        
        print(f"\n{name}:")
        print(f"  到关节中心平均距离: {dist_to_center:.4f}")
        print(f"  解的平均标准差: {np.mean(std_dev):.4f}")
        print(f"  平均关节范围利用率: {np.mean(joint_range_utilization)*100:.1f}%")


def demo_comparison_near_singularity():
    print("\n" + "=" * 60)
    print("稳定性对比: 接近奇异位形时")
    print("=" * 60)
    
    arm = SevenDOFArm()
    
    print("\n测试目标: 逐步接近工作空间边界")
    print("-" * 50)
    
    distances = [0.5, 0.7, 0.85, 0.9, 0.95]
    
    for d in distances:
        target_pos = np.array([d, 0.0, 0.3])
        
        arm.reset_previous_solution()
        sol_old, success_old = arm.inverse_kinematics_damped(
            target_pos,
            initial_guess=np.zeros(7),
            max_iterations=3000,
            tolerance=1e-4,
            step_size=0.03
        )
        
        arm.reset_previous_solution()
        sol_new, success_new = arm.inverse_kinematics_stable(
            target_pos,
            initial_guess=np.zeros(7),
            max_iterations=3000,
            tolerance=1e-4,
            step_size=0.03,
            continuity_weight=0.5,
            elbow_preference_weight=1.0,
            center_weight=0.1
        )
        
        pos_old, _, _ = arm.forward_kinematics(sol_old)
        pos_new, _, _ = arm.forward_kinematics(sol_new)
        err_old = np.linalg.norm(target_pos - pos_old)
        err_new = np.linalg.norm(target_pos - pos_new)
        
        print(f"\n距离 {d} m:")
        print(f"  原算法: 成功={success_old}, 误差={err_old:.2e}, 肘角={sol_old[3]:.4f}")
        print(f"  新算法: 成功={success_new}, 误差={err_new:.2e}, 肘角={sol_new[3]:.4f}")


def demo_recommended_config():
    print("\n" + "=" * 60)
    print("推荐配置示例")
    print("=" * 60)
    
    arm = SevenDOFArm()
    
    print("\n推荐参数组合:")
    print("  elbow_preference_weight = 1.0  (肘部向上偏好)")
    print("  center_weight = 0.05           (关节中心正则化)")
    print("  continuity_weight = 1.0        (连续性约束，轨迹跟踪时)")
    print("  null_space_weight = 0.1        (零空间投影权重)")
    
    print("\n平滑轨迹跟踪演示:")
    print("-" * 50)
    
    num_points = 30
    trajectory = []
    for t in np.linspace(0, 4 * np.pi, num_points):
        x = 0.5 + 0.2 * np.cos(t)
        y = 0.2 * np.sin(t)
        z = 0.5 + 0.1 * np.sin(t/2)
        trajectory.append(np.array([x, y, z]))
    
    arm.reset_previous_solution()
    solutions, successes = arm.inverse_kinematics_track(
        trajectory,
        max_iterations=2000,
        tolerance=1e-4,
        step_size=0.05,
        continuity_weight=1.0,
        elbow_preference_weight=1.0,
        center_weight=0.05,
        null_space_weight=0.1
    )
    
    jumps = [np.max(np.abs(solutions[i] - solutions[i-1])) for i in range(1, len(solutions))]
    elbow_angles = [sol[3] for sol in solutions]
    
    print(f"  成功率: {np.mean(successes)*100:.1f}%")
    print(f"  平均关节跳变: {np.mean(jumps):.4f} rad")
    print(f"  最大关节跳变: {np.max(jumps):.4f} rad")
    print(f"  肘部角度范围: [{np.min(elbow_angles):.4f}, {np.max(elbow_angles):.4f}] rad")
    print(f"  肘部角度标准差: {np.std(elbow_angles):.4f} rad")
    
    print("\n前5个点的肘部角度（展示平滑过渡）:")
    for i in range(min(5, len(elbow_angles))):
        print(f"  点{i}: {elbow_angles[i]:.4f} rad")


def print_joint_limits():
    print("\n" + "=" * 60)
    print("关节限位信息")
    print("=" * 60)
    
    arm = SevenDOFArm()
    for i, (min_angle, max_angle) in enumerate(arm.joint_limits):
        min_deg = np.degrees(min_angle)
        max_deg = np.degrees(max_angle)
        center = (min_angle + max_angle) / 2
        print(f"关节 {i+1}: [{min_deg:.1f}°, {max_deg:.1f}°], 中心={center:.4f} rad")


if __name__ == "__main__":
    np.random.seed(42)
    
    print_joint_limits()
    demo_problem_statement()
    demo_elbow_preference()
    demo_continuity()
    demo_center_regularization()
    demo_comparison_near_singularity()
    demo_recommended_config()
    
    print("\n" + "=" * 60)
    print("所有演示完成!")
    print("=" * 60)
