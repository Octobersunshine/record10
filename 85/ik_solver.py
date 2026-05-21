import numpy as np
from scipy.spatial.transform import Rotation


class SevenDOFArm:
    def __init__(self):
        self.dh_params = [
            {'alpha': np.pi/2, 'a': 0, 'd': 0.333, 'theta': 0},
            {'alpha': -np.pi/2, 'a': 0, 'd': 0, 'theta': 0},
            {'alpha': np.pi/2, 'a': 0, 'd': 0.316, 'theta': 0},
            {'alpha': -np.pi/2, 'a': 0.0825, 'd': 0, 'theta': 0},
            {'alpha': np.pi/2, 'a': 0, 'd': 0.384, 'theta': 0},
            {'alpha': -np.pi/2, 'a': -0.0825, 'd': 0, 'theta': 0},
            {'alpha': 0, 'a': 0, 'd': 0.2575, 'theta': 0}
        ]
        
        self.joint_limits = [
            (-2.8973, 2.8973),
            (-2.8973, 2.8973),
            (-2.8973, 2.8973),
            (-3.0718, -0.0698),
            (-2.8973, 2.8973),
            (-0.0175, 3.7525),
            (-2.8973, 2.8973)
        ]
        self.joint_limits = np.array(self.joint_limits).reshape(7, 2)
        
        self.joint_centers = (self.joint_limits[:, 0] + self.joint_limits[:, 1]) / 2
        
        self.num_joints = 7
        
        self.previous_solution = None

    def dh_transform(self, alpha, a, d, theta):
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])

    def forward_kinematics(self, joint_angles):
        T = np.eye(4)
        transforms = [T.copy()]
        
        for i in range(self.num_joints):
            dh = self.dh_params[i]
            T_i = self.dh_transform(
                dh['alpha'], dh['a'], dh['d'], 
                joint_angles[i] + dh['theta']
            )
            T = T @ T_i
            transforms.append(T.copy())
        
        position = T[:3, 3]
        rotation = T[:3, :3]
        return position, rotation, transforms

    def jacobian(self, joint_angles, transforms):
        J = np.zeros((6, self.num_joints))
        
        for i in range(self.num_joints):
            T_prev = transforms[i]
            z_prev = T_prev[:3, 2]
            p_prev = T_prev[:3, 3]
            p_end = transforms[-1][:3, 3]
            
            J[:3, i] = np.cross(z_prev, p_end - p_prev)
            J[3:, i] = z_prev
        
        return J

    def null_space_projector(self, J):
        J_pinv = np.linalg.pinv(J)
        N = np.eye(self.num_joints) - J_pinv @ J
        return N

    def elbow_cost(self, joint_angles):
        elbow_joint = joint_angles[3]
        elbow_up_target = -1.0
        elbow_down_target = -2.5
        return (elbow_joint - elbow_up_target) ** 2

    def elbow_cost_gradient(self, joint_angles):
        grad = np.zeros(self.num_joints)
        elbow_joint = joint_angles[3]
        elbow_up_target = -1.0
        grad[3] = 2 * (elbow_joint - elbow_up_target)
        return grad

    def joint_center_cost(self, joint_angles):
        return 0.1 * np.sum((joint_angles - self.joint_centers) ** 2)

    def joint_center_gradient(self, joint_angles):
        return 0.2 * (joint_angles - self.joint_centers)

    def continuity_cost(self, joint_angles, previous_solution):
        if previous_solution is None:
            return 0
        return np.sum((joint_angles - previous_solution) ** 2)

    def continuity_gradient(self, joint_angles, previous_solution):
        if previous_solution is None:
            return np.zeros(self.num_joints)
        return 2 * (joint_angles - previous_solution)

    def inverse_kinematics_stable(self, target_pos, target_rot=None, initial_guess=None,
                                   max_iterations=1000, tolerance=1e-6, step_size=0.1,
                                   damping=0.01, null_space_weight=0.1,
                                   continuity_weight=0.0, elbow_preference_weight=0.0,
                                   center_weight=0.01):
        if initial_guess is None:
            if self.previous_solution is not None:
                joint_angles = self.previous_solution.copy()
            else:
                joint_angles = np.zeros(self.num_joints)
        else:
            joint_angles = initial_guess.copy()
        
        previous_for_continuity = self.previous_solution if continuity_weight > 0 else None
        
        if target_rot is None:
            target_rot = np.eye(3)
        
        best_solution = joint_angles.copy()
        best_error = float('inf')
        
        for iteration in range(max_iterations):
            pos, rot, transforms = self.forward_kinematics(joint_angles)
            
            pos_error = target_pos - pos
            
            rot_error = np.zeros(3)
            if target_rot is not None:
                error_rot = target_rot @ rot.T
                rot_error = Rotation.from_matrix(error_rot).as_rotvec()
            
            error = np.concatenate([pos_error, rot_error])
            error_norm = np.linalg.norm(error)
            
            if error_norm < best_error:
                best_error = error_norm
                best_solution = joint_angles.copy()
            
            if error_norm < tolerance:
                self.previous_solution = joint_angles.copy()
                return joint_angles, True
            
            J = self.jacobian(joint_angles, transforms)
            
            J_damped = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(6))
            
            delta_theta_main = step_size * (J_damped @ error)
            
            N = self.null_space_projector(J)
            
            null_space_grad = np.zeros(self.num_joints)
            
            if elbow_preference_weight > 0:
                null_space_grad -= elbow_preference_weight * self.elbow_cost_gradient(joint_angles)
            
            if center_weight > 0:
                null_space_grad -= center_weight * self.joint_center_gradient(joint_angles)
            
            if continuity_weight > 0 and previous_for_continuity is not None:
                null_space_grad -= continuity_weight * self.continuity_gradient(joint_angles, previous_for_continuity)
            
            delta_theta_null = null_space_weight * (N @ null_space_grad)
            
            delta_theta = delta_theta_main + delta_theta_null
            
            joint_angles += delta_theta
            
            joint_angles = np.clip(joint_angles, self.joint_limits[:, 0], self.joint_limits[:, 1])
        
        self.previous_solution = best_solution.copy()
        if best_error < tolerance * 10:
            return best_solution, True
        return best_solution, False

    def inverse_kinematics_track(self, target_positions, target_rotations=None,
                                  initial_guess=None, **kwargs):
        num_targets = len(target_positions)
        solutions = []
        successes = []
        
        if target_rotations is None:
            target_rotations = [None] * num_targets
        
        current_guess = initial_guess
        
        for i, (target_pos, target_rot) in enumerate(zip(target_positions, target_rotations)):
            if i > 0 and 'continuity_weight' not in kwargs:
                kwargs['continuity_weight'] = 0.5
            
            sol, success = self.inverse_kinematics_stable(
                target_pos, target_rot,
                initial_guess=current_guess,
                **kwargs
            )
            
            solutions.append(sol)
            successes.append(success)
            current_guess = sol.copy()
        
        return solutions, successes

    def reset_previous_solution(self):
        self.previous_solution = None

    def inverse_kinematics_damped(self, target_pos, target_rot=None, initial_guess=None,
                                  max_iterations=1000, tolerance=1e-6, step_size=0.1, damping=0.01):
        return self.inverse_kinematics_stable(
            target_pos, target_rot, initial_guess,
            max_iterations, tolerance, step_size, damping,
            null_space_weight=0, continuity_weight=0,
            elbow_preference_weight=0, center_weight=0
        )


def main():
    arm = SevenDOFArm()
    
    print("=" * 60)
    print("7自由度机械臂逆运动学求解器 - 稳定版")
    print("=" * 60)
    
    print("\n测试1: 基础稳定性测试（同一目标多次求解）")
    print("-" * 40)
    target_pos = np.array([0.5, 0.0, 0.5])
    
    results = []
    for i in range(5):
        arm.reset_previous_solution()
        sol, success = arm.inverse_kinematics_stable(
            target_pos,
            initial_guess=np.random.uniform(-0.5, 0.5, 7),
            max_iterations=3000,
            tolerance=1e-4,
            step_size=0.05,
            elbow_preference_weight=1.0,
            center_weight=0.05
        )
        results.append(sol)
        print(f"第{i+1}次: 成功={success}, 解={np.round(sol, 4)}")
    
    results = np.array(results)
    std_dev = np.std(results, axis=0)
    print(f"\n解的标准差: {np.round(std_dev, 4)}")
    print(f"平均标准差: {np.mean(std_dev):.4f}")
    
    print("\n测试2: 轨迹连续性测试")
    print("-" * 40)
    num_points = 10
    trajectory = []
    for t in np.linspace(0, 1, num_points):
        x = 0.5 + 0.2 * t
        y = 0.0
        z = 0.5 + 0.1 * np.sin(t * np.pi)
        trajectory.append(np.array([x, y, z]))
    
    arm.reset_previous_solution()
    solutions, successes = arm.inverse_kinematics_track(
        trajectory,
        max_iterations=2000,
        tolerance=1e-4,
        step_size=0.05,
        continuity_weight=1.0,
        elbow_preference_weight=0.5,
        center_weight=0.02
    )
    
    max_jumps = []
    for i in range(1, len(solutions)):
        jump = np.max(np.abs(solutions[i] - solutions[i-1]))
        max_jumps.append(jump)
        pos_i, _, _ = arm.forward_kinematics(solutions[i])
        err = np.linalg.norm(trajectory[i] - pos_i)
        print(f"点{i}: 成功={successes[i]}, 最大关节跳变={jump:.4f}, 位置误差={err:.2e}")
    
    print(f"\n平均关节跳变: {np.mean(max_jumps):.4f}")
    print(f"最大关节跳变: {np.max(max_jumps):.4f}")
    
    print("\n测试3: 肘部姿态偏好测试")
    print("-" * 40)
    test_configs = [
        ("无肘部偏好", 0.0),
        ("肘部向上偏好", 1.0),
        ("强肘部向上偏好", 5.0),
    ]
    
    target_pos = np.array([0.4, 0.2, 0.5])
    
    for name, elbow_weight in test_configs:
        arm.reset_previous_solution()
        sol, success = arm.inverse_kinematics_stable(
            target_pos,
            max_iterations=3000,
            tolerance=1e-4,
            step_size=0.05,
            elbow_preference_weight=elbow_weight,
            center_weight=0.05
        )
        
        elbow_angle = sol[3]
        pos_verify, _, _ = arm.forward_kinematics(sol)
        err = np.linalg.norm(target_pos - pos_verify)
        
        print(f"{name}:")
        print(f"  肘部关节角: {elbow_angle:.4f} rad ({np.degrees(elbow_angle):.1f}°)")
        print(f"  位置误差: {err:.2e}")
        print(f"  完整解: {np.round(sol, 4)}")


if __name__ == "__main__":
    main()
