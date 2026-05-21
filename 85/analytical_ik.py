import numpy as np
from scipy.spatial.transform import Rotation


class Analytical7DOF:
    """
    7DOF机械臂解析逆运动学求解器
    
    基于臂型角（Arm Angle）参数化所有逆运动学解
    适用于球形腕、肩-肘-腕共面的7DOF构型（如Franka Panda）
    
    参考文献:
    - "Analytical Inverse Kinematics for 7-DOF Manipulators with Application to the Human Arm"
    - "Configuration Control of Redundant Manipulators: Theory and Implementation"
    """
    
    def __init__(self):
        # Franka Panda DH参数
        self.d1 = 0.333   # 基座到肩关节偏移
        self.d3 = 0.316   # 肩关节到肘关节偏移
        self.d5 = 0.384   # 肘关节到腕关节偏移
        self.d7 = 0.2575  # 腕关节到末端偏移
        self.a4 = 0.0825  # 肘关节偏置
        self.a6 = -0.0825 # 腕关节偏置
        
        # 连杆长度
        self.L_upper = np.sqrt(self.d3**2 + self.a4**2)  # 上臂长度
        self.L_lower = np.sqrt(self.d5**2 + self.a6**2)  # 前臂长度
        
        # 关节限位 (rad)
        self.joint_limits = np.array([
            [-2.8973, 2.8973],   # J1
            [-2.8973, 2.8973],   # J2
            [-2.8973, 2.8973],   # J3
            [-3.0718, -0.0698],  # J4 (肘关节)
            [-2.8973, 2.8973],   # J5
            [-0.0175, 3.7525],   # J6
            [-2.8973, 2.8973]    # J7
        ])
        
        self.num_joints = 7
        
    def forward_kinematics(self, q):
        """
        正运动学计算（用于验证解析IK结果）
        """
        T = np.eye(4)
        
        alpha = [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, 0]
        a = [0, 0, 0, 0.0825, 0, -0.0825, 0]
        d = [0.333, 0, 0.316, 0, 0.384, 0, 0.2575]
        
        for i in range(7):
            T = T @ self._dh_matrix(alpha[i], a[i], d[i], q[i])
        
        return T[:3, 3], T[:3, :3]
    
    def _dh_matrix(self, alpha, a, d, theta):
        """DH变换矩阵"""
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])
    
    def _check_joint_limits(self, q):
        """检查关节角是否在限位范围内"""
        for i in range(self.num_joints):
            if q[i] < self.joint_limits[i, 0] or q[i] > self.joint_limits[i, 1]:
                return False
        return True
    
    def _compute_elbow_position(self, shoulder_pos, wrist_pos, psi):
        """
        计算给定臂型角psi时的肘部位置
        
        参数:
            shoulder_pos: 肩关节位置 (3,)
            wrist_pos: 腕关节位置 (3,)
            psi: 臂型角 (arm angle)
            
        返回:
            elbow_pos: 肘部位置 (3,)
        """
        # 肩-腕向量
        v_sw = wrist_pos - shoulder_pos
        d_sw = np.linalg.norm(v_sw)
        
        # 检查可达性
        if d_sw > self.L_upper + self.L_lower or d_sw < abs(self.L_upper - self.L_lower):
            return None
        
        # 垂直于肩-腕连线的平面内的基向量
        u1 = v_sw / d_sw
        
        # 构造垂直于u1的向量
        if abs(u1[0]) < 0.9:
            tmp = np.array([1, 0, 0])
        else:
            tmp = np.array([0, 1, 0])
        
        u2 = np.cross(u1, tmp)
        u2 = u2 / np.linalg.norm(u2)
        u3 = np.cross(u1, u2)
        
        # 余弦定理计算肘部角度
        cos_theta = (self.L_upper**2 + d_sw**2 - self.L_lower**2) / (2 * self.L_upper * d_sw)
        cos_theta = np.clip(cos_theta, -1, 1)
        theta_elbow = np.arccos(cos_theta)
        
        # 肘部在肩-腕连线上的投影距离
        d_proj = self.L_upper * np.cos(theta_elbow)
        
        # 肘部到肩-腕连线的垂直距离
        d_perp = self.L_upper * np.sin(theta_elbow)
        
        # 肘部位置（在垂直平面内旋转臂型角psi）
        elbow_pos = shoulder_pos + d_proj * u1 + d_perp * (np.cos(psi) * u2 + np.sin(psi) * u3)
        
        return elbow_pos
    
    def _solve_shoulder_joints(self, shoulder_pos, elbow_pos):
        """
        求解肩关节关节角 (J1, J2, J3)
        """
        v_se = elbow_pos - shoulder_pos
        
        # 简化实现：基于几何关系求解
        # J1: 基座旋转 (绕Z轴)
        q1 = np.arctan2(v_se[1], v_se[0])
        
        # J2和J3: 肩关节的俯仰和偏转
        # 这部分需要根据具体的机械臂构型详细推导
        # 这里提供一个简化版本
        
        R_z = np.array([
            [np.cos(q1), -np.sin(q1), 0],
            [np.sin(q1), np.cos(q1), 0],
            [0, 0, 1]
        ])
        
        v_se_local = R_z.T @ v_se
        
        q2 = np.arctan2(v_se_local[2], np.sqrt(v_se_local[0]**2 + v_se_local[1]**2))
        q3 = 0.0  # 简化，实际需要更复杂的推导
        
        return np.array([q1, q2, q3])
    
    def _solve_wrist_joints(self, elbow_pos, wrist_pos, R_end):
        """
        求解腕关节关节角 (J5, J6, J7)
        """
        # 简化实现
        v_ew = wrist_pos - elbow_pos
        
        q5 = np.arctan2(v_ew[1], v_ew[0])
        q6 = 0.0
        q7 = 0.0
        
        return np.array([q5, q6, q7])
    
    def inverse_kinematics_single_psi(self, target_pos, target_rot, psi, shoulder_pos=None):
        """
        对单个臂型角psi，计算对应的逆运动学解
        
        参数:
            target_pos: 目标位置 (3,)
            target_rot: 目标旋转矩阵 (3, 3)
            psi: 臂型角
            shoulder_pos: 肩关节位置，默认为[0, 0, d1]
            
        返回:
            q: 关节角 (7,) 或 None（无解）
        """
        if shoulder_pos is None:
            shoulder_pos = np.array([0, 0, self.d1])
        
        # 计算腕关节位置（减去末端工具偏移）
        wrist_pos = target_pos - target_rot @ np.array([0, 0, self.d7])
        
        # 计算肘部位置
        elbow_pos = self._compute_elbow_position(shoulder_pos, wrist_pos, psi)
        if elbow_pos is None:
            return None
        
        # 求解各关节
        q_shoulder = self._solve_shoulder_joints(shoulder_pos, elbow_pos)
        
        # 肘关节角度 (J4)
        v_se = elbow_pos - shoulder_pos
        v_ew = wrist_pos - elbow_pos
        cos_q4 = np.dot(v_se, v_ew) / (np.linalg.norm(v_se) * np.linalg.norm(v_ew))
        cos_q4 = np.clip(cos_q4, -1, 1)
        q4 = np.arccos(cos_q4) - np.pi  # 调整符号
        
        q_wrist = self._solve_wrist_joints(elbow_pos, wrist_pos, target_rot)
        
        q = np.zeros(7)
        q[0:3] = q_shoulder
        q[3] = q4
        q[4:7] = q_wrist
        
        return q
    
    def sample_psi_space(self, target_pos, target_rot=None, num_samples=100, 
                         psi_range=(-np.pi, np.pi), shoulder_pos=None):
        """
        采样臂型角空间，获取所有可能的解
        
        参数:
            target_pos: 目标位置 (3,)
            target_rot: 目标旋转矩阵 (3, 3)，默认为单位矩阵
            num_samples: 采样数量
            psi_range: 臂型角范围 (min, max)
            shoulder_pos: 肩关节位置
            
        返回:
            solutions: 有效解列表 [(psi, q), ...]
            feasibility: 可行性标志列表
        """
        if target_rot is None:
            target_rot = np.eye(3)
        
        solutions = []
        feasibility = []
        
        psi_values = np.linspace(psi_range[0], psi_range[1], num_samples)
        
        for psi in psi_values:
            q = self.inverse_kinematics_single_psi(target_pos, target_rot, psi, shoulder_pos)
            
            if q is not None and self._check_joint_limits(q):
                solutions.append((psi, q))
                feasibility.append(True)
            else:
                feasibility.append(False)
        
        return solutions, feasibility
    
    def inverse_kinematics_optimize_psi(self, target_pos, target_rot=None, 
                                         objective='elbow_up', psi_guess=0.0,
                                         shoulder_pos=None):
        """
        优化臂型角，找到满足特定目标的最优解
        
        参数:
            target_pos: 目标位置 (3,)
            target_rot: 目标旋转矩阵 (3, 3)
            objective: 优化目标
                - 'elbow_up': 肘部尽可能向上
                - 'elbow_down': 肘部尽可能向下
                - 'center': 关节角尽可能接近中心
                - 'manipulability': 最大化可操作度
                - 'continuous': 尽可能接近前一解
            psi_guess: 初始猜测
            shoulder_pos: 肩关节位置
            
        返回:
            q_opt: 最优关节角 (7,)
            psi_opt: 最优臂型角
            success: 是否成功
        """
        if target_rot is None:
            target_rot = np.eye(3)
        
        if shoulder_pos is None:
            shoulder_pos = np.array([0, 0, self.d1])
        
        # 简单的网格搜索 + 局部优化
        psi_candidates = np.linspace(-np.pi, np.pi, 50)
        
        best_cost = float('inf')
        best_q = None
        best_psi = None
        
        for psi in psi_candidates:
            q = self.inverse_kinematics_single_psi(target_pos, target_rot, psi, shoulder_pos)
            
            if q is None or not self._check_joint_limits(q):
                continue
            
            # 计算代价
            cost = self._compute_objective(q, psi, objective)
            
            if cost < best_cost:
                best_cost = cost
                best_q = q.copy()
                best_psi = psi
        
        # 局部优化（使用简单的梯度下降）
        if best_q is not None:
            for _ in range(10):
                psi_gradient = 0.1
                for sign in [-1, 1]:
                    psi_test = best_psi + sign * psi_gradient
                    q_test = self.inverse_kinematics_single_psi(
                        target_pos, target_rot, psi_test, shoulder_pos
                    )
                    if q_test is not None and self._check_joint_limits(q_test):
                        cost_test = self._compute_objective(q_test, psi_test, objective)
                        if cost_test < best_cost:
                            best_cost = cost_test
                            best_q = q_test.copy()
                            best_psi = psi_test
        
        return best_q, best_psi, best_q is not None
    
    def _compute_objective(self, q, psi, objective):
        """计算优化目标函数值"""
        if objective == 'elbow_up':
            # 肘部向上：最大化肘部Z坐标
            # 或者最小化q4的绝对值（根据具体构型）
            return -q[3]
        
        elif objective == 'elbow_down':
            return q[3]
        
        elif objective == 'center':
            # 关节中心代价
            centers = (self.joint_limits[:, 0] + self.joint_limits[:, 1]) / 2
            return np.sum((q - centers)**2)
        
        elif objective == 'manipulability':
            # 简化：肘关节接近中间位置时可操作度较好
            q4_center = (self.joint_limits[3, 0] + self.joint_limits[3, 1]) / 2
            return (q[3] - q4_center)**2
        
        else:
            return 0.0
    
    def inverse_kinematics_all_solutions(self, target_pos, target_rot=None, 
                                          shoulder_pos=None, num_psi=100):
        """
        获取所有可行的逆运动学解（离散采样）
        
        返回:
            psi_values: 臂型角数组
            q_solutions: 关节角解列表
            valid_mask: 有效性布尔数组
        """
        if target_rot is None:
            target_rot = np.eye(3)
        
        psi_values = np.linspace(-np.pi, np.pi, num_psi)
        q_solutions = []
        valid_mask = []
        
        for psi in psi_values:
            q = self.inverse_kinematics_single_psi(target_pos, target_rot, psi, shoulder_pos)
            
            if q is not None and self._check_joint_limits(q):
                q_solutions.append(q)
                valid_mask.append(True)
            else:
                q_solutions.append(None)
                valid_mask.append(False)
        
        return psi_values, q_solutions, np.array(valid_mask)
    
    def compute_manipulability_ellipsoid(self, q):
        """
        计算可操作度椭球（用于评估不同臂型角的性能）
        """
        # 简化的雅可比矩阵计算
        J = np.zeros((6, 7))
        
        # 这里应该实现完整的雅可比计算
        # 为简化，返回条件数的近似值
        
        # 简化指标：肘关节角度距离中间位置的远近
        q4_center = (self.joint_limits[3, 0] + self.joint_limits[3, 1]) / 2
        manipulability = 1.0 / (1.0 + abs(q[3] - q4_center))
        
        return manipulability
    
    def get_elbow_position_from_psi(self, target_pos, target_rot, psi, shoulder_pos=None):
        """
        根据臂型角计算肘部位置（用于可视化）
        """
        if shoulder_pos is None:
            shoulder_pos = np.array([0, 0, self.d1])
        
        wrist_pos = target_pos - target_rot @ np.array([0, 0, self.d7])
        
        return self._compute_elbow_position(shoulder_pos, wrist_pos, psi)


class AnalyticalIKWrapper:
    """
    解析IK与数值IK的统一接口
    """
    
    def __init__(self):
        self.analytical_solver = Analytical7DOF()
        self.previous_solution = None
    
    def solve(self, target_pos, target_rot=None, method='auto', psi=None, **kwargs):
        """
        统一的逆运动学求解接口
        
        参数:
            target_pos: 目标位置 (3,)
            target_rot: 目标旋转矩阵 (3, 3)
            method: 求解方法
                - 'auto': 优先使用解析解，失败则使用数值解
                - 'analytical': 仅使用解析解
                - 'psi': 使用指定的臂型角
                - 'optimize': 优化臂型角
            psi: 臂型角（当method='psi'时使用）
            **kwargs: 其他参数（如objective, 初始猜测等）
            
        返回:
            q: 关节角 (7,)
            success: 是否成功
            info: 额外信息（如使用的方法、臂型角等）
        """
        if target_rot is None:
            target_rot = np.eye(3)
        
        info = {}
        
        if method == 'psi' and psi is not None:
            # 使用指定臂型角
            q = self.analytical_solver.inverse_kinematics_single_psi(
                target_pos, target_rot, psi
            )
            if q is not None:
                self.previous_solution = q.copy()
                info['psi'] = psi
                info['method'] = 'analytical_fixed_psi'
                return q, True, info
        
        elif method == 'optimize':
            # 优化臂型角
            objective = kwargs.get('objective', 'elbow_up')
            q, psi_opt, success = self.analytical_solver.inverse_kinematics_optimize_psi(
                target_pos, target_rot, objective=objective
            )
            if success:
                self.previous_solution = q.copy()
                info['psi'] = psi_opt
                info['objective'] = objective
                info['method'] = 'analytical_optimized'
                return q, True, info
        
        elif method in ['auto', 'analytical']:
            # 尝试优化方式
            q, psi_opt, success = self.analytical_solver.inverse_kinematics_optimize_psi(
                target_pos, target_rot, objective='center'
            )
            if success:
                self.previous_solution = q.copy()
                info['psi'] = psi_opt
                info['method'] = 'analytical_auto'
                return q, True, info
            
            if method == 'analytical':
                return None, False, info
        
        # 如果解析方法失败，回退到数值方法（需要配合原有的数值IK）
        info['method'] = 'numerical_fallback'
        info['warning'] = 'Analytical IK failed, using numerical IK'
        return self._fallback_numerical(target_pos, target_rot, **kwargs)
    
    def sample_solution_space(self, target_pos, target_rot=None, num_samples=50):
        """
        采样整个解空间，返回所有可行解和对应的臂型角
        """
        psi_values, q_solutions, valid_mask = self.analytical_solver.inverse_kinematics_all_solutions(
            target_pos, target_rot, num_psi=num_samples
        )
        
        valid_psis = psi_values[valid_mask]
        valid_qs = [q for q, v in zip(q_solutions, valid_mask) if v]
        
        return valid_psis, valid_qs, valid_mask
    
    def _fallback_numerical(self, target_pos, target_rot, **kwargs):
        """
        回退到数值方法（占位，需要导入原有的数值IK实现）
        """
        # 在实际使用中，这里应该调用数值IK求解器
        # 这里返回一个简单的零解作为示例
        return np.zeros(7), False, {}


def main():
    """演示解析IK的基本功能"""
    print("=" * 60)
    print("7DOF机械臂解析逆运动学演示")
    print("=" * 60)
    
    solver = Analytical7DOF()
    
    # 测试目标
    target_pos = np.array([0.5, 0.0, 0.5])
    target_rot = np.eye(3)
    
    print(f"\n目标位置: {target_pos}")
    print("-" * 50)
    
    # 演示1: 采样臂型角空间
    print("\n演示1: 采样臂型角空间，查找可行解")
    solutions, feasibility = solver.sample_psi_space(
        target_pos, target_rot, num_samples=20
    )
    
    print(f"  采样数量: 20")
    print(f"  可行解数量: {len(solutions)}")
    
    if len(solutions) > 0:
        print(f"  臂型角范围: [{solutions[0][0]:.3f}, {solutions[-1][0]:.3f}] rad")
        print(f"  示例解 (psi={solutions[0][0]:.3f}): {solutions[0][1].round(4)}")
    
    # 演示2: 优化臂型角
    print("\n演示2: 优化臂型角获取特定目标的解")
    objectives = ['elbow_up', 'elbow_down', 'center']
    
    for obj in objectives:
        q_opt, psi_opt, success = solver.inverse_kinematics_optimize_psi(
            target_pos, target_rot, objective=obj
        )
        
        if success:
            print(f"\n  目标 '{obj}':")
            print(f"    最优臂型角: {psi_opt:.4f} rad")
            print(f"    关节角: {q_opt.round(4)}")
            
            # 验证正运动学
            pos_fk, rot_fk = solver.forward_kinematics(q_opt)
            err = np.linalg.norm(target_pos - pos_fk)
            print(f"    位置误差: {err:.2e}")
    
    # 演示3: 获取所有解
    print("\n演示3: 获取所有可行解的臂型角范围")
    psi_values, q_solutions, valid_mask = solver.inverse_kinematics_all_solutions(
        target_pos, target_rot, num_psi=100
    )
    
    valid_psis = psi_values[valid_mask]
    if len(valid_psis) > 0:
        print(f"  可行臂型角范围: [{valid_psis.min():.4f}, {valid_psis.max():.4f}] rad")
        print(f"  可行臂型角区间长度: {valid_psis.max() - valid_psis.min():.4f} rad")
        print(f"  可行解占比: {np.mean(valid_mask)*100:.1f}%")
        
        # 计算不同臂型角对应的肘部位置
        elbow_heights = []
        for psi in valid_psis[:10]:
            elbow_pos = solver.get_elbow_position_from_psi(target_pos, target_rot, psi)
            if elbow_pos is not None:
                elbow_heights.append(elbow_pos[2])
        
        if len(elbow_heights) > 0:
            print(f"  肘部Z坐标范围: [{min(elbow_heights):.4f}, {max(elbow_heights):.4f}] m")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
