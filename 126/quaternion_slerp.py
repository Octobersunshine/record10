import math
import numpy as np


class Quaternion:
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Quaternion(w={self.w:.6f}, x={self.x:.6f}, y={self.y:.6f}, z={self.z:.6f})"

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
            x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
            y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
            z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            return Quaternion(w, x, y, z)
        elif isinstance(other, (int, float)):
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        else:
            raise TypeError("Unsupported operand type for multiplication")

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        raise TypeError("Unsupported operand type for multiplication")

    def __add__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(
                self.w + other.w,
                self.x + other.x,
                self.y + other.y,
                self.z + other.z
            )
        else:
            raise TypeError("Unsupported operand type for addition")

    def __sub__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(
                self.w - other.w,
                self.x - other.x,
                self.y - other.y,
                self.z - other.z
            )
        else:
            raise TypeError("Unsupported operand type for subtraction")

    def __neg__(self):
        return Quaternion(-self.w, -self.x, -self.y, -self.z)

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm(self):
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        n = self.norm()
        if n == 0:
            return Quaternion(1, 0, 0, 0)
        return Quaternion(self.w / n, self.x / n, self.y / n, self.z / n)

    def inverse(self):
        n_sq = self.w**2 + self.x**2 + self.y**2 + self.z**2
        if n_sq == 0:
            return Quaternion(1, 0, 0, 0)
        conj = self.conjugate()
        return Quaternion(
            conj.w / n_sq,
            conj.x / n_sq,
            conj.y / n_sq,
            conj.z / n_sq
        )

    def dot(self, other):
        return self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z

    def to_rotation_matrix(self):
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])

    def transform_vector(self, v):
        v_q = Quaternion(0, v[0], v[1], v[2])
        result = self * v_q * self.conjugate()
        return np.array([result.x, result.y, result.z])

    def to_axis_angle(self):
        w = max(-1.0, min(1.0, self.w))
        half_angle = math.acos(w)
        if abs(half_angle) < 1e-8:
            return np.array([1, 0, 0]), 0.0
        sin_half = math.sin(half_angle)
        axis = np.array([self.x, self.y, self.z]) / sin_half
        return axis, 2 * half_angle

    @staticmethod
    def from_axis_angle(axis, angle):
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)
        half_angle = angle / 2
        sin_half = math.sin(half_angle)
        return Quaternion(
            math.cos(half_angle),
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half
        )

    @staticmethod
    def from_euler(roll, pitch, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return Quaternion(w, x, y, z)

    @staticmethod
    def log(q):
        q = q.normalize()
        w = max(-1.0, min(1.0, q.w))
        half_theta = math.acos(w)
        if abs(half_theta) < 1e-8:
            return Quaternion(0, 0, 0, 0)
        sin_half_theta = math.sin(half_theta)
        factor = half_theta / sin_half_theta
        return Quaternion(0, q.x * factor, q.y * factor, q.z * factor)

    @staticmethod
    def exp(q):
        v_norm = math.sqrt(q.x**2 + q.y**2 + q.z**2)
        if v_norm < 1e-8:
            return Quaternion(math.exp(q.w), 0, 0, 0)
        half_theta = v_norm
        sin_half = math.sin(half_theta)
        cos_half = math.cos(half_theta)
        factor = sin_half / v_norm
        return Quaternion(cos_half, q.x * factor, q.y * factor, q.z * factor)


def slerp(q1, q2, t, ensure_shortest_path=True):
    q1 = q1.normalize()
    q2 = q2.normalize()

    dot = q1.dot(q2)

    if ensure_shortest_path and dot < 0.0:
        q2 = Quaternion(-q2.w, -q2.x, -q2.y, -q2.z)
        dot = -dot

    if dot > 0.9995:
        result = Quaternion(
            q1.w + t * (q2.w - q1.w),
            q1.x + t * (q2.x - q1.x),
            q1.y + t * (q2.y - q1.y),
            q1.z + t * (q2.z - q1.z)
        )
        return result.normalize()

    dot = max(-1.0, min(1.0, dot))
    
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)

    theta = theta_0 * t
    sin_theta = math.sin(theta)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return Quaternion(
        s0 * q1.w + s1 * q2.w,
        s0 * q1.x + s1 * q2.x,
        s0 * q1.y + s1 * q2.y,
        s0 * q1.z + s1 * q2.z
    )


def intermediate_quaternion(qi, qj, qk, ensure_shortest_path=True):
    qi_inv = qi.inverse()
    
    q_ij = qi_inv * qj
    q_ik = qi_inv * qk
    
    log_ij = Quaternion.log(q_ij)
    log_ik = Quaternion.log(q_ik)
    
    log_avg = (log_ij + log_ik) * (-0.25)
    exp_val = Quaternion.exp(log_avg)
    
    result = qj * exp_val
    return result.normalize()


def squad(q0, q1, q2, q3, t, ensure_shortest_path=True):
    a = intermediate_quaternion(q0, q1, q2, ensure_shortest_path)
    b = intermediate_quaternion(q1, q2, q3, ensure_shortest_path)
    
    c = slerp(q1, q2, t, ensure_shortest_path)
    d = slerp(a, b, t, ensure_shortest_path)
    
    return slerp(c, d, 2 * t * (1 - t), ensure_shortest_path)


class QuaternionSpline:
    def __init__(self, keyframes=None):
        self.keyframes = []
        if keyframes:
            self.keyframes = sorted(keyframes, key=lambda k: k['time'])

    def add_keyframe(self, time, rotation):
        self.keyframes.append({'time': time, 'rotation': rotation})
        self.keyframes.sort(key=lambda k: k['time'])

    def _get_segment(self, time):
        if len(self.keyframes) < 2:
            return None, None, 0
        
        if time <= self.keyframes[0]['time']:
            return 0, 1, 0.0
        
        if time >= self.keyframes[-1]['time']:
            return len(self.keyframes) - 2, len(self.keyframes) - 1, 1.0
        
        for i in range(len(self.keyframes) - 1):
            t1, t2 = self.keyframes[i]['time'], self.keyframes[i + 1]['time']
            if t1 <= time <= t2:
                t = (time - t1) / (t2 - t1) if t2 != t1 else 0
                return i, i + 1, t
        
        return len(self.keyframes) - 2, len(self.keyframes) - 1, 1.0

    def evaluate_slerp(self, time):
        if len(self.keyframes) == 0:
            return Quaternion(1, 0, 0, 0)
        if len(self.keyframes) == 1:
            return self.keyframes[0]['rotation']
        
        i, j, t = self._get_segment(time)
        return slerp(self.keyframes[i]['rotation'], self.keyframes[j]['rotation'], t)

    def evaluate_squad(self, time):
        n = len(self.keyframes)
        if n == 0:
            return Quaternion(1, 0, 0, 0)
        if n == 1:
            return self.keyframes[0]['rotation']
        if n == 2:
            return self.evaluate_slerp(time)
        
        i, j, t = self._get_segment(time)
        
        q0_idx = max(0, i - 1)
        q3_idx = min(n - 1, j + 1)
        
        q0 = self.keyframes[q0_idx]['rotation']
        q1 = self.keyframes[i]['rotation']
        q2 = self.keyframes[j]['rotation']
        q3 = self.keyframes[q3_idx]['rotation']
        
        return squad(q0, q1, q2, q3, t)


class DualQuaternion:
    def __init__(self, real=None, dual=None):
        if real is None:
            self.real = Quaternion(1, 0, 0, 0)
        else:
            self.real = real.normalize()
        
        if dual is None:
            self.dual = Quaternion(0, 0, 0, 0)
        else:
            self.dual = dual

    def __repr__(self):
        return f"DualQuaternion(\n  real={self.real},\n  dual={self.dual}\n)"

    def __mul__(self, other):
        if isinstance(other, DualQuaternion):
            new_real = self.real * other.real
            new_dual = self.real * other.dual + self.dual * other.real
            return DualQuaternion(new_real, new_dual)
        elif isinstance(other, (int, float)):
            return DualQuaternion(self.real * other, self.dual * other)
        else:
            raise TypeError("Unsupported operand type for multiplication")

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return DualQuaternion(self.real * other, self.dual * other)
        raise TypeError("Unsupported operand type for multiplication")

    def __add__(self, other):
        if isinstance(other, DualQuaternion):
            return DualQuaternion(
                self.real + other.real,
                self.dual + other.dual
            )
        else:
            raise TypeError("Unsupported operand type for addition")

    def conjugate(self):
        return DualQuaternion(self.real.conjugate(), self.dual.conjugate())

    def norm(self):
        real_norm = self.real.norm()
        if real_norm < 1e-8:
            return 0.0
        dual_dot = self.real.dot(self.dual)
        return real_norm + (dual_dot / real_norm)

    def normalize(self):
        real_norm = self.real.norm()
        if real_norm < 1e-8:
            return DualQuaternion()
        
        real_normalized = self.real * (1.0 / real_norm)
        real_dot_dual = self.real.dot(self.dual)
        dual_normalized = (self.dual * real_norm - self.real * real_dot_dual) / (real_norm * real_norm)
        
        return DualQuaternion(real_normalized, dual_normalized)

    def inverse(self):
        real_inv = self.real.inverse()
        dual_inv = -real_inv * self.dual * real_inv
        return DualQuaternion(real_inv, dual_inv)

    def to_matrix(self):
        R = self.real.to_rotation_matrix()
        
        t_q = 2.0 * self.dual * self.real.conjugate()
        t = np.array([t_q.x, t_q.y, t_q.z])
        
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = t
        return M

    def transform_point(self, p):
        M = self.to_matrix()
        p_hom = np.append(p, 1.0)
        result = M @ p_hom
        return result[:3]

    @staticmethod
    def from_rotation_translation(rotation, translation):
        real = rotation.normalize()
        t = Quaternion(0, translation[0], translation[1], translation[2])
        dual = 0.5 * t * real
        return DualQuaternion(real, dual)

    @staticmethod
    def from_matrix(M):
        R = M[:3, :3]
        t = M[:3, 3]
        
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        rotation = Quaternion(w, x, y, z).normalize()
        return DualQuaternion.from_rotation_translation(rotation, t)


def dual_quaternion_slerp(dq1, dq2, t, ensure_shortest_path=True):
    dq1 = dq1.normalize()
    dq2 = dq2.normalize()

    dot = dq1.real.dot(dq2.real)

    if ensure_shortest_path and dot < 0.0:
        dq2 = DualQuaternion(-dq2.real, -dq2.dual)
        dot = -dot

    real_interp = slerp(dq1.real, dq2.real, t, ensure_shortest_path=False)
    dual_interp = dq1.dual * (1 - t) + dq2.dual * t

    return DualQuaternion(real_interp, dual_interp).normalize()


def dual_quaternion_squad(dq0, dq1, dq2, dq3, t, ensure_shortest_path=True):
    def intermediate_dq(dqi, dqj, dqk):
        qi_inv = dqi.real.inverse()
        
        q_ij = qi_inv * dqj.real
        q_ik = qi_inv * dqk.real
        
        log_ij = Quaternion.log(q_ij)
        log_ik = Quaternion.log(q_ik)
        
        log_avg = (log_ij + log_ik) * (-0.25)
        exp_val = Quaternion.exp(log_avg)
        
        real_result = dqj.real * exp_val
        
        dual_result = dqj.dual * exp_val + dqj.real * Quaternion.log(exp_val) * 0.5
        
        return DualQuaternion(real_result.normalize(), dual_result)

    a = intermediate_dq(dq0, dq1, dq2)
    b = intermediate_dq(dq1, dq2, dq3)
    
    c = dual_quaternion_slerp(dq1, dq2, t, ensure_shortest_path)
    d = dual_quaternion_slerp(a, b, t, ensure_shortest_path)
    
    return dual_quaternion_slerp(c, d, 2 * t * (1 - t), ensure_shortest_path)


class DualQuaternionSpline:
    def __init__(self, keyframes=None):
        self.keyframes = []
        if keyframes:
            self.keyframes = sorted(keyframes, key=lambda k: k['time'])

    def add_keyframe(self, time, transform):
        self.keyframes.append({'time': time, 'transform': transform})
        self.keyframes.sort(key=lambda k: k['time'])

    def _get_segment(self, time):
        if len(self.keyframes) < 2:
            return None, None, 0
        
        if time <= self.keyframes[0]['time']:
            return 0, 1, 0.0
        
        if time >= self.keyframes[-1]['time']:
            return len(self.keyframes) - 2, len(self.keyframes) - 1, 1.0
        
        for i in range(len(self.keyframes) - 1):
            t1, t2 = self.keyframes[i]['time'], self.keyframes[i + 1]['time']
            if t1 <= time <= t2:
                t = (time - t1) / (t2 - t1) if t2 != t1 else 0
                return i, i + 1, t
        
        return len(self.keyframes) - 2, len(self.keyframes) - 1, 1.0

    def evaluate_slerp(self, time):
        if len(self.keyframes) == 0:
            return DualQuaternion()
        if len(self.keyframes) == 1:
            return self.keyframes[0]['transform']
        
        i, j, t = self._get_segment(time)
        return dual_quaternion_slerp(
            self.keyframes[i]['transform'], 
            self.keyframes[j]['transform'], 
            t
        )

    def evaluate_squad(self, time):
        n = len(self.keyframes)
        if n == 0:
            return DualQuaternion()
        if n == 1:
            return self.keyframes[0]['transform']
        if n == 2:
            return self.evaluate_slerp(time)
        
        i, j, t = self._get_segment(time)
        
        q0_idx = max(0, i - 1)
        q3_idx = min(n - 1, j + 1)
        
        dq0 = self.keyframes[q0_idx]['transform']
        dq1 = self.keyframes[i]['transform']
        dq2 = self.keyframes[j]['transform']
        dq3 = self.keyframes[q3_idx]['transform']
        
        return dual_quaternion_squad(dq0, dq1, dq2, dq3, t)


class SkinnedBone:
    def __init__(self, name, parent_name=None):
        self.name = name
        self.parent_name = parent_name
        self.local_transform = DualQuaternion()
        self.world_transform = DualQuaternion()


class SkinningAnimation:
    def __init__(self):
        self.bones = {}
        self.bone_splines = {}
        self.bind_poses = {}

    def add_bone(self, bone, bind_pose=None):
        self.bones[bone.name] = bone
        self.bone_splines[bone.name] = DualQuaternionSpline()
        if bind_pose:
            self.bind_poses[bone.name] = bind_pose
        else:
            self.bind_poses[bone.name] = DualQuaternion()

    def add_bone_keyframe(self, bone_name, time, local_transform):
        if bone_name in self.bone_splines:
            self.bone_splines[bone_name].add_keyframe(time, local_transform)

    def evaluate(self, time, use_squad=True):
        transforms = {}
        for bone_name, spline in self.bone_splines.items():
            if use_squad:
                transforms[bone_name] = spline.evaluate_squad(time)
            else:
                transforms[bone_name] = spline.evaluate_slerp(time)
        
        world_transforms = {}
        for bone_name, bone in self.bones.items():
            if bone.parent_name and bone.parent_name in world_transforms:
                world_transforms[bone_name] = world_transforms[bone.parent_name] * transforms[bone_name]
            else:
                world_transforms[bone_name] = transforms[bone_name]
        
        return world_transforms

    def skin_vertex(self, vertex, bone_weights, world_transforms):
        result = np.zeros(3)
        total_weight = sum(bone_weights.values())
        
        if total_weight < 1e-8:
            return vertex
        
        for bone_name, weight in bone_weights.items():
            if bone_name in world_transforms:
                inv_bind = self.bind_poses[bone_name].inverse()
                skinning_dq = world_transforms[bone_name] * inv_bind
                transformed = skinning_dq.transform_point(vertex)
                result += transformed * (weight / total_weight)
        
        return result


def compute_total_angle(q1, q2):
    dot = q1.dot(q2)
    dot = max(-1.0, min(1.0, dot))
    return 2 * math.degrees(math.acos(dot))


def main():
    print("="*70)
    print("=== 四元数与双四元数插值演示 (用于骨骼动画)")
    print("="*70 + "\n")

    print("--- 测试1: 基本Slerp插值")
    q1 = Quaternion.from_axis_angle([0, 1, 0], 0)
    q2 = Quaternion.from_axis_angle([0, 1, 0], math.pi)

    print(f"起始四元数 (0度): {q1}")
    print(f"结束四元数 (180度): {q2}\n")

    print("Slerp 插值结果:")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        q_interp = slerp(q1, q2, t)
        angle = 2 * math.acos(max(-1, min(1, q_interp.w)))
        print(f"  t={t:.2f}: {q_interp}, 角度={math.degrees(angle):.1f}度")

    print("\n" + "="*70 + "\n")

    print("--- 测试2: 符号二义性修复演示")
    q_a = Quaternion.from_axis_angle([0, 1, 0], math.radians(170))
    q_b_neg = Quaternion(-q_a.w, -q_a.x, -q_a.y, -q_a.z)
    
    print(f"q_a (170度绕Y轴): {q_a}")
    print(f"q_b_neg (-q_a, 等价于 -190度): {q_b_neg}")
    print(f"q_a 和 q_b_neg 点积: {q_a.dot(q_b_neg):.6f}")
    print()
    
    q_start = Quaternion.from_axis_angle([0, 1, 0], 0)
    
    angle_without_fix = compute_total_angle(q_start, q_a)
    angle_with_fix = compute_total_angle(q_start, q_b_neg)
    
    print(f"到 q_a 的弧长: {angle_without_fix:.1f} 度")
    print(f"到 q_b_neg 的弧长 (未修复): {angle_with_fix:.1f} 度")
    print()
    
    print("开启最短路径保证:")
    for t in [0.0, 0.5, 1.0]:
        q_interp = slerp(q_start, q_b_neg, t, ensure_shortest_path=True)
        angle = 2 * math.degrees(math.acos(max(-1, min(1, q_interp.w))))
        print(f"  t={t:.2f}: {q_interp}, 角度={angle:.1f}度")
    
    print("\n关闭最短路径保证:")
    for t in [0.0, 0.5, 1.0]:
        q_interp = slerp(q_start, q_b_neg, t, ensure_shortest_path=False)
        angle = 2 * math.degrees(math.acos(max(-1, min(1, q_interp.w))))
        print(f"  t={t:.2f}: {q_interp}, 角度={angle:.1f}度")

    print("\n" + "="*70 + "\n")

    print("--- 测试3: Squad 球面二次插值 (Slerp vs Squad 对比)")
    keyframes_rot = [
        {'time': 0.0, 'rotation': Quaternion.from_euler(0, 0, 0)},
        {'time': 1.0, 'rotation': Quaternion.from_euler(0, math.pi/2, 0)},
        {'time': 2.0, 'rotation': Quaternion.from_euler(math.pi/4, math.pi, 0)},
        {'time': 3.0, 'rotation': Quaternion.from_euler(math.pi/2, math.pi, math.pi/4)},
    ]

    spline = QuaternionSpline(keyframes_rot)
    
    print("时间    Slerp 插值角度    Squad 插值角度")
    print("-" * 50)
    for time in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        q_slerp = spline.evaluate_slerp(time)
        q_squad = spline.evaluate_squad(time)
        angle_slerp = 2 * math.degrees(math.acos(max(-1, min(1, q_slerp.w))))
        angle_squad = 2 * math.degrees(math.acos(max(-1, min(1, q_squad.w))))
        print(f"  {time:.1f}s    {angle_slerp:>8.1f}度        {angle_squad:>8.1f}度")

    print("\n" + "="*70 + "\n")

    print("--- 测试4: 双四元数 (旋转+平移) 插值")
    dq1 = DualQuaternion.from_rotation_translation(
        Quaternion.from_axis_angle([0, 1, 0], 0),
        np.array([0, 0, 0])
    )
    dq2 = DualQuaternion.from_rotation_translation(
        Quaternion.from_axis_angle([0, 1, 0], math.pi/2),
        np.array([2, 1, 0])
    )

    print("起始变换:")
    print(f"  位置: [0, 0, 0], 旋转: 0度")
    print("结束变换:")
    print(f"  位置: [2, 1, 0], 旋转: 90度\n")

    print("双四元数 Slerp 插值 (测试点 [1, 0, 0] 的变换):")
    test_point = np.array([1, 0, 0])
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        dq_interp = dual_quaternion_slerp(dq1, dq2, t)
        transformed = dq_interp.transform_point(test_point)
        
        axis, angle = dq_interp.real.to_axis_angle()
        t_q = 2.0 * dq_interp.dual * dq_interp.real.conjugate()
        pos = np.array([t_q.x, t_q.y, t_q.z])
        
        print(f"  t={t:.2f}: 位置=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], "
              f"角度={math.degrees(angle):.1f}度, 点变换=[{transformed[0]:.2f}, {transformed[1]:.2f}, {transformed[2]:.2f}]")

    print("\n" + "="*70 + "\n")

    print("--- 测试5: 蒙皮动画演示 (简单骨骼链)")
    anim = SkinningAnimation()
    
    root_bone = SkinnedBone("Root", None)
    upper_bone = SkinnedBone("UpperArm", "Root")
    lower_bone = SkinnedBone("LowerArm", "UpperArm")
    
    anim.add_bone(root_bone, DualQuaternion())
    anim.add_bone(upper_bone, DualQuaternion.from_rotation_translation(
        Quaternion(), np.array([0, 0, 0])
    ))
    anim.add_bone(lower_bone, DualQuaternion.from_rotation_translation(
        Quaternion(), np.array([1, 0, 0])
    ))
    
    anim.add_bone_keyframe("Root", 0.0, DualQuaternion())
    anim.add_bone_keyframe("Root", 2.0, DualQuaternion.from_rotation_translation(
        Quaternion.from_axis_angle([0, 0, 1], math.pi/4),
        np.array([0.5, 0, 0])
    ))
    
    anim.add_bone_keyframe("UpperArm", 0.0, DualQuaternion())
    anim.add_bone_keyframe("UpperArm", 2.0, DualQuaternion.from_rotation_translation(
        Quaternion.from_axis_angle([0, 0, 1], math.pi/3),
        np.array([0, 0, 0])
    ))
    
    anim.add_bone_keyframe("LowerArm", 0.0, DualQuaternion())
    anim.add_bone_keyframe("LowerArm", 2.0, DualQuaternion.from_rotation_translation(
        Quaternion.from_axis_angle([0, 0, 1], -math.pi/4),
        np.array([0, 0, 0])
    ))
    
    print("时间点    肘部位置    手部位置")
    print("-" * 50)
    elbow_pos_bind = np.array([1, 0, 0])
    hand_pos_bind = np.array([2, 0, 0])
    
    for time in [0.0, 0.5, 1.0, 1.5, 2.0]:
        transforms = anim.evaluate(time, use_squad=True)
        
        elbow_weights = {"UpperArm": 1.0}
        hand_weights = {"UpperArm": 0.3, "LowerArm": 0.7}
        
        elbow_world = anim.skin_vertex(elbow_pos_bind, elbow_weights, transforms)
        hand_world = anim.skin_vertex(hand_pos_bind, hand_weights, transforms)
        
        print(f"  {time:.1f}s    [{elbow_world[0]:.2f}, {elbow_world[1]:.2f}]    "
              f"[{hand_world[0]:.2f}, {hand_world[1]:.2f}]")

    print("\n" + "="*70)
    print("演示完成! 包含功能:")
    print("  1. Slerp - 球面线性插值")
    print("  2. Squad - 球面二次插值 (C1连续)")
    print("  3. DualQuaternion - 双四元数 (旋转+平移)")
    print("  4. Skinning - 蒙皮动画 (线性混合蒙皮)")
    print("="*70)


if __name__ == "__main__":
    main()
