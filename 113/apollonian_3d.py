from decimal import Decimal, getcontext
from typing import List, Tuple
import math

PRECISION = 60
getcontext().prec = PRECISION


class Vector3Decimal:
    def __init__(self, x: Decimal, y: Decimal, z: Decimal):
        self.x = Decimal(str(x)) if not isinstance(x, Decimal) else x
        self.y = Decimal(str(y)) if not isinstance(y, Decimal) else y
        self.z = Decimal(str(z)) if not isinstance(z, Decimal) else z
    
    def __add__(self, other):
        if isinstance(other, Vector3Decimal):
            return Vector3Decimal(self.x + other.x, self.y + other.y, self.z + other.z)
        return Vector3Decimal(self.x + Decimal(str(other)), 
                              self.y + Decimal(str(other)), 
                              self.z + Decimal(str(other)))
    
    def __sub__(self, other):
        if isinstance(other, Vector3Decimal):
            return Vector3Decimal(self.x - other.x, self.y - other.y, self.z - other.z)
        return Vector3Decimal(self.x - Decimal(str(other)), 
                              self.y - Decimal(str(other)), 
                              self.z - Decimal(str(other)))
    
    def __mul__(self, scalar):
        s = Decimal(str(scalar)) if not isinstance(scalar, Decimal) else scalar
        return Vector3Decimal(self.x * s, self.y * s, self.z * s)
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar):
        s = Decimal(str(scalar)) if not isinstance(scalar, Decimal) else scalar
        return Vector3Decimal(self.x / s, self.y / s, self.z / s)
    
    def dot(self, other) -> Decimal:
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def norm_sq(self) -> Decimal:
        return self.dot(self)
    
    def norm(self) -> Decimal:
        return sqrt_decimal(self.norm_sq())
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return float(self.x), float(self.y), float(self.z)
    
    def __repr__(self):
        return f"({self.x}, {self.y}, {self.z})"


def sqrt_decimal(d: Decimal) -> Decimal:
    if d < 0:
        if d > Decimal('-1e-40'):
            return Decimal('0')
        raise ValueError(f"Cannot compute sqrt of negative number: {d}")
    return d.sqrt()


def solve_k5_3d(k1: Decimal, k2: Decimal, k3: Decimal, k4: Decimal) -> Tuple[Decimal, Decimal]:
    k_sum = k1 + k2 + k3 + k4
    cross = k1*k2 + k1*k3 + k1*k4 + k2*k3 + k2*k4 + k3*k4
    if cross < 0:
        cross = max(cross, Decimal('0'))
    k_prod = Decimal('2') * sqrt_decimal(cross)
    return k_sum + k_prod, k_sum - k_prod


def solve_b5_3d(b1: Vector3Decimal, b2: Vector3Decimal, b3: Vector3Decimal, b4: Vector3Decimal,
                k1: Decimal, k2: Decimal, k3: Decimal, k4: Decimal, k5: Decimal) -> Tuple[Vector3Decimal, Vector3Decimal]:
    b_sum = b1 + b2 + b3 + b4
    
    cross_xx = (b1.x*b2.x + b1.x*b3.x + b1.x*b4.x + b2.x*b3.x + b2.x*b4.x + b3.x*b4.x)
    cross_yy = (b1.y*b2.y + b1.y*b3.y + b1.y*b4.y + b2.y*b3.y + b2.y*b4.y + b3.y*b4.y)
    cross_zz = (b1.z*b2.z + b1.z*b3.z + b1.z*b4.z + b2.z*b3.z + b2.z*b4.z + b3.z*b4.z)
    cross_xy = (b1.x*b2.y + b1.y*b2.x + b1.x*b3.y + b1.y*b3.x + b1.x*b4.y + b1.y*b4.x +
                b2.x*b3.y + b2.y*b3.x + b2.x*b4.y + b2.y*b4.x + b3.x*b4.y + b3.y*b4.x)
    cross_xz = (b1.x*b2.z + b1.z*b2.x + b1.x*b3.z + b1.z*b3.x + b1.x*b4.z + b1.z*b4.x +
                b2.x*b3.z + b2.z*b3.x + b2.x*b4.z + b2.z*b4.x + b3.x*b4.z + b3.z*b4.x)
    cross_yz = (b1.y*b2.z + b1.z*b2.y + b1.y*b3.z + b1.z*b3.y + b1.y*b4.z + b1.z*b4.y +
                b2.y*b3.z + b2.z*b3.y + b2.y*b4.z + b2.z*b4.y + b3.y*b4.z + b3.z*b4.y)
    
    M = [[cross_xx, cross_xy/Decimal('2'), cross_xz/Decimal('2')],
         [cross_xy/Decimal('2'), cross_yy, cross_yz/Decimal('2')],
         [cross_xz/Decimal('2'), cross_yz/Decimal('2'), cross_zz]]
    
    cross_vec_sq_norm = cross_xx + cross_yy + cross_zz
    
    if cross_vec_sq_norm < Decimal('1e-40'):
        sqrt_cross = Vector3Decimal(0, 0, 0)
    else:
        scale = sqrt_decimal(cross_vec_sq_norm) / Decimal('2')
        avg_x = (b1.x + b2.x + b3.x + b4.x) / Decimal('4')
        avg_y = (b1.y + b2.y + b3.y + b4.y) / Decimal('4')
        avg_z = (b1.z + b2.z + b3.z + b4.z) / Decimal('4')
        norm_factor = sqrt_decimal(avg_x*avg_x + avg_y*avg_y + avg_z*avg_z)
        if norm_factor > Decimal('1e-30'):
            sqrt_cross = Vector3Decimal(avg_x / norm_factor * scale,
                                        avg_y / norm_factor * scale,
                                        avg_z / norm_factor * scale)
        else:
            sqrt_cross = Vector3Decimal(scale, 0, 0)
    
    return (b_sum + sqrt_cross * Decimal('2')) / k5, (b_sum - sqrt_cross * Decimal('2')) / k5


def sphere_tangent(s1: Tuple[float, float, float, float], 
                   s2: Tuple[float, float, float, float], 
                   tol: float = 1e-7) -> bool:
    dx = s1[0] - s2[0]
    dy = s1[1] - s2[1]
    dz = s1[2] - s2[2]
    d_sq = dx*dx + dy*dy + dz*dz
    sum_r = s1[3] + s2[3]
    diff_r = abs(s1[3] - s2[3])
    return abs(d_sq - sum_r*sum_r) < tol or abs(d_sq - diff_r*diff_r) < tol


def sphere_key(s: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    return (int(s[0] * 1e9), int(s[1] * 1e9), int(s[2] * 1e9), int(s[3] * 1e9))


def apollonian_sphere_packing(s1: Tuple[float, float, float, float],
                              s2: Tuple[float, float, float, float],
                              s3: Tuple[float, float, float, float],
                              s4: Tuple[float, float, float, float],
                              max_iter: int = 10,
                              min_radius: float = 1e-6) -> List[Tuple[float, float, float, float]]:
    spheres = [s1, s2, s3, s4]
    seen_spheres = {sphere_key(s1), sphere_key(s2), sphere_key(s3), sphere_key(s4)}
    
    k1 = Decimal(1) / Decimal(str(s1[3]))
    k2 = Decimal(1) / Decimal(str(s2[3]))
    k3 = Decimal(1) / Decimal(str(s3[3]))
    k4 = Decimal(1) / Decimal(str(s4[3]))
    
    z1 = Vector3Decimal(Decimal(str(s1[0])), Decimal(str(s1[1])), Decimal(str(s1[2])))
    z2 = Vector3Decimal(Decimal(str(s2[0])), Decimal(str(s2[1])), Decimal(str(s2[2])))
    z3 = Vector3Decimal(Decimal(str(s3[0])), Decimal(str(s3[1])), Decimal(str(s3[2])))
    z4 = Vector3Decimal(Decimal(str(s4[0])), Decimal(str(s4[1])), Decimal(str(s4[2])))
    
    b1 = z1 * k1
    b2 = z2 * k2
    b3 = z3 * k3
    b4 = z4 * k4
    
    try:
        k5_plus, k5_minus = solve_k5_3d(k1, k2, k3, k4)
    except ValueError:
        return spheres
    
    def get_sphere_candidates(k_val: Decimal, b_a: Vector3Decimal, b_b: Vector3Decimal, 
                              b_c: Vector3Decimal, b_d: Vector3Decimal,
                              ka: Decimal, kb: Decimal, kc: Decimal, kd: Decimal) -> List[Tuple[float, float, float, float]]:
        if abs(k_val) < Decimal('1e-40'):
            return []
        try:
            z5a, z5b = solve_b5_3d(b_a, b_b, b_c, b_d, ka, kb, kc, kd, k_val)
            r_val = float(Decimal(1) / abs(k_val))
            if r_val < min_radius:
                return []
            x1, y1, z1_coord = z5a.to_tuple()
            x2, y2, z2_coord = z5b.to_tuple()
            return [(x1, y1, z1_coord, r_val), (x2, y2, z2_coord, r_val)]
        except:
            return []
    
    candidates = []
    for k in [k5_plus, k5_minus]:
        candidates.extend(get_sphere_candidates(k, b1, b2, b3, b4, k1, k2, k3, k4))
    
    inner_spheres = []
    for sx, sy, sz, sr in candidates:
        test_sphere = (sx, sy, sz, sr)
        if (sphere_tangent(test_sphere, s1, tol=1e-5) and 
            sphere_tangent(test_sphere, s2, tol=1e-5) and 
            sphere_tangent(test_sphere, s3, tol=1e-5) and
            sphere_tangent(test_sphere, s4, tol=1e-5)):
            inner_spheres.append(test_sphere)
    
    s5 = None
    if inner_spheres:
        s5 = min(inner_spheres, key=lambda x: x[3])
        k5 = sphere_key(s5)
        if k5 not in seen_spheres:
            spheres.append(s5)
            seen_spheres.add(k5)
    
    class SphereData:
        def __init__(self, s: Tuple[float, float, float, float], b: Vector3Decimal, k: Decimal):
            self.sphere = s
            self.b = b
            self.k = k
    
    sd1 = SphereData(s1, b1, k1)
    sd2 = SphereData(s2, b2, k2)
    sd3 = SphereData(s3, b3, k3)
    sd4 = SphereData(s4, b4, k4)
    
    quadruple_queue = []
    if s5:
        k5_val = Decimal(1) / Decimal(str(s5[3]))
        z5 = Vector3Decimal(Decimal(str(s5[0])), Decimal(str(s5[1])), Decimal(str(s5[2])))
        b5 = z5 * k5_val
        sd5 = SphereData(s5, b5, k5_val)
        
        quadruple_queue.append((sd1, sd2, sd3, sd5))
        quadruple_queue.append((sd1, sd2, sd4, sd5))
        quadruple_queue.append((sd1, sd3, sd4, sd5))
        quadruple_queue.append((sd2, sd3, sd4, sd5))
    
    seen_quadruples = set()
    
    for iteration in range(max_iter):
        new_quadruples = []
        
        for quad in quadruple_queue:
            quad_key = tuple(sorted([sphere_key(sd.sphere) for sd in quad]))
            if quad_key in seen_quadruples:
                continue
            seen_quadruples.add(quad_key)
            
            sa, sb, sc, sd = quad
            
            try:
                k_new_plus, k_new_minus = solve_k5_3d(sa.k, sb.k, sc.k, sd.k)
            except (ValueError, ZeroDivisionError):
                continue
            
            for k_new in [k_new_plus, k_new_minus]:
                if abs(k_new) < Decimal('1e-40'):
                    continue
                
                try:
                    r_new = float(Decimal(1) / abs(k_new))
                except ZeroDivisionError:
                    continue
                
                if r_new < min_radius:
                    continue
                
                try:
                    z_new1, z_new2 = solve_b5_3d(sa.b, sb.b, sc.b, sd.b, sa.k, sb.k, sc.k, sd.k, k_new)
                except:
                    continue
                
                for z_new in [z_new1, z_new2]:
                    x, y, z_coord = z_new.to_tuple()
                    new_sphere = (x, y, z_coord, r_new)
                    sk = sphere_key(new_sphere)
                    
                    if sk in seen_spheres:
                        continue
                    
                    if (sphere_tangent(new_sphere, sa.sphere, tol=1e-5) and
                        sphere_tangent(new_sphere, sb.sphere, tol=1e-5) and
                        sphere_tangent(new_sphere, sc.sphere, tol=1e-5) and
                        sphere_tangent(new_sphere, sd.sphere, tol=1e-5)):
                        spheres.append(new_sphere)
                        seen_spheres.add(sk)
                        new_b = z_new * k_new
                        new_sd = SphereData(new_sphere, new_b, k_new)
                        new_quadruples.append((sa, sb, sc, new_sd))
                        new_quadruples.append((sa, sb, sd, new_sd))
                        new_quadruples.append((sa, sc, sd, new_sd))
                        new_quadruples.append((sb, sc, sd, new_sd))
        
        quadruple_queue = new_quadruples
        if not quadruple_queue:
            break
    
    return spheres


def calculate_packing_density(spheres: List[Tuple[float, float, float, float]],
                              container_radius: float) -> float:
    total_volume = 0.0
    for s in spheres:
        r = s[3]
        total_volume += (4.0 / 3.0) * math.pi * (r ** 3)
    
    container_volume = (4.0 / 3.0) * math.pi * (container_radius ** 3)
    return total_volume / container_volume


def estimate_saturation_density(initial_radius: float = 1.0, 
                                max_iters: List[int] = [3, 5, 7, 9],
                                min_radii: List[float] = [1e-2, 1e-3, 1e-4, 1e-5]) -> Tuple[float, List[Tuple[int, float, float]]]:
    r = initial_radius
    s1 = (0, 0, 0, r)
    s2 = (2*r, 0, 0, r)
    s3 = (r, math.sqrt(3)*r, 0, r)
    s4 = (r, math.sqrt(3)*r/3, 2*math.sqrt(6)*r/3, r)
    
    results = []
    for max_iter in max_iters:
        for min_r in min_radii:
            spheres = apollonian_sphere_packing(s1, s2, s3, s4, 
                                                max_iter=max_iter, 
                                                min_radius=min_r)
            outer_r = 3 * r
            density = calculate_packing_density(spheres, outer_r)
            results.append((max_iter, min_r, density, len(spheres)))
    
    if results:
        results.sort(key=lambda x: (-x[0], x[1]))
        best_density = results[0][2]
    else:
        best_density = 0.0
    
    return best_density, results


def create_tetrahedral_spheres(radius: float = 1.0) -> Tuple[Tuple, Tuple, Tuple, Tuple]:
    s1 = (0.0, 0.0, 0.0, radius)
    s2 = (2.0 * radius, 0.0, 0.0, radius)
    s3 = (radius, math.sqrt(3) * radius, 0.0, radius)
    s4 = (radius, math.sqrt(3) * radius / 3.0, 2.0 * math.sqrt(6) * radius / 3.0, radius)
    return s1, s2, s3, s4


if __name__ == "__main__":
    print("=" * 60)
    print("三维阿波罗尼斯球填充模拟")
    print("=" * 60)
    
    r = 1.0
    s1, s2, s3, s4 = create_tetrahedral_spheres(r)
    
    print(f"\n初始四个正四面体内切球 (r={r}):")
    for i, s in enumerate([s1, s2, s3, s4]):
        print(f"  球 {i+1}: 球心=({s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f}), 半径={s[3]:.4f}")
    
    print(f"\n开始生成球填充...")
    spheres = apollonian_sphere_packing(s1, s2, s3, s4, max_iter=6, min_radius=0.001)
    
    print(f"\n生成了 {len(spheres)} 个球:")
    for i, s in enumerate(spheres[:15]):
        print(f"  球 {i+1:3d}: 球心=({s[0]:10.6f}, {s[1]:10.6f}, {s[2]:10.6f}), 半径={s[3]:10.6f}")
    
    if len(spheres) > 15:
        print(f"  ... 还有 {len(spheres) - 15} 个球")
    
    container_r = 3 * r
    density = calculate_packing_density(spheres, container_r)
    print(f"\n填充密度 (容器半径={container_r:.2f}): {density:.6f} ({density*100:.2f}%)")
    
    print(f"\n估计饱和堆积密度 (渐进极限)...")
    print("  (理论极限值约为 0.715 - 0.74)")
    
    best_density, results = estimate_saturation_density(r, 
                                                        max_iters=[4, 6],
                                                        min_radii=[0.01, 0.001])
    
    print(f"\n密度测试结果:")
    for iter_count, min_r, dens, count in results:
        print(f"  迭代={iter_count}, min_r={min_r:.0e}: 密度={dens:.6f}, 球数={count}")
    
    print(f"\n最佳估计密度: {best_density:.6f} ({best_density*100:.2f}%)")
    print(f"\n注意: 实际饱和堆积密度约为 0.715 - 0.74")
    print(f"      随迭代次数增加和最小半径减小，密度将趋近于此值")
