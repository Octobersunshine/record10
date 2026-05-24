from decimal import Decimal, getcontext
from typing import List, Tuple, Union
import math

PRECISION = 80
getcontext().prec = PRECISION


def sqrt_decimal(d: Decimal) -> Decimal:
    if d < 0:
        if d > Decimal('-1e-50'):
            return Decimal('0')
        raise ValueError(f"Cannot compute sqrt of negative number: {d}")
    return d.sqrt()


class VectorDecimal:
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            coords = args[0]
        else:
            coords = args
        self.coords = [Decimal(str(c)) if not isinstance(c, Decimal) else c for c in coords]
    
    @property
    def dim(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        return self.coords[idx]
    
    def __add__(self, other):
        if isinstance(other, VectorDecimal):
            return VectorDecimal([a + b for a, b in zip(self.coords, other.coords)])
        s = Decimal(str(other)) if not isinstance(other, Decimal) else other
        return VectorDecimal([c + s for c in self.coords])
    
    def __sub__(self, other):
        if isinstance(other, VectorDecimal):
            return VectorDecimal([a - b for a, b in zip(self.coords, other.coords)])
        s = Decimal(str(other)) if not isinstance(other, Decimal) else other
        return VectorDecimal([c - s for c in self.coords])
    
    def __mul__(self, scalar):
        s = Decimal(str(scalar)) if not isinstance(scalar, Decimal) else scalar
        return VectorDecimal([c * s for c in self.coords])
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar):
        s = Decimal(str(scalar)) if not isinstance(scalar, Decimal) else scalar
        return VectorDecimal([c / s for c in self.coords])
    
    def dot(self, other) -> Decimal:
        return sum(a * b for a, b in zip(self.coords, other.coords))
    
    def norm_sq(self) -> Decimal:
        return self.dot(self)
    
    def norm(self) -> Decimal:
        return sqrt_decimal(self.norm_sq())
    
    def to_float(self) -> tuple:
        return tuple(float(c) for c in self.coords)
    
    def __repr__(self):
        return f"Vector{self.coords}"


Sphere2D = Tuple[float, float, float]
Sphere3D = Tuple[float, float, float, float]


def solve_kn(curvatures: List[Decimal], n: int) -> Tuple[Decimal, Decimal]:
    d = n - 1
    k_sum = sum(curvatures)
    cross = Decimal('0')
    for i in range(len(curvatures)):
        for j in range(i + 1, len(curvatures)):
            cross += curvatures[i] * curvatures[j]
    if cross < 0:
        cross = max(cross, Decimal('0'))
    k_prod = Decimal('2') * sqrt_decimal(cross)
    return k_sum + k_prod, k_sum - k_prod


def solve_zn(centers: List[VectorDecimal], curvatures: List[Decimal], k_new: Decimal) -> Tuple[VectorDecimal, VectorDecimal]:
    n = len(centers)
    b_vectors = [c * k for c, k in zip(centers, curvatures)]
    b_sum = VectorDecimal([Decimal('0')] * centers[0].dim)
    for b in b_vectors:
        b_sum = b_sum + b
    
    d = centers[0].dim
    cross_sq_vectors = []
    for idx in range(d):
        val = Decimal('0')
        for i in range(n):
            for j in range(i + 1, n):
                val += b_vectors[i][idx] * b_vectors[j][idx]
        cross_sq_vectors.append(val)
    
    cross_sq_norm = sum(v * v for v in cross_sq_vectors)
    
    if cross_sq_norm < Decimal('1e-50'):
        sqrt_vec = VectorDecimal([Decimal('0')] * d)
    else:
        scale = sqrt_decimal(cross_sq_norm)
        avg_b = VectorDecimal([Decimal('0')] * d)
        for b in b_vectors:
            avg_b = avg_b + b
        avg_b = avg_b / Decimal(str(n))
        norm_b = avg_b.norm()
        
        if norm_b > Decimal('1e-30'):
            sqrt_vec = avg_b / norm_b * scale
        else:
            sqrt_vec = VectorDecimal([scale] + [Decimal('0')] * (d - 1))
    
    return (b_sum + sqrt_vec * Decimal('2')) / k_new, (b_sum - sqrt_vec * Decimal('2')) / k_new


def sphere_tangent_nd(s1: tuple, s2: tuple, tol: float = 1e-7) -> bool:
    d = len(s1) - 1
    d_sq = sum((s1[i] - s2[i])**2 for i in range(d))
    r1, r2 = s1[-1], s2[-1]
    sum_r = r1 + r2
    diff_r = abs(r1 - r2)
    return abs(d_sq - sum_r * sum_r) < tol or abs(d_sq - diff_r * diff_r) < tol


def sphere_key_nd(s: tuple) -> tuple:
    scale = 1e9
    return tuple(int(c * scale) for c in s)


def apollonian_packing_2d(c1: Sphere2D, c2: Sphere2D, c3: Sphere2D,
                           max_iter: int = 10, min_radius: float = 1e-6) -> List[Sphere2D]:
    circles = [c1, c2, c3]
    seen = {sphere_key_nd(c1), sphere_key_nd(c2), sphere_key_nd(c3)}
    
    k1 = Decimal(1) / Decimal(str(c1[2]))
    k2 = Decimal(1) / Decimal(str(c2[2]))
    k3 = Decimal(1) / Decimal(str(c3[2]))
    
    z1 = VectorDecimal(c1[0], c1[1])
    z2 = VectorDecimal(c2[0], c2[1])
    z3 = VectorDecimal(c3[0], c3[1])
    
    try:
        k4_plus, k4_minus = solve_kn([k1, k2, k3], 3)
    except ValueError:
        return circles
    
    def get_candidates(k_val, centers, ks):
        if abs(k_val) < Decimal('1e-50'):
            return []
        try:
            za, zb = solve_zn(centers, ks, k_val)
            r_val = float(Decimal(1) / abs(k_val))
            if r_val < min_radius:
                return []
            ca = za.to_float()
            cb = zb.to_float()
            return [(*ca, r_val), (*cb, r_val)]
        except:
            return []
    
    candidates = []
    for k in [k4_plus, k4_minus]:
        candidates.extend(get_candidates(k, [z1, z2, z3], [k1, k2, k3]))
    
    inner = []
    for cand in candidates:
        if (sphere_tangent_nd(cand, c1, 1e-5) and 
            sphere_tangent_nd(cand, c2, 1e-5) and 
            sphere_tangent_nd(cand, c3, 1e-5)):
            inner.append(cand)
    
    triple_queue = []
    if inner:
        c4 = min(inner, key=lambda x: x[2])
        k4 = sphere_key_nd(c4)
        if k4 not in seen:
            circles.append(c4)
            seen.add(k4)
            z4 = VectorDecimal(c4[0], c4[1])
            k4_val = Decimal(1) / Decimal(str(c4[2]))
            triple_queue = [
                ((z1, k1, c1), (z2, k2, c2), (z4, k4_val, c4)),
                ((z1, k1, c1), (z3, k3, c3), (z4, k4_val, c4)),
                ((z2, k2, c2), (z3, k3, c3), (z4, k4_val, c4)),
            ]
    
    seen_triples = set()
    
    for _ in range(max_iter):
        new_triples = []
        for tri in triple_queue:
            tri_key = tuple(sorted(sphere_key_nd(t[2]) for t in tri))
            if tri_key in seen_triples:
                continue
            seen_triples.add(tri_key)
            
            (za, ka, ca), (zb, kb, cb), (zc, kc, cc) = tri
            
            try:
                k_new_plus, k_new_minus = solve_kn([ka, kb, kc], 3)
            except:
                continue
            
            for k_new in [k_new_plus, k_new_minus]:
                if abs(k_new) < Decimal('1e-50'):
                    continue
                try:
                    r_new = float(Decimal(1) / abs(k_new))
                except:
                    continue
                if r_new < min_radius:
                    continue
                
                try:
                    z_new1, z_new2 = solve_zn([za, zb, zc], [ka, kb, kc], k_new)
                except:
                    continue
                
                for z_new in [z_new1, z_new2]:
                    coords = z_new.to_float()
                    new_circle = (*coords, r_new)
                    nk = sphere_key_nd(new_circle)
                    
                    if nk in seen:
                        continue
                    
                    if (sphere_tangent_nd(new_circle, ca, 1e-5) and
                        sphere_tangent_nd(new_circle, cb, 1e-5) and
                        sphere_tangent_nd(new_circle, cc, 1e-5)):
                        circles.append(new_circle)
                        seen.add(nk)
                        new_data = (z_new, k_new, new_circle)
                        new_triples.append(((za, ka, ca), (zb, kb, cb), new_data))
                        new_triples.append(((za, ka, ca), (zc, kc, cc), new_data))
                        new_triples.append(((zb, kb, cb), (zc, kc, cc), new_data))
        
        triple_queue = new_triples
        if not triple_queue:
            break
    
    return circles


def apollonian_sphere_packing_3d(s1: Sphere3D, s2: Sphere3D, s3: Sphere3D, s4: Sphere3D,
                                  max_iter: int = 10, min_radius: float = 1e-6) -> List[Sphere3D]:
    spheres = [s1, s2, s3, s4]
    seen = {sphere_key_nd(s1), sphere_key_nd(s2), sphere_key_nd(s3), sphere_key_nd(s4)}
    
    k1 = Decimal(1) / Decimal(str(s1[3]))
    k2 = Decimal(1) / Decimal(str(s2[3]))
    k3 = Decimal(1) / Decimal(str(s3[3]))
    k4 = Decimal(1) / Decimal(str(s4[3]))
    
    z1 = VectorDecimal(s1[0], s1[1], s1[2])
    z2 = VectorDecimal(s2[0], s2[1], s2[2])
    z3 = VectorDecimal(s3[0], s3[1], s3[2])
    z4 = VectorDecimal(s4[0], s4[1], s4[2])
    
    try:
        k5_plus, k5_minus = solve_kn([k1, k2, k3, k4], 4)
    except ValueError:
        return spheres
    
    def get_candidates(k_val, centers, ks):
        if abs(k_val) < Decimal('1e-50'):
            return []
        try:
            za, zb = solve_zn(centers, ks, k_val)
            r_val = float(Decimal(1) / abs(k_val))
            if r_val < min_radius:
                return []
            ca = za.to_float()
            cb = zb.to_float()
            return [(*ca, r_val), (*cb, r_val)]
        except:
            return []
    
    candidates = []
    for k in [k5_plus, k5_minus]:
        candidates.extend(get_candidates(k, [z1, z2, z3, z4], [k1, k2, k3, k4]))
    
    inner = []
    for cand in candidates:
        if (sphere_tangent_nd(cand, s1, 1e-5) and 
            sphere_tangent_nd(cand, s2, 1e-5) and 
            sphere_tangent_nd(cand, s3, 1e-5) and
            sphere_tangent_nd(cand, s4, 1e-5)):
            inner.append(cand)
    
    quadruple_queue = []
    if inner:
        s5 = min(inner, key=lambda x: x[3])
        k5 = sphere_key_nd(s5)
        if k5 not in seen:
            spheres.append(s5)
            seen.add(k5)
            z5 = VectorDecimal(s5[0], s5[1], s5[2])
            k5_val = Decimal(1) / Decimal(str(s5[3]))
            data = [(z1, k1, s1), (z2, k2, s2), (z3, k3, s3), (z4, k4, s4), (z5, k5_val, s5)]
            quadruple_queue = [
                (data[0], data[1], data[2], data[4]),
                (data[0], data[1], data[3], data[4]),
                (data[0], data[2], data[3], data[4]),
                (data[1], data[2], data[3], data[4]),
            ]
    
    seen_quadruples = set()
    
    for _ in range(max_iter):
        new_quadruples = []
        for quad in quadruple_queue:
            quad_key = tuple(sorted(sphere_key_nd(q[2]) for q in quad))
            if quad_key in seen_quadruples:
                continue
            seen_quadruples.add(quad_key)
            
            (za, ka, ca), (zb, kb, cb), (zc, kc, cc), (zd, kd, cd) = quad
            
            try:
                k_new_plus, k_new_minus = solve_kn([ka, kb, kc, kd], 4)
            except:
                continue
            
            for k_new in [k_new_plus, k_new_minus]:
                if abs(k_new) < Decimal('1e-50'):
                    continue
                try:
                    r_new = float(Decimal(1) / abs(k_new))
                except:
                    continue
                if r_new < min_radius:
                    continue
                
                try:
                    z_new1, z_new2 = solve_zn([za, zb, zc, zd], [ka, kb, kc, kd], k_new)
                except:
                    continue
                
                for z_new in [z_new1, z_new2]:
                    coords = z_new.to_float()
                    new_sphere = (*coords, r_new)
                    nk = sphere_key_nd(new_sphere)
                    
                    if nk in seen:
                        continue
                    
                    if (sphere_tangent_nd(new_sphere, ca, 1e-5) and
                        sphere_tangent_nd(new_sphere, cb, 1e-5) and
                        sphere_tangent_nd(new_sphere, cc, 1e-5) and
                        sphere_tangent_nd(new_sphere, cd, 1e-5)):
                        spheres.append(new_sphere)
                        seen.add(nk)
                        new_data = (z_new, k_new, new_sphere)
                        new_quadruples.append(((za, ka, ca), (zb, kb, cb), (zc, kc, cc), new_data))
                        new_quadruples.append(((za, ka, ca), (zb, kb, cb), (zd, kd, cd), new_data))
                        new_quadruples.append(((za, ka, ca), (zc, kc, cc), (zd, kd, cd), new_data))
                        new_quadruples.append(((zb, kb, cb), (zc, kc, cc), (zd, kd, cd), new_data))
        
        quadruple_queue = new_quadruples
        if not quadruple_queue:
            break
    
    return spheres


def calculate_packing_density(spheres: List, container_radius: float, dim: int = 3) -> float:
    if dim == 2:
        def volume(r):
            return math.pi * r * r
    else:
        def volume(r):
            return (4.0 / 3.0) * math.pi * (r ** 3)
    
    total = sum(volume(s[-1]) for s in spheres)
    container = volume(container_radius)
    return total / container


def estimate_saturation_density_3d(initial_radius: float = 1.0,
                                max_iters: List[int] = [4, 6, 8],
                                min_radii: List[float] = [1e-2, 1e-3, 1e-4]) -> Tuple[float, List]:
    r = initial_radius
    s1 = (0, 0, 0, r)
    s2 = (2*r, 0, 0, r)
    s3 = (r, math.sqrt(3)*r, 0, r)
    s4 = (r, math.sqrt(3)*r/3, 2*math.sqrt(6)*r/3, r)
    
    results = []
    for max_iter in max_iters:
        for min_r in min_radii:
            spheres = apollonian_sphere_packing_3d(s1, s2, s3, s4,
                                            max_iter=max_iter,
                                            min_radius=min_r)
            outer_r = 3 * r
            density = calculate_packing_density(spheres, outer_r, 3)
            results.append((max_iter, min_r, density, len(spheres)))
    
    if results:
        results.sort(key=lambda x: (-x[0], x[1]))
        best = results[0][2]
    else:
        best = 0.0
    
    return best, results


def create_tetrahedral_spheres(radius: float = 1.0) -> Tuple[Sphere3D, Sphere3D, Sphere3D, Sphere3D]:
    r = radius
    s1 = (0.0, 0.0, 0.0, r)
    s2 = (2.0 * r, 0.0, 0.0, r)
    s3 = (r, math.sqrt(3) * r, 0.0, r)
    s4 = (r, math.sqrt(3) * r / 3.0, 2.0 * math.sqrt(6) * r / 3.0, r)
    return s1, s2, s3, s4


def create_equilateral_circles(radius: float = 1.0) -> Tuple[Sphere2D, Sphere2D, Sphere2D]:
    r = radius
    c1 = (0.0, 0.0, r)
    c2 = (2.0 * r, 0.0, r)
    c3 = (r, math.sqrt(3) * r, r)
    return c1, c2, c3


if __name__ == "__main__":
    print("=" * 70)
    print("阿波罗尼斯填充 - 统一模块 (2D圆填充 & 3D球填充")
    print("=" * 70)
    
    print("\n" + "-" * 50)
    print("2D 阿波罗尼斯圆填充")
    print("-" * 50)
    
    c1, c2, c3 = create_equilateral_circles(1.0)
    circles = apollonian_packing_2d(c1, c2, c3, max_iter=6, min_radius=0.001)
    print(f"生成了 {len(circles)} 个圆")
    density_2d = calculate_packing_density(circles, 3.0, 2)
    print(f"填充密度: {density_2d:.6f} ({density_2d*100:.2f}%)")
    
    print("\n" + "-" * 50)
    print("3D 阿波罗尼斯球填充")
    print("-" * 50)
    
    s1, s2, s3, s4 = create_tetrahedral_spheres(1.0)
    spheres = apollonian_sphere_packing_3d(s1, s2, s3, s4, max_iter=5, min_radius=0.005)
    print(f"生成了 {len(spheres)} 个球")
    density_3d = calculate_packing_density(spheres, 3.0, 3)
    print(f"填充密度: {density_3d:.6f} ({density_3d*100:.2f}%)")
    
    print("\n" + "-" * 50)
    print("3D 饱和堆积密度估计")
    print("-" * 50)
    print("理论极限值约为: 0.715 - 0.74 (三维阿波罗尼斯球填充")
    
    best_density, results = estimate_saturation_density_3d(1.0,
                                                    max_iters=[4, 5],
                                                    min_radii=[0.01, 0.005])
    
    print("\n测试结果:")
    for iter_count, min_r, dens, count in results:
        print(f"  迭代={iter_count}, min_r={min_r:.0e}: 密度={dens:.6f}, 球数={count}")
    
    print(f"\n最佳估计密度: {best_density:.6f} ({best_density*100:.2f}%)")
