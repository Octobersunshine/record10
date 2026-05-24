from decimal import Decimal, getcontext
from typing import List, Tuple, Optional
import math

PRECISION = 80
getcontext().prec = PRECISION


class ComplexDecimal:
    def __init__(self, real: Decimal, imag: Decimal = Decimal('0')):
        self.real = Decimal(str(real)) if not isinstance(real, Decimal) else real
        self.imag = Decimal(str(imag)) if not isinstance(imag, Decimal) else imag
    
    def __add__(self, other):
        if isinstance(other, ComplexDecimal):
            return ComplexDecimal(self.real + other.real, self.imag + other.imag)
        return ComplexDecimal(self.real + Decimal(str(other)), self.imag)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, ComplexDecimal):
            return ComplexDecimal(self.real - other.real, self.imag - other.imag)
        return ComplexDecimal(self.real - Decimal(str(other)), self.imag)
    
    def __mul__(self, other):
        if isinstance(other, ComplexDecimal):
            r = self.real * other.real - self.imag * other.imag
            i = self.real * other.imag + self.imag * other.real
            return ComplexDecimal(r, i)
        other_dec = Decimal(str(other))
        return ComplexDecimal(self.real * other_dec, self.imag * other_dec)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, ComplexDecimal):
            denom = other.real * other.real + other.imag * other.imag
            r = (self.real * other.real + self.imag * other.imag) / denom
            i = (self.imag * other.real - self.real * other.imag) / denom
            return ComplexDecimal(r, i)
        other_dec = Decimal(str(other))
        return ComplexDecimal(self.real / other_dec, self.imag / other_dec)
    
    def abs_sq(self) -> Decimal:
        return self.real * self.real + self.imag * self.imag
    
    def __abs__(self) -> Decimal:
        return sqrt_decimal(self.abs_sq())
    
    def sqrt(self):
        r = abs(self)
        if r == 0:
            return ComplexDecimal(0, 0)
        theta = Decimal(str(math.atan2(float(self.imag), float(self.real))))
        half_theta = theta / Decimal('2')
        sqrt_r = sqrt_decimal(r)
        return ComplexDecimal(
            sqrt_r * Decimal(str(math.cos(float(half_theta)))),
            sqrt_r * Decimal(str(math.sin(float(half_theta))))
        )
    
    def to_tuple(self) -> Tuple[float, float]:
        return float(self.real), float(self.imag)
    
    def __repr__(self):
        return f"({self.real}+{self.imag}j)"


def sqrt_decimal(d: Decimal) -> Decimal:
    if d < 0:
        if d > Decimal('-1e-50'):
            return Decimal('0')
        raise ValueError(f"Cannot compute sqrt of negative number: {d}")
    return d.sqrt()


def solve_k4(k1: Decimal, k2: Decimal, k3: Decimal) -> Tuple[Decimal, Decimal]:
    k_sum = k1 + k2 + k3
    cross = k1 * k2 + k2 * k3 + k3 * k1
    if cross < 0:
        cross = max(cross, Decimal('0'))
    k_prod = Decimal('2') * sqrt_decimal(cross)
    return k_sum + k_prod, k_sum - k_prod


def solve_z4_decimal(z1: ComplexDecimal, z2: ComplexDecimal, z3: ComplexDecimal,
                     k1: Decimal, k2: Decimal, k3: Decimal, k4: Decimal) -> Tuple[ComplexDecimal, ComplexDecimal]:
    zk1 = z1 * k1
    zk2 = z2 * k2
    zk3 = z3 * k3
    sum_zk = zk1 + zk2 + zk3
    cross_term_sq = zk1 * zk2 + zk2 * zk3 + zk3 * zk1
    
    if cross_term_sq.abs_sq() < Decimal('1e-50'):
        cross_term = ComplexDecimal(0, 0)
    else:
        cross_term = cross_term_sq.sqrt()
    
    return (sum_zk + Decimal('2') * cross_term) / k4, (sum_zk - Decimal('2') * cross_term) / k4


def circle_tangent(c1: Tuple[float, float, float], 
                   c2: Tuple[float, float, float], 
                   tol: float = 1e-8) -> bool:
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    d_sq = dx * dx + dy * dy
    sum_r = c1[2] + c2[2]
    diff_r = abs(c1[2] - c2[2])
    return abs(d_sq - sum_r * sum_r) < tol or abs(d_sq - diff_r * diff_r) < tol


def circle_key(c: Tuple[float, float, float]) -> Tuple[int, int, int]:
    return (int(c[0] * 1e10), int(c[1] * 1e10), int(c[2] * 1e10))


def apollonian_gasket(c1: Tuple[float, float, float],
                      c2: Tuple[float, float, float],
                      c3: Tuple[float, float, float],
                      max_iter: int = 20,
                      min_radius: float = 1e-8) -> List[Tuple[float, float, float]]:
    circles = [c1, c2, c3]
    seen_circles = {circle_key(c1), circle_key(c2), circle_key(c3)}
    
    k1 = Decimal(1) / Decimal(str(c1[2]))
    k2 = Decimal(1) / Decimal(str(c2[2]))
    k3 = Decimal(1) / Decimal(str(c3[2]))
    
    z1 = ComplexDecimal(Decimal(str(c1[0])), Decimal(str(c1[1])))
    z2 = ComplexDecimal(Decimal(str(c2[0])), Decimal(str(c2[1])))
    z3 = ComplexDecimal(Decimal(str(c3[0])), Decimal(str(c3[1])))
    
    try:
        k4_plus, k4_minus = solve_k4(k1, k2, k3)
    except ValueError:
        return circles
    
    def get_circle_candidates(k_val: Decimal, z_a: ComplexDecimal, z_b: ComplexDecimal, z_c: ComplexDecimal,
                              ka: Decimal, kb: Decimal, kc: Decimal) -> List[Tuple[float, float, float]]:
        if abs(k_val) < Decimal('1e-50'):
            return []
        try:
            z4a, z4b = solve_z4_decimal(z_a, z_b, z_c, ka, kb, kc, k_val)
            r_val = float(Decimal(1) / abs(k_val))
            if r_val < min_radius:
                return []
            x1, y1 = z4a.to_tuple()
            x2, y2 = z4b.to_tuple()
            return [(x1, y1, r_val), (x2, y2, r_val)]
        except:
            return []
    
    candidates = []
    for k in [k4_plus, k4_minus]:
        candidates.extend(get_circle_candidates(k, z1, z2, z3, k1, k2, k3))
    
    inner_circles = []
    for cx, cy, cr in candidates:
        test_circle = (cx, cy, cr)
        if (circle_tangent(test_circle, c1, tol=1e-6) and 
            circle_tangent(test_circle, c2, tol=1e-6) and 
            circle_tangent(test_circle, c3, tol=1e-6)):
            inner_circles.append(test_circle)
    
    c4 = None
    if inner_circles:
        c4 = min(inner_circles, key=lambda x: x[2])
        k4 = circle_key(c4)
        if k4 not in seen_circles:
            circles.append(c4)
            seen_circles.add(k4)
    
    class CircleData:
        def __init__(self, c: Tuple[float, float, float], z: ComplexDecimal, k: Decimal):
            self.circle = c
            self.z = z
            self.k = k
    
    cd1 = CircleData(c1, z1, k1)
    cd2 = CircleData(c2, z2, k2)
    cd3 = CircleData(c3, z3, k3)
    cd4 = CircleData(c4, ComplexDecimal(Decimal(str(c4[0])), Decimal(str(c4[1]))), 
                     Decimal(1) / Decimal(str(c4[2]))) if c4 else None
    
    triple_queue = []
    if cd4:
        triple_queue.append((cd1, cd2, cd4))
        triple_queue.append((cd1, cd3, cd4))
        triple_queue.append((cd2, cd3, cd4))
    
    seen_triples = set()
    
    for iteration in range(max_iter):
        new_triples = []
        
        for tri in triple_queue:
            tri_circles = tuple(sorted([circle_key(t.circle) for t in tri]))
            if tri_circles in seen_triples:
                continue
            seen_triples.add(tri_circles)
            
            ca, cb, cc = tri
            
            try:
                k_new_plus, k_new_minus = solve_k4(ca.k, cb.k, cc.k)
            except (ValueError, ZeroDivisionError):
                continue
            
            for k_new in [k_new_plus, k_new_minus]:
                if abs(k_new) < Decimal('1e-50'):
                    continue
                
                try:
                    r_new = float(Decimal(1) / abs(k_new))
                except ZeroDivisionError:
                    continue
                
                if r_new < min_radius:
                    continue
                
                try:
                    z_new1, z_new2 = solve_z4_decimal(ca.z, cb.z, cc.z, ca.k, cb.k, cc.k, k_new)
                except:
                    continue
                
                for z_new in [z_new1, z_new2]:
                    x, y = z_new.to_tuple()
                    new_circle = (x, y, r_new)
                    ck = circle_key(new_circle)
                    
                    if ck in seen_circles:
                        continue
                    
                    if (circle_tangent(new_circle, ca.circle, tol=1e-6) and
                        circle_tangent(new_circle, cb.circle, tol=1e-6) and
                        circle_tangent(new_circle, cc.circle, tol=1e-6)):
                        circles.append(new_circle)
                        seen_circles.add(ck)
                        new_cd = CircleData(new_circle, z_new, k_new)
                        new_triples.append((ca, cb, new_cd))
                        new_triples.append((ca, cc, new_cd))
                        new_triples.append((cb, cc, new_cd))
        
        triple_queue = new_triples
        if not triple_queue:
            break
    
    return circles


if __name__ == "__main__":
    r = 1.0
    c1 = (0.0, 0.0, r)
    c2 = (2.0 * r, 0.0, r)
    c3 = (r, math.sqrt(3) * r, r)
    
    circles = apollonian_gasket(c1, c2, c3, max_iter=10, min_radius=1e-6)
    
    print(f"生成了 {len(circles)} 个圆:")
    for i, (x, y, rad) in enumerate(circles[:20]):
        print(f"  圆 {i+1:3d}: 圆心=({x:14.10f}, {y:14.10f}), 半径={rad:14.10f}")
    
    if len(circles) > 20:
        print(f"  ... 还有 {len(circles) - 20} 个圆")
    
    print(f"\n精度设置: {PRECISION} 位十进制数")
    print(f"迭代次数: 10, 最小半径: 1e-6")
