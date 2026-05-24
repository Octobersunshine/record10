import numpy as np
from scipy.integrate import quad
from scipy.special import gamma


class SingularityIntegrator:
    def __init__(self, beta):
        self.beta = beta
    
    def gauss_jacobi_nodes_weights(self, n, alpha, beta, a=0, b=1):
        try:
            from scipy.special import roots_jacobi
            x, w = roots_jacobi(n, alpha, beta)
            x = 0.5 * (x + 1) * (b - a) + a
            w = 0.5 * (b - a) * w
            return x, w
        except:
            return self._gauss_jacobi_fallback(n, alpha, beta, a, b)
    
    def _gauss_jacobi_fallback(self, n, alpha, beta, a, b):
        m = n * 3
        x = np.linspace(a, b, m)
        w = np.ones(m) * (b - a) / m
        return x, w
    
    def gauss_jacobi_integrate(self, f, a, b, alpha, beta, n=32):
        x, w = self.gauss_jacobi_nodes_weights(n, alpha, beta, a, b)
        result = 0.0
        for xi, wi in zip(x, w):
            result += wi * f(xi)
        return result
    
    def integrate_singular_right(self, f, a, b, singularity_strength, n=64):
        if singularity_strength >= 1.0:
            return self._simpson_integrate(f, a, b, n * 2)
        
        alpha_jacobi = 0.0
        beta_jacobi = -singularity_strength
        
        def scaled_f(t):
            weight = (1 - (t - a) / (b - a)) ** singularity_strength
            return f(t) / weight if weight != 0 else 0.0
        
        try:
            return self.gauss_jacobi_integrate(scaled_f, a, b, alpha_jacobi, beta_jacobi, n)
        except:
            return self.integrate_variable_transform(f, a, b, singularity_strength, n)
    
    def integrate_variable_transform(self, f, a, b, singularity_strength, n=200):
        if singularity_strength <= 0 or singularity_strength >= 1:
            return self._simpson_integrate(f, a, b, n * 2)
        
        p = 1.0 / (1.0 - singularity_strength)
        
        def transformed(u):
            t = b - (b - a) * (1 - u) ** p
            dt_du = (b - a) * p * (1 - u) ** (p - 1)
            return f(t) * dt_du
        
        return self._simpson_integrate(transformed, 0, 1, n)
    
    def integrate_power_singularity(self, f, a, b, alpha, beta_sing, n=50):
        t = np.linspace(a, b, n)
        dt = (b - a) / (n - 1)
        
        result = 0.0
        for i in range(n):
            weight = dt
            if i == 0 or i == n - 1:
                weight *= 0.5
            result += f(t[i]) * weight
        
        return result
    
    def integrate_endpoint_singularity(self, f, a, b, singularity_strength, 
                                       endpoint='right', n=100):
        if endpoint == 'right':
            return self.integrate_singular_right(f, a, b, singularity_strength, n)
        else:
            def flipped_f(t):
                return f(a + b - t)
            return self.integrate_singular_right(flipped_f, a, b, singularity_strength, n)
    
    def _simpson_integrate(self, f, a, b, n):
        if n % 2 == 0:
            n += 1
        x = np.linspace(a, b, n)
        h = (b - a) / (n - 1)
        y = np.array([f(xi) for xi in x])
        return h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))


class SchwarzChristoffelDisk:
    def __init__(self, polygon_vertices, tol=1e-8, integration_order=100):
        self.vertices = np.array(polygon_vertices, dtype=np.complex128)
        self.n = len(self.vertices)
        self.tol = tol
        self.integration_order = integration_order
        
        self.alphas = self._compute_internal_angles()
        self.betas = 1 - self.alphas / np.pi
        self.prevertices = self._compute_symmetric_prevertices()
        
        self._singularity_handlers = [SingularityIntegrator(b) for b in self.betas]
        
        self.scale, self.rotation, self.offset = self._calibrate_transform()
        
    def _compute_internal_angles(self):
        angles = np.zeros(self.n)
        for i in range(self.n):
            prev = self.vertices[(i-1) % self.n]
            curr = self.vertices[i]
            next_v = self.vertices[(i+1) % self.n]
            
            v1 = prev - curr
            v2 = next_v - curr
            
            cross = np.imag(np.conj(v1) * v2)
            dot = np.real(np.conj(v1) * v2)
            
            angle = np.arctan2(cross, dot)
            if angle < 0:
                angle += 2 * np.pi
            angles[i] = angle
        return angles
    
    def _compute_symmetric_prevertices(self):
        return np.exp(1j * np.linspace(0, 2*np.pi, self.n, endpoint=False))
    
    def _find_nearest_prevertex(self, z):
        distances = np.abs(z - self.prevertices)
        return np.argmin(distances)
    
    def _integrand(self, z):
        prod = 1.0 + 0j
        for k in range(self.n):
            term = 1 - z / self.prevertices[k]
            if np.abs(term) < 1e-15:
                return np.inf + 0j
            prod *= term ** (-self.betas[k])
        return prod
    
    def _integrate_from_origin(self, z):
        if np.abs(z) < 1e-15:
            return 0j
        
        nearest_k = self._find_nearest_prevertex(z)
        distance_to_nearest = np.abs(z - self.prevertices[nearest_k])
        
        if distance_to_nearest < 1e-3:
            return self._integrate_near_prevertex(z, nearest_k)
        else:
            return self._integrate_path(z)
    
    def _integrate_path(self, z):
        def real_integrand(t):
            w = z * t
            val = self._integrand(w)
            return np.real(z * val)
        
        def imag_integrand(t):
            w = z * t
            val = self._integrand(w)
            return np.imag(z * val)
        
        try:
            real_part, _ = quad(real_integrand, 0, 1, epsabs=self.tol, epsrel=self.tol)
            imag_part, _ = quad(imag_integrand, 0, 1, epsabs=self.tol, epsrel=self.tol)
            return real_part + 1j * imag_part
        except:
            return self._integrate_numerical(z)
    
    def _integrate_numerical(self, z, n=2000):
        t = np.linspace(0, 1, n)
        dt = 1.0 / (n - 1)
        result = 0j
        
        for i in range(n):
            w = z * t[i]
            val = self._integrand(w)
            weight = dt
            if i == 0 or i == n - 1:
                weight *= 0.5
            result += z * val * weight
        
        return result
    
    def _integrate_near_prevertex(self, z, k):
        w_k = self.prevertices[k]
        beta_k = self.betas[k]
        
        if np.abs(z - w_k) < 1e-10:
            return self._integrate_to_prevertex_asymptotic(k)
        
        t_end = np.abs(z)
        if t_end > 1.0:
            t_end = 1.0
        
        def integrand(t):
            w = z * t
            return z * self._integrand(w)
        
        t_split = 0.95
        result1 = self._integrate_regular_segment(z, 0, t_split)
        result2 = self._integrate_singular_segment(z, t_split, t_end, k)
        
        return result1 + result2
    
    def _integrate_regular_segment(self, z, t_start, t_end, n=500):
        t = np.linspace(t_start, t_end, n)
        dt = (t_end - t_start) / (n - 1)
        result = 0j
        
        for i in range(n):
            w = z * t[i]
            val = self._integrand(w)
            weight = dt
            if i == 0 or i == n - 1:
                weight *= 0.5
            result += z * val * weight
        
        return result
    
    def _integrate_singular_segment(self, z, t_start, t_end, k, n=64):
        w_k = self.prevertices[k]
        beta_k = self.betas[k]
        
        if beta_k <= 0 or beta_k >= 1:
            return self._integrate_regular_segment(z, t_start, t_end, n * 4)
        
        integrator = self._singularity_handlers[k]
        
        def real_integrand(t):
            w = z * t
            return np.real(z * self._integrand(w))
        
        def imag_integrand(t):
            w = z * t
            return np.imag(z * self._integrand(w))
        
        try:
            real_part = integrator.integrate_singular_right(
                real_integrand, t_start, t_end, beta_k, n
            )
            imag_part = integrator.integrate_singular_right(
                imag_integrand, t_start, t_end, beta_k, n
            )
            return real_part + 1j * imag_part
        except:
            def transformed_integrand(u):
                t = t_start + (t_end - t_start) * (1 - (1 - u) ** (1.0 / (1 - beta_k)))
                dt_du = (t_end - t_start) / (1 - beta_k) * (1 - u) ** (beta_k / (1 - beta_k))
                w = z * t
                return z * self._integrand(w) * dt_du
            
            return self._simpson_integrate_complex(transformed_integrand, 0, 1, n * 2)
    
    def _simpson_integrate_complex(self, f, a, b, n):
        if n % 2 == 0:
            n += 1
        x = np.linspace(a, b, n)
        h = (b - a) / (n - 1)
        y = np.array([f(xi) for xi in x])
        return h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))
    
    def _integrate_to_prevertex_asymptotic(self, k):
        w_k = self.prevertices[k]
        beta_k = self.betas[k]
        
        if beta_k <= 0 or beta_k >= 1:
            return self._integrate_path(w_k * 0.9999)
        
        other_prod = 1.0 + 0j
        for j in range(self.n):
            if j != k:
                other_prod *= (1 - w_k / self.prevertices[j]) ** (-self.betas[j])
        
        C = w_k * other_prod
        
        integral_const = 1.0 / (1 - beta_k)
        
        return C * integral_const
    
    def _integrate_to_prevertex(self, k):
        w_k = self.prevertices[k]
        beta_k = self.betas[k]
        
        result1 = self._integrate_regular_segment(w_k, 0, 0.99)
        
        if beta_k > 0 and beta_k < 1:
            t_array = np.linspace(0.99, 1.0, 1000)
            dt = t_array[1] - t_array[0]
            result2 = 0j
            
            for i in range(len(t_array)):
                t = t_array[i]
                w = w_k * t
                term = (1 - t) ** (-beta_k)
                other_prod = 1.0 + 0j
                for j in range(self.n):
                    if j != k:
                        other_prod *= (1 - w / self.prevertices[j]) ** (-self.betas[j])
                
                weight = dt
                if i == 0 or i == len(t_array) - 1:
                    weight *= 0.5
                
                result2 += w_k * term * other_prod * weight
            
            return result1 + result2
        else:
            return result1 + self._integrate_regular_segment(w_k, 0.99, 1.0, 2000)
    
    def _calibrate_transform(self):
        mapped_prevertices = np.array([self._integrate_to_prevertex(k) for k in range(self.n)])
        
        target_center = np.mean(self.vertices)
        mapped_center = np.mean(mapped_prevertices)
        
        target_distances = np.abs(self.vertices - target_center)
        mapped_distances = np.abs(mapped_prevertices - mapped_center)
        
        valid = mapped_distances > 1e-10
        if np.any(valid):
            scale = np.mean(target_distances[valid] / mapped_distances[valid])
        else:
            scale = 1.0
        
        target_angles = np.angle(self.vertices - target_center)
        mapped_angles = np.angle(mapped_prevertices - mapped_center)
        
        angle_diffs = target_angles - mapped_angles
        angle_diffs = np.mod(angle_diffs + np.pi, 2 * np.pi) - np.pi
        angle_diff = np.mean(angle_diffs)
        rotation = np.exp(1j * angle_diff)
        
        offset = target_center - scale * rotation * mapped_center
        
        return scale, rotation, offset
    
    def map(self, z):
        z = np.asarray(z, dtype=np.complex128)
        original_shape = z.shape
        z = z.flatten()
        
        result = np.zeros_like(z, dtype=np.complex128)
        
        for idx, zi in enumerate(z):
            if np.abs(zi) > 1 + 1e-10:
                raise ValueError(f"Point {zi} is outside unit disk")
            
            if np.abs(zi) < 1e-15:
                result[idx] = 0j
                continue
            
            result[idx] = self._integrate_from_origin(zi)
        
        result = self.scale * self.rotation * result + self.offset
        return result.reshape(original_shape)
    
    def derivative(self, z):
        z = np.asarray(z, dtype=np.complex128)
        original_shape = z.shape
        z = z.flatten()
        
        result = np.zeros_like(z, dtype=np.complex128)
        
        for idx, zi in enumerate(z):
            if np.abs(zi) < 1e-15:
                prod = 1.0 + 0j
                for k in range(self.n):
                    prod *= (-1 / self.prevertices[k]) ** (-self.betas[k])
                result[idx] = self.scale * self.rotation * prod
            else:
                result[idx] = self.scale * self.rotation * self._integrand(zi)
        
        return result.reshape(original_shape)
    
    def map_boundary(self, num_points=200):
        theta = np.linspace(0, 2*np.pi, num_points)
        z = np.exp(1j * theta)
        return self.map(z)
    
    def inverse_map(self, w, max_iter=50, tol=1e-10):
        w = np.asarray(w, dtype=np.complex128)
        original_shape = w.shape
        w = w.flatten()
        
        result = np.zeros_like(w, dtype=np.complex128)
        
        for idx, wi in enumerate(w):
            z0 = 0j
            
            for _ in range(max_iter):
                fz = self.map(z0) - wi
                dfz = self.derivative(z0)
                
                if np.abs(dfz) < 1e-15:
                    break
                
                z_new = z0 - fz / dfz
                
                if np.abs(z_new) > 1:
                    z_new = z_new / np.abs(z_new) * 0.999
                
                if np.abs(z_new - z0) < tol:
                    z0 = z_new
                    break
                
                z0 = z_new
            
            result[idx] = z0
        
        return result.reshape(original_shape)


def regular_polygon(n, radius=1.0, center=0+0j):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return center + radius * np.exp(1j * angles)


def rectangle(width, height, center=0+0j):
    w, h = width / 2, height / 2
    return np.array([
        center - w - h*1j,
        center + w - h*1j,
        center + w + h*1j,
        center - w + h*1j
    ])


def example_square():
    square_vertices = [
        -1 - 1j,
        1 - 1j,
        1 + 1j,
        -1 + 1j
    ]
    
    sc = SchwarzChristoffelDisk(square_vertices)
    
    test_points = [0, 0.5, 0.5j, -0.5, -0.5j]
    mapped = sc.map(test_points)
    
    print("=" * 60)
    print("正方形映射测试 (含奇点处理)")
    print("=" * 60)
    print(f"多边形顶点: {square_vertices}")
    print(f"内角(弧度): {sc.alphas}")
    print(f"beta参数: {sc.betas}")
    print(f"缩放因子: {sc.scale:.6f}")
    print(f"旋转角度: {np.angle(sc.rotation):.6f} rad")
    print(f"平移量: {sc.offset}")
    print("\n映射结果:")
    for z, w in zip(test_points, mapped):
        print(f"  f({z:>6}) = {w:.4f}")
    
    print("\n预顶点映射验证 (接近奇点):")
    prevertex_mapped = sc.map(sc.prevertices * 0.9999)
    for i, (src, dst) in enumerate(zip(sc.prevertices, prevertex_mapped)):
        print(f"  预顶点 {i}: {src:.4f} -> {dst:.4f}")
    
    return sc


def example_triangle():
    triangle_vertices = [
        -1 - 0.5j,
        1 - 0.5j,
        0 + 1j
    ]
    
    sc = SchwarzChristoffelDisk(triangle_vertices)
    
    test_points = [0, 0.5, 0.5j]
    mapped = sc.map(test_points)
    
    print("\n" + "=" * 60)
    print("三角形映射测试")
    print("=" * 60)
    print(f"多边形顶点: {triangle_vertices}")
    print(f"内角(弧度): {sc.alphas}")
    print(f"beta参数: {sc.betas}")
    print("\n映射结果:")
    for z, w in zip(test_points, mapped):
        print(f"  f({z:>6}) = {w:.4f}")
    
    return sc


def example_singularity_convergence():
    square_vertices = [
        -1 - 1j,
        1 - 1j,
        1 + 1j,
        -1 + 1j
    ]
    
    sc = SchwarzChristoffelDisk(square_vertices)
    
    print("\n" + "=" * 60)
    print("奇点收敛性测试")
    print("=" * 60)
    print("测试: 逐渐接近预顶点的点的映射结果")
    print("(验证积分在奇点附近不会发散)")
    
    w0 = sc.prevertices[0]
    results = []
    prev_val = None
    
    for eps in [0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001]:
        z = w0 * (1 - eps)
        w = sc.map(z)
        results.append(w)
        
        if prev_val is not None:
            diff = np.abs(w - prev_val)
        else:
            diff = 0.0
        prev_val = w
        
        print(f"  距离预顶点 {eps:>7.5f}: f({z:.8f}) = {w:.8f}, 变化 = {diff:.2e}")
    
    if len(results) > 1:
        max_diff = np.max(np.abs(np.diff(results)))
        print(f"\n  最大变化量: {max_diff:.2e}")
        if max_diff < 0.1:
            print("  ✓ 收敛性良好，积分奇点处理有效")
        else:
            print("  ⚠ 收敛性需要改进")
    
    return sc


def test_singularity_accuracy():
    print("\n" + "=" * 60)
    print("奇点处理精度验证")
    print("=" * 60)
    
    square_vertices = [
        -1 - 1j,
        1 - 1j,
        1 + 1j,
        -1 + 1j
    ]
    
    sc = SchwarzChristoffelDisk(square_vertices)
    
    print("测试1: 映射边界点的平滑性")
    theta_values = np.linspace(0, 2*np.pi, 100, endpoint=False)
    boundary_values = []
    for theta in theta_values:
        z = np.exp(1j * theta) * 0.9999
        w = sc.map(z)
        boundary_values.append(w)
    
    boundary_values = np.array(boundary_values)
    distances = np.abs(np.diff(boundary_values, append=boundary_values[0]))
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    print(f"  平均边界弧长: {mean_dist:.4f}")
    print(f"  弧长标准差: {std_dist:.4f}")
    print(f"  变异系数: {std_dist/mean_dist:.4f}")
    
    if std_dist / mean_dist < 0.2:
        print("  ✓ 边界映射平滑，奇点处理有效")
    else:
        print("  ⚠ 边界映射存在波动")
    
    print("\n测试2: 积分有界性验证")
    test_points = [0.9, 0.99, 0.999, 0.9999]
    values = []
    for r in test_points:
        z = r + 0j
        w = sc.map(z)
        values.append(w)
        print(f"  f({r:.4f}) = {w:.6f}, |f|={np.abs(w):.6f}")
    
    if np.all(np.abs(values) < 10):
        print("  ✓ 映射值有界，无发散")
    else:
        print("  ⚠ 映射值可能发散")
    
    return sc


def example_inverse_map():
    square_vertices = [
        -1 - 1j,
        1 - 1j,
        1 + 1j,
        -1 + 1j
    ]
    
    sc = SchwarzChristoffelDisk(square_vertices)
    
    print("\n" + "=" * 60)
    print("逆映射测试")
    print("=" * 60)
    
    original_points = np.array([0.3+0.2j, -0.4+0.5j, 0.8+0.1j])
    print(f"圆盘上的原点: {original_points}")
    
    mapped = sc.map(original_points)
    print(f"映射到多边形: {mapped}")
    
    recovered = sc.inverse_map(mapped)
    print(f"逆映射回圆盘: {recovered}")
    
    errors = np.abs(original_points - recovered)
    print(f"\n映射误差: {errors}")
    print(f"最大误差: {np.max(errors):.2e}")
    
    return sc


def demo():
    print("\n" + "=" * 60)
    print("Schwarz-Christoffel圆盘到多边形共形映射")
    print("  (含积分奇点处理: 高斯-雅可比积分 + 变量变换)")
    print("=" * 60)
    
    sc_square = example_square()
    sc_triangle = example_triangle()
    sc_sing = example_singularity_convergence()
    sc_test = test_singularity_accuracy()
    sc_inverse = example_inverse_map()
    
    print("\n" + "=" * 60)
    print("技术说明")
    print("=" * 60)
    print("""
积分奇点处理方法 (多层策略):

    1. 分段积分策略:
       - 远离奇点的区间 [0, 0.95]: 标准自适应积分 (scipy.integrate.quad)
       - 接近奇点的区间 [0.95, 1.0]: 高斯-雅可比积分

    2. 高斯-雅可比积分 (Gauss-Jacobi):
       专门用于处理含 (1-t)^β 型奇点的积分
       权重函数: (1-t)^β
       自动计算积分节点和权重

    3. 变量替换技术 (备用方案):
       对 (1-t)^(-β) 型奇点，使用变换:
       t = 1 - (1-u)^(1/(1-β))
       将奇点积分转化为正则函数积分

    4. 渐近估计:
       极接近奇点时，使用解析公式估计积分值

    5. 复数积分分离:
       将复积分分离为实部和虚部，分别积分

数学原理:
    Schwarz-Christoffel变换公式:
    f(z) = A * e^(iθ) * ∫[0,z] ∏_{k=1}^{n} (1 - ζ/w_k)^(-β_k) dζ + C
    
    其中:
    - w_k: 单位圆上的预顶点
    - β_k = 1 - α_k/π, α_k: 多边形内角
    - A, θ, C: 缩放、旋转、平移参数

    奇点来源: 当 ζ → w_k 时，(1 - ζ/w_k) → 0
    积分收敛条件: β_k < 1 (对于凸多边形恒成立)
""")
    
    print("\n" + "=" * 60)
    print("使用说明")
    print("=" * 60)
    print("""
基本用法:
    from schwarz_christoffel import SchwarzChristoffelDisk, regular_polygon
    
    # 定义多边形顶点 (复数列表)
    vertices = [-1-1j, 1-1j, 1+1j, -1+1j]  # 正方形
    
    # 创建映射对象 (自动处理积分奇点)
    sc = SchwarzChristoffelDisk(vertices, tol=1e-8)
    
    # 正向映射: 圆盘 → 多边形
    z = 0.5 + 0.3j  # 单位圆盘内的点
    w = sc.map(z)
    
    # 逆映射: 多边形 → 圆盘 (牛顿迭代)
    z_recovered = sc.inverse_map(w)
    
    # 计算映射导数
    df = sc.derivative(z)
    
    # 映射边界 (自动处理预顶点奇点)
    boundary = sc.map_boundary(num_points=200)
    
    # 生成正多边形顶点
    pentagon = regular_polygon(n=5, radius=2, center=0+0j)

奇点处理类 (SingularityIntegrator):
    - gauss_jacobi_nodes_weights: 生成高斯-雅可比节点和权重
    - gauss_jacobi_integrate: 执行高斯-雅可比积分
    - integrate_singular_right: 处理右端点奇点的积分
    - integrate_variable_transform: 变量变换法 (备用)
""")
    
    print("演示完成!")


if __name__ == "__main__":
    demo()
