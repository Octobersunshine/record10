import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar


class VelocityModel:
    def __init__(self, model_type='isotropic', params=None):
        self.model_type = model_type
        self.params = params if params is not None else {}
        self._validate_params()

    def _validate_params(self):
        if self.model_type == 'gradient':
            if 'v0' not in self.params:
                self.params['v0'] = 2000.0
            if 'k' not in self.params:
                self.params['k'] = 0.5
        elif self.model_type == 'layered':
            if 'layers' not in self.params:
                self.params['layers'] = [
                    {'z_top': 0, 'z_bottom': 1000, 'v': 2000},
                    {'z_top': 1000, 'z_bottom': 2000, 'v': 3000},
                    {'z_top': 2000, 'z_bottom': 3000, 'v': 4000}
                ]
            self.params['layers'].sort(key=lambda x: x['z_top'])
        elif self.model_type == 'hti':
            if 'vp0' not in self.params:
                self.params['vp0'] = 3000.0
            if 'vs0' not in self.params:
                self.params['vs0'] = 1500.0
            if 'epsilon' not in self.params:
                self.params['epsilon'] = 0.15
            if 'gamma' not in self.params:
                self.params['gamma'] = 0.1
            if 'delta' not in self.params:
                self.params['delta'] = 0.05
            if 'fracture_azimuth' not in self.params:
                self.params['fracture_azimuth'] = 0.0
        elif self.model_type == 'vti':
            if 'vp0' not in self.params:
                self.params['vp0'] = 3000.0
            if 'vs0' not in self.params:
                self.params['vs0'] = 1500.0
            if 'epsilon' not in self.params:
                self.params['epsilon'] = 0.2
            if 'gamma' not in self.params:
                self.params['gamma'] = 0.15
            if 'delta' not in self.params:
                self.params['delta'] = 0.1

    def velocity(self, z):
        z = abs(z)
        if self.model_type in ['gradient']:
            v0 = self.params['v0']
            k = self.params['k']
            return v0 + k * z
        elif self.model_type == 'layered':
            for layer in self.params['layers']:
                if layer['z_top'] <= z < layer['z_bottom']:
                    return layer['v']
            return self.params['layers'][-1]['v']
        elif self.model_type in ['hti', 'vti']:
            return self.params['vp0']
        else:
            return self.params.get('v0', 2000.0)

    def velocity_gradient(self, z):
        z = abs(z)
        if self.model_type == 'gradient':
            return self.params['k']
        else:
            return 0.0

    def get_layer_boundaries(self):
        if self.model_type == 'layered':
            return [layer['z_bottom'] for layer in self.params['layers'][:-1]]
        return []

    def get_layer_velocity_at_boundary(self, z, direction='down'):
        z = abs(z)
        if self.model_type == 'layered':
            for i, layer in enumerate(self.params['layers']):
                if layer['z_top'] <= z < layer['z_bottom']:
                    if direction == 'down' and i < len(self.params['layers']) - 1:
                        return layer['v'], self.params['layers'][i+1]['v']
                    elif direction == 'up' and i > 0:
                        return layer['v'], self.params['layers'][i-1]['v']
                    else:
                        return layer['v'], layer['v']
            v = self.params['layers'][-1]['v']
            return v, v
        else:
            v = self.velocity(z)
            return v, v

    def critical_angle(self, v1, v2):
        if v2 <= v1:
            return None
        return np.degrees(np.arcsin(v1 / v2))

    def snell_refraction(self, theta1, v1, v2):
        sin_theta2 = (v2 / v1) * np.sin(np.radians(theta1))
        if abs(sin_theta2) >= 1.0:
            return None
        return np.degrees(np.arcsin(sin_theta2))

    def is_isotropic(self):
        return self.model_type in ['isotropic', 'gradient', 'layered']


class AnisotropicModel(VelocityModel):
    def __init__(self, model_type='hti', params=None):
        super().__init__(model_type, params)
        self.wave_type = params.get('wave_type', 'qP') if params else 'qP'

    def christoffel_matrix_2d(self, theta):
        theta_rad = np.radians(theta)
        l = np.cos(theta_rad)
        n = np.sin(theta_rad)
        if self.model_type == 'vti':
            vp0 = self.params['vp0']
            vs0 = self.params['vs0']
            epsilon = self.params['epsilon']
            delta = self.params['delta']
            gamma = self.params['gamma']
            c11 = vp0**2 * (1 + 2 * epsilon)
            c33 = vp0**2
            c44 = vs0**2
            c66 = vs0**2 * (1 + 2 * gamma)
            c13 = np.sqrt((vp0**2 - vs0**2) * (vp0**2 * (1 + 2 * delta) - vs0**2)) - vs0**2
            Γ11 = c11 * l**2 + c44 * n**2
            Γ13 = (c13 + c44) * l * n
            Γ33 = c44 * l**2 + c33 * n**2
            Γ = np.array([[Γ11, Γ13], [Γ13, Γ33]])
            return Γ
        elif self.model_type == 'hti':
            vp0 = self.params['vp0']
            vs0 = self.params['vs0']
            epsilon = self.params['epsilon']
            delta = self.params['delta']
            gamma = self.params['gamma']
            az = np.radians(self.params['fracture_azimuth'])
            c11 = vp0**2
            c33 = vp0**2 * (1 + 2 * epsilon)
            c44 = vs0**2 * (1 + 2 * gamma)
            c66 = vs0**2
            c13 = np.sqrt((vp0**2 - vs0**2) * (vp0**2 * (1 + 2 * delta) - vs0**2)) - vs0**2
            Γ11 = c11 * l**2 + c66 * n**2
            Γ13 = (c13 + c66) * l * n
            Γ33 = c44 * l**2 + c33 * n**2
            Γ = np.array([[Γ11, Γ13], [Γ13, Γ33]])
            return Γ
        else:
            v = self.velocity(0)
            return np.array([[v**2, 0], [0, v**2]])

    def phase_velocity(self, theta):
        if self.is_isotropic():
            v = self.velocity(0)
            return v, np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])
        Γ = self.christoffel_matrix_2d(theta)
        eigenvalues, eigenvectors = np.linalg.eigh(Γ)
        eigenvalues = np.maximum(eigenvalues, 0)
        vp = np.sqrt(eigenvalues[1])
        vsv = np.sqrt(eigenvalues[0])
        if self.wave_type == 'qP':
            v_phase = vp
            polarization = eigenvectors[:, 1]
        elif self.wave_type in ['qS', 'qSV']:
            v_phase = vsv
            polarization = eigenvectors[:, 0]
        else:
            v_phase = vp
            polarization = eigenvectors[:, 1]
        return v_phase, polarization

    def group_velocity_vector(self, theta, delta_theta=0.1):
        theta_rad = np.radians(theta)
        v_phase, _ = self.phase_velocity(theta)
        v_phase_plus, _ = self.phase_velocity(theta + delta_theta)
        v_phase_minus, _ = self.phase_velocity(theta - delta_theta)
        dv_dtheta = (v_phase_plus - v_phase_minus) / (2 * np.radians(delta_theta))
        l = np.cos(theta_rad)
        n = np.sin(theta_rad)
        gx = v_phase * l - dv_dtheta * n
        gz = v_phase * n + dv_dtheta * l
        group_mag = np.sqrt(gx**2 + gz**2)
        group_angle = np.degrees(np.arctan2(gz, gx))
        return np.array([gx, gz]), group_angle, group_mag

    def phase_slowness_vector(self, theta):
        v_phase, _ = self.phase_velocity(theta)
        theta_rad = np.radians(theta)
        px = np.cos(theta_rad) / v_phase
        pz = np.sin(theta_rad) / v_phase
        return np.array([px, pz])

    def compute_slowness_surface(self, n_points=181):
        thetas = np.linspace(0, 360, n_points)
        px = []
        pz = []
        vx = []
        vz = []
        for theta in thetas:
            p = self.phase_slowness_vector(theta)
            g, _, _ = self.group_velocity_vector(theta)
            px.append(p[0])
            pz.append(p[1])
            vx.append(g[0])
            vz.append(g[1])
        return {
            'thetas': thetas,
            'px': np.array(px),
            'pz': np.array(pz),
            'vx': np.array(vx),
            'vz': np.array(vz)
        }


class RayPath:
    def __init__(self):
        self.x = []
        self.z = []
        self.s = []
        self.p = []
        self.q = []
        self.phase_angle = []
        self.group_angle = []
        self.events = []

    def add_segment(self, x_seg, z_seg, s_seg, p_seg=None, q_seg=None,
                   phase_angle_seg=None, group_angle_seg=None, event_type=None):
        self.x.extend(x_seg if hasattr(x_seg, '__iter__') else [x_seg])
        self.z.extend(z_seg if hasattr(z_seg, '__iter__') else [z_seg])
        self.s.extend(s_seg if hasattr(s_seg, '__iter__') else [s_seg])
        if p_seg is not None:
            self.p.extend(p_seg if hasattr(p_seg, '__iter__') else [p_seg])
        if q_seg is not None:
            self.q.extend(q_seg if hasattr(q_seg, '__iter__') else [q_seg])
        if phase_angle_seg is not None:
            self.phase_angle.extend(phase_angle_seg if hasattr(phase_angle_seg, '__iter__') else [phase_angle_seg])
        if group_angle_seg is not None:
            self.group_angle.extend(group_angle_seg if hasattr(group_angle_seg, '__iter__') else [group_angle_seg])
        if event_type:
            self.events.append((len(self.x)-1, event_type))

    def get_arrays(self):
        return (np.array(self.x), np.array(self.z),
                np.array(self.p), np.array(self.q),
                np.array(self.s))


class ShootingMethod:
    def __init__(self, velocity_model, debug=False):
        self.vm = velocity_model
        self.debug = debug
        self.max_reflections = 5
        self.is_anisotropic = isinstance(velocity_model, AnisotropicModel)

    def _ray_ode_isotropic(self, s, state):
        x, z, p, q = state
        v = self.vm.velocity(z)
        dv_dz = self.vm.velocity_gradient(z)
        dx_ds = p * v
        dz_ds = q * v
        dp_ds = 0.0
        dq_ds = -dv_dz / (v ** 2)
        return [dx_ds, dz_ds, dp_ds, dq_ds]

    def _ray_ode_anisotropic(self, s, state):
        x, z, phase_angle = state
        phase_angle = np.clip(phase_angle, -89, 89)
        g_vec, group_angle, v_group = self.vm.group_velocity_vector(phase_angle)
        dx_ds = g_vec[0]
        dz_ds = g_vec[1]
        dv_dz = self.vm.velocity_gradient(z)
        if abs(dv_dz) > 1e-10:
            dtheta_ds = -np.sin(np.radians(phase_angle)) * dv_dz / (v_group ** 2) * v_group
        else:
            dtheta_ds = 0.0
        return [dx_ds, dz_ds, dtheta_ds]

    def _ray_ode(self, s, state):
        if self.is_anisotropic:
            return self._ray_ode_anisotropic(s, state)
        else:
            return self._ray_ode_isotropic(s, state)

    def _crossed_boundary(self, z1, z2):
        boundaries = self.vm.get_layer_boundaries()
        z1, z2 = abs(z1), abs(z2)
        for b in boundaries:
            if min(z1, z2) < b < max(z1, z2):
                return True, b, np.sign(z2 - z1)
        return False, None, None

    def _handle_interface(self, x, z, p, q, boundary, direction):
        v1, v2 = self.vm.get_layer_velocity_at_boundary(z, 'down' if direction > 0 else 'up')
        theta1 = np.degrees(np.arctan2(abs(q), abs(p)))
        p_mag = np.sqrt(p**2 + q**2)
        critical_angle = self.vm.critical_angle(v1, v2)
        if critical_angle is not None and theta1 >= critical_angle - 1e-6:
            if self.debug:
                print(f"  全反射/临界角: θ={theta1:.2f}°, θc={critical_angle:.2f}°")
            p_new = p
            q_new = -q
            return p_new, q_new, 'total_reflection'
        theta2 = self.vm.snell_refraction(theta1, v1, v2)
        if theta2 is None:
            p_new = p
            q_new = -q
            return p_new, q_new, 'total_reflection'
        if self.debug:
            print(f"  折射: θ1={theta1:.2f}°, θ2={theta2:.2f}°, v1={v1}, v2={v2}")
        sin_theta1 = np.sin(np.radians(theta1))
        sin_theta2 = np.sin(np.radians(theta2))
        cos_theta1 = np.cos(np.radians(theta1))
        cos_theta2 = np.cos(np.radians(theta2))
        p_new = p_mag * sin_theta2 * np.sign(p)
        q_new = p_mag * cos_theta2 * np.sign(q)
        return p_new, q_new, 'refraction'

    def shoot_ray(self, x0, z0, takeoff_angle, max_s=20000, num_points_per_segment=200):
        ray = RayPath()
        if self.is_anisotropic:
            initial_phase_angle = takeoff_angle
            x, z = x0, z0
            phase_angle = initial_phase_angle
            s_total = 0
            initial_state = [x, z, phase_angle]
            try:
                sol = solve_ivp(
                    self._ray_ode,
                    [0, max_s],
                    initial_state,
                    method='RK45',
                    max_step=10.0
                )
                x_seg = sol.y[0]
                z_seg = sol.y[1]
                phase_angle_seg = sol.y[2]
                s_seg = sol.t
                group_angle_seg = []
                for pa in phase_angle_seg:
                    _, ga, _ = self.vm.group_velocity_vector(pa)
                    group_angle_seg.append(ga)
                ray.add_segment(x_seg, z_seg, s_seg,
                               phase_angle_seg=phase_angle_seg,
                               group_angle_seg=group_angle_seg)
                s_total = s_seg[-1]
            except Exception as e:
                if self.debug:
                    print(f"  射线追踪错误: {e}")
                return np.array([x0]), np.array([z0]), np.array([0])
        else:
            x, z = x0, z0
            p = np.cos(np.radians(takeoff_angle))
            q = np.sin(np.radians(takeoff_angle))
            s_total = 0
            reflection_count = 0
            while s_total < max_s and reflection_count < self.max_reflections:
                def event_boundary(s, state):
                    x, z, p, q = state
                    boundaries = self.vm.get_layer_boundaries()
                    min_dist = float('inf')
                    for b in boundaries:
                        dist = abs(abs(z) - b)
                        min_dist = min(min_dist, dist)
                    return min_dist - 1.0
                event_boundary.terminal = True
                event_boundary.direction = 0
                initial_state = [x, z, p, q]
                s_span = [0, max_s - s_total]
                try:
                    sol = solve_ivp(
                        self._ray_ode,
                        s_span,
                        initial_state,
                        method='RK45',
                        events=event_boundary,
                        max_step=10.0
                    )
                except:
                    break
                x_seg = sol.y[0]
                z_seg = sol.y[1]
                p_seg = sol.y[2]
                q_seg = sol.y[3]
                s_seg = sol.t + s_total
                if len(sol.t_events[0]) > 0:
                    idx = len(s_seg) - 1
                    x, z = x_seg[idx], z_seg[idx]
                    p, q = p_seg[idx], q_seg[idx]
                    s_total = s_seg[idx]
                    crossed, boundary, direction = self._crossed_boundary(z_seg[-2], z_seg[-1])
                    if crossed:
                        p, q, event_type = self._handle_interface(x, z, p, q, boundary, direction)
                        if event_type == 'total_reflection':
                            reflection_count += 1
                        ray.add_segment(x_seg[:-1], z_seg[:-1], s_seg[:-1],
                                      p_seg[:-1], q_seg[:-1], event_type)
                    else:
                        ray.add_segment(x_seg, z_seg, s_seg, p_seg, q_seg)
                        break
                else:
                    ray.add_segment(x_seg, z_seg, s_seg, p_seg, q_seg)
                    break
        x_arr, z_arr, p_arr, q_arr, s_arr = ray.get_arrays()
        if len(x_arr) == 0:
            return np.array([x0]), np.array([z0]), np.array([0])
        travel_time = self._compute_travel_time(x_arr, z_arr)
        return x_arr, z_arr, travel_time

    def _compute_travel_time(self, x, z):
        if len(x) < 2:
            return 0.0
        if self.is_anisotropic:
            tt = 0.0
            for i in range(len(x)-1):
                dx = x[i+1] - x[i]
                dz = z[i+1] - z[i]
                ds = np.sqrt(dx**2 + dz**2)
                z_mid = (z[i] + z[i+1]) / 2
                phase_angle = np.degrees(np.arctan2(dz, dx))
                _, _, v_group = self.vm.group_velocity_vector(phase_angle)
                tt += ds / v_group
            return tt
        else:
            v = np.array([self.vm.velocity(zi) for zi in z])
            ds = np.sqrt(np.diff(x)**2 + np.diff(z)**2)
            travel_time = np.sum(ds / v[:-1])
            return travel_time

    def _objective_x_at_z(self, angle, x0, z0, xr, zr):
        x, z, _ = self.shoot_ray(x0, z0, angle)
        if len(x) < 2:
            return 1e10
        idx = np.argmin(np.abs(z - zr))
        z_error = abs(z[idx] - zr)
        if z_error > 50:
            return 1e10 + z_error * 100
        return abs(x[idx] - xr)

    def find_ray(self, x0, z0, xr, zr, angle_range=(1, 89)):
        angles_to_test = np.linspace(angle_range[0], angle_range[1], 50)
        best_angle = None
        best_dist = float('inf')
        best_x, best_z, best_tt = None, None, None
        if self.debug:
            print(f"\n搜索射线: 目标 ({xr}, {zr})")
        for angle in angles_to_test:
            try:
                x, z, tt = self.shoot_ray(x0, z0, angle)
                if len(x) < 2:
                    continue
                idx = np.argmin(np.sqrt((x - xr)**2 + (z - zr)**2))
                dist = np.sqrt((x[idx] - xr)**2 + (z[idx] - zr)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_angle = angle
                    best_x, best_z, best_tt = x, z, tt
            except:
                continue
        if best_angle is None:
            return None, None, None, None
        try:
            def objective(angle):
                return self._objective_x_at_z(angle, x0, z0, xr, zr)
            search_range = max(5.0, (angle_range[1] - angle_range[0]) * 0.1)
            res = minimize_scalar(
                objective,
                bracket=(max(angle_range[0], best_angle - search_range),
                         best_angle,
                         min(angle_range[1], best_angle + search_range)),
                method='brent'
            )
            optimal_angle = res.x
            x, z, tt = self.shoot_ray(x0, z0, optimal_angle)
            final_dist = self._objective_x_at_z(optimal_angle, x0, z0, xr, zr)
            if final_dist < best_dist:
                if self.debug:
                    print(f"  优化成功: 初始角度 {best_angle:.2f}° -> 最优角度 {optimal_angle:.2f}°")
                    print(f"  距离误差: {best_dist:.2f}m -> {final_dist:.2f}m")
                return x, z, tt, optimal_angle
            else:
                return best_x, best_z, best_tt, best_angle
        except Exception as e:
            if self.debug:
                print(f"  优化失败: {e}, 使用初始角度 {best_angle:.2f}°")
            return best_x, best_z, best_tt, best_angle

    def compute_rays(self, source, receivers, angle_range=(1, 89)):
        rays = []
        for receiver in receivers:
            x, z, tt, angle = self.find_ray(
                source[0], source[1],
                receiver[0], receiver[1],
                angle_range
            )
            if x is not None and len(x) > 1:
                rays.append({
                    'receiver': receiver,
                    'x': x,
                    'z': z,
                    'travel_time': tt,
                    'takeoff_angle': angle
                })
        return rays


def plot_rays(rays, source, vm, xlim=None, zlim=None, title=None):
    plt.figure(figsize=(12, 8))
    z_max = zlim[0] if zlim else 3000
    z_plot = np.linspace(0, z_max, 500)
    v_plot = np.array([vm.velocity(z) for z in z_plot])
    plt.subplot(2, 1, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(rays))) if rays else []
    for i, ray in enumerate(rays):
        plt.plot(ray['x'], ray['z'], color=colors[i], linewidth=2,
                 label=f"Receiver {i+1}: {ray['travel_time']:.3f}s, {ray['takeoff_angle']:.1f}°")
    plt.scatter(source[0], source[1], c='red', s=300, marker='*', edgecolors='black', label='Source', zorder=10)
    for i, ray in enumerate(rays):
        plt.scatter(ray['receiver'][0], ray['receiver'][1], c=colors[i], s=150, marker='v',
                    edgecolors='black', zorder=10, label=f'_nolegend_')
    if vm.model_type == 'layered':
        for boundary in vm.get_layer_boundaries():
            b_xmin = xlim[0] if xlim else (min([ray['x'].min() for ray in rays]) if rays else 0)
            b_xmax = xlim[1] if xlim else (max([ray['x'].max() for ray in rays]) if rays else 10000)
            plt.hlines(boundary, b_xmin, b_xmax, colors='gray', linestyles='--', alpha=0.7,
                      label='Layer Boundary' if boundary == vm.get_layer_boundaries()[0] else '_nolegend_')
    plt.xlabel('X (m)')
    plt.ylabel('Depth Z (m)')
    model_type = f"{vm.model_type.upper()}" if isinstance(vm, AnisotropicModel) else vm.model_type
    plt_title = f"Ray Paths - {model_type} Model" + (f" ({title})" if title else "")
    plt.title(plt_title)
    plt.legend(fontsize=8, loc='upper right')
    plt.grid(True, alpha=0.3)
    if xlim:
        plt.xlim(xlim)
    if zlim:
        plt.ylim(zlim)
    plt.gca().invert_yaxis()
    plt.subplot(2, 1, 2)
    plt.plot(v_plot, z_plot, 'g-', linewidth=2, label='Velocity')
    if vm.model_type == 'layered':
        for boundary in vm.get_layer_boundaries():
            plt.hlines(boundary, v_plot.min(), v_plot.max(), colors='gray', linestyles='--', alpha=0.7)
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Depth Z (m)')
    plt.title('Velocity Profile')
    plt.grid(True, alpha=0.3)
    plt.legend()
    if zlim:
        plt.ylim(zlim)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    filename = f"ray_paths_{model_type}_{title if title else 'model'}.png".replace(" ", "_")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"射线路径图已保存为 {filename}")
    plt.close()


def plot_slowness_surface(anisotropic_model, title=None):
    surface = anisotropic_model.compute_slowness_surface(360)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1 = axes[0]
    ax1.plot(surface['px'], surface['pz'], 'b-', linewidth=2, label='Slowness')
    ax1.set_xlabel('Px (s/m)')
    ax1.set_ylabel('Pz (s/m)')
    model_type = anisotropic_model.model_type.upper()
    wave_type = anisotropic_model.wave_type
    ax1.set_title(f'{model_type} Slowness Surface - {wave_type} wave')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax1.legend()
    ax1.set_aspect('equal')
    ax2 = axes[1]
    ax2.plot(surface['vx'], surface['vz'], 'r-', linewidth=2, label='Group Velocity')
    v_iso = anisotropic_model.params.get('vp0', 3000)
    theta_circle = np.linspace(0, 2*np.pi, 100)
    ax2.plot(v_iso * np.cos(theta_circle), v_iso * np.sin(theta_circle),
             'k--', alpha=0.5, label='Isotropic Reference')
    ax2.set_xlabel('Vx (m/s)')
    ax2.set_ylabel('Vz (m/s)')
    ax2.set_title(f'{model_type} Group Velocity Surface - {wave_type} wave')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax2.legend()
    ax2.set_aspect('equal')
    plt.tight_layout()
    filename = f"slowness_{model_type}_{wave_type}.png".replace(" ", "_")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"慢度/群速度面图已保存为 {filename}")
    plt.close()


def demo_critical_angle():
    print("\n" + "=" * 70)
    print("演示: 临界角与折射盲区分析")
    print("=" * 70)
    vm = VelocityModel(
        model_type='layered',
        params={
            'layers': [
                {'z_top': 0, 'z_bottom': 1000, 'v': 2000},
                {'z_top': 1000, 'z_bottom': 3000, 'v': 3000}
            ]
        }
    )
    v1, v2 = 2000, 3000
    theta_c = np.degrees(np.arcsin(v1 / v2))
    print(f"\n两层模型: v1={v1}m/s, v2={v2}m/s")
    print(f"临界角 θc = arcsin(v1/v2) = {theta_c:.2f}°")
    print(f"\n出射角 < {theta_c:.2f}°: 发生折射")
    print(f"出射角 ≈ {theta_c:.2f}°: 发生折射盲区/滑行波")
    print(f"出射角 > {theta_c:.2f}°: 发生全反射")
    sm = ShootingMethod(vm, debug=True)
    source = (0, 0)
    angles = [20, 40, 41.8, 45, 60]
    print("\n测试不同出射角的射线:")
    print("-" * 70)
    rays_for_plot = []
    for angle in angles:
        print(f"\n出射角: {angle}°")
        x, z, tt = sm.shoot_ray(0, 0, angle)
        print(f"  最终位置: ({x[-1]:.1f}, {z[-1]:.1f})")
        print(f"  走时: {tt:.4f}s")
        rays_for_plot.append({
            'receiver': (x[-1], z[-1]),
            'x': x,
            'z': z,
            'travel_time': tt,
            'takeoff_angle': angle
        })
    plot_rays(rays_for_plot, source, vm, xlim=(-500, 6000), zlim=(2000, -100),
              title="Critical Angle Demo")
    print("\n" + "=" * 70)


def demo_anisotropic():
    print("\n" + "=" * 70)
    print("演示: 各向异性介质射线追踪 (裂缝HTI模型)")
    print("=" * 70)
    hti_model = AnisotropicModel(
        model_type='hti',
        params={
            'vp0': 3000.0,
            'vs0': 1500.0,
            'epsilon': 0.15,
            'gamma': 0.10,
            'delta': 0.05,
