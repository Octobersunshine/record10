import numpy as np
import matplotlib.pyplot as plt


class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0.0, output_limits=(-1, 1), 
                 anti_windup=True, anti_windup_gain=0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.anti_windup = anti_windup
        self.anti_windup_gain = anti_windup_gain
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
    
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
    
    def update(self, measurement, current_time):
        error = self.setpoint - measurement
        
        if self.prev_time is None:
            dt = 0.0
        else:
            dt = current_time - self.prev_time
        
        proportional = self.kp * error
        
        if dt > 0:
            self.integral += error * dt
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0.0
        
        integral = self.ki * self.integral
        derivative = self.kd * derivative
        
        output = proportional + integral + derivative
        
        min_output, max_output = self.output_limits
        output_clipped = np.clip(output, min_output, max_output)
        
        if self.anti_windup and self.ki > 0:
            output_saturation = output_clipped - output
            self.integral += self.anti_windup_gain * output_saturation / self.ki
        
        self.prev_error = error
        self.prev_time = current_time
        
        return output_clipped


class DroneAttitudeDynamics:
    def __init__(self, inertia=0.01, damping=0.05):
        self.inertia = inertia
        self.damping = damping
        self.angle = 0.0
        self.angular_velocity = 0.0
    
    def reset(self, initial_angle=0.0):
        self.angle = initial_angle
        self.angular_velocity = 0.0
    
    def update(self, torque, dt):
        angular_acceleration = (torque - self.damping * self.angular_velocity) / self.inertia
        self.angular_velocity += angular_acceleration * dt
        self.angle += self.angular_velocity * dt
        return self.angle


class RelayFeedbackTuner:
    def __init__(self, plant, relay_amplitude=1.0, hysteresis=0.5, simulation_time=20.0, dt=0.01):
        self.plant = plant
        self.relay_amplitude = relay_amplitude
        self.hysteresis = hysteresis
        self.simulation_time = simulation_time
        self.dt = dt
    
    def run_relay_test(self, setpoint=10.0):
        time_steps = int(self.simulation_time / self.dt)
        time_array = np.linspace(0, self.simulation_time, time_steps)
        output_history = np.zeros(time_steps)
        input_history = np.zeros(time_steps)
        
        self.plant.reset()
        relay_output = self.relay_amplitude
        prev_error = 0
        
        for i, t in enumerate(time_array):
            error = setpoint - self.plant.angle
            
            if error > self.hysteresis and prev_error <= self.hysteresis:
                relay_output = self.relay_amplitude
            elif error < -self.hysteresis and prev_error >= -self.hysteresis:
                relay_output = -self.relay_amplitude
            
            prev_error = error
            
            self.plant.update(relay_output, self.dt)
            output_history[i] = self.plant.angle
            input_history[i] = relay_output
        
        return time_array, output_history, input_history, setpoint
    
    def analyze_oscillation(self, time_array, output_history, setpoint):
        deviation = output_history - setpoint
        
        peaks = []
        peak_times = []
        for i in range(2, len(deviation) - 2):
            if (deviation[i] > deviation[i-1] and deviation[i] > deviation[i-2] and
                deviation[i] > deviation[i+1] and deviation[i] > deviation[i+2]):
                peaks.append(deviation[i])
                peak_times.append(time_array[i])
        
        if len(peaks) < 3:
            return None, None
        
        peak_amps = np.abs(peaks[-3:])
        a_peak = np.mean(peak_amps)
        
        if len(peak_times) >= 2:
            t_cycle = peak_times[-1] - peak_times[-3]
        else:
            t_cycle = None
        
        return a_peak, t_cycle
    
    def calculate_pid_params(self, a_peak, t_cycle, method='ziegler-nichols'):
        if a_peak is None or t_cycle is None:
            return None
        
        d = self.relay_amplitude
        a = a_peak
        
        k_crit = (4 * d) / (np.pi * a)
        t_crit = t_cycle
        
        if method == 'ziegler-nichols':
            kp = 0.6 * k_crit
            ki = 2 * kp / t_crit
            kd = kp * t_crit / 8
        elif method == 'pessen-integral':
            kp = 0.7 * k_crit
            ki = 2.5 * kp / t_crit
            kd = 0.15 * kp * t_crit
        elif method == 'some-overshoot':
            kp = 0.33 * k_crit
            ki = 2 * kp / t_crit
            kd = kp * t_crit / 3
        elif method == 'no-overshoot':
            kp = 0.2 * k_crit
            ki = 2 * kp / t_crit
            kd = 0.5 * kp * t_crit
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            'kp': kp,
            'ki': ki,
            'kd': kd,
            'k_crit': k_crit,
            't_crit': t_crit,
            'a_peak': a_peak,
            'method': method
        }
    
    def auto_tune(self, setpoint=10.0, method='ziegler-nichols'):
        time_array, output_history, input_history, _ = self.run_relay_test(setpoint)
        a_peak, t_cycle = self.analyze_oscillation(time_array, output_history, setpoint)
        params = self.calculate_pid_params(a_peak, t_cycle, method)
        
        return params, time_array, output_history, input_history


def plot_relay_test(time_array, output_history, input_history, setpoint, params=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(time_array, output_history, 'b-', linewidth=2, label='系统输出')
    ax1.axhline(y=setpoint, color='r', linestyle='--', label='设定值')
    ax1.axhline(y=setpoint + 0.5, color='g', linestyle=':', alpha=0.5, label='滞环上限')
    ax1.axhline(y=setpoint - 0.5, color='g', linestyle=':', alpha=0.5, label='滞环下限')
    ax1.set_ylabel('角度 (度)', fontsize=12)
    ax1.set_title('继电反馈测试 - 自持振荡', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(time_array, input_history, 'r-', linewidth=2, label='继电输出')
    ax2.set_xlabel('时间 (秒)', fontsize=12)
    ax2.set_ylabel('控制输入', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    if params is not None:
        info_text = f"临界增益 K_crit = {params['k_crit']:.4f}\n"
        info_text += f"临界周期 T_crit = {params['t_crit']:.4f} s\n"
        info_text += f"整定方法: {params['method']}"
        ax1.text(0.02, 0.95, info_text, transform=ax1.transAxes, 
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('relay_feedback_test.png', dpi=300, bbox_inches='tight')
    plt.close()


def simulate_pid(kp, ki, kd, target_angle=30.0, simulation_time=10.0, dt=0.01, 
                 anti_windup=True, anti_windup_gain=0.5):
    pid = PIDController(kp=kp, ki=ki, kd=kd, setpoint=target_angle, output_limits=(-5, 5),
                        anti_windup=anti_windup, anti_windup_gain=anti_windup_gain)
    drone = DroneAttitudeDynamics()
    
    time_steps = int(simulation_time / dt)
    time_array = np.linspace(0, simulation_time, time_steps)
    angle_history = np.zeros(time_steps)
    torque_history = np.zeros(time_steps)
    integral_history = np.zeros(time_steps)
    target_history = np.full(time_steps, target_angle)
    
    pid.reset()
    drone.reset()
    
    for i, t in enumerate(time_array):
        torque = pid.update(drone.angle, t)
        drone.update(torque, dt)
        angle_history[i] = drone.angle
        torque_history[i] = torque
        integral_history[i] = pid.integral
    
    return time_array, angle_history, torque_history, integral_history, target_history


def plot_results(time_array, angle_history, torque_history, integral_history, target_history, 
                 kp, ki, kd, anti_windup_label=''):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    title = f'PID控制器响应 (Kp={kp}, Ki={ki}, Kd={kd})'
    if anti_windup_label:
        title += f' - {anti_windup_label}'
    
    ax1.plot(time_array, angle_history, 'b-', linewidth=2, label='实际角度')
    ax1.plot(time_array, target_history, 'r--', linewidth=2, label='目标角度')
    ax1.set_ylabel('角度 (度)', fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(time_array, torque_history, 'g-', linewidth=2, label='控制力矩')
    ax2.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='输出上限')
    ax2.axhline(y=-5, color='r', linestyle='--', alpha=0.5, label='输出下限')
    ax2.set_ylabel('力矩 (N·m)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(time_array, integral_history, 'm-', linewidth=2, label='积分项累积')
    ax3.set_xlabel('时间 (秒)', fontsize=12)
    ax3.set_ylabel('积分项', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = 'pid_response_anti_windup.png' if '启用' in anti_windup_label else 'pid_response_no_anti_windup.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_comparison(time_array_no_aw, angle_history_no_aw, integral_history_no_aw,
                    time_array_aw, angle_history_aw, integral_history_aw,
                    target_history, kp, ki, kd):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(time_array_no_aw, angle_history_no_aw, 'r-', linewidth=2, label='无抗积分饱和')
    ax1.plot(time_array_aw, angle_history_aw, 'b-', linewidth=2, label='有抗积分饱和')
    ax1.plot(time_array_no_aw, target_history, 'k--', linewidth=2, label='目标角度')
    ax1.set_ylabel('角度 (度)', fontsize=12)
    ax1.set_title(f'抗积分饱和效果对比 (Kp={kp}, Ki={ki}, Kd={kd})', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(time_array_no_aw, integral_history_no_aw, 'r-', linewidth=2, label='积分项(无抗饱和)')
    ax2.plot(time_array_aw, integral_history_aw, 'b-', linewidth=2, label='积分项(有抗饱和)')
    ax2.set_xlabel('时间 (秒)', fontsize=12)
    ax2.set_ylabel('积分项累积', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pid_anti_windup_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def calculate_performance_metrics(time_array, angle_history, target_angle):
    steady_state_error = abs(angle_history[-1] - target_angle)
    
    overshoot = np.max(angle_history) - target_angle
    overshoot_percent = (overshoot / target_angle) * 100 if target_angle != 0 else 0
    
    rise_time_idx = np.where(angle_history >= target_angle * 0.9)[0]
    rise_time = time_array[rise_time_idx[0]] if len(rise_time_idx) > 0 else None
    
    settling_time_idx = np.where(np.abs(angle_history - target_angle) <= target_angle * 0.02)[0]
    if len(settling_time_idx) > 0:
        for i in range(len(settling_time_idx) - 1, -1, -1):
            if settling_time_idx[i] != settling_time_idx[-1] - (len(settling_time_idx) - 1 - i):
                settling_time = time_array[settling_time_idx[i + 1]]
                break
        else:
            settling_time = time_array[settling_time_idx[0]]
    else:
        settling_time = None
    
    return {
        '稳态误差': steady_state_error,
        '超调量(%)': overshoot_percent,
        '上升时间(s)': rise_time,
        '调节时间(s)': settling_time
    }


def print_auto_tune_results(params, method_name):
    print(f"\n【{method_name}整定结果】")
    print(f"  临界增益 K_crit = {params['k_crit']:.4f}")
    print(f"  临界周期 T_crit = {params['t_crit']:.4f} s")
    print(f"  振荡幅值 a = {params['a_peak']:.4f}°")
    print()
    print(f"  PID参数:")
    print(f"    Kp = {params['kp']:.4f}")
    print(f"    Ki = {params['ki']:.4f}")
    print(f"    Kd = {params['kd']:.4f}")


def main():
    print("=" * 70)
    print("无人机姿态PID控制器仿真 - 自动调参演示")
    print("=" * 70)
    print()
    
    target_angle = 30.0
    anti_windup_gain = 1.0
    
    print("=" * 70)
    print("【步骤1: 继电反馈法自动获取临界参数】")
    print()
    
    drone = DroneAttitudeDynamics()
    tuner = RelayFeedbackTuner(
        plant=drone,
        relay_amplitude=2.0,
        hysteresis=0.3,
        simulation_time=15.0,
        dt=0.01
    )
    
    print("正在进行继电反馈测试...")
    params_zn, time_array, output_history, input_history = tuner.auto_tune(
        setpoint=10.0, method='ziegler-nichols'
    )
    
    if params_zn is None:
        print("错误: 未能检测到自持振荡，请调整继电参数")
        return
    
    print_auto_tune_results(params_zn, "Ziegler-Nichols")
    
    methods = ['pessen-integral', 'some-overshoot', 'no-overshoot']
    method_names = ['Pessen Integral', '少量超调', '无超调']
    all_params = {'ziegler-nichols': params_zn}
    
    for method, name in zip(methods, method_names):
        params, _, _, _ = tuner.auto_tune(setpoint=10.0, method=method)
        all_params[method] = params
        print_auto_tune_results(params, name)
    
    print()
    print("正在生成继电反馈测试图...")
    plot_relay_test(time_array, output_history, input_history, 10.0, params_zn)
    print("已保存: relay_feedback_test.png")
    
    print()
    print("=" * 70)
    print("【步骤2: 使用自动整定参数进行仿真验证】")
    
    best_method = 'ziegler-nichols'
    kp_auto = all_params[best_method]['kp']
    ki_auto = all_params[best_method]['ki']
    kd_auto = all_params[best_method]['kd']
    
    print(f"\n使用 {best_method} 参数进行仿真:")
    print(f"  Kp = {kp_auto:.4f}, Ki = {ki_auto:.4f}, Kd = {kd_auto:.4f}")
    print(f"  目标角度: {target_angle}°")
    
    time_array_auto, angle_history_auto, torque_history_auto, integral_history_auto, target_history = simulate_pid(
        kp=kp_auto, ki=ki_auto, kd=kd_auto, target_angle=target_angle,
        anti_windup=True, anti_windup_gain=anti_windup_gain
    )
    
    metrics_auto = calculate_performance_metrics(time_array_auto, angle_history_auto, target_angle)
    print("\n【自动整定参数性能指标】")
    for key, value in metrics_auto.items():
        if value is not None:
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: N/A")
    
    print()
    print("=" * 70)
    print("【步骤3: 抗积分饱和效果验证】")
    print()
    
    time_array_no_aw, angle_history_no_aw, torque_history_no_aw, integral_history_no_aw, _ = simulate_pid(
        kp=kp_auto, ki=ki_auto, kd=kd_auto, target_angle=target_angle,
        anti_windup=False, anti_windup_gain=0
    )
    
    metrics_no_aw = calculate_performance_metrics(time_array_no_aw, angle_history_no_aw, target_angle)
    print("【无抗积分饱和性能指标】")
    for key, value in metrics_no_aw.items():
        if value is not None:
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: N/A")
    
    print()
    print("【抗积分饱和改进效果】")
    print(f"  超调量减少: {metrics_no_aw['超调量(%)'] - metrics_auto['超调量(%)']:.2f}%")
    print(f"  调节时间改善: {metrics_no_aw['调节时间(s)'] - metrics_auto['调节时间(s)']:.4f} s")
    
    print()
    print("正在生成仿真曲线...")
    plot_results(time_array_auto, angle_history_auto, torque_history_auto,
                 integral_history_auto, target_history, kp_auto, ki_auto, kd_auto,
                 '自动整定+抗积分饱和')
    plot_results(time_array_no_aw, angle_history_no_aw, torque_history_no_aw,
                 integral_history_no_aw, target_history, kp_auto, ki_auto, kd_auto,
                 '自动整定-无抗积分饱和')
    plot_comparison(time_array_no_aw, angle_history_no_aw, integral_history_no_aw,
                    time_array_auto, angle_history_auto, integral_history_auto,
                    target_history, kp_auto, ki_auto, kd_auto)
    
    print("仿真曲线已保存:")
    print("  - pid_response_自动整定+抗积分饱和.png")
    print("  - pid_response_自动整定-无抗积分饱和.png")
    print("  - pid_anti_windup_comparison.png")
    
    print()
    print("=" * 70)
    print("【自动调参原理说明】")
    print()
    print("继电反馈法:")
    print("  1. 使用继电控制器代替PID，使系统产生自持振荡")
    print("  2. 测量振荡幅值(a)和周期(T_crit)")
    print("  3. 计算临界增益: K_crit = 4d / (πa)  (d为继电幅值)")
    print()
    print("Ziegler-Nichols公式:")
    print("  Kp = 0.6 * K_crit")
    print("  Ki = 2 * Kp / T_crit")
    print("  Kd = Kp * T_crit / 8")
    print()
    print("不同整定方法特点:")
    print("  - ziegler-nichols: 经典方法，响应快但有一定超调")
    print("  - pessen-integral: 积分误差最小，超调适中")
    print("  - some-overshoot: 允许少量超调，平衡性能")
    print("  - no-overshoot: 追求无超调，响应较慢")
    print()
    print("使用建议:")
    print("  1. 先用继电反馈法获取初始PID参数")
    print("  2. 根据实际需求选择整定方法")
    print("  3. 启用抗积分饱和机制改善大阶跃响应")
    print("  4. 必要时可微调参数以获得更佳性能")


if __name__ == "__main__":
    main()
