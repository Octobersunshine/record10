import numpy as np
from online_changepoint_detection import OnlineBayesianChangepointDetection, AdaptiveOnlineChangepointDetection


def example_basic_usage():
    """基本使用示例"""
    print("="*70)
    print("示例1: 基本在线变点检测")
    print("="*70)
    
    np.random.seed(42)
    
    data = np.concatenate([
        np.random.normal(0, 1, 100),
        np.random.normal(5, 1.2, 100),
        np.random.normal(2, 0.8, 100)
    ])
    
    detector = OnlineBayesianChangepointDetection(
        hazard=0.01,
        threshold=0.5
    )
    
    print("\n逐点处理数据（模拟实时流）：")
    print("-"*70)
    print(f"{'时间':>6} {'数据值':>8} {'变点概率':>12} {'贝叶斯因子':>12}")
    print("-"*70)
    
    for t, x in enumerate(data):
        cp_prob = detector.update(x)
        bf = detector.get_bayes_factor()
        
        if t % 30 == 0 or cp_prob > 0.3:
            flag = " *" if cp_prob > 0.5 else ""
            print(f"{t:6d} {x:8.3f} {cp_prob:12.4f} {bf:12.2f}{flag}")
    
    print("\n" + "="*70)
    detector.print_summary()
    return detector


def example_adaptive_detection():
    """自适应在线变点检测示例"""
    print("\n" + "="*70)
    print("示例2: 自适应在线变点检测（危险率自动调整）")
    print("="*70)
    
    np.random.seed(123)
    
    data = np.concatenate([
        np.random.normal(0, 0.5, 150),
        np.random.normal(3, 0.8, 150)
    ])
    
    detector = AdaptiveOnlineChangepointDetection(
        hazard=0.005,
        hazard_min=0.001,
        hazard_max=0.08,
        adaptation_rate=0.03,
        threshold=0.4
    )
    
    print("\n处理数据...")
    for t, x in enumerate(data):
        detector.update(x)
    
    print(f"\n危险率范围: {min(detector.hazard_history):.4f} ~ {max(detector.hazard_history):.4f}")
    detector.print_summary()
    return detector


def example_real_time_monitoring():
    """实时监控场景示例"""
    print("\n" + "="*70)
    print("示例3: 实时监控场景 - 工业传感器监控")
    print("="*70)
    
    class SensorMonitor:
        def __init__(self):
            self.detector = OnlineBayesianChangepointDetection(
                hazard=0.005,
                threshold=0.6
            )
            self.alerts = []
        
        def process_reading(self, timestamp, value):
            cp_prob = self.detector.update(value)
            bf = self.detector.get_bayes_factor()
            
            if cp_prob > self.detector.threshold:
                alert = {
                    'timestamp': timestamp,
                    'value': value,
                    'probability': cp_prob,
                    'bayes_factor': bf
                }
                self.alerts.append(alert)
                
                print(f"⚠  告警 @ t={timestamp}: 变点概率={cp_prob:.4f}, BF={bf:.2f}")
            
            return cp_prob, bf
    
    np.random.seed(789)
    
    print("\n模拟工业传感器数据（正常运行 -> 故障 -> 恢复）:")
    print("-"*70)
    
    monitor = SensorMonitor()
    
    # 正常运行
    normal_data = np.random.normal(25, 0.5, 200)
    # 发生故障
    fault_data = np.random.normal(40, 2, 150)
    # 恢复
    recovery_data = np.random.normal(26, 0.6, 150)
    
    all_data = np.concatenate([normal_data, fault_data, recovery_data])
    
    print("开始监控...")
    for t, val in enumerate(all_data):
        monitor.process_reading(t, val)
    
    print(f"\n共触发 {len(monitor.alerts)} 次告警")
    print(f"检测到变点位置: {monitor.detector.changepoints}")
    
    return monitor


def example_multiple_sensors():
    """多传感器监控示例"""
    print("\n" + "="*70)
    print("示例4: 多传感器并行监控")
    print("="*70)
    
    np.random.seed(456)
    
    n_sensors = 3
    n_timesteps = 500
    
    sensors = []
    for i in range(n_sensors):
        detector = OnlineBayesianChangepointDetection(
            hazard=0.01,
            threshold=0.5
        )
        sensors.append({
            'name': f'Sensor_{i+1}',
            'detector': detector
        })
    
    print(f"\n模拟 {n_sensors} 个传感器的实时数据...")
    
    for t in range(n_timesteps):
        for i, sensor in enumerate(sensors):
            if t < 150:
                val = np.random.normal(i * 2, 0.5)
            elif t < 350:
                val = np.random.normal(i * 2 + 5, 0.8)
            else:
                val = np.random.normal(i * 2 + 2, 0.6)
            
            cp_prob = sensor['detector'].update(val)
            
            if cp_prob > 0.6 and t % 50 == 0:
                print(f"t={t:3d} | {sensor['name']}: 变点概率={cp_prob:.4f}")
    
    print("\n各传感器检测结果:")
    for sensor in sensors:
        cps = sensor['detector'].changepoints
        print(f"  {sensor['name']}: 检测到 {len(cps)} 个变点 @ {cps}")
    
    return sensors


def example_bayes_factor_analysis():
    """贝叶斯因子分析示例"""
    print("\n" + "="*70)
    print("示例5: 贝叶斯因子分析 - 量化变点证据强度")
    print("="*70)
    
    np.random.seed(999)
    
    # 不同强度的变化
    data_small = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(1, 1, 100)])
    data_medium = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(3, 1, 100)])
    data_large = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(6, 1, 100)])
    
    scenarios = [
        ("微弱变化 (Δμ=1)", data_small),
        ("中等变化 (Δμ=3)", data_medium),
        ("显著变化 (Δμ=6)", data_large)
    ]
    
    for name, data in scenarios:
        detector = OnlineBayesianChangepointDetection(hazard=0.01)
        detector.update_batch(data)
        bf = detector.get_bayes_factor()
        max_prob = max(detector.changepoint_prob_history)
        
        evidence = "无"
        if bf > 100: evidence = "决定性"
        elif bf > 10: evidence = "强烈"
        elif bf > 3: evidence = "实质性"
        elif bf > 1: evidence = "微弱"
        
        print(f"\n{name}:")
        print(f"  最大变点概率: {max_prob:.4f}")
        print(f"  最终贝叶斯因子: {bf:.2f}")
        print(f"  证据强度: {evidence}")
    
    print("\n贝叶斯因子解释:")
    print("  BF < 1:     支持无变点")
    print("  1 < BF < 3: 微弱支持变点")
    print("  3 < BF < 10:实质性支持变点")
    print("  BF > 10:    强烈支持变点")
    print("  BF > 100:   决定性支持变点")


def main():
    print("在线贝叶斯变点检测功能演示")
    print("基于 Bayesian Online Changepoint Detection (Adams & MacKay, 2007)")
    
    # 示例1: 基本使用
    detector1 = example_basic_usage()
    
    # 示例2: 自适应检测
    detector2 = example_adaptive_detection()
    
    # 示例3: 实时监控
    monitor = example_real_time_monitoring()
    
    # 示例4: 多传感器监控
    sensors = example_multiple_sensors()
    
    # 示例5: 贝叶斯因子分析
    example_bayes_factor_analysis()
    
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    print("在线变点检测的优势:")
    print("  ✓ 逐点处理，无需存储全部历史数据")
    print("  ✓ 使用贝叶斯因子量化证据强度")
    print("  ✓ 支持自适应参数调整")
    print("  ✓ 适用于实时流数据监控")
    print("  ✓ 可并行监控多数据流")
    print()
    print("适用场景:")
    print("  • 工业传感器监控")
    print("  • 金融市场异常检测")
    print("  • 网络流量监控")
    print("  • 医疗健康监测")
    print("  • 质量控制过程监控")
    print("="*70)


if __name__ == '__main__':
    main()
