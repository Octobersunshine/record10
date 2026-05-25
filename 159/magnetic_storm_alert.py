import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum
from datetime import datetime, timedelta
import threading
import time
from collections import deque
import json


class StormLevel(Enum):
    QUIET = "平静"
    MINOR = "小磁暴"
    MODERATE = "中等磁暴"
    STRONG = "强磁暴"
    SEVERE = "烈磁暴"
    EXTREME = "特大磁暴"


class RiskLevel(Enum):
    NONE = "无风险"
    LOW = "低风险"
    MEDIUM = "中风险"
    HIGH = "高风险"
    CRITICAL = "严重风险"


@dataclass
class SpaceWeatherData:
    timestamp: datetime
    dst_index: float
    solar_wind_speed: float
    solar_wind_density: float
    bz: float
    by: float
    bx: float
    kp_index: float
    ae_index: float


@dataclass
class GICRiskAssessment:
    timestamp: datetime
    risk_level: RiskLevel
    max_gic_predicted: float
    risk_score: float
    contributing_factors: Dict[str, float]
    recommended_actions: List[str]
    affected_substations: List[Tuple[str, float]]


@dataclass
class AlertThreshold:
    dst_warning: float = -50.0
    dst_critical: float = -100.0
    bz_warning: float = -5.0
    bz_critical: float = -10.0
    speed_warning: float = 500.0
    speed_critical: float = 700.0
    density_warning: float = 5.0
    density_critical: float = 10.0
    gic_warning: float = 10.0
    gic_critical: float = 50.0


class SpaceWeatherSimulator:
    def __init__(self):
        self.current_time = datetime.now()
        self.base_dst = 0.0
        self.storm_active = False
        self.storm_start_time = None
        self.storm_intensity = 0.0
        
    def generate_data(self, current_time: Optional[datetime] = None) -> SpaceWeatherData:
        if current_time is None:
            current_time = self.current_time
            
        t = (current_time - self.current_time).total_seconds() / 3600.0
        
        if not self.storm_active and np.random.random() < 0.01:
            self.storm_active = True
            self.storm_start_time = current_time
            self.storm_intensity = np.random.uniform(0.3, 1.0)
        
        dst = self.base_dst
        bz = np.random.normal(0, 2)
        speed = np.random.normal(350, 50)
        density = np.random.normal(3, 1)
        kp = np.random.uniform(0, 3)
        ae = np.random.normal(100, 50)
        
        if self.storm_active:
            storm_hours = (current_time - self.storm_start_time).total_seconds() / 3600.0
            
            if storm_hours < 6:
                dst = -200 * self.storm_intensity * (1 - np.exp(-storm_hours / 1.5))
                bz = -15 * self.storm_intensity * (1 - np.exp(-storm_hours / 1.0)) + np.random.normal(0, 3)
                speed = 350 + 400 * self.storm_intensity * (1 - np.exp(-storm_hours / 2.0))
                density = 3 + 15 * self.storm_intensity * (1 - np.exp(-storm_hours / 1.0))
                kp = 5 + 4 * self.storm_intensity * (1 - np.exp(-storm_hours / 2.0))
                ae = 500 + 1500 * self.storm_intensity * (1 - np.exp(-storm_hours / 1.0))
            elif storm_hours < 24:
                recovery = storm_hours - 6
                dst = -200 * self.storm_intensity * np.exp(-recovery / 8.0)
                bz = 5 * np.sin(2 * np.pi * storm_hours / 4.0) + np.random.normal(0, 2)
                speed = 350 + 400 * self.storm_intensity * np.exp(-recovery / 10.0)
                density = 3 + 10 * self.storm_intensity * np.exp(-recovery / 5.0)
                kp = 3 + 4 * self.storm_intensity * np.exp(-recovery / 6.0)
                ae = 300 + 800 * self.storm_intensity * np.exp(-recovery / 4.0)
            else:
                self.storm_active = False
                self.storm_start_time = None
        
        return SpaceWeatherData(
            timestamp=current_time,
            dst_index=dst + np.random.normal(0, 5),
            solar_wind_speed=speed + np.random.normal(0, 20),
            solar_wind_density=max(0.1, density + np.random.normal(0, 0.5)),
            bz=bz,
            by=np.random.normal(0, 2),
            bx=np.random.normal(0, 1),
            kp_index=min(9, max(0, kp + np.random.normal(0, 0.3))),
            ae_index=max(0, ae + np.random.normal(0, 30))
        )


class GICRiskModel:
    def __init__(self, threshold: AlertThreshold = None):
        self.threshold = threshold or AlertThreshold()
        self.weights = {
            'dst': 0.30,
            'bz': 0.25,
            'speed': 0.20,
            'density': 0.10,
            'kp': 0.10,
            'ae': 0.05
        }
        
    def calculate_risk_score(self, sw_data: SpaceWeatherData) -> Tuple[float, Dict[str, float]]:
        scores = {}
        
        scores['dst'] = self._normalize_dst(sw_data.dst_index)
        scores['bz'] = self._normalize_bz(sw_data.bz)
        scores['speed'] = self._normalize_speed(sw_data.solar_wind_speed)
        scores['density'] = self._normalize_density(sw_data.solar_wind_density)
        scores['kp'] = self._normalize_kp(sw_data.kp_index)
        scores['ae'] = self._normalize_ae(sw_data.ae_index)
        
        total_score = sum(scores[k] * self.weights[k] for k in self.weights)
        
        return total_score, scores
    
    def _normalize_dst(self, dst: float) -> float:
        if dst >= -30:
            return 0.0
        elif dst >= -100:
            return (abs(dst) - 30) / 70.0
        elif dst >= -300:
            return 0.7 + 0.3 * (abs(dst) - 100) / 200.0
        else:
            return 1.0
    
    def _normalize_bz(self, bz: float) -> float:
        if bz >= 0:
            return 0.0
        elif bz >= -5:
            return abs(bz) / 10.0
        elif bz >= -20:
            return 0.5 + 0.5 * (abs(bz) - 5) / 15.0
        else:
            return 1.0
    
    def _normalize_speed(self, speed: float) -> float:
        if speed <= 350:
            return 0.0
        elif speed <= 700:
            return (speed - 350) / 700.0
        elif speed <= 1000:
            return 0.5 + 0.5 * (speed - 700) / 300.0
        else:
            return 1.0
    
    def _normalize_density(self, density: float) -> float:
        if density <= 3:
            return 0.0
        elif density <= 20:
            return (density - 3) / 34.0
        elif density <= 50:
            return 0.5 + 0.5 * (density - 20) / 30.0
        else:
            return 1.0
    
    def _normalize_kp(self, kp: float) -> float:
        return kp / 9.0
    
    def _normalize_ae(self, ae: float) -> float:
        if ae <= 200:
            return 0.0
        elif ae <= 1000:
            return (ae - 200) / 1600.0
        elif ae <= 3000:
            return 0.5 + 0.5 * (ae - 1000) / 2000.0
        else:
            return 1.0
    
    def predict_max_gic(self, sw_data: SpaceWeatherData, 
                         base_ground_conductivity: float = 0.01) -> float:
        epsilon = 1e-10
        dB_dt_est = abs(sw_data.bz) * sw_data.solar_wind_speed / 1000.0
        gic_est = dB_dt_est * (1.0 / (base_ground_conductivity + epsilon)) * 0.1
        dst_factor = max(0, abs(sw_data.dst_index) / 100.0)
        kp_factor = sw_data.kp_index / 5.0
        density_factor = sw_data.solar_wind_density / 5.0
        
        total_gic = gic_est * (1 + 0.5 * dst_factor + 0.3 * kp_factor + 0.2 * density_factor)
        
        return min(total_gic, 200.0)
    
    def get_risk_level(self, risk_score: float) -> RiskLevel:
        if risk_score < 0.2:
            return RiskLevel.NONE
        elif risk_score < 0.4:
            return RiskLevel.LOW
        elif risk_score < 0.6:
            return RiskLevel.MEDIUM
        elif risk_score < 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def get_recommended_actions(self, risk_level: RiskLevel, 
                                  sw_data: SpaceWeatherData) -> List[str]:
        actions = []
        
        if risk_level == RiskLevel.NONE:
            actions.append("电网正常运行，持续监测空间天气")
        elif risk_level == RiskLevel.LOW:
            actions.append("加强空间天气数据监测频率")
            actions.append("通知电网调度人员注意")
        elif risk_level == RiskLevel.MEDIUM:
            actions.append("启动GIC监测预警系统")
            actions.append("检查关键变电站变压器中性点运行状态")
            actions.append("准备备用无功补偿设备")
        elif risk_level == RiskLevel.HIGH:
            actions.append("发布电网GIC预警")
            actions.append("限制长距离输电线路传输功率")
            actions.append("增加变压器中性点监测频次")
            actions.append("准备应急响应预案")
        elif risk_level == RiskLevel.CRITICAL:
            actions.append("发布电网GIC紧急警报")
            actions.append("降低关键输电线路负载至70%以下")
            actions.append("启动电网应急响应机制")
            actions.append("通知所有相关场站加强值守")
            actions.append("准备隔离受影响设备")
        
        if sw_data.dst_index < self.threshold.dst_critical:
            actions.append("Dst指数过低，特别注意变压器直流偏磁")
        if sw_data.bz < self.threshold.bz_critical:
            actions.append("Bz南向分量大，感应电场强")
        if sw_data.solar_wind_speed > self.threshold.speed_critical:
            actions.append("太阳风速度过高，警惕持续影响")
        
        return actions


class RealTimeGICAlertSystem:
    def __init__(self, grid_calculator=None):
        self.risk_model = GICRiskModel()
        self.space_weather_simulator = SpaceWeatherSimulator()
        self.grid_calculator = grid_calculator
        self.data_history = deque(maxlen=288)
        self.risk_history = deque(maxlen=288)
        self.alert_callbacks = []
        self.is_running = False
        self.update_interval = 300
        self.substations_info = {}
        
    def register_substation(self, substation_id: int, name: str, 
                             grounding_resistance: float, 
                             critical_load: bool = False):
        self.substations_info[substation_id] = {
            'name': name,
            'grounding_resistance': grounding_resistance,
            'critical_load': critical_load
        }
    
    def add_alert_callback(self, callback: Callable[[GICRiskAssessment], None]):
        self.alert_callbacks.append(callback)
    
    def process_data(self, sw_data: SpaceWeatherData) -> GICRiskAssessment:
        risk_score, factors = self.risk_model.calculate_risk_score(sw_data)
        risk_level = self.risk_model.get_risk_level(risk_score)
        max_gic = self.risk_model.predict_max_gic(sw_data)
        
        affected_substations = []
        for sid, info in self.substations_info.items():
            substation_gic = max_gic * (0.5 / info['grounding_resistance'])
            if substation_gic > self.risk_model.threshold.gic_warning:
                affected_substations.append((info['name'], substation_gic))
        
        affected_substations.sort(key=lambda x: x[1], reverse=True)
        
        actions = self.risk_model.get_recommended_actions(risk_level, sw_data)
        
        assessment = GICRiskAssessment(
            timestamp=sw_data.timestamp,
            risk_level=risk_level,
            max_gic_predicted=max_gic,
            risk_score=risk_score,
            contributing_factors=factors,
            recommended_actions=actions,
            affected_substations=affected_substations[:5]
        )
        
        self.data_history.append(sw_data)
        self.risk_history.append(assessment)
        
        for callback in self.alert_callbacks:
            callback(assessment)
        
        return assessment
    
    def start(self):
        self.is_running = True
        print("实时GIC预警系统已启动...")
        print(f"更新间隔: {self.update_interval}秒")
        
        while self.is_running:
            try:
                current_time = datetime.now()
                sw_data = self.space_weather_simulator.generate_data(current_time)
                assessment = self.process_data(sw_data)
                
                self._print_status(assessment)
                
                time.sleep(self.update_interval)
            except KeyboardInterrupt:
                print("\n正在停止预警系统...")
                self.stop()
                break
            except Exception as e:
                print(f"处理错误: {e}")
                time.sleep(self.update_interval)
    
    def stop(self):
        self.is_running = False
        print("实时GIC预警系统已停止")
    
    def _print_status(self, assessment: GICRiskAssessment):
        print("\n" + "=" * 70)
        print(f"GIC实时预警 - {assessment.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        risk_colors = {
            RiskLevel.NONE: "\033[92m",
            RiskLevel.LOW: "\033[94m",
            RiskLevel.MEDIUM: "\033[93m",
            RiskLevel.HIGH: "\033[91m",
            RiskLevel.CRITICAL: "\033[95m"
        }
        reset_color = "\033[0m"
        
        color = risk_colors.get(assessment.risk_level, "")
        print(f"风险等级: {color}{assessment.risk_level.value}{reset_color}")
        print(f"风险评分: {assessment.risk_score:.2f} / 1.0")
        print(f"预测最大GIC: {assessment.max_gic_predicted:.2f} A")
        
        print("\n风险贡献因子:")
        for factor, score in sorted(assessment.contributing_factors.items(), 
                                     key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 30)
            print(f"  {factor.upper():8s}: {bar} {score:.2f}")
        
        if assessment.affected_substations:
            print("\n受影响变电站:")
            for name, gic in assessment.affected_substations:
                print(f"  {name}: {gic:.2f} A")
        
        print("\n建议措施:")
        for i, action in enumerate(assessment.recommended_actions[:5], 1):
            print(f"  {i}. {action}")
        
        print("=" * 70)
    
    def generate_report(self, output_file: str = "gic_risk_report.json"):
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_records': len(self.risk_history),
            'max_risk_score': max(a.risk_score for a in self.risk_history),
            'max_predicted_gic': max(a.max_gic_predicted for a in self.risk_history),
            'risk_levels_count': {},
            'recommendations_summary': []
        }
        
        for assessment in self.risk_history:
            level = assessment.risk_level.value
            report['risk_levels_count'][level] = report['risk_levels_count'].get(level, 0) + 1
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"风险报告已生成: {output_file}")


class GridDecisionSupportSystem:
    def __init__(self, alert_system: RealTimeGICAlertSystem):
        self.alert_system = alert_system
        self.alert_history = []
        self.action_taken = []
        
    def analyze_grid_vulnerability(self) -> Dict:
        vulnerabilities = {
            'high_risk_substations': [],
            'critical_lines': [],
            'recommendations': []
        }
        
        for sid, info in self.alert_system.substations_info.items():
            if info['grounding_resistance'] < 0.3:
                vulnerabilities['high_risk_substations'].append({
                    'id': sid,
                    'name': info['name'],
                    'reason': '接地电阻较低',
                    'mitigation': '考虑增加临时接地电阻或加强监测'
                })
            if info['critical_load']:
                vulnerabilities['high_risk_substations'].append({
                    'id': sid,
                    'name': info['name'],
                    'reason': '承载关键负荷',
                    'mitigation': '准备备用电源方案'
                })
        
        return vulnerabilities
    
    def generate_decision_support(self, assessment: GICRiskAssessment) -> Dict:
        decision = {
            'timestamp': assessment.timestamp.isoformat(),
            'risk_level': assessment.risk_level.value,
            'risk_score': assessment.risk_score,
            'max_gic': assessment.max_gic_predicted,
            'operational_actions': [],
            'monitoring_actions': [],
            'emergency_preparedness': []
        }
        
        if assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            decision['operational_actions'] = [
                "降低长距离输电线路功率至额定值的70-80%",
                "投入备用无功补偿设备",
                "调整变压器分接头以应对电压波动"
            ]
            decision['monitoring_actions'] = [
                "增加变压器振动和噪声监测",
                "监测变压器中性点电流波形",
                "检查继电保护整定值"
            ]
            decision['emergency_preparedness'] = [
                "应急抢修队伍待命",
                "关键备件准备就绪",
                "与上级调度保持密切联系"
            ]
        
        return decision
    
    def display_dashboard(self):
        print("\n" + "=" * 80)
        print("电网调度GIC决策支持系统")
        print("=" * 80)
        
        if self.alert_system.risk_history:
            latest = self.alert_system.risk_history[-1]
            sw_data = self.alert_system.data_history[-1]
            
            print(f"\n当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"\n空间天气参数:")
            print(f"  Dst指数: {sw_data.dst_index:.1f} nT")
            print(f"  Bz分量: {sw_data.bz:.1f} nT")
            print(f"  太阳风速度: {sw_data.solar_wind_speed:.1f} km/s")
            print(f"  太阳风密度: {sw_data.solar_wind_density:.1f} p/cc")
            print(f"  Kp指数: {sw_data.kp_index:.1f}")
            print(f"  AE指数: {sw_data.ae_index:.1f} nT")
            
            print(f"\nGIC风险评估:")
            print(f"  风险等级: {latest.risk_level.value}")
            print(f"  风险评分: {latest.risk_score:.2f}")
            print(f"  预测最大GIC: {latest.max_gic_predicted:.2f} A")
            
            vuln = self.analyze_grid_vulnerability()
            if vuln['high_risk_substations']:
                print(f"\n脆弱性分析 - 高风险变电站:")
                for sub in vuln['high_risk_substations']:
                    print(f"  - {sub['name']}: {sub['reason']}")
            
            decision = self.generate_decision_support(latest)
            if decision['operational_actions']:
                print(f"\n调度建议 - 运行操作:")
                for action in decision['operational_actions']:
                    print(f"  → {action}")
            
            print("\n" + "=" * 80)


def main():
    print("=" * 80)
    print("磁暴GIC预警与电网调度决策支持系统")
    print("=" * 80)
    
    alert_system = RealTimeGICAlertSystem()
    
    alert_system.register_substation(1, "变电站A", 0.5, critical_load=True)
    alert_system.register_substation(2, "变电站B", 0.3, critical_load=True)
    alert_system.register_substation(3, "变电站C", 0.4, critical_load=False)
    alert_system.register_substation(4, "变电站D", 0.6, critical_load=False)
    alert_system.register_substation(5, "变电站E", 0.45, critical_load=True)
    
    dss = GridDecisionSupportSystem(alert_system)
    
    print("\n生成模拟磁暴事件进行测试...")
    test_times = []
    test_data = []
    base_time = datetime.now()
    
    for i in range(48):
        current_time = base_time + timedelta(minutes=i * 30)
        sw_data = alert_system.space_weather_simulator.generate_data(current_time)
        assessment = alert_system.process_data(sw_data)
        test_times.append(current_time)
        test_data.append((sw_data, assessment))
    
    print(f"已生成 {len(test_data)} 条模拟数据")
    
    dss.display_dashboard()
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    times = [t for t, _ in test_data]
    dst_values = [d.dst_index for d, _ in test_data]
    bz_values = [d.bz for d, _ in test_data]
    speed_values = [d.solar_wind_speed for d, _ in test_data]
    risk_scores = [a.risk_score for _, a in test_data]
    gic_values = [a.max_gic_predicted for _, a in test_data]
    
    ax = axes[0, 0]
    ax.plot(times, dst_values, 'b-', linewidth=2)
    ax.axhline(y=-50, color='y', linestyle='--', label='预警阈值')
    ax.axhline(y=-100, color='r', linestyle='--', label='危急阈值')
    ax.set_ylabel('Dst (nT)')
    ax.set_title('Dst指数变化')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    ax = axes[0, 1]
    ax.plot(times, bz_values, 'r-', linewidth=2)
    ax.axhline(y=-5, color='y', linestyle='--')
    ax.axhline(y=-10, color='r', linestyle='--')
    ax.set_ylabel('Bz (nT)')
    ax.set_title('IMF Bz分量')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    ax = axes[1, 0]
    ax.plot(times, speed_values, 'g-', linewidth=2)
    ax.axhline(y=500, color='y', linestyle='--')
    ax.axhline(y=700, color='r', linestyle='--')
    ax.set_ylabel('太阳风速度 (km/s)')
    ax.set_title('太阳风速度')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    ax = axes[1, 1]
    ax.plot(times, risk_scores, 'm-', linewidth=2)
    ax.axhline(y=0.4, color='y', linestyle='--', label='中风险')
    ax.axhline(y=0.6, color='orange', linestyle='--', label='高风险')
    ax.axhline(y=0.8, color='r', linestyle='--', label='严重风险')
    ax.set_ylabel('风险评分')
    ax.set_title('GIC风险评分')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    ax = axes[2, 0]
    ax.plot(times, gic_values, 'b-', linewidth=2)
    ax.axhline(y=10, color='y', linestyle='--', label='预警')
    ax.axhline(y=50, color='r', linestyle='--', label='危急')
    ax.set_ylabel('预测GIC (A)')
    ax.set_title('预测最大GIC')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    ax = axes[2, 1]
    factor_names = ['Dst', 'Bz', 'Speed', 'Density', 'Kp', 'AE']
    if test_data:
        factors = list(test_data[-1][1].contributing_factors.values())
        colors = ['red' if f > 0.6 else 'orange' if f > 0.3 else 'green' for f in factors]
        ax.bar(factor_names, factors, color=colors)
    ax.set_ylabel('贡献度')
    ax.set_title('风险因子贡献')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('magnetic_storm_alert_dashboard.png', dpi=150, bbox_inches='tight')
    print("\n预警仪表板图已保存为: magnetic_storm_alert_dashboard.png")
    
    alert_system.generate_report("gic_risk_report.json")
    
    print("\n" + "=" * 80)
    print("系统测试完成!")
    print("=" * 80)
    print("\n主要功能:")
    print("  1. 空间天气数据模拟与处理 (Dst, Bz, 太阳风参数等)")
    print("  2. 多因子GIC风险评估模型")
    print("  3. 变电站脆弱性分析")
    print("  4. 分级预警与调度决策支持")
    print("  5. 实时监测与告警回调机制")
    print("  6. 风险报告生成")


if __name__ == "__main__":
    main()
