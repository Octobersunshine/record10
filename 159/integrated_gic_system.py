import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from multilayer_earth_model import (
    LayeredEarthModel, Layer, AdvancedGICCalculator,
    Substation, TransmissionLine, generate_storm_profile
)
from magnetic_storm_alert import (
    RealTimeGICAlertSystem, GridDecisionSupportSystem,
    SpaceWeatherSimulator
)


class IntegratedGICSystem:
    def __init__(self):
        self.earth_model = None
        self.grid_calculator = None
        self.alert_system = None
        self.dss = None
        self.historical_data = []
        
    def setup_earth_model_from_mt(self, mt_inversion_result: List[Layer]):
        self.earth_model = LayeredEarthModel(mt_inversion_result)
        print(f"已设置地电阻率模型: {len(mt_inversion_result)}层")
        
    def setup_sample_earth_model(self):
        layers = [
            Layer(thickness=500, resistivity=100),
            Layer(thickness=5000, resistivity=500),
            Layer(thickness=20000, resistivity=10),
            Layer(thickness=float('inf'), resistivity=1000)
        ]
        self.setup_earth_model_from_mt(layers)
        
    def setup_grid(self, substations: List[Substation], lines: List[TransmissionLine]):
        self.grid_calculator = AdvancedGICCalculator(self.earth_model)
        
        for sub in substations:
            self.grid_calculator.add_substation(sub)
            
        for line in lines:
            self.grid_calculator.add_line(line)
            
        self.alert_system = RealTimeGICAlertSystem(self.grid_calculator)
        
        for sub in substations:
            self.alert_system.register_substation(
                sub.id, sub.name, sub.grounding_resistance,
                critical_load=(sub.grounding_resistance < 0.4)
            )
            
        self.dss = GridDecisionSupportSystem(self.alert_system)
        
        print(f"电网配置完成: {len(substations)}个变电站, {len(lines)}条线路")
        
    def setup_sample_grid(self):
        substations = [
            Substation(1, "变电站A", 0, 0, 0.5),
            Substation(2, "变电站B", 100, 0, 0.3),
            Substation(3, "变电站C", 50, 80, 0.4),
            Substation(4, "变电站D", 150, 60, 0.6),
            Substation(5, "变电站E", 200, 100, 0.45)
        ]
        
        lines = [
            TransmissionLine(1, 1, 2, 100, 0.05),
            TransmissionLine(2, 1, 3, 95, 0.048),
            TransmissionLine(3, 2, 3, 90, 0.045),
            TransmissionLine(4, 2, 4, 70, 0.035),
            TransmissionLine(5, 3, 4, 100, 0.05),
            TransmissionLine(6, 4, 5, 80, 0.04),
            TransmissionLine(7, 3, 5, 130, 0.065)
        ]
        
        self.setup_grid(substations, lines)
        
    def run_offline_analysis(self, duration_hours: float = 48.0, 
                              dt_minutes: float = 5.0) -> Dict:
        print(f"\n开始离线GIC分析: {duration_hours}小时, 时间步长{dt_minutes}分钟")
        
        t, dB_dt = generate_storm_profile(duration_hours, dt_minutes)
        
        print("计算大地电场和GIC...")
        gic_results, E_time = self.grid_calculator.calculate_gic(
            t, dB_dt, e_field_angle=45.0, use_fem=False
        )
        
        print("进行空间天气风险评估...")
        base_time = datetime.now()
        sw_simulator = SpaceWeatherSimulator()
        
        assessments = []
        for i in range(len(t)):
            current_time = base_time + timedelta(seconds=t[i])
            sw_data = sw_simulator.generate_data(current_time)
            assessment = self.alert_system.process_data(sw_data)
            assessments.append(assessment)
        
        results = {
            'time': t,
            'dB_dt': dB_dt,
            'electric_field': E_time,
            'gic_results': gic_results,
            'assessments': assessments
        }
        
        self.historical_data.append(results)
        
        self._display_offline_results(results)
        
        return results
    
    def _display_offline_results(self, results: Dict):
        print("\n" + "=" * 80)
        print("离线分析结果摘要")
        print("=" * 80)
        
        t = results['time']
        E_time = results['electric_field']
        gic_results = results['gic_results']
        
        print(f"\n大地电场统计:")
        print(f"  最大值: {np.max(np.abs(E_time)) * 1000:.2f} mV/km")
        print(f"  平均值: {np.mean(np.abs(E_time)) * 1000:.2f} mV/km")
        
        print(f"\n各变电站GIC统计:")
        for sid, gic in gic_results.items():
            sub = self.grid_calculator.substations[sid]
            max_gic = np.max(np.abs(gic))
            print(f"  {sub.name}: 最大GIC = {max_gic:.2f} A, RMS = {np.sqrt(np.mean(gic**2)):.2f} A")
        
        vuln = self.dss.analyze_grid_vulnerability()
        if vuln['high_risk_substations']:
            print(f"\n脆弱性分析 - 高风险变电站:")
            for sub in vuln['high_risk_substations']:
                print(f"  - {sub['name']}: {sub['reason']}")
                print(f"    缓解措施: {sub['mitigation']}")
        
        self._plot_integrated_results(results)
        
    def _plot_integrated_results(self, results: Dict):
        t = results['time'] / 3600
        dB_dt = results['dB_dt']
        E_time = results['electric_field']
        gic_results = results['gic_results']
        assessments = results['assessments']
        
        risk_scores = [a.risk_score for a in assessments]
        
        fig = plt.figure(figsize=(16, 14))
        
        plt.subplot(4, 2, 1)
        plt.plot(t, dB_dt, 'b-', linewidth=1.5)
        plt.ylabel('dB/dt (nT/min)')
        plt.title('地磁场变化率')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(4, 2, 2)
        plt.plot(t, E_time * 1000, 'g-', linewidth=1.5)
        plt.ylabel('地表电场 (mV/km)')
        plt.title('感应地表电场')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(4, 2, 3)
        for sid, gic in gic_results.items():
            sub = self.grid_calculator.substations[sid]
            plt.plot(t, gic, label=sub.name, linewidth=1.5)
        plt.ylabel('GIC (A)')
        plt.title('各变电站GIC')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(4, 2, 4)
        for sid, gic in gic_results.items():
            sub = self.grid_calculator.substations[sid]
            plt.plot(t, np.abs(gic), label=sub.name, linewidth=1.5)
        plt.ylabel('|GIC| (A)')
        plt.title('GIC幅值')
        plt.yscale('log')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(4, 2, 5)
        plt.plot(t, risk_scores, 'm-', linewidth=1.5)
        plt.axhline(y=0.4, color='y', linestyle='--', label='中风险')
        plt.axhline(y=0.6, color='orange', linestyle='--', label='高风险')
        plt.axhline(y=0.8, color='r', linestyle='--', label='严重风险')
        plt.ylabel('风险评分')
        plt.title('GIC风险评分')
        plt.legend()
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        ax6 = plt.subplot(4, 2, 6)
        max_gics = []
        sub_names = []
        for sid, gic in gic_results.items():
            sub = self.grid_calculator.substations[sid]
            max_gics.append(np.max(np.abs(gic)))
            sub_names.append(sub.name)
        
        colors = ['red' if g > 50 else 'orange' if g > 20 else 'green' for g in max_gics]
        ax6.bar(sub_names, max_gics, color=colors)
        ax6.set_ylabel('最大GIC (A)')
        ax6.set_title('各变电站最大GIC')
        ax6.grid(True, alpha=0.3, axis='y')
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)
        
        plt.subplot(4, 1, 4)
        latest_assessment = assessments[-1]
        factors = latest_assessment.contributing_factors
        factor_names = [k.upper() for k in factors.keys()]
        factor_values = list(factors.values())
        colors = ['red' if v > 0.6 else 'orange' if v > 0.3 else 'green' for v in factor_values]
        plt.bar(factor_names, factor_values, color=colors)
        plt.ylabel('贡献度')
        plt.title('当前风险因子贡献分析')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3, axis='y')
        
        for i, (name, value) in enumerate(zip(factor_names, factor_values)):
            plt.text(i, value + 0.02, f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('integrated_gic_analysis.png', dpi=150, bbox_inches='tight')
        print("\n综合分析图已保存为: integrated_gic_analysis.png")
        
    def generate_comprehensive_report(self, results: Dict, output_file: str = "comprehensive_gic_report.txt"):
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("GIC综合分析报告\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("1. 地电阻率模型参数\n")
            f.write("-" * 80 + "\n")
            for i, layer in enumerate(self.earth_model.layers):
                if layer.thickness == float('inf'):
                    f.write(f"  层{i+1}: 半空间, 电阻率 = {layer.resistivity:.1f} Ω·m\n")
                else:
                    f.write(f"  层{i+1}: 厚度 = {layer.thickness:.0f}m, 电阻率 = {layer.resistivity:.1f} Ω·m\n")
            f.write("\n")
            
            f.write("-" * 80 + "\n")
            f.write("2. 电网拓扑信息\n")
            f.write("-" * 80 + "\n")
            f.write(f"变电站数量: {len(self.grid_calculator.substations)}\n")
            for sub in self.grid_calculator.substations.values():
                f.write(f"  {sub.name}: 接地电阻 = {sub.grounding_resistance} Ω\n")
            f.write(f"\n输电线路数量: {len(self.grid_calculator.lines)}\n")
            f.write("\n")
            
            f.write("-" * 80 + "\n")
            f.write("3. GIC计算结果\n")
            f.write("-" * 80 + "\n")
            gic_results = results['gic_results']
            E_time = results['electric_field']
            
            f.write(f"\n大地电场统计:\n")
            f.write(f"  最大值: {np.max(np.abs(E_time)) * 1000:.2f} mV/km\n")
            f.write(f"  平均值: {np.mean(np.abs(E_time)) * 1000:.2f} mV/km\n")
            f.write(f"  均方根值: {np.sqrt(np.mean(E_time**2)) * 1000:.2f} mV/km\n")
            
            f.write(f"\n各变电站GIC统计:\n")
            for sid, gic in gic_results.items():
                sub = self.grid_calculator.substations[sid]
                f.write(f"\n  {sub.name}:\n")
                f.write(f"    最大GIC: {np.max(np.abs(gic)):.2f} A\n")
                f.write(f"    最小GIC: {np.min(gic):.2f} A\n")
                f.write(f"    平均GIC: {np.mean(np.abs(gic)):.2f} A\n")
                f.write(f"    均方根GIC: {np.sqrt(np.mean(gic**2)):.2f} A\n")
            
            f.write("\n" + "-" * 80 + "\n")
            f.write("4. 风险评估与建议\n")
            f.write("-" * 80 + "\n")
            
            vuln = self.dss.analyze_grid_vulnerability()
            if vuln['high_risk_substations']:
                f.write("\n脆弱性分析 - 高风险变电站:\n")
                for sub in vuln['high_risk_substations']:
                    f.write(f"\n  {sub['name']}:\n")
                    f.write(f"    原因: {sub['reason']}\n")
                    f.write(f"    缓解措施: {sub['mitigation']}\n")
            
            latest_assessment = results['assessments'][-1]
            f.write(f"\n当前风险等级: {latest_assessment.risk_level.value}\n")
            f.write(f"风险评分: {latest_assessment.risk_score:.2f} / 1.0\n")
            
            f.write(f"\n建议措施:\n")
            for i, action in enumerate(latest_assessment.recommended_actions, 1):
                f.write(f"  {i}. {action}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("报告结束\n")
            f.write("=" * 80 + "\n")
        
        print(f"综合报告已生成: {output_file}")


def main():
    print("=" * 80)
    print("GIC综合分析系统 - 集成MT地电阻率模型+实时预警")
    print("=" * 80)
    
    system = IntegratedGICSystem()
    
    print("\n步骤1: 设置地电阻率模型")
    system.setup_sample_earth_model()
    
    print("\n步骤2: 配置电网模型")
    system.setup_sample_grid()
    
    print("\n步骤3: 运行离线GIC分析")
    results = system.run_offline_analysis(duration_hours=48.0, dt_minutes=5.0)
    
    print("\n步骤4: 生成综合报告")
    system.generate_comprehensive_report(results)
    
    print("\n" + "=" * 80)
    print("系统功能说明")
    print("=" * 80)
    print("\n1. 多层地电阻率模型:")
    print("   - 支持MT测深数据反演")
    print("   - 传播矩阵法计算地表阻抗")
    print("   - 有限元法求解2D大地电场")
    
    print("\n2. GIC计算模块:")
    print("   - 频域-时域转换计算时变电场")
    print("   - 节点导纳矩阵法求解电网GIC")
    print("   - 考虑输电线路走向与电场夹角")
    
    print("\n3. 磁暴预警系统:")
    print("   - 多因子风险评估 (Dst, Bz, 太阳风速度, 密度, Kp, AE)")
    print("   - 分级预警机制 (无/低/中/高/严重风险)")
    print("   - 受影响变电站识别")
    print("   - 实时监测与告警回调")
    
    print("\n4. 决策支持系统:")
    print("   - 电网脆弱性分析")
    print("   - 分级调度建议")
    print("   - 应急预案准备")
    print("   - 综合报告生成")
    
    print("\n" + "=" * 80)
    print("所有分析已完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
