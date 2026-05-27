import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from battery_thermal_simulation import BatteryPackSimulator, SEIAgingModel, ThermalManagementOptimizer

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("=" * 70)
    print("  锂离子电池组热-电-老化耦合仿真")
    print("  可视化结果生成")
    print("=" * 70)

    # 1. 风冷 vs 液冷温度分布对比
    print("\n[1/4] 生成风冷 vs 液冷温度分布对比图...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sim_air = BatteryPackSimulator(
        num_cells=(4, 4),
        cell_size=(0.02, 0.1, 0.15),
        cooling_type='air',
        cell_capacity=50.0,
        discharge_current=50.0,
        enable_aging=True
    )
    time_air, max_temp_air, min_temp_air = sim_air.run_simulation(
        total_time=3600.0, T_ambient=298.15
    )

    sim_liquid = BatteryPackSimulator(
        num_cells=(4, 4),
        cell_size=(0.02, 0.1, 0.15),
        cooling_type='liquid',
        cell_capacity=50.0,
        discharge_current=50.0,
        enable_aging=True
    )
    time_liquid, max_temp_liquid, min_temp_liquid = sim_liquid.run_simulation(
        total_time=3600.0, T_ambient=298.15
    )

    ax = axes[0, 0]
    ax.plot(np.array(time_air) / 60.0, np.array(max_temp_air) - 273.15, 'r-', label='最高温度')
    ax.plot(np.array(time_air) / 60.0, np.array(min_temp_air) - 273.15, 'b-', label='最低温度')
    ax.fill_between(np.array(time_air) / 60.0,
                    np.array(min_temp_air) - 273.15,
                    np.array(max_temp_air) - 273.15, alpha=0.3, color='orange')
    ax.set_xlabel('时间 (min)')
    ax.set_ylabel('温度 (°C)')
    ax.set_title('风冷条件下温度分布')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(np.array(time_liquid) / 60.0, np.array(max_temp_liquid) - 273.15, 'r-', label='最高温度')
    ax.plot(np.array(time_liquid) / 60.0, np.array(min_temp_liquid) - 273.15, 'b-', label='最低温度')
    ax.fill_between(np.array(time_liquid) / 60.0,
                    np.array(min_temp_liquid) - 273.15,
                    np.array(max_temp_liquid) - 273.15, alpha=0.3, color='orange')
    ax.set_xlabel('时间 (min)')
    ax.set_ylabel('温度 (°C)')
    ax.set_title('液冷条件下温度分布')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(np.array(time_air) / 60.0, np.array(max_temp_air) - 273.15, 'r-', label='风冷-最高')
    ax.plot(np.array(time_air) / 60.0, np.array(min_temp_air) - 273.15, 'r--', label='风冷-最低')
    ax.plot(np.array(time_liquid) / 60.0, np.array(max_temp_liquid) - 273.15, 'b-', label='液冷-最高')
    ax.plot(np.array(time_liquid) / 60.0, np.array(min_temp_liquid) - 273.15, 'b--', label='液冷-最低')
    ax.set_xlabel('时间 (min)')
    ax.set_ylabel('温度 (°C)')
    ax.set_title('风冷 vs 液冷温度对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    spread_air = np.array(max_temp_air) - np.array(min_temp_air)
    spread_liquid = np.array(max_temp_liquid) - np.array(min_temp_liquid)
    ax.plot(np.array(time_air) / 60.0, spread_air, 'r-', label='风冷温差')
    ax.plot(np.array(time_liquid) / 60.0, spread_liquid, 'b-', label='液冷温差')
    ax.set_xlabel('时间 (min)')
    ax.set_ylabel('温差 (K)')
    ax.set_title('风冷 vs 液冷温差对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cooling_comparison.png', dpi=150)
    plt.close()
    print("  ✓ 已保存 cooling_comparison.png")

    # 2. SEI膜生长与容量衰减
    print("\n[2/4] 生成SEI膜生长与容量衰减压...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    temperatures = [-20, -10, 0, 10, 20, 25, 30, 40, 50, 60]
    cycle_lives = []
    calendar_lives = []

    for T in temperatures:
        aging_model = SEIAgingModel(
            nominal_capacity=50.0,
            Ea_sei=30000.0,
            k_sei=0.05,
            alpha_capacity=0.001,
            alpha_resistance=0.002
        )
        result = aging_model.predict_cycle_life(
            T_stored=T + 273.15,
            capacity_fade_threshold=0.2
        )
        cycle_lives.append(result['cycle_life'])
        calendar_lives.append(result['calendar_life_years'])

    ax = axes[0, 0]
    ax.plot(temperatures, calendar_lives, 'bo-', linewidth=2, markersize=8)
    ax.fill_between(temperatures, 0, calendar_lives, alpha=0.3, color='blue')
    ax.set_xlabel('存储温度 (°C)')
    ax.set_ylabel('日历寿命 (年)')
    ax.set_title('日历寿命与存储温度的关系')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=10, color='r', linestyle='--', alpha=0.7, label='10年基准线')
    ax.legend()

    ax = axes[0, 1]
    ax.plot(temperatures, cycle_lives, 'ro-', linewidth=2, markersize=8)
    ax.fill_between(temperatures, 0, cycle_lives, alpha=0.3, color='red')
    ax.set_xlabel('存储温度 (°C)')
    ax.set_ylabel('循环寿命 (次)')
    ax.set_title('循环寿命与存储温度的关系')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    T_arrhenius = np.linspace(-30, 70, 100)
    k_sei = 0.05
    Ea_sei = 30000.0
    R_gas = 8.314
    arrhenius_factor = np.exp(-Ea_sei / (R_gas * (T_arrhenius + 273.15)))
    growth_rate = k_sei * arrhenius_factor * 1.25 * 86400 * 365
    ax.semilogy(T_arrhenius, growth_rate, 'g-', linewidth=2)
    ax.set_xlabel('温度 (°C)')
    ax.set_ylabel('SEI生长速率 (nm/year)')
    ax.set_title('SEI膜生长速率与温度的关系 (Arrhenius)')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    years = np.linspace(0, 30, 100)
    for T in [0, 25, 40, 60]:
        arrhenius_factor = np.exp(-Ea_sei / (R_gas * (T + 273.15)))
        growth_rate = k_sei * arrhenius_factor * 1.25
        sei_thickness = growth_rate * years * 365 * 86400
        capacity_fade = 0.001 * sei_thickness * 100
        ax.plot(years, capacity_fade, linewidth=2, label=f'{T}°C')
    ax.set_xlabel('时间 (年)')
    ax.set_ylabel('容量衰减 (%)')
    ax.set_title('不同温度下容量衰减随时间的变化')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=20, color='r', linestyle='--', alpha=0.7, label='20%阈值')

    plt.tight_layout()
    plt.savefig('aging_prediction.png', dpi=150)
    plt.close()
    print("  ✓ 已保存 aging_prediction.png")

    # 3. 热-电-老化耦合效果
    print("\n[3/4] 生成热-电-老化耦合效果...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sim_coupled = BatteryPackSimulator(
        num_cells=(4, 4),
        cell_size=(0.02, 0.1, 0.15),
        cooling_type='air',
        cell_capacity=50.0,
        discharge_current=50.0,
        enable_aging=True
    )
    time_coupled, max_temp_coupled, min_temp_coupled = sim_coupled.run_simulation(
        total_time=3600.0, T_ambient=298.15
    )

    sim_uncoupled = BatteryPackSimulator(
        num_cells=(4, 4),
        cell_size=(0.02, 0.1, 0.15),
        cooling_type='air',
        cell_capacity=50.0,
        discharge_current=50.0,
        enable_aging=False
    )
    time_uncoupled, max_temp_uncoupled, min_temp_uncoupled = sim_uncoupled.run_simulation(
        total_time=3600.0, T_ambient=298.15
    )

    ax = axes[0, 0]
    ax.plot(np.array(time_coupled) / 60.0, np.array(max_temp_coupled) - 273.15, 'r-', label='耦合-最高')
    ax.plot(np.array(time_coupled) / 60.0, np.array(min_temp_coupled) - 273.15, 'r--', label='耦合-最低')
    ax.plot(np.array(time_uncoupled) / 60.0, np.array(max_temp_uncoupled) - 273.15, 'b-', label='非耦合-最高')
    ax.plot(np.array(time_uncoupled) / 60.0, np.array(min_temp_uncoupled) - 273.15, 'b--', label='非耦合-最低')
    ax.set_xlabel('时间 (min)')
    ax.set_ylabel('温度 (°C)')
    ax.set_title('热-电-老化耦合 vs 非耦合温度对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if sim_coupled.aging_model is not None:
        state = sim_coupled.aging_model.get_state()
        capacity_fade_coupled = state['capacity_fade'] * 100
        resistance_growth_coupled = state['resistance_growth'] * 100
        ax.bar(['容量衰减', '内阻增长'],
               [capacity_fade_coupled, resistance_growth_coupled],
               color=['#ff6b6b', '#4ecdc4'], alpha=0.8)
        ax.set_ylabel('变化量 (%)')
        ax.set_title('1小时放电后的老化状态')
        for i, v in enumerate([capacity_fade_coupled, resistance_growth_coupled]):
            ax.text(i, v + 0.00005, f'{v:.4f}%', ha='center', va='bottom')
        ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 0]
    if sim_coupled.aging_model is not None:
        time_hist = sim_coupled.aging_model.history_time
        temp_hist = sim_coupled.aging_model.history_temperature
        fade_hist = sim_coupled.aging_model.history_capacity_fade
        if len(time_hist) > 1:
            ax.plot(np.array(time_hist) / 60.0, np.array(temp_hist) - 273.15, 'r-', label='平均温度')
            ax.set_xlabel('时间 (min)')
            ax.set_ylabel('温度 (°C)', color='r')
            ax.tick_params(axis='y', labelcolor='r')
            ax2 = ax.twinx()
            ax2.plot(np.array(time_hist) / 60.0, np.array(fade_hist) * 100, 'b-', label='容量衰减')
            ax2.set_ylabel('容量衰减 (%)', color='b')
            ax2.tick_params(axis='y', labelcolor='b')
            ax.set_title('温度与容量衰减的耦合关系')
            ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    soc_values = np.linspace(0, 1, 100)
    temp_values = [0, 25, 40, 60]
    for T in temp_values:
        resistance = []
        for soc in soc_values:
            r = sim_coupled.heat_model._lookup_resistance(soc, T + 273.15)
            resistance.append(r * 1000)
        ax.plot(soc_values * 100, resistance, linewidth=2, label=f'{T}°C')
    ax.set_xlabel('SOC (%)')
    ax.set_ylabel('内阻 (mΩ)')
    ax.set_title('内阻与SOC和温度的关系')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('coupling_effects.png', dpi=150)
    plt.close()
    print("  ✓ 已保存 coupling_effects.png")

    # 4. 热管理策略优化
    print("\n[4/4] 生成热管理策略优化结果...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    optimizer = ThermalManagementOptimizer()

    strategies = {
        '弱风冷': {'cooling_type': 'air', 'h_conv': 25.0, 'T_coolant': 298.15},
        '标准风冷': {'cooling_type': 'air', 'h_conv': 50.0, 'T_coolant': 298.15},
        '强制风冷': {'cooling_type': 'air', 'h_conv': 100.0, 'T_coolant': 298.15},
        '弱液冷': {'cooling_type': 'liquid', 'h_conv': 250.0, 'T_coolant': 288.15},
        '标准液冷': {'cooling_type': 'liquid', 'h_conv': 500.0, 'T_coolant': 288.15},
        '强化液冷': {'cooling_type': 'liquid', 'h_conv': 1000.0, 'T_coolant': 283.15}
    }

    strategy_results = {}
    for name, params in strategies.items():
        print(f"\n  评估策略: {name}")
        result = optimizer.evaluate_strategy(
            cooling_type=params['cooling_type'],
            h_conv=params['h_conv'],
            T_coolant=params['T_coolant'],
            discharge_current=50.0,
            total_time=3600.0,
            T_ambient=298.15
        )
        strategy_results[name] = result

    names = list(strategy_results.keys())
    T_max_list = [strategy_results[n]['T_max'] for n in names]
    T_avg_list = [strategy_results[n]['T_avg'] for n in names]
    T_spread_list = [strategy_results[n]['T_spread'] for n in names]

    ax = axes[0, 0]
    x = np.arange(len(names))
    width = 0.25
    ax.bar(x - width, T_max_list, width, label='最高温度', color='#ff6b6b', alpha=0.8)
    ax.bar(x, T_avg_list, width, label='平均温度', color='#4ecdc4', alpha=0.8)
    ax.bar(x + width, T_spread_list, width, label='温差', color='#ffe66d', alpha=0.8)
    ax.set_xlabel('冷却策略')
    ax.set_ylabel('温度 (°C/K)')
    ax.set_title('不同冷却策略的温度指标对比')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[0, 1]
    h_air = [25, 50, 100]
    h_liquid = [250, 500, 1000]
    T_max_air = []
    T_max_liquid = []
    
    for h in h_air:
        result = optimizer.evaluate_strategy(
            cooling_type='air', h_conv=h, T_coolant=298.15,
            discharge_current=50.0,
            total_time=3600.0, T_ambient=298.15
        )
        T_max_air.append(result['T_max'])
    
    for h in h_liquid:
        result = optimizer.evaluate_strategy(
            cooling_type='liquid', h_conv=h, T_coolant=288.15,
            discharge_current=50.0,
            total_time=3600.0, T_ambient=298.15
        )
        T_max_liquid.append(result['T_max'])

    ax.plot(h_air, T_max_air, 'ro-', linewidth=2, markersize=8, label='风冷')
    ax.plot(h_liquid, T_max_liquid, 'bo-', linewidth=2, markersize=8, label='液冷')
    ax.set_xlabel('对流换热系数 h (W/m²·K)')
    ax.set_ylabel('最高温度 (°C)')
    ax.set_title('最高温度与换热系数的关系')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    coolant_temps = [283.15, 288.15, 293.15, 298.15]
    T_max_coolant = []
    for T_c in coolant_temps:
        result = optimizer.evaluate_strategy(
            cooling_type='liquid', h_conv=500.0, T_coolant=T_c,
            discharge_current=50.0,
            total_time=3600.0, T_ambient=298.15
        )
        T_max_coolant.append(result['T_max'])

    ax.plot(np.array(coolant_temps) - 273.15, T_max_coolant, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('冷却液温度 (°C)')
    ax.set_ylabel('最高温度 (°C)')
    ax.set_title('最高温度与冷却液温度的关系')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.axis('off')
    ax.text(0.1, 0.9, '热管理策略优化结论', fontsize=14, fontweight='bold',
            transform=ax.transAxes)
    conclusions = [
        f"• 最优风冷策略: 强制风冷 (h=100)",
        f"  最高温度: {min(T_max_air):.1f}°C",
        "",
        f"• 最优液冷策略: 标准液冷 (h=500, T=15°C)",
        f"  最高温度: {min(T_max_liquid):.1f}°C",
        "",
        f"• 液冷优势: {min(T_max_air) - min(T_max_liquid):.1f} K",
        "",
        "• 推荐: 标准液冷在散热和能耗间取得平衡"
    ]
    for i, text in enumerate(conclusions):
        ax.text(0.1, 0.8 - i * 0.06, text, fontsize=11, transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig('optimization_results.png', dpi=150)
    plt.close()
    print("  ✓ 已保存 optimization_results.png")

    # 打印总结
    print("\n" + "=" * 70)
    print("  仿真结果总结")
    print("=" * 70)
    print(f"\n  风冷 vs 液冷:")
    print(f"    风冷最高温度: {np.max(max_temp_air) - 273.15:.1f} °C")
    print(f"    液冷最高温度: {np.max(max_temp_liquid) - 273.15:.1f} °C")
    print(f"    温差: {np.max(max_temp_air) - np.max(max_temp_liquid):.1f} K")

    print(f"\n  日历寿命预测 (20%容量衰减):")
    print(f"    25°C: {calendar_lives[5]:.1f} 年")
    print(f"    40°C: {calendar_lives[7]:.1f} 年")
    print(f"    60°C: {calendar_lives[9]:.1f} 年")

    print(f"\n  热管理优化:")
    print(f"    最优风冷最高温度: {min(T_max_air):.1f}°C")
    print(f"    最优液冷最高温度: {min(T_max_liquid):.1f}°C")

    print("\n  生成的图表:")
    print("    1. cooling_comparison.png - 冷却方式对比")
    print("    2. aging_prediction.png - 老化预测")
    print("    3. coupling_effects.png - 耦合效果")
    print("    4. optimization_results.png - 优化结果")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
