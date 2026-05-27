import numpy as np
import matplotlib.pyplot as plt
from coupled_flow_transport import CoupledFlowTransportModel
import warnings
warnings.filterwarnings('ignore')


def example_contaminant_leakage():
    print("=" * 70)
    print("污染物泄漏迁移预测示例")
    print("=" * 70)
    
    nx, ny = 40, 30
    dx, dy = 2.0, 2.0
    
    model = CoupledFlowTransportModel(nx, ny, dx, dy, dt_flow=0.5, dt_transport=0.05)
    
    print("\n设置土壤参数 (砂壤土):")
    model.set_soil_properties(
        theta_r=0.08, theta_s=0.43, alpha=3.5, n=1.6, 
        Ks=0.0005, l=0.5
    )
    
    print("设置溶质运移参数:")
    model.set_transport_params(
        DL=0.5, DT=0.05, rho_b=1.6, Kd=0.1, lambda_decay=0.0001
    )
    
    print("设置初始条件:")
    model.set_initial_conditions(h0=-2.0, C0=0.0, theta0=0.25)
    
    print("设置边界条件:")
    flow_bc = []
    for j in range(ny):
        flow_bc.append(('dirichlet', 0, j, -1.0))
        flow_bc.append(('dirichlet', nx-1, j, -3.0))
    
    for i in range(nx):
        flow_bc.append(('flux', i, 0, 0.0))
        flow_bc.append(('flux', i, ny-1, 0.0))
    
    transport_bc = []
    for j in range(ny):
        transport_bc.append(('dirichlet', 0, j, 0.0))
        transport_bc.append(('dirichlet', nx-1, j, 0.0))
    
    model.set_boundary_conditions(flow_bc=flow_bc, transport_bc=transport_bc)
    
    print("设置污染源 (污染物泄漏点):")
    leak_i, leak_j = 10, 15
    source_strength = 100.0
    transport_sources = [(leak_i, leak_j, source_strength)]
    model.set_sources(transport_sources=transport_sources)
    
    print("\n开始模拟...")
    total_time = 100.0
    output_interval = 10.0
    
    h, C = model.simulate(total_time, output_interval)
    
    print("\n模拟完成!")
    print("=" * 70)
    
    u, v = model.flow_solver.compute_velocity()
    
    print(f"\n水头范围: {h.min():.4f} m ~ {h.max():.4f} m")
    print(f"含水量范围: {model.flow_solver.theta.min():.4f} ~ {model.flow_solver.theta.max():.4f}")
    print(f"浓度范围: {C.min():.6f} ~ {C.max():.6f}")
    
    total_mass, flux_mass = model.compute_mass_balance()
    print(f"总污染物质量: {total_mass:.6f} kg")
    print(f"边界质量通量: {flux_mass:.6e} kg/s")
    
    times, concentrations = model.get_concentration_history()
    
    print(f"\n输出时间点: {times}")
    
    return model, times, concentrations


def example_remediation_design():
    print("\n\n" + "=" * 70)
    print("修复设计示例: 抽水-处理系统")
    print("=" * 70)
    
    nx, ny = 50, 40
    dx, dy = 1.5, 1.5
    
    model = CoupledFlowTransportModel(nx, ny, dx, dy, dt_flow=1.0, dt_transport=0.1)
    
    model.set_soil_properties(
        theta_r=0.10, theta_s=0.45, alpha=4.0, n=1.8, 
        Ks=0.001, l=0.5
    )
    
    model.set_transport_params(
        DL=1.0, DT=0.1, rho_b=1.5, Kd=0.05, lambda_decay=0.0
    )
    
    model.set_initial_conditions(h0=-1.5, C0=0.0, theta0=0.30)
    
    flow_bc = []
    for j in range(ny):
        flow_bc.append(('dirichlet', 0, j, 0.0))
        flow_bc.append(('dirichlet', nx-1, j, -5.0))
    
    for i in range(nx):
        flow_bc.append(('flux', i, 0, 0.0))
        flow_bc.append(('flux', i, ny-1, 0.0))
    
    transport_bc = []
    for j in range(ny):
        transport_bc.append(('dirichlet', 0, j, 0.0))
        transport_bc.append(('dirichlet', nx-1, j, 0.0))
    
    model.set_boundary_conditions(flow_bc=flow_bc, transport_bc=transport_bc)
    
    print("\n设置污染源区:")
    for i in range(10, 20):
        for j in range(15, 25):
            model.transport_solver.C[j, i] = 50.0
    
    print("设置抽水井 (修复井):")
    pumping_wells = [
        (30, 20),
        (35, 15),
        (35, 25)
    ]
    
    flow_sources = []
    for well_i, well_j in pumping_wells:
        flow_sources.append((well_i, well_j, -0.001))
    
    model.set_sources(flow_sources=flow_sources)
    
    print("\n开始修复模拟...")
    total_time = 200.0
    output_interval = 20.0
    
    h, C = model.simulate(total_time, output_interval)
    
    print("\n修复模拟完成!")
    print("=" * 70)
    
    u, v = model.flow_solver.compute_velocity()
    
    print(f"\n修复后浓度范围: {C.min():.6f} ~ {C.max():.6f}")
    
    source_zone_C = C[15:25, 10:20]
    print(f"源区平均浓度: {source_zone_C.mean():.4f}")
    
    well_concentrations = []
    for well_i, well_j in pumping_wells:
        well_concentrations.append(C[well_j, well_i])
        print(f"抽水井 ({well_i}, {well_j}) 浓度: {C[well_j, well_i]:.4f}")
    
    total_mass, flux_mass = model.compute_mass_balance()
    print(f"剩余污染物质量: {total_mass:.6f} kg")
    
    times, concentrations = model.get_concentration_history()
    
    mass_history = []
    for C_t in concentrations:
        mass_history.append(np.sum(C_t * model.flow_solver.theta) * dx * dy)
    
    print(f"\n质量随时间变化:")
    for t, m in zip(times, mass_history):
        print(f"  t={t:6.1f}天: 总质量={m:.4f} kg")
    
    return model, times, concentrations, pumping_wells


def plot_contaminant_results(model, times, concentrations, title="污染物迁移模拟"):
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    x = np.linspace(0, (model.nx - 1) * model.dx, model.nx)
    y = np.linspace(0, (model.ny - 1) * model.dy, model.ny)
    X, Y = np.meshgrid(x, y)
    
    plot_indices = [0, len(times)//4, len(times)//2, 3*len(times)//4, -1]
    plot_indices = [i for i in plot_indices if i < len(times)]
    
    for idx, plot_idx in enumerate(plot_indices[:4]):
        ax = axes[0, idx]
        im = ax.contourf(X, Y, concentrations[plot_idx], levels=20, cmap='Reds')
        ax.set_title(f'浓度分布 t={times[plot_idx]:.1f}天')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        plt.colorbar(im, ax=ax)
    
    ax = axes[0, 3]
    im = ax.contourf(X, Y, model.flow_solver.h, levels=20, cmap='Blues')
    ax.set_title('水头分布')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    plt.colorbar(im, ax=ax)
    
    u, v = model.flow_solver.compute_velocity()
    
    ax = axes[1, 0]
    speed = np.sqrt(u**2 + v**2)
    im = ax.contourf(X, Y, speed, levels=20, cmap='hot')
    ax.set_title('流速大小')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 1]
    ax.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2], scale=10)
    ax.set_title('流速矢量场')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    
    ax = axes[1, 2]
    ax.contourf(X, Y, model.flow_solver.theta, levels=20, cmap='PuBuGn')
    ax.set_title('含水量分布')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    
    mid_j = model.ny // 2
    ax = axes[1, 3]
    for plot_idx in plot_indices[:4]:
        ax.plot(x, concentrations[plot_idx][mid_j, :], label=f't={times[plot_idx]:.0f}天')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('浓度')
    ax.set_title(f'中线浓度剖面 (y={mid_j*model.dy:.0f}m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    mass_history = []
    for C_t in concentrations:
        mass_history.append(np.sum(C_t * model.flow_solver.theta) * model.dx * model.dy)
    
    ax = axes[2, 0]
    ax.plot(times, mass_history, 'b-o', linewidth=2)
    ax.set_xlabel('时间 (天)')
    ax.set_ylabel('总质量 (kg)')
    ax.set_title('污染物质量变化')
    ax.grid(True, alpha=0.3)
    
    max_conc_history = [C_t.max() for C_t in concentrations]
    
    ax = axes[2, 1]
    ax.plot(times, max_conc_history, 'r-s', linewidth=2)
    ax.set_xlabel('时间 (天)')
    ax.set_ylabel('最大浓度')
    ax.set_title('最大浓度变化')
    ax.grid(True, alpha=0.3)
    
    plume_area_history = []
    threshold = max_conc_history[0] * 0.01
    for C_t in concentrations:
        plume_area = np.sum(C_t > threshold) * model.dx * model.dy
        plume_area_history.append(plume_area)
    
    ax = axes[2, 2]
    ax.plot(times, plume_area_history, 'g-^', linewidth=2)
    ax.set_xlabel('时间 (天)')
    ax.set_ylabel('羽状体面积 (m²)')
    ax.set_title('污染羽范围变化')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 3]
    h_vals = np.linspace(-5, 0, 100)
    theta_vals = [model.flow_solver.soil_model.moisture_content(h) for h in h_vals]
    K_vals = [model.flow_solver.soil_model.hydraulic_conductivity(h) for h in h_vals]
    
    ax_twin = ax.twinx()
    ax.plot(h_vals, theta_vals, 'b-', label='含水量')
    ax_twin.semilogy(h_vals, K_vals, 'r--', label='渗透系数')
    ax.set_xlabel('水头 h (m)')
    ax.set_ylabel('含水量')
    ax_twin.set_ylabel('渗透系数 (m/s)')
    ax.set_title('土壤水分特征曲线')
    ax.legend(loc='upper left')
    ax_twin.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig('contaminant_transport_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n结果已保存为 contaminant_transport_results.png")


def example_saturated_unsaturated():
    print("\n\n" + "=" * 70)
    print("饱和-非饱和带耦合模拟示例")
    print("=" * 70)
    
    nx, ny = 50, 40
    dx, dy = 1.0, 1.0
    
    model = CoupledFlowTransportModel(nx, ny, dx, dy, dt_flow=0.5, dt_transport=0.05)
    
    model.set_soil_properties(
        theta_r=0.08, theta_s=0.43, alpha=3.5, n=1.6, 
        Ks=0.0005, l=0.5
    )
    
    model.set_transport_params(
        DL=0.5, DT=0.05, rho_b=1.6, Kd=0.1, lambda_decay=0.0001
    )
    
    h_initial = np.zeros((ny, nx))
    for j in range(ny):
        for i in range(nx):
            z = (ny - 1 - j) * dy
            h_initial[j, i] = -z + 10.0
    
    model.set_initial_conditions(h0=h_initial, C0=0.0)
    
    flow_bc = []
    for j in range(ny):
        if j < ny * 0.7:
            flow_bc.append(('dirichlet', 0, j, -5.0))
            flow_bc.append(('dirichlet', nx-1, j, -8.0))
        else:
            flow_bc.append(('flux', 0, j, 0.0))
            flow_bc.append(('flux', nx-1, j, 0.0))
    
    for i in range(nx):
        flow_bc.append(('flux', i, 0, 0.0))
        flow_bc.append(('flux', i, ny-1, -0.0001))
    
    transport_bc = []
    for j in range(ny):
        transport_bc.append(('dirichlet', 0, j, 0.0))
        transport_bc.append(('dirichlet', nx-1, j, 0.0))
    
    model.set_boundary_conditions(flow_bc=flow_bc, transport_bc=transport_bc)
    
    leak_i, leak_j = 25, 30
    transport_sources = [(leak_i, leak_j, 50.0)]
    model.set_sources(transport_sources=transport_sources)
    
    print("\n开始模拟...")
    total_time = 150.0
    output_interval = 15.0
    
    h, C = model.simulate(total_time, output_interval)
    
    print("\n模拟完成!")
    print("=" * 70)
    
    theta = model.flow_solver.theta
    saturated_zone = theta > 0.40
    print(f"饱和带面积: {np.sum(saturated_zone) * dx * dy:.1f} m²")
    
    return model


if __name__ == "__main__":
    print("地下水流动与溶质运移综合模拟系统")
    print("=" * 70)
    print("物理模块:")
    print("  1. 非饱和水流: Richards方程 + van Genuchten/Mualem模型")
    print("  2. 饱和水流: 达西方程")
    print("  3. 溶质运移: 对流-弥散方程 (含吸附、衰减)")
    print("=" * 70)
    
    model1, times1, conc1 = example_contaminant_leakage()
    plot_contaminant_results(model1, times1, conc1, "污染物泄漏迁移预测")
    
    model2, times2, conc2, wells = example_remediation_design()
    plot_contaminant_results(model2, times2, conc2, "抽水-处理修复系统模拟")
    
    model3 = example_saturated_unsaturated()
    
    print("\n\n" + "=" * 70)
    print("所有模拟完成!")
    print("=" * 70)
    print("\n结果文件:")
    print("  - contaminant_transport_results.png: 可视化结果")
