import numpy as np
import matplotlib.pyplot as plt
from rrt_star import RRTStar
from path_smoothing import PathSmoother, calculate_path_length


def main():
    start = [0.0, 0.0]
    goal = [6.0, 6.0]

    obstacle_list = [
        [3.0, 3.0, 1.0],
        [2.0, 5.0, 0.8],
        [5.0, 2.0, 0.6],
        [[1.0, 1.0], [1.0, 2.0], [2.0, 2.0], [2.0, 1.0]],
        [[4.0, 4.0], [4.5, 5.0], [5.0, 4.5], [4.5, 4.0]]
    ]

    search_area = [0, 7]

    print("开始 RRT* 路径规划...")
    rrt_star = RRTStar(
        start=start,
        goal=goal,
        obstacle_list=obstacle_list,
        search_area=search_area,
        expand_dis=0.5,
        goal_sample_rate=10,
        max_iter=800,
        connect_circle_dist=1.5
    )

    path = rrt_star.planning(animation=False)

    if path is None:
        print("未找到路径！")
        return

    print(f"原始路径长度: {calculate_path_length(path):.2f}")

    print("开始路径平滑...")
    smoother = PathSmoother(obstacle_list=obstacle_list)
    
    smooth_path_bspline = smoother.smooth_path(path, method='b_spline', s=0.3, num_points=100)
    smooth_path_bezier = smoother.smooth_path(path, method='bezier', num_points=100)

    print(f"B样条平滑路径长度: {calculate_path_length(smooth_path_bspline):.2f}")
    print(f"贝塞尔平滑路径长度: {calculate_path_length(smooth_path_bezier):.2f}")

    plot_results(rrt_star, path, smooth_path_bspline, smooth_path_bezier, obstacle_list, start, goal)


def plot_results(rrt_star, raw_path, smooth_path_bspline, smooth_path_bezier, obstacle_list, start, goal):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    titles = ['RRT* 原始路径', 'B样条平滑路径', '贝塞尔平滑路径']
    paths = [raw_path, smooth_path_bspline, smooth_path_bezier]
    colors = ['r-', 'm-', 'c-']
    line_widths = [2, 3, 3]

    for ax, title, path, color, lw in zip(axes, titles, paths, colors, line_widths):
        for obs in obstacle_list:
            if len(obs) == 3:
                x, y, r = obs
                circle = plt.Circle((x, y), r, color='b', alpha=0.5)
                ax.add_artist(circle)
            else:
                poly = np.array(obs)
                ax.fill(poly[:, 0], poly[:, 1], 'b', alpha=0.5)

        path_np = np.array(path)
        ax.plot(path_np[:, 0], path_np[:, 1], color, linewidth=lw, label=title)

        ax.plot(start[0], start[1], "go", markersize=12, label="起点")
        ax.plot(goal[0], goal[1], "ro", markersize=12, label="终点")

        ax.set_xlim(0, 7)
        ax.set_ylim(0, 7)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('rrt_star_result.png', dpi=150, bbox_inches='tight')
    print("结果图已保存为 rrt_star_result.png")
    plt.show()


def custom_example():
    print("\n自定义示例: 复杂障碍物环境")
    
    start = [1.0, 1.0]
    goal = [9.0, 9.0]

    obstacle_list = [
        [5.0, 5.0, 2.0],
        [3.0, 7.0, 1.0],
        [7.0, 3.0, 1.0],
        [2.0, 4.0, 0.8],
        [8.0, 6.0, 0.8],
        [[0.5, 0.5], [0.5, 2.5], [2.5, 2.5], [2.5, 0.5]],
        [[7.5, 7.5], [7.5, 9.5], [9.5, 9.5], [9.5, 7.5]],
        [[4.0, 7.5], [4.5, 8.5], [5.0, 8.0], [4.5, 7.5]]
    ]

    search_area = [0, 10]

    rrt_star = RRTStar(
        start=start,
        goal=goal,
        obstacle_list=obstacle_list,
        search_area=search_area,
        expand_dis=0.6,
        goal_sample_rate=15,
        max_iter=1000,
        connect_circle_dist=2.0
    )

    path = rrt_star.planning(animation=True)

    if path is None:
        print("未找到路径！")
        return

    smoother = PathSmoother(obstacle_list=obstacle_list)
    smooth_path = smoother.smooth_path(path, method='b_spline', s=0.5)

    print(f"原始路径长度: {calculate_path_length(path):.2f}")
    print(f"平滑路径长度: {calculate_path_length(smooth_path):.2f}")

    plt.figure(figsize=(10, 10))
    
    for obs in obstacle_list:
        if len(obs) == 3:
            x, y, r = obs
            circle = plt.Circle((x, y), r, color='b', alpha=0.5)
            plt.gcf().gca().add_artist(circle)
        else:
            poly = np.array(obs)
            plt.fill(poly[:, 0], poly[:, 1], 'b', alpha=0.5)

    path_np = np.array(path)
    smooth_np = np.array(smooth_path)
    
    plt.plot(path_np[:, 0], path_np[:, 1], 'r--', linewidth=2, label='原始路径')
    plt.plot(smooth_np[:, 0], smooth_np[:, 1], 'g-', linewidth=3, label='平滑路径')
    
    plt.plot(start[0], start[1], "go", markersize=15, label="起点")
    plt.plot(goal[0], goal[1], "ro", markersize=15, label="终点")
    
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.title('RRT* 路径规划 - 复杂环境', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig('rrt_star_complex.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    print("=" * 50)
    print("RRT* 最优路径规划与平滑")
    print("=" * 50)
    
    main()
    
    print("\n是否运行复杂环境示例？(y/n)")
    choice = input().strip().lower()
    if choice == 'y' or choice == 'yes':
        custom_example()
