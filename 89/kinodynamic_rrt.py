import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Point, LineString, Polygon
from scipy.spatial import cKDTree
import time


class CarState:
    def __init__(self, x, y, theta, v=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.parent = None
        self.u = None
        self.dt = None
        self.cost = 0.0

    def to_array(self):
        return np.array([self.x, self.y, np.cos(self.theta), np.sin(self.theta)])

    def __repr__(self):
        return f"State(x={self.x:.2f}, y={self.y:.2f}, theta={np.degrees(self.theta):.1f}°, v={self.v:.2f})"


class KinodynamicRRT:
    def __init__(self, start, goal, obstacle_list, arena_bounds,
                 wheelbase=2.5, max_steer=np.radians(30), max_accel=2.0,
                 max_speed=5.0, dt=0.2, max_iter=1000, goal_sample_rate=10):
        self.start = CarState(start[0], start[1], start[2], start[3])
        self.goal = CarState(goal[0], goal[1], goal[2], goal[3])
        self.obstacle_list = obstacle_list
        self.obstacle_polygons = []
        
        for obs in obstacle_list:
            if len(obs) == 3:
                x, y, r = obs
                self.obstacle_polygons.append(Point(x, y).buffer(r))
            else:
                self.obstacle_polygons.append(Polygon(obs))
        
        self.arena_bounds = arena_bounds
        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.max_accel = max_accel
        self.max_speed = max_speed
        self.dt = dt
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        
        self.node_list = []
        self.state_coords = np.empty((0, 4), dtype=np.float64)
        self.kdtree = None
        
        self.car_length = 4.5
        self.car_width = 1.8
        self.rear_overhang = 1.0

    def bicycle_model(self, state, u):
        steer = np.clip(u[0], -self.max_steer, self.max_steer)
        accel = np.clip(u[1], -self.max_accel, self.max_accel)
        
        v_new = np.clip(state.v + accel * self.dt, -self.max_speed/2, self.max_speed)
        
        if abs(v_new) > 1e-3:
            theta_dot = v_new * np.tan(steer) / self.wheelbase
        else:
            theta_dot = 0.0
        
        theta_new = state.theta + theta_dot * self.dt
        x_new = state.x + v_new * np.cos(theta_new) * self.dt
        y_new = state.y + v_new * np.sin(theta_new) * self.dt
        
        return CarState(x_new, y_new, theta_new, v_new)

    def get_random_control(self):
        steer = np.random.uniform(-self.max_steer, self.max_steer)
        accel = np.random.uniform(-self.max_accel, self.max_accel)
        duration = np.random.uniform(self.dt, self.dt * 5)
        return [steer, accel], duration

    def get_random_state(self):
        if np.random.randint(0, 100) > self.goal_sample_rate:
            x = np.random.uniform(self.arena_bounds[0], self.arena_bounds[2])
            y = np.random.uniform(self.arena_bounds[1], self.arena_bounds[3])
            theta = np.random.uniform(-np.pi, np.pi)
            v = np.random.uniform(-self.max_speed/2, self.max_speed)
            return CarState(x, y, theta, v)
        else:
            return self.goal

    def distance_metric(self, state1, state2):
        pos_dist = np.sqrt((state1.x - state2.x)**2 + (state1.y - state2.y)**2)
        theta_diff = abs(((state1.theta - state2.theta + np.pi) % (2*np.pi)) - np.pi)
        theta_weight = 2.0
        v_diff = abs(state1.v - state2.v)
        v_weight = 0.5
        return pos_dist + theta_weight * theta_diff + v_weight * v_diff

    def get_nearest_state_index(self, target_state):
        if len(self.node_list) < 100:
            dists = [self.distance_metric(node, target_state) for node in self.node_list]
            return dists.index(min(dists))
        
        query = target_state.to_array()
        _, idx = self.kdtree.query(query, k=1)
        return idx

    def steer(self, from_state, to_state, num_steps=5):
        best_state = None
        best_dist = float('inf')
        best_u = None
        best_duration = 0
        
        for _ in range(20):
            u, duration = self.get_random_control()
            current = CarState(from_state.x, from_state.y, from_state.theta, from_state.v)
            steps = int(duration / self.dt)
            
            trajectory = [current]
            for _ in range(steps):
                current = self.bicycle_model(current, u)
                trajectory.append(current)
            
            if self.check_trajectory_collision(trajectory):
                final_dist = self.distance_metric(trajectory[-1], to_state)
                if final_dist < best_dist:
                    best_dist = final_dist
                    best_state = trajectory[-1]
                    best_state.parent = from_state
                    best_state.u = u
                    best_state.dt = duration
                    best_state.cost = from_state.cost + duration
        
        return best_state

    def get_car_corners(self, state):
        rear_x = state.x - self.rear_overhang * np.cos(state.theta)
        rear_y = state.y - self.rear_overhang * np.sin(state.theta)
        front_x = rear_x + self.car_length * np.cos(state.theta)
        front_y = rear_y + self.car_length * np.sin(state.theta)
        
        half_width = self.car_width / 2
        perp_x = -np.sin(state.theta) * half_width
        perp_y = np.cos(state.theta) * half_width
        
        corners = [
            (rear_x + perp_x, rear_y + perp_y),
            (rear_x - perp_x, rear_y - perp_y),
            (front_x - perp_x, front_y - perp_y),
            (front_x + perp_x, front_y + perp_y),
        ]
        return corners

    def check_state_collision(self, state):
        corners = self.get_car_corners(state)
        car_poly = Polygon(corners)
        
        if not (self.arena_bounds[0] <= state.x <= self.arena_bounds[2] and
                self.arena_bounds[1] <= state.y <= self.arena_bounds[3]):
            return False
        
        for obs in self.obstacle_polygons:
            if car_poly.intersects(obs):
                return False
        
        return True

    def check_trajectory_collision(self, trajectory):
        for state in trajectory:
            if not self.check_state_collision(state):
                return False
        return True

    def is_goal_reached(self, state):
        pos_tol = 1.0
        theta_tol = np.radians(30)
        
        pos_dist = np.sqrt((state.x - self.goal.x)**2 + (state.y - self.goal.y)**2)
        theta_diff = abs(((state.theta - self.goal.theta + np.pi) % (2*np.pi)) - np.pi)
        
        return pos_dist < pos_tol and theta_diff < theta_tol

    def planning(self, animation=False):
        self.node_list = [self.start]
        self.state_coords = np.array([self.start.to_array()])
        path_found = False
        goal_node = None

        for i in range(self.max_iter):
            if i % 100 == 0:
                print(f"迭代 {i}/{self.max_iter}, 节点数: {len(self.node_list)}")

            rand_state = self.get_random_state()
            nearest_idx = self.get_nearest_state_index(rand_state)
            nearest_state = self.node_list[nearest_idx]

            new_state = self.steer(nearest_state, rand_state)

            if new_state is not None:
                self.node_list.append(new_state)
                self.state_coords = np.vstack([self.state_coords, new_state.to_array()])
                
                if len(self.node_list) % 50 == 0:
                    self.kdtree = cKDTree(self.state_coords)

                if self.is_goal_reached(new_state):
                    print(f"在迭代 {i} 找到路径！")
                    path_found = True
                    goal_node = new_state
                    break

            if animation and i % 50 == 0:
                self.draw_scene(rand_state)

        if not path_found:
            print("未找到可行路径")
            return None

        return self.extract_trajectory(goal_node)

    def extract_trajectory(self, goal_node):
        trajectory = []
        current = goal_node
        
        while current is not None:
            trajectory.append(current)
            current = current.parent
        
        trajectory.reverse()
        
        full_traj = []
        for i in range(len(trajectory) - 1):
            state = trajectory[i]
            next_state = trajectory[i + 1]
            if next_state.u is not None:
                steps = int(next_state.dt / self.dt)
                temp = CarState(state.x, state.y, state.theta, state.v)
                full_traj.append(CarState(temp.x, temp.y, temp.theta, temp.v))
                for _ in range(steps):
                    temp = self.bicycle_model(temp, next_state.u)
                    full_traj.append(CarState(temp.x, temp.y, temp.theta, temp.v))
        
        if len(full_traj) == 0:
            full_traj = trajectory
        
        return full_traj

    def draw_scene(self, rand_state=None, trajectory=None):
        plt.clf()
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlim(self.arena_bounds[0], self.arena_bounds[2])
        ax.set_ylim(self.arena_bounds[1], self.arena_bounds[3])
        ax.grid(True, alpha=0.3)

        for obs in self.obstacle_list:
            if len(obs) == 3:
                x, y, r = obs
                circle = plt.Circle((x, y), r, color='r', alpha=0.5)
                ax.add_artist(circle)
            else:
                poly = np.array(obs)
                ax.fill(poly[:, 0], poly[:, 1], 'r', alpha=0.5)

        self.draw_car(ax, self.start, 'g', '起点')
        self.draw_car(ax, self.goal, 'b', '终点')

        if trajectory:
            xs = [s.x for s in trajectory]
            ys = [s.y for s in trajectory]
            ax.plot(xs, ys, 'm-', linewidth=2, label='轨迹')

        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.legend()
        plt.pause(0.01)

    def draw_car(self, ax, state, color, label=None):
        corners = self.get_car_corners(state)
        car_poly = patches.Polygon(corners, linewidth=2, edgecolor=color,
                                   facecolor=color, alpha=0.5, label=label)
        ax.add_patch(car_poly)
        
        arrow_length = 2.0
        ax.arrow(state.x, state.y,
                 arrow_length * np.cos(state.theta),
                 arrow_length * np.sin(state.theta),
                 head_width=0.5, head_length=0.5, fc=color, ec=color)


def reverse_parking_scenario():
    print("=" * 70)
    print("Kinodynamic RRT - 汽车倒车入库场景")
    print("=" * 70)

    start = [0.0, 5.0, 0.0, 0.0]
    goal = [15.0, 0.0, np.radians(90), 0.0]

    obstacle_list = [
        [[8.0, -2.0], [8.0, 3.0], [12.0, 3.0], [12.0, -2.0]],
        [[18.0, -2.0], [18.0, 3.0], [22.0, 3.0], [22.0, -2.0]],
        [[6.0, -4.0], [24.0, -4.0], [24.0, -3.0], [6.0, -3.0]],
    ]

    arena_bounds = [-5.0, -6.0, 30.0, 15.0]

    rrt = KinodynamicRRT(
        start=start,
        goal=goal,
        obstacle_list=obstacle_list,
        arena_bounds=arena_bounds,
        wheelbase=2.5,
        max_steer=np.radians(35),
        max_accel=1.5,
        max_speed=3.0,
        dt=0.2,
        max_iter=5000,
        goal_sample_rate=15
    )

    print("\n开始规划倒车入库轨迹...")
    t_start = time.time()
    trajectory = rrt.planning(animation=False)
    t_total = time.time() - t_start

    if trajectory is None:
        print("规划失败，请调整参数")
        return

    print(f"规划耗时: {t_total:.2f} 秒")
    print(f"轨迹总点数: {len(trajectory)}")
    print(f"总时间: {len(trajectory) * rrt.dt:.1f} 秒")

    plt.figure(figsize=(14, 8))
    rrt.draw_scene(trajectory=trajectory)
    plt.title('Kinodynamic RRT - 汽车倒车入库轨迹', fontsize=14, fontweight='bold')
    
    for i, state in enumerate(trajectory):
        if i % 10 == 0:
            rrt.draw_car(plt.gca(), state, 'm')
    
    plt.tight_layout()
    plt.savefig('reverse_parking.png', dpi=150, bbox_inches='tight')
    print("结果图已保存为 reverse_parking.png")
    plt.show()

    plot_trajectory_details(trajectory, rrt.dt)


def plot_trajectory_details(trajectory, dt):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    times = np.arange(len(trajectory)) * dt
    xs = [s.x for s in trajectory]
    ys = [s.y for s in trajectory]
    thetas = [np.degrees(s.theta) for s in trajectory]
    vs = [s.v for s in trajectory]

    axes[0, 0].plot(xs, ys, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('X (m)', fontsize=11)
    axes[0, 0].set_ylabel('Y (m)', fontsize=11)
    axes[0, 0].set_title('车辆行驶轨迹', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_aspect('equal')

    axes[0, 1].plot(times, thetas, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('时间 (s)', fontsize=11)
    axes[0, 1].set_ylabel('航向角 (°)', fontsize=11)
    axes[0, 1].set_title('航向角变化', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(times, vs, 'g-', linewidth=2)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('时间 (s)', fontsize=11)
    axes[1, 0].set_ylabel('速度 (m/s)', fontsize=11)
    axes[1, 0].set_title('速度变化（负值表示倒车）', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    accel = np.diff(vs) / dt
    axes[1, 1].plot(times[:-1], accel, 'm-', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('时间 (s)', fontsize=11)
    axes[1, 1].set_ylabel('加速度 (m/s²)', fontsize=11)
    axes[1, 1].set_title('加速度变化', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('trajectory_details.png', dpi=150, bbox_inches='tight')
    print("轨迹详情图已保存为 trajectory_details.png")
    plt.show()


def parallel_parking_scenario():
    print("\n" + "=" * 70)
    print("Kinodynamic RRT - 平行泊车场景")
    print("=" * 70)

    start = [0.0, 0.0, 0.0, 0.0]
    goal = [12.0, 0.0, 0.0, 0.0]

    obstacle_list = [
        [6.0, 0.0, 2.3],
        [18.0, 0.0, 2.3],
        [[3.0, 1.0], [9.0, 1.0], [9.0, 2.5], [3.0, 2.5]],
        [[15.0, 1.0], [21.0, 1.0], [21.0, 2.5], [15.0, 2.5]],
    ]

    arena_bounds = [-5.0, -3.0, 25.0, 6.0]

    rrt = KinodynamicRRT(
        start=start,
        goal=goal,
        obstacle_list=obstacle_list,
        arena_bounds=arena_bounds,
        wheelbase=2.5,
        max_steer=np.radians(40),
        max_accel=1.0,
        max_speed=2.0,
        dt=0.2,
        max_iter=8000,
        goal_sample_rate=20
    )

    print("\n开始规划平行泊车轨迹...")
    t_start = time.time()
    trajectory = rrt.planning(animation=False)
    t_total = time.time() - t_start

    if trajectory is None:
        print("规划失败，请调整参数")
        return

    print(f"规划耗时: {t_total:.2f} 秒")
    print(f"轨迹总点数: {len(trajectory)}")
    print(f"总时间: {len(trajectory) * rrt.dt:.1f} 秒")

    plt.figure(figsize=(16, 6))
    rrt.draw_scene(trajectory=trajectory)
    plt.title('Kinodynamic RRT - 平行泊车轨迹', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('parallel_parking.png', dpi=150, bbox_inches='tight')
    print("结果图已保存为 parallel_parking.png")
    plt.show()


if __name__ == '__main__':
    reverse_parking_scenario()
    
    print("\n" + "=" * 70)
    print("是否运行平行泊车场景？(y/n)")
    print("=" * 70)
    choice = input().strip().lower()
    if choice == 'y':
        parallel_parking_scenario()
