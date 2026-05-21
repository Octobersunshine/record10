import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import nearest_points
from scipy.spatial import cKDTree
import time


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0


class RRTStar:
    def __init__(self, start, goal, obstacle_list, search_area,
                 expand_dis=0.5, goal_sample_rate=5, max_iter=500,
                 connect_circle_dist=1.0, kdtree_rebuild_freq=50):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.obstacle_list = obstacle_list
        self.obstacle_polygons = []
        
        for obs in obstacle_list:
            if len(obs) == 3:
                x, y, r = obs
                self.obstacle_polygons.append(Point(x, y).buffer(r))
            else:
                self.obstacle_polygons.append(Polygon(obs))
        
        self.min_rand = search_area[0]
        self.max_rand = search_area[1]
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.connect_circle_dist = connect_circle_dist
        self.kdtree_rebuild_freq = kdtree_rebuild_freq
        
        self.node_list = []
        self.node_coords = np.empty((0, 2), dtype=np.float64)
        self.kdtree = None
        self.perf_stats = {'search_time': 0, 'search_count': 0}

    def planning(self, animation=True):
        self.node_list = [self.start]
        self.node_coords = np.array([[self.start.x, self.start.y]], dtype=np.float64)
        self.kdtree = cKDTree(self.node_coords)
        path_found = False

        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            
            t_start = time.perf_counter()
            nearest_ind = self.get_nearest_node_index_kdtree(rnd_node)
            self.perf_stats['search_time'] += time.perf_counter() - t_start
            self.perf_stats['search_count'] += 1
            
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node):
                t_start = time.perf_counter()
                near_inds = self.find_near_nodes_kdtree(new_node)
                self.perf_stats['search_time'] += time.perf_counter() - t_start
                self.perf_stats['search_count'] += 1
                
                new_node = self.choose_parent(new_node, near_inds)
                if new_node.parent is not None:
                    self.node_list.append(new_node)
                    self.node_coords = np.vstack([self.node_coords, [new_node.x, new_node.y]])
                    
                    if len(self.node_list) % self.kdtree_rebuild_freq == 0:
                        self.kdtree = cKDTree(self.node_coords)
                    
                    self.rewire(new_node, near_inds)

            if animation and i % 5 == 0:
                self.draw_graph(rnd_node)

            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.goal, self.expand_dis)
                if self.check_collision(final_node):
                    path_found = True
                    break

        if not path_found:
            print("未找到路径")
            return None

        path = self.generate_final_course(len(self.node_list) - 1)
        return path

    def get_performance_stats(self):
        avg_time = self.perf_stats['search_time'] / max(1, self.perf_stats['search_count'])
        return {
            'total_search_time': self.perf_stats['search_time'],
            'search_count': self.perf_stats['search_count'],
            'avg_search_time': avg_time,
            'node_count': len(self.node_list)
        }

    def steer(self, from_node, to_node, extend_length=float('inf')):
        new_node = Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.x += min(extend_length, d) * np.cos(theta)
        new_node.y += min(extend_length, d) * np.sin(theta)
        new_node.cost = from_node.cost + min(extend_length, d)
        new_node.parent = from_node

        return new_node

    def get_random_node(self):
        if np.random.randint(0, 100) > self.goal_sample_rate:
            rnd = Node(np.random.uniform(self.min_rand, self.max_rand),
                       np.random.uniform(self.min_rand, self.max_rand))
        else:
            rnd = Node(self.goal.x, self.goal.y)
        return rnd

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2
                 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    def get_nearest_node_index_kdtree(self, rnd_node):
        query_point = np.array([rnd_node.x, rnd_node.y], dtype=np.float64)
        
        if len(self.node_list) % self.kdtree_rebuild_freq != 0 and len(self.node_list) > 1:
            distances, indices = self.kdtree.query(query_point, k=min(self.kdtree_rebuild_freq, len(self.node_list)))
            if isinstance(indices, np.ndarray):
                min_idx = indices[0]
                min_dist = distances[0]
                for i in range(len(self.node_coords), len(self.node_list)):
                    node = self.node_list[i]
                    d = (node.x - query_point[0]) ** 2 + (node.y - query_point[1]) ** 2
                    if d < min_dist:
                        min_dist = d
                        min_idx = i
                return min_idx
            else:
                min_idx = indices
                min_dist = distances
                for i in range(len(self.node_coords), len(self.node_list)):
                    node = self.node_list[i]
                    d = (node.x - query_point[0]) ** 2 + (node.y - query_point[1]) ** 2
                    if d < min_dist:
                        min_dist = d
                        min_idx = i
                return min_idx
        
        _, min_idx = self.kdtree.query(query_point, k=1)
        return min_idx

    def check_collision(self, node):
        if node is None:
            return False
        
        point = Point(node.x, node.y)
        
        for polygon in self.obstacle_polygons:
            if polygon.contains(point):
                return False
        
        if node.parent is not None:
            line = LineString([(node.parent.x, node.parent.y), (node.x, node.y)])
            for polygon in self.obstacle_polygons:
                if line.intersects(polygon):
                    return False
        
        return True

    def find_near_nodes(self, new_node):
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * np.sqrt((np.log(nnode) / nnode))
        r = min(r, self.expand_dis)
        dist_list = [(node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2
                     for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r ** 2]
        return near_inds

    def find_near_nodes_kdtree(self, new_node):
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * np.sqrt((np.log(nnode) / nnode))
        r = min(r, self.expand_dis)
        
        query_point = np.array([new_node.x, new_node.y], dtype=np.float64)
        
        if len(self.node_list) % self.kdtree_rebuild_freq != 0 and len(self.node_list) > 1:
            near_inds = self.kdtree.query_ball_point(query_point, r)
            near_inds_set = set(near_inds)
            for i in range(len(self.node_coords), len(self.node_list)):
                node = self.node_list[i]
                d_sq = (node.x - query_point[0]) ** 2 + (node.y - query_point[1]) ** 2
                if d_sq <= r ** 2:
                    near_inds_set.add(i)
            return list(near_inds_set)
        
        near_inds = self.kdtree.query_ball_point(query_point, r)
        return near_inds

    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return new_node

        dlist = []
        for i in near_inds:
            near_node = self.node_list[i]
            d, _ = self.calc_distance_and_angle(near_node, new_node)
            t_node = self.steer(near_node, new_node, d)
            if self.check_collision(t_node):
                dlist.append(near_node.cost + d)
            else:
                dlist.append(float('inf'))

        min_cost = min(dlist)
        min_ind = near_inds[dlist.index(min_cost)]

        if min_cost == float('inf'):
            return new_node

        new_node.cost = min_cost
        new_node.parent = self.node_list[min_ind]
        return new_node

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            d, _ = self.calc_distance_and_angle(new_node, near_node)
            new_cost = new_node.cost + d

            if near_node.cost > new_cost:
                if self.check_collision(self.steer(new_node, near_node, d)):
                    near_node.parent = new_node
                    near_node.cost = new_cost

    def calc_dist_to_goal(self, x, y):
        dx = x - self.goal.x
        dy = y - self.goal.y
        return np.hypot(dx, dy)

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)
        return d, theta

    def generate_final_course(self, goal_ind):
        path = [[self.goal.x, self.goal.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path[::-1]

    def draw_graph(self, rnd=None):
        plt.clf()
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")

        for node in self.node_list:
            if node.parent:
                plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g")

        for obs in self.obstacle_list:
            if len(obs) == 3:
                x, y, r = obs
                circle = plt.Circle((x, y), r, color='b', alpha=0.5)
                plt.gcf().gca().add_artist(circle)
            else:
                poly = np.array(obs)
                plt.fill(poly[:, 0], poly[:, 1], 'b', alpha=0.5)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.goal.x, self.goal.y, "xr")
        plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])
        plt.grid(True)
        plt.pause(0.01)
