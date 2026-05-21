import numpy as np
from scipy.interpolate import splprep, splev
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import nearest_points


class PathSmoother:
    def __init__(self, obstacle_list=None):
        self.obstacle_polygons = []
        if obstacle_list:
            for obs in obstacle_list:
                if len(obs) == 3:
                    x, y, r = obs
                    self.obstacle_polygons.append(Point(x, y).buffer(r))
                else:
                    self.obstacle_polygons.append(Polygon(obs))

    def smooth_path(self, path, method='b_spline', **kwargs):
        if path is None or len(path) < 2:
            return path

        if method == 'b_spline':
            return self.b_spline_smooth(path, **kwargs)
        elif method == 'gradient_descent':
            return self.gradient_descent_smooth(path, **kwargs)
        elif method == 'bezier':
            return self.bezier_smooth(path, **kwargs)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

    def b_spline_smooth(self, path, s=0.5, k=3, num_points=100):
        path = np.array(path)
        x = path[:, 0]
        y = path[:, 1]

        if len(path) <= k:
            k = len(path) - 1 if len(path) > 1 else 1

        tck, u = splprep([x, y], s=s, k=k)
        u_new = np.linspace(u.min(), u.max(), num_points)
        x_new, y_new = splev(u_new, tck, der=0)

        smooth_path = np.column_stack((x_new, y_new))

        if self.obstacle_polygons:
            smooth_path = self.ensure_collision_free(smooth_path)

        return smooth_path.tolist()

    def gradient_descent_smooth(self, path, alpha=0.1, beta=0.1, iterations=100):
        path = np.array(path, dtype=np.float64)
        n = len(path)

        if n < 3:
            return path.tolist()

        for _ in range(iterations):
            new_path = path.copy()

            for i in range(1, n - 1):
                smooth_term = alpha * (path[i - 1] - 2 * path[i] + path[i + 1])
                obstacle_term = beta * self.obstacle_gradient(path[i])
                new_path[i] += smooth_term - obstacle_term

            path = new_path

        return path.tolist()

    def obstacle_gradient(self, point):
        gradient = np.zeros(2)
        point_obj = Point(point[0], point[1])

        for polygon in self.obstacle_polygons:
            if polygon.distance(point_obj) < 1.0:
                nearest_pt = nearest_points(point_obj, polygon.boundary)[1]
                vec = np.array([point[0] - nearest_pt.x, point[1] - nearest_pt.y])
                dist = np.linalg.norm(vec)
                if dist > 0:
                    gradient += vec / (dist ** 2)

        return gradient

    def bezier_smooth(self, path, num_points=100):
        path = np.array(path)
        n = len(path)

        if n < 2:
            return path.tolist()

        t = np.linspace(0, 1, num_points)
        smooth_path = np.zeros((num_points, 2))

        for i in range(num_points):
            smooth_path[i] = self.de_casteljau(path, t[i])

        if self.obstacle_polygons:
            smooth_path = self.ensure_collision_free(smooth_path)

        return smooth_path.tolist()

    def de_casteljau(self, control_points, t):
        n = len(control_points)
        points = control_points.copy()

        for k in range(1, n):
            for i in range(n - k):
                points[i] = (1 - t) * points[i] + t * points[i + 1]

        return points[0]

    def ensure_collision_free(self, path):
        valid_path = [path[0]]

        for i in range(1, len(path)):
            line = LineString([(path[i-1][0], path[i-1][1]),
                             (path[i][0], path[i][1])])

            collision = False
            for polygon in self.obstacle_polygons:
                if line.intersects(polygon):
                    collision = True
                    break

            if not collision:
                valid_path.append(path[i])
            else:
                mid_point = [(path[i-1][0] + path[i][0]) / 2,
                            (path[i-1][1] + path[i][1]) / 2]
                if not self.is_point_collision(mid_point):
                    valid_path.append(mid_point)
                    valid_path.append(path[i])

        return np.array(valid_path)

    def is_point_collision(self, point):
        point_obj = Point(point[0], point[1])
        for polygon in self.obstacle_polygons:
            if polygon.contains(point_obj):
                return True
        return False


def calculate_path_length(path):
    if path is None or len(path) < 2:
        return 0

    path = np.array(path)
    diffs = np.diff(path, axis=0)
    lengths = np.sqrt((diffs ** 2).sum(axis=1))
    return lengths.sum()
