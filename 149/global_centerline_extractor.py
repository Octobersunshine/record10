import numpy as np
import heapq
import networkx as nx
from scipy import ndimage
from scipy import spatial
from skimage import morphology
from collections import deque, defaultdict


class GlobalCenterlineExtractor:
    def __init__(self, spacing=(1.0, 1.0, 1.0)):
        self.spacing = np.array(spacing, dtype=np.float64)

    def extract_topological_skeleton(self, binary_image):
        skeleton = morphology.skeletonize_3d(binary_image)
        skeleton_points = np.argwhere(skeleton)
        return skeleton, skeleton_points

    def prune_skeleton(self, skeleton, min_branch_length=5):
        points = np.argwhere(skeleton)
        if len(points) == 0:
            return skeleton, points

        graph = self._build_graph_from_skeleton(skeleton)
        pruned_graph = self._prune_graph(graph, min_branch_length)

        pruned_skeleton = np.zeros_like(skeleton, dtype=bool)
        for node in pruned_graph.nodes():
            pos = pruned_graph.nodes[node]['position']
            idx = tuple(pos.astype(int))
            if all(0 <= idx[i] < skeleton.shape[i] for i in range(3)):
                pruned_skeleton[idx] = True

        pruned_points = np.argwhere(pruned_skeleton)
        return pruned_skeleton, pruned_points

    def _build_graph_from_skeleton(self, skeleton):
        G = nx.Graph()
        points = np.argwhere(skeleton)

        if len(points) == 0:
            return G

        for i, point in enumerate(points):
            node_id = f"skel_{i}"
            G.add_node(node_id, position=point * self.spacing, index=point)

        point_tuple_to_node = {tuple(p): f"skel_{i}" for i, p in enumerate(points)}

        for i, point in enumerate(points):
            node_id = f"skel_{i}"
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        neighbor = (point[0] + dx, point[1] + dy, point[2] + dz)
                        if neighbor in point_tuple_to_node:
                            neighbor_node = point_tuple_to_node[neighbor]
                            if not G.has_edge(node_id, neighbor_node):
                                dist = np.linalg.norm(
                                    (np.array(neighbor) - point) * self.spacing
                                )
                                G.add_edge(node_id, neighbor_node, weight=dist)

        return G

    def _prune_graph(self, graph, min_branch_length):
        if graph.number_of_nodes() == 0:
            return graph

        degrees = dict(graph.degree())
        endpoints = [n for n, d in degrees.items() if d == 1]

        for endpoint in endpoints:
            path = []
            current = endpoint
            prev = None
            path_length = 0

            while True:
                path.append(current)
                neighbors = list(graph.neighbors(current))
                next_nodes = [n for n in neighbors if n != prev]

                if len(next_nodes) == 0:
                    break
                elif len(next_nodes) == 1 and degrees[current] <= 2:
                    edge_data = graph.get_edge_data(current, next_nodes[0])
                    path_length += edge_data.get('weight', 1.0)
                    prev = current
                    current = next_nodes[0]
                else:
                    break

            if path_length < min_branch_length and len(path) >= 2:
                for node in path[:-1]:
                    if node in graph:
                        graph.remove_node(node)

        return graph

    def multi_source_fast_marching_tree(self, speed_image, binary_mask, num_sources=10):
        shape = speed_image.shape
        time_map = np.full(shape, np.inf, dtype=np.float64)
        source_map = np.full(shape + (3,), -1, dtype=np.int32)
        state = np.zeros(shape, dtype=np.int8)

        FAR, NARROW, ALIVE = 0, 1, 2

        dt = ndimage.distance_transform_edt(binary_mask, sampling=self.spacing)
        source_points = self._find_multiple_seeds(dt, binary_mask, num_sources)

        heap = []
        for source in source_points:
            src = tuple(source.astype(int))
            if binary_mask[src]:
                time_map[src] = 0
                state[src] = NARROW
                heapq.heappush(heap, (0.0, src))

        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbors.append((dx, dy, dz))

        while heap:
            current_time, current = heapq.heappop(heap)

            if state[current] == ALIVE:
                continue

            state[current] = ALIVE

            for neighbor in neighbors:
                ni = (current[0] + neighbor[0],
                      current[1] + neighbor[1],
                      current[2] + neighbor[2])

                if (0 <= ni[0] < shape[0] and
                    0 <= ni[1] < shape[1] and
                    0 <= ni[2] < shape[2] and
                    state[ni] != ALIVE and
                    binary_mask[ni]):

                    step = np.sqrt((neighbor[0] * self.spacing[0])**2 +
                                   (neighbor[1] * self.spacing[1])**2 +
                                   (neighbor[2] * self.spacing[2])**2)

                    speed = (speed_image[current] + speed_image[ni]) / 2.0
                    if speed <= 0:
                        continue

                    new_time = current_time + step / speed

                    if new_time < time_map[ni]:
                        time_map[ni] = new_time
                        source_map[ni] = current
                        if state[ni] == FAR:
                            state[ni] = NARROW
                            heapq.heappush(heap, (new_time, ni))

        return time_map, source_map, source_points

    def _find_multiple_seeds(self, distance_transform, binary_mask, num_seeds):
        dt = distance_transform.copy()
        dt[~binary_mask] = -1

        seeds = []
        for _ in range(num_seeds):
            if np.max(dt) <= 0:
                break
            seed = np.unravel_index(np.argmax(dt), dt.shape)
            seeds.append(np.array(seed))

            r = max(3, int(dt[seed] / np.min(self.spacing)))
            x, y, z = seed
            x_slice = slice(max(0, x - r), min(dt.shape[0], x + r + 1))
            y_slice = slice(max(0, y - r), min(dt.shape[1], y + r + 1))
            z_slice = slice(max(0, z - r), min(dt.shape[2], z + r + 1))
            dt[x_slice, y_slice, z_slice] = -1

        return seeds

    def build_gradient_tree(self, time_map, source_map, binary_mask):
        tree_points = []

        mask_indices = np.argwhere(binary_mask)
        for idx in mask_indices:
            idx_tuple = tuple(idx)

            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        ni = (idx[0] + dx, idx[1] + dy, idx[2] + dz)
                        if (0 <= ni[0] < binary_mask.shape[0] and
                            0 <= ni[1] < binary_mask.shape[1] and
                            0 <= ni[2] < binary_mask.shape[2] and
                            binary_mask[ni]):
                            neighbors.append((ni, time_map[ni]))

            if len(neighbors) > 0:
                neighbors.sort(key=lambda x: x[1])
                min_time_neighbor = neighbors[0][0]

                if time_map[idx_tuple] > time_map[min_time_neighbor]:
                    tree_points.append(idx)

        return np.array(tree_points)

    def power_watershed_centerline(self, binary_mask, distance_transform=None,
                                     num_sources=5, lambda_cost=0.5):
        if distance_transform is None:
            distance_transform = ndimage.distance_transform_edt(
                binary_mask, sampling=self.spacing
            )

        speed = distance_transform / (np.max(distance_transform) + 1e-8)
        speed[~binary_mask] = 0

        time_map, source_map, seeds = self.multi_source_fast_marching_tree(
            speed, binary_mask, num_sources
        )

        tree_points = self.build_gradient_tree(time_map, source_map, binary_mask)

        return tree_points, time_map, seeds

    def extract_complete_vessel_tree(self, binary_image, method='skeleton', **kwargs):
        if method == 'skeleton':
            min_branch_length = kwargs.get('min_branch_length', 10)
            skeleton, raw_points = self.extract_topological_skeleton(binary_image)
            pruned_skeleton, points = self.prune_skeleton(skeleton, min_branch_length)

            return {
                'method': 'topological_skeleton',
                'skeleton': pruned_skeleton,
                'centerline_points': points,
                'raw_points': raw_points
            }

        elif method == 'power_watershed':
            num_sources = kwargs.get('num_sources', 8)
            min_branch_length = kwargs.get('min_branch_length', 8)

            dt = ndimage.distance_transform_edt(binary_image, sampling=self.spacing)
            points, time_map, seeds = self.power_watershed_centerline(
                binary_image, dt, num_sources
            )

            return {
                'method': 'power_watershed',
                'centerline_points': points,
                'time_map': time_map,
                'seed_points': seeds,
                'distance_transform': dt
            }

        elif method == 'hybrid':
            skeleton_result = self.extract_complete_vessel_tree(
                binary_image, 'skeleton', **kwargs
            )
            pw_result = self.extract_complete_vessel_tree(
                binary_image, 'power_watershed', **kwargs
            )

            combined_mask = np.zeros_like(binary_image, dtype=bool)
            for point in skeleton_result['centerline_points']:
                combined_mask[tuple(point.astype(int))] = True
            for point in pw_result['centerline_points']:
                combined_mask[tuple(point.astype(int))] = True

            combined_points = np.argwhere(combined_mask)

            return {
                'method': 'hybrid',
                'centerline_points': combined_points,
                'skeleton_result': skeleton_result,
                'power_watershed_result': pw_result
            }

        else:
            raise ValueError(f"Unknown method: {method}. Use 'skeleton', 'power_watershed', or 'hybrid'")
