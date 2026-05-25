import numpy as np
from scipy import ndimage
from skimage import morphology, graph
from collections import deque
import heapq


class DistanceTransformCenterline:
    def __init__(self, spacing=(1.0, 1.0, 1.0)):
        self.spacing = np.array(spacing, dtype=np.float64)

    def compute_distance_transform(self, binary_image):
        dt = ndimage.distance_transform_edt(binary_image, sampling=self.spacing)
        return dt

    def find_source_points(self, distance_transform, binary_image, num_sources=1):
        dt = distance_transform.copy()
        dt[~binary_image] = -1
        
        sources = []
        for _ in range(num_sources):
            idx = np.unravel_index(np.argmax(dt), dt.shape)
            if dt[idx] <= 0:
                break
            sources.append(np.array(idx))
            
            mask = np.ones(dt.shape, dtype=bool)
            r = max(3, int(dt[idx] / np.min(self.spacing)))
            x, y, z = idx
            x_slice = slice(max(0, x - r), min(dt.shape[0], x + r + 1))
            y_slice = slice(max(0, y - r), min(dt.shape[1], y + r + 1))
            z_slice = slice(max(0, z - r), min(dt.shape[2], z + r + 1))
            mask[x_slice, y_slice, z_slice] = False
            dt[mask] = -1
        
        return sources

    def compute_geodesic_distance(self, binary_image, source_points, distance_transform):
        penalty = 1.0 / (distance_transform + 1e-8)
        penalty[~binary_image] = np.inf
        
        dist = np.full(binary_image.shape, np.inf)
        prev = np.full(binary_image.shape + (3,), -1, dtype=np.int32)
        
        heap = []
        for source in source_points:
            src = tuple(source.astype(int))
            if binary_image[src]:
                dist[src] = 0
                heapq.heappush(heap, (0, src))
        
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbors.append((dx, dy, dz))
        
        while heap:
            current_dist, current = heapq.heappop(heap)
            
            if current_dist > dist[current]:
                continue
            
            for neighbor in neighbors:
                ni = (current[0] + neighbor[0], 
                      current[1] + neighbor[1], 
                      current[2] + neighbor[2])
                
                if (0 <= ni[0] < binary_image.shape[0] and
                    0 <= ni[1] < binary_image.shape[1] and
                    0 <= ni[2] < binary_image.shape[2]):
                    
                    step = np.sqrt((neighbor[0] * self.spacing[0])**2 +
                                   (neighbor[1] * self.spacing[1])**2 +
                                   (neighbor[2] * self.spacing[2])**2)
                    
                    cost = step * (penalty[current] + penalty[ni]) / 2.0
                    new_dist = current_dist + cost
                    
                    if new_dist < dist[ni]:
                        dist[ni] = new_dist
                        prev[ni] = current
                        heapq.heappush(heap, (new_dist, ni))
        
        return dist, prev

    def extract_centerline_points(self, binary_image, distance_transform, threshold=0.5):
        dt = distance_transform.copy()
        dt[~binary_image] = 0
        
        max_filtered = ndimage.maximum_filter(dt, size=3)
        ridge_points = (dt == max_filtered) & (dt > 0)
        
        dt_normalized = dt / (np.max(dt) + 1e-8)
        thresholded = dt_normalized > threshold
        
        centerline_candidates = ridge_points & thresholded & binary_image
        
        labeled, num_labels = ndimage.label(centerline_candidates)
        sizes = ndimage.sum(centerline_candidates, labeled, range(1, num_labels + 1))
        
        if len(sizes) > 0:
            largest_label = np.argmax(sizes) + 1
            main_centerline = labeled == largest_label
        else:
            main_centerline = centerline_candidates
        
        return main_centerline

    def thin_centerline(self, binary_centerline):
        skeleton = morphology.skeletonize_3d(binary_centerline)
        return skeleton

    def trace_centerline(self, skeleton, start_point=None):
        if start_point is None:
            endpoints = self._find_endpoints(skeleton)
            if len(endpoints) == 0:
                return []
            start_point = endpoints[0]
        
        visited = np.zeros_like(skeleton, dtype=bool)
        path = []
        stack = [tuple(start_point)]
        
        while stack:
            current = stack.pop()
            if visited[current]:
                continue
            visited[current] = True
            path.append(np.array(current))
            
            neighbors = self._get_neighbors(current, skeleton)
            for neighbor in neighbors:
                if not visited[neighbor]:
                    stack.append(neighbor)
        
        return path

    def _find_endpoints(self, skeleton):
        endpoints = []
        indices = np.argwhere(skeleton)
        
        for idx in indices:
            idx_tuple = tuple(idx)
            neighbors = self._get_neighbors(idx_tuple, skeleton)
            if len(neighbors) == 1:
                endpoints.append(idx)
        
        return endpoints

    def _get_neighbors(self, point, mask):
        neighbors = []
        x, y, z = point
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if (0 <= nx < mask.shape[0] and
                        0 <= ny < mask.shape[1] and
                        0 <= nz < mask.shape[2] and
                        mask[nx, ny, nz]):
                        neighbors.append((nx, ny, nz))
        return neighbors

    def extract(self, binary_image, method='ridge'):
        dt = self.compute_distance_transform(binary_image)
        
        if method == 'ridge':
            centerline_mask = self.extract_centerline_points(binary_image, dt)
            skeleton = self.thin_centerline(centerline_mask)
            points = self.trace_centerline(skeleton)
        elif method == 'geodesic':
            sources = self.find_source_points(dt, binary_image, num_sources=1)
            dist, prev = self.compute_geodesic_distance(binary_image, sources, dt)
            
            max_dist = np.max(dist[np.isfinite(dist)])
            end_point = np.unravel_index(np.argmax(dist == max_dist), dist.shape)
            
            points = self._backtrack_path(prev, end_point)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            'distance_transform': dt,
            'centerline_points': np.array(points),
            'method': method
        }

    def _backtrack_path(self, prev, end_point):
        path = []
        current = end_point
        
        while current[0] != -1:
            path.append(np.array(current))
            current = tuple(prev[current])
        
        return path[::-1]
