import numpy as np
import heapq
from scipy import ndimage
from skimage import morphology


class FastMarchingCenterline:
    def __init__(self, spacing=(1.0, 1.0, 1.0)):
        self.spacing = np.array(spacing, dtype=np.float64)

    def compute_speed_image(self, image, binary_mask=None, alpha=0.5):
        if binary_mask is not None:
            speed = np.zeros_like(image, dtype=np.float64)
            speed[binary_mask] = 1.0
        else:
            image_norm = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
            speed = alpha * image_norm + (1 - alpha) * 0.5
        
        return speed

    def fast_marching(self, speed_image, start_points, max_time=np.inf):
        shape = speed_image.shape
        time_map = np.full(shape, np.inf, dtype=np.float64)
        prev = np.full(shape + (3,), -1, dtype=np.int32)
        state = np.zeros(shape, dtype=np.int8)
        
        FAR, NARROW, ALIVE = 0, 1, 2
        
        heap = []
        
        for start in start_points:
            start = tuple(np.array(start).astype(int))
            if (0 <= start[0] < shape[0] and
                0 <= start[1] < shape[1] and
                0 <= start[2] < shape[2]):
                time_map[start] = 0
                state[start] = NARROW
                heapq.heappush(heap, (0.0, start))
        
        neighbors_26 = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbors_26.append((dx, dy, dz))
        
        while heap:
            current_time, current = heapq.heappop(heap)
            
            if state[current] == ALIVE:
                continue
            
            if current_time > max_time:
                break
            
            state[current] = ALIVE
            
            for neighbor in neighbors_26:
                ni = (current[0] + neighbor[0],
                      current[1] + neighbor[1],
                      current[2] + neighbor[2])
                
                if (0 <= ni[0] < shape[0] and
                    0 <= ni[1] < shape[1] and
                    0 <= ni[2] < shape[2] and
                    state[ni] != ALIVE and
                    np.isfinite(speed_image[ni])):
                    
                    step = np.sqrt((neighbor[0] * self.spacing[0])**2 +
                                   (neighbor[1] * self.spacing[1])**2 +
                                   (neighbor[2] * self.spacing[2])**2)
                    
                    speed = (speed_image[current] + speed_image[ni]) / 2.0
                    if speed <= 0:
                        continue
                    
                    new_time = current_time + step / speed
                    
                    if new_time < time_map[ni]:
                        time_map[ni] = new_time
                        prev[ni] = current
                        if state[ni] == FAR:
                            state[ni] = NARROW
                            heapq.heappush(heap, (new_time, ni))
                        else:
                            for i, (t, p) in enumerate(heap):
                                if p == ni:
                                    heap[i] = (new_time, ni)
                                    heapq.heapify(heap)
                                    break
        
        return time_map, prev, state

    def extract_minimal_path(self, prev, end_point):
        path = []
        current = tuple(np.array(end_point).astype(int))
        
        while current[0] != -1:
            path.append(np.array(current))
            prev_idx = prev[current]
            if prev_idx[0] == -1:
                break
            current = tuple(prev_idx)
        
        return path[::-1]

    def find_endpoints_from_time_map(self, time_map, binary_mask, num_endpoints=1):
        valid_times = time_map.copy()
        if binary_mask is not None:
            valid_times[~binary_mask] = -np.inf
        
        endpoints = []
        for _ in range(num_endpoints):
            if np.all(~np.isfinite(valid_times)):
                break
            endpoint = np.unravel_index(np.argmax(valid_times), valid_times.shape)
            if valid_times[endpoint] <= 0:
                break
            endpoints.append(np.array(endpoint))
            
            r = 5
            x, y, z = endpoint
            x_slice = slice(max(0, x - r), min(valid_times.shape[0], x + r + 1))
            y_slice = slice(max(0, y - r), min(valid_times.shape[1], y + r + 1))
            z_slice = slice(max(0, z - r), min(valid_times.shape[2], z + r + 1))
            valid_times[x_slice, y_slice, z_slice] = -np.inf
        
        return endpoints

    def find_vessel_seed_points(self, binary_image, distance_transform=None, num_seeds=2):
        if distance_transform is None:
            distance_transform = ndimage.distance_transform_edt(
                binary_image, sampling=self.spacing
            )
        
        dt = distance_transform.copy()
        dt[~binary_image] = -1
        
        seeds = []
        for _ in range(num_seeds):
            if np.max(dt) <= 0:
                break
            seed = np.unravel_index(np.argmax(dt), dt.shape)
            seeds.append(np.array(seed))
            
            r = max(5, int(dt[seed] / np.min(self.spacing)))
            x, y, z = seed
            x_slice = slice(max(0, x - r), min(dt.shape[0], x + r + 1))
            y_slice = slice(max(0, y - r), min(dt.shape[1], y + r + 1))
            z_slice = slice(max(0, z - r), min(dt.shape[2], z + r + 1))
            dt[x_slice, y_slice, z_slice] = -1
        
        return seeds

    def extract_centerline_from_path(self, path, binary_mask, num_points=None):
        if len(path) == 0:
            return np.array([])
        
        path_arr = np.array(path)
        
        if num_points is not None and len(path) > num_points:
            indices = np.linspace(0, len(path) - 1, num_points, dtype=int)
            path_arr = path_arr[indices]
        
        return path_arr

    def multiple_fast_marching(self, speed_image, binary_mask, num_sources=2, num_paths=5):
        dt = ndimage.distance_transform_edt(binary_mask, sampling=self.spacing)
        source_points = self.find_vessel_seed_points(binary_mask, dt, num_seeds=num_sources)
        
        if len(source_points) == 0:
            return []
        
        all_paths = []
        
        for source in source_points:
            time_map, prev, _ = self.fast_marching(speed_image, [source])
            
            endpoints = self.find_endpoints_from_time_map(time_map, binary_mask, num_endpoints=num_paths)
            
            for endpoint in endpoints:
                path = self.extract_minimal_path(prev, endpoint)
                if len(path) > 5:
                    all_paths.append(path)
        
        return all_paths

    def merge_paths(self, paths, threshold=2.0):
        if len(paths) == 0:
            return []
        
        all_points = []
        for path in paths:
            all_points.extend(path)
        all_points = np.array(all_points)
        
        merged_mask = np.zeros(tuple(int(np.max(all_points[:, i])) + 3 for i in range(3)), dtype=bool)
        for point in all_points:
            x, y, z = point.astype(int)
            if (0 <= x < merged_mask.shape[0] and
                0 <= y < merged_mask.shape[1] and
                0 <= z < merged_mask.shape[2]):
                merged_mask[x, y, z] = True
        
        skeleton = morphology.skeletonize_3d(merged_mask)
        skeleton_points = np.argwhere(skeleton)
        
        return skeleton_points

    def extract(self, image, binary_mask, source_points=None, end_points=None):
        speed = self.compute_speed_image(image, binary_mask)
        
        if source_points is None:
            dt = ndimage.distance_transform_edt(binary_mask, sampling=self.spacing)
            source_points = self.find_vessel_seed_points(binary_mask, dt, num_seeds=1)
        
        if len(source_points) == 0:
            return {
                'time_map': None,
                'centerline_points': np.array([]),
                'speed_image': speed
            }
        
        time_map, prev, state = self.fast_marching(speed, source_points)
        
        if end_points is None:
            end_points = self.find_endpoints_from_time_map(time_map, binary_mask, num_endpoints=1)
        
        all_paths = []
        for end_point in end_points:
            path = self.extract_minimal_path(prev, end_point)
            if len(path) > 5:
                all_paths.append(path)
        
        if len(all_paths) > 1:
            centerline_points = self.merge_paths(all_paths)
        elif len(all_paths) == 1:
            centerline_points = np.array(all_paths[0])
        else:
            centerline_points = np.array([])
        
        return {
            'time_map': time_map,
            'centerline_points': centerline_points,
            'speed_image': speed,
            'source_points': source_points,
            'end_points': end_points,
            'paths': all_paths
        }
