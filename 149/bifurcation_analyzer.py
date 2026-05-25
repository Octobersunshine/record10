import numpy as np
import networkx as nx
from scipy import ndimage, spatial
from scipy.signal import savgol_filter
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class BifurcationAnalyzer:
    def __init__(self, spacing=(1.0, 1.0, 1.0)):
        self.spacing = np.array(spacing, dtype=np.float64)
        self.bifurcations = []
        self.bifurcation_graph = None

    def detect_bifurcations_from_topology(self, topology_tree):
        self.bifurcations = []
        
        junction_nodes = [n for n in topology_tree.graph.nodes() 
                         if topology_tree.graph.nodes[n]['type'] == 'junction']
        
        for node_id in junction_nodes:
            node_info = topology_tree.node_info[node_id]
            position = node_info['position']
            index = node_info['index']
            radius = node_info['radius']
            
            neighbors = list(topology_tree.graph.neighbors(node_id))
            
            bifurcation = {
                'node_id': node_id,
                'position': position,
                'index': index,
                'radius': radius,
                'level': node_info.get('level', -1),
                'num_branches': len(neighbors),
                'branch_nodes': neighbors,
                'branch_directions': [],
                'branch_angles': [],
                'branch_radii': [],
                'branch_lengths': [],
                'detection_method': 'topology'
            }
            
            for neighbor in neighbors:
                neighbor_info = topology_tree.node_info[neighbor]
                direction = neighbor_info['position'] - position
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 0:
                    direction = direction / direction_norm
                bifurcation['branch_directions'].append(direction)
                bifurcation['branch_radii'].append(neighbor_info['radius'])
            
            self._compute_bifurcation_angles(bifurcation)
            self.bifurcations.append(bifurcation)
        
        return self.bifurcations

    def _compute_bifurcation_angles(self, bifurcation):
        directions = bifurcation['branch_directions']
        n = len(directions)
        
        angles = []
        for i in range(n):
            for j in range(i + 1, n):
                cos_theta = np.dot(directions[i], directions[j])
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angle = np.arccos(cos_theta) * 180.0 / np.pi
                
                angles.append({
                    'branch_pair': (i, j),
                    'angle_deg': angle,
                    'angle_rad': angle * np.pi / 180.0
                })
        
        bifurcation['branch_angles'] = angles
        
        if len(angles) > 0:
            bifurcation['min_angle'] = min(a['angle_deg'] for a in angles)
            bifurcation['max_angle'] = max(a['angle_deg'] for a in angles)
            bifurcation['mean_angle'] = np.mean([a['angle_deg'] for a in angles])
        
        return bifurcation

    def detect_bifurcations_from_centerline(self, centerline_points, binary_mask=None, 
                                            distance_transform=None, window_size=5):
        points = np.array(centerline_points) * self.spacing
        n_points = len(points)
        
        if n_points < window_size * 2:
            return []
        
        curvatures = self._compute_curvature(points, window_size)
        
        if distance_transform is not None:
            radii = []
            for pt in centerline_points:
                idx = tuple(np.array(pt).astype(int))
                if all(0 <= idx[i] < distance_transform.shape[i] for i in range(3)):
                    radii.append(distance_transform[idx])
                else:
                    radii.append(0)
            radii = np.array(radii)
            radius_changes = np.abs(np.gradient(radii))
        else:
            radius_changes = np.zeros(n_points)
        
        curvature_threshold = np.percentile(curvatures, 90)
        radius_threshold = np.percentile(radius_changes, 90)
        
        candidate_indices = np.where(
            (curvatures > curvature_threshold) | 
            (radius_changes > radius_threshold)
        )[0]
        
        candidate_indices = self._suppress_non_maxima(
            candidate_indices, curvatures + radius_changes, min_distance=window_size
        )
        
        self.bifurcations = []
        for idx in candidate_indices:
            bifurcation = {
                'position': points[idx],
                'index': centerline_points[idx],
                'curvature': curvatures[idx],
                'radius_change': radius_changes[idx] if len(radius_changes) > idx else 0,
                'num_branches': self._count_local_branches(centerline_points, idx, window_size),
                'detection_method': 'curvature_radius'
            }
            self.bifurcations.append(bifurcation)
        
        return self.bifurcations

    def _compute_curvature(self, points, window_size=5):
        n_points = len(points)
        curvatures = np.zeros(n_points)
        
        half_window = window_size // 2
        
        for i in range(half_window, n_points - half_window):
            window_points = points[i - half_window:i + half_window + 1]
            
            if len(window_points) < 3:
                continue
            
            try:
                t = np.arange(len(window_points))
                x_smooth = savgol_filter(window_points[:, 0], window_size, 3)
                y_smooth = savgol_filter(window_points[:, 1], window_size, 3)
                z_smooth = savgol_filter(window_points[:, 2], window_size, 3)
                
                dx = np.gradient(x_smooth)
                dy = np.gradient(y_smooth)
                dz = np.gradient(z_smooth)
                ddx = np.gradient(dx)
                ddy = np.gradient(dy)
                ddz = np.gradient(dz)
                
                mid = len(dx) // 2
                cross = np.array([
                    dy[mid] * ddz[mid] - dz[mid] * ddy[mid],
                    dz[mid] * ddx[mid] - dx[mid] * ddz[mid],
                    dx[mid] * ddy[mid] - dy[mid] * ddx[mid]
                ])
                cross_norm = np.linalg.norm(cross)
                tangent_norm = np.linalg.norm([dx[mid], dy[mid], dz[mid]])
                
                if tangent_norm > 1e-8:
                    curvatures[i] = cross_norm / (tangent_norm ** 3)
            except:
                pass
        
        return curvatures

    def _suppress_non_maxima(self, indices, scores, min_distance=3):
        if len(indices) == 0:
            return []
        
        sorted_indices = indices[np.argsort(scores[indices])[::-1]]
        selected = []
        selected_indices = set()
        
        for idx in sorted_indices:
            if idx not in selected_indices:
                selected.append(idx)
                for i in range(max(0, idx - min_distance), idx + min_distance + 1):
                    selected_indices.add(i)
        
        return np.array(selected)

    def _count_local_branches(self, points, center_idx, radius=5):
        center_point = points[center_idx]
        
        neighbors = []
        for i, point in enumerate(points):
            if i == center_idx:
                continue
            dist = np.linalg.norm(np.array(point) - np.array(center_point))
            if dist <= radius:
                neighbors.append(i)
        
        if len(neighbors) <= 2:
            return len(neighbors) + 1
        
        return len(neighbors)

    def analyze_bifurcation_morphology(self, bifurcation, segments):
        node_id = bifurcation['node_id']
        
        connected_segments = []
        for seg in segments:
            if seg['start_node'] == node_id or seg['end_node'] == node_id:
                connected_segments.append(seg)
        
        if len(connected_segments) < 2:
            return bifurcation
        
        parent_segment = None
        child_segments = []
        
        for seg in connected_segments:
            if seg['start_node'] == node_id:
                child_segments.append(seg)
            else:
                parent_segment = seg
        
        if parent_segment is None and len(connected_segments) >= 1:
            parent_segment = connected_segments[0]
            child_segments = connected_segments[1:]
        
        if parent_segment is not None:
            bifurcation['parent_radius'] = parent_segment['avg_radius']
            bifurcation['parent_length'] = parent_segment['length']
        
        if len(child_segments) > 0:
            bifurcation['child_radii'] = [seg['avg_radius'] for seg in child_segments]
            bifurcation['child_lengths'] = [seg['length'] for seg in child_segments]
            
            if 'parent_radius' in bifurcation and bifurcation['parent_radius'] > 0:
                bifurcation['radius_ratio'] = [
                    r / bifurcation['parent_radius'] for r in bifurcation['child_radii']
                ]
                
                bifurcation['area_ratio'] = sum(
                    (r / bifurcation['parent_radius']) ** 2 
                    for r in bifurcation['child_radii']
                )
        
        return bifurcation

    def compute_bifurcation_risk_score(self, bifurcation):
        score = 0.0
        factors = {}
        
        if 'mean_angle' in bifurcation:
            if bifurcation['mean_angle'] < 30:
                score += 2.0
                factors['sharp_angle'] = 2.0
            elif bifurcation['mean_angle'] < 45:
                score += 1.0
                factors['moderate_angle'] = 1.0
        
        if 'area_ratio' in bifurcation:
            if bifurcation['area_ratio'] > 1.2:
                score += 1.5
                factors['high_area_ratio'] = 1.5
            elif bifurcation['area_ratio'] < 0.7:
                score += 1.0
                factors['low_area_ratio'] = 1.0
        
        if bifurcation['num_branches'] >= 4:
            score += 1.0
            factors['multi_branch'] = 1.0
        
        if 'radius' in bifurcation and bifurcation['radius'] > 0:
            for child_r in bifurcation.get('child_radii', []):
                if child_r > 0 and abs(child_r - bifurcation['radius']) / bifurcation['radius'] > 0.5:
                    score += 0.5
                    factors['radius_mismatch'] = 0.5
                    break
        
        bifurcation['risk_score'] = score
        bifurcation['risk_factors'] = factors
        bifurcation['risk_level'] = 'low' if score < 1.5 else 'medium' if score < 3.0 else 'high'
        
        return bifurcation

    def comprehensive_bifurcation_analysis(self, topology_tree):
        print("\n" + "=" * 70)
        print("COMPREHENSIVE BIFURCATION ANALYSIS")
        print("=" * 70)
        
        bifurcations = self.detect_bifurcations_from_topology(topology_tree)
        segments = topology_tree.segments
        
        print(f"\nDetected {len(bifurcations)} bifurcation points")
        
        for i, bifurcation in enumerate(bifurcations):
            self.analyze_bifurcation_morphology(bifurcation, segments)
            self.compute_bifurcation_risk_score(bifurcation)
            
            print(f"\n--- Bifurcation #{i + 1} (Node: {bifurcation['node_id']}) ---")
            print(f"  Position: {bifurcation['position']}")
            print(f"  Number of branches: {bifurcation['num_branches']}")
            print(f"  Bifurcation radius: {bifurcation['radius']:.2f}")
            print(f"  Tree level: {bifurcation['level']}")
            
            if 'mean_angle' in bifurcation:
                print(f"  Mean branch angle: {bifurcation['mean_angle']:.1f}°")
                print(f"  Min/Max angle: {bifurcation['min_angle']:.1f}° / {bifurcation['max_angle']:.1f}°")
            
            if 'area_ratio' in bifurcation:
                print(f"  Area ratio (sum(child²)/parent²): {bifurcation['area_ratio']:.3f}")
            
            print(f"  Risk level: {bifurcation['risk_level'].upper()} (score: {bifurcation['risk_score']:.1f})")
            
            if bifurcation['risk_factors']:
                print(f"  Risk factors: {', '.join(bifurcation['risk_factors'].keys())}")
        
        return bifurcations

    def visualize_bifurcations_3d(self, topology_tree, figsize=(14, 10), save_path=None):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        positions = np.array([topology_tree.node_info[n]['position'] 
                             for n in topology_tree.graph.nodes()])
        
        for seg in topology_tree.segments:
            seg_positions = np.array([topology_tree.node_info[n]['position'] for n in seg['nodes']])
            ax.plot(seg_positions[:, 0], seg_positions[:, 1], seg_positions[:, 2], 
                    'b-', linewidth=2, alpha=0.5)
        
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                   c='lightgray', s=20, alpha=0.3)
        
        risk_colors = {'low': 'green', 'medium': 'orange', 'high': 'red'}
        
        for i, bifurcation in enumerate(self.bifurcations):
            pos = bifurcation['position']
            color = risk_colors.get(bifurcation.get('risk_level', 'low'), 'blue')
            size = 100 + bifurcation['num_branches'] * 30
            
            ax.scatter(pos[0], pos[1], pos[2], c=color, s=size, marker='o',
                       edgecolors='black', linewidths=2, zorder=10,
                       label=f"Bifurcation #{i+1} ({bifurcation['risk_level']})" 
                       if i < 5 else "")
            
            if 'branch_directions' in bifurcation:
                for j, direction in enumerate(bifurcation['branch_directions']):
                    end_pos = pos + direction * 15
                    ax.plot([pos[0], end_pos[0]], [pos[1], end_pos[1]], [pos[2], end_pos[2]],
                            color=color, linestyle='--', linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Vessel Bifurcations with Risk Assessment')
        
        legend_elements = [
            plt.scatter([], [], c='green', s=100, label='Low Risk'),
            plt.scatter([], [], c='orange', s=100, label='Medium Risk'),
            plt.scatter([], [], c='red', s=100, label='High Risk')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Bifurcation visualization saved to {save_path}")
        
        plt.close()
        return fig

    def generate_bifurcation_report(self, topology_tree, save_path=None):
        bifurcations = self.comprehensive_bifurcation_analysis(topology_tree)
        
        report = []
        report.append("=" * 70)
        report.append("VESSEL BIFURCATION ANALYSIS REPORT")
        report.append("=" * 70)
        report.append(f"")
        report.append(f"Total bifurcations detected: {len(bifurcations)}")
        
        risk_counts = defaultdict(int)
        for b in bifurcations:
            risk_counts[b.get('risk_level', 'unknown')] += 1
        report.append(f"Risk distribution:")
        for level in ['high', 'medium', 'low']:
            report.append(f"  - {level.upper()}: {risk_counts[level]}")
        report.append(f"")
        
        report.append("-" * 70)
        report.append("SURGICAL PLANNING RECOMMENDATIONS")
        report.append("-" * 70)
        
        high_risk = [b for b in bifurcations if b.get('risk_level') == 'high']
        medium_risk = [b for b in bifurcations if b.get('risk_level') == 'medium']
        
        if high_risk:
            report.append(f"\n⚠️  HIGH PRIORITY BIFURCATIONS ({len(high_risk)}):")
            for i, b in enumerate(high_risk):
                report.append(f"  #{i+1}: Node {b['node_id']} at level {b['level']}")
                report.append(f"    - Position: {np.round(b['position'], 2)}")
                report.append(f"    - Angles: {b.get('min_angle', 0):.1f}° / {b.get('max_angle', 0):.1f}°")
                report.append(f"    - Risk factors: {', '.join(b.get('risk_factors', {}).keys())}")
                report.append(f"    - Recommendation: Careful navigation, consider pre-dilation")
        
        if medium_risk:
            report.append(f"\n⚡ MEDIUM PRIORITY BIFURCATIONS ({len(medium_risk)}):")
            for i, b in enumerate(medium_risk[:5]):
                report.append(f"  #{i+1}: Node {b['node_id']} at level {b['level']}")
        
        report.append(f"\n" + "-" * 70)
        report.append("INTERVENTION PLANNING GUIDANCE")
        report.append("-" * 70)
        report.append(f"1. Approach order: Start from root, work distally")
        report.append(f"2. Stent sizing: Use parent vessel diameter at bifurcations")
        report.append(f"3. Side branch protection: Consider for angles < 45°")
        report.append(f"4. Wire selection: Supportive wires for sharp angles")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"\nReport saved to {save_path}")
        
        return report_text

    def export_bifurcations_to_csv(self, filename):
        import csv
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            headers = [
                'bifurcation_id', 'node_id', 'position_x', 'position_y', 'position_z',
                'num_branches', 'radius', 'level', 'mean_angle_deg', 'min_angle_deg',
                'max_angle_deg', 'area_ratio', 'risk_score', 'risk_level'
            ]
            writer.writerow(headers)
            
            for i, b in enumerate(self.bifurcations):
                row = [
                    i,
                    b.get('node_id', ''),
                    b['position'][0] if len(b['position']) > 0 else '',
                    b['position'][1] if len(b['position']) > 1 else '',
                    b['position'][2] if len(b['position']) > 2 else '',
                    b.get('num_branches', ''),
                    b.get('radius', ''),
                    b.get('level', ''),
                    b.get('mean_angle', ''),
                    b.get('min_angle', ''),
                    b.get('max_angle', ''),
                    b.get('area_ratio', ''),
                    b.get('risk_score', ''),
                    b.get('risk_level', '')
                ]
                writer.writerow(row)
        
        print(f"Bifurcation data exported to {filename}")
