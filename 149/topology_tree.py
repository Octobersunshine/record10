import numpy as np
import networkx as nx
from scipy import ndimage, spatial
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class VesselTopologyTree:
    def __init__(self, spacing=(1.0, 1.0, 1.0)):
        self.spacing = np.array(spacing, dtype=np.float64)
        self.graph = nx.Graph()
        self.root_node = None
        self.node_info = {}
        self.segments = []

    def build_from_skeleton(self, skeleton_mask, distance_transform=None):
        skeleton_points = np.argwhere(skeleton_mask)
        return self.build_from_centerline_points(
            skeleton_points, 
            binary_image=skeleton_mask, 
            distance_transform=distance_transform
        )

    def build_from_centerline_points(self, centerline_points, binary_image=None, 
                                      distance_transform=None, k_neighbors=6):
        if len(centerline_points) == 0:
            return None
        
        points = np.array(centerline_points, dtype=np.float64)
        points_spaced = points * self.spacing
        
        self.graph = nx.Graph()
        
        for i, point in enumerate(centerline_points):
            node_id = f"node_{i}"
            
            attrs = {
                'position': point * self.spacing,
                'index': np.array(point, dtype=int),
                'radius': 0.0
            }
            
            if binary_image is not None and distance_transform is not None:
                idx = tuple(np.array(point).astype(int))
                if (0 <= idx[0] < binary_image.shape[0] and
                    0 <= idx[1] < binary_image.shape[1] and
                    0 <= idx[2] < binary_image.shape[2] and
                    binary_image[idx]):
                    attrs['radius'] = distance_transform[idx]
            
            self.graph.add_node(node_id, **attrs)
            self.node_info[node_id] = attrs
        
        if len(points_spaced) > 1:
            tree = spatial.cKDTree(points_spaced)
            distances, indices = tree.query(points_spaced, k=min(k_neighbors + 1, len(points_spaced)))
            
            for i in range(len(points_spaced)):
                for j in range(1, len(indices[i])):
                    if distances[i, j] < np.max(self.spacing) * 3:
                        node1 = f"node_{i}"
                        node2 = f"node_{indices[i, j]}"
                        if not self.graph.has_edge(node1, node2):
                            self.graph.add_edge(node1, node2, weight=distances[i, j])
        
        self._compute_node_types()
        self._find_root()
        self._compute_hierarchy()
        self.segments = self.extract_vessel_segments()
        
        return self.graph

    def _compute_node_types(self):
        for node in self.graph.nodes():
            degree = self.graph.degree(node)
            if degree == 1:
                node_type = 'endpoint'
            elif degree == 2:
                node_type = 'internal'
            elif degree > 2:
                node_type = 'junction'
            else:
                node_type = 'isolated'
            
            self.graph.nodes[node]['type'] = node_type
            self.node_info[node]['type'] = node_type

    def _find_root(self):
        endpoints = [n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'endpoint']
        
        if len(endpoints) == 0:
            junctions = [n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'junction']
            if len(junctions) > 0:
                self.root_node = junctions[0]
            elif self.graph.number_of_nodes() > 0:
                self.root_node = list(self.graph.nodes())[0]
            else:
                self.root_node = None
            return
        
        max_eccentricity = -1
        root = endpoints[0]
        
        for endpoint in endpoints:
            try:
                lengths = nx.single_source_dijkstra_path_length(self.graph, endpoint)
                if len(lengths) > 0:
                    eccentricity = max(lengths.values())
                    if eccentricity > max_eccentricity:
                        max_eccentricity = eccentricity
                        root = endpoint
            except:
                continue
        
        self.root_node = root
        if root in self.node_info:
            self.node_info[root]['is_root'] = True

    def _compute_hierarchy(self):
        if self.root_node is None:
            return
        
        for node in self.graph.nodes():
            self.graph.nodes[node]['level'] = -1
        
        queue = deque([(self.root_node, 0)])
        visited = set([self.root_node])
        
        while queue:
            node, level = queue.popleft()
            self.graph.nodes[node]['level'] = level
            self.node_info[node]['level'] = level
            
            for neighbor in self.graph.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, level + 1))

    def extract_vessel_segments(self):
        if self.graph.number_of_nodes() == 0:
            return []
        
        if self.root_node is None:
            self._find_root()
        
        junctions = [n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'junction']
        endpoints = [n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'endpoint']
        
        critical_points = set(junctions + endpoints)
        if self.root_node and self.root_node not in critical_points:
            critical_points.add(self.root_node)
        
        segments = []
        visited_edges = set()
        
        for start_node in critical_points:
            for neighbor in self.graph.neighbors(start_node):
                edge_key = tuple(sorted([start_node, neighbor]))
                if edge_key in visited_edges:
                    continue
                
                segment = self._trace_segment(start_node, neighbor, critical_points, visited_edges)
                if len(segment) >= 2:
                    segments.append({
                        'nodes': segment,
                        'start_node': segment[0],
                        'end_node': segment[-1],
                        'start_position': self.node_info[segment[0]]['position'],
                        'end_position': self.node_info[segment[-1]]['position'],
                        'length': self._compute_segment_length(segment),
                        'avg_radius': self._compute_segment_avg_radius(segment),
                        'start_type': self.node_info[segment[0]]['type'],
                        'end_type': self.node_info[segment[-1]]['type']
                    })
        
        return segments

    def _trace_segment(self, start_node, first_neighbor, critical_points, visited_edges):
        segment = [start_node]
        current_node = first_neighbor
        prev_node = start_node
        
        while True:
            segment.append(current_node)
            edge_key = tuple(sorted([prev_node, current_node]))
            visited_edges.add(edge_key)
            
            if current_node in critical_points:
                break
            
            neighbors = list(self.graph.neighbors(current_node))
            next_nodes = [n for n in neighbors if n != prev_node]
            
            if len(next_nodes) == 0:
                break
            
            prev_node = current_node
            current_node = next_nodes[0]
        
        return segment

    def _compute_segment_length(self, segment_nodes):
        length = 0.0
        for i in range(len(segment_nodes) - 1):
            pos1 = self.node_info[segment_nodes[i]]['position']
            pos2 = self.node_info[segment_nodes[i + 1]]['position']
            length += np.linalg.norm(pos2 - pos1)
        return length

    def _compute_segment_avg_radius(self, segment_nodes):
        radii = [self.node_info[n]['radius'] for n in segment_nodes if self.node_info[n]['radius'] > 0]
        return np.mean(radii) if radii else 0.0

    def build_hierarchical_tree(self):
        if len(self.segments) == 0:
            self.segments = self.extract_vessel_segments()
        
        tree = nx.DiGraph()
        
        for segment in self.segments:
            start = segment['start_node']
            end = segment['end_node']
            
            for node in [start, end]:
                if not tree.has_node(node):
                    tree.add_node(node, **self.node_info[node])
            
            edge_attrs = {k: v for k, v in segment.items() 
                         if k not in ['nodes', 'start_node', 'end_node']}
            tree.add_edge(start, end, **edge_attrs)
        
        return tree

    def get_tree_statistics(self):
        if self.graph.number_of_nodes() == 0:
            return {}
        
        if len(self.segments) == 0:
            self.segments = self.extract_vessel_segments()
            
        junction_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'junction']
        end_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'endpoint']
        internal_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'internal']
        
        total_length = sum(s['length'] for s in self.segments)
        avg_segment_length = np.mean([s['length'] for s in self.segments]) if self.segments else 0
        
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_junctions': len(junction_nodes),
            'num_endpoints': len(end_nodes),
            'num_internal_nodes': len(internal_nodes),
            'num_segments': len(self.segments),
            'total_vessel_length': total_length,
            'average_segment_length': avg_segment_length,
            'max_level': max([d.get('level', 0) for d in self.node_info.values()]) if self.node_info else 0
        }

    def print_tree_summary(self):
        stats = self.get_tree_statistics()
        print("=" * 60)
        print("Vessel Topology Tree Summary")
        print("=" * 60)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key:30s}: {value:.4f}")
            else:
                print(f"  {key:30s}: {value}")
        
        if len(self.segments) > 0:
            print("-" * 60)
            print(f"  Segments Details:")
            for i, seg in enumerate(self.segments[:10]):
                print(f"    Segment {i:2d}: length={seg['length']:.2f}, "
                      f"radius={seg['avg_radius']:.2f}, "
                      f"{seg['start_type']} -> {seg['end_type']}")
            if len(self.segments) > 10:
                print(f"    ... and {len(self.segments) - 10} more segments")
        print("=" * 60)

    def visualize_3d(self, binary_image=None, figsize=(12, 10), save_path=None, 
                     show_segment_labels=False):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        if binary_image is not None:
            from skimage import measure
            try:
                verts, faces, _, _ = measure.marching_cubes(binary_image, level=0.5, spacing=self.spacing)
                ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                                alpha=0.1, color='cyan', edgecolor='none')
            except:
                pass
        
        positions = np.array([self.node_info[n]['position'] for n in self.graph.nodes()])
        
        cmap = plt.cm.get_cmap('tab10')
        for i, seg in enumerate(self.segments):
            color = cmap(i % 10)
            seg_positions = np.array([self.node_info[n]['position'] for n in seg['nodes']])
            ax.plot(seg_positions[:, 0], seg_positions[:, 1], seg_positions[:, 2], 
                    color=color, linewidth=2, alpha=0.8)
        
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                   c='r', s=15, alpha=0.6, label='Nodes')
        
        endpoints = [n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'endpoint']
        if endpoints:
            end_positions = np.array([self.node_info[n]['position'] for n in endpoints])
            ax.scatter(end_positions[:, 0], end_positions[:, 1], end_positions[:, 2], 
                       c='g', s=100, marker='*', label='Endpoints', zorder=10)
        
        junctions = [n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'junction']
        if junctions:
            junction_positions = np.array([self.node_info[n]['position'] for n in junctions])
            ax.scatter(junction_positions[:, 0], junction_positions[:, 1], junction_positions[:, 2], 
                       c='m', s=80, marker='s', label='Junctions', zorder=9)
        
        if self.root_node:
            root_pos = self.node_info[self.root_node]['position']
            ax.scatter([root_pos[0]], [root_pos[1]], [root_pos[2]], 
                       c='yellow', s=200, marker='o', label='Root', 
                       edgecolors='black', linewidths=2, zorder=11)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Vessel Topology Tree')
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.close()
        return fig

    def visualize_tree_graph(self, figsize=(14, 10), save_path=None, prog='neato'):
        plt.figure(figsize=figsize)
        
        try:
            if self.root_node:
                pos = nx.nx_agraph.graphviz_layout(self.graph, prog=prog, root=self.root_node)
            else:
                pos = nx.nx_agraph.graphviz_layout(self.graph, prog=prog)
        except:
            pos = nx.spring_layout(self.graph, k=2.0, iterations=50)
        
        node_colors = []
        for node in self.graph.nodes():
            ntype = self.graph.nodes[node]['type']
            if node == self.root_node:
                node_colors.append('gold')
            elif ntype == 'junction':
                node_colors.append('magenta')
            elif ntype == 'endpoint':
                node_colors.append('limegreen')
            elif ntype == 'internal':
                node_colors.append('cyan')
            else:
                node_colors.append('gray')
        
        node_sizes = []
        for node in self.graph.nodes():
            if node == self.root_node:
                node_sizes.append(600)
            elif self.graph.nodes[node]['type'] == 'junction':
                node_sizes.append(400)
            elif self.graph.nodes[node]['type'] == 'endpoint':
                node_sizes.append(300)
            else:
                node_sizes.append(100)
        
        nx.draw_networkx_edges(self.graph, pos, alpha=0.5, width=2)
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=node_sizes,
                               edgecolors='black', linewidths=0.5)
        
        label_pos = {k: (v[0], v[1] + 3) for k, v in pos.items()}
        label_dict = {n: f"{self.graph.nodes[n]['type'][:2].upper()}\nL{self.graph.nodes[n].get('level', '?')}" 
                      for n in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, label_pos, label_dict, font_size=8)
        
        legend_elements = [
            plt.scatter([], [], c='gold', s=300, label='Root'),
            plt.scatter([], [], c='magenta', s=200, label='Junction'),
            plt.scatter([], [], c='limegreen', s=150, label='Endpoint'),
            plt.scatter([], [], c='cyan', s=50, label='Internal')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title('Vessel Topology Tree Graph')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.close()

    def export_to_vtk(self, filename):
        try:
            import vtk
            from vtk.util import numpy_support
        except ImportError:
            print("VTK not available, cannot export")
            return
        
        points = vtk.vtkPoints()
        node_to_idx = {}
        for i, node in enumerate(self.graph.nodes()):
            pos = self.node_info[node]['position']
            points.InsertNextPoint(pos[0], pos[1], pos[2])
            node_to_idx[node] = i
        
        lines = vtk.vtkCellArray()
        for edge in self.graph.edges():
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, node_to_idx[edge[0]])
            line.GetPointIds().SetId(1, node_to_idx[edge[1]])
            lines.InsertNextCell(line)
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)
        
        node_type_array = vtk.vtkIntArray()
        node_type_array.SetName("NodeType")
        type_map = {'endpoint': 0, 'junction': 1, 'internal': 2, 'root': 3}
        for node in self.graph.nodes():
            ntype = self.graph.nodes[node]['type']
            if node == self.root_node:
                node_type_array.InsertNextValue(3)
            else:
                node_type_array.InsertNextValue(type_map.get(ntype, 2))
        polydata.GetPointData().AddArray(node_type_array)
        
        radius_array = vtk.vtkFloatArray()
        radius_array.SetName("Radius")
        for node in self.graph.nodes():
            radius_array.InsertNextValue(self.node_info[node]['radius'])
        polydata.GetPointData().AddArray(radius_array)
        
        level_array = vtk.vtkIntArray()
        level_array.SetName("Level")
        for node in self.graph.nodes():
            level_array.InsertNextValue(self.graph.nodes[node].get('level', -1))
        polydata.GetPointData().AddArray(level_array)
        
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(polydata)
        writer.Write()
        
        print(f"VTK file saved to {filename}")

    def export_to_graphml(self, filename):
        for node in self.graph.nodes():
            for key, value in self.node_info[node].items():
                if isinstance(value, np.ndarray):
                    self.graph.nodes[node][key] = str(value.tolist())
                elif isinstance(value, (np.int64, np.float64)):
                    self.graph.nodes[node][key] = float(value)
        
        nx.write_graphml(self.graph, filename)
        print(f"GraphML file saved to {filename}")
