import numpy as np
from itertools import combinations


class MWPMDecoder:
    def __init__(self, surface_code, p_error=0.01):
        self.sc = surface_code
        self.d = surface_code.d
        self.p_error = p_error
    
    def set_error_rate(self, p_error):
        self.p_error = p_error
    
    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _add_boundary_defects(self, defects, stab_type):
        d = self.d
        boundary_defects = []
        boundary_positions = []
        
        for defect in defects:
            pos = self.sc.get_stab_position(defect, stab_type)
            boundary_defects.append(defect)
            boundary_positions.append(pos)
        
        virtual_defects = []
        virtual_positions = []
        
        if stab_type == 'x':
            top_virtual = len(boundary_defects)
            bottom_virtual = len(boundary_defects) + 1
            virtual_defects.append(top_virtual)
            virtual_defects.append(bottom_virtual)
            virtual_positions.append((-0.5, (d - 1) / 2.0))
            virtual_positions.append((d - 0.5, (d - 1) / 2.0))
        else:
            left_virtual = len(boundary_defects)
            right_virtual = len(boundary_defects) + 1
            virtual_defects.append(left_virtual)
            virtual_defects.append(right_virtual)
            virtual_positions.append(((d - 1) / 2.0, -0.5))
            virtual_positions.append(((d - 1) / 2.0, d - 0.5))
        
        all_defects = boundary_defects + virtual_defects
        all_positions = boundary_positions + virtual_positions
        
        return all_defects, all_positions
    
    def _negative_log_likelihood(self, distance):
        if distance <= 0:
            return 0.0
        p = self.p_error
        return -distance * np.log(p)
    
    def _build_matching_graph(self, defects, positions):
        n = len(defects)
        graph = {}
        
        for i, j in combinations(range(n), 2):
            distance = self.manhattan_distance(positions[i], positions[j])
            weight = self._negative_log_likelihood(distance)
            graph[(i, j)] = weight
            graph[(j, i)] = weight
        
        return graph
    
    def _greedy_matching(self, graph, n):
        if n == 0:
            return []
        
        matched = [False] * n
        pairs = []
        
        edges = []
        for (i, j), weight in graph.items():
            if i < j:
                edges.append((weight, i, j))
        
        edges.sort()
        
        for weight, i, j in edges:
            if not matched[i] and not matched[j]:
                pairs.append((i, j))
                matched[i] = True
                matched[j] = True
        
        return pairs
    
    def decode(self, stab_type='x'):
        defects = self.sc.get_defects(stab_type)
        
        if len(defects) == 0:
            return []
        
        all_defects, all_positions = self._add_boundary_defects(defects, stab_type)
        graph = self._build_matching_graph(all_defects, all_positions)
        
        n = len(all_defects)
        matching = self._greedy_matching(graph, n)
        
        real_matching = []
        for i, j in matching:
            if i < len(defects) and j < len(defects):
                real_matching.append((defects[i], defects[j]))
        
        return real_matching
    
    def get_correction_path(self, stab1, stab2, stab_type):
        pos1 = self.sc.get_stab_position(stab1, stab_type)
        pos2 = self.sc.get_stab_position(stab2, stab_type)
        
        path = []
        r1, c1 = pos1
        r2, c2 = pos2
        
        r, c = int(r1), int(c1)
        while r != int(r2):
            if r < int(r2):
                r += 1
            else:
                r -= 1
            path.append((r, c))
        
        while c != int(c2):
            if c < int(c2):
                c += 1
            else:
                c -= 1
            path.append((r, c))
        
        return path
    
    def apply_correction(self, matching, stab_type='x'):
        d = self.d
        
        for stab1, stab2 in matching:
            path = self.get_correction_path(stab1, stab2, stab_type)
            
            for (r, c) in path:
                if 0 <= r < d and 0 <= c < d:
                    qubit_idx = r * d + c
                    if stab_type == 'x':
                        self.sc.apply_bit_flip(qubit_idx)
                    else:
                        self.sc.apply_phase_flip(qubit_idx)
