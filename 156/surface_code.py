import numpy as np
from itertools import combinations
import random


class SurfaceCode:
    def __init__(self, distance):
        self.d = distance
        self.n_data = distance * distance
        self.n_ancilla_x = (distance - 1) * distance
        self.n_ancilla_z = distance * (distance - 1)
        
        self.data_qubits = np.zeros(self.n_data, dtype=bool)
        self.x_errors = np.zeros(self.n_data, dtype=bool)
        self.z_errors = np.zeros(self.n_data, dtype=bool)
        
        self.x_stabilizers = np.zeros(self.n_ancilla_z, dtype=bool)
        self.z_stabilizers = np.zeros(self.n_ancilla_x, dtype=bool)
        
        self._build_stabilizer_maps()
    
    def _build_stabilizer_maps(self):
        d = self.d
        
        self.z_stab_to_data = {}
        for i in range(d):
            for j in range(d - 1):
                ancilla_idx = i * (d - 1) + j
                data1 = i * d + j
                data2 = i * d + j + 1
                data3 = (i + 1) * d + j if i + 1 < d else None
                data4 = (i + 1) * d + j + 1 if i + 1 < d else None
                
                qubits = [data1, data2]
                if data3 is not None:
                    qubits.append(data3)
                if data4 is not None:
                    qubits.append(data4)
                self.z_stab_to_data[ancilla_idx] = qubits
        
        self.x_stab_to_data = {}
        for i in range(d - 1):
            for j in range(d):
                ancilla_idx = i * d + j
                data1 = i * d + j
                data2 = (i + 1) * d + j
                data3 = i * d + j + 1 if j + 1 < d else None
                data4 = (i + 1) * d + j + 1 if j + 1 < d else None
                
                qubits = [data1, data2]
                if data3 is not None:
                    qubits.append(data3)
                if data4 is not None:
                    qubits.append(data4)
                self.x_stab_to_data[ancilla_idx] = qubits
    
    def apply_bit_flip(self, qubit_idx):
        self.x_errors[qubit_idx] = not self.x_errors[qubit_idx]
    
    def apply_phase_flip(self, qubit_idx):
        self.z_errors[qubit_idx] = not self.z_errors[qubit_idx]
    
    def measure_stabilizers(self):
        d = self.d
        
        self.x_stabilizers.fill(False)
        for ancilla_idx, data_qubits in self.z_stab_to_data.items():
            parity = sum(self.x_errors[q] for q in data_qubits) % 2
            self.x_stabilizers[ancilla_idx] = parity
        
        self.z_stabilizers.fill(False)
        for ancilla_idx, data_qubits in self.x_stab_to_data.items():
            parity = sum(self.z_errors[q] for q in data_qubits) % 2
            self.z_stabilizers[ancilla_idx] = parity
        
        return self.x_stabilizers.copy(), self.z_stabilizers.copy()
    
    def get_defects(self, stab_type='x'):
        if stab_type == 'x':
            return np.where(self.x_stabilizers)[0]
        else:
            return np.where(self.z_stabilizers)[0]
    
    def get_stab_position(self, stab_idx, stab_type='x'):
        d = self.d
        if stab_type == 'x':
            row = stab_idx // (d - 1)
            col = stab_idx % (d - 1)
            return (row + 0.5, col + 0.5)
        else:
            row = stab_idx // d
            col = stab_idx % d
            return (row + 0.5, col + 0.5)
    
    def get_logical_error(self):
        d = self.d
        
        x_logical = sum(self.x_errors[i * d] for i in range(d)) % 2
        z_logical = sum(self.z_errors[i] for i in range(d)) % 2
        
        return x_logical, z_logical
    
    def reset(self):
        self.x_errors.fill(False)
        self.z_errors.fill(False)
        self.x_stabilizers.fill(False)
        self.z_stabilizers.fill(False)
