import numpy as np


class ErrorModel:
    def __init__(self, p_bit_flip=0.0, p_phase_flip=0.0):
        self.p_bit_flip = p_bit_flip
        self.p_phase_flip = p_phase_flip
    
    def apply_errors(self, surface_code):
        n_qubits = surface_code.n_data
        
        for i in range(n_qubits):
            if np.random.random() < self.p_bit_flip:
                surface_code.apply_bit_flip(i)
            
            if np.random.random() < self.p_phase_flip:
                surface_code.apply_phase_flip(i)


class DepolarizingErrorModel:
    def __init__(self, p=0.0):
        self.p = p
    
    def apply_errors(self, surface_code):
        n_qubits = surface_code.n_data
        p_single = self.p / 3.0
        
        for i in range(n_qubits):
            r = np.random.random()
            if r < p_single:
                surface_code.apply_bit_flip(i)
            elif r < 2 * p_single:
                surface_code.apply_phase_flip(i)
            elif r < self.p:
                surface_code.apply_bit_flip(i)
                surface_code.apply_phase_flip(i)
