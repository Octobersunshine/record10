import numpy as np
from surface_code import SurfaceCode
from mwpm_decoder import MWPMDecoder
from error_model import ErrorModel, DepolarizingErrorModel


class SurfaceCodeSimulator:
    def __init__(self, distance, error_model=None):
        self.distance = distance
        self.sc = SurfaceCode(distance)
        self.decoder = MWPMDecoder(self.sc)
        
        if error_model is None:
            self.error_model = ErrorModel(p_bit_flip=0.01, p_phase_flip=0.01)
        else:
            self.error_model = error_model
    
    def _get_effective_error_rate(self, error_model):
        if hasattr(error_model, 'p'):
            return error_model.p
        elif hasattr(error_model, 'p_bit_flip') and hasattr(error_model, 'p_phase_flip'):
            return max(error_model.p_bit_flip, error_model.p_phase_flip)
        return 0.01
    
    def run_single_trial(self, error_model=None):
        if error_model is None:
            error_model = self.error_model
        
        p_eff = self._get_effective_error_rate(error_model)
        self.decoder.set_error_rate(p_eff)
        
        self.sc.reset()
        
        error_model.apply_errors(self.sc)
        
        self.sc.measure_stabilizers()
        
        x_matching = self.decoder.decode('x')
        self.decoder.apply_correction(x_matching, 'x')
        
        z_matching = self.decoder.decode('z')
        self.decoder.apply_correction(z_matching, 'z')
        
        x_logical, z_logical = self.sc.get_logical_error()
        
        return x_logical or z_logical
    
    def estimate_logical_error_rate(self, n_trials, error_model=None, verbose=False):
        logical_errors = 0
        
        for i in range(n_trials):
            if self.run_single_trial(error_model):
                logical_errors += 1
            
            if verbose and (i + 1) % max(1, n_trials // 10) == 0:
                progress = (i + 1) / n_trials * 100
                print(f"Progress: {progress:.0f}% - Logical errors: {logical_errors}/{i+1}")
        
        p_logical = logical_errors / n_trials
        error = np.sqrt(p_logical * (1 - p_logical) / n_trials)
        
        return p_logical, error
    
    def threshold_scan(self, distances, error_rates, n_trials, error_type='bit_flip', verbose=True):
        results = {}
        
        for d in distances:
            results[d] = {'p_phys': [], 'p_logical': [], 'error': []}
            
            simulator = SurfaceCodeSimulator(d)
            
            for p in error_rates:
                if error_type == 'bit_flip':
                    error_model = ErrorModel(p_bit_flip=p, p_phase_flip=0)
                elif error_type == 'phase_flip':
                    error_model = ErrorModel(p_bit_flip=0, p_phase_flip=p)
                elif error_type == 'both':
                    error_model = ErrorModel(p_bit_flip=p, p_phase_flip=p)
                elif error_type == 'depolarizing':
                    error_model = DepolarizingErrorModel(p=p)
                else:
                    raise ValueError(f"Unknown error type: {error_type}")
                
                p_logical, err = simulator.estimate_logical_error_rate(n_trials, error_model)
                
                results[d]['p_phys'].append(p)
                results[d]['p_logical'].append(p_logical)
                results[d]['error'].append(err)
                
                if verbose:
                    print(f"d={d}, p_phys={p:.4f}, p_logical={p_logical:.6f} ± {err:.6f}")
        
        return results
