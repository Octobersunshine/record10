import numpy as np
import time
from surface_code import SurfaceCode
from mwpm_decoder import MWPMDecoder
from nn_decoder import NeuralDecoder, FastNeuralDecoder, GNNDecoderTrainer
from error_model import ErrorModel, DepolarizingErrorModel


class NeuralSurfaceCodeSimulator:
    def __init__(self, distance, decoder_type='fast', error_model=None):
        self.distance = distance
        self.sc = SurfaceCode(distance)
        self.decoder_type = decoder_type
        
        if decoder_type == 'mwpm':
            self.decoder = MWPMDecoder(self.sc)
        elif decoder_type == 'gnn':
            self.decoder = NeuralDecoder(distance)
        elif decoder_type == 'fast':
            self.decoder = FastNeuralDecoder(distance)
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")
        
        if error_model is None:
            self.error_model = ErrorModel(p_bit_flip=0.01, p_phase_flip=0.01)
        else:
            self.error_model = error_model
        
        self.decoding_times = []
    
    def train_decoder(self, n_samples=5000, p_error=0.05, verbose=True):
        if self.decoder_type == 'mwpm':
            if verbose:
                print("MWPM decoder does not require training")
            return
        
        if verbose:
            print(f"Training {self.decoder_type.upper()} decoder...")
            print(f"  Distance: {self.distance}")
            print(f"  Samples: {n_samples}")
            print(f"  Error rate: {p_error}")
        
        self.decoder.train(n_samples=n_samples, p_error=p_error, verbose=verbose)
        
        if verbose:
            print("Training complete!")
    
    def run_single_trial(self, error_model=None, measure_time=False):
        if error_model is None:
            error_model = self.error_model
        
        self.sc.reset()
        
        error_model.apply_errors(self.sc)
        
        x_syndrome, z_syndrome = self.sc.measure_stabilizers()
        
        if measure_time:
            start_time = time.perf_counter()
        
        if self.decoder_type == 'mwpm':
            if hasattr(self.decoder, 'set_error_rate'):
                p_eff = max(getattr(error_model, 'p_bit_flip', 0.01),
                           getattr(error_model, 'p_phase_flip', 0.01))
                self.decoder.set_error_rate(p_eff)
            
            x_matching = self.decoder.decode('x')
            self.decoder.apply_correction(x_matching, 'x')
            
            z_matching = self.decoder.decode('z')
            self.decoder.apply_correction(z_matching, 'z')
        else:
            x_correction, z_correction = self.decoder.decode(x_syndrome, z_syndrome)
            self.decoder.apply_correction(self.sc, x_correction, z_correction)
        
        if measure_time:
            elapsed = time.perf_counter() - start_time
            self.decoding_times.append(elapsed)
        
        x_logical, z_logical = self.sc.get_logical_error()
        
        return x_logical or z_logical
    
    def estimate_logical_error_rate(self, n_trials, error_model=None, verbose=False, measure_time=False):
        logical_errors = 0
        self.decoding_times = []
        
        for i in range(n_trials):
            if self.run_single_trial(error_model, measure_time=measure_time):
                logical_errors += 1
            
            if verbose and (i + 1) % max(1, n_trials // 10) == 0:
                progress = (i + 1) / n_trials * 100
                print(f"Progress: {progress:.0f}% - Logical errors: {logical_errors}/{i+1}")
        
        p_logical = logical_errors / n_trials
        error = np.sqrt(p_logical * (1 - p_logical) / n_trials)
        
        avg_time = np.mean(self.decoding_times) if self.decoding_times else 0
        
        return p_logical, error, avg_time
    
    def get_average_decoding_time(self):
        if not self.decoding_times:
            return 0
        return np.mean(self.decoding_times)


def compare_decoders(distance=3, p_error=0.05, n_trials=200, n_train_samples=1000):
    print("=" * 70)
    print("解码器性能对比")
    print("=" * 70)
    
    error_model = ErrorModel(p_bit_flip=p_error, p_phase_flip=p_error)
    
    results = {}
    
    for decoder_type in ['mwpm', 'fast']:
        print(f"\n{'='*70}")
        print(f"测试解码器: {decoder_type.upper()}")
        print(f"{'='*70}")
        
        sim = NeuralSurfaceCodeSimulator(distance, decoder_type=decoder_type, error_model=error_model)
        
        if decoder_type != 'mwpm':
            print(f"\n训练中... (n={n_train_samples} samples)")
            sim.train_decoder(n_samples=n_train_samples, p_error=p_error, verbose=False)
            print("训练完成!")
        
        print(f"\n运行测试中... (n={n_trials} trials)")
        p_logical, err, avg_time = sim.estimate_logical_error_rate(
            n_trials, error_model, verbose=False, measure_time=True
        )
        
        results[decoder_type] = {
            'p_logical': p_logical,
            'error': err,
            'avg_time': avg_time
        }
        
        print(f"  逻辑错误率: {p_logical:.6f} ± {err:.6f}")
        print(f"  平均解码时间: {avg_time*1000:.2f} ms")
    
    print(f"\n{'='*70}")
    print("性能汇总:")
    print(f"{'='*70}")
    print(f"{'解码器':<10} {'逻辑错误率':<15} {'平均时间(ms)':<15} {'加速比':<10}")
    print("-" * 70)
    
    mwpm_time = results.get('mwpm', {}).get('avg_time', 1)
    
    for decoder_type, res in results.items():
        speedup = mwpm_time / res['avg_time'] if res['avg_time'] > 0 else float('inf')
        print(f"{decoder_type:<10} {res['p_logical']:<15.6f} {res['avg_time']*1000:<15.2f} {speedup:<10.1f}x")
    
    print("=" * 70)
    
    return results


def benchmark_scaling(max_distance=7, p_error=0.05, n_trials=100):
    print("\n" + "=" * 70)
    print("解码速度可扩展性测试")
    print("=" * 70)
    
    distances = [3, 5, 7]
    distances = [d for d in distances if d <= max_distance]
    
    error_model = ErrorModel(p_bit_flip=p_error, p_phase_flip=p_error)
    
    results = {'d': [], 'mwpm_time': [], 'fast_time': []}
    
    for d in distances:
        print(f"\n码距 d={d}")
        
        sim_mwpm = NeuralSurfaceCodeSimulator(d, 'mwpm', error_model)
        _, _, mwpm_time = sim_mwpm.estimate_logical_error_rate(
            n_trials, error_model, verbose=False, measure_time=True
        )
        
        sim_fast = NeuralSurfaceCodeSimulator(d, 'fast', error_model)
        sim_fast.train_decoder(n_samples=500, p_error=p_error, verbose=False)
        _, _, fast_time = sim_fast.estimate_logical_error_rate(
            n_trials, error_model, verbose=False, measure_time=True
        )
        
        results['d'].append(d)
        results['mwpm_time'].append(mwpm_time * 1000)
        results['fast_time'].append(fast_time * 1000)
        
        speedup = mwpm_time / fast_time if fast_time > 0 else float('inf')
        print(f"  MWPM: {mwpm_time*1000:.2f} ms")
        print(f"  Fast NN: {fast_time*1000:.2f} ms")
        print(f"  加速比: {speedup:.1f}x")
    
    return results


if __name__ == "__main__":
    compare_decoders(distance=3, p_error=0.05, n_trials=200, n_train_samples=1000)
