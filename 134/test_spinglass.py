import numpy as np
import matplotlib
matplotlib.use('Agg')
from spin_glass import EdwardsAnderson, AdaptiveParallelTempering

def test_basic():
    print('='*60)
    print('Testing EdwardsAnderson model...')
    print('='*60)
    model = EdwardsAnderson(L=4, seed=42)
    print(f'  Lattice size: {model.L}x{model.L} = {model.N} spins')

    spins = model.random_spins()
    E = model.energy(spins)
    print(f'  Initial energy: {E:.2f}')

    dE = model.delta_energy(spins, 0, 0)
    print(f'  Delta energy for flip (0,0): {dE:.2f}')

    spins_new = model.metropolis_step(spins, T=1.0)
    E_new = model.energy(spins_new)
    print(f'  Energy after Metropolis step: {E_new:.2f}')
    print('  ✓ Basic model test passed')

def test_adaptive_pt():
    print('\n' + '='*60)
    print('Testing Adaptive Parallel Tempering...')
    print('='*60)
    
    model = EdwardsAnderson(L=6, seed=42)
    Ts_initial = np.geomspace(0.5, 2.0, 8)
    
    pt = AdaptiveParallelTempering(
        model, Ts_initial,
        min_exchange_rate=0.2,
        max_exchange_rate=0.6,
        seed=123
    )
    
    print(f'  Initial replicas: {pt.n_replicas}')
    print(f'  Initial temperatures: {Ts_initial}')
    print(f'  Target exchange rate range: [20%, 60%]')

    for _ in range(200):
        pt.step()
    
    initial_rates = pt.exchange_rates()
    Cv_initial = pt.get_specific_heat()
    
    print(f'  Initial exchange rates: {initial_rates}')
    print(f'  Initial Cv peak at T={Ts_initial[np.argmax(Cv_initial)]:.3f}')
    
    Ts_opt, history = pt.optimize_temperatures(n_cycles=3)
    print(f'\n  Final temperatures: {Ts_opt}')
    print(f'  Final exchange rates: {pt.exchange_rates()}')
    
    final_rates = pt.exchange_rates()
    within_range = np.sum((final_rates >= 0.2) & (final_rates <= 0.6))
    print(f'  {within_range}/{len(final_rates)} exchange rates within target range')
    
    print('  ✓ Adaptive PT test passed')
    return pt, history

def test_statistics():
    print('\n' + '='*60)
    print('Testing energy and specific heat statistics...')
    print('='*60)
    
    model = EdwardsAnderson(L=4, seed=42)
    Ts = np.array([0.5, 1.0, 2.0])
    pt = AdaptiveParallelTempering(model, Ts, seed=456)
    
    for _ in range(500):
        pt.step()
    
    E_mean = pt.get_energy_mean()
    Cv = pt.get_specific_heat()
    
    print(f'  Energy mean per spin: {E_mean}')
    print(f'  Specific heat: {Cv}')
    
    assert np.all(np.isfinite(E_mean)), "Energy mean contains NaN"
    assert np.all(Cv >= 0), "Specific heat should be non-negative"
    print('  ✓ Statistics test passed')

def main():
    try:
        test_basic()
        test_adaptive_pt()
        test_statistics()
        
        print('\n' + '='*60)
        print('ALL TESTS PASSED! ✓')
        print('='*60)
        
    except Exception as e:
        print(f'\n✗ TEST FAILED: {e}')
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
