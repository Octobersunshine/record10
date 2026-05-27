import sys
sys.path.insert(0, r'e:\temp\record10\213')
from black_oil_impes import *
import numpy as np

# Quick test with both solvers
nx, ny = 8, 8
t_max = 30.0

print("=" * 70)
print("  IMPES vs FIM Mass Balance Comparison (Quick Test)")
print("=" * 70)

print("\n" + "-" * 70)
print("  Running IMPES Solver...")
print("-" * 70)
results_impes = main(use_fim=False, nx=nx, ny=ny, t_max=t_max)

print("\n" + "-" * 70)
print("  Running FIM Solver...")
print("-" * 70)
results_fim = main(use_fim=True, nx=nx, ny=ny, t_max=t_max)

print("\n" + "=" * 70)
print("  Mass Balance Comparison")
print("=" * 70)

mb_impes = results_impes.get('mass_balance_errors', [])
mb_fim = results_fim.get('mass_balance_errors', [])

if len(mb_impes) > 0:
    avg_mb_impes = np.mean([r['water'] for r in mb_impes])
    max_mb_impes = np.max([r['water'] for r in mb_impes])
    print(f"\n  IMPES Water Mass Balance:")
    print(f"    Mean residual: {avg_mb_impes:.2e} m3/day")
    print(f"    Max residual:  {max_mb_impes:.2e} m3/day")

if len(mb_fim) > 0:
    avg_mb_fim = np.mean([r['water'] for r in mb_fim])
    max_mb_fim = np.max([r['water'] for r in mb_fim])
    print(f"\n  FIM Water Mass Balance:")
    print(f"    Mean residual: {avg_mb_fim:.2e} m3/day")
    print(f"    Max residual:  {max_mb_fim:.2e} m3/day")

if len(mb_impes) > 0 and len(mb_fim) > 0:
    improvement = avg_mb_impes / avg_mb_fim
    print(f"\n  Mass Balance Improvement: {improvement:.1f}x better with FIM")

avg_corr = np.mean(results_fim.get('correction_iters', [0]))
print(f"\n  FIM Performance:")
print(f"    Average correction iterations per step: {avg_corr:.1f}")

# Compare production
prod_impes = results_impes['production']['PROD-1']
prod_fim = results_fim['production']['PROD-1']

cum_oil_impes = np.sum([d['oil_rate'] for d in prod_impes])
cum_oil_fim = np.sum([d['oil_rate'] for d in prod_fim])

print(f"\n  Production Comparison:")
print(f"    IMPES Cumulative Oil: {cum_oil_impes:.0f} m3")
print(f"    FIM Cumulative Oil:   {cum_oil_fim:.0f} m3")
print(f"    Difference:           {abs(cum_oil_impes - cum_oil_fim):.0f} m3 ({abs(cum_oil_impes - cum_oil_fim)/max(cum_oil_impes,1e-10)*100:.2f}%)")

print("\n" + "=" * 70)
