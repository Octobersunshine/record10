from main import calculate_default_nlags

print('序列长度 vs 默认延迟阶数:')
for n in [10, 20, 30, 50, 100, 200, 500]:
    nlags = calculate_default_nlags(n)
    print(f'  len={n:3d} -> nlags={nlags:3d}')

print()
print('改进说明:')
print('- 短序列 (<30): nobs // 2')
print('- 中等序列 (30-100): max(nobs // 3, 20)')
print('- 长序列 (>100): max(nobs // 4, 50)')
print('- 最大不超过 nobs - 1')
print()
print('优势: 对于长周期序列(周期>10)，不会丢失信息')