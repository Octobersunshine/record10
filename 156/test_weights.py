import sys
sys.path.insert(0, r'e:\temp\record10\156')

import numpy as np
from surface_code import SurfaceCode
from mwpm_decoder import MWPMDecoder
from error_model import ErrorModel


def test_weight_calculation():
    print("=" * 60)
    print("测试: 负对数似然权重计算")
    print("=" * 60)
    
    sc = SurfaceCode(5)
    decoder = MWPMDecoder(sc, p_error=0.05)
    
    print(f"\n物理错误率 p = {decoder.p_error}")
    print(f"-log(p) = {-np.log(decoder.p_error):.4f}")
    
    print("\n距离与权重对应关系:")
    print("-" * 40)
    for d in [1, 2, 3, 4, 5, 10]:
        weight = decoder._negative_log_likelihood(d)
        print(f"  距离 d={d:2d}: 权重 = {weight:.4f}")
    
    print("\n" + "=" * 60)
    print("结论: 权重 = -d * log(p)")
    print("  - 距离越远, 权重越大")
    print("  - 错误率越低, 相同距离的权重越大")
    print("  - 最小权匹配优先选择近距离错误链")
    print("=" * 60)


def test_decoding_with_weights():
    print("\n" + "=" * 60)
    print("测试: 使用正确权重的解码过程")
    print("=" * 60)
    
    d = 5
    sc = SurfaceCode(d)
    decoder = MWPMDecoder(sc, p_error=0.05)
    
    print(f"\n码距 d = {d}")
    print(f"物理错误率 p = {decoder.p_error}")
    
    error_positions = [6, 7, 11]
    print(f"\n施加错误的量子比特: {error_positions}")
    for pos in error_positions:
        sc.apply_bit_flip(pos)
    
    x_stabs, z_stabs = sc.measure_stabilizers()
    x_defects = np.where(x_stabs)[0]
    print(f"X稳定子缺陷: {x_defects}")
    print(f"缺陷数目: {len(x_defects)}")
    
    print(f"\n解码前解码器错误率: {decoder.p_error}")
    x_matching = decoder.decode('x')
    print(f"匹配结果: {x_matching}")
    
    decoder.apply_correction(x_matching, 'x')
    
    x_logical, z_logical = sc.get_logical_error()
    print(f"\n解码后逻辑X错误: {x_logical}")
    print(f"逻辑错误发生: {x_logical or z_logical}")


def compare_different_error_rates():
    print("\n" + "=" * 60)
    print("比较: 不同错误率下的权重差异")
    print("=" * 60)
    
    sc = SurfaceCode(5)
    
    print("\n距离 d=3 时不同错误率的权重:")
    print("-" * 50)
    print(f"  {'p':>10} | {'-log(p)':>12} | {'权重(3*-log(p))':>18}")
    print("-" * 50)
    
    for p in [0.001, 0.01, 0.05, 0.1, 0.2]:
        decoder = MWPMDecoder(sc, p_error=p)
        weight = decoder._negative_log_likelihood(3)
        print(f"  {p:>10.4f} | {-np.log(p):>12.4f} | {weight:>18.4f}")


if __name__ == "__main__":
    test_weight_calculation()
    test_decoding_with_weights()
    compare_different_error_rates()
    
    print("\n" + "=" * 60)
    print("✓ 所有权重测试完成!")
    print("=" * 60)
