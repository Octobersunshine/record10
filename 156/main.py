#!/usr/bin/env python3

import numpy as np
import argparse
from simulator import SurfaceCodeSimulator
from error_model import ErrorModel, DepolarizingErrorModel


def demo_single_trial():
    print("=" * 60)
    print("演示: 单次表面码纠错试验")
    print("=" * 60)
    
    d = 3
    simulator = SurfaceCodeSimulator(d)
    
    print(f"\n码距: {d}")
    print(f"数据量子比特数: {simulator.sc.n_data}")
    print(f"X稳定子数: {simulator.sc.n_ancilla_z}")
    print(f"Z稳定子数: {simulator.sc.n_ancilla_x}")
    
    error_model = ErrorModel(p_bit_flip=0.1, p_phase_flip=0.0)
    simulator.sc.reset()
    error_model.apply_errors(simulator.sc)
    
    print(f"\nX错误位置: {np.where(simulator.sc.x_errors)[0]}")
    print(f"Z错误位置: {np.where(simulator.sc.z_errors)[0]}")
    
    x_stabs, z_stabs = simulator.sc.measure_stabilizers()
    print(f"X稳定子缺陷: {np.where(x_stabs)[0]}")
    print(f"Z稳定子缺陷: {np.where(z_stabs)[0]}")
    
    x_matching = simulator.decoder.decode('x')
    z_matching = simulator.decoder.decode('z')
    print(f"X型匹配: {x_matching}")
    print(f"Z型匹配: {z_matching}")
    
    simulator.decoder.apply_correction(x_matching, 'x')
    simulator.decoder.apply_correction(z_matching, 'z')
    
    x_logical, z_logical = simulator.sc.get_logical_error()
    print(f"\n解码后逻辑X错误: {x_logical}")
    print(f"解码后逻辑Z错误: {z_logical}")
    print(f"逻辑错误发生: {x_logical or z_logical}")


def estimate_error_rate():
    print("\n" + "=" * 60)
    print("估计逻辑错误率")
    print("=" * 60)
    
    d = 3
    p_phys = 0.05
    n_trials = 1000
    
    print(f"\n码距: {d}")
    print(f"物理错误率: {p_phys}")
    print(f"试验次数: {n_trials}")
    
    simulator = SurfaceCodeSimulator(d)
    error_model = ErrorModel(p_bit_flip=p_phys, p_phase_flip=0)
    
    p_logical, error = simulator.estimate_logical_error_rate(
        n_trials, error_model, verbose=True
    )
    
    print(f"\n逻辑错误率: {p_logical:.6f} ± {error:.6f}")


def threshold_scan():
    print("\n" + "=" * 60)
    print("阈值扫描")
    print("=" * 60)
    
    distances = [3, 5]
    error_rates = [0.01, 0.05, 0.1]
    n_trials = 200
    
    print(f"\n码距: {distances}")
    print(f"物理错误率范围: {error_rates}")
    print(f"每点试验次数: {n_trials}")
    print("\n开始扫描...\n")
    
    simulator = SurfaceCodeSimulator(3)
    results = simulator.threshold_scan(
        distances, error_rates, n_trials, 
        error_type='bit_flip', verbose=True
    )
    
    print("\n" + "=" * 60)
    print("结果汇总:")
    print("=" * 60)
    for d in distances:
        print(f"\nd={d}:")
        for i, p in enumerate(results[d]['p_phys']):
            pl = results[d]['p_logical'][i]
            err = results[d]['error'][i]
            print(f"  p_phys={p:.4f}: p_logical={pl:.6f} ± {err:.6f}")


def main():
    parser = argparse.ArgumentParser(description='表面码量子纠错模拟')
    parser.add_argument('--mode', type=str, default='demo',
                        choices=['demo', 'estimate', 'threshold'],
                        help='运行模式: demo(演示), estimate(估计错误率), threshold(阈值扫描)')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        demo_single_trial()
    elif args.mode == 'estimate':
        estimate_error_rate()
    elif args.mode == 'threshold':
        threshold_scan()


if __name__ == "__main__":
    main()
