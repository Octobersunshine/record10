import numpy as np
from star_sensor import (
    Star,
    generate_catalog_stars,
    generate_observed_stars,
    star_identification_triangle,
    euler_angles_from_rotation_matrix,
    solve_wahba_problem,
    solve_wahba_problem_quest,
    orthogonal_projection_svd
)


def rotation_matrix_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    roll, pitch, yaw = np.radians([roll, pitch, yaw])
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    return Rz @ Ry @ Rx


def test_star_identification():
    print("=" * 60)
    print("星敏感器星图识别测试 - 三角形算法")
    print("=" * 60)
    
    true_roll, true_pitch, true_yaw = 15.0, -10.0, 45.0
    print(f"\n真实姿态角: Roll={true_roll}°, Pitch={true_pitch}°, Yaw={true_yaw}°")
    
    R_true = rotation_matrix_from_euler(true_roll, true_pitch, true_yaw)
    print("\n真实旋转矩阵 R_true:")
    print(np.array2string(R_true, precision=6, suppress_small=True))
    
    catalog_stars = generate_catalog_stars()
    print(f"\n星表恒星数量: {len(catalog_stars)}")
    
    observed_stars = generate_observed_stars(R_true, catalog_stars, noise_level=0.005)
    print(f"观测恒星数量: {len(observed_stars)}")
    
    print("\n" + "-" * 60)
    print("开始星图识别...")
    R_est, matches = star_identification_triangle(observed_stars, catalog_stars, tolerance=0.02)
    print("星图识别完成!")
    
    print("\n" + "-" * 60)
    print("星匹配结果 (观测星索引 -> 星表星ID):")
    for obs_idx, cat_id in matches:
        print(f"  观测星 {obs_idx} -> 星表星 {cat_id}")
    print(f"成功匹配星数: {len(matches)}")
    
    print("\n" + "-" * 60)
    print("估计旋转矩阵 R_est:")
    print(np.array2string(R_est, precision=6, suppress_small=True))
    
    R_error = R_est - R_true
    print("\n旋转矩阵误差:")
    print(np.array2string(R_error, precision=6, suppress_small=True))
    
    est_roll, est_pitch, est_yaw = euler_angles_from_rotation_matrix(R_est)
    R_est_body = R_est.T
    est_roll_body, est_pitch_body, est_yaw_body = euler_angles_from_rotation_matrix(R_est_body)
    
    print("\n" + "-" * 60)
    print("姿态角估计结果 (从观测到星表的旋转矩阵转置):")
    print(f"  Roll:  真实={true_roll:.2f}°, 估计={est_roll_body:.2f}°, 误差={abs(est_roll_body-true_roll):.4f}°")
    print(f"  Pitch: 真实={true_pitch:.2f}°, 估计={est_pitch_body:.2f}°, 误差={abs(est_pitch_body-true_pitch):.4f}°")
    print(f"  Yaw:   真实={true_yaw:.2f}°, 估计={est_yaw_body:.2f}°, 误差={abs(est_yaw_body-true_yaw):.4f}°")
    
    ortho_check = R_est @ R_est.T
    print("\n" + "-" * 60)
    print("正交性验证 (R_est @ R_est.T 应为单位矩阵):")
    print(np.array2string(ortho_check, precision=6, suppress_small=True))
    
    det_est = np.linalg.det(R_est)
    print(f"\n行列式验证 (应为1): det(R_est) = {det_est:.6f}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    
    return R_est, R_true


def test_custom_data():
    print("\n" + "=" * 60)
    print("自定义观测数据测试")
    print("=" * 60)
    
    catalog_vectors = [
        np.array([0.98480775, 0.17364818, 0.0]),
        np.array([0.0, 0.98480775, 0.17364818]),
        np.array([0.17364818, 0.0, 0.98480775]),
        np.array([0.57735027, 0.57735027, 0.57735027]),
        np.array([-0.5, 0.5, 0.70710678]),
    ]
    catalog_stars = [Star(v, i) for i, v in enumerate(catalog_vectors)]
    
    R_true = rotation_matrix_from_euler(30, 0, 60)
    
    observed_vectors = [R_true @ v for v in catalog_vectors]
    observed_stars = [Star(v, None) for v in observed_vectors]
    
    R_est, matches = star_identification_triangle(observed_stars, catalog_stars, tolerance=0.01)
    
    print("\n匹配结果:")
    for obs_idx, cat_id in matches:
        print(f"  观测星 {obs_idx} -> 星表星 {cat_id}")
    
    R_est_body = R_est.T
    est_roll, est_pitch, est_yaw = euler_angles_from_rotation_matrix(R_est_body)
    true_roll, true_pitch, true_yaw = 30, 0, 60
    
    print(f"\nRoll误差: {abs(est_roll-true_roll):.4f}°")
    print(f"Pitch误差: {abs(est_pitch-true_pitch):.4f}°")
    print(f"Yaw误差: {abs(est_yaw-true_yaw):.4f}°")
    
    return R_est


def test_quest_orthogonal_projection():
    print("\n" + "=" * 60)
    print("QUEST算法与SVD正交投影测试")
    print("=" * 60)
    
    true_roll, true_pitch, true_yaw = 20.0, -15.0, 30.0
    R_true = rotation_matrix_from_euler(true_roll, true_pitch, true_yaw)
    
    obs_vectors = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.57735027, 0.57735027, 0.57735027])
    ]
    
    cat_vectors = [R_true.T @ v for v in obs_vectors]
    
    print("\n测试1: QUEST算法求解旋转矩阵")
    R_quest = solve_wahba_problem_quest(obs_vectors, cat_vectors)
    print(f"QUEST算法输出旋转矩阵行列式: {np.linalg.det(R_quest):.10f}")
    
    ortho_check = R_quest @ R_quest.T
    ortho_error = np.max(np.abs(ortho_check - np.eye(3)))
    print(f"QUEST算法输出正交性误差: {ortho_error:.2e}")
    
    print("\n测试2: SVD正交投影修正")
    R_corrected = orthogonal_projection_svd(R_quest)
    print(f"正交投影后旋转矩阵行列式: {np.linalg.det(R_corrected):.10f}")
    
    ortho_check_corrected = R_corrected @ R_corrected.T
    ortho_error_corrected = np.max(np.abs(ortho_check_corrected - np.eye(3)))
    print(f"正交投影后正交性误差: {ortho_error_corrected:.2e}")
    
    print("\n测试3: 完整solve_wahba_problem函数 (QUEST + SVD投影)")
    R_final = solve_wahba_problem(obs_vectors, cat_vectors)
    print(f"最终旋转矩阵行列式: {np.linalg.det(R_final):.10f}")
    
    ortho_check_final = R_final @ R_final.T
    ortho_error_final = np.max(np.abs(ortho_check_final - np.eye(3)))
    print(f"最终正交性误差: {ortho_error_final:.2e}")
    
    print("\n测试4: 姿态角估计精度")
    R_final_body = R_final.T
    est_roll, est_pitch, est_yaw = euler_angles_from_rotation_matrix(R_final_body)
    print(f"Roll:  真实={true_roll:.2f}°, 估计={est_roll:.2f}°, 误差={abs(est_roll-true_roll):.4f}°")
    print(f"Pitch: 真实={true_pitch:.2f}°, 估计={est_pitch:.2f}°, 误差={abs(est_pitch-true_pitch):.4f}°")
    print(f"Yaw:   真实={true_yaw:.2f}°, 估计={est_yaw:.2f}°, 误差={abs(est_yaw-true_yaw):.4f}°")
    
    print("\n测试5: 非正交矩阵修正演示")
    R_degenerate = np.array([
        [0.9, 0.1, 0.0],
        [0.0, 0.9, 0.1],
        [0.1, 0.0, 0.9]
    ])
    det_before = np.linalg.det(R_degenerate)
    ortho_before = np.max(np.abs(R_degenerate @ R_degenerate.T - np.eye(3)))
    print(f"退化矩阵行列式: {det_before:.6f}")
    print(f"退化矩阵正交性误差: {ortho_before:.2e}")
    
    R_fixed = orthogonal_projection_svd(R_degenerate)
    det_after = np.linalg.det(R_fixed)
    ortho_after = np.max(np.abs(R_fixed @ R_fixed.T - np.eye(3)))
    print(f"修正后矩阵行列式: {det_after:.10f}")
    print(f"修正后正交性误差: {ortho_after:.2e}")
    
    print("\n" + "=" * 60)
    print("QUEST算法与SVD正交投影测试完成!")
    print("=" * 60)
    
    return R_final


if __name__ == "__main__":
    np.random.seed(42)
    
    R_est1, R_true1 = test_star_identification()
    R_est2 = test_custom_data()
    R_est3 = test_quest_orthogonal_projection()
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)
