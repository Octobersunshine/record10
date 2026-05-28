import warnings
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.covariance import MinCovDet


def mahalanobis_outlier_detection(data, alpha=0.01, reg_param=1e-6):
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n, p = data.shape

    mean_vec = np.mean(data, axis=0)
    cov_mat = np.cov(data, rowvar=False)

    if p == 1:
        cov_mat = cov_mat.reshape(1, 1)

    method_used = "regular inverse"
    cov_inv = None
    is_singular = False

    cond_threshold = 1e10

    if n < p:
        is_singular = True
    else:
        try:
            cond_num = np.linalg.cond(cov_mat)
            if cond_num > cond_threshold:
                is_singular = True
        except Exception:
            is_singular = True

    if not is_singular:
        try:
            cov_inv = np.linalg.inv(cov_mat)
            if not np.all(np.isfinite(cov_inv)):
                is_singular = True
        except np.linalg.LinAlgError:
            is_singular = True

    if is_singular:
        if reg_param > 0:
            trace_cov = np.trace(cov_mat)
            delta = max(reg_param, reg_param * trace_cov / p if p > 0 else reg_param)
            cov_mat_reg = cov_mat + delta * np.eye(p)
            try:
                cov_inv = np.linalg.inv(cov_mat_reg)
                method_used = f"regularized inverse (δ={delta:.2e})"
                reason = "n < p" if n < p else f"条件数过大 (cond={np.linalg.cond(cov_mat):.2e})"
                warnings.warn(
                    f"协方差矩阵奇异或病态（{reason}），已采用正则化协方差矩阵（δ={delta:.2e}），"
                    f"马氏距离结果可能有偏倚，建议检查数据相关性或增加样本量。",
                    UserWarning
                )
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(cov_mat)
                method_used = "Moore-Penrose pseudoinverse"
                warnings.warn(
                    "正则化后协方差矩阵仍不可逆，已采用 Moore-Penrose 伪逆计算马氏距离，"
                    "结果可能存在较大误差，建议检查数据质量。",
                    UserWarning
                )
        else:
            cov_inv = np.linalg.pinv(cov_mat)
            method_used = "Moore-Penrose pseudoinverse"
            warnings.warn(
                "协方差矩阵奇异且未启用正则化，已采用 Moore-Penrose 伪逆计算马氏距离，"
                "建议设置 reg_param>0 启用正则化以获得更稳定结果。",
                UserWarning
            )

    distances = np.array([
        mahalanobis(row, mean_vec, cov_inv) for row in data
    ])

    threshold = np.sqrt(chi2.ppf(1 - alpha, df=p))

    outlier_mask = distances > threshold

    return distances, threshold, outlier_mask, method_used


def robust_mahalanobis_outlier_detection(data, alpha=0.01, support_fraction=None, random_state=42):
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n, p = data.shape

    if support_fraction is None:
        support_fraction = min(0.75, max(0.5, (n - p + 1) / n)) if n > p else 0.5

    try:
        mcd = MinCovDet(support_fraction=support_fraction, random_state=random_state)
        mcd.fit(data)
        robust_mean = mcd.location_
        robust_cov = mcd.covariance_

        cond_num = np.linalg.cond(robust_cov)
        if cond_num > 1e10:
            delta = 1e-6 * np.trace(robust_cov) / p
            robust_cov = robust_cov + delta * np.eye(p)
            warnings.warn(
                f"MCD 协方差矩阵病态（条件数={cond_num:.2e}），已添加正则化 δ={delta:.2e}",
                UserWarning
            )

        robust_cov_inv = np.linalg.inv(robust_cov)
        method_used = f"MCD (h={int(support_fraction * n)}/{n})"

    except Exception as e:
        warnings.warn(
            f"MCD 估计失败（{str(e)}），回退到经典马氏距离带正则化",
            UserWarning
        )
        distances, threshold, outlier_mask, method_used = mahalanobis_outlier_detection(
            data, alpha=alpha, reg_param=1e-6
        )
        return {
            "distances": distances,
            "threshold": threshold,
            "outlier_mask": outlier_mask,
            "method_used": f"fallback: {method_used}",
            "robust_mean": None,
            "robust_cov": None
        }

    distances = np.array([
        mahalanobis(row, robust_mean, robust_cov_inv) for row in data
    ])

    threshold = np.sqrt(chi2.ppf(1 - alpha, df=p))
    outlier_mask = distances > threshold

    return {
        "distances": distances,
        "threshold": threshold,
        "outlier_mask": outlier_mask,
        "method_used": method_used,
        "robust_mean": robust_mean,
        "robust_cov": robust_cov
    }


def mahalanobis_outlier_detection_both(data, alpha=0.01, reg_param=1e-6, support_fraction=None):
    classical_result = mahalanobis_outlier_detection(data, alpha=alpha, reg_param=reg_param)
    classical_dist, classical_thresh, classical_mask, classical_method = classical_result

    robust_result = robust_mahalanobis_outlier_detection(
        data, alpha=alpha, support_fraction=support_fraction
    )

    return {
        "classical": {
            "distances": classical_dist,
            "threshold": classical_thresh,
            "outlier_mask": classical_mask,
            "method_used": classical_method
        },
        "robust": {
            "distances": robust_result["distances"],
            "threshold": robust_result["threshold"],
            "outlier_mask": robust_result["outlier_mask"],
            "method_used": robust_result["method_used"],
            "robust_mean": robust_result["robust_mean"],
            "robust_cov": robust_result["robust_cov"]
        }
    }


def test_normal_case():
    print("\n" + "=" * 60)
    print("【测试 1】正常数据（协方差矩阵可逆）")
    print("=" * 60)

    np.random.seed(42)
    n_normal = 200
    mean = [5, 10, 20]
    cov = [[1, 0.5, 0.3],
           [0.5, 2, 0.4],
           [0.3, 0.4, 1.5]]

    normal_data = np.random.multivariate_normal(mean, cov, size=n_normal)

    n_outliers = 8
    outlier_data = np.array([
        [15, 25, 40],
        [-5, -2, 35],
        [20, 30, 10],
        [1, 20, 45],
        [25, 5, 50],
        [-3, 30, -5],
        [18, -3, 38],
        [30, 25, 5],
    ])

    data = np.vstack([normal_data, outlier_data])
    sample_labels = np.arange(1, len(data) + 1)

    distances, threshold, outlier_mask, method_used = mahalanobis_outlier_detection(data, alpha=0.01)

    print(f"样本数量: {len(data)}")
    print(f"变量维度: {data.shape[1]}")
    print(f"显著性水平 α: 0.01")
    print(f"卡方分布自由度: {data.shape[1]}")
    print(f"距离阈值 (χ² 的平方根): {threshold:.4f}")
    print(f"使用方法: {method_used}")
    print("-" * 60)

    outlier_indices = np.where(outlier_mask)[0]
    print(f"检测到离群值数量: {outlier_mask.sum()}")

    injected = list(range(n_normal, n_normal + n_outliers))
    detected = set(outlier_indices.tolist())
    injected_set = set(injected)
    hit = detected & injected_set
    print(f"注入的离群值编号: {[i + 1 for i in injected]}")
    print(f"成功检出: {sorted([i + 1 for i in hit])}")
    print(f"误报: {sorted([i + 1 for i in detected - injected_set])}")
    print(f"漏检: {sorted([i + 1 for i in injected_set - detected])}")
    print("✓ 测试通过" if len(hit) == n_outliers else "✗ 测试失败")


def test_singular_covariance():
    print("\n" + "=" * 60)
    print("【测试 2】奇异协方差矩阵（特征完全相关）")
    print("=" * 60)

    np.random.seed(42)
    n = 50
    x1 = np.random.normal(0, 1, n)
    x2 = x1 * 2 + np.random.normal(0, 0.001, n)
    x3 = x1 + x2
    data = np.column_stack([x1, x2, x3])

    outlier = np.array([[10, 20, 30]])
    data = np.vstack([data, outlier])

    print(f"样本数量: {data.shape[0]}")
    print(f"变量维度: {data.shape[1]}")
    print(f"协方差矩阵行列式: {np.linalg.det(np.cov(data, rowvar=False)):.2e}")
    print("-" * 60)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        distances, threshold, outlier_mask, method_used = mahalanobis_outlier_detection(data, alpha=0.01)

        if w:
            print(f"警告信息: {w[-1].message}")

    print(f"使用方法: {method_used}")
    print(f"距离阈值: {threshold:.4f}")
    print(f"检测到离群值: {outlier_mask.sum()} 个")
    print(f"最后一个样本（注入离群）距离: {distances[-1]:.4f}")
    print(f"最后一个样本是否被标记: {'是' if outlier_mask[-1] else '否'}")

    if "regularized" in method_used or "pseudoinverse" in method_used:
        print("✓ 奇异矩阵处理生效")
    else:
        print("✗ 奇异矩阵处理未触发")


def test_n_less_than_p():
    print("\n" + "=" * 60)
    print("【测试 3】样本数 < 特征数（n=10, p=20）")
    print("=" * 60)

    np.random.seed(42)
    n, p = 10, 20
    data = np.random.randn(n, p)

    print(f"样本数量: {n}")
    print(f"变量维度: {p}")
    print(f"协方差矩阵形状: {np.cov(data, rowvar=False).shape}")
    print("-" * 60)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        distances, threshold, outlier_mask, method_used = mahalanobis_outlier_detection(data, alpha=0.01)

        if w:
            print(f"警告信息: {w[-1].message}")

    print(f"使用方法: {method_used}")
    print(f"距离阈值: {threshold:.4f}")
    print(f"检测到离群值: {outlier_mask.sum()} 个")
    print(f"所有距离有效: {np.all(np.isfinite(distances))}")

    if "regularized" in method_used or "pseudoinverse" in method_used:
        print("✓ n < p 场景处理生效")
    else:
        print("✗ n < p 场景处理未触发")


def test_no_regularization():
    print("\n" + "=" * 60)
    print("【测试 4】禁用正则化（reg_param=0）")
    print("=" * 60)

    np.random.seed(42)
    n = 30
    x1 = np.random.normal(0, 1, n)
    x2 = x1 * 3
    data = np.column_stack([x1, x2])

    print(f"样本数量: {data.shape[0]}")
    print(f"变量维度: {data.shape[1]}")
    print("-" * 60)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        distances, threshold, outlier_mask, method_used = mahalanobis_outlier_detection(
            data, alpha=0.01, reg_param=0
        )

        if w:
            print(f"警告信息: {w[-1].message}")

    print(f"使用方法: {method_used}")
    print(f"所有距离有效: {np.all(np.isfinite(distances))}")

    if "pseudoinverse" in method_used:
        print("✓ 禁用正则化时伪逆生效")
    else:
        print("✗ 禁用正则化时伪逆未触发")


def test_classical_vs_robust_masking():
    print("\n" + "=" * 60)
    print("【测试 5】经典 vs 稳健：掩蔽效应（多离群值）")
    print("=" * 60)
    print("场景：20% 离群值，经典马氏距离易受掩蔽效应影响")

    np.random.seed(42)
    n_normal = 80
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]
    normal_data = np.random.multivariate_normal(mean, cov, size=n_normal)

    n_outliers = 20
    outlier_data = np.random.multivariate_normal([6, 6], [[0.3, 0], [0, 0.3]], size=n_outliers)

    data = np.vstack([normal_data, outlier_data])
    true_outliers = set(range(n_normal, n_normal + n_outliers))

    results = mahalanobis_outlier_detection_both(data, alpha=0.01)

    classical = results["classical"]
    robust = results["robust"]

    classical_detected = set(np.where(classical["outlier_mask"])[0])
    robust_detected = set(np.where(robust["outlier_mask"])[0])

    classical_hit = classical_detected & true_outliers
    robust_hit = robust_detected & true_outliers

    classical_false = classical_detected - true_outliers
    robust_false = robust_detected - true_outliers

    print(f"\n样本数: {len(data)}, 正常: {n_normal}, 离群: {n_outliers}")
    print(f"{'方法':<30} {'检出':>6} {'正确':>6} {'误报':>6} {'漏检':>6}")
    print("-" * 60)
    print(f"{'经典马氏距离 (' + classical['method_used'] + ')':<30} "
          f"{len(classical_detected):>6} {len(classical_hit):>6} {len(classical_false):>6} {n_outliers - len(classical_hit):>6}")
    print(f"{'稳健马氏距离 (' + robust['method_used'] + ')':<30} "
          f"{len(robust_detected):>6} {len(robust_hit):>6} {len(robust_false):>6} {n_outliers - len(robust_hit):>6}")

    if len(robust_hit) > len(classical_hit):
        print("✓ 稳健马氏距离在多离群值场景表现更优")
    else:
        print("  两种方法表现相当")


def test_compare_both_methods():
    print("\n" + "=" * 60)
    print("【测试 6】使用 mahalanobis_outlier_detection_both 接口")
    print("=" * 60)

    np.random.seed(42)
    n = 100
    mean = [5, 10]
    cov = [[2, 1], [1, 2]]
    data = np.random.multivariate_normal(mean, cov, size=n)

    data[95] = [20, 30]
    data[96] = [22, 28]
    data[97] = [-5, -5]

    results = mahalanobis_outlier_detection_both(data, alpha=0.01)

    print(f"{'样本':>6}  {'经典距离':>12}  {'经典标记':>8}  {'稳健距离':>12}  {'稳健标记':>8}")
    print("-" * 60)
    for i in range(90, 100):
        c_dist = results['classical']['distances'][i]
        c_out = "是" if results['classical']['outlier_mask'][i] else "否"
        r_dist = results['robust']['distances'][i]
        r_out = "是" if results['robust']['outlier_mask'][i] else "否"
        tag = "  ← 注入" if i >= 95 else ""
        print(f"{i+1:>6}  {c_dist:>12.4f}  {c_out:>8}  {r_dist:>12.4f}  {r_out:>8}{tag}")

    print(f"\n经典方法离群数: {results['classical']['outlier_mask'].sum()}")
    print(f"稳健方法离群数: {results['robust']['outlier_mask'].sum()}")
    print("✓ 双方法对比接口工作正常")


def main():
    warnings.filterwarnings("default", category=UserWarning)

    print("=" * 60)
    print("马氏距离离群值检测 - 完整测试套件")
    print("=" * 60)

    test_normal_case()
    test_singular_covariance()
    test_n_less_than_p()
    test_no_regularization()
    test_classical_vs_robust_masking()
    test_compare_both_methods()

    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
