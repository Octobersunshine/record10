from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

DEFAULT_RCOND = 1e-10
DEFAULT_REGULARIZATION = 1e-6
CONDITION_THRESHOLD = 1e10
DEFAULT_ENERGY_RATIO = 0.95

DEFAULT_ITERATIVE_REFINE = True
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_RESIDUAL_TOLERANCE = 1e-12


def truncated_svd_solve(A_np, b_np, energy_ratio=DEFAULT_ENERGY_RATIO):
    U, s, Vt = np.linalg.svd(A_np, full_matrices=False)
    
    total_energy = np.sum(s ** 2)
    cumulative_energy = np.cumsum(s ** 2)
    energy_ratios = cumulative_energy / total_energy
    
    k = np.searchsorted(energy_ratios, energy_ratio) + 1
    k = min(k, len(s))
    
    s_trunc = s[:k]
    U_trunc = U[:, :k]
    Vt_trunc = Vt[:k, :]
    
    s_inv = 1.0 / s_trunc
    if b_np.ndim == 1:
        x = Vt_trunc.T @ (s_inv * (U_trunc.T @ b_np))
    else:
        x = Vt_trunc.T @ (s_inv.reshape(-1, 1) * (U_trunc.T @ b_np))
    
    return x, k, len(s), float(energy_ratios[k-1] if k > 0 else 0)


def compute_residual(A, x, b):
    return np.linalg.norm(np.dot(A, x) - b)


def compute_relative_residual(A, x, b):
    residual = np.linalg.norm(np.dot(A, x) - b)
    b_norm = np.linalg.norm(b)
    return residual / b_norm if b_norm > 0 else residual


def solve_lu(A, b):
    return np.linalg.solve(A, b)


def solve_pinv(A, b, rcond=DEFAULT_RCOND):
    A_pinv = np.linalg.pinv(A, rcond=rcond)
    return np.dot(A_pinv, b)


def solve_tikhonov(A, b, alpha=DEFAULT_REGULARIZATION):
    n = A.shape[0]
    A_reg = np.dot(A.T, A) + alpha * np.eye(n)
    b_reg = np.dot(A.T, b)
    return np.linalg.solve(A_reg, b_reg)


def iterative_refinement(A_np, b_np, x_initial, solve_method, 
                        max_iterations=DEFAULT_MAX_ITERATIONS,
                        tol=DEFAULT_RESIDUAL_TOLERANCE, **solve_kwargs):
    x = x_initial.copy()
    residuals = []
    
    initial_residual = compute_relative_residual(A_np, x, b_np)
    residuals.append(float(initial_residual))
    
    for iteration in range(max_iterations):
        r = b_np - np.dot(A_np, x)
        
        try:
            delta = solve_method(A_np, r, **solve_kwargs)
        except np.linalg.LinAlgError:
            A_pinv = np.linalg.pinv(A_np, rcond=DEFAULT_RCOND)
            delta = np.dot(A_pinv, r)
        
        x = x + delta
        
        current_residual = compute_relative_residual(A_np, x, b_np)
        residuals.append(float(current_residual))
        
        if current_residual < tol:
            break
        
        if len(residuals) >= 3 and abs(residuals[-1] - residuals[-2]) < 1e-15:
            break
    
    return x, len(residuals) - 1, residuals


def solve_single(A, b, method='auto', rcond=DEFAULT_RCOND, 
                 alpha=DEFAULT_REGULARIZATION, energy_ratio=DEFAULT_ENERGY_RATIO,
                 iterative_refine=DEFAULT_ITERATIVE_REFINE,
                 max_iterations=DEFAULT_MAX_ITERATIONS,
                 residual_tolerance=DEFAULT_RESIDUAL_TOLERANCE):
    A_np = np.array(A, dtype=np.float64)
    b_np = np.array(b, dtype=np.float64)
    
    if A_np.ndim != 2:
        raise ValueError("矩阵A必须是二维数组")
    if b_np.ndim not in [1, 2]:
        raise ValueError("向量b必须是一维或二维数组")
    
    n = A_np.shape[0]
    if A_np.shape[1] != n:
        raise ValueError("矩阵A必须是方阵")
    
    if b_np.ndim == 1:
        if b_np.shape[0] != n:
            raise ValueError(f"向量b的长度必须为{n}")
    else:
        if b_np.shape[0] != n:
            raise ValueError(f"向量b的行数必须为{n}")
    
    cond = np.linalg.cond(A_np)
    
    is_near_singular = cond > CONDITION_THRESHOLD
    
    used_method = method
    svd_info = None
    refinement_info = None
    
    if method == 'auto':
        if is_near_singular:
            used_method = 'truncated_svd'
        else:
            used_method = 'lu'
    
    solve_method = None
    solve_kwargs = {}
    
    if used_method == 'lu':
        solve_method = solve_lu
    elif used_method == 'pinv':
        solve_method = solve_pinv
        solve_kwargs = {'rcond': rcond}
    elif used_method == 'tikhonov':
        solve_method = solve_tikhonov
        solve_kwargs = {'alpha': alpha}
    elif used_method == 'truncated_svd':
        solve_method = truncated_svd_solve
        solve_kwargs = {'energy_ratio': energy_ratio}
    else:
        raise ValueError(f"不支持的求解方法: {method}")
    
    try:
        if used_method == 'truncated_svd':
            x_initial, k, total_singular, achieved_energy = solve_method(A_np, b_np, **solve_kwargs)
            svd_info = {
                'total_singular_values': total_singular,
                'retained_singular_values': k,
                'energy_ratio': achieved_energy,
                'target_energy_ratio': energy_ratio
            }
        else:
            x_initial = solve_method(A_np, b_np, **solve_kwargs)
    except np.linalg.LinAlgError as e:
        x_initial, k, total_singular, achieved_energy = truncated_svd_solve(A_np, b_np, energy_ratio)
        used_method = 'truncated_svd_fallback'
        svd_info = {
            'total_singular_values': total_singular,
            'retained_singular_values': k,
            'energy_ratio': achieved_energy,
            'target_energy_ratio': energy_ratio
        }
    
    if iterative_refine:
        if used_method == 'truncated_svd':
            def wrapped_truncated_svd(A, b, **kwargs):
                x, _, _, _ = truncated_svd_solve(A, b, **kwargs)
                return x
            refine_solve_method = wrapped_truncated_svd
            refine_kwargs = solve_kwargs
        elif used_method == 'truncated_svd_fallback':
            def wrapped_truncated_svd(A, b, **kwargs):
                x, _, _, _ = truncated_svd_solve(A, b, **kwargs)
                return x
            refine_solve_method = wrapped_truncated_svd
            refine_kwargs = {'energy_ratio': energy_ratio}
        else:
            refine_solve_method = solve_method
            refine_kwargs = solve_kwargs
        
        x_final, iterations, residual_history = iterative_refinement(
            A_np, b_np, x_initial, refine_solve_method,
            max_iterations=max_iterations, tol=residual_tolerance, **refine_kwargs
        )
        
        refinement_info = {
            'enabled': True,
            'iterations': iterations,
            'max_iterations': max_iterations,
            'residual_tolerance': residual_tolerance,
            'initial_residual': residual_history[0],
            'final_residual': residual_history[-1],
            'residual_history': residual_history,
            'residual_reduction_ratio': residual_history[0] / residual_history[-1] if residual_history[-1] > 0 else float('inf')
        }
    else:
        x_final = x_initial
        refinement_info = {
            'enabled': False
        }
    
    rel_residual = compute_relative_residual(A_np, x_final, b_np)
    
    result = {
        'x': x_final.tolist(),
        'condition_number': float(cond),
        'condition_threshold': CONDITION_THRESHOLD,
        'is_near_singular': bool(is_near_singular),
        'used_method': used_method,
        'relative_residual': float(rel_residual),
        'iterative_refinement': refinement_info
    }
    
    if svd_info is not None:
        result['svd_info'] = svd_info
    
    return result


@app.route('/solve', methods=['POST'])
def solve():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "请求体不能为空"}), 400
        
        A = data.get('A')
        b = data.get('b')
        method = data.get('method', 'auto')
        rcond = data.get('rcond', DEFAULT_RCOND)
        alpha = data.get('alpha', DEFAULT_REGULARIZATION)
        energy_ratio = data.get('energy_ratio', DEFAULT_ENERGY_RATIO)
        iterative_refine = data.get('iterative_refine', DEFAULT_ITERATIVE_REFINE)
        max_iterations = data.get('max_iterations', DEFAULT_MAX_ITERATIONS)
        residual_tolerance = data.get('residual_tolerance', DEFAULT_RESIDUAL_TOLERANCE)
        
        if A is None or b is None:
            return jsonify({"error": "必须提供矩阵A和向量b"}), 400
        
        result = solve_single(A, b, method=method, rcond=rcond, 
                             alpha=alpha, energy_ratio=energy_ratio,
                             iterative_refine=iterative_refine,
                             max_iterations=max_iterations,
                             residual_tolerance=residual_tolerance)
        
        return jsonify({
            "success": True,
            **result
        })
        
    except np.linalg.LinAlgError as e:
        return jsonify({
            "success": False,
            "error": "线性代数错误",
            "details": str(e)
        }), 400
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": "输入错误",
            "details": str(e)
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "服务器错误",
            "details": str(e)
        }), 500


@app.route('/batch_solve', methods=['POST'])
def batch_solve():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "请求体不能为空"}), 400
        
        problems = data.get('problems')
        method = data.get('method', 'auto')
        rcond = data.get('rcond', DEFAULT_RCOND)
        alpha = data.get('alpha', DEFAULT_REGULARIZATION)
        energy_ratio = data.get('energy_ratio', DEFAULT_ENERGY_RATIO)
        iterative_refine = data.get('iterative_refine', DEFAULT_ITERATIVE_REFINE)
        max_iterations = data.get('max_iterations', DEFAULT_MAX_ITERATIONS)
        residual_tolerance = data.get('residual_tolerance', DEFAULT_RESIDUAL_TOLERANCE)
        
        if not problems or not isinstance(problems, list):
            return jsonify({"error": "必须提供problems数组"}), 400
        
        results = []
        for i, problem in enumerate(problems):
            try:
                A = problem.get('A')
                b = problem.get('b')
                
                if A is None or b is None:
                    results.append({
                        "index": i,
                        "success": False,
                        "error": "每个问题必须提供矩阵A和向量b"
                    })
                    continue
                
                problem_method = problem.get('method', method)
                problem_rcond = problem.get('rcond', rcond)
                problem_alpha = problem.get('alpha', alpha)
                problem_energy_ratio = problem.get('energy_ratio', energy_ratio)
                problem_iterative_refine = problem.get('iterative_refine', iterative_refine)
                problem_max_iterations = problem.get('max_iterations', max_iterations)
                problem_residual_tolerance = problem.get('residual_tolerance', residual_tolerance)
                
                solve_result = solve_single(A, b, method=problem_method, 
                                           rcond=problem_rcond, alpha=problem_alpha,
                                           energy_ratio=problem_energy_ratio,
                                           iterative_refine=problem_iterative_refine,
                                           max_iterations=problem_max_iterations,
                                           residual_tolerance=problem_residual_tolerance)
                results.append({
                    "index": i,
                    "success": True,
                    **solve_result
                })
                
            except Exception as e:
                results.append({
                    "index": i,
                    "success": False,
                    "error": str(e)
                })
        
        success_count = sum(1 for r in results if r["success"])
        
        return jsonify({
            "total": len(problems),
            "success_count": success_count,
            "failed_count": len(problems) - success_count,
            "results": results
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "服务器错误",
            "details": str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok", 
        "numpy_version": np.__version__,
        "condition_threshold": CONDITION_THRESHOLD,
        "default_energy_ratio": DEFAULT_ENERGY_RATIO,
        "default_iterative_refine": DEFAULT_ITERATIVE_REFINE,
        "default_max_iterations": DEFAULT_MAX_ITERATIONS,
        "default_residual_tolerance": DEFAULT_RESIDUAL_TOLERANCE
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
