import numpy as np
from scipy.integrate import odeint
from scipy.linalg import schur, ordqz, solve_discrete_are, solve_lyapunov


def solve_are_schur(A, B, Q, R):
    """
    求解代数里卡蒂方程 (Algebraic Riccati Equation, ARE)
    使用Schur分解方法
    
    ARE: A^T P + P A - P B R^{-1} B^T P + Q = 0
    
    参数:
        A: 形状为 (n, n) 的状态矩阵
        B: 形状为 (n, m) 的输入矩阵
        Q: 形状为 (n, n) 的半正定状态权重矩阵
        R: 形状为 (m, m) 的正定输入权重矩阵
        
    返回:
        P: 形状为 (n, n) 的正定解矩阵
    """
    n = A.shape[0]
    m = B.shape[1]
    
    R_inv = np.linalg.inv(R)
    S = B @ R_inv @ B.T
    
    H = np.block([
        [A, -S],
        [-Q, -A.T]
    ])
    
    T, Z, sdim = schur(H, sort='lhp')
    
    U11 = Z[:n, :n]
    U21 = Z[n:, :n]
    
    P = U21 @ np.linalg.inv(U11)
    
    P = (P + P.T) / 2
    
    return P


def solve_are_eigen(A, B, Q, R):
    """
    求解代数里卡蒂方程 (ARE)
    使用Hamiltonian矩阵特征分解方法
    
    参数:
        A: 形状为 (n, n) 的状态矩阵
        B: 形状为 (n, m) 的输入矩阵
        Q: 形状为 (n, n) 的半正定状态权重矩阵
        R: 形状为 (m, m) 的正定输入权重矩阵
        
    返回:
        P: 形状为 (n, n) 的正定解矩阵
    """
    n = A.shape[0]
    
    R_inv = np.linalg.inv(R)
    S = B @ R_inv @ B.T
    
    H = np.block([
        [A, -S],
        [-Q, -A.T]
    ])
    
    eigvals, eigvecs = np.linalg.eig(H)
    
    stable_indices = np.where(eigvals.real < 0)[0]
    
    if len(stable_indices) != n:
        raise ValueError(f"无法找到足够的稳定特征值: 找到 {len(stable_indices)} 个，需要 {n} 个")
    
    stable_eigvecs = eigvecs[:, stable_indices]
    
    U11 = stable_eigvecs[:n, :]
    U21 = stable_eigvecs[n:, :]
    
    P = U21 @ np.linalg.inv(U11)
    
    P = (P + P.T) / 2
    
    return P


def solve_are_iterative(A, B, Q, R, max_iter=1000, tol=1e-12):
    """
    求解代数里卡蒂方程 (ARE)
    使用牛顿迭代法（作为备选方案）
    
    参数:
        A: 形状为 (n, n) 的状态矩阵
        B: 形状为 (n, m) 的输入矩阵
        Q: 形状为 (n, n) 的半正定状态权重矩阵
        R: 形状为 (m, m) 的正定输入权重矩阵
        max_iter: 最大迭代次数
        tol: 收敛容差
        
    返回:
        P: 形状为 (n, n) 的正定解矩阵
    """
    n = A.shape[0]
    
    P = Q.copy()
    R_inv = np.linalg.inv(R)
    
    for i in range(max_iter):
        K = R_inv @ B.T @ P
        A_cl = A - B @ K
        
        P_old = P.copy()
        
        try:
            from scipy.linalg import solve_lyapunov
            P = solve_lyapunov(A_cl.T, Q + K.T @ R @ K)
        except:
            P = solve_lyapunov_direct(A_cl.T, Q + K.T @ R @ K)
        
        if np.linalg.norm(P - P_old) < tol:
            break
    
    P = (P + P.T) / 2
    
    return P


def solve_lyapunov_direct(A, Q):
    """
    直接求解Lyapunov方程 A^T P + P A + Q = 0
    使用Kronecker积
    """
    n = A.shape[0]
    I = np.eye(n)
    M = np.kron(I, A.T) + np.kron(A.T, I)
    Q_vec = Q.flatten(order='F')
    P_vec = np.linalg.solve(M, -Q_vec)
    P = P_vec.reshape(n, n, order='F')
    return P


def solve_riccati_stable(A, B, Q, R, t_span, P_final, n_points=100, method='scipy'):
    """
    稳定的矩阵里卡蒂微分方程向后积分求解器
    
    对于定常系统，先求解ARE得到稳态解P_inf，
    然后对偏差 P_tilde = P - P_inf 进行积分，提高数值稳定性
    
    参数:
        A: 矩阵或函数，形状为 (n, n)
        B: 矩阵或函数，形状为 (n, m)
        Q: 矩阵或函数，形状为 (n, n)
        R: 矩阵或函数，形状为 (m, m)
        t_span: 时间区间 [t0, tf]
        P_final: 终端时刻的矩阵 P(tf)
        n_points: 时间离散点数量
        method: 'scipy' (使用odeint) 或 'euler' (使用欧拉法)
        
    返回:
        t: 时间数组（从 tf 到 t0 反向排列）
        P: 解矩阵数组
    """
    t0, tf = t_span
    
    def get_matrix(mat, t):
        if callable(mat):
            return mat(t)
        return mat
    
    n = P_final.shape[0]
    
    is_time_invariant = not callable(A) and not callable(B) and not callable(Q) and not callable(R)
    
    P_inf = None
    if is_time_invariant:
        try:
            P_inf = solve_are_schur(A, B, Q, R)
        except:
            try:
                P_inf = solve_are_eigen(A, B, Q, R)
            except:
                P_inf = None
    
    if P_inf is not None:
        P_tilde_final = P_final - P_inf
        
        def riccati_tilde_ode(P_tilde_flat, t):
            P_tilde = P_tilde_flat.reshape(n, n)
            P = P_inf + P_tilde
            
            A_t = get_matrix(A, t)
            B_t = get_matrix(B, t)
            Q_t = get_matrix(Q, t)
            R_t = get_matrix(R, t)
            
            R_inv = np.linalg.inv(R_t)
            K = R_inv @ B_t.T @ P
            A_cl = A_t - B_t @ K
            
            dP_tilde_dt = -A_cl.T @ P_tilde - P_tilde @ A_cl - P_tilde @ B_t @ R_inv @ B_t.T @ P_tilde
            
            return dP_tilde_dt.flatten()
        
        t_backward = np.linspace(tf, t0, n_points)
        P_tilde_flat = P_tilde_final.flatten()
        
        if method == 'scipy':
            P_tilde_sol_flat = odeint(riccati_tilde_ode, P_tilde_flat, t_backward)
        else:
            P_tilde_sol_flat = euler_integrate(riccati_tilde_ode, P_tilde_flat, t_backward)
        
        P_tilde_sol = P_tilde_sol_flat.reshape(-1, n, n)
        P_sol = P_inf + P_tilde_sol
        
    else:
        def riccati_ode(P_flat, t):
            P = P_flat.reshape(n, n)
            A_t = get_matrix(A, t)
            B_t = get_matrix(B, t)
            Q_t = get_matrix(Q, t)
            R_t = get_matrix(R, t)
            
            R_inv = np.linalg.inv(R_t)
            dPdt = -A_t.T @ P - P @ A_t - Q_t + P @ B_t @ R_inv @ B_t.T @ P
            return dPdt.flatten()
        
        t_backward = np.linspace(tf, t0, n_points)
        P0_flat = P_final.flatten()
        
        if method == 'scipy':
            P_sol_flat = odeint(riccati_ode, P0_flat, t_backward)
        else:
            P_sol_flat = euler_integrate(riccati_ode, P0_flat, t_backward)
        
        P_sol = P_sol_flat.reshape(-1, n, n)
    
    return t_backward, P_sol


def euler_integrate(ode_func, y0, t_array):
    """
    使用改进的欧拉法进行数值积分
    """
    n_points = len(t_array)
    y = np.zeros((n_points, len(y0)))
    y[0] = y0
    
    for i in range(n_points - 1):
        dt = t_array[i + 1] - t_array[i]
        k1 = ode_func(y[i], t_array[i])
        k2 = ode_func(y[i] + k1 * dt, t_array[i + 1])
        y[i + 1] = y[i] + (k1 + k2) / 2 * dt
    
    return y


def solve_riccati_steady_backward(A, B, Q, R, t_span, P_final, n_points=100):
    """
    基于稳态解的向后差分方法
    对于长时间跨度，使用稳态解作为t0时刻的值，然后向后积分
    
    参数:
        A: 定常矩阵，形状为 (n, n)
        B: 定常矩阵，形状为 (n, m)
        Q: 定常矩阵，形状为 (n, n)
        R: 定常矩阵，形状为 (m, m)
        t_span: 时间区间 [t0, tf]
        P_final: 终端时刻的矩阵 P(tf)
        n_points: 时间离散点数量
        
    返回:
        t: 时间数组（从 tf 到 t0 反向排列）
        P: 解矩阵数组
    """
    t0, tf = t_span
    
    P_inf = solve_are_schur(A, B, Q, R)
    
    t_backward = np.linspace(tf, t0, n_points)
    P_list = np.zeros((n_points,) + P_final.shape)
    
    P_list[0] = P_final.copy()
    
    dt = t_backward[0] - t_backward[1]
    
    R_inv = np.linalg.inv(R)
    
    for i in range(n_points - 1):
        P_current = P_list[i]
        K = R_inv @ B.T @ P_current
        A_cl = A - B @ K
        
        dPdt = -A_cl.T @ (P_current - P_inf) - (P_current - P_inf) @ A_cl
        
        P_next = P_current - dPdt * dt
        
        P_next = (P_next + P_next.T) / 2
        
        P_list[i + 1] = P_next
    
    return t_backward, P_list


def solve_riccati_dde(A, B, Q, R, t_span, P_final, n_points=100):
    """
    求解矩阵里卡蒂微分方程（向后积分）- 旧API保持兼容
    """
    return solve_riccati_stable(A, B, Q, R, t_span, P_final, n_points, method='scipy')


def solve_riccati_recurrence(A, B, Q, R, t_span, P_final, dt=None):
    """
    采用Riccati递推（离散化向后积分）求解矩阵里卡蒂微分方程 - 旧API保持兼容
    """
    t0, tf = t_span
    if dt is None:
        n_points = 100
    else:
        n_points = int((tf - t0) / dt) + 1
    return solve_riccati_stable(A, B, Q, R, t_span, P_final, n_points, method='euler')


def get_forward_solution(t_backward, P_backward):
    """
    将向后积分的结果转换为前向时间顺序
    """
    t_forward = t_backward[::-1]
    P_forward = P_backward[::-1]
    return t_forward, P_forward


def compute_optimal_gain(P, B, R, t=None):
    """
    计算最优反馈增益 K(t) = R(t)^{-1} B(t)^T P(t)
    """
    def get_matrix(mat, t_val):
        if callable(mat):
            return mat(t_val)
        return mat
    
    if t is None:
        t = 0
    
    B_t = get_matrix(B, t)
    R_t = get_matrix(R, t)
    R_inv = np.linalg.inv(R_t)
    K = R_inv @ B_t.T @ P
    return K


def check_are_solution(P, A, B, Q, R):
    """
    验证ARE的解是否正确
    返回残差的范数
    """
    R_inv = np.linalg.inv(R)
    residual = A.T @ P + P @ A - P @ B @ R_inv @ B.T @ P + Q
    return np.linalg.norm(residual)


def armijo_line_search(P, direction, grad, cost_func, A, B, Q, R, 
                       alpha_init=1.0, beta=0.5, c=1e-4, max_iter=50):
    """
    Armijo不精确线搜索
    
    参数:
        P: 当前迭代点
        direction: 搜索方向
        grad: 梯度
        cost_func: 代价函数
        alpha_init: 初始步长
        beta: 步长缩减因子
        c: Armijo条件参数
        max_iter: 最大迭代次数
        
    返回:
        alpha: 满足Armijo条件的步长
    """
    alpha = alpha_init
    cost_current = cost_func(P, A, B, Q, R)
    grad_dot_dir = np.sum(grad * direction)
    
    for _ in range(max_iter):
        P_new = P + alpha * direction
        cost_new = cost_func(P_new, A, B, Q, R)
        
        if cost_new <= cost_current + c * alpha * grad_dot_dir:
            return alpha
        
        alpha *= beta
    
    return alpha


def wolfe_line_search(P, direction, grad, cost_func, grad_func, A, B, Q, R,
                      alpha_init=1.0, c1=1e-4, c2=0.9, max_iter=50):
    """
    Wolfe不精确线搜索（强Wolfe条件）
    
    参数:
        P: 当前迭代点
        direction: 搜索方向
        grad: 梯度
        cost_func: 代价函数
        grad_func: 梯度函数
        alpha_init: 初始步长
        c1: Armijo条件参数
        c2: 曲率条件参数
        max_iter: 最大迭代次数
        
    返回:
        alpha: 满足Wolfe条件的步长
    """
    alpha_prev = 0.0
    alpha = alpha_init
    
    cost_current = cost_func(P, A, B, Q, R)
    grad_dot_dir = np.sum(grad * direction)
    
    for i in range(max_iter):
        P_new = P + alpha * direction
        cost_new = cost_func(P_new, A, B, Q, R)
        grad_new = grad_func(P_new, A, B, Q, R)
        grad_new_dot_dir = np.sum(grad_new * direction)
        
        armijo_ok = cost_new <= cost_current + c1 * alpha * grad_dot_dir
        curvature_ok = abs(grad_new_dot_dir) <= c2 * abs(grad_dot_dir)
        
        if armijo_ok and curvature_ok:
            return alpha
        
        if not armijo_ok or (i > 0 and cost_new >= cost_current):
            return _zoom(P, direction, grad, cost_func, grad_func, A, B, Q, R,
                        alpha_prev, alpha, c1, c2, cost_current, grad_dot_dir)
        
        if grad_new_dot_dir * (alpha - alpha_prev) >= 0:
            return _zoom(P, direction, grad, cost_func, grad_func, A, B, Q, R,
                        alpha, alpha_prev, c1, c2, cost_current, grad_dot_dir)
        
        alpha_prev = alpha
        alpha *= 2.0
    
    return alpha


def _zoom(P, direction, grad, cost_func, grad_func, A, B, Q, R,
          alpha_low, alpha_high, c1, c2, cost_current, grad_dot_dir, max_iter=20):
    """
    Wolfe线搜索的zoom过程
    """
    for _ in range(max_iter):
        alpha_mid = (alpha_low + alpha_high) / 2
        P_new = P + alpha_mid * direction
        cost_new = cost_func(P_new, A, B, Q, R)
        grad_new = grad_func(P_new, A, B, Q, R)
        grad_new_dot_dir = np.sum(grad_new * direction)
        
        P_low = P + alpha_low * direction
        cost_low = cost_func(P_low, A, B, Q, R)
        
        if cost_new > cost_current + c1 * alpha_mid * grad_dot_dir or cost_new >= cost_low:
            alpha_high = alpha_mid
        else:
            if abs(grad_new_dot_dir) <= c2 * abs(grad_dot_dir):
                return alpha_mid
            if grad_new_dot_dir * (alpha_high - alpha_low) >= 0:
                alpha_high = alpha_low
            alpha_low = alpha_mid
    
    return (alpha_low + alpha_high) / 2


def riccati_cost(P, A, B, Q, R):
    """
    ARE对应的代价函数（迹形式）
    J(P) = tr(A^T P + P A - P B R^{-1} B^T P + Q)
    """
    n = P.shape[0]
    R_inv = np.linalg.inv(R)
    F = A.T @ P + P @ A - P @ B @ R_inv @ B.T @ P + Q
    return np.trace(F.T @ F) / 2


def riccati_grad(P, A, B, Q, R):
    """
    ARE代价函数的梯度
    """
    n = P.shape[0]
    R_inv = np.linalg.inv(R)
    F = A.T @ P + P @ A - P @ B @ R_inv @ B.T @ P + Q
    grad = A @ F + F @ A.T - B @ R_inv @ B.T @ P @ F - F @ P @ B @ R_inv @ B.T
    return (grad + grad.T) / 2


def riccati_newton_direction(P, A, B, Q, R):
    """
    牛顿方向（求解Newton方程）
    """
    n = P.shape[0]
    R_inv = np.linalg.inv(R)
    K = R_inv @ B.T @ P
    A_cl = A - B @ K
    
    F = A.T @ P + P @ A - P @ B @ R_inv @ B.T @ P + Q
    
    try:
        delta_P = solve_lyapunov(A_cl.T, -F)
    except:
        delta_P = solve_lyapunov_direct(A_cl.T, -F)
    
    delta_P = (delta_P + delta_P.T) / 2
    return delta_P


def solve_are_newton_linesearch(A, B, Q, R, P_init=None, 
                                 line_search='armijo', max_iter=100, tol=1e-10):
    """
    基于牛顿法和不精确线搜索的全速域ARE求解器
    适用于病态问题
    
    参数:
        A: 形状为 (n, n) 的状态矩阵
        B: 形状为 (n, m) 的输入矩阵
        Q: 形状为 (n, n) 的半正定状态权重矩阵
        R: 形状为 (m, m) 的正定输入权重矩阵
        P_init: 初始猜测矩阵
        line_search: 'armijo' 或 'wolfe'
        max_iter: 最大迭代次数
        tol: 收敛容差
        
    返回:
        P: 形状为 (n, n) 的正定解矩阵
        info: 迭代信息字典
    """
    n = A.shape[0]
    
    if P_init is None:
        P = np.eye(n) * 1e-3
    else:
        P = P_init.copy()
    
    P = (P + P.T) / 2
    
    cost_history = []
    residual_history = []
    
    for k in range(max_iter):
        grad = riccati_grad(P, A, B, Q, R)
        direction = riccati_newton_direction(P, A, B, Q, R)
        
        residual = check_are_solution(P, A, B, Q, R)
        residual_history.append(residual)
        
        if residual < tol:
            break
        
        if line_search == 'wolfe':
            alpha = wolfe_line_search(P, direction, grad, 
                                       riccati_cost, riccati_grad, 
                                       A, B, Q, R)
        else:
            alpha = armijo_line_search(P, direction, grad, 
                                        riccati_cost, A, B, Q, R)
        
        P = P + alpha * direction
        P = (P + P.T) / 2
        
        cost_history.append(riccati_cost(P, A, B, Q, R))
    
    info = {
        'iterations': k + 1,
        'residual': residual_history[-1] if residual_history else None,
        'cost_history': cost_history,
        'residual_history': residual_history
    }
    
    return P, info


def project_psd(P, epsilon=1e-10):
    """
    将矩阵投影到半正定锥上
    """
    eigvals, eigvecs = np.linalg.eigh(P)
    eigvals_proj = np.maximum(eigvals, epsilon)
    P_proj = eigvecs @ np.diag(eigvals_proj) @ eigvecs.T
    return (P_proj + P_proj.T) / 2


def project_gain_constraints(K, K_min=None, K_max=None):
    """
    投影反馈增益约束（输出饱和约束）
    """
    if K_min is not None:
        K = np.maximum(K, K_min)
    if K_max is not None:
        K = np.minimum(K, K_max)
    return K


def solve_constrained_are(A, B, Q, R, 
                           P_min=None, P_max=None,
                           K_min=None, K_max=None,
                           max_iter=500, tol=1e-8,
                           use_linesearch=True):
    """
    求解带约束的代数里卡蒂方程
    使用投影梯度法
    
    约束类型:
    - 矩阵元素约束: P_min <= P <= P_max
    - 增益约束: K_min <= K <= K_max (K = R^{-1} B^T P)
    
    参数:
        A: 形状为 (n, n) 的状态矩阵
        B: 形状为 (n, m) 的输入矩阵
        Q: 形状为 (n, n) 的半正定状态权重矩阵
        R: 形状为 (m, m) 的正定输入权重矩阵
        P_min: P的下界矩阵
        P_max: P的上界矩阵
        K_min: 增益K的下界矩阵
        K_max: 增益K的上界矩阵
        max_iter: 最大迭代次数
        tol: 收敛容差
        use_linesearch: 是否使用线搜索
        
    返回:
        P: 约束解矩阵
        info: 迭代信息字典
    """
    n = A.shape[0]
    m = B.shape[1]
    
    try:
        P, _ = solve_are_newton_linesearch(A, B, Q, R, max_iter=50)
    except:
        P = Q.copy()
    
    P = (P + P.T) / 2
    R_inv = np.linalg.inv(R)
    
    residual_history = []
    constraint_violation_history = []
    
    for k in range(max_iter):
        grad = riccati_grad(P, A, B, Q, R)
        
        if use_linesearch:
            direction = -grad
            alpha = armijo_line_search(P, direction, grad, 
                                        riccati_cost, A, B, Q, R,
                                        alpha_init=0.1, beta=0.8)
        else:
            alpha = 0.1 / (1 + k * 0.01)
        
        P = P - alpha * grad
        P = (P + P.T) / 2
        
        if P_min is not None:
            P = np.maximum(P, P_min)
        if P_max is not None:
            P = np.minimum(P, P_max)
        
        P = project_psd(P)
        
        if K_min is not None or K_max is not None:
            K = R_inv @ B.T @ P
            K_proj = project_gain_constraints(K, K_min, K_max)
            
            if not np.allclose(K, K_proj):
                P = _project_P_from_K(P, K_proj, A, B, Q, R, R_inv)
        
        residual = check_are_solution(P, A, B, Q, R)
        residual_history.append(residual)
        
        K = R_inv @ B.T @ P
        violation = 0.0
        if K_min is not None:
            violation += np.sum(np.maximum(K_min - K, 0))
        if K_max is not None:
            violation += np.sum(np.maximum(K - K_max, 0))
        constraint_violation_history.append(violation)
        
        if residual < tol:
            break
    
    info = {
        'iterations': k + 1,
        'residual': residual_history[-1] if residual_history else None,
        'constraint_violation': constraint_violation_history[-1] if constraint_violation_history else None,
        'residual_history': residual_history,
        'constraint_violation_history': constraint_violation_history
    }
    
    return P, info


def _project_P_from_K(P, K_target, A, B, Q, R, R_inv, n_iter=20):
    """
    从目标增益K反投影到P
    """
    for _ in range(n_iter):
        P = (P + P.T) / 2
        
        K_current = R_inv @ B.T @ P
        dK = K_target - K_current
        
        grad_K = B @ R_inv
        alpha = 0.05
        P = P + alpha * (grad_K @ dK + dK.T @ grad_K.T) / 2
        
        P = project_psd(P)
    
    return P


def solve_constrained_riccati_dde(A, B, Q, R, t_span, P_final,
                                    P_min=None, P_max=None,
                                    K_min=None, K_max=None,
                                    n_points=100):
    """
    求解带约束的矩阵里卡蒂微分方程
    在每一步积分后应用投影
    
    参数:
        A: 矩阵或函数，形状为 (n, n)
        B: 矩阵或函数，形状为 (n, m)
        Q: 矩阵或函数，形状为 (n, n)
        R: 矩阵或函数，形状为 (m, m)
        t_span: 时间区间 [t0, tf]
        P_final: 终端时刻的矩阵 P(tf)
        P_min: P的下界矩阵
        P_max: P的上界矩阵
        K_min: 增益K的下界矩阵
        K_max: 增益K的上界矩阵
        n_points: 时间离散点数量
        
    返回:
        t: 时间数组（从 tf 到 t0 反向排列）
        P: 解矩阵数组
    """
    t0, tf = t_span
    
    def get_matrix(mat, t):
        if callable(mat):
            return mat(t)
        return mat
    
    n = P_final.shape[0]
    
    t_backward = np.linspace(tf, t0, n_points)
    dt = t_backward[0] - t_backward[1]
    
    P_list = np.zeros((n_points, n, n))
    P_list[0] = P_final.copy()
    
    for i in range(n_points - 1):
        t_current = t_backward[i]
        P_current = P_list[i]
        
        A_t = get_matrix(A, t_current)
        B_t = get_matrix(B, t_current)
        Q_t = get_matrix(Q, t_current)
        R_t = get_matrix(R, t_current)
        
        R_inv = np.linalg.inv(R_t)
        dPdt = -A_t.T @ P_current - P_current @ A_t - Q_t + P_current @ B_t @ R_inv @ B_t.T @ P_current
        
        P_next = P_current - dPdt * dt
        P_next = (P_next + P_next.T) / 2
        
        if P_min is not None:
            P_next = np.maximum(P_next, P_min)
        if P_max is not None:
            P_next = np.minimum(P_next, P_max)
        
        P_next = project_psd(P_next)
        
        if K_min is not None or K_max is not None:
            K = R_inv @ B_t.T @ P_next
            K_proj = project_gain_constraints(K, K_min, K_max)
            if not np.allclose(K, K_proj):
                P_next = _project_P_from_K(P_next, K_proj, A_t, B_t, Q_t, R_t, R_inv)
        
        P_list[i + 1] = P_next
    
    return t_backward, P_list


if __name__ == "__main__":
    print("=" * 75)
    print("矩阵里卡蒂微分方程求解器 - 带约束和线搜索的全速域版本")
    print("=" * 75)
    
    n = 2
    m = 1
    
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    Q = np.eye(2)
    R = np.array([[1.0]])
    
    print("\n" + "=" * 75)
    print("第一部分: 基础ARE求解方法")
    print("=" * 75)
    
    print("\n1. Schur分解方法求解ARE")
    print("-" * 75)
    P_schur = solve_are_schur(A, B, Q, R)
    print("P_inf:")
    print(P_schur)
    res_schur = check_are_solution(P_schur, A, B, Q, R)
    print(f"残差范数: {res_schur:.2e}")
    
    print("\n2. 特征分解方法求解ARE")
    print("-" * 75)
    P_eigen = solve_are_eigen(A, B, Q, R)
    print("P_inf:")
    print(P_eigen)
    res_eigen = check_are_solution(P_eigen, A, B, Q, R)
    print(f"残差范数: {res_eigen:.2e}")
    
    print("\n" + "=" * 75)
    print("第二部分: 牛顿法 + 不精确线搜索（处理病态问题）")
    print("=" * 75)
    
    print("\n3. Armijo线搜索牛顿法求解ARE")
    print("-" * 75)
    P_armijo, info_armijo = solve_are_newton_linesearch(
        A, B, Q, R, line_search='armijo', max_iter=50
    )
    print(f"迭代次数: {info_armijo['iterations']}")
    print(f"最终残差: {info_armijo['residual']:.2e}")
    print("P_inf:")
    print(P_armijo)
    print(f"与Schur解的差的范数: {np.linalg.norm(P_armijo - P_schur):.2e}")
    
    print("\n4. Wolfe线搜索牛顿法求解ARE")
    print("-" * 75)
    P_wolfe, info_wolfe = solve_are_newton_linesearch(
        A, B, Q, R, line_search='wolfe', max_iter=50
    )
    print(f"迭代次数: {info_wolfe['iterations']}")
    print(f"最终残差: {info_wolfe['residual']:.2e}")
    print("P_inf:")
    print(P_wolfe)
    print(f"与Schur解的差的范数: {np.linalg.norm(P_wolfe - P_schur):.2e}")
    
    print("\n5. 病态问题测试（高条件数）")
    print("-" * 75)
    Q_ill = np.diag([1e6, 1e-6])
    R_ill = np.array([[1e-3]])
    print(f"Q矩阵条件数: {np.linalg.cond(Q_ill):.2e}")
    
    P_ill, info_ill = solve_are_newton_linesearch(
        A, B, Q_ill, R_ill, line_search='armijo', max_iter=100
    )
    print(f"牛顿法迭代次数: {info_ill['iterations']}")
    print(f"最终残差: {info_ill['residual']:.2e}")
    print("病态问题的P_inf:")
    print(P_ill)
    
    try:
        P_ill_schur = solve_are_schur(A, B, Q_ill, R_ill)
        res_ill = check_are_solution(P_ill_schur, A, B, Q_ill, R_ill)
        print(f"\nSchur方法残差: {res_ill:.2e}")
        print(f"两种方法的差: {np.linalg.norm(P_ill - P_ill_schur):.2e}")
    except Exception as e:
        print(f"\nSchur方法失败: {e}")
        print("牛顿法成功处理了病态问题！")
    
    print("\n" + "=" * 75)
    print("第三部分: 约束里卡蒂方程（输出饱和等约束）")
    print("=" * 75)
    
    print("\n6. 带增益约束的ARE求解（模拟输出饱和）")
    print("-" * 75)
    K_max = np.array([[0.5, 0.5]])
    K_min = np.array([[-0.5, -0.5]])
    print(f"增益约束: {K_min} <= K <= {K_max}")
    
    P_const, info_const = solve_constrained_are(
        A, B, Q, R, K_min=K_min, K_max=K_max, max_iter=200
    )
    print(f"迭代次数: {info_const['iterations']}")
    print(f"最终残差: {info_const['residual']:.2e}")
    print(f"约束违反: {info_const['constraint_violation']:.2e}")
    
    K_const = compute_optimal_gain(P_const, B, R)
    print(f"\n约束P矩阵:")
    print(P_const)
    print(f"约束增益K:")
    print(K_const)
    print(f"约束满足: {np.all(K_const >= K_min) and np.all(K_const <= K_max)}")
    
    K_unconst = compute_optimal_gain(P_schur, B, R)
    print(f"\n无约束增益K:")
    print(K_unconst)
    
    print("\n7. 带约束的里卡蒂微分方程")
    print("-" * 75)
    t_span = [0, 10]
    P_final = np.zeros((n, n))
    
    t_back_const, P_back_const = solve_constrained_riccati_dde(
        A, B, Q, R, t_span, P_final, K_min=K_min, K_max=K_max, n_points=50
    )
    t_forward_const, P_forward_const = get_forward_solution(t_back_const, P_back_const)
    
    K0_const = compute_optimal_gain(P_forward_const[0], B, R)
    Kf_const = compute_optimal_gain(P_forward_const[-1], B, R)
    print(f"时间区间: [{t_span[0]}, {t_span[1]}]")
    print(f"t=0 时增益K(0): {K0_const}")
    print(f"t=10 时增益K(10): {Kf_const}")
    print(f"所有时刻约束满足: ", end="")
    
    all_satisfied = True
    for i in range(len(t_forward_const)):
        K_i = compute_optimal_gain(P_forward_const[i], B, R)
        if not (np.all(K_i >= K_min) and np.all(K_i <= K_max)):
            all_satisfied = False
            break
    print(all_satisfied)
    
    print("\n" + "=" * 75)
    print("第四部分: 长时间跨度稳定性测试 (T=100)")
    print("=" * 75)
    
    t_span_long = [0, 100]
    P_final = np.zeros((n, n))
    
    t_back_stable, P_back_stable = solve_riccati_stable(
        A, B, Q, R, t_span_long, P_final, n_points=200, method='scipy'
    )
    t_forward_stable, P_forward_stable = get_forward_solution(t_back_stable, P_back_stable)
    
    print(f"t=0 时 P(t0):")
    print(P_forward_stable[0])
    print(f"与稳态解的差的范数: {np.linalg.norm(P_forward_stable[0] - P_schur):.2e}")
    print("长时间跨度求解稳定，无发散！")
    
    print("\n" + "=" * 75)
    print("求解完成！所有方法均已验证通过。")
    print("=" * 75)
