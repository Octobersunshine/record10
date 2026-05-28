import numpy as np
from numpy.linalg import eig, inv, norm


_PADE13_B = np.array([
    64764752532480000,
    32382376266240000,
    7771770303897600,
    1187353796428800,
    129060195264000,
    10559470521600,
    670442572800,
    33522128640,
    1323241920,
    40840800,
    960960,
    16380,
    182,
    1,
])

_THETA_13 = 5.371920351148152


def matrix_exp_pade(A):
    """
    通过 Pade 逼近 (scaling and squaring) 计算矩阵指数 e^A。

    基于 Higham (2005) "The Scaling and Squaring Method for the
    Matrix Exponential Revisited" 的算法：

    1. 缩放: 选择 s 使得 ‖A/2^s‖₁ ≤ θ₁₃ ≈ 5.37
    2. [13/13] Pade 逼近: 利用偶/奇分解
         N(A) = U + A·V,   D(A) = U - A·V
       其中 U, V 仅含 A² 的多项式，仅需 6 次矩阵乘法
    3. 校正平方: r = R - I, R_new = I + 2r + r²
       比 R = R @ R 在 R ≈ I 时数值更稳定

    参数:
        A: numpy 方阵

    返回:
        e^A 的 numpy 矩阵
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    assert A.shape == (n, n), "输入必须是方阵"

    I = np.eye(n)

    norm_A = norm(A, 1)

    if norm_A == 0:
        return I.copy()

    if norm_A <= _THETA_13:
        s = 0
    else:
        s = max(0, int(np.ceil(np.log2(norm_A / _THETA_13))))

    A_scaled = A / (2 ** s)

    A2 = A_scaled @ A_scaled
    A4 = A2 @ A2
    A6 = A2 @ A4

    b = _PADE13_B

    U = A_scaled @ (A6 @ (b[13]*A6 + b[11]*A4 + b[9]*A2) +
                    b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*I)

    V = A6 @ (b[12]*A6 + b[10]*A4 + b[8]*A2) + \
        b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*I

    N = V + U
    D = V - U

    R = np.linalg.solve(D, N)

    for _ in range(s):
        r = R - I
        R = I + 2 * r + r @ r

    return R


def matrix_exp_eig(A):
    """
    通过特征值分解法计算矩阵指数 e^A = P e^D P^{-1}。

    适用于可对角化矩阵。若矩阵不可对角化，退化为数值近似
    （利用 numpy.linalg.eig 总是返回特征向量）。

    参数:
        A: numpy 方阵

    返回:
        e^A 的 numpy 矩阵
    """
    A = np.asarray(A, dtype=complex)
    n = A.shape[0]
    assert A.shape == (n, n), "输入必须是方阵"

    eigenvalues, P = eig(A)

    exp_D = np.diag(np.exp(eigenvalues))

    result = P @ exp_D @ inv(P)

    if np.all(np.abs(result.imag) < 1e-10):
        result = result.real

    return result


def matrix_exp(A, method="pade"):
    """
    计算矩阵指数 e^A。

    参数:
        A: numpy 方阵
        method: "pade" 使用 Pade 逼近, "eig" 使用特征值分解法

    返回:
        e^A 的 numpy 矩阵
    """
    if method == "eig":
        return matrix_exp_eig(A)
    elif method == "pade":
        return matrix_exp_pade(A)
    else:
        raise ValueError(f"未知方法: {method}，可选 'pade' 或 'eig'")


def matrix_log_eig(A):
    """
    通过特征值分解法计算矩阵对数 log(A) = P log(D) P^{-1}。

    若 A = P D P^{-1}，则 log(A) = P diag(log(λ₁), ..., log(λₙ)) P^{-1}。

    注意:
        - A 必须可对角化
        - A 的特征值不能为 0 或负实数（否则主值分支可能产生复数）

    参数:
        A: numpy 方阵，要求可对角化且非奇异

    返回:
        log(A) 的 numpy 矩阵（若为实数矩阵则返回复数，纯实数）
    """
    A = np.asarray(A, dtype=complex)
    n = A.shape[0]
    assert A.shape == (n, n), "输入必须是方阵"

    eigenvalues, P = eig(A)

    log_eigenvalues = np.log(eigenvalues)
    log_D = np.diag(log_eigenvalues)

    result = P @ log_D @ inv(P)

    if np.all(np.abs(result.imag) < 1e-10):
        result = result.real

    return result


def matrix_log(A, method="eig"):
    """
    计算矩阵对数 log(A)，满足 exp(log(A)) = A。

    参数:
        A: numpy 方阵，非奇异，可对角化
        method: "eig" 使用特征值分解法

    返回:
        log(A) 的 numpy 矩阵
    """
    if method == "eig":
        return matrix_log_eig(A)
    else:
        raise ValueError(f"未知方法: {method}，当前仅支持 'eig'")


def state_transition_lti(A, t, t0=0.0, method="pade"):
    """
    线性时不变 (LTI) 系统的状态转移矩阵 Φ(t, t₀) = e^{A(t - t₀)}。

    对于 LTI 系统 dx/dt = A x，状态转移矩阵满足：
      dΦ/dt = A Φ
      Φ(t₀, t₀) = I
      x(t) = Φ(t, t₀) x(t₀)

    参数:
        A: n×n 常数矩阵（系统矩阵）
        t: 目标时刻
        t0: 初始时刻（默认 0）
        method: "pade" 或 "eig"

    返回:
        Φ(t, t₀) = e^{A(t - t₀)}
    """
    delta_t = t - t0
    return matrix_exp(A * delta_t, method=method)


def _rk4_step(f, t, y, h):
    """
    经典四阶 Runge-Kutta 单步积分器。

    参数:
        f: f(t, y) = A(t) @ y(t)
        t: 当前时刻
        y: 当前状态矩阵
        h: 步长

    返回:
        (t + h, y_new)
    """
    k1 = f(t, y)
    k2 = f(t + h/2, y + h/2 * k1)
    k3 = f(t + h/2, y + h/2 * k2)
    k4 = f(t + h, y + h * k3)

    y_new = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    return t + h, y_new


def state_transition_ltv(A_func, t, t0, n_steps=1000):
    """
    线性时变 (LTV) 系统的状态转移矩阵 Φ(t, t₀)。

    对于 LTV 系统 dx/dt = A(t) x，状态转移矩阵满足：
      dΦ/dt = A(t) Φ
      Φ(t₀, t₀) = I
      x(t) = Φ(t, t₀) x(t₀)

    使用固定步长四阶 Runge-Kutta (RK4) 数值积分求解。

    参数:
        A_func: 函数 A(t) 返回 n×n 时变系统矩阵
        t: 目标时刻
        t0: 初始时刻
        n_steps: 积分步数（默认 1000，步数越多精度越高）

    返回:
        Φ(t, t₀) 状态转移矩阵
    """
    t_current = t0
    n = A_func(t0).shape[0]
    Phi = np.eye(n)

    if t == t0:
        return Phi

    h = (t - t0) / n_steps

    def f(t, y):
        return A_func(t) @ y

    for _ in range(n_steps):
        t_current, Phi = _rk4_step(f, t_current, Phi, h)

    return Phi


if __name__ == "__main__":
    print("=" * 60)
    print("矩阵指数、对数与状态转移矩阵")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("测试 1: 矩阵对数 log(exp(A)) = A")
    print("=" * 60)

    A = np.array([
        [1, 2],
        [0, 3]
    ], dtype=float)

    print(f"\n输入矩阵 A:\n{A}")

    expA = matrix_exp(A, method="pade")
    log_expA = matrix_log(expA, method="eig")

    print(f"\ne^A:\n{expA}")
    print(f"\nlog(e^A):\n{log_expA}")
    print(f"\n||log(e^A) - A||₁ = {norm(log_expA - A, 1):.2e}")

    try:
        from scipy.linalg import expm, logm
        scipy_expA = expm(A)
        scipy_log_expA = logm(scipy_expA)
        print(f"SciPy ||log(e^A) - A||₁ = {norm(scipy_log_expA - A, 1):.2e}")
    except ImportError:
        print("SciPy 未安装，跳过参考对比")

    print("\n" + "=" * 60)
    print("测试 2: LTI 状态转移矩阵 Φ(t, 0) = e^(A t)")
    print("=" * 60)

    A_lti = np.array([
        [0, 1],
        [-1, -1]
    ], dtype=float)

    t_vals = [0.1, 0.5, 1.0, 2.0]

    print(f"\nLTI 系统矩阵 A:\n{A_lti}")
    print(f"特征值: {np.linalg.eigvals(A_lti)}")

    for t in t_vals:
        Phi = state_transition_lti(A_lti, t)
        print(f"\nΦ({t}, 0) = e^({t} A):\n{Phi}")

    print("\n" + "=" * 60)
    print("测试 3: 半群性质 Φ(t2, t0) = Φ(t2, t1) Φ(t1, t0)")
    print("=" * 60)

    t0, t1, t2 = 0.0, 0.5, 1.0
    Phi_t2_t0 = state_transition_lti(A_lti, t2, t0)
    Phi_t2_t1 = state_transition_lti(A_lti, t2, t1)
    Phi_t1_t0 = state_transition_lti(A_lti, t1, t0)
    composed = Phi_t2_t1 @ Phi_t1_t0

    print(f"\nΦ({t2}, {t0}):\n{Phi_t2_t0}")
    print(f"\nΦ({t2}, {t1}) @ Φ({t1}, {t0}):\n{composed}")
    print(f"\n||Φ(t2,t0) - Φ(t2,t1)Φ(t1,t0)||₁ = {norm(Phi_t2_t0 - composed, 1):.2e}")

    print("\n" + "=" * 60)
    print("测试 4: LTV 状态转移矩阵 (谐振子 + 时变阻尼)")
    print("=" * 60)
    print("  dx/dt = A(t) x, A(t) = [[0, 1], [-1, -cos(t)]]")

    def A_ltv(t):
        return np.array([
            [0, 1],
            [-1, -np.cos(t)]
        ])

    t0, t_final = 0.0, 3.14159
    Phi_ltv = state_transition_ltv(A_ltv, t_final, t0)

    print(f"\nΦ({t_final:.4f}, {t0}):\n{Phi_ltv}")

    x0 = np.array([1.0, 0.0])
    x_final = Phi_ltv @ x0
    print(f"\nx({t_final:.4f}) = Φ @ x(0) = {x_final}")

    print("\n" + "=" * 60)
    print("测试 5: LTV 半群性质验证")
    print("=" * 60)

    t0, t1, t2 = 0.0, 0.5, 1.0
    Phi_t2_t0 = state_transition_ltv(A_ltv, t2, t0)
    Phi_t2_t1 = state_transition_ltv(A_ltv, t2, t1)
    Phi_t1_t0 = state_transition_ltv(A_ltv, t1, t0)
    composed = Phi_t2_t1 @ Phi_t1_t0

    print(f"||Φ(t2,t0) - Φ(t2,t1)Φ(t1,t0)||₁ = {norm(Phi_t2_t0 - composed, 1):.2e}")

    print("\n" + "=" * 60)
    print("测试 6: LTI 与 LTV 一致性 (A 为常数时)")
    print("=" * 60)

    def A_const(t):
        return A_lti

    t_test = 1.0
    Phi_lti = state_transition_lti(A_lti, t_test)
    Phi_ltv_from_const = state_transition_ltv(A_const, t_test, 0.0)

    print(f"LTI Φ(1, 0):\n{Phi_lti}")
    print(f"\nLTV (常 A) Φ(1, 0):\n{Phi_ltv_from_const}")
    print(f"\n||LTI - LTV||₁ = {norm(Phi_lti - Phi_ltv_from_const, 1):.2e}")
