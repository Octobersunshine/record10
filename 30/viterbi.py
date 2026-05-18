import numpy as np

def forward(pi, A, B, observations, eps=1e-10):
    """
    前向算法，计算alpha概率
    
    参数:
        pi: 初始状态概率向量 (N,)
        A: 状态转移矩阵 (N, N)
        B: 发射概率矩阵 (N, M)
        observations: 观测序列 (T,)
        eps: 平滑参数
    
    返回:
        alpha: 前向概率矩阵 (T, N)
    """
    N = len(pi)
    T = len(observations)
    M = B.shape[1]
    
    pi_smooth = np.maximum(pi, eps)
    A_smooth = np.maximum(A, eps)
    B_smooth = np.maximum(B, eps)
    
    pi_smooth = pi_smooth / np.sum(pi_smooth)
    for j in range(N):
        A_smooth[j, :] = A_smooth[j, :] / np.sum(A_smooth[j, :])
        B_smooth[j, :] = B_smooth[j, :] / np.sum(B_smooth[j, :])
    
    alpha = np.zeros((T, N))
    alpha[0] = pi_smooth * B_smooth[:, observations[0] % M]
    
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.sum(alpha[t-1] * A_smooth[:, j]) * B_smooth[j, observations[t] % M]
    
    return alpha

def backward(pi, A, B, observations, eps=1e-10):
    """
    后向算法，计算beta概率
    
    参数:
        pi: 初始状态概率向量 (N,)
        A: 状态转移矩阵 (N, N)
        B: 发射概率矩阵 (N, M)
        observations: 观测序列 (T,)
        eps: 平滑参数
    
    返回:
        beta: 后向概率矩阵 (T, N)
    """
    N = len(pi)
    T = len(observations)
    M = B.shape[1]
    
    A_smooth = np.maximum(A, eps)
    B_smooth = np.maximum(B, eps)
    
    for j in range(N):
        A_smooth[j, :] = A_smooth[j, :] / np.sum(A_smooth[j, :])
        B_smooth[j, :] = B_smooth[j, :] / np.sum(B_smooth[j, :])
    
    beta = np.zeros((T, N))
    beta[T-1] = np.ones(N)
    
    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[t, i] = np.sum(A_smooth[i, :] * B_smooth[:, observations[t+1] % M] * beta[t+1])
    
    return beta

def baum_welch(observations, n_states, n_obs, max_iter=100, tol=1e-6, eps=1e-10):
    """
    Baum-Welch算法（EM算法）学习HMM参数
    
    参数:
        observations: 观测序列 (T,)
        n_states: 隐藏状态数量 N
        n_obs: 观测值数量 M
        max_iter: 最大迭代次数
        tol: 收敛阈值
        eps: 平滑参数
    
    返回:
        pi: 学习到的初始状态概率 (N,)
        A: 学习到的状态转移矩阵 (N, N)
        B: 学习到的发射概率矩阵 (N, M)
    """
    T = len(observations)
    N = n_states
    M = n_obs
    
    pi = np.random.rand(N)
    pi = pi / np.sum(pi)
    
    A = np.random.rand(N, N)
    for i in range(N):
        A[i, :] = A[i, :] / np.sum(A[i, :])
    
    B = np.random.rand(N, M)
    for i in range(N):
        B[i, :] = B[i, :] / np.sum(B[i, :])
    
    log_likelihood_prev = -np.inf
    
    for iteration in range(max_iter):
        alpha = forward(pi, A, B, observations, eps)
        beta = backward(pi, A, B, observations, eps)
        
        gamma = np.zeros((T, N))
        for t in range(T):
            denom = np.sum(alpha[t] * beta[t])
            if denom < eps:
                denom = eps
            gamma[t] = (alpha[t] * beta[t]) / denom
        
        xi = np.zeros((T-1, N, N))
        for t in range(T-1):
            denom = np.sum(alpha[t, :, np.newaxis] * A * B[:, observations[t+1] % M] * beta[t+1])
            if denom < eps:
                denom = eps
            xi[t] = (alpha[t, :, np.newaxis] * A * B[:, observations[t+1] % M] * beta[t+1]) / denom
        
        pi_new = gamma[0]
        
        A_new = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                numer = np.sum(xi[:, i, j])
                denom = np.sum(gamma[:-1, i])
                if denom < eps:
                    denom = eps
                A_new[i, j] = numer / denom
        
        B_new = np.zeros((N, M))
        for j in range(N):
            for k in range(M):
                mask = (observations % M == k)
                numer = np.sum(gamma[mask, j])
                denom = np.sum(gamma[:, j])
                if denom < eps:
                    denom = eps
                B_new[j, k] = numer / denom
        
        pi_new = np.maximum(pi_new, eps)
        A_new = np.maximum(A_new, eps)
        B_new = np.maximum(B_new, eps)
        
        pi_new = pi_new / np.sum(pi_new)
        for i in range(N):
            A_new[i, :] = A_new[i, :] / np.sum(A_new[i, :])
            B_new[i, :] = B_new[i, :] / np.sum(B_new[i, :])
        
        pi, A, B = pi_new, A_new, B_new
        
        alpha_final = forward(pi, A, B, observations, eps)
        log_likelihood = np.log(np.sum(alpha_final[-1]))
        
        if abs(log_likelihood - log_likelihood_prev) < tol:
            break
        
        log_likelihood_prev = log_likelihood
    
    return pi, A, B

def viterbi(pi, A, B, observations, eps=1e-10, use_log=True):
    """
    Viterbi算法实现（修复发射概率为0的问题）
    
    参数:
        pi: 初始状态概率向量，形状为 (N,)
        A: 状态转移矩阵，形状为 (N, N)
        B: 发射概率矩阵，形状为 (N, M)
        observations: 观测序列，形状为 (T,)
        eps: 平滑参数，用于避免概率为0
        use_log: 是否使用对数概率计算，避免数值下溢
    
    返回:
        best_path: 最可能的状态序列
        best_path_prob: 最可能路径的概率（原始概率，非对数）
    """
    N = len(pi)
    T = len(observations)
    M = B.shape[1]
    
    pi_smooth = np.maximum(pi, eps)
    A_smooth = np.maximum(A, eps)
    B_smooth = np.maximum(B, eps)
    
    for j in range(N):
        if j < len(pi_smooth):
            pi_smooth[j] = pi_smooth[j] / np.sum(pi_smooth)
        A_smooth[j, :] = A_smooth[j, :] / np.sum(A_smooth[j, :])
        B_smooth[j, :] = B_smooth[j, :] / np.sum(B_smooth[j, :])
    
    if use_log:
        log_pi = np.log(pi_smooth)
        log_A = np.log(A_smooth)
        log_B = np.log(B_smooth)
        
        log_delta = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)
        
        log_delta[0] = log_pi + log_B[:, observations[0] % M]
        
        for t in range(1, T):
            for j in range(N):
                temp = log_delta[t-1] + log_A[:, j]
                log_delta[t, j] = np.max(temp) + log_B[j, observations[t] % M]
                psi[t, j] = np.argmax(temp)
        
        best_log_prob = np.max(log_delta[T-1])
        best_last_state = np.argmax(log_delta[T-1])
        best_path_prob = np.exp(best_log_prob)
    else:
        delta = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)
        
        delta[0] = pi_smooth * B_smooth[:, observations[0] % M]
        
        for t in range(1, T):
            for j in range(N):
                temp = delta[t-1] * A_smooth[:, j]
                delta[t, j] = np.max(temp) * B_smooth[j, observations[t] % M]
                psi[t, j] = np.argmax(temp)
        
        best_path_prob = np.max(delta[T-1])
        best_last_state = np.argmax(delta[T-1])
    
    best_path = np.zeros(T, dtype=int)
    best_path[T-1] = best_last_state
    
    for t in range(T-2, -1, -1):
        best_path[t] = psi[t+1, best_path[t+1]]
    
    return best_path, best_path_prob

if __name__ == "__main__":
    pi = np.array([0.6, 0.4])
    A = np.array([[0.7, 0.3],
                  [0.4, 0.6]])
    B = np.array([[0.1, 0.4, 0.5],
                  [0.6, 0.3, 0.1]])
    
    print("=== 正常情况测试 ===")
    observations = np.array([0, 1, 2])
    best_path, best_prob = viterbi(pi, A, B, observations)
    print(f"最可能的状态序列: {best_path}")
    print(f"最可能路径的概率: {best_prob}")
    
    print("\n=== 未见过的观测值测试 ===")
    observations_unk = np.array([0, 1, 5])
    best_path_unk, best_prob_unk = viterbi(pi, A, B, observations_unk)
    print(f"包含未见过观测值的观测序列: {observations_unk}")
    print(f"最可能的状态序列: {best_path_unk}")
    print(f"最可能路径的概率: {best_prob_unk}")
    
    print("\n=== 发射概率含0的测试 ===")
    B_with_zero = np.array([[0.0, 0.4, 0.6],
                            [0.6, 0.4, 0.0]])
    observations_zero = np.array([0, 2, 1])
    best_path_zero, best_prob_zero = viterbi(pi, A, B_with_zero, observations_zero)
    print(f"发射概率矩阵:\n{B_with_zero}")
    print(f"观测序列: {observations_zero}")
    print(f"最可能的状态序列: {best_path_zero}")
    print(f"最可能路径的概率: {best_prob_zero}")
    
    print("\n=== Baum-Welch参数学习测试 ===")
    np.random.seed(42)
    
    true_pi = np.array([0.6, 0.4])
    true_A = np.array([[0.7, 0.3],
                       [0.4, 0.6]])
    true_B = np.array([[0.1, 0.4, 0.5],
                       [0.6, 0.3, 0.1]])
    
    T_long = 1000
    states = np.zeros(T_long, dtype=int)
    obs_train = np.zeros(T_long, dtype=int)
    
    states[0] = np.random.choice(2, p=true_pi)
    obs_train[0] = np.random.choice(3, p=true_B[states[0]])
    for t in range(1, T_long):
        states[t] = np.random.choice(2, p=true_A[states[t-1]])
        obs_train[t] = np.random.choice(3, p=true_B[states[t]])
    
    print(f"生成的观测序列长度: {T_long}")
    print(f"真实初始概率: {true_pi}")
    print(f"真实转移矩阵:\n{true_A}")
    print(f"真实发射矩阵:\n{true_B}")
    
    learned_pi, learned_A, learned_B = baum_welch(obs_train, n_states=2, n_obs=3, max_iter=50)
    
    print("\n学习到的初始概率:", learned_pi)
    print("学习到的转移矩阵:\n", learned_A)
    print("学习到的发射矩阵:\n", learned_B)
    
    obs_test = np.array([0, 1, 2, 0, 1])
    best_path_learned, best_prob_learned = viterbi(learned_pi, learned_A, learned_B, obs_test)
    print(f"\n用学习到的参数进行Viterbi解码:")
    print(f"观测序列: {obs_test}")
    print(f"最可能状态序列: {best_path_learned}")
    print(f"路径概率: {best_prob_learned}")
