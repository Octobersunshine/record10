import numpy as np


def logsumexp(x, axis=None):
    x_max = np.max(x, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(x - x_max), axis=axis)) + np.squeeze(x_max, axis=axis)


class HMM:
    def __init__(self, A, B, pi):
        self.A = np.array(A)
        self.B = np.array(B)
        self.pi = np.array(pi)
        self.N = self.A.shape[0]
        self.M = self.B.shape[1]
        self.log_A = np.log(self.A)
        self.log_B = np.log(self.B)
        self.log_pi = np.log(self.pi)

    def forward_log(self, observations):
        T = len(observations)
        log_alpha = np.zeros((T, self.N))
        log_alpha[0] = self.log_pi + self.log_B[:, observations[0]]
        for t in range(1, T):
            for j in range(self.N):
                log_alpha[t, j] = logsumexp(log_alpha[t-1] + self.log_A[:, j]) + self.log_B[j, observations[t]]
        return log_alpha

    def backward_log(self, observations):
        T = len(observations)
        log_beta = np.zeros((T, self.N))
        log_beta[T-1] = np.zeros(self.N)
        for t in range(T-2, -1, -1):
            for i in range(self.N):
                log_beta[t, i] = logsumexp(self.log_A[i, :] + self.log_B[:, observations[t+1]] + log_beta[t+1])
        return log_beta

    def filtering(self, observations):
        log_alpha = self.forward_log(observations)
        log_P_O = logsumexp(log_alpha[-1])
        filtering_probs = np.exp(log_alpha - log_P_O)
        return filtering_probs

    def smoothing(self, observations):
        T = len(observations)
        log_alpha = self.forward_log(observations)
        log_beta = self.backward_log(observations)
        log_P_O = logsumexp(log_alpha[-1])
        smoothing_probs = np.zeros((T, self.N))
        for t in range(T):
            smoothing_probs[t] = np.exp(log_alpha[t] + log_beta[t] - log_P_O)
        return smoothing_probs

    def forward(self, observations):
        return np.exp(self.forward_log(observations))

    def backward(self, observations):
        return np.exp(self.backward_log(observations))


def main():
    A = [[0.7, 0.3],
         [0.4, 0.6]]
    B = [[0.1, 0.4, 0.5],
         [0.7, 0.2, 0.1]]
    pi = [0.6, 0.4]
    hmm = HMM(A, B, pi)
    
    observations = [0, 1, 2]
    print("=" * 60)
    print("短序列测试 (T=3)")
    print("=" * 60)
    print("观测序列:", observations)
    
    filter_probs = hmm.filtering(observations)
    print("\n滤波概率 P(X_t | Y_{1:t}):")
    for t in range(len(observations)):
        print(f"t={t+1}: {filter_probs[t]}")
    
    smooth_probs = hmm.smoothing(observations)
    print("\n平滑概率 P(X_t | Y_{1:T}):")
    for t in range(len(observations)):
        print(f"t={t+1}: {smooth_probs[t]}")
    
    print("\n" + "=" * 60)
    print("长序列稳定性验证 (T=100)")
    print("=" * 60)
    
    np.random.seed(42)
    long_observations = np.random.randint(0, 3, size=100)
    
    log_alpha = hmm.forward_log(long_observations)
    log_beta = hmm.backward_log(long_observations)
    
    print(f"对数域前向变量范围:")
    print(f"  log_alpha[0] = {log_alpha[0]}")
    print(f"  log_alpha[50] = {log_alpha[50]}")
    print(f"  log_alpha[-1] = {log_alpha[-1]}")
    
    print(f"\n对数域后向变量范围:")
    print(f"  log_beta[0] = {log_beta[0]}")
    print(f"  log_beta[50] = {log_beta[50]}")
    print(f"  log_beta[-1] = {log_beta[-1]}")
    
    long_filter_probs = hmm.filtering(long_observations)
    print(f"\n长序列滤波概率 (t=1, 50, 100):")
    print(f"  t=1: {long_filter_probs[0]}")
    print(f"  t=50: {long_filter_probs[49]}")
    print(f"  t=100: {long_filter_probs[99]}")
    
    print(f"\n验证概率归一性检查:")
    print(f"  t=50 滤波概率和: {np.sum(long_filter_probs[49]):.6f}")


def multinomial_resampling(weights, N):
    indices = np.random.choice(N, size=N, p=weights)
    return indices


def systematic_resampling(weights, N):
    positions = (np.arange(N) + np.random.uniform(0, 1)) / N
    indices = np.zeros(N, dtype=int)
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
    return indices


class ParticleFilter:
    def __init__(self, num_particles, state_dim, obs_dim,
                 transition_fn, observation_fn,
                 initial_particle_fn, resampling_method='systematic'):
        self.N = num_particles
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.transition_fn = transition_fn
        self.observation_fn = observation_fn
        self.initial_particle_fn = initial_particle_fn
        self.resampling_method = resampling_method
        self.particles = None
        self.weights = None
        self.ess_history = []

    def initialize(self):
        self.particles = self.initial_particle_fn(self.N)
        self.weights = np.ones(self.N) / self.N

    def predict(self):
        self.particles = self.transition_fn(self.particles)

    def update(self, observation):
        likelihoods = self.observation_fn(self.particles, observation)
        self.weights *= likelihoods
        self.weights /= np.sum(self.weights)

    def resample(self):
        ess = 1.0 / np.sum(self.weights ** 2)
        self.ess_history.append(ess)
        if ess < self.N / 2:
            if self.resampling_method == 'multinomial':
                indices = multinomial_resampling(self.weights, self.N)
            elif self.resampling_method == 'systematic':
                indices = systematic_resampling(self.weights, self.N)
            else:
                raise ValueError(f"Unknown resampling method: {self.resampling_method}")
            self.particles = self.particles[indices]
            self.weights = np.ones(self.N) / self.N

    def filter(self, observations):
        T = len(observations)
        self.initialize()
        estimates = np.zeros((T, self.state_dim))
        for t in range(T):
            self.predict()
            self.update(observations[t])
            self.resample()
            estimates[t] = np.sum(self.particles * self.weights[:, np.newaxis], axis=0)
        return estimates

    def get_particles(self):
        return self.particles

    def get_weights(self):
        return self.weights


def nonlinear_transition(particles):
    return 0.5 * particles + 25 * particles / (1 + particles ** 2) + np.random.randn(*particles.shape) * np.sqrt(10)


def nonlinear_observation(particles, observation):
    obs_pred = particles ** 2 / 20
    diff = observation - obs_pred
    return np.exp(-0.5 * np.sum(diff ** 2, axis=1) / 1.0)


def initial_particles(N):
    return np.random.randn(N, 1) * 5


def main_particle_filter():
    print("\n" + "=" * 60)
    print("粒子滤波测试 (非线性非高斯模型)")
    print("=" * 60)

    np.random.seed(42)
    T = 50
    true_states = np.zeros((T, 1))
    true_states[0] = np.random.randn() * 5
    for t in range(1, T):
        true_states[t] = 0.5 * true_states[t-1] + 25 * true_states[t-1] / (1 + true_states[t-1] ** 2) + 8 * np.cos(1.2 * t) + np.random.randn() * np.sqrt(10)
    observations = true_states ** 2 / 20 + np.random.randn(*true_states.shape)

    print(f"\n真实状态 (前5个):", true_states[:5].flatten())
    print(f"观测值 (前5个):", observations[:5].flatten())

    print("\n--- 系统重采样 ---")
    pf_sys = ParticleFilter(
        num_particles=500,
        state_dim=1,
        obs_dim=1,
        transition_fn=nonlinear_transition,
        observation_fn=nonlinear_observation,
        initial_particle_fn=initial_particles,
        resampling_method='systematic'
    )
    estimates_sys = pf_sys.filter(observations)
    mse_sys = np.mean((estimates_sys - true_states) ** 2)
    print(f"系统重采样 MSE: {mse_sys:.4f}")
    print(f"ESS 范围: [{min(pf_sys.ess_history):.1f} ~ {max(pf_sys.ess_history):.1f}]")

    print("\n--- 多项式重采样 ---")
    pf_mult = ParticleFilter(
        num_particles=500,
        state_dim=1,
        obs_dim=1,
        transition_fn=nonlinear_transition,
        observation_fn=nonlinear_observation,
        initial_particle_fn=initial_particles,
        resampling_method='multinomial'
    )
    estimates_mult = pf_mult.filter(observations)
    mse_mult = np.mean((estimates_mult - true_states) ** 2)
    print(f"多项式重采样 MSE: {mse_mult:.4f}")
    print(f"ESS 范围: [{min(pf_mult.ess_history):.1f} ~ {max(pf_mult.ess_history):.1f}]")

    print("\n状态估计 (前5个时间步):")
    print(f"  真实值: {true_states[:5].flatten()}")
    print(f"  系统重采样估计: {estimates_sys[:5].flatten()}")
    print(f"  多项式重采样估计: {estimates_mult[:5].flatten()}")


if __name__ == "__main__":
    main()
    main_particle_filter()
