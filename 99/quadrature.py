import numpy as np


def gauss_legendre_quadrature(n):
    if n % 2 != 0:
        raise ValueError("n must be even for symmetric quadrature")
    
    mu, w = np.polynomial.legendre.leggauss(n)
    return mu, w


def level_symmetric_quadrature(n):
    if n not in [2, 4, 6, 8, 12, 16]:
        raise ValueError("Level symmetric quadrature available for n = 2, 4, 6, 8, 12, 16")
    
    quadrature_data = {
        2: {
            'mu': [0.577350269189626],
            'w': [1.000000000000000]
        },
        4: {
            'mu': [0.350021174581547, 0.868890300662212],
            'w': [0.652145154862546, 0.347854845137454]
        },
        6: {
            'mu': [0.238619186083197, 0.661209386466265, 0.932469514203152],
            'w': [0.467913934572691, 0.360761573048139, 0.171324492379170]
        },
        8: {
            'mu': [0.183434642495650, 0.525532409916329, 0.796666477413627, 0.960289856497536],
            'w': [0.362683783378362, 0.313706645877887, 0.222381034453374, 0.101228536290376]
        },
        12: {
            'mu': [0.125233408511469, 0.367831498998180, 0.587317954286617,
                   0.769902674194305, 0.904117256370475, 0.981560634246719],
            'w': [0.249147045813403, 0.233492536538355, 0.203167426723066,
                  0.160078328543346, 0.106939325995318, 0.047175336386512]
        },
        16: {
            'mu': [0.095012509837637, 0.281603550779259, 0.458016777657227,
                   0.617876244402644, 0.755404408355003, 0.865631202387832,
                   0.944575023073233, 0.989400934991650],
            'w': [0.189450610455068, 0.182603415044924, 0.169156519395003,
                  0.149595988816577, 0.124628971255534, 0.095158511682493,
                  0.062253523938648, 0.027152459411754]
        }
    }
    
    mu_pos = np.array(quadrature_data[n]['mu'])
    w_pos = np.array(quadrature_data[n]['w'])
    
    mu = np.concatenate([-mu_pos[::-1], mu_pos])
    w = np.concatenate([w_pos[::-1], w_pos])
    
    return mu, w


def gauss_chebyshev_quadrature(n):
    if n % 2 != 0:
        raise ValueError("n must be even for symmetric quadrature")
    
    k = np.arange(1, n + 1)
    mu = np.cos((2 * k - 1) * np.pi / (2 * n))
    w = np.pi / n * np.ones(n)
    
    idx = np.argsort(mu)
    mu = mu[idx]
    w = w[idx]
    
    return mu, w


def double_gauss_quadrature(n):
    if n % 4 != 0:
        raise ValueError("n must be divisible by 4 for double Gauss quadrature")
    
    n_half = n // 2
    mu_half, w_half = np.polynomial.legendre.leggauss(n_half)
    
    mu_pos = 0.5 * (mu_half + 1)
    mu_neg = 0.5 * (mu_half - 1)
    w_pos = 0.5 * w_half
    w_neg = 0.5 * w_half
    
    mu = np.concatenate([mu_neg, mu_pos])
    w = np.concatenate([w_neg, w_pos])
    
    idx = np.argsort(mu)
    mu = mu[idx]
    w = w[idx]
    
    return mu, w


def get_quadrature(n, quadrature_type='legendre'):
    if quadrature_type == 'legendre':
        mu, w = gauss_legendre_quadrature(n)
    elif quadrature_type == 'level_symmetric':
        mu, w = level_symmetric_quadrature(n)
    elif quadrature_type == 'chebyshev':
        mu, w = gauss_chebyshev_quadrature(n)
    elif quadrature_type == 'double_gauss':
        mu, w = double_gauss_quadrature(n)
    else:
        raise ValueError(f"Unknown quadrature type: {quadrature_type}")
    
    if not np.all(w > 0):
        raise ValueError(f"Quadrature weights must be positive! Got min(w) = {np.min(w)}")
    
    return {'mu': mu, 'w': w}


def verify_quadrature(mu, w, verbose=True):
    results = {
        'n_directions': len(mu),
        'sum_weights': np.sum(w),
        'sum_mu_w': np.sum(mu * w),
        'sum_mu2_w': np.sum(mu**2 * w),
        'sum_mu3_w': np.sum(mu**3 * w),
        'sum_mu4_w': np.sum(mu**4 * w),
        'min_weight': np.min(w),
        'max_weight': np.max(w)
    }
    
    if verbose:
        print(f"Number of directions: {results['n_directions']}")
        print(f"Sum of weights: {results['sum_weights']:.10f} (should be 2.0)")
        print(f"Sum(mu * w): {results['sum_mu_w']:.10e} (should be 0)")
        print(f"Sum(mu^2 * w): {results['sum_mu2_w']:.10f} (should be 2/3 ≈ 0.6666666667)")
        print(f"Sum(mu^3 * w): {results['sum_mu3_w']:.10e} (should be 0)")
        print(f"Sum(mu^4 * w): {results['sum_mu4_w']:.10f} (should be 2/5 = 0.4)")
        print(f"Min weight: {results['min_weight']:.10f} (should be > 0)")
        print(f"Max weight: {results['max_weight']:.10f}")
    
    return results


def list_available_quadratures():
    quadratures = [
        ('legendre', 'Gauss-Legendre', 'Any even N'),
        ('level_symmetric', 'Level Symmetric (SN standard)', 'N = 2, 4, 6, 8, 12, 16'),
        ('chebyshev', 'Gauss-Chebyshev', 'Any even N'),
        ('double_gauss', 'Double Gauss', 'N divisible by 4')
    ]
    
    print("Available quadrature types:")
    print("-" * 60)
    for qtype, name, notes in quadratures:
        print(f"  {qtype:15s} - {name:25s} ({notes})")
    print("-" * 60)
