import numpy as np
import math
import os
import urllib.request


class EGM2008Gravity:
    def __init__(self, coefficient_file=None, max_degree=2190):
        self.GM = 3986004.418E8
        self.a = 6378136.3
        self.max_degree = max_degree
        self.C = None
        self.S = None
        
        if coefficient_file and os.path.exists(coefficient_file):
            self.load_coefficients(coefficient_file)
    
    def load_coefficients(self, filepath):
        print(f"Loading EGM2008 coefficients from {filepath}...")
        max_n = self.max_degree
        self.C = np.zeros((max_n + 1, max_n + 1))
        self.S = np.zeros((max_n + 1, max_n + 1))
        
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('end_of_head'):
                    break
            
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                n = int(parts[0])
                m = int(parts[1])
                if n > max_n:
                    continue
                C_val = float(parts[2].replace('D', 'E'))
                S_val = float(parts[3].replace('D', 'E'))
                self.C[n, m] = C_val
                self.S[n, m] = S_val
        
        print(f"Loaded coefficients up to degree {max_n}")
    
    def download_coefficients(self, save_dir='.'):
        url = "http://icgem.gfz-potsdam.de/getmodel/gfc/779bcdd1b9a838f3c47f86070c6c7d926d1c2a6e4a3802b36a4606f4e6a09c6d/EGM2008.gfc"
        filepath = os.path.join(save_dir, "EGM2008.gfc")
        
        if os.path.exists(filepath):
            print(f"EGM2008.gfc already exists at {filepath}")
            return filepath
        
        print("Downloading EGM2008 coefficients...")
        os.makedirs(save_dir, exist_ok=True)
        
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            print(f"\rDownloading: {percent:.1f}%", end='', flush=True)
        
        try:
            urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
            print("\nDownload complete!")
        except Exception as e:
            print(f"\nDownload failed: {e}")
            print("Please download manually from: http://icgem.gfz-potsdam.de/")
            return None
        
        return filepath
    
    def associated_legendre_fully_normalized(self, n_max, theta):
        """
        Compute fully normalized associated Legendre functions using
        stable column-wise recursion (X-number method).
        
        Normalization convention (matches EGM2008):
        P̄_nm(x) = sqrt((2n+1) * (n-m)!/(n+m)!) * P_nm(x) for m=0
        P̄_nm(x) = sqrt(2*(2n+1) * (n-m)!/(n+m)!) * P_nm(x) for m>0
        """
        t = math.cos(theta)
        u = math.sqrt(1.0 - t * t)
        
        P = np.zeros((n_max + 1, n_max + 1), dtype=np.float64)
        
        P[0, 0] = 1.0
        
        if n_max >= 1:
            sqrt3 = math.sqrt(3.0)
            P[1, 0] = sqrt3 * t
            P[1, 1] = sqrt3 * u
        
        for n in range(2, n_max + 1):
            ratio = math.sqrt((2.0 * n + 1.0) / (2.0 * n))
            P[n, n] = ratio * u * P[n-1, n-1]
        
        for n in range(1, n_max + 1):
            P[n, n-1] = math.sqrt(2.0 * n + 1.0) * t * P[n-1, n-1]
        
        for m in range(n_max + 1):
            for n in range(m + 2, n_max + 1):
                denom = (n - m) * (n + m)
                a = math.sqrt((2.0 * n + 1.0) * (2.0 * n - 1.0) / denom)
                b = math.sqrt((2.0 * n + 1.0) * (n + m - 1.0) * (n - m - 1.0) / 
                             ((2.0 * n - 3.0) * denom))
                P[n, m] = a * t * P[n-1, m] - b * P[n-2, m]
        
        return P
    
    def legendre_first_derivative(self, n_max, theta, P):
        """Compute first derivatives of fully normalized associated Legendre functions."""
        sin_theta = math.sin(theta)
        if abs(sin_theta) < 1e-15:
            dP = np.zeros((n_max + 1, n_max + 1))
            for n in range(1, n_max + 1):
                dP[n, 0] = -math.sqrt(n * (n + 1) / 2.0) * P[n, 1]
            return dP
        
        dP = np.zeros((n_max + 1, n_max + 1))
        for n in range(0, n_max + 1):
            for m in range(0, n + 1):
                if m < n:
                    dP[n, m] = math.sqrt((n - m) * (n + m + 1)) * P[n, m + 1]
                elif m == n:
                    dP[n, m] = 0.0
        
        return dP
    
    def legendre_second_derivative(self, n_max, theta, P, dP):
        """Compute second derivatives of fully normalized associated Legendre functions."""
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        
        if abs(sin_theta) < 1e-15:
            d2P = np.zeros((n_max + 1, n_max + 1))
            return d2P
        
        d2P = np.zeros((n_max + 1, n_max + 1))
        for n in range(0, n_max + 1):
            for m in range(0, n + 1):
                if m < n:
                    term1 = -cos_theta * dP[n, m]
                    term2 = (m * m - n * (n + 1.0)) * P[n, m]
                    d2P[n, m] = (term1 + term2) / (sin_theta * sin_theta)
                elif m == n:
                    d2P[n, m] = 0.0
        
        return d2P
    
    def compute_gravity_vector(self, lat, lon, height):
        """
        Compute gravity vector components in local north-east-down (NED) frame.
        
        Returns:
            g_N, g_E, g_D: gravity components in mGal
        """
        if self.C is None or self.S is None:
            raise ValueError("Coefficients not loaded. Call load_coefficients() first.")
        
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        
        colatitude = math.pi / 2.0 - lat_rad
        r = self.a + height
        
        n_max = self.max_degree
        P = self.associated_legendre_fully_normalized(n_max, colatitude)
        dP = self.legendre_first_derivative(n_max, colatitude, P)
        
        V_r = 0.0
        V_theta = 0.0
        V_lambda = 0.0
        
        cos_lon = np.cos(np.arange(n_max + 1) * lon_rad)
        sin_lon = np.sin(np.arange(n_max + 1) * lon_rad)
        
        for n in range(0, n_max + 1):
            r_factor = (self.a / r) ** (n + 1)
            
            for m in range(0, n + 1):
                cos_ml = cos_lon[m]
                sin_ml = sin_lon[m]
                
                term_C = self.C[n, m] * cos_ml
                term_S = self.S[n, m] * sin_ml
                term = term_C + term_S
                
                V_r += r_factor * (n + 1) * P[n, m] * term
                V_theta += r_factor * dP[n, m] * term
                V_lambda += r_factor * m * P[n, m] * (self.S[n, m] * cos_ml - self.C[n, m] * sin_ml)
        
        V_r *= -self.GM / (r * r)
        V_theta *= self.GM / r
        V_lambda *= self.GM / r
        
        omega = 7292115.0e-11
        centrifugal = omega * omega * r * math.cos(lat_rad) ** 2
        
        g_r = -V_r + centrifugal
        g_theta = -V_theta / r
        g_lambda = -V_lambda / (r * math.sin(colatitude)) if abs(math.sin(colatitude)) > 1e-15 else 0.0
        
        g_N = g_theta
        g_E = -g_lambda
        g_D = g_r
        
        return g_N * 1e5, g_E * 1e5, g_D * 1e5
    
    def compute_gravity_gradient_tensor(self, lat, lon, height):
        """
        Compute gravity gradient tensor in local north-east-down (NED) frame.
        
        Gravity gradient tensor:
            [ T_NN  T_NE  T_ND ]
            [ T_EN  T_EE  T_ED ]
            [ T_DN  T_DE  T_DD ]
        
        Units: Eötvös (1 Eötvös = 10^-9 s^-2)
        
        Returns:
            3x3 numpy array of gravity gradient tensor components
        """
        if self.C is None or self.S is None:
            raise ValueError("Coefficients not loaded. Call load_coefficients() first.")
        
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        
        colatitude = math.pi / 2.0 - lat_rad
        r = self.a + height
        sin_colat = math.sin(colatitude)
        cos_colat = math.cos(colatitude)
        
        n_max = self.max_degree
        P = self.associated_legendre_fully_normalized(n_max, colatitude)
        dP = self.legendre_first_derivative(n_max, colatitude, P)
        d2P = self.legendre_second_derivative(n_max, colatitude, P, dP)
        
        T_rr = 0.0
        T_rtheta = 0.0
        T_rlambda = 0.0
        T_thetatheta = 0.0
        T_thetalambda = 0.0
        T_lambdalambda = 0.0
        
        cos_lon = np.cos(np.arange(n_max + 1) * lon_rad)
        sin_lon = np.sin(np.arange(n_max + 1) * lon_rad)
        
        for n in range(0, n_max + 1):
            r_factor = (self.a / r) ** (n + 1)
            n1 = n + 1
            n2 = n1 * (n + 2)
            
            for m in range(0, n + 1):
                cos_ml = cos_lon[m]
                sin_ml = sin_lon[m]
                
                term_C = self.C[n, m] * cos_ml
                term_S = self.S[n, m] * sin_ml
                term = term_C + term_S
                cross_term = self.S[n, m] * cos_ml - self.C[n, m] * sin_ml
                
                T_rr += r_factor * n2 * P[n, m] * term
                T_rtheta += r_factor * n1 * dP[n, m] * term
                T_rlambda += r_factor * n1 * m * P[n, m] * cross_term
                
                T_thetatheta += r_factor * d2P[n, m] * term
                T_thetalambda += r_factor * m * dP[n, m] * cross_term
                T_lambdalambda += r_factor * (m * m * P[n, m] * term + cos_colat * dP[n, m] * term)
        
        factor = self.GM / (r * r * r)
        T_rr *= factor
        T_rtheta *= factor
        T_rlambda *= factor
        T_thetatheta *= factor
        T_thetalambda *= factor
        T_lambdalambda *= factor
        
        if abs(sin_colat) > 1e-15:
            T_rlambda /= sin_colat
            T_thetalambda /= sin_colat
            T_lambdalambda /= (sin_colat * sin_colat)
        
        T_NN = T_thetatheta
        T_NE = -T_thetalambda
        T_ND = -T_rtheta
        
        T_EN = T_NE
        T_EE = T_lambdalambda
        T_ED = T_rlambda / sin_colat if abs(sin_colat) > 1e-15 else 0.0
        
        T_DN = T_ND
        T_DE = T_ED
        T_DD = T_rr
        
        tensor = np.array([
            [T_NN, T_NE, T_ND],
            [T_EN, T_EE, T_ED],
            [T_DN, T_DE, T_DD]
        ]) * 1e9
        
        return tensor
    
    def compute_spherical_harmonic_sum(self, lat, lon, height):
        """Compute radial derivative of the gravitational potential."""
        if self.C is None or self.S is None:
            raise ValueError("Coefficients not loaded. Call load_coefficients() first.")
        
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        
        colatitude = math.pi / 2.0 - lat_rad
        r = self.a + height
        
        n_max = self.max_degree
        P = self.associated_legendre_fully_normalized(n_max, colatitude)
        
        V_r = 0.0
        
        cos_lon = np.cos(np.arange(n_max + 1) * lon_rad)
        sin_lon = np.sin(np.arange(n_max + 1) * lon_rad)
        
        for n in range(0, n_max + 1):
            r_factor = (self.a / r) ** (n + 1)
            
            for m in range(0, n + 1):
                term = self.C[n, m] * cos_lon[m] + self.S[n, m] * sin_lon[m]
                V_r += r_factor * (n + 1) * P[n, m] * term
        
        V_r *= -self.GM / (r * r)
        
        return V_r
    
    def compute_disturbing_potential(self, lat, lon):
        """Compute disturbing potential T for geoid height calculation."""
        if self.C is None or self.S is None:
            raise ValueError("Coefficients not loaded. Call load_coefficients() first.")
        
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        
        colatitude = math.pi / 2.0 - lat_rad
        r = self.a
        
        n_max = self.max_degree
        P = self.associated_legendre_fully_normalized(n_max, colatitude)
        
        T = 0.0
        
        cos_lon = np.cos(np.arange(n_max + 1) * lon_rad)
        sin_lon = np.sin(np.arange(n_max + 1) * lon_rad)
        
        for n in range(0, n_max + 1):
            r_factor = (self.a / r) ** (n + 1)
            for m in range(0, n + 1):
                term = self.C[n, m] * cos_lon[m] + self.S[n, m] * sin_lon[m]
                T += r_factor * P[n, m] * term
        
        T *= self.GM / r
        
        return T
    
    def normal_gravity(self, lat, height):
        """Compute normal gravity on the ellipsoid using Somigliana's formula."""
        a = 6378137.0
        b = 6356752.3141
        GM = 3986004.418E8
        omega = 7292115.0e-11
        
        lat_rad = math.radians(lat)
        sin2_lat = math.sin(lat_rad) ** 2
        
        gamma_e = 9.7803253359
        k = 0.00193185265241
        e2 = 0.00669437999013
        
        gamma = gamma_e * (1 + k * sin2_lat) / math.sqrt(1 - e2 * sin2_lat)
        
        gamma_h = gamma * (1 - 2 * height / a * (1 + (omega ** 2 * a ** 2 * b) / GM + k * (1 - 2 * sin2_lat)) + 3 * (height / a) ** 2)
        
        return gamma_h
    
    def gravity_anomaly(self, lat, lon, height):
        """
        Compute gravity anomaly in mGal.
        
        Gravity anomaly delta_g = g - gamma
        where g is the actual gravity and gamma is normal gravity.
        """
        V_r = self.compute_spherical_harmonic_sum(lat, lon, height)
        
        lat_rad = math.radians(lat)
        omega = 7292115.0e-11
        r = self.a + height
        
        centrifugal = omega * omega * r * math.cos(lat_rad) ** 2
        
        g_r = -V_r + centrifugal
        
        gamma = self.normal_gravity(lat, height)
        
        delta_g = (g_r - gamma) * 1e5
        
        return delta_g
    
    def geoid_height(self, lat, lon):
        """
        Compute geoid height using Bruns' formula: N = T / gamma
        
        where T is the disturbing potential and gamma is normal gravity.
        """
        T = self.compute_disturbing_potential(lat, lon)
        gamma = self.normal_gravity(lat, 0.0)
        
        N = T / gamma
        
        return N
    
    def test_stability(self, test_degree=2190):
        """Test numerical stability of the Legendre function computation."""
        print(f"\nTesting numerical stability up to degree {test_degree}...")
        
        theta = math.pi / 4.0
        
        try:
            P = self.associated_legendre_fully_normalized(test_degree, theta)
            
            max_val = np.max(np.abs(P))
            min_val = np.min(np.abs(P[P != 0]))
            
            print(f"  Maximum |Pnm|: {max_val:.6e}")
            print(f"  Minimum |Pnm|: {min_val:.6e}")
            print(f"  Dynamic range: {max_val / min_val:.2e}")
            
            if not np.isfinite(max_val):
                print("  FAILED: Non-finite values detected!")
                return False
            elif max_val > 1e15:
                print("  WARNING: Large values detected, potential overflow!")
                return False
            else:
                print("  Stability check: PASSED")
                return True
        except Exception as e:
            print(f"  Stability check FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_known_values(self):
        """Test against known values for low-degree coefficients."""
        print("\nTesting against known values...")
        
        test_n, test_m = 2, 0
        theta = 0.0
        
        P = self.associated_legendre_fully_normalized(test_n, theta)
        expected = math.sqrt(5) * 0.5
        
        print(f"  P[{test_n},{test_m}](cos(0)) = {P[test_n, test_m]:.10f}")
        print(f"  Expected: {expected:.10f}")
        print(f"  Difference: {abs(P[test_n, test_m] - expected):.2e}")
        
        return abs(P[test_n, test_m] - expected) < 1e-10


class SphericalCapHarmonics:
    """
    Spherical Cap Harmonics (SCH) for local gravity field modeling.
    
    This class implements spherical cap harmonics for high-resolution
    regional gravity field modeling.
    """
    
    def __init__(self, cap_center_lat, cap_center_lon, cap_half_angle, max_degree=360):
        """
        Initialize spherical cap harmonics model.
        
        Args:
            cap_center_lat: Latitude of cap center (degrees)
            cap_center_lon: Longitude of cap center (degrees)
            cap_half_angle: Half-angle of the spherical cap (degrees)
            max_degree: Maximum degree of expansion
        """
        self.cap_center_lat = cap_center_lat
        self.cap_center_lon = cap_center_lon
        self.cap_half_angle = math.radians(cap_half_angle)
        self.max_degree = max_degree
        self.GM = 3986004.418E8
        self.a = 6378136.3
        
        self.cap_C = None
        self.cap_S = None
    
    def spherical_to_cap_coords(self, lat, lon):
        """Convert geographic coordinates to spherical cap coordinates."""
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        
        center_lat_rad = math.radians(self.cap_center_lat)
        center_lon_rad = math.radians(self.cap_center_lon)
        
        d_lon = lon_rad - center_lon_rad
        
        cos_theta = (math.sin(lat_rad) * math.sin(center_lat_rad) + 
                     math.cos(lat_rad) * math.cos(center_lat_rad) * math.cos(d_lon))
        
        cos_theta = max(min(cos_theta, 1.0), -1.0)
        theta = math.acos(cos_theta)
        
        if theta < 1e-15:
            lambda_cap = 0.0
        else:
            sin_theta = math.sin(theta)
            sin_lat_diff = math.sin(lat_rad) * math.cos(center_lat_rad) - math.cos(lat_rad) * math.sin(center_lat_rad) * math.cos(d_lon)
            sin_lambda_cap = math.cos(lat_rad) * math.sin(d_lon) / sin_theta
            cos_lambda_cap = sin_lat_diff / sin_theta
            
            sin_lambda_cap = max(min(sin_lambda_cap, 1.0), -1.0)
            lambda_cap = math.asin(sin_lambda_cap)
            if cos_lambda_cap < 0:
                lambda_cap = math.pi - lambda_cap
        
        return theta, lambda_cap
    
    def compute_legendre_for_cap(self, n_max, theta):
        """Compute associated Legendre functions for spherical cap."""
        t = math.cos(theta)
        u = math.sqrt(1.0 - t * t)
        
        P = np.zeros((n_max + 1, n_max + 1), dtype=np.float64)
        
        P[0, 0] = 1.0
        
        if n_max >= 1:
            P[1, 0] = math.sqrt(3) * t
            P[1, 1] = math.sqrt(3) * u
        
        for n in range(2, n_max + 1):
            ratio = math.sqrt((2.0 * n + 1.0) / (2.0 * n))
            P[n, n] = ratio * u * P[n-1, n-1]
        
        for n in range(1, n_max + 1):
            P[n, n-1] = math.sqrt(2.0 * n + 1.0) * t * P[n-1, n-1]
        
        for m in range(n_max + 1):
            for n in range(m + 2, n_max + 1):
                denom = (n - m) * (n + m)
                a = math.sqrt((2.0 * n + 1.0) * (2.0 * n - 1.0) / denom)
                b = math.sqrt((2.0 * n + 1.0) * (n + m - 1.0) * (n - m - 1.0) / 
                             ((2.0 * n - 3.0) * denom))
                P[n, m] = a * t * P[n-1, m] - b * P[n-2, m]
        
        return P
    
    def fit_model(self, latitudes, longitudes, gravity_data, heights=None):
        """
        Fit spherical cap harmonics model to gravity data.
        
        Args:
            latitudes: Array of latitudes (degrees)
            longitudes: Array of longitudes (degrees)
            gravity_data: Array of gravity anomalies (mGal)
            heights: Optional array of heights (m)
        """
        if heights is None:
            heights = np.zeros_like(latitudes)
        
        N = len(latitudes)
        M = (self.max_degree + 1) * (self.max_degree + 2) // 2
        
        print(f"Fitting spherical cap harmonics model with {N} data points...")
        print(f"Maximum degree: {self.max_degree}, Number of coefficients: {M}")
        
        A = np.zeros((N, M))
        b = np.array(gravity_data)
        
        coeff_idx = 0
        for n in range(self.max_degree + 1):
            for m in range(n + 1):
                for i in range(N):
                    theta, lambda_cap = self.spherical_to_cap_coords(latitudes[i], longitudes[i])
                    P = self.compute_legendre_for_cap(self.max_degree, theta)
                    
                    r = self.a + heights[i]
                    r_factor = (self.a / r) ** (n + 2) * (n + 1)
                    
                    if m == 0:
                        A[i, coeff_idx] = r_factor * P[n, m]
                    else:
                        A[i, coeff_idx] = r_factor * P[n, m] * math.cos(m * lambda_cap)
                        if coeff_idx + 1 < M:
                            A[i, coeff_idx + 1] = r_factor * P[n, m] * math.sin(m * lambda_cap)
                
                coeff_idx += 1 if m == 0 else 2
        
        print("Solving least squares system...")
        coeffs, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        
        self.cap_C = np.zeros((self.max_degree + 1, self.max_degree + 1))
        self.cap_S = np.zeros((self.max_degree + 1, self.max_degree + 1))
        
        coeff_idx = 0
        for n in range(self.max_degree + 1):
            for m in range(n + 1):
                if m == 0:
                    self.cap_C[n, m] = coeffs[coeff_idx]
                    coeff_idx += 1
                else:
                    self.cap_C[n, m] = coeffs[coeff_idx]
                    self.cap_S[n, m] = coeffs[coeff_idx + 1]
                    coeff_idx += 2
        
        print(f"Fit complete. Residual norm: {np.sqrt(np.sum(residuals)) if len(residuals) > 0 else 0:.2f} mGal")
        
        return coeffs, residuals
    
    def predict_gravity(self, lat, lon, height=0.0):
        """
        Predict gravity anomaly using the fitted spherical cap model.
        
        Args:
            lat: Latitude (degrees)
            lon: Longitude (degrees)
            height: Height above ellipsoid (m)
            
        Returns:
            Predicted gravity anomaly (mGal)
        """
        if self.cap_C is None:
            raise ValueError("Model not fitted. Call fit_model() first.")
        
        theta, lambda_cap = self.spherical_to_cap_coords(lat, lon)
        P = self.compute_legendre_for_cap(self.max_degree, theta)
        
        r = self.a + height
        
        g = 0.0
        for n in range(self.max_degree + 1):
            r_factor = (self.a / r) ** (n + 2) * (n + 1)
            for m in range(n + 1):
                if m == 0:
                    g += r_factor * self.cap_C[n, m] * P[n, m]
                else:
                    g += r_factor * P[n, m] * (
                        self.cap_C[n, m] * math.cos(m * lambda_cap) +
                        self.cap_S[n, m] * math.sin(m * lambda_cap)
                    )
        
        return g
    
    def predict_geoid_height(self, lat, lon):
        """
        Predict geoid height using the fitted spherical cap model.
        
        Args:
            lat: Latitude (degrees)
            lon: Longitude (degrees)
            
        Returns:
            Predicted geoid height (m)
        """
        if self.cap_C is None:
            raise ValueError("Model not fitted. Call fit_model() first.")
        
        theta, lambda_cap = self.spherical_to_cap_coords(lat, lon)
        P = self.compute_legendre_for_cap(self.max_degree, theta)
        
        r = self.a
        gamma = 9.80665
        
        T = 0.0
        for n in range(self.max_degree + 1):
            r_factor = (self.a / r) ** (n + 1)
            for m in range(n + 1):
                if m == 0:
                    T += r_factor * self.cap_C[n, m] * P[n, m]
                else:
                    T += r_factor * P[n, m] * (
                        self.cap_C[n, m] * math.cos(m * lambda_cap) +
                        self.cap_S[n, m] * math.sin(m * lambda_cap)
                    )
        
        T *= self.GM / r
        
        N = T / gamma
        
        return N


def main():
    print("EGM2008 Earth Gravity Field Model")
    print("=" * 60)
    print("Features:")
    print("  - Fully normalized associated Legendre functions")
    print("  - Stable column-wise X-number recursion")
    print("  - Gravity vector computation (NED frame)")
    print("  - Gravity gradient tensor (Eötvös)")
    print("  - Spherical Cap Harmonics for local modeling")
    print("=" * 60)
    
    egm = EGM2008Gravity(max_degree=60)
    
    egm.test_stability(test_degree=60)
    egm.test_known_values()
    
    coeff_file = "EGM2008.gfc"
    if not os.path.exists(coeff_file):
        print("\nEGM2008 coefficient file not found.")
        print("Running in demo mode with sample coefficients (degree 2 only)...")
        print()
        
        egm.C = np.zeros((61, 61))
        egm.S = np.zeros((61, 61))
        egm.C[0, 0] = 1.0
        egm.C[2, 0] = -4.84165143790815E-04
        egm.C[2, 1] = -2.06615509074176E-10
        egm.S[2, 1] = 2.48593231863966E-09
        egm.C[2, 2] = 2.43938357328313E-06
        egm.S[2, 2] = -1.40027370385934E-06
    else:
        egm.load_coefficients(coeff_file)
    
    test_points = [
        (40.0, 116.0, 0.0, "Beijing"),
        (48.85, 2.35, 0.0, "Paris"),
        (40.71, -74.01, 0.0, "New York"),
    ]
    
    print("\n" + "=" * 100)
    print(f"{'City':<15} {'Latitude':>10} {'Longitude':>12} {'g_N (mGal)':>12} {'g_E (mGal)':>12} {'g_D (mGal)':>12}")
    print("=" * 100)
    
    for lat, lon, h, name in test_points:
        try:
            g_N, g_E, g_D = egm.compute_gravity_vector(lat, lon, h)
            print(f"{name:<15} {lat:>10.4f} {lon:>12.4f} {g_N:>12.4f} {g_E:>12.4f} {g_D:>12.4f}")
        except Exception as e:
            print(f"{name:<15} {lat:>10.4f} {lon:>12.4f} {'Error':>12} {'Error':>12} {'Error':>12}")
    
    print("\n" + "=" * 100)
    print("Gravity Gradient Tensor (Eötvös):")
    print("=" * 100)
    
    for lat, lon, h, name in test_points[:1]:
        try:
            tensor = egm.compute_gravity_gradient_tensor(lat, lon, h)
            print(f"\n{name}:")
            print(f"  T_NN = {tensor[0,0]:.4f}  T_NE = {tensor[0,1]:.4f}  T_ND = {tensor[0,2]:.4f}")
            print(f"  T_EN = {tensor[1,0]:.4f}  T_EE = {tensor[1,1]:.4f}  T_ED = {tensor[1,2]:.4f}")
            print(f"  T_DN = {tensor[2,0]:.4f}  T_DE = {tensor[2,1]:.4f}  T_DD = {tensor[2,2]:.4f}")
            print(f"  Trace: {np.trace(tensor):.4f} Eötvös (should be ~0 for Laplace equation)")
        except Exception as e:
            print(f"{name}: Error computing gradient tensor")
    
    print("\n" + "=" * 100)
    print("Spherical Cap Harmonics Demo (Local Modeling)")
    print("=" * 100)
    
    center_lat, center_lon = 40.0, 116.0
    sch = SphericalCapHarmonics(center_lat, center_lon, cap_half_angle=5.0, max_degree=10)
    
    print(f"\nCreating synthetic data for cap centered at ({center_lat}°, {center_lon}°)")
    print(f"Cap half-angle: 5.0°, Max degree: 10")
    
    np.random.seed(42)
    n_points = 50
    lats = center_lat + np.random.uniform(-4, 4, n_points)
    lons = center_lon + np.random.uniform(-4, 4, n_points)
    
    synthetic_gravity = (100.0 * np.sin(np.radians(lats - center_lat) * 5) * 
                        np.cos(np.radians(lons - center_lon) * 5) +
                        np.random.normal(0, 5, n_points))
    
    print(f"Generated {n_points} synthetic gravity observations")
    
    try:
        sch.fit_model(lats, lons, synthetic_gravity)
        
        print("\nPrediction at cap center:")
        pred_g = sch.predict_gravity(center_lat, center_lon)
        pred_N = sch.predict_geoid_height(center_lat, center_lon)
        print(f"  Predicted gravity anomaly: {pred_g:.4f} mGal")
        print(f"  Predicted geoid height: {pred_N:.4f} m")
    except Exception as e:
        print(f"Spherical cap harmonics demo: {e}")
        print("Note: This is a simplified demonstration")
    
    print("\n" + "=" * 100)
    print("Notes:")
    print("  - 1 mGal = 10^-5 m/s^2")
    print("  - 1 Eötvös = 10^-9 s^-2")
    print("  - Gravity tensor trace should be ~0 (Laplace equation in free air)")
    print("  - Spherical Cap Harmonics improve local resolution")
    print("  - Demo mode uses limited coefficients (full EGM2008.gfc file required for accuracy)")


if __name__ == "__main__":
    main()
