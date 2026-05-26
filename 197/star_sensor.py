import numpy as np
from typing import List, Tuple, Optional


class Star:
    def __init__(self, vector: np.ndarray, star_id: Optional[int] = None):
        self.vector = vector / np.linalg.norm(vector)
        self.id = star_id

    def angle_with(self, other: 'Star') -> float:
        dot = np.clip(np.dot(self.vector, other.vector), -1.0, 1.0)
        return np.arccos(dot)


class Triangle:
    def __init__(self, stars: List[Star]):
        assert len(stars) == 3
        self.stars = stars
        sides_with_indices = [
            (stars[0].angle_with(stars[1]), 0, 1),
            (stars[1].angle_with(stars[2]), 1, 2),
            (stars[2].angle_with(stars[0]), 2, 0)
        ]
        sides_with_indices.sort(key=lambda x: x[0])
        self.sides = [s[0] for s in sides_with_indices]
        self.side_indices = [(s[1], s[2]) for s in sides_with_indices]

    def match(self, other: 'Triangle', tolerance: float) -> bool:
        return all(abs(a - b) < tolerance for a, b in zip(self.sides, other.sides))

    def get_correspondence(self, other: 'Triangle') -> List[Tuple[int, int]]:
        obs_vertices = self._get_vertex_order()
        cat_vertices = other._get_vertex_order()
        return list(zip(obs_vertices, cat_vertices))

    def _get_vertex_order(self) -> List[int]:
        (s0_i1, s0_i2), (s1_i1, s1_i2), (s2_i1, s2_i2) = self.side_indices
        common_in_s0_s1 = self._get_common(s0_i1, s0_i2, s1_i1, s1_i2)
        common_in_s1_s2 = self._get_common(s1_i1, s1_i2, s2_i1, s2_i2)
        common_in_s0_s2 = self._get_common(s0_i1, s0_i2, s2_i1, s2_i2)
        return [common_in_s0_s2, common_in_s0_s1, common_in_s1_s2]

    def _get_common(self, a1: int, a2: int, b1: int, b2: int) -> int:
        s = {a1, a2} & {b1, b2}
        return s.pop()

    def _get_unique(self, a1: int, a2: int, common: int) -> int:
        return a1 if a2 == common else a2


def generate_catalog_stars() -> List[Star]:
    catalog_vectors = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([-1.0, 1.0, 0.5]),
    ]
    return [Star(v, i) for i, v in enumerate(catalog_vectors)]


def generate_observed_stars(rotation_matrix: np.ndarray, 
                            catalog_stars: List[Star],
                            noise_level: float = 0.01) -> List[Star]:
    observed = []
    for star in catalog_stars:
        rotated_vector = rotation_matrix @ star.vector
        noise = np.random.normal(0, noise_level, 3)
        noisy_vector = rotated_vector + noise
        observed.append(Star(noisy_vector, star.id))
    return observed


def build_catalog_triangles(catalog_stars: List[Star], 
                            min_angle: float = 0.1,
                            max_angle: float = np.pi) -> List[Triangle]:
    triangles = []
    n = len(catalog_stars)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                tri = Triangle([catalog_stars[i], catalog_stars[j], catalog_stars[k]])
                if min_angle <= tri.sides[0] and tri.sides[2] <= max_angle:
                    triangles.append(tri)
    return triangles


def find_matching_triangles(observed_triangle: Triangle,
                            catalog_triangles: List[Triangle],
                            tolerance: float = 0.01) -> List[Tuple[Triangle, List[int]]]:
    matches = []
    for cat_tri in catalog_triangles:
        if observed_triangle.match(cat_tri, tolerance):
            matches.append((cat_tri, [star.id for star in cat_tri.stars]))
    return matches


def orthogonal_projection_svd(A: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def solve_wahba_problem_quest(observed_vectors: List[np.ndarray],
                               catalog_vectors: List[np.ndarray],
                               weights: Optional[List[float]] = None) -> np.ndarray:
    n = len(observed_vectors)
    if weights is None:
        weights = [1.0] * n
    
    B = np.zeros((3, 3))
    for w, v_obs, v_cat in zip(weights, observed_vectors, catalog_vectors):
        B += w * np.outer(v_obs, v_cat)
    
    S = B + B.T
    sigma = np.trace(B)
    Z = np.array([
        B[1, 2] - B[2, 1],
        B[2, 0] - B[0, 2],
        B[0, 1] - B[1, 0]
    ])
    
    K = np.zeros((4, 4))
    K[0, 0] = sigma
    K[0, 1:4] = Z
    K[1:4, 0] = Z
    K[1:4, 1:4] = S - sigma * np.eye(3)
    
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    max_idx = np.argmax(eigenvalues)
    q = eigenvectors[:, max_idx]
    
    q4 = q[0]
    q_vec = q[1:4]
    
    if q4 < 0:
        q4 = -q4
        q_vec = -q_vec
    
    q1, q2, q3 = q_vec
    
    R = np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q4*q3), 2*(q1*q3 + q4*q2)],
        [2*(q1*q2 + q4*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q4*q1)],
        [2*(q1*q3 - q4*q2), 2*(q2*q3 + q4*q1), 1 - 2*(q1**2 + q2**2)]
    ])
    
    return R


def solve_wahba_problem(observed_vectors: List[np.ndarray],
                        catalog_vectors: List[np.ndarray],
                        weights: Optional[List[float]] = None) -> np.ndarray:
    R_raw = solve_wahba_problem_quest(observed_vectors, catalog_vectors, weights)
    R_ortho = orthogonal_projection_svd(R_raw)
    return R_ortho


def star_identification_triangle(observed_stars: List[Star],
                                  catalog_stars: List[Star],
                                  tolerance: float = 0.02) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    catalog_triangles = build_catalog_triangles(catalog_stars)
    
    best_match = None
    best_rms = float('inf')
    best_matches = []
    best_match_count = 0
    
    obs_indices_list = []
    n_obs = len(observed_stars)
    for i in range(n_obs):
        for j in range(i + 1, n_obs):
            for k in range(j + 1, n_obs):
                obs_indices_list.append((i, j, k))
    
    for obs_indices in obs_indices_list:
        i, j, k = obs_indices
        obs_tri = Triangle([observed_stars[i], observed_stars[j], observed_stars[k]])
        matches = find_matching_triangles(obs_tri, catalog_triangles, tolerance)
        
        for cat_tri, cat_ids in matches:
            correspondence = obs_tri.get_correspondence(cat_tri)
            
            obs_local_to_global = [i, j, k]
            obs_vectors = []
            cat_vectors = []
            for obs_local_idx, cat_local_idx in correspondence:
                obs_global_idx = obs_local_to_global[obs_local_idx]
                obs_vectors.append(observed_stars[obs_global_idx].vector)
                cat_vectors.append(cat_tri.stars[cat_local_idx].vector)
            
            R = solve_wahba_problem(obs_vectors, cat_vectors)
            
            rms = 0
            star_matches = []
            for obs_idx, obs_star in enumerate(observed_stars):
                rotated = R @ obs_star.vector
                best_cat_id = -1
                min_angle = float('inf')
                for cat_star in catalog_stars:
                    angle = np.arccos(np.clip(np.dot(rotated, cat_star.vector), -1, 1))
                    if angle < min_angle:
                        min_angle = angle
                        best_cat_id = cat_star.id
                if min_angle < tolerance * 3:
                    star_matches.append((obs_idx, best_cat_id))
                    rms += min_angle ** 2
            
            if len(star_matches) >= 3:
                rms = np.sqrt(rms / len(star_matches))
                if len(star_matches) > best_match_count or (len(star_matches) == best_match_count and rms < best_rms):
                    best_match_count = len(star_matches)
                    best_rms = rms
                    best_match = R
                    best_matches = star_matches
    
    if best_match is None:
        raise ValueError("No valid star identification found")
    
    return best_match, best_matches


def euler_angles_from_rotation_matrix(R: np.ndarray) -> Tuple[float, float, float]:
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    
    return np.degrees(x), np.degrees(y), np.degrees(z)
