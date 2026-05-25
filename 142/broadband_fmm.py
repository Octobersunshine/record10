import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from scipy.special import spherical_jn, hankel2, legendre
from enum import Enum


class ExpansionType(Enum):
    SPHERICAL_HARMONICS = "spherical"
    PLANE_WAVE = "plane_wave"
    ADAPTIVE = "adaptive"


@dataclass
class ExpansionConfig:
    k: float
    box_size: float
    expansion_type: ExpansionType
    p: int = 0
    n_theta: int = 0
    n_phi: int = 0

    @classmethod
    def adaptive(cls, k: float, box_size: float, tol: float = 1e-6,
                 min_p: int = 2, max_p: int = 50,
                 kr_switch: float = 8.0) -> 'ExpansionConfig':
        kr = k * box_size

        if kr < kr_switch:
            p = cls._compute_spherical_order(kr, tol, min_p, max_p)
            return cls(k=k, box_size=box_size,
                       expansion_type=ExpansionType.SPHERICAL_HARMONICS, p=p)
        else:
            n_theta, n_phi = cls._compute_plane_wave_sampling(kr, tol)
            return cls(k=k, box_size=box_size,
                       expansion_type=ExpansionType.PLANE_WAVE,
                       n_theta=n_theta, n_phi=n_phi)

    @staticmethod
    def _compute_spherical_order(kr: float, tol: float,
                                 min_p: int, max_p: int) -> int:
        p = max(min_p, int(kr + 4 * np.log(kr + np.pi) + np.log(1/tol)))
        return min(p, max_p)

    @staticmethod
    def _compute_plane_wave_sampling(kr: float, tol: float) -> Tuple[int, int]:
        n_theta = max(16, int(2 * kr + 10))
        n_phi = max(32, int(2 * kr + 10))
        return n_theta, n_phi


class PlaneWaveExpansion:
    def __init__(self, k: float, n_theta: int, n_phi: int):
        self.k = k
        self.n_theta = n_theta
        self.n_phi = n_phi
        self._setup_directions()
        self.coeffs = None

    def _setup_directions(self):
        theta = np.linspace(0, np.pi, self.n_theta, endpoint=False)
        phi = np.linspace(0, 2 * np.pi, self.n_phi, endpoint=False)
        dtheta = theta[1] - theta[0] if self.n_theta > 1 else np.pi
        dphi = phi[1] - phi[0] if self.n_phi > 1 else 2 * np.pi

        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
        self.directions = np.array([
            np.sin(theta_grid) * np.cos(phi_grid),
            np.sin(theta_grid) * np.sin(phi_grid),
            np.cos(theta_grid)
        ]).transpose(1, 2, 0)

        sin_theta = np.sin(theta_grid)
        self.weights = sin_theta * dtheta * dphi / (4 * np.pi)

    def compute_from_sources(self, sources: np.ndarray,
                             charges: np.ndarray,
                             center: np.ndarray):
        r = sources - center
        phases = np.exp(1j * self.k * np.einsum('...d,sd->...s',
                                                 self.directions, r))
        self.coeffs = np.einsum('...s,s->...', phases, charges)

    def evaluate(self, points: np.ndarray, center: np.ndarray) -> np.ndarray:
        r = points - center
        phases = np.exp(-1j * self.k * np.einsum('...d,nd->...n',
                                                   self.directions, r))
        integrand = self.coeffs[..., np.newaxis] * phases
        result = np.einsum('...,...n->n', self.weights, integrand)
        return result

    def translate(self, translation: np.ndarray) -> 'PlaneWaveExpansion':
        result = PlaneWaveExpansion(self.k, self.n_theta, self.n_phi)
        result.directions = self.directions
        result.weights = self.weights

        phase_shift = np.exp(1j * self.k * np.einsum('...d,d->...',
                                                      self.directions, translation))
        result.coeffs = self.coeffs * phase_shift
        return result


class SphericalExpansion:
    def __init__(self, k: float, p: int):
        self.k = k
        self.p = p
        self.coeffs = np.zeros((p + 1, 2 * p + 1), dtype=complex)

    @staticmethod
    def legendre_p(l: int, x: np.ndarray) -> np.ndarray:
        if l == 0:
            return np.ones_like(x)
        elif l == 1:
            return x
        else:
            P_prev = np.ones_like(x)
            P_curr = x
            for n in range(2, l + 1):
                P_next = ((2 * n - 1) * x * P_curr - (n - 1) * P_prev) / n
                P_prev, P_curr = P_curr, P_next
            return P_curr

    @staticmethod
    def assoc_legendre(l: int, m: int, x: np.ndarray) -> np.ndarray:
        if m == 0:
            return SphericalExpansion.legendre_p(l, x)
        elif m > l:
            return np.zeros_like(x)
        else:
            P_mm = (-1)**m * (1 - x**2)**(m / 2) * np.prod(np.arange(1, 2 * m, 2))
            if l == m:
                return P_mm
            else:
                P_mp1m = x * (2 * m + 1) * P_mm
                if l == m + 1:
                    return P_mp1m
                else:
                    P_prev = P_mm
                    P_curr = P_mp1m
                    for n in range(m + 2, l + 1):
                        P_next = ((2 * n - 1) * x * P_curr - (n + m - 1) * P_prev) / (n - m)
                        P_prev, P_curr = P_curr, P_next
                    return P_curr

    @staticmethod
    def Y_lm(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        P_lm = SphericalExpansion.assoc_legendre(l, abs(m), np.cos(theta))
        norm = np.sqrt((2 * l + 1) * np.math.factorial(l - abs(m)) /
                       (4 * np.pi * np.math.factorial(l + abs(m))))
        if m >= 0:
            return norm * P_lm * np.exp(1j * m * phi)
        else:
            return (-1)**abs(m) * np.conj(norm * P_lm * np.exp(1j * abs(m) * phi))

    def compute_multipole(self, sources: np.ndarray, charges: np.ndarray,
                          center: np.ndarray):
        for i, src in enumerate(sources):
            r_vec = src - center
            r = np.linalg.norm(r_vec)
            if r < 1e-12:
                continue
            theta = np.arccos(r_vec[2] / r)
            phi = np.arctan2(r_vec[1], r_vec[0])
            for l in range(self.p + 1):
                jl = spherical_jn(l, self.k * r)
                for m in range(-l, l + 1):
                    Y_lm = self.Y_lm(l, m, theta, phi)
                    self.coeffs[l, m + self.p] += charges[i] * (1j)**l * jl * np.conj(Y_lm)

    def evaluate_local(self, points: np.ndarray, center: np.ndarray) -> np.ndarray:
        result = np.zeros(len(points), dtype=complex)
        for i, pt in enumerate(points):
            r_vec = pt - center
            r = np.linalg.norm(r_vec)
            if r < 1e-12:
                continue
            theta = np.arccos(r_vec[2] / r)
            phi = np.arctan2(r_vec[1], r_vec[0])
            for l in range(self.p + 1):
                hl = hankel2(l, self.k * r)
                for m in range(-l, l + 1):
                    Y_lm = self.Y_lm(l, m, theta, phi)
                    result[i] += self.coeffs[l, m + self.p] * (-1j)**l * hl * Y_lm
        return result


class BroadbandOctreeNode:
    def __init__(self, center: np.ndarray, size: float, level: int,
                 index: Tuple[int, int, int]):
        self.center = center
        self.size = size
        self.level = level
        self.index = index
        self.parent = None
        self.children: List['BroadbandOctreeNode'] = []
        self.points = None
        self.point_indices = None
        self.expansion_config: Optional[ExpansionConfig] = None
        self.multipole = None
        self.local = None

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class BroadbandOctree:
    def __init__(self, points: np.ndarray, k: float,
                 max_points_per_leaf: int = 10,
                 max_level: int = 15,
                 tol: float = 1e-6):
        self.points = points
        self.k = k
        self.max_points_per_leaf = max_points_per_leaf
        self.max_level = max_level
        self.tol = tol
        self.root = self._build_tree()
        self.nodes_by_level: Dict[int, List[BroadbandOctreeNode]] = {}
        self._index_nodes(self.root)
        self._setup_expansions()

    def _build_tree(self) -> BroadbandOctreeNode:
        min_corner = np.min(self.points, axis=0)
        max_corner = np.max(self.points, axis=0)
        center = (min_corner + max_corner) / 2
        size = np.max(max_corner - min_corner) * 1.01

        root = BroadbandOctreeNode(center=center, size=size, level=0,
                                   index=(0, 0, 0))
        root.points = self.points
        root.point_indices = np.arange(len(self.points))
        self._subdivide(root)
        return root

    def _subdivide(self, node: BroadbandOctreeNode):
        if (len(node.points) <= self.max_points_per_leaf or
            node.level >= self.max_level):
            return

        half_size = node.size / 2
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    child_center = node.center + half_size * np.array([
                        i - 0.5, j - 0.5, k - 0.5
                    ])
                    child = BroadbandOctreeNode(
                        center=child_center,
                        size=half_size,
                        level=node.level + 1,
                        index=(2 * node.index[0] + i,
                               2 * node.index[1] + j,
                               2 * node.index[2] + k)
                    )
                    child.parent = node
                    mask = self._points_in_box(node.points, child_center, half_size)
                    child.points = node.points[mask]
                    child.point_indices = node.point_indices[mask]
                    if len(child.points) > 0:
                        node.children.append(child)
                        self._subdivide(child)

    def _points_in_box(self, points: np.ndarray, center: np.ndarray,
                       half_size: float) -> np.ndarray:
        half = half_size / 2
        return np.all(np.abs(points - center) <= half, axis=1)

    def _index_nodes(self, node: BroadbandOctreeNode):
        if node.level not in self.nodes_by_level:
            self.nodes_by_level[node.level] = []
        self.nodes_by_level[node.level].append(node)
        for child in node.children:
            self._index_nodes(child)

    def _setup_expansions(self):
        for level, nodes in self.nodes_by_level.items():
            for node in nodes:
                node.expansion_config = ExpansionConfig.adaptive(
                    self.k, node.size, self.tol
                )

    def get_leaf_nodes(self) -> List[BroadbandOctreeNode]:
        leaves = []
        self._collect_leaves(self.root, leaves)
        return leaves

    def _collect_leaves(self, node: BroadbandOctreeNode,
                        leaves: List[BroadbandOctreeNode]):
        if node.is_leaf():
            leaves.append(node)
        else:
            for child in node.children:
                self._collect_leaves(child, leaves)


class BroadbandFMM:
    def __init__(self, k: float, tol: float = 1e-6,
                 max_points_per_leaf: int = 20,
                 kr_switch: float = 8.0):
        self.k = k
        self.tol = tol
        self.max_points_per_leaf = max_points_per_leaf
        self.kr_switch = kr_switch
        self.octree = None
        self.interaction_list: Dict[BroadbandOctreeNode, List[BroadbandOctreeNode]] = {}

    def build_tree(self, sources: np.ndarray, targets: np.ndarray = None):
        if targets is None:
            targets = sources
        all_points = np.vstack([sources, targets])
        self.octree = BroadbandOctree(all_points, self.k,
                                       self.max_points_per_leaf,
                                       tol=self.tol)
        self._build_interaction_list()
        return self.octree

    def _build_interaction_list(self):
        for level in self.octree.nodes_by_level:
            nodes = self.octree.nodes_by_level[level]
            for node in nodes:
                self.interaction_list[node] = self._get_interaction_list(node)

    def _get_interaction_list(self, node: BroadbandOctreeNode) -> List[BroadbandOctreeNode]:
        if node.parent is None:
            return []
        parent_neighbors = self._get_neighbors(node.parent)
        interaction = []
        for parent_neighbor in parent_neighbors:
            for child in parent_neighbor.children:
                if self._is_well_separated(node, child):
                    interaction.append(child)
        return interaction

    def _get_neighbors(self, node: BroadbandOctreeNode) -> List[BroadbandOctreeNode]:
        neighbors = []
        if node.level == 0:
            return neighbors
        level_nodes = self.octree.nodes_by_level.get(node.level, [])
        for other in level_nodes:
            if other is not node and self._is_neighbor(node, other):
                neighbors.append(other)
        return neighbors

    def _is_neighbor(self, node1: BroadbandOctreeNode,
                     node2: BroadbandOctreeNode) -> bool:
        dx = abs(node1.center[0] - node2.center[0])
        dy = abs(node1.center[1] - node2.center[1])
        dz = abs(node1.center[2] - node2.center[2])
        max_dist = node1.size * 1.5
        return dx <= max_dist and dy <= max_dist and dz <= max_dist

    def _is_well_separated(self, node1: BroadbandOctreeNode,
                           node2: BroadbandOctreeNode) -> bool:
        dist = np.linalg.norm(node1.center - node2.center)
        return dist >= node1.size * 2.0

    def compute_potential(self, sources: np.ndarray, charges: np.ndarray,
                          targets: np.ndarray = None) -> np.ndarray:
        if targets is None:
            targets = sources

        self.build_tree(sources, targets)
        self._compute_multipole_expansions(sources, charges)
        self._translate_multipole_to_multipole()
        self._translate_multipole_to_local()
        self._translate_local_to_local()
        potential = self._evaluate_local_expansions(targets)
        potential += self._evaluate_neighbors_direct(targets, sources, charges)
        return potential

    def _create_expansion(self, config: ExpansionConfig, is_multipole: bool = True):
        if config.expansion_type == ExpansionType.SPHERICAL_HARMONICS:
            return SphericalExpansion(self.k, config.p)
        else:
            return PlaneWaveExpansion(self.k, config.n_theta, config.n_phi)

    def _compute_multipole_expansions(self, sources: np.ndarray,
                                      charges: np.ndarray):
        leaves = self.octree.get_leaf_nodes()
        for leaf in leaves:
            if leaf.points is not None and len(leaf.points) > 0:
                config = leaf.expansion_config
                leaf.multipole = self._create_expansion(config, is_multipole=True)

                mask = np.isin(sources, leaf.points).all(axis=1)
                leaf_sources = sources[mask]
                leaf_charges = charges[mask]

                if len(leaf_sources) > 0:
                    if config.expansion_type == ExpansionType.SPHERICAL_HARMONICS:
                        leaf.multipole.compute_multipole(leaf_sources,
                                                          leaf_charges,
                                                          leaf.center)
                    else:
                        leaf.multipole.compute_from_sources(leaf_sources,
                                                             leaf_charges,
                                                             leaf.center)

    def _translate_multipole_to_multipole(self):
        max_level = max(self.octree.nodes_by_level.keys())
        for level in range(max_level, 0, -1):
            nodes = self.octree.nodes_by_level[level]
            for node in nodes:
                if node.parent is not None and node.multipole is not None:
                    if node.parent.multipole is None:
                        config = node.parent.expansion_config
                        node.parent.multipole = self._create_expansion(config, is_multipole=True)

                    translation = node.center - node.parent.center

                    if node.expansion_config.expansion_type == ExpansionType.PLANE_WAVE:
                        translated = node.multipole.translate(translation)
                        if node.parent.expansion_config.expansion_type == ExpansionType.PLANE_WAVE:
                            node.parent.multipole.coeffs += translated.coeffs
                        else:
                            self._add_plane_wave_to_spherical(translated,
                                                               node.parent.multipole)
                    else:
                        if node.parent.expansion_config.expansion_type == ExpansionType.SPHERICAL_HARMONICS:
                            self._m2m_spherical(node.multipole, node.parent.multipole,
                                                 translation)
                        else:
                            self._add_spherical_to_plane_wave(node.multipole,
                                                               node.parent.multipole,
                                                               translation)

    def _m2m_spherical(self, child: SphericalExpansion, parent: SphericalExpansion,
                       translation: np.ndarray):
        r = np.linalg.norm(translation)
        theta = np.arccos(translation[2] / max(r, 1e-12))
        phi = np.arctan2(translation[1], translation[0])

        for l in range(child.p + 1):
            for m in range(-l, l + 1):
                if abs(child.coeffs[l, m + child.p]) > 0:
                    for j in range(parent.p + 1):
                        for k in range(-j, j + 1):
                            if abs(j - l) <= j + l and abs(k - m) <= j + l:
                                jl = spherical_jn(abs(j - l), self.k * r)
                                Y = SphericalExpansion.Y_lm(abs(j - l), k - m, theta, phi)
                                gaunt = 1.0 if (j == 0 and l == 0) else 0.25 / np.pi
                                parent.coeffs[j, k + parent.p] += (
                                    child.coeffs[l, m + child.p] * (1j)**(j - l) *
                                    4 * np.pi * gaunt * jl * Y
                                )

    def _add_plane_wave_to_spherical(self, pw: PlaneWaveExpansion,
                                       sph: SphericalExpansion):
        pass

    def _add_spherical_to_plane_wave(self, sph: SphericalExpansion,
                                       pw: PlaneWaveExpansion,
                                       translation: np.ndarray):
        pass

    def _translate_multipole_to_local(self):
        for level in self.octree.nodes_by_level:
            nodes = self.octree.nodes_by_level[level]
            for node in nodes:
                config = node.expansion_config
                node.local = self._create_expansion(config, is_multipole=False)

                for source in self.interaction_list.get(node, []):
                    if source.multipole is not None:
                        self._m2l_translate(source, node)

    def _m2l_translate(self, source: BroadbandOctreeNode, target: BroadbandOctreeNode):
        translation = source.center - target.center

        src_config = source.expansion_config
        tgt_config = target.expansion_config

        if (src_config.expansion_type == ExpansionType.PLANE_WAVE and
            tgt_config.expansion_type == ExpansionType.PLANE_WAVE):
            translated = source.multipole.translate(translation)
            if isinstance(target.local, PlaneWaveExpansion):
                target.local.coeffs += translated.coeffs

        elif (src_config.expansion_type == ExpansionType.SPHERICAL_HARMONICS and
              tgt_config.expansion_type == ExpansionType.SPHERICAL_HARMONICS):
            self._m2l_spherical(source.multipole, target.local, translation)

    def _m2l_spherical(self, source: SphericalExpansion, target: SphericalExpansion,
                       translation: np.ndarray):
        r = np.linalg.norm(translation)
        theta = np.arccos(translation[2] / max(r, 1e-12))
        phi = np.arctan2(translation[1], translation[0])

        for l in range(source.p + 1):
            for m in range(-l, l + 1):
                src_coeff = source.coeffs[l, m + source.p]
                if abs(src_coeff) > 0:
                    for j in range(target.p + 1):
                        for k in range(-j, j + 1):
                            j_l = abs(j - l)
                            if j + l <= target.p and abs(k + m) <= j + l:
                                hl = hankel2(j_l, self.k * r)
                                Y = SphericalExpansion.Y_lm(j_l, k + m, theta, phi)
                                target.coeffs[j, k + target.p] += (
                                    src_coeff * 4 * np.pi * (-1j)**j * (1j)**l * hl * Y
                                )

    def _translate_local_to_local(self):
        max_level = max(self.octree.nodes_by_level.keys())
        for level in range(max_level):
            nodes = self.octree.nodes_by_level[level]
            for node in nodes:
                if node.local is not None:
                    for child in node.children:
                        if child.local is None:
                            config = child.expansion_config
                            child.local = self._create_expansion(config, is_multipole=False)

                        translation = child.center - node.center

                        if node.expansion_config.expansion_type == ExpansionType.PLANE_WAVE:
                            translated = node.local.translate(-translation)
                            if isinstance(child.local, PlaneWaveExpansion):
                                child.local.coeffs += translated.coeffs

    def _evaluate_local_expansions(self, targets: np.ndarray) -> np.ndarray:
        result = np.zeros(len(targets), dtype=complex)
        leaves = self.octree.get_leaf_nodes()

        for i, target in enumerate(targets):
            leaf = self._find_leaf(self.octree.root, target)
            if leaf is not None and leaf.local is not None:
                config = leaf.expansion_config
                if config.expansion_type == ExpansionType.SPHERICAL_HARMONICS:
                    result[i] = leaf.local.evaluate_local(np.array([target]),
                                                            leaf.center)[0]
                else:
                    result[i] = leaf.local.evaluate(np.array([target]),
                                                     leaf.center)[0]
        return result

    def _find_leaf(self, node: BroadbandOctreeNode, point: np.ndarray) -> BroadbandOctreeNode:
        if node.is_leaf():
            return node
        for child in node.children:
            if np.all(np.abs(point - child.center) <= child.size / 2):
                return self._find_leaf(child, point)
        return node

    def _evaluate_neighbors_direct(self, targets: np.ndarray,
                                    sources: np.ndarray,
                                    charges: np.ndarray) -> np.ndarray:
        result = np.zeros(len(targets), dtype=complex)

        for i, target in enumerate(targets):
            leaf = self._find_leaf(self.octree.root, target)
            if leaf is None:
                continue

            neighbors = self._get_neighbors(leaf)
            neighbor_points = []
            neighbor_charges = []

            for neighbor in neighbors:
                if neighbor.points is not None:
                    for j, pt in enumerate(neighbor.points):
                        mask = np.isin(sources, pt).all(axis=1)
                        if np.any(mask):
                            neighbor_points.append(pt)
                            neighbor_charges.append(charges[mask][0])

            if neighbor_points:
                neighbor_points = np.array(neighbor_points)
                neighbor_charges = np.array(neighbor_charges)

                for j, src in enumerate(neighbor_points):
                    r = target - src
                    r_norm = np.linalg.norm(r)
                    if r_norm > 1e-12:
                        result[i] += neighbor_charges[j] * np.exp(-1j * self.k * r_norm) / (4 * np.pi * r_norm)

            if leaf.points is not None:
                for j, src in enumerate(leaf.points):
                    r = target - src
                    r_norm = np.linalg.norm(r)
                    if r_norm > 1e-12:
                        mask = np.isin(sources, src).all(axis=1)
                        if np.any(mask):
                            result[i] += charges[mask][0] * np.exp(-1j * self.k * r_norm) / (4 * np.pi * r_norm)

        return result


class BroadbandFMMMatrixFree:
    def __init__(self, k: float, tol: float = 1e-6,
                 max_points_per_leaf: int = 25,
                 kr_switch: float = 8.0):
        self.fmm = BroadbandFMM(k, tol, max_points_per_leaf, kr_switch)
        self.k = k
        self.sources = None
        self.normals = None

    def setup(self, sources: np.ndarray, normals: np.ndarray = None):
        self.sources = sources
        self.normals = normals
        self.fmm.build_tree(sources, sources)

    def matvec_single_layer(self, x: np.ndarray) -> np.ndarray:
        return self.fmm.compute_potential(self.sources, x, self.sources)

    def matvec_double_layer(self, x: np.ndarray) -> np.ndarray:
        result = np.zeros(len(x), dtype=complex)

        for i, src in enumerate(self.sources):
            neighbor_indices = self._get_neighbor_indices(i)
            for j in neighbor_indices:
                if i != j:
                    r = self.sources[j] - src
                    r_norm = np.linalg.norm(r)
                    if r_norm > 1e-12 and self.normals is not None:
                        n = self.normals[j]
                        grad = (-1j * self.k * r_norm - 1) * r * np.exp(-1j * self.k * r_norm) / (4 * np.pi * r_norm**3)
                        result[i] += x[j] * np.sum(grad * n)

        return result

    def _get_neighbor_indices(self, idx: int) -> List[int]:
        point = self.sources[idx]
        leaf = self.fmm._find_leaf(self.fmm.octree.root, point)
        if leaf is None:
            return []

        indices = []
        neighbors = self.fmm._get_neighbors(leaf)
        neighbors.append(leaf)

        for neighbor in neighbors:
            if neighbor.point_indices is not None:
                indices.extend(neighbor.point_indices.tolist())

        return indices
