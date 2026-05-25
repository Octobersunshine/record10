import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass, field
from scipy.special import spherical_jn, hankel2

from .kernels import SphericalHarmonics, MultipoleExpansion, LocalExpansion


@dataclass
class OctreeNode:
    center: np.ndarray
    size: float
    level: int
    index: Tuple[int, int, int]
    parent: 'OctreeNode' = None
    children: List['OctreeNode'] = field(default_factory=list)
    points: np.ndarray = None
    point_indices: np.ndarray = None
    multipole: MultipoleExpansion = None
    local: LocalExpansion = None

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class Octree:
    def __init__(self, points: np.ndarray, max_points_per_leaf: int = 10,
                 max_level: int = 10):
        self.points = points
        self.max_points_per_leaf = max_points_per_leaf
        self.max_level = max_level
        self.root = self._build_tree()
        self.nodes_by_level: Dict[int, List[OctreeNode]] = {}
        self._index_nodes(self.root)

    def _build_tree(self) -> OctreeNode:
        min_corner = np.min(self.points, axis=0)
        max_corner = np.max(self.points, axis=0)
        center = (min_corner + max_corner) / 2
        size = np.max(max_corner - min_corner) * 1.01

        root = OctreeNode(center=center, size=size, level=0, index=(0, 0, 0))
        root.points = self.points
        root.point_indices = np.arange(len(self.points))
        self._subdivide(root)
        return root

    def _subdivide(self, node: OctreeNode):
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
                    child = OctreeNode(
                        center=child_center,
                        size=half_size,
                        level=node.level + 1,
                        index=(2 * node.index[0] + i,
                               2 * node.index[1] + j,
                               2 * node.index[2] + k),
                        parent=node
                    )
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

    def _index_nodes(self, node: OctreeNode):
        if node.level not in self.nodes_by_level:
            self.nodes_by_level[node.level] = []
        self.nodes_by_level[node.level].append(node)
        for child in node.children:
            self._index_nodes(child)

    def get_leaf_nodes(self) -> List[OctreeNode]:
        leaves = []
        self._collect_leaves(self.root, leaves)
        return leaves

    def _collect_leaves(self, node: OctreeNode, leaves: List[OctreeNode]):
        if node.is_leaf():
            leaves.append(node)
        else:
            for child in node.children:
                self._collect_leaves(child, leaves)


class FastMultipoleMethod:
    def __init__(self, k: float, p: int = 6, max_points_per_leaf: int = 20):
        self.k = k
        self.p = p
        self.max_points_per_leaf = max_points_per_leaf
        self.octree = None
        self.interaction_list: Dict[OctreeNode, List[OctreeNode]] = {}

    def build_tree(self, sources: np.ndarray, targets: np.ndarray = None):
        if targets is None:
            targets = sources
        all_points = np.vstack([sources, targets])
        self.octree = Octree(all_points, self.max_points_per_leaf)
        self._build_interaction_list()
        return self.octree

    def _build_interaction_list(self):
        for level in self.octree.nodes_by_level:
            nodes = self.octree.nodes_by_level[level]
            for node in nodes:
                self.interaction_list[node] = self._get_interaction_list(node)

    def _get_interaction_list(self, node: OctreeNode) -> List[OctreeNode]:
        if node.parent is None:
            return []
        parent_neighbors = self._get_neighbors(node.parent)
        interaction = []
        for parent_neighbor in parent_neighbors:
            for child in parent_neighbor.children:
                if self._is_well_separated(node, child):
                    interaction.append(child)
        return interaction

    def _get_neighbors(self, node: OctreeNode) -> List[OctreeNode]:
        neighbors = []
        if node.level == 0:
            return neighbors
        level_nodes = self.octree.nodes_by_level.get(node.level, [])
        for other in level_nodes:
            if other is not node and self._is_neighbor(node, other):
                neighbors.append(other)
        return neighbors

    def _is_neighbor(self, node1: OctreeNode, node2: OctreeNode) -> bool:
        dx = abs(node1.center[0] - node2.center[0])
        dy = abs(node1.center[1] - node2.center[1])
        dz = abs(node1.center[2] - node2.center[2])
        max_dist = node1.size * 1.5
        return dx <= max_dist and dy <= max_dist and dz <= max_dist

    def _is_well_separated(self, node1: OctreeNode, node2: OctreeNode) -> bool:
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
        return self._evaluate_local_expansions(targets)

    def _compute_multipole_expansions(self, sources: np.ndarray, charges: np.ndarray):
        leaves = self.octree.get_leaf_nodes()
        for leaf in leaves:
            if leaf.points is not None and len(leaf.points) > 0:
                leaf.multipole = MultipoleExpansion(self.k, self.p)
                mask = np.isin(sources, leaf.points).all(axis=1)
                leaf_sources = sources[mask]
                leaf_charges = charges[mask]
                if len(leaf_sources) > 0:
                    leaf.multipole.compute_local(leaf_sources, leaf_charges, leaf.center)

    def _translate_multipole_to_multipole(self):
        max_level = max(self.octree.nodes_by_level.keys())
        for level in range(max_level, 0, -1):
            nodes = self.octree.nodes_by_level[level]
            for node in nodes:
                if node.parent is not None and node.multipole is not None:
                    if node.parent.multipole is None:
                        node.parent.multipole = MultipoleExpansion(self.k, self.p)
                    translation = node.center - node.parent.center
                    translated = node.multipole.translate(translation)
                    node.parent.multipole.coeffs += translated.coeffs

    def _translate_multipole_to_local(self):
        for level in self.octree.nodes_by_level:
            nodes = self.octree.nodes_by_level[level]
            for node in nodes:
                node.local = LocalExpansion(self.k, self.p)
                for source in self.interaction_list.get(node, []):
                    if source.multipole is not None:
                        self._m2l_translate(source, node)

    def _m2l_translate(self, source: OctreeNode, target: OctreeNode):
        translation = source.center - target.center
        r = np.linalg.norm(translation)
        theta = np.arccos(translation[2] / max(r, 1e-12))
        phi = np.arctan2(translation[1], translation[0])

        for l in range(self.p + 1):
            for m in range(-l, l + 1):
                src_coeff = source.multipole.coeffs[l, m + self.p]
                if abs(src_coeff) > 0:
                    for j in range(self.p + 1):
                        for k in range(-j, j + 1):
                            j_l = abs(j - l)
                            if j + l <= self.p and abs(k + m) <= j + l:
                                hl = hankel2(j_l, self.k * r)
                                Y = SphericalHarmonics.Y_lm(j_l, k + m, theta, phi)
                                target.local.coeffs[j, k + self.p] += (
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
                            child.local = LocalExpansion(self.k, self.p)
                        translation = child.center - node.center
                        translated = node.local.translate(translation)
                        child.local.coeffs += translated.coeffs

    def _evaluate_local_expansions(self, targets: np.ndarray) -> np.ndarray:
        result = np.zeros(len(targets), dtype=complex)
        leaves = self.octree.get_leaf_nodes()

        for i, target in enumerate(targets):
            leaf = self._find_leaf(self.octree.root, target)
            if leaf is not None and leaf.local is not None:
                result[i] = leaf.local.evaluate(np.array([target]), leaf.center)[0]
            result[i] += self._evaluate_neighbors(leaf, target)
        return result

    def _find_leaf(self, node: OctreeNode, point: np.ndarray) -> OctreeNode:
        if node.is_leaf():
            return node
        for child in node.children:
            if np.all(np.abs(point - child.center) <= child.size / 2):
                return self._find_leaf(child, point)
        return node

    def _evaluate_neighbors(self, leaf: OctreeNode, point: np.ndarray) -> complex:
        result = 0j
        if leaf is None:
            return result
        neighbors = self._get_neighbors(leaf)
        for neighbor in neighbors:
            if neighbor.points is not None:
                for src in neighbor.points:
                    r = point - src
                    r_norm = np.linalg.norm(r)
                    if r_norm > 1e-12:
                        result += np.exp(-1j * self.k * r_norm) / (4 * np.pi * r_norm)
        return result


class FMMMatrixFree:
    def __init__(self, k: float, p: int = 8, max_points_per_leaf: int = 25):
        self.fmm = FastMultipoleMethod(k, p, max_points_per_leaf)
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
            for j, tgt in enumerate(self.sources):
                if i != j:
                    r = tgt - src
                    r_norm = np.linalg.norm(r)
                    if r_norm > 1e-12 and self.normals is not None:
                        n = self.normals[j]
                        grad = (-1j * self.k * r_norm - 1) * r * np.exp(-1j * self.k * r_norm) / (4 * np.pi * r_norm**3)
                        result[i] += x[j] * np.sum(grad * n)
        return result
