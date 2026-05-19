import numpy as np
import math
from typing import List, Tuple, Optional


class Particle:
    def __init__(self, x: float, y: float, charge: float):
        self.x = x
        self.y = y
        self.charge = charge
        self.potential = 0.0


class QuadNode:
    def __init__(self, x0: float, y0: float, size: float, level: int = 0):
        self.x0 = x0
        self.y0 = y0
        self.size = size
        self.level = level
        self.particles: List[Particle] = []
        self.children: List[Optional[QuadNode]] = [None, None, None, None]
        self.parent: Optional[QuadNode] = None
        self.multipole: Optional[np.ndarray] = None
        self.local: Optional[np.ndarray] = None

    def contains(self, particle: Particle) -> bool:
        return (self.x0 <= particle.x < self.x0 + self.size and
                self.y0 <= particle.y < self.y0 + self.size)

    def is_leaf(self) -> bool:
        return all(child is None for child in self.children)


class FMM2D:
    def __init__(self, p: int = 10, max_particles_per_leaf: int = 10):
        self.p = p
        self.max_particles_per_leaf = max_particles_per_leaf
        self.root: Optional[QuadNode] = None

    def _build_tree(self, particles: List[Particle]) -> None:
        x_coords = [p.x for p in particles]
        y_coords = [p.y for p in particles]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        size = max(x_max - x_min, y_max - y_min) * 1.01
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        self.root = QuadNode(center_x - size / 2, center_y - size / 2, size)

        for particle in particles:
            self._insert_particle(self.root, particle)

    def _insert_particle(self, node: QuadNode, particle: Particle) -> None:
        if not node.contains(particle):
            return

        if node.is_leaf() and len(node.particles) < self.max_particles_per_leaf:
            node.particles.append(particle)
            return

        if node.is_leaf():
            self._subdivide(node)

        for child in node.children:
            if child is not None and child.contains(particle):
                self._insert_particle(child, particle)
                break

    def _subdivide(self, node: QuadNode) -> None:
        half_size = node.size / 2
        x0, y0 = node.x0, node.y0

        node.children[0] = QuadNode(x0, y0 + half_size, half_size, node.level + 1)
        node.children[1] = QuadNode(x0 + half_size, y0 + half_size, half_size, node.level + 1)
        node.children[2] = QuadNode(x0, y0, half_size, node.level + 1)
        node.children[3] = QuadNode(x0 + half_size, y0, half_size, node.level + 1)

        for child in node.children:
            child.parent = node

        for particle in node.particles:
            for child in node.children:
                if child is not None and child.contains(particle):
                    child.particles.append(particle)
                    break

        node.particles = []

    def _compute_multipole(self, node: QuadNode) -> None:
        if node.is_leaf():
            node.multipole = np.zeros(self.p + 1, dtype=complex)
            center_x = node.x0 + node.size / 2
            center_y = node.y0 + node.size / 2
            for particle in node.particles:
                dx = particle.x - center_x
                dy = particle.y - center_y
                z = complex(dx, dy)
                for k in range(self.p + 1):
                    if k == 0:
                        node.multipole[k] += particle.charge
                    else:
                        node.multipole[k] += particle.charge * (z ** k) / k
        else:
            for child in node.children:
                if child is not None:
                    self._compute_multipole(child)

            node.multipole = np.zeros(self.p + 1, dtype=complex)
            center_x = node.x0 + node.size / 2
            center_y = node.y0 + node.size / 2
            for child in node.children:
                if child is None:
                    continue
                child_cx = child.x0 + child.size / 2
                child_cy = child.y0 + child.size / 2
                z0 = complex(child_cx - center_x, child_cy - center_y)
                for k in range(self.p + 1):
                    node.multipole[k] += child.multipole[k]
                    for l in range(k):
                        if l == 0:
                            node.multipole[k] += child.multipole[l] * (z0 ** k) / k
                        else:
                            binom = math.comb(k - 1, l - 1)
                            node.multipole[k] += child.multipole[l] * binom * (z0 ** (k - l))

    def _is_well_separated(self, node1: QuadNode, node2: QuadNode) -> bool:
        dx = abs((node1.x0 + node1.size / 2) - (node2.x0 + node2.size / 2))
        dy = abs((node1.y0 + node1.size / 2) - (node2.y0 + node2.size / 2))
        return max(dx, dy) >= node1.size * 1.5

    def _get_all_nodes_at_level(self, node: QuadNode, level: int, nodes: List[QuadNode]) -> None:
        if node.level == level:
            nodes.append(node)
        elif not node.is_leaf():
            for child in node.children:
                if child is not None:
                    self._get_all_nodes_at_level(child, level, nodes)

    def _interaction_list(self, node: QuadNode) -> List[QuadNode]:
        if node.parent is None:
            return []

        all_same_level = []
        self._get_all_nodes_at_level(self.root, node.level, all_same_level)

        parent_neighbors_set = set()
        for candidate in all_same_level:
            if candidate.parent == node.parent:
                continue
            if not self._is_well_separated(node.parent, candidate.parent):
                parent_neighbors_set.add(candidate.parent)
        parent_neighbors = list(parent_neighbors_set)

        interaction = []
        for pn in parent_neighbors:
            if pn.is_leaf():
                if self._is_well_separated(node, pn):
                    interaction.append(pn)
            else:
                for child in pn.children:
                    if child is not None and self._is_well_separated(node, child):
                        interaction.append(child)

        return interaction

    def _multipole_to_local(self, node: QuadNode, source: QuadNode) -> None:
        if node.local is None:
            node.local = np.zeros(self.p + 1, dtype=complex)

        node_cx = node.x0 + node.size / 2
        node_cy = node.y0 + node.size / 2
        src_cx = source.x0 + source.size / 2
        src_cy = source.y0 + source.size / 2

        z0 = complex(node_cx - src_cx, node_cy - src_cy)
        if abs(z0) < 1e-15:
            return

        for l in range(self.p + 1):
            if l == 0:
                node.local[l] += -source.multipole[0] * np.log(z0)
                for k in range(1, self.p + 1):
                    node.local[l] += source.multipole[k] / (z0 ** k)
            else:
                sign = (-1) ** l
                node.local[l] += source.multipole[0] * sign / (l * (z0 ** l))
                for k in range(1, self.p + 1):
                    binom = math.comb(k + l - 1, l)
                    node.local[l] += source.multipole[k] * binom * sign / (z0 ** (k + l))

    def _local_to_local(self, node: QuadNode) -> None:
        if node.parent is None or node.parent.local is None:
            return

        if node.local is None:
            node.local = np.zeros(self.p + 1, dtype=complex)

        node_cx = node.x0 + node.size / 2
        node_cy = node.y0 + node.size / 2
        parent_cx = node.parent.x0 + node.parent.size / 2
        parent_cy = node.parent.y0 + node.parent.size / 2

        z0 = complex(node_cx - parent_cx, node_cy - parent_cy)

        for l in range(self.p + 1):
            for k in range(l, self.p + 1):
                binom = math.comb(k, l)
                node.local[l] += node.parent.local[k] * (z0 ** (k - l))

    def _evaluate_local(self, node: QuadNode) -> None:
        if node.local is None:
            return

        center_x = node.x0 + node.size / 2
        center_y = node.y0 + node.size / 2

        for particle in node.particles:
            dx = particle.x - center_x
            dy = particle.y - center_y
            z = complex(dx, dy)

            pot = 0.0
            for l in range(self.p + 1):
                pot += (node.local[l] * (z ** l)).real
            particle.potential += pot

    def _direct_interactions(self, node: QuadNode) -> None:
        all_particles = []
        self._collect_neighbor_particles(node, all_particles)

        for p1 in node.particles:
            for p2 in all_particles:
                if p1 is p2:
                    continue
                dx = p1.x - p2.x
                dy = p1.y - p2.y
                r2 = dx * dx + dy * dy
                if r2 > 1e-15:
                    p1.potential -= p2.charge * 0.5 * np.log(r2)

    def _collect_neighbor_particles(self, node: QuadNode, particles: List[Particle]) -> None:
        all_leaf_nodes = []
        self._collect_all_leaf_nodes(self.root, all_leaf_nodes)

        for leaf in all_leaf_nodes:
            if not self._is_well_separated(node, leaf):
                particles.extend(leaf.particles)

    def _collect_all_leaf_nodes(self, node: QuadNode, leaf_nodes: List[QuadNode]) -> None:
        if node.is_leaf():
            leaf_nodes.append(node)
        else:
            for child in node.children:
                if child is not None:
                    self._collect_all_leaf_nodes(child, leaf_nodes)

    def _collect_all_particles(self, node: QuadNode, particles: List[Particle]) -> None:
        if node.is_leaf():
            particles.extend(node.particles)
        else:
            for child in node.children:
                if child is not None:
                    self._collect_all_particles(child, particles)

    def _upward_pass(self, node: QuadNode) -> None:
        if node.is_leaf():
            self._compute_multipole(node)
        else:
            for child in node.children:
                if child is not None:
                    self._upward_pass(child)
            self._compute_multipole(node)

    def _downward_pass(self, node: QuadNode) -> None:
        if not node.is_leaf():
            self._local_to_local(node)

            interaction = self._interaction_list(node)
            for src in interaction:
                self._multipole_to_local(node, src)

            for child in node.children:
                if child is not None:
                    self._downward_pass(child)
        else:
            self._evaluate_local(node)
            self._direct_interactions(node)

    def compute_potential(self, particles: List[Particle]) -> List[float]:
        for p in particles:
            p.potential = 0.0

        self._build_tree(particles)
        self._upward_pass(self.root)

        self.root.local = np.zeros(self.p + 1, dtype=complex)
        self._downward_pass(self.root)

        return [p.potential for p in particles]


def direct_potential(particles: List[Particle]) -> List[float]:
    n = len(particles)
    potentials = np.zeros(n)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dx = particles[i].x - particles[j].x
            dy = particles[i].y - particles[j].y
            r2 = dx * dx + dy * dy
            if r2 > 1e-15:
                potentials[i] -= particles[j].charge * 0.5 * np.log(r2)

    return potentials.tolist()


if __name__ == "__main__":
    np.random.seed(42)
    n_particles = 100

    particles = []
    for i in range(n_particles):
        x = np.random.random()
        y = np.random.random()
        charge = np.random.random() * 2 - 1
        particles.append(Particle(x, y, charge))

    print(f"计算 {n_particles} 个粒子的静电势...")

    fmm = FMM2D(p=10, max_particles_per_leaf=10)
    fmm_potentials = fmm.compute_potential(particles)

    direct_potentials = direct_potential(particles)

    error = np.mean(np.abs(np.array(fmm_potentials) - np.array(direct_potentials)))
    max_error = np.max(np.abs(np.array(fmm_potentials) - np.array(direct_potentials)))

    print(f"\n结果验证:")
    print(f"平均误差: {error:.2e}")
    print(f"最大误差: {max_error:.2e}")

    print(f"\n前10个粒子的电势对比:")
    for i in range(min(10, n_particles)):
        print(f"粒子 {i:2d}: FMM = {fmm_potentials[i]:.6f}, 直接 = {direct_potentials[i]:.6f}, 差 = {abs(fmm_potentials[i] - direct_potentials[i]):.2e}")
