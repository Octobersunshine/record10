import numpy as np
import math
from typing import List, Tuple, Optional, Dict
import time

try:
    import cupy as cp
    from cupy import cusolver
    CUPY_AVAILABLE = True
    print(f"CuPy 已加载，GPU: {cp.cuda.get_device_name(0)}")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy 未找到，将使用CPU模式")


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
        self.particle_indices: List[int] = []

    def contains(self, particle: Particle) -> bool:
        return (self.x0 <= particle.x < self.x0 + self.size and
                self.y0 <= particle.y < self.y0 + self.size)

    def is_leaf(self) -> bool:
        return all(child is None for child in self.children)


class FMM2DGPU:
    def __init__(self, p: int = 10, max_particles_per_leaf: int = 10, 
                 use_gpu: bool = True, gpu_threshold: int = 100):
        self.p = p
        self.max_particles_per_leaf = max_particles_per_leaf
        self.root: Optional[QuadNode] = None
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.gpu_threshold = gpu_threshold
        self.xp = cp if self.use_gpu else np
        
        if self.use_gpu:
            print("GPU模式已启用")
            self._precompute_binomial_coefficients()
        else:
            print("CPU模式已启用")

    def _precompute_binomial_coefficients(self) -> None:
        max_k = self.p + 1
        self.binom_m2m = np.zeros((max_k, max_k), dtype=np.float64)
        self.binom_m2l = np.zeros((max_k, max_k), dtype=np.float64)
        
        for k in range(max_k):
            for l in range(k):
                if l > 0:
                    self.binom_m2m[k, l] = math.comb(k - 1, l - 1)
        
        for k in range(max_k):
            for l in range(max_k):
                self.binom_m2l[k, l] = math.comb(k + l - 1, l)

    def _build_tree(self, particles: List[Particle]) -> None:
        x_coords = [p.x for p in particles]
        y_coords = [p.y for p in particles]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        size = max(x_max - x_min, y_max - y_min) * 1.01
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        self.root = QuadNode(center_x - size / 2, center_y - size / 2, size)

        for idx, particle in enumerate(particles):
            particle.index = idx
            self._insert_particle(self.root, particle, idx)

    def _insert_particle(self, node: QuadNode, particle: Particle, idx: int) -> None:
        if not node.contains(particle):
            return

        if node.is_leaf() and len(node.particles) < self.max_particles_per_leaf:
            node.particles.append(particle)
            node.particle_indices.append(idx)
            return

        if node.is_leaf():
            self._subdivide(node)

        for child in node.children:
            if child is not None and child.contains(particle):
                self._insert_particle(child, particle, idx)
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

        for i, particle in enumerate(node.particles):
            for child in node.children:
                if child is not None and child.contains(particle):
                    child.particles.append(particle)
                    child.particle_indices.append(node.particle_indices[i])
                    break

        node.particles = []
        node.particle_indices = []

    def _compute_multipole_gpu_batch(self, nodes: List[QuadNode], 
                                     x_all: np.ndarray, y_all: np.ndarray, 
                                     charge_all: np.ndarray) -> None:
        if not self.use_gpu or len(nodes) == 0:
            return

        n_nodes = len(nodes)
        max_particles = max(len(n.particle_indices) for n in nodes)
        
        centers_x = self.xp.array([n.x0 + n.size / 2 for n in nodes], dtype=self.xp.float64)
        centers_y = self.xp.array([n.y0 + n.size / 2 for n in nodes], dtype=self.xp.float64)
        
        indices_list = [self.xp.array(n.particle_indices, dtype=self.xp.int32) for n in nodes]
        max_len = max(len(indices) for indices in indices_list)
        
        indices_padded = self.xp.zeros((n_nodes, max_len), dtype=self.xp.int32)
        mask = self.xp.zeros((n_nodes, max_len), dtype=self.xp.bool_)
        
        for i, indices in enumerate(indices_list):
            indices_padded[i, :len(indices)] = indices
            mask[i, :len(indices)] = True

        x_gpu = self.xp.array(x_all, dtype=self.xp.float64)
        y_gpu = self.xp.array(y_all, dtype=self.xp.float64)
        charge_gpu = self.xp.array(charge_all, dtype=self.xp.float64)

        x_batch = x_gpu[indices_padded] - centers_x[:, self.xp.newaxis]
        y_batch = y_gpu[indices_padded] - centers_y[:, self.xp.newaxis]
        charge_batch = charge_gpu[indices_padded]

        z = x_batch + 1j * y_batch

        multipoles = self.xp.zeros((n_nodes, self.p + 1), dtype=self.xp.complex128)
        multipoles[:, 0] = self.xp.sum(charge_batch * mask, axis=1)

        for k in range(1, self.p + 1):
            z_pow = z ** k
            multipoles[:, k] = self.xp.sum(charge_batch * z_pow * mask, axis=1) / k

        multipoles_cpu = multipoles.get()
        for i, node in enumerate(nodes):
            node.multipole = multipoles_cpu[i]

    def _compute_multipole(self, node: QuadNode, x_all: np.ndarray, y_all: np.ndarray, 
                          charge_all: np.ndarray, use_batch: bool = True) -> None:
        if node.is_leaf():
            if self._compute_multipole_gpu_batch([node], x_all, y_all, charge_all)
        else:
            leaf_children = [child for child in node.children if child is not None and child.is_leaf()]
            non_leaf_children = [child for child in node.children if child is not None and not child.is_leaf()]

            if self.use_gpu and len(leaf_children) >= 4:
                self._compute_multipole_gpu_batch(leaf_children, x_all, y_all, charge_all)
                for child in non_leaf_children:
                    self._compute_multipole(child, x_all, y_all, charge_all)
            else:
                for child in node.children:
                    if child is not None:
                        self._compute_multipole(child, x_all, y_all, charge_all)

            node.multipole = np.zeros(self.p + 1), dtype=complex)
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

    def _multipole_to_local_batch(self, nodes: List[Tuple[QuadNode, List[QuadNode]]]) -> None:
        if not self.use_gpu or len(nodes) == 0:
            for node, sources in nodes:
                for source in sources:
                    self._multipole_to_local_single(node, source)
            return

        n_nodes = len(nodes)
        n_sources_max = max(len(sources) for _, sources in nodes)

        node_cx = []
        node_cy = []
        src_cx = []
        src_cy = []
        src_multipoles = []

        for node, sources in nodes:
            node_cx.append(node.x0 + node.size / 2)
            node_cy.append(node.y0 + node.size / 2)
            for source in sources:
                src_cx.append(source.x0 + source.size / 2)
                src_cy.append(source.y0 + source.size / 2)
                src_multipoles.append(source.multipole)

        node_cx_gpu = self.xp.array(node_cx, dtype=self.xp.float64)
        node_cy_gpu = self.xp.array(node_cy, dtype=self.xp.float64)
        src_cx_gpu = self.xp.array(src_cx, dtype=self.xp.float64)
        src_cy_gpu = self.xp.array(src_cy, dtype=self.xp.float64)
        src_multipoles_gpu = self.xp.array(src_multipoles, dtype=self.xp.complex128)

        z0 = (node_cx_gpu[:, self.xp.newaxis] - src_cx_gpu[self.xp.newaxis, :]) + \
             1j * (node_cy_gpu[:, self.xp.newaxis] - src_cy_gpu[self.xp.newaxis, :])

        z0_abs = self.xp.abs(z0)
        z0_safe = self.xp.where(z0_abs > 1e-15, z0, 1.0)

        local_expansions = self.xp.zeros((n_nodes, self.p + 1), dtype=self.xp.complex128)

        local_expansions[:, 0] = -self.xp.sum(src_multipoles_gpu[:, 0] * self.xp.log(z0_safe), axis=1)
        for k in range(1, self.p + 1):
            local_expansions[:, 0] += self.xp.sum(src_multipoles_gpu[:, k] / (z0_safe ** k), axis=1)

        for l in range(1, self.p + 1):
            sign = (-1) ** l
            local_expansions[:, l] = sign * self.xp.sum(src_multipoles_gpu[:, 0] / (l * (z0_safe ** l)), axis=1)
            for k in range(1, self.p + 1):
                binom = math.comb(k + l - 1, l)
                local_expansions[:, l] += sign * binom * self.xp.sum(
                    src_multipoles_gpu[:, k] / (z0_safe ** (k + l)), axis=1)

        local_cpu = local_expansions.get()
        for i, (node, _) in enumerate(nodes):
            if node.local is None:
                node.local = local_cpu[i]
            else:
                node.local += local_cpu[i]

    def _multipole_to_local_single(self, node: QuadNode, source: QuadNode) -> None:
        if node.local is None:
            node.local = np.zeros(self.p + 1), dtype=complex)

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
            node.local = np.zeros(self.p + 1), dtype=complex)

        node_cx = node.x0 + node.size / 2
        node_cy = node.y0 + node.size / 2
        parent_cx = node.parent.x0 + node.parent.size / 2
        parent_cy = node.parent.y0 + node.parent.size / 2

        z0 = complex(node_cx - parent_cx, node_cy - parent_cy)

        for l in range(self.p + 1):
            for k in range(l, self.p + 1):
                binom = math.comb(k, l)
                node.local[l] += node.parent.local[k] * (z0 ** (k - l))

    def _evaluate_local_gpu(self, nodes: List[QuadNode], x_all: np.ndarray, y_all: np.ndarray) -> None:
        if not self.use_gpu or len(nodes) == 0:
            for node in nodes:
                self._evaluate_local_single(node)
            return

        valid_nodes = [n for n in nodes if n.local is not None and len(n.particles) > 0]
        if len(valid_nodes) == 0:
            return

        n_nodes = len(valid_nodes)
        max_particles = max(len(n.particle_indices) for n in valid_nodes)

        centers_x = self.xp.array([n.x0 + n.size / 2 for n in valid_nodes], dtype=self.xp.float64)
        centers_y = self.xp.array([n.y0 + n.size / 2 for n in valid_nodes], dtype=self.xp.float64)
        locals_gpu = self.xp.array([n.local for n in valid_nodes], dtype=self.xp.complex128)

        indices_list = [self.xp.array(n.particle_indices, dtype=self.xp.int32) for n in valid_nodes]
        max_len = max(len(indices) for indices in indices_list)

        indices_padded = self.xp.zeros((n_nodes, max_len)), dtype=self.xp.int32)
        mask = self.xp.zeros((n_nodes, max_len)), dtype=self.xp.bool_)

        for i, indices in enumerate(indices_list):
            indices_padded[i, :len(indices)] = indices
            mask[i, :len(indices)] = True

        x_gpu = self.xp.array(x_all, dtype=self.xp.float64)
        y_gpu = self.xp.array(y_all, dtype=self.xp.float64)

        x_batch = x_gpu[indices_padded] - centers_x[:, self.xp.newaxis]
        y_batch = y_gpu[indices_padded] - centers_y[:, self.xp.newaxis]

        z = x_batch + 1j * y_batch

        potentials = self.xp.zeros((n_nodes, max_len)), dtype=self.xp.float64)

        for l in range(self.p + 1):
            z_pow = z ** l
            contribution = (locals_gpu[:, l][:, self.xp.newaxis] * z_pow).real
            potentials += contribution * mask

        potentials_cpu = potentials.get()

        for i, node in enumerate(valid_nodes):
            for j, p in enumerate(node.particles):
                p.potential += potentials_cpu[i, j]

    def _evaluate_local_single(self, node: QuadNode) -> None:
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

    def _direct_interactions_gpu(self, all_particles: List[Particle]) -> None:
        n = len(all_particles)
        if n == 0:
            return

        x_gpu = self.xp.array([p.x for p in all_particles], dtype=self.xp.float64)
        y_gpu = self.xp.array([p.y for p in all_particles], dtype=self.xp.float64)
        charge_gpu = self.xp.array([p.charge for p in all_particles], dtype=self.xp.float64)

        dx = x_gpu[:, self.xp.newaxis] - x_gpu[self.xp.newaxis, :]
        dy = y_gpu[:, self.xp.newaxis] - y_gpu[self.xp.newaxis, :]
        r2 = dx ** 2 + dy ** 2

        mask = r2 > 1e-15
        r2_safe = self.xp.where(mask, r2, 1.0)
        log_term = self.xp.log(r2_safe)

        pot_matrix = -0.5 * charge_gpu[self.xp.newaxis, :] * log_term
        pot_matrix = self.xp.where(mask, pot_matrix, 0.0)

        potentials_gpu = self.xp.sum(pot_matrix, axis=1)

        if self.use_gpu:
            potentials_cpu = potentials_gpu.get()
        else:
            potentials_cpu = potentials_gpu

        for i, p in enumerate(all_particles):
            p.potential += potentials_cpu[i]

    def _direct_interactions(self, node: QuadNode) -> None:
        all_particles = []
        self._collect_neighbor_particles(node, all_particles)

        if len(all_particles) == 0:
            return

        if self.use_gpu and len(all_particles) > self.gpu_threshold:
            self._direct_interactions_gpu(all_particles)
        else:
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

    def _upward_pass(self, x_all: np.ndarray, y_all: np.ndarray, charge_all: np.ndarray) -> None:
        self._compute_multipole(self.root, x_all, y_all, charge_all)

    def _downward_pass(self, node: QuadNode, x_all: np.ndarray, y_all: np.ndarray) -> None:
        if not node.is_leaf():
            self._local_to_local(node)

            interaction = self._interaction_list(node)
            if self.use_gpu and len(interaction) > 0:
                self._multipole_to_local_batch([(node, interaction)])
            else:
                for src in interaction:
                    self._multipole_to_local_single(node, src)

            for child in node.children:
                if child is not None:
                    self._downward_pass(child, x_all, y_all)
        else:
            self._evaluate_local_gpu([node], x_all, y_all)
            self._direct_interactions(node)

    def compute_potential(self, particles: List[Particle]) -> List[float]:
        for p in particles:
            p.potential = 0.0

        x_all = np.array([p.x for p in particles], dtype=np.float64)
        y_all = np.array([p.y for p in particles], dtype=np.float64)
        charge_all = np.array([p.charge for p in particles], dtype=np.float64)

        self._build_tree(particles)
        self._upward_pass(x_all, y_all, charge_all)

        self.root.local = np.zeros(self.p + 1), dtype=complex)
        self._downward_pass(self.root, x_all, y_all)

        return [p.potential for p in particles]


def direct_potential(particles: List[Particle], use_gpu: bool = True) -> List[float]:
    n = len(particles)

    use_gpu_final = use_gpu and CUPY_AVAILABLE
    xp = cp if use_gpu_final else np

    x = xp.array([p.x for p in particles], dtype=xp.float64)
    y = xp.array([p.y for p in particles], dtype=xp.float64)
    charge = xp.array([p.charge for p in particles], dtype=xp.float64)

    dx = x[:, xp.newaxis] - x[xp.newaxis, :]
    dy = y[:, xp.newaxis] - y[xp.newaxis, :]
    r2 = dx ** 2 + dy ** 2

    mask = r2 > 1e-15
    r2_safe = xp.where(mask, r2, 1.0)
    log_term = xp.log(r2_safe)

    pot_matrix = -0.5 * charge[xp.newaxis, :] * log_term
    pot_matrix = xp.where(mask, pot_matrix, 0.0)

    potentials = xp.sum(pot_matrix, axis=1)

    if use_gpu_final:
        potentials = potentials.get()

    return potentials.tolist()


if __name__ == "__main__":
    np.random.seed(42)
    n_particles = 1000

    particles = []
    for i in range(n_particles):
        x = np.random.random()
        y = np.random.random()
        charge = np.random.random() * 2 - 1
        particles.append(Particle(x, y, charge))

    print(f"计算 {n_particles} 个粒子的静电势...")
    print(f"GPU可用: {CUPY_AVAILABLE}")

    print("\n=== FMM GPU 计算 ===")
    start_time = time.time()
    fmm_gpu = FMM2DGPU(p=10, max_particles_per_leaf=10, use_gpu=True)
    fmm_gpu_potentials = fmm_gpu.compute_potential(particles)
    fmm_gpu_time = time.time() - start_time
    print(f"FMM GPU 耗时: {fmm_gpu_time:.4f} 秒")

    print("\n=== FMM CPU 计算 ===")
    start_time = time.time()
    fmm_cpu = FMM2DGPU(p=10, max_particles_per_leaf=10, use_gpu=False)
    fmm_cpu_potentials = fmm_cpu.compute_potential(particles)
    fmm_cpu_time = time.time() - start_time
    print(f"FMM CPU 耗时: {fmm_cpu_time:.4f} 秒")
    print(f"FMM GPU 加速比: {fmm_cpu_time / fmm_gpu_time:.2f}x")

    print("\n=== 直接计算 (CPU) ===")
    start_time = time.time()
    direct_potentials_cpu = direct_potential(particles, use_gpu=False)
    direct_cpu_time = time.time() - start_time
    print(f"直接计算 (CPU) 耗时: {direct_cpu_time:.4f} 秒")

    if CUPY_AVAILABLE:
        print("\n=== 直接计算 (GPU) ===")
        start_time = time.time()
        direct_potentials_gpu = direct_potential(particles, use_gpu=True)
        direct_gpu_time = time.time() - start_time
        print(f"直接计算 (GPU) 耗时: {direct_gpu_time:.4f} 秒")
        print(f"直接计算 GPU 加速比: {direct_cpu_time / direct_gpu_time:.2f}x")

    error = np.mean(np.abs(np.array(fmm_gpu_potentials) - np.array(direct_potentials_cpu)))
    max_error = np.max(np.abs(np.array(fmm_gpu_potentials) - np.array(direct_potentials_cpu))))

    print(f"\n结果验证:")
    print(f"平均误差: {error:.2e}")
    print(f"最大误差: {max_error:.2e}")

    print(f"\n前10个粒子的电势对比:")
    for i in range(min(10, n_particles)):
        print(f"粒子 {i:2d}: FMM = {fmm_gpu_potentials[i]:.6f}, 直接 = {direct_potentials_cpu[i]:.6f}, 差 = {abs(fmm_gpu_potentials[i] - direct_potentials_cpu[i]):.2e}")
