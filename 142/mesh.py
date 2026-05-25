import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Element:
    vertices: np.ndarray
    index: int

    @property
    def centroid(self) -> np.ndarray:
        return np.mean(self.vertices, axis=0)

    @property
    def area(self) -> float:
        if len(self.vertices) == 3:
            v1 = self.vertices[1] - self.vertices[0]
            v2 = self.vertices[2] - self.vertices[0]
            return 0.5 * np.linalg.norm(np.cross(v1, v2))
        elif len(self.vertices) == 4:
            v1 = self.vertices[1] - self.vertices[0]
            v2 = self.vertices[3] - self.vertices[0]
            return np.linalg.norm(np.cross(v1, v2))
        return 0.0

    @property
    def normal(self) -> np.ndarray:
        if len(self.vertices) == 3:
            v1 = self.vertices[1] - self.vertices[0]
            v2 = self.vertices[2] - self.vertices[0]
            n = np.cross(v1, v2)
        elif len(self.vertices) == 4:
            v1 = self.vertices[1] - self.vertices[0]
            v2 = self.vertices[3] - self.vertices[0]
            n = np.cross(v1, v2)
        else:
            return np.zeros(3)
        norm = np.linalg.norm(n)
        return n / norm if norm > 1e-12 else np.zeros(3)


class Mesh:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices = np.array(vertices, dtype=np.float64)
        self.faces = np.array(faces, dtype=np.int32)
        self.elements: List[Element] = []
        self._build_elements()
        self._compute_centroids()
        self._compute_normals()
        self._compute_areas()

    def _build_elements(self):
        self.elements = []
        for i, face in enumerate(self.faces):
            verts = self.vertices[face]
            self.elements.append(Element(vertices=verts, index=i))

    def _compute_centroids(self):
        self.centroids = np.array([elem.centroid for elem in self.elements])

    def _compute_normals(self):
        self.normals = np.array([elem.normal for elem in self.elements])

    def _compute_areas(self):
        self.areas = np.array([elem.area for elem in self.elements])

    @property
    def num_elements(self) -> int:
        return len(self.elements)

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    def get_collocation_points(self) -> np.ndarray:
        return self.centroids.copy()

    def get_element(self, index: int) -> Element:
        return self.elements[index]


def generate_sphere_mesh(radius: float = 1.0,
                         n_theta: int = 20,
                         n_phi: int = 40) -> Mesh:
    vertices = []
    faces = []

    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    for i, t in enumerate(theta):
        for j, p in enumerate(phi):
            x = radius * np.sin(t) * np.cos(p)
            y = radius * np.sin(t) * np.sin(p)
            z = radius * np.cos(t)
            vertices.append([x, y, z])

    vertices = np.array(vertices)

    for i in range(n_theta - 1):
        for j in range(n_phi):
            v1 = i * n_phi + j
            v2 = i * n_phi + (j + 1) % n_phi
            v3 = (i + 1) * n_phi + (j + 1) % n_phi
            v4 = (i + 1) * n_phi + j
            faces.append([v1, v2, v3, v4])

    return Mesh(vertices, np.array(faces))


def generate_cube_mesh(size: float = 1.0, n_per_side: int = 10) -> Mesh:
    vertices = []
    faces = []
    half = size / 2

    grid = np.linspace(-half, half, n_per_side)
    X, Y = np.meshgrid(grid, grid)

    for face_idx in range(6):
        face_verts = []
        for i in range(n_per_side):
            for j in range(n_per_side):
                x, y = X[i, j], Y[i, j]
                if face_idx == 0:
                    face_verts.append([x, y, half])
                elif face_idx == 1:
                    face_verts.append([x, y, -half])
                elif face_idx == 2:
                    face_verts.append([x, half, y])
                elif face_idx == 3:
                    face_verts.append([x, -half, y])
                elif face_idx == 4:
                    face_verts.append([half, x, y])
                elif face_idx == 5:
                    face_verts.append([-half, x, y])

        start_idx = len(vertices)
        vertices.extend(face_verts)

        for i in range(n_per_side - 1):
            for j in range(n_per_side - 1):
                v1 = start_idx + i * n_per_side + j
                v2 = start_idx + i * n_per_side + j + 1
                v3 = start_idx + (i + 1) * n_per_side + j + 1
                v4 = start_idx + (i + 1) * n_per_side + j
                faces.append([v1, v2, v3, v4])

    return Mesh(np.array(vertices), np.array(faces))


def generate_cylinder_mesh(radius: float = 1.0,
                           height: float = 2.0,
                           n_theta: int = 40,
                           n_height: int = 20) -> Mesh:
    vertices = []
    faces = []

    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    z = np.linspace(-height / 2, height / 2, n_height)

    for zi in z:
        for t in theta:
            x = radius * np.cos(t)
            y = radius * np.sin(t)
            vertices.append([x, y, zi])

    for i in range(n_height - 1):
        for j in range(n_theta):
            v1 = i * n_theta + j
            v2 = i * n_theta + (j + 1) % n_theta
            v3 = (i + 1) * n_theta + (j + 1) % n_theta
            v4 = (i + 1) * n_theta + j
            faces.append([v1, v2, v3, v4])

    top_center = len(vertices)
    vertices.append([0, 0, height / 2])
    for j in range(n_theta):
        v1 = (n_height - 1) * n_theta + j
        v2 = (n_height - 1) * n_theta + (j + 1) % n_theta
        faces.append([v1, v2, top_center])

    bottom_center = len(vertices)
    vertices.append([0, 0, -height / 2])
    for j in range(n_theta):
        v1 = j
        v2 = (j + 1) % n_theta
        faces.append([v2, v1, bottom_center])

    return Mesh(np.array(vertices), np.array(faces))


def refine_mesh(mesh: Mesh, times: int = 1) -> Mesh:
    for _ in range(times):
        new_vertices = list(mesh.vertices.copy())
        new_faces = []

        edge_midpoints = {}

        def get_midpoint(v1: int, v2: int) -> int:
            key = tuple(sorted([v1, v2]))
            if key not in edge_midpoints:
                p1 = mesh.vertices[v1]
                p2 = mesh.vertices[v2]
                mid = (p1 + p2) / 2
                edge_midpoints[key] = len(new_vertices)
                new_vertices.append(mid)
            return edge_midpoints[key]

        for face in mesh.faces:
            if len(face) == 3:
                v0, v1, v2 = face
                m01 = get_midpoint(v0, v1)
                m12 = get_midpoint(v1, v2)
                m20 = get_midpoint(v2, v0)
                new_faces.append([v0, m01, m20])
                new_faces.append([v1, m12, m01])
                new_faces.append([v2, m20, m12])
                new_faces.append([m01, m12, m20])
            elif len(face) == 4:
                v0, v1, v2, v3 = face
                m01 = get_midpoint(v0, v1)
                m12 = get_midpoint(v1, v2)
                m23 = get_midpoint(v2, v3)
                m30 = get_midpoint(v3, v0)
                center = len(new_vertices)
                new_vertices.append(np.mean([mesh.vertices[v] for v in face], axis=0))
                new_faces.append([v0, m01, center, m30])
                new_faces.append([v1, m12, center, m01])
                new_faces.append([v2, m23, center, m12])
                new_faces.append([v3, m30, center, m23])

        mesh = Mesh(np.array(new_vertices), np.array(new_faces))

    return mesh
