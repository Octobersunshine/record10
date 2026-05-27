import math
import random


class Vec3:
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
        return Vec3(self.x + other, self.y + other, self.z + other)

    def __sub__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
        return Vec3(self.x - other, self.y - other, self.z - other)

    def __mul__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)
        return Vec3(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.x / other.x, self.y / other.y, self.z / other.z)
        return Vec3(self.x / other, self.y / other, self.z / other)

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        raise IndexError("Vec3 index out of range")

    def __repr__(self):
        return f"Vec3({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"


def dot(a: Vec3, b: Vec3) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z


def cross(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    )


def length(v: Vec3) -> float:
    return math.sqrt(dot(v, v))


def normalize(v: Vec3) -> Vec3:
    l = length(v)
    if l < 1e-8:
        return Vec3(0, 0, 0)
    return v / l


def reflect(v: Vec3, n: Vec3) -> Vec3:
    return v - n * (2.0 * dot(v, n))


def refract(uv: Vec3, n: Vec3, etai_over_etat: float) -> Vec3:
    cos_theta = min(dot(-uv, n), 1.0)
    r_out_perp = (uv + n * cos_theta) * etai_over_etat
    r_out_parallel = n * (-math.sqrt(abs(1.0 - dot(r_out_perp, r_out_perp))))
    return r_out_perp + r_out_parallel


def random_unit_vector() -> Vec3:
    while True:
        p = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
        len_sq = dot(p, p)
        if 1e-16 < len_sq <= 1.0:
            return p / math.sqrt(len_sq)


def random_on_hemisphere(normal: Vec3) -> Vec3:
    on_unit_sphere = random_unit_vector()
    if dot(on_unit_sphere, normal) > 0.0:
        return on_unit_sphere
    else:
        return -on_unit_sphere


def random_in_unit_disk() -> Vec3:
    while True:
        p = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), 0)
        if dot(p, p) < 1:
            return p


def clamp(x: float, min_val: float, max_val: float) -> float:
    if x < min_val:
        return min_val
    if x > max_val:
        return max_val
    return x


class Color(Vec3):
    def __init__(self, r=0.0, g=0.0, b=0.0):
        super().__init__(r, g, b)

    @property
    def r(self):
        return self.x

    @property
    def g(self):
        return self.y

    @property
    def b(self):
        return self.z

    def __mul__(self, other):
        if isinstance(other, Color):
            return Color(self.r * other.r, self.g * other.g, self.b * other.b)
        return Color(self.r * other, self.g * other, self.b * other)

    def __add__(self, other):
        if isinstance(other, Color):
            return Color(self.r + other.r, self.g + other.g, self.b + other.b)
        return Color(self.r + other, self.g + other, self.b + other)

    def __truediv__(self, other):
        if isinstance(other, Color):
            return Color(self.r / other.r, self.g / other.g, self.b / other.b)
        return Color(self.r / other, self.g / other, self.b / other)

    def to_rgb(self, samples_per_pixel: int) -> tuple:
        scale = 1.0 / samples_per_pixel
        r = math.sqrt(self.r * scale)
        g = math.sqrt(self.g * scale)
        b = math.sqrt(self.b * scale)
        return (
            int(256 * clamp(r, 0.0, 0.999)),
            int(256 * clamp(g, 0.0, 0.999)),
            int(256 * clamp(b, 0.0, 0.999))
        )


class Ray:
    __slots__ = ('origin', 'direction', 'time')

    def __init__(self, origin: Vec3, direction: Vec3, time: float = 0.0):
        self.origin = origin
        self.direction = normalize(direction)
        self.time = time

    def at(self, t: float) -> Vec3:
        return self.origin + self.direction * t
