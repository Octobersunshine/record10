import math
from dataclasses import dataclass
from typing import Optional

from mathlib import Vec3, Ray, dot, normalize


@dataclass
class HitRecord:
    point: Vec3
    normal: Vec3
    t: float
    material: object
    front_face: bool = True

    def set_face_normal(self, ray: Ray, outward_normal: Vec3):
        self.front_face = dot(ray.direction, outward_normal) < 0
        self.normal = outward_normal if self.front_face else -outward_normal


class Hittable:
    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        raise NotImplementedError


class Sphere(Hittable):
    def __init__(self, center: Vec3, radius: float, material: object):
        self.center = center
        self.radius = max(0.0, radius)
        self.material = material

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        oc = ray.origin - self.center
        a = dot(ray.direction, ray.direction)
        half_b = dot(oc, ray.direction)
        c = dot(oc, oc) - self.radius * self.radius

        discriminant = half_b * half_b - a * c
        if discriminant < 0:
            return None

        sqrtd = math.sqrt(discriminant)

        root = (-half_b - sqrtd) / a
        if root < t_min or t_max < root:
            root = (-half_b + sqrtd) / a
            if root < t_min or t_max < root:
                return None

        t = root
        point = ray.at(t)
        outward_normal = (point - self.center) / self.radius

        rec = HitRecord(point, outward_normal, t, self.material)
        rec.set_face_normal(ray, outward_normal)
        return rec


class Plane(Hittable):
    def __init__(self, point: Vec3, normal: Vec3, material: object):
        self.point = point
        self.normal = normalize(normal)
        self.material = material

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        denom = dot(self.normal, ray.direction)
        if abs(denom) < 1e-8:
            return None

        t = dot(self.point - ray.origin, self.normal) / denom
        if t < t_min or t > t_max:
            return None

        point = ray.at(t)
        rec = HitRecord(point, self.normal, t, self.material)
        rec.set_face_normal(ray, self.normal)
        return rec


class Triangle(Hittable):
    def __init__(self, v0: Vec3, v1: Vec3, v2: Vec3, material: object):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.material = material

        edge1 = v1 - v0
        edge2 = v2 - v0
        self.normal = normalize(Vec3(
            edge1.y * edge2.z - edge1.z * edge2.y,
            edge1.z * edge2.x - edge1.x * edge2.z,
            edge1.x * edge2.y - edge1.y * edge2.x
        ))

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        edge1 = self.v1 - self.v0
        edge2 = self.v2 - self.v0
        h = Vec3(
            ray.direction.y * edge2.z - ray.direction.z * edge2.y,
            ray.direction.z * edge2.x - ray.direction.x * edge2.z,
            ray.direction.x * edge2.y - ray.direction.y * edge2.x
        )
        a = dot(edge1, h)

        if abs(a) < 1e-8:
            return None

        f = 1.0 / a
        s = ray.origin - self.v0
        u = f * dot(s, h)

        if u < 0.0 or u > 1.0:
            return None

        q = Vec3(
            s.y * edge1.z - s.z * edge1.y,
            s.z * edge1.x - s.x * edge1.z,
            s.x * edge1.y - s.y * edge1.x
        )
        v = f * dot(ray.direction, q)

        if v < 0.0 or u + v > 1.0:
            return None

        t = f * dot(edge2, q)

        if t < t_min or t > t_max:
            return None

        point = ray.at(t)
        rec = HitRecord(point, self.normal, t, self.material)
        rec.set_face_normal(ray, self.normal)
        return rec


class HittableList(Hittable):
    def __init__(self):
        self.objects: list[Hittable] = []

    def add(self, obj: Hittable):
        self.objects.append(obj)

    def clear(self):
        self.objects.clear()

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        closest = None
        closest_t = t_max

        for obj in self.objects:
            hit = obj.hit(ray, t_min, closest_t)
            if hit is not None:
                closest = hit
                closest_t = hit.t

        return closest
