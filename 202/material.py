import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple

from mathlib import Vec3, Color, Ray, dot, reflect, refract, random_unit_vector, normalize
from geometry import HitRecord


@dataclass
class ScatterRecord:
    attenuation: Color
    scattered: Ray
    pdf: float = 1.0


class Material:
    def scatter(self, ray_in: Ray, rec: HitRecord) -> Optional[ScatterRecord]:
        return None

    def emitted(self, rec: HitRecord, u: float, v: float, point: Vec3) -> Color:
        return Color(0, 0, 0)

    def scattering_pdf(self, ray_in: Ray, rec: HitRecord, scattered: Ray) -> float:
        return 0.0


class Lambertian(Material):
    def __init__(self, albedo: Color):
        self.albedo = albedo

    def scatter(self, ray_in: Ray, rec: HitRecord) -> Optional[ScatterRecord]:
        scatter_direction = rec.normal + random_unit_vector()
        if dot(scatter_direction, scatter_direction) < 1e-8:
            scatter_direction = rec.normal

        scattered = Ray(rec.point, scatter_direction, ray_in.time)
        pdf = dot(rec.normal, normalize(scattered.direction)) / math.pi
        return ScatterRecord(self.albedo, scattered, pdf)

    def scattering_pdf(self, ray_in: Ray, rec: HitRecord, scattered: Ray) -> float:
        cosine = dot(rec.normal, normalize(scattered.direction))
        return max(0.0, cosine / math.pi)


class Metal(Material):
    def __init__(self, albedo: Color, fuzz: float = 0.0):
        self.albedo = albedo
        self.fuzz = min(1.0, max(0.0, fuzz))

    def scatter(self, ray_in: Ray, rec: HitRecord) -> Optional[ScatterRecord]:
        reflected = reflect(normalize(ray_in.direction), rec.normal)
        reflected = reflected + random_unit_vector() * self.fuzz
        scattered = Ray(rec.point, reflected, ray_in.time)

        if dot(scattered.direction, rec.normal) <= 0:
            return None

        return ScatterRecord(self.albedo, scattered, 1.0)


class Dielectric(Material):
    def __init__(self, ir: float):
        self.ir = ir

    def scatter(self, ray_in: Ray, rec: HitRecord) -> Optional[ScatterRecord]:
        attenuation = Color(1.0, 1.0, 1.0)
        refraction_ratio = (1.0 / self.ir) if rec.front_face else self.ir

        unit_direction = normalize(ray_in.direction)
        cos_theta = min(dot(-unit_direction, rec.normal), 1.0)
        sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)

        cannot_refract = refraction_ratio * sin_theta > 1.0

        if cannot_refract or self._reflectance(cos_theta, refraction_ratio) > random.random():
            direction = reflect(unit_direction, rec.normal)
        else:
            direction = refract(unit_direction, rec.normal, refraction_ratio)

        scattered = Ray(rec.point, direction, ray_in.time)
        return ScatterRecord(attenuation, scattered, 1.0)

    def _reflectance(self, cosine: float, ref_idx: float) -> float:
        r0 = (1 - ref_idx) / (1 + ref_idx)
        r0 = r0 * r0
        return r0 + (1 - r0) * math.pow(1 - cosine, 5)


class DiffuseLight(Material):
    def __init__(self, emit: Color):
        self.emit = emit

    def scatter(self, ray_in: Ray, rec: HitRecord) -> Optional[ScatterRecord]:
        return None

    def emitted(self, rec: HitRecord, u: float, v: float, point: Vec3) -> Color:
        if rec.front_face:
            return self.emit
        return Color(0, 0, 0)


class Isotropic(Material):
    def __init__(self, albedo: Color):
        self.albedo = albedo

    def scatter(self, ray_in: Ray, rec: HitRecord) -> Optional[ScatterRecord]:
        scattered = Ray(rec.point, random_unit_vector(), ray_in.time)
        return ScatterRecord(self.albedo, scattered, 1.0 / (4.0 * math.pi))

    def scattering_pdf(self, ray_in: Ray, rec: HitRecord, scattered: Ray) -> float:
        return 1.0 / (4.0 * math.pi)
