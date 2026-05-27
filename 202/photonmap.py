import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

from mathlib import Vec3, Color, Ray, dot, normalize, random_unit_vector, length
from geometry import Hittable, HitRecord, Sphere
from material import Material, DiffuseLight, Dielectric, Metal, Lambertian


@dataclass
class Photon:
    position: Vec3
    direction: Vec3
    power: Color
    normal: Vec3


@dataclass
class PhotonMap:
    photons: List[Photon] = field(default_factory=list)
    max_photons: int = 500000

    def add(self, photon: Photon):
        if len(self.photons) < self.max_photons:
            self.photons.append(photon)

    def __len__(self):
        return len(self.photons)

    def query(self, point: Vec3, normal: Vec3, max_dist: float, max_count: int) -> List[Photon]:
        candidates = []

        for photon in self.photons:
            dist = length(photon.position - point)
            if dist < max_dist:
                cosine = dot(photon.direction, -normal)
                if cosine > 0:
                    candidates.append((dist, photon))

        candidates.sort(key=lambda x: x[0])
        return [p for _, p in candidates[:max_count]]


class PhotonTracer:
    def __init__(
        self,
        world: Hittable,
        lights: List[Sphere],
        max_photon_bounces: int = 10,
        caustic_only: bool = False
    ):
        self.world = world
        self.lights = lights
        self.max_photon_bounces = max_photon_bounces
        self.caustic_only = caustic_only

    def _sample_light_position(self, light: Sphere) -> tuple:
        theta = random.uniform(0, 2 * math.pi)
        phi = random.uniform(0, math.pi)
        sin_phi = math.sin(phi)

        x = light.radius * sin_phi * math.cos(theta)
        y = light.radius * sin_phi * math.sin(phi)
        z = light.radius * math.cos(phi)

        pos = Vec3(
            light.center.x + x,
            light.center.y + y,
            light.center.z + z
        )
        normal = normalize(pos - light.center)
        return pos, normal

    def _sample_direction_on_hemisphere(self, normal: Vec3) -> Vec3:
        r1 = random.random()
        r2 = random.random()

        sin_theta = math.sqrt(max(0.0, 1.0 - r1 * r1))
        phi = 2 * math.pi * r2

        x = sin_theta * math.cos(phi)
        y = sin_theta * math.sin(phi)
        z = r1

        tangent = Vec3(1, 0, 0)
        if abs(normal.x) < 0.9:
            tangent = Vec3(1, 0, 0)
        else:
            tangent = Vec3(0, 1, 0)

        bitangent = normalize(Vec3(
            normal.y * tangent.z - normal.z * tangent.y,
            normal.z * tangent.x - normal.x * tangent.z,
            normal.x * tangent.y - normal.y * tangent.x
        ))
        tangent = Vec3(
            bitangent.y * normal.z - bitangent.z * normal.y,
            bitangent.z * normal.x - bitangent.x * normal.z,
            bitangent.x * normal.y - bitangent.y * normal.x
        )

        return normalize(
            tangent * x + bitangent * y + normal * z
        )

    def trace_photon(
        self,
        ray: Ray,
        power: Color,
        depth: int = 0,
        has_specular_bounce: bool = False
    ) -> Optional[Photon]:
        if depth >= self.max_photon_bounces:
            return None

        rec = self.world.hit(ray, 0.001, float('inf'))
        if rec is None:
            return None

        if isinstance(rec.material, DiffuseLight):
            return None

        is_specular = isinstance(rec.material, (Dielectric, Metal))
        is_diffuse = isinstance(rec.material, Lambertian)

        if is_diffuse and (not self.caustic_only or has_specular_bounce):
            return Photon(
                position=rec.point,
                direction=ray.direction,
                power=power,
                normal=rec.normal
            )

        scatter = rec.material.scatter(ray, rec)
        if scatter is None:
            return None

        new_has_specular = has_specular_bounce or is_specular

        if is_diffuse and not self.caustic_only:
            if random.random() < 0.5:
                return Photon(
                    position=rec.point,
                    direction=ray.direction,
                    power=power * 2.0,
                    normal=rec.normal
                )
            else:
                new_power = power * 2.0
        else:
            new_power = power

        return self.trace_photon(
            scatter.scattered,
            new_power * scatter.attenuation,
            depth + 1,
            new_has_specular
        )

    def generate_photons(self, num_photons: int, caustic_map: bool = False) -> PhotonMap:
        photon_map = PhotonMap(max_photons=num_photons)

        if not self.lights:
            return photon_map

        total_light_power = 0.0
        for light in self.lights:
            emit = light.material.emit
            total_light_power += emit.r + emit.g + emit.b

        if total_light_power < 1e-8:
            return photon_map

        photons_emitted = 0
        while len(photon_map) < num_photons and photons_emitted < num_photons * 10:
            light = random.choice(self.lights)

            light_power = light.material.emit.r + light.material.emit.g + light.material.emit.b
            probability = light_power / total_light_power

            if random.random() > probability:
                continue

            pos, normal = self._sample_light_position(light)
            direction = self._sample_direction_on_hemisphere(normal)

            ray = Ray(pos, direction)

            power = light.material.emit / probability

            if caustic_map:
                self.caustic_only = True
                has_specular = False
            else:
                has_specular = False

            rec = self.world.hit(ray, 0.001, float('inf'))
            if rec is None:
                photons_emitted += 1
                continue

            if isinstance(rec.material, DiffuseLight):
                photons_emitted += 1
                continue

            is_specular = isinstance(rec.material, (Dielectric, Metal))
            new_has_specular = has_specular or is_specular

            scatter = rec.material.scatter(ray, rec)
            if scatter is None:
                photons_emitted += 1
                continue

            if isinstance(rec.material, Lambertian):
                if not caustic_map or new_has_specular:
                    cosine = dot(rec.normal, normalize(scatter.scattered.direction))
                    if cosine > 0:
                        photon = Photon(
                            position=rec.point,
                            direction=ray.direction,
                            power=power,
                            normal=rec.normal
                        )
                        photon_map.add(photon)

            photon = self.trace_photon(
                scatter.scattered,
                power * scatter.attenuation,
                depth=1,
                has_specular_bounce=new_has_specular
            )
            if photon is not None:
                photon_map.add(photon)

            photons_emitted += 1

        return photon_map


def estimate_radiance_with_photons(
    hit_point: Vec3,
    hit_normal: Vec3,
    global_map: PhotonMap,
    caustic_map: Optional[PhotonMap] = None,
    search_radius: float = 0.5,
    max_photons: int = 100
) -> Color:
    radiance = Color(0, 0, 0)

    if caustic_map is not None and len(caustic_map) > 0:
        caustic_photons = caustic_map.query(hit_point, hit_normal, search_radius * 0.5, max_photons // 2)
        if caustic_photons:
            area = math.pi * (search_radius * 0.5) ** 2
            for photon in caustic_photons:
                radiance = radiance + photon.power / area

    if len(global_map) > 0:
        global_photons = global_map.query(hit_point, hit_normal, search_radius, max_photons)
        if global_photons:
            area = math.pi * search_radius ** 2
            for photon in global_photons:
                radiance = radiance + photon.power / area

    return radiance / math.pi
