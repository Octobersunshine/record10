import math
import random

from mathlib import Vec3, Color, Ray, dot, normalize, random_unit_vector, length
from geometry import Hittable, HitRecord, Sphere
from material import Material, ScatterRecord, Lambertian, DiffuseLight


def power_heuristic(nf: float, f_pdf: float, ng: float, g_pdf: float) -> float:
    f = nf * f_pdf
    g = ng * g_pdf
    return (f * f) / (f * f + g * g)


def balance_heuristic(nf: float, f_pdf: float, ng: float, g_pdf: float) -> float:
    return (nf * f_pdf) / (nf * f_pdf + ng * g_pdf)


class PathTracer:
    def __init__(
        self,
        world: Hittable,
        max_depth: int = 50,
        background: Color = None,
        lights: Hittable = None
    ):
        self.world = world
        self.max_depth = max_depth
        self.background = background if background is not None else Color(0, 0, 0)
        self.lights = lights
        self.light_objects = []
        self.light_surface_area = 0.0
        self._build_light_cache()

    def _build_light_cache(self):
        self.light_objects = []
        self.light_surface_area = 0.0

        if self.lights is None:
            return

        if hasattr(self.lights, 'objects'):
            for obj in self.lights.objects:
                if hasattr(obj, 'material') and isinstance(obj.material, DiffuseLight):
                    self.light_objects.append(obj)
                    if hasattr(obj, 'radius'):
                        self.light_surface_area += 4.0 * math.pi * obj.radius * obj.radius

    def _sample_light_position(self, light: Sphere) -> tuple:
        theta = random.uniform(0, 2 * math.pi)
        phi = random.uniform(0, math.pi)
        sin_phi = math.sin(phi)

        x = light.radius * sin_phi * math.cos(theta)
        y = light.radius * sin_phi * math.sin(theta)
        z = light.radius * math.cos(phi)

        return Vec3(
            light.center.x + x,
            light.center.y + y,
            light.center.z + z
        )

    def _light_sampling_pdf(self, light: Sphere, hit_point: Vec3, light_sample: Vec3, light_normal: Vec3) -> float:
        to_light = light_sample - hit_point
        distance = length(to_light)
        light_dir = to_light / distance

        cos_theta = abs(dot(light_normal, -light_dir))
        if cos_theta < 1e-8:
            return 0.0

        pdf_solid_angle = (distance * distance) / (cos_theta * 4.0 * math.pi * light.radius * light.radius)
        return pdf_solid_angle

    def trace_ray(self, ray: Ray, depth: int = 0) -> Color:
        if depth >= self.max_depth:
            return Color(0, 0, 0)

        rec = self.world.hit(ray, 0.001, float('inf'))
        if rec is None:
            return self.background

        emitted = rec.material.emitted(rec, 0, 0, rec.point)
        scatter_result = rec.material.scatter(ray, rec)

        if scatter_result is None:
            return emitted

        if self.light_objects and isinstance(rec.material, Lambertian):
            return self._trace_ray_with_mis(ray, rec, scatter_result, depth)
        else:
            return self._trace_ray_simple(ray, rec, scatter_result, emitted, depth)

    def _trace_ray_simple(
        self,
        ray: Ray,
        rec: HitRecord,
        scatter: ScatterRecord,
        emitted: Color,
        depth: int
    ) -> Color:
        scattered = scatter.scattered
        attenuation = scatter.attenuation
        pdf = scatter.pdf

        if pdf > 1e-8:
            incoming = self.trace_ray(scattered, depth + 1)
            cos_term = max(0.0, dot(rec.normal, normalize(scattered.direction)))
            return emitted + attenuation * incoming * cos_term / pdf
        else:
            return emitted

    def _trace_ray_with_mis(
        self,
        ray: Ray,
        rec: HitRecord,
        scatter: ScatterRecord,
        depth: int
    ) -> Color:
        emitted = rec.material.emitted(rec, 0, 0, rec.point)

        if not self.light_objects:
            scattered = scatter.scattered
            attenuation = scatter.attenuation
            pdf = scatter.pdf
            if pdf > 1e-8:
                incoming = self.trace_ray(scattered, depth + 1)
                cos_term = max(0.0, dot(rec.normal, normalize(scattered.direction)))
                return emitted + attenuation * incoming * cos_term / pdf
            return emitted

        light = random.choice(self.light_objects)

        light_sample = self._sample_light_position(light)
        to_light = light_sample - rec.point
        light_distance = length(to_light)
        light_dir = to_light / light_distance

        if dot(rec.normal, light_dir) <= 0:
            return self._trace_ray_simple(ray, rec, scatter, emitted, depth)

        shadow_ray = Ray(rec.point, light_dir, ray.time)
        shadow_hit = self.world.hit(shadow_ray, 0.001, light_distance - 0.001)

        light_sample_normal = (light_sample - light.center) / light.radius

        light_pdf = self._light_sampling_pdf(light, rec.point, light_sample, light_sample_normal)
        brdf_pdf = dot(rec.normal, light_dir) / math.pi

        if light_pdf < 1e-8 or brdf_pdf < 1e-8:
            return self._trace_ray_simple(ray, rec, scatter, emitted, depth)

        mis_weight = power_heuristic(1.0, light_pdf, 1.0, brdf_pdf)

        light_contribution = Color(0, 0, 0)
        if shadow_hit is None:
            cos_theta = abs(dot(light_sample_normal, -light_dir))
            geometric_term = cos_theta / (light_distance * light_distance)
            light_contribution = scatter.attenuation * light.material.emit * geometric_term / light_pdf
            light_contribution = light_contribution * mis_weight

        scattered = scatter.scattered
        attenuation = scatter.attenuation
        pdf = scatter.pdf

        if pdf > 1e-8:
            incoming = self.trace_ray(scattered, depth + 1)
            cos_term = max(0.0, dot(rec.normal, normalize(scattered.direction)))
            brdf_contribution = attenuation * incoming * cos_term / pdf

            scattered_dir = normalize(scattered.direction)
            to_light_dir = normalize(light_sample - rec.point)
            brdf_mis_weight = power_heuristic(1.0, pdf, 1.0, light_pdf)
            brdf_contribution = brdf_contribution * brdf_mis_weight
        else:
            brdf_contribution = Color(0, 0, 0)

        return emitted + light_contribution + brdf_contribution
