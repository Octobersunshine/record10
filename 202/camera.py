import math
import random

from mathlib import Vec3, Ray, random_in_unit_disk, normalize


class Camera:
    def __init__(
        self,
        lookfrom: Vec3,
        lookat: Vec3,
        vup: Vec3,
        vfov: float,
        aspect_ratio: float,
        aperture: float = 0.0,
        focus_dist: float = 1.0
    ):
        theta = math.radians(vfov)
        h = math.tan(theta / 2)
        viewport_height = 2.0 * h
        viewport_width = aspect_ratio * viewport_height

        self.w = normalize(lookfrom - lookat)
        self.u = normalize(Vec3(
            vup.y * self.w.z - vup.z * self.w.y,
            vup.z * self.w.x - vup.x * self.w.z,
            vup.x * self.w.y - vup.y * self.w.x
        ))
        self.v = Vec3(
            self.w.y * self.u.z - self.w.z * self.u.y,
            self.w.z * self.u.x - self.w.x * self.u.z,
            self.w.x * self.u.y - self.w.y * self.u.x
        )

        self.origin = lookfrom
        self.horizontal = self.u * (viewport_width * focus_dist)
        self.vertical = self.v * (viewport_height * focus_dist)
        self.lower_left_corner = (
            self.origin
            - self.horizontal / 2
            - self.vertical / 2
            - self.w * focus_dist
        )

        self.lens_radius = aperture / 2.0

    def get_ray(self, s: float, t: float) -> Ray:
        rd = random_in_unit_disk() * self.lens_radius
        offset = self.u * rd.x + self.v * rd.y

        return Ray(
            self.origin + offset,
            (
                self.lower_left_corner
                + self.horizontal * s
                + self.vertical * t
                - self.origin
                - offset
            )
        )
