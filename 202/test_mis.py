import sys
import random
import time

from mathlib import Vec3, Color
from geometry import HittableList, Sphere, Plane
from material import Lambertian, Metal, Dielectric, DiffuseLight
from camera import Camera
from pathtracer import PathTracer
from renderer import Renderer


def test_mis_comparison():
    print("Testing Multiple Importance Sampling (MIS)")
    print("=" * 60)
    print()

    random.seed(42)

    world = HittableList()
    lights = HittableList()

    ground = Lambertian(Color(0.8, 0.8, 0.8))
    red = Lambertian(Color(0.9, 0.1, 0.1))
    blue = Lambertian(Color(0.1, 0.1, 0.9))
    white = Lambertian(Color(0.9, 0.9, 0.9))

    world.add(Plane(Vec3(0, -1, 0), Vec3(0, 1, 0), ground))

    small_light = DiffuseLight(Color(30, 30, 30))
    light_sphere = Sphere(Vec3(0, 5, -2), 0.3, small_light)
    world.add(light_sphere)
    lights.add(light_sphere)

    world.add(Sphere(Vec3(-2, 0, 0), 1.0, red))
    world.add(Sphere(Vec3(0, 0, 0), 1.0, white))
    world.add(Sphere(Vec3(2, 0, 0), 1.0, blue))

    camera = Camera(
        lookfrom=Vec3(0, 2, -8),
        lookat=Vec3(0, 0, 0),
        vup=Vec3(0, 1, 0),
        vfov=45,
        aspect_ratio=4.0 / 3.0,
        aperture=0.0,
        focus_dist=8.0
    )

    background = Color(0.1, 0.1, 0.15)

    print("Scene: Small bright light source (good for showing fireflies)")
    print("-" * 60)

    renderer = Renderer(
        width=300,
        height=225,
        samples_per_pixel=16,
        max_depth=50
    )

    print(f"\nResolution: {renderer.width}x{renderer.height}")
    print(f"Samples per pixel: {renderer.samples_per_pixel}")
    print()

    print("Rendering WITH MIS...")
    start = time.time()

    pathtracer_mis = PathTracer(world, renderer.max_depth, background, lights)

    pixels_mis = []
    for j in range(renderer.height - 1, -1, -1):
        print(f"  Scanline {renderer.height - j}/{renderer.height}", end='\r', flush=True)
        row = []
        for i in range(renderer.width):
            pixel_color = Color(0, 0, 0)
            for _ in range(renderer.samples_per_pixel):
                u = (i + random.random()) / (renderer.width - 1)
                v = (j + random.random()) / (renderer.height - 1)
                ray = camera.get_ray(u, v)
                pixel_color = pixel_color + pathtracer_mis.trace_ray(ray)
            row.append(pixel_color.to_rgb(renderer.samples_per_pixel))
        pixels_mis.append(row)

    time_mis = time.time() - start
    print(f"\n  Completed in {time_mis:.2f} seconds")
    renderer.save_png(pixels_mis, "test_mis_enabled.png")

    print()
    print("Rendering WITHOUT MIS (simple sampling)...")
    start = time.time()

    class SimplePathTracer(PathTracer):
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

            scattered = scatter_result.scattered
            attenuation = scatter_result.attenuation
            pdf = scatter_result.pdf

            if pdf > 1e-8:
                incoming = self.trace_ray(scattered, depth + 1)
                cos_term = max(0.0, dot(rec.normal, normalize(scattered.direction)))
                return emitted + attenuation * incoming * cos_term / pdf
            else:
                return emitted

    pathtracer_simple = SimplePathTracer(world, renderer.max_depth, background, lights)

    pixels_simple = []
    for j in range(renderer.height - 1, -1, -1):
        print(f"  Scanline {renderer.height - j}/{renderer.height}", end='\r', flush=True)
        row = []
        for i in range(renderer.width):
            pixel_color = Color(0, 0, 0)
            for _ in range(renderer.samples_per_pixel):
                u = (i + random.random()) / (renderer.width - 1)
                v = (j + random.random()) / (renderer.height - 1)
                ray = camera.get_ray(u, v)
                pixel_color = pixel_color + pathtracer_simple.trace_ray(ray)
            row.append(pixel_color.to_rgb(renderer.samples_per_pixel))
        pixels_simple.append(row)

    time_simple = time.time() - start
    print(f"\n  Completed in {time_simple:.2f} seconds")
    renderer.save_png(pixels_simple, "test_mis_disabled.png")

    print()
    print("=" * 60)
    print("Comparison Summary:")
    print(f"  With MIS:    {time_mis:.2f}s -> test_mis_enabled.png")
    print(f"  Without MIS: {time_simple:.2f}s -> test_mis_disabled.png")
    print()
    print("Note: The 'without MIS' version may show bright firefly noise")
    print("especially near the small bright light source.")
    print("=" * 60)

    return True


if __name__ == "__main__":
    from mathlib import dot, normalize

    try:
        success = test_mis_comparison()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
