import sys
import random
import time
from mathlib import Vec3, Color, dot, normalize
from geometry import HittableList, Sphere, Plane
from material import Lambertian, Metal, Dielectric, DiffuseLight
from camera import Camera
from pathtracer import PathTracer
from photonmap import PhotonTracer, estimate_radiance_with_photons, PhotonMap
from renderer import Renderer


def create_caustic_scene():
    print("Creating caustics scene...")
    print("=" * 60)
    print()

    world = HittableList()
    lights = HittableList()

    ground = Lambertian(Color(0.3, 0.3, 0.35))
    world.add(Plane(Vec3(0, -1, 0), Vec3(0, 1, 0), ground))

    back_wall = Lambertian(Color(0.5, 0.5, 0.55))
    world.add(Plane(Vec3(0, 0, 10), Vec3(0, 0, -1), back_wall))

    left_wall = Lambertian(Color(0.6, 0.2, 0.2))
    world.add(Plane(Vec3(-6, 0, 0), Vec3(1, 0, 0), left_wall))

    right_wall = Lambertian(Color(0.2, 0.2, 0.6))
    world.add(Plane(Vec3(6, 0, 0), Vec3(-1, 0, 0), right_wall))

    ceiling = Lambertian(Color(0.7, 0.7, 0.7))
    world.add(Plane(Vec3(0, 6, 0), Vec3(0, -1, 0), ceiling))

    light = DiffuseLight(Color(20, 20, 20))
    light_sphere = Sphere(Vec3(0, 4.5, 2), 0.8, light)
    world.add(light_sphere)
    lights.add(light_sphere)

    glass = Dielectric(1.5)
    glass_sphere = Sphere(Vec3(0, 0.5, 2), 1.2, glass)
    world.add(glass_sphere)

    metal_sphere = Sphere(Vec3(-2.5, 0.5, 3), 1.0, Metal(Color(0.9, 0.9, 0.9), 0.1))
    world.add(metal_sphere)

    red_sphere = Sphere(Vec3(2.5, 0.5, 1.5), 1.0, Lambertian(Color(0.8, 0.2, 0.2)))
    world.add(red_sphere)

    camera = Camera(
        lookfrom=Vec3(0, 2, -6),
        lookat=Vec3(0, 1, 2),
        vup=Vec3(0, 1, 0),
        vfov=50,
        aspect_ratio=4.0 / 3.0,
        aperture=0.0,
        focus_dist=8.0
    )

    background = Color(0.1, 0.1, 0.15)

    return world, camera, background, lights


def render_with_photon_mapping():
    print("Rendering with Photon Mapping for Caustics")
    print("=" * 60)
    print()

    random.seed(42)

    world, camera, background, lights = create_caustic_scene()

    width = 400
    height = 300
    samples_per_pixel = 20
    max_depth = 50

    print(f"Resolution: {width}x{height}")
    print(f"Samples per pixel: {samples_per_pixel}")
    print(f"Max depth: {max_depth}")
    print()

    light_list = []
    if hasattr(lights, 'objects'):
        light_list = lights.objects

    print("Generating global photon map...")
    start = time.time()
    photon_tracer = PhotonTracer(world, light_list, max_photon_bounces=8)
    global_photon_map = photon_tracer.generate_photons(20000, caustic_map=False)
    print(f"  Generated {len(global_photon_map)} global photons in {time.time() - start:.2f}s")
    print()

    print("Generating caustic photon map...")
    start = time.time()
    caustic_tracer = PhotonTracer(world, light_list, max_photon_bounces=10, caustic_only=True)
    caustic_photon_map = caustic_tracer.generate_photons(10000, caustic_map=True)
    print(f"  Generated {len(caustic_photon_map)} caustic photons in {time.time() - start:.2f}s")
    print()

    pathtracer = PathTracer(world, max_depth, background, lights)

    print("Rendering image (path tracing + photon mapping)...")
    start = time.time()
    pixels = []

    for j in range(height - 1, -1, -1):
        print(f"  Scanline {height - j}/{height}", end='\r', flush=True)
        row = []
        for i in range(width):
            pixel_color = Color(0, 0, 0)
            for _ in range(samples_per_pixel):
                u = (i + random.random()) / (width - 1)
                v = (j + random.random()) / (height - 1)
                ray = camera.get_ray(u, v)
                pixel_color = pixel_color + _trace_ray_with_photons(
                    ray, pathtracer, world,
                    global_photon_map, caustic_photon_map,
                    max_depth, 0
                )
            row.append(pixel_color.to_rgb(samples_per_pixel))
        pixels.append(row)

    elapsed = time.time() - start
    print(f"\n  Rendering completed in {elapsed:.2f} seconds")

    output_file = "caustics_photon_map.png"
    renderer = Renderer(width, height, samples_per_pixel, max_depth)
    renderer.save_png(pixels, output_file)
    print(f"  Saved: {output_file}")
    print()

    print("Rendering without photon mapping (for comparison)...")
    start = time.time()
    pixels_no_pm = []

    for j in range(height - 1, -1, -1):
        print(f"  Scanline {height - j}/{height}", end='\r', flush=True)
        row = []
        for i in range(width):
            pixel_color = Color(0, 0, 0)
            for _ in range(samples_per_pixel):
                u = (i + random.random()) / (width - 1)
                v = (j + random.random()) / (height - 1)
                ray = camera.get_ray(u, v)
                pixel_color = pixel_color + pathtracer.trace_ray(ray)
            row.append(pixel_color.to_rgb(samples_per_pixel))
        pixels_no_pm.append(row)

    elapsed = time.time() - start
    print(f"\n  Rendering completed in {elapsed:.2f} seconds")

    output_file_no_pm = "caustics_no_photon_map.png"
    renderer.save_png(pixels_no_pm, output_file_no_pm)
    print(f"  Saved: {output_file_no_pm}")
    print()

    print("=" * 60)
    print("Comparison:")
    print(f"  With Photon Mapping:    {output_file}")
    print(f"  Without Photon Mapping: {output_file_no_pm}")
    print()
    print("The photon mapping version should show clearer caustic")
    print("patterns under the glass and metal spheres.")
    print("=" * 60)

    return True


def _trace_ray_with_photons(
    ray,
    pathtracer,
    world,
    global_map,
    caustic_map,
    max_depth,
    depth
):
    if depth >= max_depth:
        return Color(0, 0, 0)

    rec = world.hit(ray, 0.001, float('inf'))
    if rec is None:
        return pathtracer.background

    emitted = rec.material.emitted(rec, 0, 0, rec.point)
    scatter_result = rec.material.scatter(ray, rec)

    if scatter_result is None:
        return emitted

    if depth >= 2:
        photon_radiance = estimate_radiance_with_photons(
            rec.point, rec.normal,
            global_map, caustic_map,
            search_radius=0.3, max_photons=50
        )
        return emitted + scatter_result.attenuation * photon_radiance

    scattered = scatter_result.scattered
    attenuation = scatter_result.attenuation
    pdf = scatter_result.pdf

    if pdf > 1e-8:
        incoming = _trace_ray_with_photons(
            scattered, pathtracer, world,
            global_map, caustic_map,
            max_depth, depth + 1
        )
        cos_term = max(0.0, dot(rec.normal, normalize(scattered.direction)))
        return emitted + attenuation * incoming * cos_term / pdf
    else:
        return emitted


if __name__ == "__main__":
    try:
        success = render_with_photon_mapping()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
