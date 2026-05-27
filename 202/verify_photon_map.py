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


def verify_photon_mapping():
    print("Verifying Photon Mapping Implementation")
    print("=" * 60)
    print()

    random.seed(42)

    world = HittableList()
    lights = HittableList()

    ground = Lambertian(Color(0.3, 0.3, 0.35))
    world.add(Plane(Vec3(0, -1, 0), Vec3(0, 1, 0), ground))

    light = DiffuseLight(Color(20, 20, 20))
    light_sphere = Sphere(Vec3(0, 3, 0), 0.5, light)
    world.add(light_sphere)
    lights.add(light_sphere)

    glass = Dielectric(1.5)
    glass_sphere = Sphere(Vec3(0, 0.5, 0), 1.0, glass)
    world.add(glass_sphere)

    camera = Camera(
        lookfrom=Vec3(0, 2, -5),
        lookat=Vec3(0, 0.5, 0),
        vup=Vec3(0, 1, 0),
        vfov=45,
        aspect_ratio=1.0,
        aperture=0.0,
        focus_dist=5.0
    )

    background = Color(0.1, 0.1, 0.15)

    light_list = lights.objects

    print("Step 1: Generating global photon map...")
    start = time.time()
    photon_tracer = PhotonTracer(world, light_list, max_photon_bounces=6)
    global_map = photon_tracer.generate_photons(5000, caustic_map=False)
    print(f"  Generated {len(global_map)} global photons in {time.time() - start:.2f}s")

    print("\nStep 2: Generating caustic photon map...")
    start = time.time()
    caustic_tracer = PhotonTracer(world, light_list, max_photon_bounces=8, caustic_only=True)
    caustic_map = caustic_tracer.generate_photons(3000, caustic_map=True)
    print(f"  Generated {len(caustic_map)} caustic photons in {time.time() - start:.2f}s")

    print("\nStep 3: Testing photon query...")
    test_point = Vec3(0, -0.9, 0)
    test_normal = Vec3(0, 1, 0)

    photons = global_map.query(test_point, test_normal, 2.0, 10)
    print(f"  Found {len(photons)} photons near test point")
    if photons:
        print(f"  Sample photon power: ({photons[0].power.r:.2f}, {photons[0].power.g:.2f}, {photons[0].power.b:.2f})")

    print("\nStep 4: Testing radiance estimation...")
    radiance = estimate_radiance_with_photons(
        test_point, test_normal,
        global_map, caustic_map,
        search_radius=1.0, max_photons=50
    )
    print(f"  Estimated radiance: ({radiance.r:.4f}, {radiance.g:.4f}, {radiance.b:.4f})")

    print("\nStep 5: Rendering small test image...")
    width = 150
    height = 150
    samples_per_pixel = 8
    max_depth = 30

    pathtracer = PathTracer(world, max_depth, background, lights)

    start = time.time()
    pixels_pm = []

    for j in range(height - 1, -1, -1):
        print(f"  Scanline {height - j}/{height}", end='\r', flush=True)
        row = []
        for i in range(width):
            pixel_color = Color(0, 0, 0)
            for _ in range(samples_per_pixel):
                u = (i + random.random()) / (width - 1)
                v = (j + random.random()) / (height - 1)
                ray = camera.get_ray(u, v)
                pixel_color = pixel_color + _trace_with_pm(
                    ray, pathtracer, world,
                    global_map, caustic_map,
                    max_depth, 0
                )
            row.append(pixel_color.to_rgb(samples_per_pixel))
        pixels_pm.append(row)

    print(f"\n  With PM: {time.time() - start:.2f}s")
    renderer = Renderer(width, height, samples_per_pixel, max_depth)
    renderer.save_png(pixels_pm, "photon_map_verify_pm.png")

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

    print(f"\n  Without PM: {time.time() - start:.2f}s")
    renderer.save_png(pixels_no_pm, "photon_map_verify_no_pm.png")

    print("\n" + "=" * 60)
    print("Verification Complete!")
    print("  photon_map_verify_pm.png    - with photon mapping")
    print("  photon_map_verify_no_pm.png - standard path tracing")
    print("=" * 60)

    return True


def _trace_with_pm(ray, pathtracer, world, global_map, caustic_map, max_depth, depth):
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
            search_radius=0.5, max_photons=30
        )
        return emitted + scatter_result.attenuation * photon_radiance

    scattered = scatter_result.scattered
    attenuation = scatter_result.attenuation
    pdf = scatter_result.pdf

    if pdf > 1e-8:
        incoming = _trace_with_pm(
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
        success = verify_photon_mapping()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
