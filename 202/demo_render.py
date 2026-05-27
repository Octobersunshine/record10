import sys
import random
import time
from mathlib import Vec3, Color, dot, normalize
from geometry import HittableList, Sphere, Plane
from material import Lambertian, Metal, Dielectric, DiffuseLight
from camera import Camera
from pathtracer import PathTracer
from renderer import Renderer


def create_cornell_box_mis_demo():
    print("Creating Cornell Box scene with MIS...")
    print("=" * 60)
    print()

    world = HittableList()
    lights = HittableList()

    red = Lambertian(Color(0.65, 0.05, 0.05))
    white = Lambertian(Color(0.73, 0.73, 0.73))
    green = Lambertian(Color(0.12, 0.45, 0.15))
    light = DiffuseLight(Color(15, 15, 15))

    light_sphere = Sphere(Vec3(278, 500, 278), 80, light)
    world.add(light_sphere)
    lights.add(light_sphere)

    world.add(Plane(Vec3(0, 0, 0), Vec3(0, 1, 0), white))
    world.add(Plane(Vec3(0, 555, 0), Vec3(0, -1, 0), white))
    world.add(Plane(Vec3(0, 0, 0), Vec3(1, 0, 0), green))
    world.add(Plane(Vec3(555, 0, 0), Vec3(-1, 0, 0), red))
    world.add(Plane(Vec3(0, 0, 555), Vec3(0, 0, -1), white))

    world.add(Sphere(Vec3(180, 90, 200), 90, white))
    world.add(Sphere(Vec3(380, 90, 350), 90, white))

    camera = Camera(
        lookfrom=Vec3(278, 278, -800),
        lookat=Vec3(278, 278, 0),
        vup=Vec3(0, 1, 0),
        vfov=40,
        aspect_ratio=1.0,
        aperture=0.0,
        focus_dist=10.0
    )

    background = Color(0, 0, 0)

    return world, camera, background, lights


def create_multi_material_scene():
    print("Creating multi-material scene with MIS...")
    print("=" * 60)
    print()

    world = HittableList()
    lights = HittableList()

    ground = Lambertian(Color(0.5, 0.5, 0.5))
    world.add(Sphere(Vec3(0, -1000, 0), 1000, ground))

    light_material = DiffuseLight(Color(4, 4, 4))
    light_sphere = Sphere(Vec3(0, 5, -1), 1.5, light_material)
    world.add(light_sphere)
    lights.add(light_sphere)

    random.seed(42)
    for a in range(-5, 5):
        for b in range(-5, 5):
            choose_mat = random.random()
            center = Vec3(a + 0.9 * random.random(), 0.2, b + 0.9 * random.random())

            dx = center.x - 4
            dy = center.y - 0.2
            dz = center.z
            if dx * dx + dy * dy + dz * dz > 0.9:
                if choose_mat < 0.6:
                    albedo = Color(random.random(), random.random(), random.random()) * Color(random.random(), random.random(), random.random())
                    sphere_material = Lambertian(albedo)
                elif choose_mat < 0.85:
                    albedo = Color(random.uniform(0.5, 1), random.uniform(0.5, 1), random.uniform(0.5, 1))
                    fuzz = random.uniform(0, 0.5)
                    sphere_material = Metal(albedo, fuzz)
                else:
                    sphere_material = Dielectric(1.5)

                world.add(Sphere(center, 0.2, sphere_material))

    material1 = Dielectric(1.5)
    world.add(Sphere(Vec3(0, 1, 0), 1.0, material1))

    material2 = Lambertian(Color(0.4, 0.2, 0.1))
    world.add(Sphere(Vec3(-4, 1, 0), 1.0, material2))

    material3 = Metal(Color(0.7, 0.6, 0.5), 0.0)
    world.add(Sphere(Vec3(4, 1, 0), 1.0, material3))

    camera = Camera(
        lookfrom=Vec3(13, 2, 3),
        lookat=Vec3(0, 0, 0),
        vup=Vec3(0, 1, 0),
        vfov=20,
        aspect_ratio=16.0 / 9.0,
        aperture=0.1,
        focus_dist=10.0
    )

    background = Color(0.70, 0.80, 1.00)

    return world, camera, background, lights


def render_scene_direct(scene_name, create_scene_func, width=400, height=300, samples=50, max_depth=50):
    world, camera, background, lights = create_scene_func()

    print(f"Rendering: {scene_name}")
    print(f"Resolution: {width}x{height}")
    print(f"Samples per pixel: {samples}")
    print(f"Max depth: {max_depth}")
    print()

    pathtracer = PathTracer(world, max_depth, background, lights)

    start = time.time()
    pixels = []

    for j in range(height - 1, -1, -1):
        print(f"  Scanline {height - j}/{height}", end='\r', flush=True)
        row = []
        for i in range(width):
            pixel_color = Color(0, 0, 0)
            for _ in range(samples):
                u = (i + random.random()) / (width - 1)
                v = (j + random.random()) / (height - 1)
                ray = camera.get_ray(u, v)
                pixel_color = pixel_color + pathtracer.trace_ray(ray)
            row.append(pixel_color.to_rgb(samples))
        pixels.append(row)

    elapsed = time.time() - start
    print(f"\n  Completed in {elapsed:.2f} seconds")

    output_file = f"{scene_name.lower().replace(' ', '_')}_mis.png"
    renderer = Renderer(width, height, samples, max_depth)
    renderer.save_png(pixels, output_file)
    print(f"  Saved: {output_file}")
    print()

    return pixels


def main():
    print("=" * 60)
    print("Monte Carlo Path Tracer with MIS Demo")
    print("=" * 60)
    print()

    print("Available scenes:")
    print("  1. Cornell Box")
    print("  2. Multi-material Spheres")
    print("  3. Both scenes")
    print()

    choice = input("Select scene [1-3] (default=3): ").strip() or "3"

    if choice == "1":
        render_scene_direct("Cornell Box", create_cornell_box_mis_demo,
                          width=400, height=400, samples=100, max_depth=50)
    elif choice == "2":
        render_scene_direct("Multi-material Spheres", create_multi_material_scene,
                          width=640, height=360, samples=50, max_depth=50)
    else:
        print("Rendering both scenes...")
        print()
        render_scene_direct("Cornell Box", create_cornell_box_mis_demo,
                          width=300, height=300, samples=50, max_depth=50)
        render_scene_direct("Multi-material Spheres", create_multi_material_scene,
                          width=480, height=270, samples=25, max_depth=50)

    print("=" * 60)
    print("Rendering complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
