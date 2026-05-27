import sys
import random

from mathlib import Vec3, Color
from geometry import HittableList, Sphere, Plane
from material import Lambertian, Metal, Dielectric, DiffuseLight
from camera import Camera
from pathtracer import PathTracer
from renderer import Renderer


def test_simple_scene():
    print("Testing Monte Carlo Path Tracer...")
    print("=" * 50)

    random.seed(42)

    renderer = Renderer(
        width=200,
        height=150,
        samples_per_pixel=10,
        max_depth=20
    )

    world = HittableList()
    lights = HittableList()

    ground = Lambertian(Color(0.8, 0.8, 0.8))
    red = Lambertian(Color(0.9, 0.1, 0.1))
    blue = Lambertian(Color(0.1, 0.1, 0.9))
    metal = Metal(Color(0.8, 0.8, 0.8), 0.3)
    glass = Dielectric(1.5)
    light = DiffuseLight(Color(8, 8, 8))

    world.add(Plane(Vec3(0, -1, 0), Vec3(0, 1, 0), ground))

    light_sphere = Sphere(Vec3(0, 5, -2), 1.0, light)
    world.add(light_sphere)
    lights.add(light_sphere)

    world.add(Sphere(Vec3(-2, 0, 0), 1.0, red))
    world.add(Sphere(Vec3(0, 0, 0), 1.0, glass))
    world.add(Sphere(Vec3(2, 0, 0), 1.0, blue))
    world.add(Sphere(Vec3(-4, 0, 1), 1.0, metal))

    camera = Camera(
        lookfrom=Vec3(0, 2, -8),
        lookat=Vec3(0, 0, 0),
        vup=Vec3(0, 1, 0),
        vfov=45,
        aspect_ratio=4.0 / 3.0,
        aperture=0.0,
        focus_dist=8.0
    )

    background = Color(0.5, 0.7, 1.0)

    print(f"Scene: {len(world.objects)} objects")
    print(f"Resolution: {renderer.width}x{renderer.height}")
    print(f"Samples per pixel: {renderer.samples_per_pixel}")
    print(f"Max depth: {renderer.max_depth}")
    print()

    pixels = renderer.render_scene(world, camera, background, lights, num_workers=2)

    output_file = "test_render.png"
    renderer.save_png(pixels, output_file)

    print("\n" + "=" * 50)
    print("Test completed successfully!")
    print(f"Output: {output_file}")
    return True


if __name__ == "__main__":
    try:
        success = test_simple_scene()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
