import random
import time
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

from mathlib import Vec3, Color, Ray
from geometry import HittableList, Sphere, Plane
from material import Lambertian, Metal, Dielectric, DiffuseLight
from camera import Camera
from pathtracer import PathTracer


class Renderer:
    def __init__(
        self,
        width: int = 400,
        height: int = 300,
        samples_per_pixel: int = 100,
        max_depth: int = 50
    ):
        self.width = width
        self.height = height
        self.samples_per_pixel = samples_per_pixel
        self.max_depth = max_depth

    def render_scene(
        self,
        world: HittableList,
        camera: Camera,
        background: Color = None,
        lights: HittableList = None,
        num_workers: int = None
    ) -> List[List[Tuple[int, int, int]]]:
        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)

        pathtracer = PathTracer(world, self.max_depth, background, lights)

        print(f"Rendering {self.width}x{self.height} image with {self.samples_per_pixel} samples per pixel")
        print(f"Using {num_workers} worker processes")
        start_time = time.time()

        rows = []
        for j in range(self.height - 1, -1, -1):
            rows.append((j, world, camera, pathtracer))

        with Pool(processes=num_workers) as pool:
            results = pool.map(self._render_row, rows)

        pixels = [row for _, row in sorted(results, key=lambda x: x[0])]

        elapsed = time.time() - start_time
        print(f"\nRendering completed in {elapsed:.2f} seconds")

        return pixels

    def _render_row(self, args):
        j, world, camera, pathtracer = args
        print(f"Scanline {self.height - j}/{self.height}", end='\r', flush=True)

        row = []
        for i in range(self.width):
            pixel_color = Color(0, 0, 0)
            for _ in range(self.samples_per_pixel):
                u = (i + random.random()) / (self.width - 1)
                v = (j + random.random()) / (self.height - 1)
                ray = camera.get_ray(u, v)
                pixel_color = pixel_color + pathtracer.trace_ray(ray)
            row.append(pixel_color.to_rgb(self.samples_per_pixel))

        return (j, row)

    def save_ppm(self, pixels: List[List[Tuple[int, int, int]]], filename: str):
        with open(filename, 'w') as f:
            f.write("P3\n")
            f.write(f"{self.width} {self.height}\n")
            f.write("255\n")
            for row in pixels:
                for r, g, b in row:
                    f.write(f"{r} {g} {b}\n")
        print(f"Image saved to {filename}")

    def save_png(self, pixels: List[List[Tuple[int, int, int]]], filename: str):
        try:
            from PIL import Image
            img = Image.new('RGB', (self.width, self.height))
            img_data = []
            for row in pixels:
                img_data.extend(row)
            img.putdata(img_data)
            img.save(filename)
            print(f"Image saved to {filename}")
        except ImportError:
            print("PIL/Pillow not available, saving as PPM instead")
            self.save_ppm(pixels, filename.replace('.png', '.ppm'))


def create_cornell_box_scene() -> Tuple[HittableList, Camera, Color, HittableList]:
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


def create_random_spheres_scene() -> Tuple[HittableList, Camera, Color, HittableList]:
    world = HittableList()
    lights = HittableList()

    ground_material = Lambertian(Color(0.5, 0.5, 0.5))
    world.add(Sphere(Vec3(0, -1000, 0), 1000, ground_material))

    light_material = DiffuseLight(Color(4, 4, 4))
    light_sphere = Sphere(Vec3(0, 5, -1), 1.5, light_material)
    world.add(light_sphere)
    lights.add(light_sphere)

    for a in range(-11, 11):
        for b in range(-11, 11):
            choose_mat = random.random()
            center = Vec3(a + 0.9 * random.random(), 0.2, b + 0.9 * random.random())

            dx = center.x - 4
            dy = center.y - 0.2
            dz = center.z - 0
            if dx * dx + dy * dy + dz * dz > 0.9:
                if choose_mat < 0.8:
                    albedo = Color(random.random(), random.random(), random.random()) * Color(random.random(), random.random(), random.random())
                    sphere_material = Lambertian(albedo)
                elif choose_mat < 0.95:
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


def main():
    random.seed(42)

    renderer = Renderer(
        width=400,
        height=300,
        samples_per_pixel=50,
        max_depth=50
    )

    scene_type = input("Choose scene (1=Cornell Box, 2=Random Spheres, default=1): ").strip() or "1"

    if scene_type == "2":
        world, camera, background, lights = create_random_spheres_scene()
        renderer.width = 640
        renderer.height = 360
        output_file = "random_spheres.png"
    else:
        world, camera, background, lights = create_cornell_box_scene()
        renderer.width = 400
        renderer.height = 400
        output_file = "cornell_box.png"

    pixels = renderer.render_scene(world, camera, background, lights)
    renderer.save_png(pixels, output_file)


if __name__ == "__main__":
    main()
