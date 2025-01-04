
#include "cuda_path_tracer/camera.cuh"
#include "cuda_path_tracer/image.cuh"
#include "cuda_path_tracer/lambertian.cuh"
#include "cuda_path_tracer/metal.cuh"
#include "cuda_path_tracer/sphere.cuh"
#include <cstdint>
#include <cstdlib>
#include <vector_functions.h>
#include <vector_types.h>

auto main() -> int {
  constexpr uint16_t image_width = 512;
  constexpr uint16_t image_height = 512;

  uchar4 *image = new uchar4[image_width * image_height];

  auto scene = std::make_shared<Scene>(image_width, image_height);
  scene->addShape(Sphere{{0, 0, -1.2}, 0.5, Lambertian(Vec3{0.1, 0.2, 0.5})});
  scene->addShape(Sphere{{-1, 0, -1}, 0.5, Metal(Vec3{0.8, 0.8, 0.8}, 0.3)});
  scene->addShape(Sphere{{1, 0, -1}, 0.5, Metal(Vec3{0.8, 0.6, 0.2}, 1.0)});
  scene->addShape(
      Sphere{{0, -100.5, -1}, 100, Lambertian(Vec3{0.8, 0.8, 0.0})});

  Camera camera = {Vec3(-2, 2, 1)};
  camera.render(scene, image);

  const std::vector<uchar4> image_v(image, image + image_width * image_height);

  saveImageAsPPM("test_image.ppm", image_width, image_height, image_v);

  return EXIT_SUCCESS;
}
