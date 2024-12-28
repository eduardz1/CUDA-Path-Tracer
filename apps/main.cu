
#include "cuda_path_tracer/camera.cuh"
#include "cuda_path_tracer/image.cuh"
#include "cuda_path_tracer/sphere.cuh"
#include <cstdlib>
#include <thrust/host_vector.h>
#include <vector_functions.h>
#include <vector_types.h>

auto main() -> int {
  constexpr auto image_width = 512;
  constexpr auto image_height = 512;
  constexpr auto num_pixels = image_width * image_height;

  thrust::host_vector<uchar4> image(num_pixels);

  auto scene = std::make_shared<Scene>(image_width, image_height);
  scene->addShape(Sphere{{0, 0, -1.2}, 0.5});
  scene->addShape(Sphere{{-1, 0, -1}, 0.5});
  scene->addShape(Sphere{{1, 0, -1}, 0.5});
  scene->addShape(Sphere{{0, -100.5, -1}, 100});

  Camera camera = {Vec3(-2, 2, 1)};
  camera.render(scene, image);

  saveImageAsPPM("test_image.ppm", image_width, image_height, image);

  return EXIT_SUCCESS;
}
