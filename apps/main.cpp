
#include "cuda_path_tracer/camera.cuh"
#include "cuda_path_tracer/image.cuh"
#include "cuda_path_tracer/sphere.cuh"
#include <cstdint>
#include <cstdlib>
#include <vector_functions.h>
#include <vector_types.h>

auto main() -> int {
  constexpr uint16_t image_width = 128;
  constexpr uint16_t image_height = 128;

  uchar4 *image = new uchar4[image_width * image_height];

  Sphere sphere = {Vec3(2, 2, 5), 3};

  auto scene = std::make_shared<Scene>(image_width, image_height);
  scene->addShape(sphere);

  Camera camera = {Vec3(0.0, 0.0, -10.0)};
  camera.render(scene, image);

  const std::vector<uchar4> image_v(image, image + image_width * image_height);

  saveImageAsPPM("test_image.ppm", image_width, image_height, image_v);

  return EXIT_SUCCESS;
}
