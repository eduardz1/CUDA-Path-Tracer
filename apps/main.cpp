
#include "cuda_path_tracer/camera.cuh"
#include "cuda_path_tracer/image.hpp"
#include "cuda_path_tracer/sphere.cuh"
#include <cstdint>
#include <cstdlib>
#include <vector_functions.h>
#include <vector_types.h>

auto main() -> int {
  constexpr uint16_t image_width = 64;
  constexpr uint16_t image_height = 64;

  uchar4 *image = new uchar4[image_width * image_height];

  Sphere sphere = {Vec3(2, 2, 5), 3};

  auto scene = std::make_shared<Scene>(image_width, image_height);
  scene->addShape(sphere);

  Camera camera = {Vec3(0.0, 0.0, -10.0)};
  camera.render(scene, image);

  // make a cool gradient with a sphere :)
  // for (int i = 0; i < image_height; i++) {
  //   for (int j = 0; j < image_width; j++) {
  //     float r = static_cast<float>(i) / (image_width - 1);
  //     float g = static_cast<float>(j) / (image_height - 1);
  //     float b = 0.25;

  //     if (sphere.hit(Ray(Vec3(i, j, -1), Vec3(0, 0, 1)))) {
  //       r = 0;
  //       g = 0;
  //       b = 1;
  //     }

  //     image[i * image_width + j] =
  //         convertColorTo8Bit(make_float4(r, g, b, 1.0));
  //   }
  // }

  const std::vector<uchar4> image_v(image, image + image_width *
  image_height);

  saveImageAsPPM("test_image.ppm", image_width, image_height, image_v);

  return EXIT_SUCCESS;
}

// Assigns every element in an array with its index.
// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

// #include "cuda_path_tracer/test.cuh"

// auto main() -> int {
//   simple();
//   return 0;
// }
