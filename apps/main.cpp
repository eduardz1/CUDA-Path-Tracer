
#include "cuda_path_tracer/image.hpp"
#include "cuda_path_tracer/sphere.cuh"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector_functions.h>
#include <vector_types.h>

auto main() -> int {
  constexpr uint8_t image_width = 255;
  constexpr uint8_t image_height = 255;

  std::vector<uchar4> image(static_cast<size_t>(image_width * image_height));

  const Sphere sphere = {Vec3(40.0, 40.0, 0.0), 10};

  // make a cool gradient with a sphere :)
  for (int i = 0; i < image_height; i++) {
    for (int j = 0; j < image_width; j++) {
      float r = static_cast<float>(i) / (image_width - 1);
      float g = static_cast<float>(j) / (image_height - 1);
      float b = 0.25;

      if (sphere.hit(Ray(Vec3(i, j, -1), Vec3(0, 0, 1)))) {
        r = 0;
        g = 0;
        b = 1;
      }

      image[i * image_width + j] =
          convertColorTo8Bit(make_float4(r, g, b, 1.0));
    }
  }

  saveImageAsPPM("test_image.ppm", image_width, image_height, image);

  return EXIT_SUCCESS;
}
