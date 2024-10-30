
#include "cuda_path_tracer/image.hpp"
#include <vector_functions.h>
#include <vector_types.h>

auto main() -> int {
  int image_width = 256;
  int image_height = 256;

  auto *image =
      new uchar4[static_cast<unsigned long>(image_width * image_height)];

  // make a cool gradient :)
  for (int i = 0; i < image_height; i++) {
    for (int j = 0; j < image_width; j++) {
      float r = double(i) / (image_width - 1);
      float g = double(j) / (image_height - 1);
      float b = 0.25;

      image[i * image_width + j] =
          convertColorTo8Bit(make_float4(r, g, b, 1.0));
    }
  }

  saveImageAsPPM("test_image.ppm", image_width, image_height, image);

  delete[] image;

  return 0;
}
