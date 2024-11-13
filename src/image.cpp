/**
 * @file image.cpp
 * @author Eduard Occhipinti (occhipinti.eduard@icloud.com)
 * @brief Implementation file for image.hh, which contains the functions to save
 * an image as a PPM file
 * @version 0.1
 * @date 2024-10-26
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "cuda_path_tracer/image.hpp"
#include <algorithm>
#include <fstream>
#include <vector>
#include <vector_functions.h>

void saveImageAsPPM(const char *filename, const int width, const int height,
                    const std::vector<uchar4> &image) {
  std::ofstream file(filename);

  file << "P3\n";
  file << width << " " << height << "\n";
  file << "255\n";

  for (int i = 0; i < width * height; i++) {
    file << static_cast<int>(image[i].x) << " " << static_cast<int>(image[i].y)
         << " " << static_cast<int>(image[i].z) << "\n";
  }

  file.close();
}

auto convertColorTo8Bit(const float4 color) -> uchar4 {
  return make_uchar4(
      static_cast<unsigned char>(static_cast<float>(UCHAR_MAX) *
                                 std::clamp(color.x, 0.0f, 1.0f)),
      static_cast<unsigned char>(static_cast<float>(UCHAR_MAX) *
                                 std::clamp(color.y, 0.0f, 1.0f)),
      static_cast<unsigned char>(static_cast<float>(UCHAR_MAX) *
                                 std::clamp(color.z, 0.0f, 1.0f)),
      static_cast<unsigned char>(static_cast<float>(UCHAR_MAX) *
                                 std::clamp(color.w, 0.0f, 1.0f)));
}
