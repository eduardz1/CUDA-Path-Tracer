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
#include <fstream>
#include <iostream>
#include <vector_functions.h>

void saveImageAsPPM(const char *filename, int width, int height,
                    uchar4 *image) {
  std::ofstream file(filename);

  file << "P3\n";
  file << width << " " << height << "\n";
  file << "255\n";

  for (int i = 0; i < width * height; i++) {
    file << (int)image[i].x << " " << (int)image[i].y << " " << (int)image[i].z
         << " ";
  }

  file.close();
}

auto convertColorTo8Bit(float4 color) -> uchar4 {
  return make_uchar4((char)(255.999 * color.x), (char)(255.999 * color.y),
                     (char)(255.999 * color.z), (char)(255.999 * color.w));
}
