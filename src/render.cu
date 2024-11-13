/**
 * @file render.cu
 * @author Eduard Occhipinti (occhipinti.eduard@icloud.com)
 * @brief Implementation file for render.cuh, which contains the kernel for
 * rendering the image
 * @version 0.1
 * @date 2024-10-26
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "cuda_path_tracer/ray.cuh"
#include "cuda_path_tracer/render.cuh"

#include <vector_functions.h>

__global__ void renderImage(int width, int height, uchar4 *image) {
  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  auto y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  auto index = y * width + x;

  Ray const r(Vec3(0, 0, 0), Vec3(0, 0, 1));

  // Save the pixel for the R G B and Alpha values
  image[index] = make_uchar4(UCHAR_MAX, 0, 0, UCHAR_MAX); // TODO: Make it query a ray
}
