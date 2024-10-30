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

#include "cuda_path_tracer/render.cuh"

__global__ void renderImage(int width, int height, uchar4 *image) {
  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  auto y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  auto index = y * width + x;

  // Save the pixel for the R G B and Alpha values
  image[index] = make_uchar4(255, 0, 0, 255); // TODO: Make it query a ray
}
