#pragma once

#include "cuda_path_tracer/vec3.cuh"
#include <cstdint>
#include <driver_types.h>
#include <string>
#include <thrust/host_vector.h>
#include <vector_types.h>

/**
 * @brief Save the image as a PPM file
 *
 * @param filename Name of the file
 * @param width Width of the image
 * @param height Height of the image
 * @param image Image to save
 */
__host__ void saveImageAsPPM(const std::string &filename, const uint16_t width,
                             const uint16_t height,
                             const thrust::host_vector<uchar4> &image);

/**
 * @brief Convert a Vec3 color to a char4 color
 *
 * @param color Vec3 color with values between 0 and 1
 * @return char4 color with values between 0 and 255
 */
__device__ auto convertColorTo8Bit(Vec3 color) -> uchar4;
