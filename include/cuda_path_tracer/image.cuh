#pragma once

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
