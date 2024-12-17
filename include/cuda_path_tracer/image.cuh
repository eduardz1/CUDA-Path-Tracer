/**
 * @file image.hpp
 * @author Eduard Occhipinti (occhipinti.eduard@icloud.com)
 * @brief Header file for image.cu, which contains the functions to save an
 * image as a PPM file
 * @version 0.1
 * @date 2024-10-26
 *
 * @copyright Copyright (c) 2024
 *
 */

#pragma once

#include <driver_types.h>
#include <vector>
#include <vector_types.h>

/**
 * @brief Save the image as a PPM file
 *
 * @param filename Name of the file
 * @param width Width of the image
 * @param height Height of the image
 * @param image Image to save
 */
__host__ void saveImageAsPPM(const char *filename, int width, int height,
                             const std::vector<uchar4> &image);

/**
 * @brief Convert a float4 color to a char4 color
 *
 * @param color float4 color with values between 0 and 1
 * @return char4 color with values between 0 and 255
 */
__device__ auto convertColorTo8Bit(float4 color) -> uchar4;
