/**
 * @file render.cuh
 * @author Eduard Occhipinti (occhipinti.eduard@icloud.com)
 * @brief Header file for render.cu, which contains the kernel for rendering the
 * image
 * @version 0.1
 * @date 2024-10-26
 *
 * @copyright Copyright (c) 2024
 *
 */

#pragma once

#include <driver_types.h>

/**
 * @brief Kernel for rendering the image
 *
 * @param width Width of the image
 * @param height Height of the image
 * @param image Image to render
 */
__global__ void renderImage(int width, int height, uchar4 *image);
