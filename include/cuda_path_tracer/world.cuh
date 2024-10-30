/**
 * @file world.cuh
 * @author Eduard Occhipinti (occhipinti.eduard@icloud.com)
 * @brief // TODO: Add brief
 * @version 0.1
 * @date 2024-10-30
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "cuda_path_tracer/vec3.cuh"
#include <driver_types.h>

#define SKY_COLOR make_uchar4(53, 81, 92, 255)

__device__ __host__ auto groundColor(const Vec3 &origin,
                                     const Vec3 &direction) -> uchar4;

/**
 * @brief Returns the color of the sky based on the constant SKY_COLOR and the
 * direction of the ray
 *
 * @param direction direction of the ray
 * @return uchar4 color of the sky
 */
__device__ __host__ auto skyColor(const Vec3 &direction) -> uchar4;