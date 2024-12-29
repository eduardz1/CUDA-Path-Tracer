/**
 * @file shape.cuh
 * @author Eduard Occhipinti (occhipinti.eduard@icloud.com)
 * @brief Abstract class for a shape in the scene
 * @version 0.1
 * @date 2024-10-30
 *
 * @copyright Copyright (c) 2024
 *
 */

#pragma once

#include "cuda_path_tracer/shapes/parallelogram.cuh"
#include "cuda_path_tracer/shapes/rectangular_cuboid.cuh"
#include "cuda_path_tracer/shapes/sphere.cuh"
#include <cuda/std/variant>
#include <thrust/device_vector.h>

using Shape = cuda::std::variant<Sphere, Parallelogram, RectangularCuboid>;
