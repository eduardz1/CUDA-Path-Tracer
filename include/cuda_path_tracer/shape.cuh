#pragma once

#include "cuda_path_tracer/shapes/parallelogram.cuh"
#include "cuda_path_tracer/shapes/rectangular_cuboid.cuh"
#include "cuda_path_tracer/shapes/sphere.cuh"
#include <cuda/std/variant>
#include <thrust/device_vector.h>

using Shape = cuda::std::variant<Sphere, Parallelogram, RectangularCuboid>;
