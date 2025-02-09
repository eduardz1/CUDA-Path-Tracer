#pragma once

#include "cuda_path_tracer/materials/dielectric.cuh"
#include "cuda_path_tracer/materials/lambertian.cuh"
#include "cuda_path_tracer/materials/metal.cuh"
#include <cuda/std/variant>

using Material = cuda::std::variant<Lambertian, Metal, Dielectric>;
