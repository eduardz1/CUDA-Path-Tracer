#pragma once


#include "lambertian.cuh"
#include "metal.cuh"
#include "dielectric.cuh"
#include <cuda/std/variant>

using Material = cuda::std::variant<Lambertian, Metal, Dielectric>;
