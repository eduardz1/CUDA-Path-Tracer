#pragma once

#include "lambertian.cuh"
#include "metal.cuh"
#include "dielectric.cuh"
#include "light.cuh"
#include <cuda/std/variant>

using Material = cuda::std::variant<Lambertian, Metal, Dielectric, Light>;
