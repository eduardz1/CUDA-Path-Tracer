#pragma once


#include "lambertian.cuh"
#include <cuda/std/variant>

using Material = cuda::std::variant<Lambertian>;
