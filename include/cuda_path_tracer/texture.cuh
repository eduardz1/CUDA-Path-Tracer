#pragma once

#include "cuda_path_tracer/checker.cuh"
#include "cuda_path_tracer/color.cuh"
#include <cuda/std/variant>

using Texture = cuda::std::variant<Checker, Color>;
