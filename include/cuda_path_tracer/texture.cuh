#pragma once

#include "checker.cuh"
#include "solid.cuh"
#include <cuda/std/variant>

using Texture = cuda::std::variant<Checker, Solid>;
