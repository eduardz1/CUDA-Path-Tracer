#include "cuda_path_tracer/materials/light.cuh"

__device__ auto Light::emitted() const -> Color { return nonNormalizedColor; }