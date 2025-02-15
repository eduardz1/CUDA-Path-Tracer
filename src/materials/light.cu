#include "cuda_path_tracer/materials/light.cuh"
#include "cuda_path_tracer/utilities.cuh"

__device__ auto Light::emitted(Vec3 &point) const -> Vec3 {
  return cuda::std::visit(
      overload{[&point](const Checker &checker) {
                 return checker.texture_value(point);
               },
               [](const Color &color) { return Vec3{color}; }},
      texture);
}