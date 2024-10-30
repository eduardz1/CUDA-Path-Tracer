#include "cuda_path_tracer/ray.cuh"
#include "cuda_path_tracer/vec3.cuh"

__host__ __device__ ray::ray() : origin(0), direction(0) {};
__host__ __device__ ray::ray(const vec3 &origin, const vec3 &direction)
    : origin(origin), direction(direction) {};

__host__ __device__ auto ray::getOrigin() const -> vec3 { return origin; }
__host__ __device__ auto ray::getDirection() const -> vec3 { return direction; }

__host__ __device__ auto ray::at(float t) const -> vec3 {
  return origin + direction * t;
}
