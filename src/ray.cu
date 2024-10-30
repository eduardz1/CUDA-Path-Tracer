#include "cuda_path_tracer/ray.cuh"
#include "cuda_path_tracer/vec3.cuh"

__host__ __device__ Ray::Ray() : origin(0), direction(0) {};
__host__ __device__ Ray::Ray(const Vec3 &origin, const Vec3 &direction)
    : origin(origin), direction(direction) {};

__host__ __device__ auto Ray::getOrigin() const -> Vec3 { return origin; }
__host__ __device__ auto Ray::getDirection() const -> Vec3 { return direction; }

__host__ __device__ auto Ray::at(float t) const -> Vec3 {
  return origin + direction * t;
}
