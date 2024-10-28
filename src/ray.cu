#include "cuda_path_tracer/ray.cuh"
#include "cuda_path_tracer/vec3.cuh"

ray::ray() : origin(0), direction(0){};
ray::ray(const vec3 &origin, const vec3 &direction)
    : origin(origin), direction(direction){};

auto ray::getOrigin() const -> vec3 { return origin; }
auto ray::getDirection() const -> vec3 { return direction; }

auto ray::at(float t) const -> vec3 { return origin + direction * t; }
