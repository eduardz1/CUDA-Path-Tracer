#include "cuda_path_tracer/sphere.cuh"

__device__ Sphere::Sphere(const Vec3 &center, float radius)
    : center(center), radius(radius) {}

__device__ auto Sphere::hit(const Ray &r) const -> bool {
  Vec3 oc = r.getOrigin() - center;
  float a = r.getDirection().dot(r.getDirection());
  float b = 2.0f * oc.dot(r.getDirection());
  float c = oc.dot(oc) - radius * radius;
  float discriminant = b * b - 4 * a * c;
  return discriminant > 0;
}