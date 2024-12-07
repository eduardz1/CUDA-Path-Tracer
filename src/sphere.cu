#include "cuda_path_tracer/sphere.cuh"

#ifndef __NVCC__
#ifndef CXA_PURE_VIRTUAL_WORKAROUND
#define CXA_PURE_VIRTUAL_WORKAROUND
// Workaround for CUDA compilation with clang
// https://bugs.llvm.org/show_bug.cgi?id=49839
extern "C" __device__ void __cxa_pure_virtual() { // NOLINT
  while (1) {
  }
}
#endif
#endif

__host__ Sphere::Sphere(const Vec3 &center, const float radius)
    : Shape(), center(center), radius(radius) {}

__host__ __device__ auto Sphere::hit(const Ray &r) const -> bool {
  Vec3 const oc = r.getOrigin() - center;
  float const a = r.getDirection().dot(r.getDirection());
  float const b = 2.0f * oc.dot(r.getDirection());
  float const c = oc.dot(oc) - radius * radius;
  float const discriminant = b * b - 4 * a * c;
  return discriminant > 0;
}