#include "cuda_path_tracer/hit_info.cuh"
#include "cuda_path_tracer/materials/lambertian.cuh"

__host__ __device__ HitInfo::HitInfo()
    : point(0), normal(0), time(0),
      material(Material(Lambertian(Colors::Black))) {}
__host__ __device__ HitInfo::HitInfo(const Vec3 &point, const Vec3 &normal,
                                     const float time, const Material &material)
    : point(point), normal(normal), time(time), material(material) {}

__device__ void HitInfo::setNormal(const Ray &r, const Vec3 &outward_normal) {
  this->front = dot(r.getDirection(), outward_normal) < 0;
  normal = this->front ? outward_normal : -outward_normal;
}