#include "cuda_path_tracer/hit_info.cuh"
#include "cuda_path_tracer/lambertian.cuh"

__device__ HitInfo::HitInfo()
    : point(0), normal(0), time(0), material(Material(Lambertian(Vec3{0, 0, 0}))) {}
__device__ HitInfo::HitInfo(const Vec3 &point, const Vec3 &normal,
                            const float time, const Material &material)
    : point(point), normal(normal), time(time), material(material) {}

__device__ auto HitInfo::getPoint() const -> Vec3 { return point; }
__device__ auto HitInfo::getNormal() const -> Vec3 { return normal; }
__device__ auto HitInfo::getTime() const -> float { return time; }

__device__ auto HitInfo::getFront() const -> bool { return front; }

__device__ auto HitInfo::getMaterial() const -> Material { return material; }
__device__ void HitInfo::setPoint(const Vec3 &point) { this->point = point; }
__device__ void HitInfo::setTime(const float time) { this->time = time; }
__device__ void HitInfo::setMaterial(const Material &material) {
  this->material = material;
}
__device__ void HitInfo::setNormal(const Ray &r, const Vec3 &outward_normal) {
  this->front = dot(r.getDirection(), outward_normal) < 0;
  normal = this->front ? outward_normal : -outward_normal;
}