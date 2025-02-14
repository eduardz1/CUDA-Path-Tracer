#include "cuda_path_tracer/hit_info.cuh"
#include "cuda_path_tracer/shapes/parallelogram.cuh"
#include "cuda_path_tracer/vec3.cuh"

__host__ Parallelogram::Parallelogram() = default;
__host__ Parallelogram::Parallelogram(const Vec3 &origin, const Vec3 &u,
                                      const Vec3 &v, const Material &material)
    : origin(origin), u(u), v(v), material(material) {
  const auto n = cross(u, v);
  normal = makeUnitVector(n);
  area = dot(normal, origin); // NOLINT
  w = n / dot(n, n);
};

__device__ auto Parallelogram::hit(const Ray &r, const float hit_t_min,
                                   const float hit_t_max, HitInfo &hi) const
    -> bool {
  const auto denominator = dot(normal, r.getDirection());

  if (fabs(denominator) < 1e-6) { // NOLINT ray is parallel to the plane
    return false;
  }

  const auto t = (area - dot(normal, r.getOrigin())) / denominator;
  if (t < hit_t_min || t > hit_t_max) {
    return false;
  }

  const auto point = r.at(t);

  const auto p = point - origin;
  const auto alpha = dot(w, cross(p, v));
  const auto beta = dot(w, cross(u, p));

  if (alpha < 0 || alpha > 1 || beta < 0 || beta > 1) {
    return false;
  }

  hi.time = t;
  hi.point = point;
  hi.setNormal(r, normal);
  hi.material = material;

  return true;
}