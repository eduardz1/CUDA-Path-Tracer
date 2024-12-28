#include "cuda_path_tracer/hit_info.cuh"
#include "cuda_path_tracer/sphere.cuh"
#include "cuda_path_tracer/material.cuh"

__host__ Sphere::Sphere()
    : center(Vec3(0,0,0)), radius(0), material(Material(Lambertian(Vec3(0,0,0)))) {}

__host__ Sphere::Sphere(const Vec3 &center, const float radius, const Material &material)
    : center(center), radius(static_cast<float>(std::fmax(0, radius))), material(material) {}

__device__ auto Sphere::hit(const Ray &r, const float hit_t_min,
                            const float hit_t_max, HitInfo &hi) const -> bool {
  // Calculate the discriminant of the quadratic equation, if it is less than 0
  // then the ray does not intersect the sphere. The formula is derived from
  // the equation of a sphere and the parametric equation of a ray.
  //
  // The discriminant is calculated as follows:
  // d = (-b +- sqrt(b^2 - 4ac)) / 2a
  //
  // where:
  // a = dot(r.direction, r.direction)
  // b = dot(-2(r.direction), (sphere.center - r.origin))
  // c = dot(sphere.center - r.origin,sphere.center - r.origin)- sphere.radius^2
  //
  // We simplify the formula by using the negative half of b, h = -b/2

  const Vec3 oc = this->center - r.getOrigin();

  const auto a = r.getDirection().getLengthSquared();
  const auto h = dot(r.getDirection(), oc);
  const auto c = oc.getLengthSquared() - radius * radius;

  const auto discriminant = h * h - a * c;

  if (discriminant < 0) {
    return false;
  }

  const auto sqrtd = sqrt(discriminant);

  // Finds the smallest root that is between the minimum and maximum t or exits
  auto root = (h - sqrtd) / a;

  if (root < hit_t_min || hit_t_max < root) {
    root = (h + sqrtd) / a;
    if (root < hit_t_min || hit_t_max < root) {
      return false;
    }
  }

  hi.setTime(root);
  hi.setPoint(r.at(root));
  hi.setNormal(r, (hi.getPoint() - center) / radius);
  hi.setMaterial(material);

  return true;
}

__device__ auto Sphere::getCenter() const -> Vec3 { return center; }
__device__ auto Sphere::getMaterial() const -> Material { return material; }