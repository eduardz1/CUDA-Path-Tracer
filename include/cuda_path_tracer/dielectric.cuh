#pragma once

class Dielectric {

public:
  __host__ __device__ Dielectric(const double refraction)
      : refraction(refraction) {}

  __device__ bool scatter(const Ray &ray, Vec3 &normal, Vec3 &point, bool front,
                          Vec3 &attenuation, Ray &scattered,
                          curandState &state) {
    double ri = front ? (1.0f / refraction) : refraction;
    auto scatter_direction = makeUnitVector(ray.getDirection());
    auto refracted = refract(scatter_direction, normal, ri);
    scattered = Ray(point, refracted);
    attenuation = Vec3{1.0f, 1.0f, 1.0f};
    return true;
  }

private:
  double refraction;
};
