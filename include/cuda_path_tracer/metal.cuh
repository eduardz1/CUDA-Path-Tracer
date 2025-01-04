#pragma once

class Metal {

public:
  __host__ __device__ Metal(const Vec3 albedo, double fuzz)
      : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {}

  __device__ bool scatter(const Ray &ray, Vec3 &normal, Vec3 &point, bool front,
                          Vec3 &attenuation, Ray &scattered,
                          curandState &state) {
    auto reflected_direction = reflect(ray.getDirection(), normal);
    reflected_direction = makeUnitVector(reflected_direction) +
                          (fuzz * makeUnitVector(randomVector(state)));
    scattered = Ray(point, reflected_direction);
    attenuation = albedo;
    return (dot(scattered.getDirection(), normal) > 0);
  }

private:
  Vec3 albedo;
  double fuzz;
};
