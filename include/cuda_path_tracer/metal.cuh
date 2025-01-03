#pragma once

class Metal {

public:
  __host__ __device__ Metal(const Vec3 albedo) : albedo(albedo) {}

  __device__ bool scatter(const Ray &ray, Vec3 &normal, Vec3 &point,
                          Vec3 &attenuation, Ray &scattered,
                          curandState &state) {
    auto reflected_direction = reflect(ray.getDirection(), normal);
    scattered = Ray(point, reflected_direction);
    attenuation = albedo;
    return true;
  }

private:
  Vec3 albedo;
};
