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

    double cos_theta = fmin(dot(-scatter_direction, normal), 1.0f);
    double sin_theta = sqrt(1.0f - cos_theta * cos_theta);
    bool reflecting = ri * sin_theta > 1;

    if (reflecting) {
      scattered = Ray(point, reflect(scatter_direction, normal));
    } else {
      scattered = Ray(point, refract(scatter_direction, normal, ri));
    }
    attenuation = Vec3{1.0f, 1.0f, 1.0f};
    return true;
  }

private:
  double refraction;

  static double reflectance(double cos, double refraction) {
    auto r0 = (1 - refraction) / (1 + refraction);
    return r0 * r0 + (1 - r0 * r0) * pow((1 - cos), 5);
  }
};
