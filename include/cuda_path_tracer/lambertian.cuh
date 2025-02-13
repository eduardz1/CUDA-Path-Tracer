#pragma once
#include "texture.cuh"

class Lambertian {

public:
  __host__ __device__ Lambertian(const Vec3 albedo) : texture(Solid(albedo)) {}
  __host__ __device__ Lambertian(const Texture texture) : texture(texture) {}

  __device__ bool scatter(const Ray &ray, Vec3 &normal, Vec3 &point, bool front,
                          Vec3 &attenuation, Ray &scattered,
                          curandStatePhilox4_32_10_t &state) {
    auto scatter_direction = normal + vectorOnHemisphere(normal, state);
    scatter_direction = roundScatterDirection(scatter_direction, normal);
    scattered = Ray(point, scatter_direction);

    attenuation = cuda::std::visit(
        [&point](auto &texture) { return texture.texture_value(point); },
        texture);
    return true;
  }

private:
  Texture texture;
};
