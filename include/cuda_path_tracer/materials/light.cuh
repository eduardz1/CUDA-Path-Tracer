#pragma once

class Light {
public:
  __host__ __device__ Light(Texture texture) : texture(texture) {}
  __host__ __device__ Light(Vec3 emit_color) : texture(Solid(emit_color)) {}

  __device__ Vec3 emitted(Vec3 &point) {
    return cuda::std::visit(
        [&point](auto &texture) { return texture.texture_value(point); },
        texture);
  }

  __device__ bool scatter(const Ray &ray, Vec3 &normal, Vec3 &point, bool front,
                          Vec3 &attenuation, Ray &scattered,
                          curandStatePhilox4_32_10_t &state) {
    return false;
  }

private:
  Texture texture;
};