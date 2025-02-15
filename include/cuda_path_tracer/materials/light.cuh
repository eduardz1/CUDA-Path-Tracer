#pragma once

#include "cuda_path_tracer/texture.cuh"

class Light {
public:
  __host__ __device__ Light(Color emit_color) : texture(emit_color) {}

  __device__ auto emitted(Vec3 &point) const -> Vec3;

private:
  Texture texture;
};