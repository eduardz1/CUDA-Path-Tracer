#pragma once

#include "cuda_path_tracer/color.cuh"

class Light {
public:
  __host__ __device__ Light(Color emit_color)
      : nonNormalizedColor(emit_color) {}

  __device__ auto emitted() const -> Color;

private:
  /**
   * @brief The color of a light does not have to be normalized. Values can be
   * greater than 1.
   */
  Color nonNormalizedColor;
};