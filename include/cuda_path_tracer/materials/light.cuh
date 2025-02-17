#pragma once

#include "cuda_path_tracer/color.cuh"

/**
 * @brief Light class, used to represent a light source
 */
class Light {
public:
  __host__ __device__ Light(Color emit_color)
      : nonNormalizedColor(emit_color) {}

  /**
   * @brief Get the emitted color of the light
   *
   * @return Color emitted color
   */
  __device__ auto emitted() const -> Color;

private:
  /**
   * @brief The color of a light does not have to be normalized. Values can be
   * greater than 1.
   */
  Color nonNormalizedColor;
};