#pragma once

#include "cuda_path_tracer/shape.cuh"
#include <cstdint>
#include <driver_types.h>
#include <thrust/device_vector.h>

class Scene {
public:
  __host__ Scene();
  __host__ Scene(uint16_t width, uint16_t height);
  __host__ Scene(uint16_t width, uint16_t height,
                 thrust::device_vector<Shape> shapes);

  __host__ auto getWidth() const -> uint16_t;
  __host__ auto getHeight() const -> uint16_t;
  __host__ auto getShapes() -> thrust::device_vector<Shape> &;
  __host__ auto addShape(Shape shape) -> void;

private:
  thrust::device_vector<Shape> shapes;

  /**
   * @brief The width and height of the image in pixels.
   */
  uint16_t width, height;
};