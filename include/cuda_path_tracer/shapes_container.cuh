#pragma once
#include "shape.cuh"
#include <cstdint>
#include <vector>



class ShapesContainer {

public:

  __host__ __device__ auto getShapeAt(size_t idx) -> char *;
  __host__ __device__ auto getShapeTypeAt(size_t idx) -> ShapeType;
  __host__ __device__ auto getNumShapes() -> uint16_t;
  __host__ auto copyShapesToDevice(const std::vector<Shape *> &shapes) -> void;

  // __host__ __device__ ~ShapesContainer();

private:
  /**
   * @brief Continuous memory block containing the shapes in a form suitable for
   * the device.
   */
  char *shapes = nullptr;
  /**
   * @brief Number of shapes in the shapes list.
   */
  uint16_t numShapes = 0;
  /**
   * @brief List of the shape types corresponding to the shapes in the shapes
   * memory block.
   */
  ShapeType *shapeTypes = nullptr;
  /**
   * @brief List of the offsets for iterating through shapes - each offset
   * corresponds to the offset of shape in the memory block.
   */
  size_t *offsets = nullptr;
};
