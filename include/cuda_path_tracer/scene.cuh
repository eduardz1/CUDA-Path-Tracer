#include "cuda_path_tracer/shape.cuh"
#include <cstdint>
#include <driver_types.h>
#include <vector>

class Scene {
public:
  __host__ Scene();
  __host__ Scene(uint16_t width, uint16_t height);
  __host__ Scene(uint16_t width, uint16_t height, std::vector<ShapeH> shapes);

  __host__ auto getWidth() -> uint16_t;
  __host__ auto getHeight() -> uint16_t;
  __host__ auto getShapes() -> std::vector<ShapeH> &;
  __host__ auto addShape(ShapeH shape) -> void;

  // TODO: deconstructor

private:
  std::vector<ShapeH> shapes;

  /**
   * @brief The width and height of the image in pixels.
   */
  uint16_t width, height;
};