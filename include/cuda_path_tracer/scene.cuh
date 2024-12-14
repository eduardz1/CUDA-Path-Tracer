#include "cuda_path_tracer/shape.cuh"
#include <cstdint>
#include <driver_types.h>
#include <vector>

class Scene {
public:
  __host__ __device__ Scene();
  __host__ __device__ Scene(uint16_t width, uint16_t height);
  __host__ __device__ Scene(uint16_t width, uint16_t height,
                            std::vector<Shape> shapes);

  __host__ __device__ auto getWidth() -> uint16_t;
  __host__ __device__ auto getHeight() -> uint16_t;
  __host__ __device__ auto getShapes() -> std::vector<Shape> &;
  __host__ __device__ auto addShape(Shape shape) -> void;

  // TODO: deconstructor

private:
  std::vector<Shape> shapes;

  /**
   * @brief The width and height of the image in pixels.
   */
  uint16_t width, height;
};