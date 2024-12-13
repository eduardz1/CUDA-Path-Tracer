#include "cuda_path_tracer/shape.cuh"
#include "cuda_path_tracer/sphere.cuh"
#include <cstdint>
#include <driver_types.h>
#include <variant>
#include <vector>

class Scene {
public:
  __host__ Scene();
  __host__ Scene(uint16_t width, uint16_t height);
  // __host__ Scene(uint16_t width, uint16_t height, std::vector<Shape *> shapes);
  __host__ Scene(uint16_t width, uint16_t height, std::vector<std::variant<Sphere>> shapes);

  __host__ __device__ auto getWidth() -> uint16_t;
  __host__ __device__ auto getHeight() -> uint16_t;
  // __host__ __device__ auto getShapes() -> std::vector<Shape *> &;
  __host__ __device__ auto getShapes() -> std::vector<std::variant<Sphere>> &;

  // __host__ auto addShape(Shape *shape) -> void;
  __host__ auto addShape(std::variant<Sphere> shape) -> void;

  // TODO: deconstructor

private:
  // std::vector<Shape *> shapes;
  std::vector<std::variant<Sphere>> shapes;

  /**
   * @brief The width and height of the image in pixels.
   */
  uint16_t width, height;
};