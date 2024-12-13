#include "cuda_path_tracer/scene.cuh"
#include <cstdint>
#include <variant>

__host__ Scene::Scene() : width(1), height(1) {}

__host__ Scene::Scene(uint16_t width, uint16_t height)
    : width(width), height(height) {}

__host__ __device__ auto Scene::getWidth() -> uint16_t { return width; }
__host__ __device__ auto Scene::getHeight() -> uint16_t { return height; }
__host__ __device__ auto
Scene::getShapes() -> std::vector<std::variant<Sphere>> & {
  return shapes;
}

__host__ auto Scene::addShape(std::variant<Sphere> shape) -> void {
  shapes.push_back(shape);
}
