#include "cuda_path_tracer/scene.cuh"
#include <cstdint>
#include <utility>

__host__ Scene::Scene() : width(1), height(1) {}
__host__ Scene::Scene(uint16_t width, uint16_t height)
    : width(width), height(height) {}
__host__ Scene::Scene(uint16_t width, uint16_t height,
                      thrust::device_vector<Shape> shapes)
    : width(width), height(height), shapes(std::move(shapes)) {}

__host__ auto Scene::getWidth() const -> uint16_t { return width; }
__host__ auto Scene::getHeight() const -> uint16_t { return height; }
__host__ auto Scene::getShapes() -> thrust::device_vector<Shape> & {
  return shapes;
}

__host__ auto Scene::addShape(Shape shape) -> void { shapes.push_back(shape); }
