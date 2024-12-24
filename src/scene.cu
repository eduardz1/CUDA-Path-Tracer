#include "cuda_path_tracer/scene.cuh"
#include <cstdint>

__host__ Scene::Scene() : width(1), height(1) {}

__host__ Scene::Scene(uint16_t width, uint16_t height)
    : width(width), height(height) {}

__host__ auto Scene::getWidth() -> uint16_t { return width; }
__host__ auto Scene::getHeight() -> uint16_t { return height; }
__host__ auto Scene::getShapes() -> std::vector<Shape> & { return shapes; }

__host__ auto Scene::addShape(Shape shape) -> void {
  shapes.emplace_back(shape);
}
