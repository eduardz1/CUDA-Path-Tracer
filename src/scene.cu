#include "cuda_path_tracer/error.cuh"
#include "cuda_path_tracer/scene.cuh"
#include <cstdint>

__host__ Scene::Scene() : width(1), height(1) {
  CUDA_ERROR_CHECK(cudaMalloc(&image, sizeof(uchar4)));
}

__host__ Scene::Scene(uint16_t width, uint16_t height)
    : width(width), height(height) {
  CUDA_ERROR_CHECK(
      cudaMalloc(&image, static_cast<long>(width) * height * sizeof(uchar4)));
}

__host__ __device__ auto Scene::getWidth() -> uint16_t { return width; }
__host__ __device__ auto Scene::getHeight() -> uint16_t { return height; }
__host__ __device__ auto Scene::getShapes() -> std::vector<Shape *> & {
  return shapes;
}
__host__ __device__ auto Scene::getImage() -> uchar4 * { return image; }

__host__ auto Scene::addShape(Shape *shape) -> void { shapes.push_back(shape); }

__host__ Scene::~Scene() {
  CUDA_ERROR_CHECK(cudaFree(image));
  // for (auto shape : shapes) {
  //   delete shape;
  // }
}