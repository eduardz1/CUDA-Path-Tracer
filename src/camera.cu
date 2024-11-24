#include <cstdint>

#include "cuda_path_tracer/camera.cuh"
#include "cuda_path_tracer/error.cuh"
#include "cuda_path_tracer/ray.cuh"
#include "cuda_path_tracer/vec3.cuh"

/**
 * @brief Kernel for rendering the image, works by calculating the pixel index
 * in the image, computing the Ray that goes from the camera's origin to the
 * pixel center, querying it for a color and then saving this color value in the
 * image buffer.
 *
 * @param width Width of the image
 * @param height Height of the image
 * @param image Image to render
 * @param camera Camera to render the image with
 */
__global__ static void renderImage(const uint16_t width, const uint16_t height,
                                   uchar4 *image, Camera *camera) {
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const auto index = y * width + x;

  const auto r = camera->getRay(x, y);

  const auto color = camera->getColor(r);

  image[index] = color;
}

__host__ Camera::Camera() : origin() {}
__host__ Camera::Camera(const Vec3 &origin) : origin(origin) {}

__host__ void Camera::render(const std::shared_ptr<Scene> &scene,
                             uchar4 *image) {
  const auto width = scene->getWidth();
  const auto height = scene->getHeight();

  viewportWidth = (float(width) / float(height)) * viewportHeight;

  auto viewportU = Vec3(viewportWidth, 0, 0);
  auto viewportV = Vec3(0, -viewportHeight, 0);

  deltaU = viewportU / float(width);
  deltaV = viewportV / float(height);

  pixel00 = (origin - viewportU / 2 - viewportV / 2) + (deltaU + deltaV) / 2;

  CUDA_ERROR_CHECK(
      cudaMalloc(&image, static_cast<long>(width) * height * sizeof(uchar4)));

  renderImage<<<dim3(width / 16, height / 16), dim3(16, 16)>>>(width, height,
                                                               image, this);
}

__device__ auto Camera::getRay(const uint16_t x, const uint16_t y) -> Ray {
  auto center = pixel00 + deltaU * x + deltaV * y;
  return {origin, center - origin};
}

__device__ auto Camera::getColor(const Ray &ray) -> uchar4 {
  // TODO: Implement this
  return make_uchar4(0, 0, 0, UCHAR_MAX);
}