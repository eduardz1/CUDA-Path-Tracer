#include <utility>

#include "cuda_path_tracer/camera.cuh"
#include "cuda_path_tracer/error.cuh"
#include "cuda_path_tracer/ray.cuh"

__host__ Camera::Camera() : scene(nullptr), origin(Vec3(0, 0, 0)) {}
__host__ Camera::Camera(const Vec3 &origin, std::shared_ptr<Scene> scene)
    : scene(std::move(scene)), origin(origin) {
  viewportWidth =
      (float(this->scene->getWidth()) / float(this->scene->getHeight())) *
      viewportHeight;
}

/**
 * @brief Kernel for rendering the image
 *
 * @param width Width of the image
 * @param height Height of the image
 * @param image Image to render
 */
__global__ static void renderImage(uint16_t width, uint16_t height,
                                   uchar4 *image) {
  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  auto y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  auto index = y * width + x;

  // Ray const r(Vec3(0, 0, 0), Vec3(0, 0, 1));

  // Save the pixel for the R G B and Alpha values
  image[index] =
      make_uchar4(UCHAR_MAX, 0, 0, UCHAR_MAX); // TODO: Make it query a ray
}

__host__ void Camera::render() const {
  // Allocate memory for the image
  //   uchar4 *image;
  //   cudaMallocManaged(&image, width * height * sizeof(uchar4));
  renderImage<<<dim3(scene->getWidth() / 16, scene->getHeight() / 16),
                dim3(16, 16)>>>(scene->getWidth(), scene->getHeight(),
                                scene->getImage());
}
