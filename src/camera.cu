#include <cstdint>

#include "cuda_path_tracer/camera.cuh"
#include "cuda_path_tracer/error.cuh"
#include "cuda_path_tracer/ray.cuh"
#include "cuda_path_tracer/vec3.cuh"

namespace {
constexpr dim3 BLOCK_SIZE(16, 16);

__device__ auto getRay(const Vec3 origin, const Vec3 pixel00, const Vec3 deltaU,
                       const Vec3 deltaV, const uint16_t x,
                       const uint16_t y) -> Ray {
  auto center = pixel00 + deltaU * x + deltaV * y;
  return {origin, center - origin};
}

__device__ auto getColor(const Ray &ray, Shape **shapes,
                         const uint16_t numShapes) -> uchar4 {
  // printf("%p:\n", shapes[0]);  
  // if (shapes != nullptr){// && shapes[0] != nullptr && shapes[0]->hit(ray)) {
  //     return make_uchar4(1, 0, 0, UCHAR_MAX);
  // }

  return make_uchar4(0, 0, 1, UCHAR_MAX);
}

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
__global__ void renderImage(const uint16_t width, const uint16_t height,
                            uchar4 *image, const Vec3 origin,
                            const Vec3 pixel00, const Vec3 deltaU,
                            const Vec3 deltaV, Shape **shapes, const uint16_t numShapes) {
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const auto index = y * width + x;
  const auto ray = getRay(origin, pixel00, deltaU, deltaV, x, y);
  image[index] = getColor(ray, shapes, numShapes);
}
} // namespace

__host__ Camera::Camera() : origin() {}
__host__ Camera::Camera(const Vec3 &origin) : origin(origin) {}

__host__ void Camera::render(const std::shared_ptr<Scene> &scene,
                             uchar4 *image) {
  const auto width = scene->getWidth();
  const auto height = scene->getHeight();
  std::vector<Shape *> &h_shapes = scene->getShapes();
  const auto num_shapes = h_shapes.size();
  
  Shape **d_shapes;
  CUDA_ERROR_CHECK(cudaMalloc(&d_shapes, num_shapes * sizeof(Shape *)));
  for (int i = 0; i < num_shapes; i++) {
    Shape *d_shape;
    CUDA_ERROR_CHECK(cudaMalloc(&d_shape, sizeof(Shape)));
    CUDA_ERROR_CHECK(cudaMemcpy(d_shape, h_shapes[i], sizeof(Shape),
                                cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(&d_shapes[i], &d_shape, sizeof(Shape *),
                                cudaMemcpyHostToDevice));
  }
 
  viewportWidth = (float(width) / float(height)) * viewportHeight;

  auto viewportU = Vec3(viewportWidth, 0, 0);
  auto viewportV = Vec3(0, -viewportHeight, 0);

  deltaU = viewportU / float(width);
  deltaV = viewportV / float(height);

  pixel00 = (origin - viewportU / 2 - viewportV / 2) + (deltaU + deltaV) / 2;

  uchar4 *image_device;

  const auto size = static_cast<long>(width) * height * sizeof(uchar4);

  CUDA_ERROR_CHECK(cudaMalloc((void **)&image_device, size));

  dim3 grid((width + BLOCK_SIZE.x - 1) / BLOCK_SIZE.x,
            (height + BLOCK_SIZE.y - 1) / BLOCK_SIZE.y);

  renderImage<<<grid, BLOCK_SIZE>>>(width, height, image_device, origin,
                                    pixel00, deltaU, deltaV, d_shapes,
                                    num_shapes);

  CUDA_ERROR_CHECK(cudaGetLastError());
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());

  for (int i = 0; i < h_shapes.size(); i++) {
    CUDA_ERROR_CHECK(cudaFree(d_shapes[i]));
  }
  CUDA_ERROR_CHECK(cudaFree(d_shapes));
  
  CUDA_ERROR_CHECK(
      cudaMemcpy(image, image_device, size, cudaMemcpyDeviceToHost));
  CUDA_ERROR_CHECK(cudaGetLastError());
}
