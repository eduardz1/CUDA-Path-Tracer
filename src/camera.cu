#include <climits>
#include <cstdint>
#include <cuda_runtime_api.h>

#include "cuda_path_tracer/camera.cuh"
#include "cuda_path_tracer/error.cuh"
#include "cuda_path_tracer/hit_info.cuh"
#include "cuda_path_tracer/ray.cuh"
#include "cuda_path_tracer/shape.cuh"
#include "cuda_path_tracer/vec3.cuh"

namespace {
constexpr dim3 BLOCK_SIZE(16, 16);

__device__ auto getRay(const Vec3 origin, const Vec3 pixel00, const Vec3 deltaU,
                       const Vec3 deltaV, const uint16_t x, const uint16_t y)
    -> Ray {
  auto center = pixel00 + deltaU * x + deltaV * y;
  return {origin, center - origin};
}

/**
 * @brief Saves the closest hit information in the HitInfo struct from the given
 * ray and shapes. Returns true if a hit was found, false otherwise.
 *
 * @param ray Ray to check for hits
 * @param shapes Array of shapes to check for hits
 * @param num_shapes Number of shapes in the array
 * @param hi HitInfo struct to save the hit information
 * @return bool true if a hit was found, false otherwise
 */
__device__ auto hitShapes(const Ray &ray, const Shape *shapes,
                          const size_t num_shapes, HitInfo &hi) -> bool {
  auto tmp = HitInfo();
  auto closest = RAY_T_MAX;
  auto hit_anything = false;
  for (size_t i = 0; i < num_shapes; i++) {
    const bool hit = cuda::std::visit(
        [&ray, &tmp, closest](const auto &shape) {
          return shape.hit(ray, RAY_T_MIN, closest, tmp);
        },
        shapes[i]);

    if (hit) {
      hit_anything = true;
      closest = tmp.getTime();
      hi = tmp;
    }
  }
  return hit_anything;
}

__device__ auto getColor(const Ray &ray, const Shape *shapes,
                         const size_t num_shapes) -> uchar4 {
  auto hi = HitInfo();
  const bool hit = hitShapes(ray, shapes, num_shapes, hi);

  if (hit) {
    auto normal = hi.getNormal();
    return {static_cast<unsigned char>(UCHAR_MAX * (normal.getX() + 1) / 2),
            static_cast<unsigned char>(UCHAR_MAX * (normal.getY() + 1) / 2),
            static_cast<unsigned char>(UCHAR_MAX * (normal.getZ() + 1) / 2),
            UCHAR_MAX};
  }

  auto unit_direction = makeUnitVector(ray.getDirection());
  auto t = (unit_direction.getY() + 1.0f) / 2;
  // TODO: Fix the background color and maybe separate into a function
  return {static_cast<unsigned char>(UCHAR_MAX * (1.0f - t)),
          static_cast<unsigned char>(UCHAR_MAX * t),
          static_cast<unsigned char>(UCHAR_MAX * t), UCHAR_MAX};
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
                            const Vec3 deltaV, const Shape *shapes,
                            const size_t num_shapes) {
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const auto index = y * width + x;
  const auto ray = getRay(origin, pixel00, deltaU, deltaV, x, y);
  image[index] = getColor(ray, shapes, num_shapes);
}
} // namespace

__host__ Camera::Camera() : origin() {}
__host__ Camera::Camera(const Vec3 &origin) : origin(origin) {}

__host__ void Camera::render(const std::shared_ptr<Scene> &scene,
                             uchar4 *image) {
  const auto width = scene->getWidth();
  const auto height = scene->getHeight();

  const std::vector<Shape> &h_shapes = scene->getShapes();
  const size_t num_shapes = h_shapes.size();
  Shape *d_shapes;
  CUDA_ERROR_CHECK(cudaMalloc((void **)&d_shapes, num_shapes * sizeof(Shape)));
  CUDA_ERROR_CHECK(cudaMemcpy(d_shapes, h_shapes.data(),
                              num_shapes * sizeof(Sphere),
                              cudaMemcpyHostToDevice));

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
  cudaDeviceSynchronize();
  CUDA_ERROR_CHECK(cudaGetLastError());

  CUDA_ERROR_CHECK(
      cudaMemcpy(image, image_device, size, cudaMemcpyDeviceToHost));

  cudaFree(d_shapes);
  CUDA_ERROR_CHECK(cudaGetLastError());
}
