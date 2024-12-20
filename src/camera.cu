#include <climits>
#include <cstdint>
#include <cuda_runtime_api.h>

#include "cuda_path_tracer/camera.cuh"
#include "cuda_path_tracer/error.cuh"
#include "cuda_path_tracer/hit_info.cuh"
#include "cuda_path_tracer/image.cuh"
#include "cuda_path_tracer/ray.cuh"
#include "cuda_path_tracer/shape.cuh"
#include "cuda_path_tracer/vec3.cuh"

namespace {
constexpr dim3 BLOCK_SIZE(16, 16);
constexpr size_t MAX_NUM_SHAPES_BEFORE_PARALLEL = 128;

__device__ auto getRay(const Vec3 origin, const Vec3 pixel00, const Vec3 deltaU,
                       const Vec3 deltaV, const uint16_t x, const uint16_t y)
    -> Ray {
  auto center = pixel00 + deltaU * x + deltaV * y;
  return {origin, center * 64};
}

__device__ auto _hitShapesSequential(const Ray &ray, const Shape *shapes,
                                     const size_t num_shapes, HitInfo &hi)
    -> bool {
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

__global__ void _hitShapesParallel(const Ray &ray, const Shape *shapes,
                                   HitInfo &hi) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;

  // TODO: check if it's correct to do "&shapes[i]"
  const bool hit = _hitShapesSequential(ray, &shapes[i], 1, hi);
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
  // If the number of shapes is small enough a simple for loop is faster
  if (num_shapes < MAX_NUM_SHAPES_BEFORE_PARALLEL) {
    return _hitShapesSequential(ray, shapes, num_shapes, hi);
  }

  dim3 grid_size((num_shapes + BLOCK_SIZE.x - 1) / BLOCK_SIZE.x,
                 (num_shapes + BLOCK_SIZE.y - 1) / BLOCK_SIZE.y);
  _hitShapesParallel<<<grid_size, BLOCK_SIZE>>>(ray, shapes, hi);
  __syncthreads();
  // TODO: Reduction to find the closest hit
  return false;
}

__device__ auto getColor(const Ray &ray, const Shape *shapes,
                         const size_t num_shapes) -> float4 {
  auto hi = HitInfo();
  const bool hit = hitShapes(ray, shapes, num_shapes, hi);

  if (hit) {
    return 0.5f * (hi.getNormal() + 1.0f);
  }

  auto unit_direction = makeUnitVector(ray.getDirection());
  auto t = 0.5f * (unit_direction.getY() + 1.0f);
  return (1.0f - t) * Vec3{1.0f} + t * Vec3{0.5f, 0.7f, 1.0f};
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
  image[index] = convertColorTo8Bit(getColor(ray, shapes, num_shapes));
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

  pixel00 = (origin - viewportU / 2 - viewportV / 2 +
             origin); //+ (deltaU + deltaV) / 2;

  uchar4 *image_device;

  const auto size = static_cast<size_t>(width) * height * sizeof(uchar4);

  CUDA_ERROR_CHECK(cudaMalloc((void **)&image_device, size));

  // Calculate the optimal block size and grid size for the renderImage kernel

  int block_size = 0;
  int min_grid_size = 0;
  int grid_size = 0;

  CUDA_ERROR_CHECK(cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &block_size, renderImage, 0, 0));

  grid_size = (width * height + block_size - 1) / block_size;

  renderImage<<<grid_size, block_size>>>(width, height, image_device, origin,
                                         pixel00, deltaU, deltaV, d_shapes,
                                         num_shapes);
  cudaDeviceSynchronize();
  CUDA_ERROR_CHECK(cudaGetLastError());

  CUDA_ERROR_CHECK(
      cudaMemcpy(image, image_device, size, cudaMemcpyDeviceToHost));

  cudaFree(d_shapes);
  CUDA_ERROR_CHECK(cudaGetLastError());
}
