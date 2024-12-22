#include <climits>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#include "cuda_path_tracer/camera.cuh"
#include "cuda_path_tracer/error.cuh"
#include "cuda_path_tracer/hit_info.cuh"
#include "cuda_path_tracer/image.cuh"
#include "cuda_path_tracer/ray.cuh"
#include "cuda_path_tracer/shape.cuh"
#include "cuda_path_tracer/vec3.cuh"

namespace {
constexpr unsigned long long SEED = 0xba0bab;
constexpr dim3 BLOCK_SIZE(16, 16);
constexpr uint16_t MAX_NUM_SHAPES_BEFORE_PARALLEL = 128;

__device__ auto getRay(const Vec3 origin, const Vec3 pixel00, const Vec3 deltaU,
                       const Vec3 deltaV, const uint16_t x, const uint16_t y,
                       curandState &state) -> Ray {
  // We sample an area of "half pixel" around the pixel centers
  auto offset =
      Vec3{curand_uniform(&state) - 0.5f, curand_uniform(&state) - 0.5f, 0};
  auto sample = pixel00 + ((float(x) + offset.getX()) * deltaU) +
                ((float(y) + offset.getY()) * deltaV);

  return {origin, sample - origin};
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
                          const uint16_t num_shapes, HitInfo &hi) -> bool {
  auto tmp = HitInfo();
  auto closest = RAY_T_MAX;
  auto hit_anything = false;

  for (auto i = 0; i < num_shapes; i++) {
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
                         const uint16_t num_shapes) -> Vec3 {
  auto hi = HitInfo();
  const bool hit = hitShapes(ray, shapes, num_shapes, hi);

  if (hit) {
    return 0.5f * (hi.getNormal() + 1.0f);
  }

  auto unit_direction = makeUnitVector(ray.getDirection());
  auto t = 0.5f * (unit_direction.getY() + 1.0f);
  return (1.0f - t) * Vec3{1.0f} + t * Vec3{0.5f, 0.7f, 1.0f};
}

__global__ void getColorParallel(const Ray &ray, const Shape *shapes,
                                 const uint16_t num_shapes, Vec3 *color) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= num_shapes) {
    return;
  }

  auto hi = HitInfo();

  const bool hit = cuda::std::visit(
      [&ray, &hi](const auto &shape) {
        return shape.hit(ray, RAY_T_MIN, RAY_T_MAX, hi);
      },
      shapes[index]);

  if (hit) {
    color[index] = 0.5f * (hi.getNormal() + 1.0f);
  } else {
    auto unit_direction = makeUnitVector(ray.getDirection());
    auto t = 0.5f * (unit_direction.getY() + 1.0f);
    color[index] = (1.0f - t) * Vec3{1.0f} + t * Vec3{0.5f, 0.7f, 1.0f};
  }
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
 * @param origin Camera's origin
 * @param pixel00 Pixel at the top left corner of the image
 * @param deltaU Horizontal vector of the distance between each pixel center
 * @param deltaV Vertical vector of the distance between each pixel center
 * @param shapes Array of shapes to check for hits
 * @param num_shapes Number of shapes in the array
 * @param states Random number generator states for each pixel
 * @param num_samples_ppx Number of samples for each pixel
 */
__global__ void renderImage(const uint16_t width, const uint16_t height,
                            uchar4 *image, const Vec3 origin,
                            const Vec3 pixel00, const Vec3 deltaU,
                            const Vec3 deltaV, const Shape *shapes,
                            const uint16_t num_shapes, curandState *states,
                            const uint8_t num_samples_ppx) {
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const auto index = y * width + x;
  curand_init(SEED, index, 0, &states[index]);

  curandState state = states[index];
  auto color = Vec3{};
  for (auto s = 0; s < num_samples_ppx; s++) {
    const auto ray = getRay(origin, pixel00, deltaU, deltaV, x, y, state);

    // If the number of shapes is small enough a simple for loop is faster
    if (num_shapes < MAX_NUM_SHAPES_BEFORE_PARALLEL) {
      color += getColor(ray, shapes, num_shapes);
    } else {
      int block_size = 0;
      int min_grid_size = 0;
      int grid_size = 0;

      cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                         getColorParallel, 0, 0);
      grid_size = (num_shapes + block_size - 1) / block_size;

      extern __shared__ Vec3 colors[];
      auto shared_memory_size = num_shapes * sizeof(Vec3);
      getColorParallel<<<grid_size, block_size, shared_memory_size>>>(
          ray, shapes, num_shapes, colors);
      __syncthreads();
      // TODO: Reduction to find the closest hit (suggestion:
      // https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/reduction/reduction_kernel.cu)
      __syncthreads();
    }
  }

  image[index] = convertColorTo8Bit(color / float(num_samples_ppx));
}
} // namespace

__host__ Camera::Camera() : origin() {}
__host__ Camera::Camera(const Vec3 &origin) : origin(origin) {}

__host__ void Camera::render(const std::shared_ptr<Scene> &scene,
                             uchar4 *image) {
  const auto width = scene->getWidth();
  const auto height = scene->getHeight();
  curandState *states;
  CUDA_ERROR_CHECK(
      cudaMalloc((void **)&states, width * height * sizeof(curandState)));

  const std::vector<Shape> &h_shapes = scene->getShapes();
  const uint16_t num_shapes = h_shapes.size();
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

  pixel00 =
      (origin - viewportU / 2 - viewportV / 2 + origin) + (deltaU + deltaV) / 2;

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

  renderImage<<<grid_size, block_size>>>(
      width, height, image_device, origin, pixel00, deltaU, deltaV, d_shapes,
      num_shapes, states, this->num_samples_ppx);
  cudaDeviceSynchronize();
  CUDA_ERROR_CHECK(cudaGetLastError());

  CUDA_ERROR_CHECK(
      cudaMemcpy(image, image_device, size, cudaMemcpyDeviceToHost));

  cudaFree(d_shapes);
  CUDA_ERROR_CHECK(cudaGetLastError());
}
