#include "cuda_path_tracer/camera.cuh"
#include "cuda_path_tracer/error.cuh"
#include "cuda_path_tracer/hit_info.cuh"
#include "cuda_path_tracer/image.cuh"
#include "cuda_path_tracer/ray.cuh"
#include "cuda_path_tracer/shape.cuh"
#include "cuda_path_tracer/utilities.cuh"
#include "cuda_path_tracer/vec3.cuh"
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cuda/std/span>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

namespace {
constexpr unsigned long long SEED = 0xba0bab;
constexpr dim3 BLOCK_SIZE(8, 8);

__device__ auto randomInUnitDisk(curandState &state) -> Vec3 {
  while (true) {
    auto p = Vec3{2.0f * curand_uniform(&state) - 1.0f,
                  2.0f * curand_uniform(&state) - 1.0f, 0};

    if (p.getLengthSquared() < 1.0f) {
      return p;
    }
  }
}

__device__ auto defocusDiskSample(curandState &state, const Vec3 &center,
                                  const Vec3 &u, const Vec3 &v) -> Vec3 {
  auto p = randomInUnitDisk(state);
  return center + p.x * u + p.y * v;
}

__device__ auto getRay(const Vec3 &origin, const Vec3 &pixel00,
                       const Vec3 &deltaU, const Vec3 &deltaV,
                       const Vec3 &defocusDiskU, const Vec3 &defocusDiskV,
                       const float defocusAngle, const uint16_t x,
                       const uint16_t y, curandState &state) -> Ray {
  // We sample an area of "half pixel" around the pixel centers
  auto offset =
      Vec3{curand_uniform(&state) - 0.5f, curand_uniform(&state) - 0.5f, 0};

  auto sample = pixel00 + ((float(x) + offset.x) * deltaU) +
                ((float(y) + offset.y) * deltaV);

  auto newOrigin =
      defocusAngle <= 0
          ? origin
          : defocusDiskSample(state, origin, defocusDiskU, defocusDiskV);
  auto direction = sample - newOrigin;

  return {newOrigin, direction};
}

/**
 * @brief Saves the closest hit information in the HitInfo struct from the
 * given ray and shapes. Returns true if a hit was found, false otherwise.
 *
 * @param ray Ray to check for hits
 * @param shapes Array of shapes to check for hits
 * @param num_shapes Number of shapes in the array
 * @param hi HitInfo struct to save the hit information
 * @return bool true if a hit was found, false otherwise
 */
__device__ auto hitShapes(const Ray &ray, cuda::std::span<const Shape> shapes,
                          HitInfo &hi) -> bool {
  auto tmp = HitInfo();
  auto closest = RAY_T_MAX;
  auto hit_anything = false;

  for (const auto &shape : shapes) {
    const bool hit = cuda::std::visit(
        [&ray, &tmp, closest](const auto &shape) {
          return shape.hit(ray, RAY_T_MIN, closest, tmp);
        },
        shape);

    if (hit) {
      hit_anything = true;
      closest = tmp.getTime();
      hi = tmp;
    }
  }

  return hit_anything;
}

__device__ auto getColor(const Ray &ray,
                         cuda::std::span<const Shape> shapes) -> Vec3 {
  auto hi = HitInfo();
  const bool hit = hitShapes(ray, shapes, hi);

  if (hit) {
    return 0.5f * (hi.getNormal() + 1.0f);
  }

  auto unit_direction = makeUnitVector(ray.getDirection());
  auto t = 0.5f * (unit_direction.y + 1.0f);
  return (1.0f - t) * Vec3{1.0f} + t * Vec3{0.5f, 0.7f, 1.0f};
}

/**
 * @brief Kernel for rendering the image, works by calculating the pixel index
 * in the image, computing the Ray that goes from the camera's origin to the
 * pixel center, querying it for a color and then saving this color value in
 * the image buffer.
 *
 * @param width Width of the image
 * @param height Height of the image
 * @param image Image to render
 * @param origin Camera's origin
 * @param pixel00 Pixel at the top left corner of the image
 * @param deltaU Horizontal vector of the distance between each pixel center
 * @param deltaV Vertical vector of the distance between each pixel center
 * @param defocusDiskU Horizontal vector of the defocus disk
 * @param defocusDiskV Vertical vector of the defocus disk
 * @param defocusAngle Angle of the defocus disk
 * @param shapes Array of shapes to check for hits
 * @param num_shapes Number of shapes in the array
 * @param states Random number generator states for each pixel
 */
__global__ void renderImage(const uint16_t width, const uint16_t height,
                            cuda::std::span<Vec3> image, const Vec3 origin,
                            const Vec3 pixel00, const Vec3 deltaU,
                            const Vec3 deltaV, const Vec3 defocusDiskU,
                            const Vec3 defocusDiskV, const float defocusAngle,
                            cuda::std::span<const Shape> shapes,
                            cuda::std::span<curandState> states,
                            const size_t stream_index) {
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const auto index = y * width + x;
  curand_init(SEED, index + (stream_index * width * height), 0, &states[index]);

  curandState state = states[index];
  auto color = Vec3{};
  for (auto s = 0; s < NUM_SAMPLES; s++) {
    const auto ray = getRay(origin, pixel00, deltaU, deltaV, defocusDiskU,
                            defocusDiskV, defocusAngle, x, y, state);
    color += getColor(ray, shapes);
  }

  image[index] = color;
}

// __global__ void averagePixels(const uint16_t width, const uint16_t height,
//                               const float scale, const Vec3 *image,
//                               uchar4 *image_out) {
//   const auto x = blockIdx.x * blockDim.x + threadIdx.x;
//   const auto y = blockIdx.y * blockDim.y + threadIdx.y;

//   if (x >= width || y >= height) {
//     return;
//   }

//   const auto index = y * width + x;

//   auto sum = Vec3{};
//   for (auto i = 0; i < NUM_IMAGES; i++) {
//     const auto index_stream = i * width * height + index;
//     sum += image[index_stream];
//   }

//   image_out[index] = convertColorTo8Bit(sum * scale);
// }
} // namespace

__host__ Camera::Camera() : origin() {}

__host__ void Camera::init(const std::shared_ptr<Scene> &scene) {
  const auto width = scene->getWidth();
  const auto height = scene->getHeight();

  const auto theta = DEGREE_TO_RADIAN(this->verticalFov);
  const auto h = std::tan(theta / 2);

  this->viewportHeight *= h * this->focusDistance;
  this->viewportWidth = (float(width) / float(height)) * viewportHeight;

  const auto w = makeUnitVector(this->origin - this->lookAt);
  const auto u = makeUnitVector(cross(this->up, w));
  const auto v = cross(w, u);

  const auto viewportU = u * viewportWidth;
  const auto viewportV = -v * viewportHeight;

  this->deltaU = viewportU / float(width);
  this->deltaV = viewportV / float(height);

  const auto viewportUpperLeft =
      this->origin - (this->focusDistance * w) - viewportU / 2 - viewportV / 2;
  this->pixel00 = 0.5f * (deltaU + deltaV) + viewportUpperLeft;

  const float defocusRadius =
      this->focusDistance *
      std::tan(DEGREE_TO_RADIAN(this->defocusAngle) / 2.0f);
  this->defocusDiskU = u * defocusRadius;
  this->defocusDiskV = v * defocusRadius;
}

__host__ void Camera::render(const std::shared_ptr<Scene> &scene,
                             universal_host_pinned_vector<uchar4> &image) {
  this->init(scene);

  const auto width = scene->getWidth();
  const auto height = scene->getHeight();

  const auto num_pixels = static_cast<size_t>(width * height);

  std::array<cudaStream_t, NUM_IMAGES> streams{};
  for (auto &stream : streams) {
    CUDA_ERROR_CHECK(cudaStreamCreate(&stream));
  }

  const auto shapes = scene->getShapes();
  thrust::device_vector<curandState> states(num_pixels * NUM_IMAGES);
  thrust::device_vector<Vec3> image_3d(num_pixels * NUM_IMAGES);

  cuda::std::span<curandState> states_span{
      thrust::raw_pointer_cast(states.data()), states.size()};
  cuda::std::span<const Shape> shapes_span{
      thrust::raw_pointer_cast(shapes.data()), shapes.size()};
  cuda::std::span<Vec3> image_3d_span{thrust::raw_pointer_cast(image_3d.data()),
                                      image_3d.size()};

  dim3 grid((width + BLOCK_SIZE.x - 1) / BLOCK_SIZE.x,
            (height + BLOCK_SIZE.y - 1) / BLOCK_SIZE.y);

  for (auto i = 0; i < NUM_IMAGES; i++) {
    renderImage<<<grid, BLOCK_SIZE, 0, streams.at(i)>>>(
        width, height, image_3d_span.subspan(i * num_pixels), origin, pixel00,
        deltaU, deltaV, defocusDiskU, defocusDiskV, defocusAngle, shapes_span,
        states_span.subspan(i * num_pixels), i);
  }

  constexpr float scale = 1.0f / (NUM_IMAGES * NUM_SAMPLES);

  // averagePixels<<<grid, BLOCK_SIZE>>>(width, height, scale, image_3d_ptr,
  //                                     image_ptr);
  // CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  // CUDA_ERROR_CHECK(cudaGetLastError());
  thrust::transform(thrust::make_counting_iterator<size_t>(0),
                    thrust::make_counting_iterator<size_t>(num_pixels),
                    image.begin(), [=] __device__(const auto pixel_idx) {
                      Vec3 sum{};
                      for (int img = 0; img < NUM_IMAGES; img++) {
                        sum += image_3d_span[img * num_pixels + pixel_idx];
                      }
                      return convertColorTo8Bit(sum * scale);
                    });

  for (auto &stream : streams) {
    CUDA_ERROR_CHECK(cudaStreamDestroy(stream));
  }
}

__host__ CameraBuilder::CameraBuilder() : camera() {}
__host__ auto CameraBuilder::origin(const Vec3 &origin) -> CameraBuilder & {
  this->camera.origin = origin;
  return *this;
}
__host__ auto CameraBuilder::lookAt(const Vec3 &lookAt) -> CameraBuilder & {
  this->camera.lookAt = lookAt;
  return *this;
}
__host__ auto CameraBuilder::up(const Vec3 &up) -> CameraBuilder & {
  this->camera.up = up;
  return *this;
}
__host__ auto
CameraBuilder::verticalFov(const float verticalFov) -> CameraBuilder & {
  this->camera.verticalFov = verticalFov;
  return *this;
}
__host__ auto
CameraBuilder::defocusAngle(const float defocusAngle) -> CameraBuilder & {
  this->camera.defocusAngle = defocusAngle;
  return *this;
}
__host__ auto
CameraBuilder::focusDistance(const float focusDistance) -> CameraBuilder & {
  this->camera.focusDistance = focusDistance;
  return *this;
}
__host__ auto CameraBuilder::build() -> Camera { return this->camera; }
