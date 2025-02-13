#include "cuda_path_tracer/camera.cuh"
#include "cuda_path_tracer/hit_info.cuh"
#include "cuda_path_tracer/image.cuh"
#include "cuda_path_tracer/lambertian.cuh"
#include "cuda_path_tracer/ray.cuh"
#include "cuda_path_tracer/shape.cuh"
#include "cuda_path_tracer/utilities.cuh"
#include "cuda_path_tracer/vec3.cuh"
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cuda/std/span>
#include <cuda/std/tuple>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

namespace {
constexpr uint64_t SEED = 0xba0bab;
constexpr dim3 BLOCK_SIZE(8, 8);
constexpr float RENDER_SCALE = 1.0F / (NUM_IMAGES * NUM_SAMPLES);
constexpr uint64_t DEPTH = 20;

__device__ auto randomInUnitDisk(curandStatePhilox4_32_10_t &state) -> Vec3 {
  // Iterate two at a time for better chances at each loop
  // TODO(eduard): calculate 2 at a time with
  // https://stats.stackexchange.com/questions/481543/generating-random-points-uniformly-on-a-disk
  // and why it's faster (or not) compared to reject sampling, which normally
  // would take pi/4 iterations but was already optimised to pi/2
  while (true) {
    const auto values = curand_uniform4(&state);

    const auto p = Vec3{2.0F * values.w - 1.0F, 2.0F * values.x - 1.0F, 0};
    const auto q = Vec3{2.0F * values.y - 1.0F, 2.0F * values.z - 1.0F, 0};

    if (p.getLengthSquared() < 1.0F) {
      return p;
    }
    if (q.getLengthSquared() < 1.0F) {
      return q;
    }
  }
}

__device__ auto defocusDiskSample(curandStatePhilox4_32_10_t &state,
                                  const Vec3 &center, const Vec3 &u,
                                  const Vec3 &v) -> Vec3 {
  auto p = randomInUnitDisk(state);
  return center + p.x * u + p.y * v;
}

__device__ auto get2Rays(const Vec3 &origin, const Vec3 &pixel00,
                         const Vec3 &deltaU, const Vec3 &deltaV,
                         const Vec3 &defocusDiskU, const Vec3 &defocusDiskV,
                         const float defocusAngle, const uint16_t x,
                         const uint16_t y, curandStatePhilox4_32_10_t &state)
    -> cuda::std::tuple<Ray, Ray> {
  const auto values = curand_uniform4(&state);

  // We sample an area of "half pixel" around the pixel centers to achieve
  // anti-aliasing. This helps in reducing the jagged edges by averaging the
  // colors of multiple samples within each pixel, resulting in smoother
  // transitions and more realistic images.
  const auto offsetA = Vec3{values.z - 0.5F, values.w - 0.5F, 0};
  const auto offsetB = Vec3{values.x - 0.5F, values.y - 0.5F, 0};

  const auto sampleA = pixel00 +
                       ((static_cast<float>(x) + offsetA.x) * deltaU) +
                       ((static_cast<float>(y) + offsetA.y) * deltaV);
  const auto sampleB = pixel00 +
                       ((static_cast<float>(x) + offsetB.x) * deltaU) +
                       ((static_cast<float>(y) + offsetB.y) * deltaV);

  auto newOriginA = origin;
  auto newOriginB = origin;

  if (defocusAngle > 0) {
    newOriginA = defocusDiskSample(state, origin, defocusDiskU, defocusDiskV);
    newOriginB = defocusDiskSample(state, origin, defocusDiskU, defocusDiskV);
  }

  const auto directionA = sampleA - newOriginA;
  const auto directionB = sampleB - newOriginB;

  return {{newOriginA, directionA}, {newOriginB, directionB}};
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
__device__ auto hitShapes(const Ray &ray,
                          const cuda::std::span<const Shape> shapes,
                          HitInfo &hi) -> bool {
  auto tmp = HitInfo();
  auto closest = RAY_T_MAX;
  auto hit_anything = false;

  for (const auto &shape : shapes) {
    const bool hit = cuda::std::visit(
        [&ray, &tmp, &closest](const auto &shape) {
          return shape.hit(ray, RAY_T_MIN, closest, tmp);
        },
        shape);

    if (hit) {
      hit_anything = true;
      closest = tmp.time;
      hi = tmp;
    }
  }

  return hit_anything;
}

__device__ auto getColor(const Ray &ray,
                         const cuda::std::span<const Shape> shapes,
                         curandStatePhilox4_32_10_t &state, int depth, Vec3 background) -> Vec3 {

  Vec3 color = Vec3{1.0f, 1.0f, 1.0f};
  Ray current = ray;

  for (int i = 0; i < depth; i++) {
    auto hi = HitInfo();
    bool hit = hitShapes(current, shapes, hi);

    if(i == 0 && !hit){
      return background;
    }

    if (hit) {// could possibly remove the shadow acne problem but this is a little change
      Ray scattered;
      Vec3 attenuation;

      Vec3 normal = hi.normal;
      Vec3 point = hi.point;
      Material material = hi.material;
      bool front = hi.front;

      bool scatter = cuda::std::visit(
          [&current, &normal, &point, front, &attenuation, &scattered,
           &state](auto &material) {
            return material.scatter(current, normal, point, front, attenuation,
                                    scattered, state);
          },
          material);

      if (scatter) {
        color = color * attenuation;
        current = scattered;
      } else {
        return Vec3{0.0f, 0.0f, 0.0f};
      }

    } else {
      auto unit_direction = makeUnitVector(current.getDirection());
      auto t = 0.5f * (unit_direction.y + 1.0f);
      return color * (1.0f - t) * Vec3{1.0f, 1.0f, 1.0f} +
             t * Vec3{0.5f, 0.7f, 1.0f};
    }
  }
  return Vec3{0, 0, 0};
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
 * @param shapes span of shapes to check for hits
 * @param stream_index Index of the stream to use
 */
__global__ void renderImage(const uint16_t width, const uint16_t height,
                            const cuda::std::span<Vec3> image,
                            const Vec3 origin, const Vec3 pixel00,
                            const Vec3 deltaU, const Vec3 deltaV,
                            const Vec3 defocusDiskU, const Vec3 defocusDiskV,
                            const float defocusAngle, const Vec3 background,
                            const cuda::std::span<const Shape> shapes,
                            const size_t stream_index) {
  // TODO(eduard): make use of shared memory

  // Initialize states with the Philox4_32_10_t initializer. This enables us, at
  // the cost of 16 extra bytes per thread, to generate efficiently four random
  // numbers at a time.
  curandStatePhilox4_32_10_t states;

  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const auto index = y * width + x;

  // Random state generation is extremely expensive, generating it with a
  // separate kernel does not improve perfomance, talk about it in the report
  // TODO(eduard): https://docs.nvidia.com/cuda/pdf/CURAND_Library.pdf (3.6)
  // It would be much better, however, to memoize the random states so that
  // rendering multiple images with the same resolution would not require
  // generating the random states again.
  curand_init(SEED, index + (stream_index * width * height), 0, &states);

  auto color = Vec3{};
  for (auto s = 0; s < (NUM_SAMPLES >> 1); s++) {
    const auto ray = get2Rays(origin, pixel00, deltaU, deltaV, defocusDiskU,
                              defocusDiskV, defocusAngle, x, y, states);

    color += getColor(cuda::std::get<0>(ray), shapes, states, DEPTH, background);
    color += getColor(cuda::std::get<1>(ray), shapes, states, DEPTH, background);
  }

  image[index] = color;
}

#if AVERAGE_WITH_THRUST == false
__global__ void averagePixels(const uint16_t width, const uint16_t height,
                              const uint16_t padded_width, const float scale,
                              const cuda::std::span<Vec3> image,
                              cuda::std::span<uchar4> image_out) {
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const auto output_idx = y * width + x;
  const auto padded_idx = y * padded_width + x;

  auto sum = Vec3{};
  for (auto img = 0; img < NUM_IMAGES; img++) {
    sum += images[img * (padded_width * height) + padded_idx];
  }

  image_out[output_idx] = convertColorTo8Bit(sum * scale);
}
#endif
} // namespace

__host__ Camera::Camera() = default;

__host__ void Camera::init(const std::shared_ptr<Scene> &scene) {
  const auto width = scene->getWidth();
  const auto height = scene->getHeight();

  const auto theta = DEGREE_TO_RADIAN(this->verticalFov);
  const auto h = std::tan(theta / 2);

  this->viewportHeight *= h * this->focusDistance;
  this->viewportWidth =
      (static_cast<float>(width) / static_cast<float>(height)) * viewportHeight;

  const auto w = makeUnitVector(this->origin - this->lookAt);
  const auto u = makeUnitVector(cross(this->up, w));
  const auto v = cross(w, u);

  const auto viewportU = u * viewportWidth;
  const auto viewportV = -v * viewportHeight;

  this->deltaU = viewportU / static_cast<float>(width);
  this->deltaV = viewportV / static_cast<float>(height);

  const auto viewportUpperLeft =
      this->origin - (this->focusDistance * w) - viewportU / 2 - viewportV / 2;
  this->pixel00 = 0.5F * (deltaU + deltaV) + viewportUpperLeft;

  const float defocusRadius =
      this->focusDistance *
      std::tan(DEGREE_TO_RADIAN(this->defocusAngle) / 2.0F);
  this->defocusDiskU = u * defocusRadius;
  this->defocusDiskV = v * defocusRadius;
}

// Align to prevent control divergence
// TODO(eduard): Write about it in the report
__host__ inline auto getPaddedSize(size_t size, size_t alignment = WARP_SIZE)
    -> size_t {
  return (size + alignment - 1) & ~(alignment - 1);
}

__host__ void
Camera::render(const std::shared_ptr<Scene> &scene,
               thrust::universal_host_pinned_vector<uchar4> &image) {
  this->init(scene);

  const auto width = scene->getWidth();
  const auto height = scene->getHeight();
  const auto padded_width = getPaddedSize(width);
  const auto padded_height = getPaddedSize(height);
  const auto num_padded_pixels = padded_width * padded_height;

  assert(image.size() == static_cast<size_t>(width * height) &&
         ("Image size does not match the scene's width and height. Actual: " +
          std::to_string(image.size()) +
          ", Expected: " + std::to_string(width * height))
             .c_str());

  thrust::device_vector<Vec3> image_3d(num_padded_pixels * NUM_IMAGES);

  cuda::std::span<const Shape> shapes_span{
      thrust::raw_pointer_cast(scene->getShapes().data()),
      scene->getShapes().size()};
  cuda::std::span<Vec3> image_3d_span{thrust::raw_pointer_cast(image_3d.data()),
                                      image_3d.size()};

  const dim3 grid(std::ceil(padded_width / BLOCK_SIZE.x),
                  std::ceil(padded_height / BLOCK_SIZE.y));

  std::array<StreamGuard, NUM_IMAGES> streams{};

  for (auto i = 0; i < NUM_IMAGES; i++) {
    renderImage<<<grid, BLOCK_SIZE, 0, streams.at(i)>>>(
        padded_width, padded_height,
        image_3d_span.subspan(i * num_padded_pixels), origin, pixel00, deltaU,
        deltaV, defocusDiskU, defocusDiskV, defocusAngle, background, shapes_span, i);
  }

  averageRenderedImages(image, image_3d_span, width, height, padded_width);
}

__host__ void Camera::averageRenderedImages(
    thrust::universal_host_pinned_vector<uchar4> &output,
    const cuda::std::span<Vec3> &images, const uint16_t width,
    const uint16_t height, const uint16_t padded_width) {
#if AVERAGE_WITH_THRUST
  const auto num_pixels = output.size();
  thrust::transform(thrust::make_counting_iterator<size_t>(0),
                    thrust::make_counting_iterator<size_t>(num_pixels),
                    output.begin(), [=] __device__(const auto idx) {
                      const auto row = idx / width;
                      const auto col = idx % width;
                      const auto padded_idx = row * padded_width + col;

                      Vec3 sum{};
                      for (int img = 0; img < NUM_IMAGES; img++) {
                        sum +=
                            images[img * (padded_width * height) + padded_idx];
                      }
                      return convertColorTo8Bit(sum * RENDER_SCALE);
                    });
#else
  const dim3 grid(std::ceil(width / BLOCK_SIZE.x),
                  std::ceil(height / BLOCK_SIZE.y));

  cuda::std::span<uchar4> output_span{thrust::raw_pointer_cast(output.data()),
                                      output.size()};

  averagePixels<<<grid, BLOCK_SIZE>>>(width, height, padded_width, RENDER_SCALE,
                                      images, output_span);
#endif
}

__host__ CameraBuilder::CameraBuilder() = default;
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
__host__ auto CameraBuilder::verticalFov(const float verticalFov)
    -> CameraBuilder & {
  this->camera.verticalFov = verticalFov;
  return *this;
}
__host__ auto CameraBuilder::defocusAngle(const float defocusAngle)
    -> CameraBuilder & {
  this->camera.defocusAngle = defocusAngle;
  return *this;
}
__host__ auto CameraBuilder::background(const Vec3 &background)
    -> CameraBuilder & {
  this->camera.background = background;
  return *this;
}
__host__ auto CameraBuilder::focusDistance(const float focusDistance)
    -> CameraBuilder & {
  this->camera.focusDistance = focusDistance;
  return *this;
}
__host__ auto CameraBuilder::build() -> Camera { return this->camera; }
