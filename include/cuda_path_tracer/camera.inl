#pragma once

#include "cuda_path_tracer/camera.cuh"
#include "cuda_path_tracer/color.cuh"
#include "cuda_path_tracer/error.cuh"
#include "cuda_path_tracer/utilities.cuh"
#include <algorithm>
#include <cstdint>
#include <sys/types.h>

namespace {
constexpr uint64_t SEED = 0xba0bab; // Seed for the random number generator
constexpr uint64_t MIN_DEPTH = 3;   // Minimum depth for Russian Roulette

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

/**
 * @brief Returns the emitted color of the hit material.
 *
 * @param hi HitInfo struct containing the hit information
 * @return Color Emitted color of the hit material
 */
__device__ auto getEmittedColor(const HitInfo &hi) -> Color {
  return cuda::std::visit( // At the moment only Light has an emitted color
      overload{[](const Light &light) { return light.emitted(); },
               [](const auto &) { return Colors::Black; }},
      hi.material);
}

/**
 * @brief Tries to scatter the ray with the hit material and updates the
 * attenuation color and the scattered ray accordingly.
 *
 * @tparam State curand state type
 * @param ray Ray to scatter
 * @param hi HitInfo struct containing the hit information
 * @param attenuation Attenuation color
 * @param scattered Scattered ray
 * @param state curand state
 * @return true if the ray was scattered, false otherwise
 */
template <typename State>
__device__ auto tryScatter(const Ray &ray, const HitInfo &hi,
                           Color &attenuation, Ray &scattered,
                           State &state) -> bool {
  return cuda::std::visit(
      overload{
          [&hi, &attenuation, &scattered, &state](const Lambertian &material) {
            return material.scatter<State>(hi.normal, hi.point, attenuation,
                                           scattered, state);
          },
          [&ray, &hi, &attenuation, &scattered, &state](const Metal &material) {
            return material.scatter<State>(ray, hi.normal, hi.point,
                                           attenuation, scattered, state);
          },
          [&ray, &hi, &attenuation, &scattered,
           &state](const Dielectric &material) {
            return material.scatter(ray, hi.normal, hi.point, hi.front,
                                    attenuation, scattered, state);
          },
          [](const auto &) { return false; }}, // Light
      hi.material);
}

/**
 * @brief Computes the color corresponding to the given ray by tracing it
 * through the scene, meaning that it checks for hits with the shapes and
 * bounces the ray iteratively until it reaches the maximum depth or it's forced
 * to terminate early by the Russian Roulette.
 *
 * @tparam State curand state type
 * @tparam Depth Maximum depth of the ray
 * @param ray Ray to compute the color for
 * @param shapes Collection of shapes to check for hits
 * @param state curand state
 * @param background Background color
 * @return Color Color of the ray
 */
template <typename State, uint16_t Depth>
__device__ auto getColor(const Ray &ray,
                         const cuda::std::span<const Shape> shapes,
                         State &state, const Color background) -> Color {
  Vec3 throughput{1.0F};
  Vec3 color{0.0F};
  Ray current = ray;

  for (int i = 0; i < Depth; i++) {
    HitInfo hi{};
    const bool hit = hitShapes(current, shapes, hi);

    if (!hit) {
      color += throughput * background;
      break;
    }

    const auto emitted = getEmittedColor(hi);
    color += throughput * emitted;

    Ray scattered;
    Color attenuation;
    const bool scatter =
        tryScatter<State>(current, hi, attenuation, scattered, state);

    if (!scatter) {
      break;
    }

    throughput *= attenuation;
    current = scattered;

    // Russian roulette to terminate early if the throughput is too low (carries
    // very little information)
    if (i > MIN_DEPTH) {
      const float p = std::max({throughput.x, throughput.y, throughput.z});
      if (curand_uniform(&state) > p) {
        break;
      }
      // Adds the energy we lost by randomly terminating early
      throughput /= p;
    }
  }

  return {color};
}

/**
 * @brief Similar to the getColor function, but computes the color for 4 rays at
 * a time.
 *
 * @tparam State curand state type
 * @tparam Depth Maximum depth of the ray
 * @param rays Tuple of 4 rays to compute the color for
 * @param shapes Collection of shapes to check for hits
 * @param state curand state
 * @param background Background color
 * @return std::tuple<Color, Color, Color, Color> Tuple of colors for the 4 rays
 */
template <typename State, uint16_t Depth>
__device__ auto
get4Colors(const std::tuple<Ray, Ray, Ray, Ray> &rays,
           const cuda::std::span<const Shape> shapes, State &state,
           const Color background) -> std::tuple<Color, Color, Color, Color> {

  cuda::std::array<Vec3, 4> throughput = {Vec3{1.0F}, Vec3{1.0F}, Vec3{1.0F},
                                          Vec3{1.0F}};
  cuda::std::array<Vec3, 4> colors = {Vec3{0.0F}, Vec3{0.0F}, Vec3{0.0F},
                                      Vec3{0.0F}};
  cuda::std::array<Ray, 4> currents = {std::get<0>(rays), std::get<1>(rays),
                                       std::get<2>(rays), std::get<3>(rays)};
  cuda::std::array<bool, 4> active = {true, true, true, true};
  int active_count = 4;

  for (int d = 0; d < Depth && active_count > 0; d++) {
    cuda::std::array<HitInfo, 4> hits;

#pragma unroll
    for (int r = 0; r < 4; r++) {
      if (!active[r]) {
        continue;
      }

      const bool hit = hitShapes(currents[r], shapes, hits[r]);

      if (!hit) {
        colors[r] += throughput[r] * background;
        active[r] = false;
        active_count--;
        continue;
      }

      const auto emitted = getEmittedColor(hits[r]);
      colors[r] += throughput[r] * emitted;

      Ray scattered;
      Color attenuation;
      const bool scatter = tryScatter<State>(currents[r], hits[r], attenuation,
                                             scattered, state);

      if (!scatter) {
        active[r] = false;
        active_count--;
        continue;
      }

      throughput[r] *= attenuation;
      currents[r] = scattered;

      // Russian roulette
      if (d > MIN_DEPTH) {
        const float p =
            std::max({throughput[r].x, throughput[r].y, throughput[r].z});
        if (curand_uniform(&state) > p) {
          active[r] = false;
          active_count--;
          continue;
        }
        throughput[r] /= p;
      }
    }
  }

  return {Color{colors[0]}, Color{colors[1]}, Color{colors[2]},
          Color{colors[3]}};
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
template <typename State, uint16_t NumSamples, uint16_t Depth>
__global__ void
renderImage(const uint16_t width, const uint16_t height,
            const cuda::std::span<Vec3> image, const Vec3 origin,
            const Vec3 pixel00, const Vec3 deltaU, const Vec3 deltaV,
            const Vec3 defocusDiskU, const Vec3 defocusDiskV,
            const float defocusAngle, const cuda::std::span<const Shape> shapes,
            const size_t stream_index, const Color background) {
  State states; // NOLINT

  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const auto index = y * width + x;

  curand_init(SEED, index + (stream_index * width * height), 0, &states);

  constexpr auto SAMPLE_SCALE = 1.0F / static_cast<float>(NumSamples);
  auto color = Vec3{};

  // Evaluates at compile time the type of the state
  if constexpr (std::is_same_v<State, curandStatePhilox4_32_10_t>) {
    // When using the Philox4_32_10_t generator, we can generate 4 random
    // uniformly distributed numbers at once, meaning that we can generate 4
    // rays at once
    for (auto s = 0; s < (NumSamples >> 2); s++) {
      const auto ray = get4Rays(origin, pixel00, deltaU, deltaV, defocusDiskU,
                                defocusDiskV, defocusAngle, x, y, states);

      const auto colors =
          get4Colors<State, Depth>(ray, shapes, states, background);

      color += (cuda::std::get<0>(colors) + cuda::std::get<1>(colors) +
                cuda::std::get<2>(colors) + cuda::std::get<3>(colors)) *
               SAMPLE_SCALE;
    }
  } else {
    for (auto s = 0; s < NumSamples; s++) {
      const auto ray = getRay(origin, pixel00, deltaU, deltaV, defocusDiskU,
                              defocusDiskV, defocusAngle, x, y, states);

      color += getColor<State, Depth>(ray, shapes, states, background) *
               SAMPLE_SCALE;
    }
  }

  image[index] = color;
}

/**
 * @brief Kernel for averaging the pixel values of the images to get the final
 * image.
 *
 * @tparam NumImages number of images to average
 * @param width width of the output image
 * @param height height of the output image
 * @param padded_width width of the padded image, used to account for alignment
 * @param images collection of images to average
 * @param image_out output image
 */
template <uint16_t NumImages>
__global__ void averagePixels(const uint16_t width, const uint16_t height,
                              const uint16_t padded_width,
                              const cuda::std::span<Vec3> images,
                              cuda::std::span<uchar4> image_out) {
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const auto output_idx = y * width + x;
  const auto padded_idx = y * padded_width + x;

  constexpr float IMAGE_SCALE = 1.0F / static_cast<float>(NumImages);
  auto sum = Vec3{};
  for (auto img = 0; img < NumImages; img++) {
    sum += images[img * (padded_width * height) + padded_idx];
  }

  image_out[output_idx] = Color(sum * IMAGE_SCALE).correctGamma().to8Bit();
}

/**
 * @brief Returns the next multiple of the alignment that is greater than or
 * equal to the given size. This is done to reduce control divergence in the
 * kernel.
 *
 * @param size Size to align
 * @param alignment Alignment value (default is WARP_SIZE)
 * @return size_t Aligned size
 */
__host__ constexpr auto getPaddedSize(size_t size,
                                      size_t alignment = WARP_SIZE) -> size_t {
  return (size + alignment - 1) & ~(alignment - 1);
}
} // namespace

template <typename Params>
__host__ Camera<Params>::Camera()
    : CameraInterface(), viewportWidth(2.0F), defocusAngle(0.0F),
      // NOLINTNEXTLINE
      focusDistance(10.0F), verticalFov(90.0F), background(Colors::Black) {}

template <typename Params>
__host__ void Camera<Params>::init(const std::shared_ptr<Scene> &scene) {
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

template <typename Params>
__host__ void
Camera<Params>::render(const std::shared_ptr<Scene> &scene,
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

  // Calling cudaGetLastError() here to clear any previous errors
  CUDA_ERROR_CHECK(cudaGetLastError());
  for (auto i = 0; i < NUM_IMAGES; i++) {
    renderImage<State, NUM_SAMPLES, DEPTH>
        <<<grid, BLOCK_SIZE, 0, streams.at(i)>>>(
            padded_width, padded_height,
            image_3d_span.subspan(i * num_padded_pixels), origin, pixel00,
            deltaU, deltaV, defocusDiskU, defocusDiskV, defocusAngle,
            shapes_span, i, background);
  }
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_ERROR_CHECK(cudaGetLastError());

  averageRenderedImages(image, image_3d_span, width, height, padded_width);
}

template <typename Params>
__host__ void Camera<Params>::averageRenderedImages(
    thrust::universal_host_pinned_vector<uchar4> &output,
    const cuda::std::span<Vec3> &images, const uint16_t width,
    const uint16_t height, const uint16_t padded_width) {

  if constexpr (AVG_WITH_THRUST) {
    const auto num_pixels = output.size();
    thrust::transform(
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(num_pixels), output.begin(),
        [=] __device__(const auto idx) {
          const auto row = idx / width;
          const auto col = idx % width;
          const auto padded_idx = row * padded_width + col;

          Vec3 sum{};
          constexpr float IMAGE_SCALE = 1.0F / static_cast<float>(NUM_IMAGES);
          for (int img = 0; img < NUM_IMAGES; img++) {
            sum += images[img * (padded_width * height) + padded_idx];
          }
          return Color(sum * IMAGE_SCALE).correctGamma().to8Bit();
        });
  } else {
    const dim3 grid(std::ceil(width / BLOCK_SIZE.x),
                    std::ceil(height / BLOCK_SIZE.y));

    cuda::std::span<uchar4> output_span{thrust::raw_pointer_cast(output.data()),
                                        output.size()};

    averagePixels<NUM_IMAGES><<<grid, BLOCK_SIZE>>>(width, height, padded_width,
                                                    images, output_span);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUDA_ERROR_CHECK(cudaGetLastError());
  }
}

// Camera Builder

template <typename Params>
__host__ CameraBuilder<Params>::CameraBuilder() = default;

template <typename Params>
__host__ auto
CameraBuilder<Params>::origin(const Vec3 &origin) -> CameraBuilder<Params> & {
  this->camera.origin = origin;
  return *this;
}

template <typename Params>
__host__ auto
CameraBuilder<Params>::lookAt(const Vec3 &lookAt) -> CameraBuilder<Params> & {
  this->camera.lookAt = lookAt;
  return *this;
}

template <typename Params>
__host__ auto
CameraBuilder<Params>::up(const Vec3 &up) -> CameraBuilder<Params> & {
  this->camera.up = up;
  return *this;
}

template <typename Params>
__host__ auto CameraBuilder<Params>::verticalFov(const float verticalFov)
    -> CameraBuilder<Params> & {
  this->camera.verticalFov = verticalFov;
  return *this;
}

template <typename Params>
__host__ auto CameraBuilder<Params>::defocusAngle(const float defocusAngle)
    -> CameraBuilder<Params> & {
  this->camera.defocusAngle = defocusAngle;
  return *this;
}

template <typename Params>
__host__ auto CameraBuilder<Params>::focusDistance(const float focusDistance)
    -> CameraBuilder<Params> & {
  this->camera.focusDistance = focusDistance;
  return *this;
}

template <typename Params>
__host__ auto CameraBuilder<Params>::background(const Color &background)
    -> CameraBuilder<Params> & {
  this->camera.background = background;
  return *this;
}

template <typename Params>
__host__ auto CameraBuilder<Params>::build() -> Camera<Params> {
  return this->camera;
}
