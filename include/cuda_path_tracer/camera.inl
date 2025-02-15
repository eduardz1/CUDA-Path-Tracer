#pragma once

#include "cuda_path_tracer/camera.cuh"
#include "cuda_path_tracer/error.cuh"
#include "cuda_path_tracer/image.cuh"
#include "cuda_path_tracer/utilities.cuh"

namespace {
constexpr uint64_t SEED = 0xba0bab;
constexpr uint64_t DEPTH = 20;

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

// TODO(eduard): remove depth and make it an hyperparameter
template <typename State>
__device__ auto getColor(const Ray &ray,
                         const cuda::std::span<const Shape> shapes,
                         State &state, int depth, Vec3 background) -> Vec3 {
  // TODO(eduard): make use of shared memory

  Vec3 color = Vec3{1.0F, 1.0F, 1.0F};
  Ray current = ray;

  for (int i = 0; i < depth; i++) {
    auto hi = HitInfo();
    bool hit = hitShapes(current, shapes, hi);

    if (i == 0 && !hit) {
      return background;
    }

    // could possibly remove the shadow acne problem but this is a little change
    if (hit) {
      Ray scattered;
      Vec3 attenuation;

      Vec3 normal = hi.normal;
      Vec3 point = hi.point;
      Material material = hi.material;
      bool front = hi.front;

      const auto emitted =
          cuda::std::visit(overload{[&point](const Light &light) {
                                      return light.emitted(point);
                                    },
                                    [](const auto) { return Vec3{0}; }},
                           material);

      const bool scatter = cuda::std::visit(
          overload{[&normal, &point, &attenuation, &scattered,
                    &state](const Lambertian &material) {
                     return material.scatter<State>(normal, point, attenuation,
                                                    scattered, state);
                   },
                   [&current, &normal, &point, &attenuation, &scattered,
                    &state](const Metal &material) {
                     return material.scatter<State>(
                         current, normal, point, attenuation, scattered, state);
                   },
                   [&current, &normal, &point, front, &attenuation,
                    &scattered](const Dielectric &material) {
                     return material.scatter(current, normal, point, front,
                                             attenuation, scattered);
                   },
                   [](const auto) { return false; }

          },
          material);

      if (scatter) {
        color = color * attenuation + emitted;
        current = scattered;
      } else {
        return emitted;
      }
    } else {
      auto unit_direction = makeUnitVector(current.getDirection());
      auto t = 0.5F * (unit_direction.y + 1.0F);
      return color * (1.0F - t) * Vec3{1.0F, 1.0F, 1.0F} +
             t * Vec3{0.8F, 0.85F, 1.0F};
    }
  }
  return Vec3{0};
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
template <typename State, uint16_t NumSamples>
__global__ void
renderImage(const uint16_t width, const uint16_t height,
            const cuda::std::span<Vec3> image, const Vec3 origin,
            const Vec3 pixel00, const Vec3 deltaU, const Vec3 deltaV,
            const Vec3 defocusDiskU, const Vec3 defocusDiskV,
            const float defocusAngle, const cuda::std::span<const Shape> shapes,
            const size_t stream_index, const Vec3 background) {
  State states; // NOLINT

  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const auto index = y * width + x;

  curand_init(SEED, index + (stream_index * width * height), 0, &states);

  auto color = Vec3{};

  // Evaluates at compile time the type of the state
  if constexpr (std::is_same_v<State, curandStatePhilox4_32_10_t>) {
    // When using the Philox4_32_10_t generator, we can generate 4 random
    // uniformly distributed numbers at once, meaning that we can generate 4
    // rays at once
    for (auto s = 0; s < (NumSamples >> 2); s++) {
      const auto ray = get4Rays(origin, pixel00, deltaU, deltaV, defocusDiskU,
                                defocusDiskV, defocusAngle, x, y, states);

      color +=
          getColor(cuda::std::get<0>(ray), shapes, states, DEPTH, background);
      color +=
          getColor(cuda::std::get<1>(ray), shapes, states, DEPTH, background);
      color +=
          getColor(cuda::std::get<2>(ray), shapes, states, DEPTH, background);
      color +=
          getColor(cuda::std::get<3>(ray), shapes, states, DEPTH, background);
    }
  } else {
    for (auto s = 0; s < NumSamples; s++) {
      const auto ray = getRay(origin, pixel00, deltaU, deltaV, defocusDiskU,
                              defocusDiskV, defocusAngle, x, y, states);

      color += getColor<State>(ray, shapes, states, DEPTH, background);
    }
  }

  image[index] = color;
}

template <uint16_t NumImages>
__global__ void averagePixels(const uint16_t width, const uint16_t height,
                              const uint16_t padded_width, const float scale,
                              const cuda::std::span<Vec3> images,
                              cuda::std::span<uchar4> image_out) {
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const auto output_idx = y * width + x;
  const auto padded_idx = y * padded_width + x;

  auto sum = Vec3{};
  for (auto img = 0; img < NumImages; img++) {
    sum += images[img * (padded_width * height) + padded_idx];
  }

  image_out[output_idx] = convertColorTo8Bit(sum * scale);
}

// Align to prevent control divergence
// TODO(eduard): Write about it in the report
__host__ inline auto getPaddedSize(size_t size,
                                   size_t alignment = WARP_SIZE) -> size_t {
  return (size + alignment - 1) & ~(alignment - 1);
}
} // namespace

template <dim3 BlockSize, uint16_t NumSamples, uint16_t NumImages,
          bool AverageWithThrust, typename State>
__host__
Camera<BlockSize, NumSamples, NumImages, AverageWithThrust, State>::Camera() =
    default;

template <dim3 BlockSize, uint16_t NumSamples, uint16_t NumImages,
          bool AverageWithThrust, typename State>
__host__ void
Camera<BlockSize, NumSamples, NumImages, AverageWithThrust, State>::init(
    const std::shared_ptr<Scene> &scene) {
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

template <dim3 BlockSize, uint16_t NumSamples, uint16_t NumImages,
          bool AverageWithThrust, typename State>
__host__ void
Camera<BlockSize, NumSamples, NumImages, AverageWithThrust, State>::render(
    const std::shared_ptr<Scene> &scene,
    thrust::universal_host_pinned_vector<uchar4> &image) {
  this->init(scene);

  const auto width = scene->getWidth();
  const auto height = scene->getHeight();
  const auto padded_width = getPaddedSize(width);
  const auto padded_height = getPaddedSize(height);
  const auto num_padded_pixels = padded_width * padded_height;

  constexpr float render_scale = 1.0F / (NumImages * NumSamples);

  assert(image.size() == static_cast<size_t>(width * height) &&
         ("Image size does not match the scene's width and height. Actual: " +
          std::to_string(image.size()) +
          ", Expected: " + std::to_string(width * height))
             .c_str());

  thrust::device_vector<Vec3> image_3d(num_padded_pixels * NumImages);

  cuda::std::span<const Shape> shapes_span{
      thrust::raw_pointer_cast(scene->getShapes().data()),
      scene->getShapes().size()};
  cuda::std::span<Vec3> image_3d_span{thrust::raw_pointer_cast(image_3d.data()),
                                      image_3d.size()};

  const dim3 grid(std::ceil(padded_width / BlockSize.x),
                  std::ceil(padded_height / BlockSize.y));

  std::array<StreamGuard, NumImages> streams{};

  // Calling cudaGetLastError() here to clear any previous errors
  CUDA_ERROR_CHECK(cudaGetLastError());
  for (auto i = 0; i < NumImages; i++) {
    renderImage<State, NumSamples><<<grid, BlockSize, 0, streams.at(i)>>>(
        padded_width, padded_height,
        image_3d_span.subspan(i * num_padded_pixels), origin, pixel00, deltaU,
        deltaV, defocusDiskU, defocusDiskV, defocusAngle, shapes_span, i,
        background);
  }
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_ERROR_CHECK(cudaGetLastError());

  averageRenderedImages(image, image_3d_span, width, height, padded_width);
}

template <dim3 BlockSize, uint16_t NumSamples, uint16_t NumImages,
          bool AverageWithThrust, typename State>
__host__ void
Camera<BlockSize, NumSamples, NumImages, AverageWithThrust, State>::
    averageRenderedImages(thrust::universal_host_pinned_vector<uchar4> &output,
                          const cuda::std::span<Vec3> &images,
                          const uint16_t width, const uint16_t height,
                          const uint16_t padded_width) {
  constexpr float render_scale = 1.0F / (NumImages * NumSamples);

  if constexpr (AverageWithThrust) {
    const auto num_pixels = output.size();
    thrust::transform(
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(num_pixels), output.begin(),
        [=] __device__(const auto idx) {
          const auto row = idx / width;
          const auto col = idx % width;
          const auto padded_idx = row * padded_width + col;

          Vec3 sum{};
          for (int img = 0; img < NumImages; img++) {
            sum += images[img * (padded_width * height) + padded_idx];
          }
          return convertColorTo8Bit(sum * render_scale);
        });
  } else {
    const dim3 grid(std::ceil(width / BlockSize.x),
                    std::ceil(height / BlockSize.y));

    cuda::std::span<uchar4> output_span{thrust::raw_pointer_cast(output.data()),
                                        output.size()};

    averagePixels<NumImages><<<grid, BlockSize>>>(
        width, height, padded_width, render_scale, images, output_span);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUDA_ERROR_CHECK(cudaGetLastError());
  }
}

// Camera Builder

template <dim3 BlockSize, uint16_t NumSamples, uint16_t NumImages,
          bool AverageWithThrust, typename State>
__host__ CameraBuilder<BlockSize, NumSamples, NumImages, AverageWithThrust,
                       State>::CameraBuilder() = default;

template <dim3 BlockSize, uint16_t NumSamples, uint16_t NumImages,
          bool AverageWithThrust, typename State>
__host__ auto CameraBuilder<BlockSize, NumSamples, NumImages, AverageWithThrust,
                            State>::origin(const Vec3 &origin)
    -> CameraBuilder<BlockSize, NumSamples, NumImages, AverageWithThrust, State>
        & {
  this->camera.origin = origin;
  return *this;
}
template <dim3 BlockSize, uint16_t NumSamples, uint16_t NumImages,
          bool AverageWithThrust, typename State>
__host__ auto CameraBuilder<BlockSize, NumSamples, NumImages, AverageWithThrust,
                            State>::lookAt(const Vec3 &lookAt)
    -> CameraBuilder<BlockSize, NumSamples, NumImages, AverageWithThrust, State>
        & {
  this->camera.lookAt = lookAt;
  return *this;
}
template <dim3 BlockSize, uint16_t NumSamples, uint16_t NumImages,
          bool AverageWithThrust, typename State>
__host__ auto
CameraBuilder<BlockSize, NumSamples, NumImages, AverageWithThrust, State>::up(
    const Vec3 &up) -> CameraBuilder<BlockSize, NumSamples, NumImages,
                                     AverageWithThrust, State> & {
  this->camera.up = up;
  return *this;
}
template <dim3 BlockSize, uint16_t NumSamples, uint16_t NumImages,
          bool AverageWithThrust, typename State>
__host__ auto CameraBuilder<BlockSize, NumSamples, NumImages, AverageWithThrust,
                            State>::verticalFov(const float verticalFov)
    -> CameraBuilder<BlockSize, NumSamples, NumImages, AverageWithThrust, State>
        & {
  this->camera.verticalFov = verticalFov;
  return *this;
}
template <dim3 BlockSize, uint16_t NumSamples, uint16_t NumImages,
          bool AverageWithThrust, typename State>
__host__ auto CameraBuilder<BlockSize, NumSamples, NumImages, AverageWithThrust,
                            State>::defocusAngle(const float defocusAngle)
    -> CameraBuilder<BlockSize, NumSamples, NumImages, AverageWithThrust, State>
        & {
  this->camera.defocusAngle = defocusAngle;
  return *this;
}
template <dim3 BlockSize, uint16_t NumSamples, uint16_t NumImages,
          bool AverageWithThrust, typename State>
__host__ auto CameraBuilder<BlockSize, NumSamples, NumImages, AverageWithThrust,
                            State>::focusDistance(const float focusDistance)
    -> CameraBuilder<BlockSize, NumSamples, NumImages, AverageWithThrust, State>
        & {
  this->camera.focusDistance = focusDistance;
  return *this;
}
template <dim3 BlockSize, uint16_t NumSamples, uint16_t NumImages,
          bool AverageWithThrust, typename State>
__host__ auto CameraBuilder<BlockSize, NumSamples, NumImages, AverageWithThrust,
                            State>::background(const Vec3 &background)
    -> CameraBuilder<BlockSize, NumSamples, NumImages, AverageWithThrust, State>
        & {
  this->camera.background = background;
  return *this;
}
template <dim3 BlockSize, uint16_t NumSamples, uint16_t NumImages,
          bool AverageWithThrust, typename State>
__host__ auto CameraBuilder<BlockSize, NumSamples, NumImages, AverageWithThrust,
                            State>::build()
    -> Camera<BlockSize, NumSamples, NumImages, AverageWithThrust, State> {
  return this->camera;
}
