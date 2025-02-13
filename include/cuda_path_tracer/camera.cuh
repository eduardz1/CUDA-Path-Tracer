#pragma once

#include "cuda_path_tracer/scene.cuh"
#include "cuda_path_tracer/vec3.cuh"
#include <driver_types.h>
#include <memory>
#include <cuda/std/span>
#include <thrust/host_vector.h>
#include <thrust/universal_vector.h>

/**
 * @brief Default hyper parameters for the camera.
 */
struct CameraHyperParams {
  static constexpr dim3 default_block_size = dim3(8, 8);
  static constexpr uint16_t default_num_samples = 128;
  static constexpr uint16_t default_num_images = 32;
  static constexpr bool default_average_with_thrust = true;
  using default_state = curandState_t;
};

/**
 * @brief The camera class is responsible for rendering the scene.
 *
 * @tparam BlockSize the dim3 block size for the kernel.
 * @tparam NumSamples the number of samples for each pixel.
 * @tparam NumImages  number of images to render and then average together,
 * provides a similar effect to the NumSamples parameter but it's more
 * parallelizable; ideally try to balance the two hyper parameters. They are
 * pretty much the same thing, so 16x32 is the same as 32x16.
 * @tparam AverageWithThrust if true, the images will be averaged using thrust,
 * otherwise they will be averaged using a custom kernel.
 * @tparam State the curandState_t type to use for the random number generator
 * in the kernel.
 */
template <dim3 BlockSize = CameraHyperParams::default_block_size,
          uint16_t NumSamples = CameraHyperParams::default_num_samples,
          uint16_t NumImages = CameraHyperParams::default_num_images,
          bool AverageWithThrust =
              CameraHyperParams::default_average_with_thrust,
          typename State = CameraHyperParams::default_state>
class Camera {
public:
  template <dim3 B, uint16_t NS, uint16_t NI, bool AWT, typename S>
  friend class CameraBuilder;

  __host__ Camera();
  __host__ void render(const std::shared_ptr<Scene> &scene,
                       thrust::universal_host_pinned_vector<uchar4> &image);
  __host__ static void
  averageRenderedImages(thrust::universal_host_pinned_vector<uchar4> &output,
                        const cuda::std::span<Vec3> &images,
                        const uint16_t width, const uint16_t height,
                        const uint16_t padded_width);

private:
  __host__ void init(const std::shared_ptr<Scene> &scene);

  /**
   * @brief viewportWidth and viewportHeight are the dimensions of the camera's
   * view. The viewport is a rectangle that represents the camera's view (our
   * world coordinates).
   */
  float viewportWidth, viewportHeight = 2.0F;

  /**
   * @brief They are, respectively, the horizontal and vertical vectors of the
   * distance between each pixel center in world coordinates.
   */
  Vec3 deltaU, deltaV;

  /**
   * @brief Location of the center of the pixel at the top left corner of the
   * image.
   */
  Vec3 pixel00;

  /**
   * @brief The X Y Z coordinates of the camera's origin.
   */
  Vec3 origin;

  /**
   * @brief The point the camera is looking at.
   */
  Vec3 lookAt;

  /**
   * @brief The camera's up vector.
   */
  Vec3 up;

  Vec3 defocusDiskU, defocusDiskV;

  /**
   * @brief The defocus angle is the angle of the cone of the defocus disk from
   * the camera center to the focus plane. It's similar to the aperture of a
   * camera, but, contrary to aperture, an higher value offers more bokeh
   * effect. The focus distance indicates the distance to the focus plane.
   */
  float defocusAngle, focusDistance;
  float verticalFov;
};

template <dim3 BlockSize = CameraHyperParams::default_block_size,
          uint16_t NumSamples = CameraHyperParams::default_num_samples,
          uint16_t NumImages = CameraHyperParams::default_num_images,
          bool AverageWithThrust =
              CameraHyperParams::default_average_with_thrust,
          typename State = CameraHyperParams::default_state>
class CameraBuilder {
public:
  __host__ CameraBuilder();
  __host__ auto origin(const Vec3 &origin) -> CameraBuilder &;
  __host__ auto lookAt(const Vec3 &lookAt) -> CameraBuilder &;
  __host__ auto up(const Vec3 &up) -> CameraBuilder &;
  __host__ auto verticalFov(float verticalFov) -> CameraBuilder &;
  __host__ auto focusDistance(float focusDistance) -> CameraBuilder &;
  __host__ auto defocusAngle(float defocusAngle) -> CameraBuilder &;
  __host__ auto
  build() -> Camera<BlockSize, NumSamples, NumImages, AverageWithThrust, State>;

private:
  Camera<BlockSize, NumSamples, NumImages, AverageWithThrust, State> camera;
};

__device__ auto defocusDiskSample(curandStatePhilox4_32_10_t &state,
                                  const Vec3 &center, const Vec3 &u,
                                  const Vec3 &v) -> Vec3;
__device__ auto defocusDiskSample(curandState_t &state, const Vec3 &center,
                                  const Vec3 &u, const Vec3 &v) -> Vec3;
__device__ auto
defocusDisk4Samples(curandStatePhilox4_32_10_t &state, const Vec3 &center,
                    const Vec3 &u,
                    const Vec3 &v) -> cuda::std::tuple<Vec3, Vec3, Vec3, Vec3>;

#include "camera.inl"