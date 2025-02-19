#pragma once

#include "cuda_path_tracer/scene.cuh"
#include "cuda_path_tracer/vec3.cuh"
#include <cuda/std/span>
#include <driver_types.h>
#include <memory>
#include <thrust/host_vector.h>
#include <thrust/universal_vector.h>

/**
 * @brief Hyper parameters for the camera.
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
template <dim3 BlockSize, uint16_t NumSamples, uint16_t NumImages,
          uint16_t Depth, bool AvgWithThrust, typename State>
struct CameraParams {
  static constexpr dim3 block_size = BlockSize;
  static constexpr uint16_t num_samples = NumSamples;
  static constexpr uint16_t num_images = NumImages;
  static constexpr uint16_t depth = Depth;
  static constexpr bool avg_with_thrust = AvgWithThrust;
  using default_state = curandState_t;
};

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers, readability-magic-numbers)
static constexpr uint16_t HIGH_QUALITY_DEPTH = 64;
using HighQuality = CameraParams<dim3(8, 8), 2048, 8, HIGH_QUALITY_DEPTH, true,
                                 curandStatePhilox4_32_10_t>;
static constexpr uint16_t MEDIUM_QUALITY_DEPTH = 16;
using MediumQuality = CameraParams<dim3(8, 8), 256, 8, MEDIUM_QUALITY_DEPTH,
                                   false, curandState_t>;
static constexpr uint16_t LOW_QUALITY_DEPTH = 4;
using LowQuality =
    CameraParams<dim3(8, 8), 64, 4, LOW_QUALITY_DEPTH, true, curandState_t>;
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers, readability-magic-numbers)

/**
 * @brief Camera interface needed to initialize the camera from the JSON parser
 */
class CameraInterface {
public:
  CameraInterface() = default;
  CameraInterface(const CameraInterface &) = default;
  CameraInterface(CameraInterface &&) = delete;
  auto operator=(const CameraInterface &) -> CameraInterface & = default;
  auto operator=(CameraInterface &&) -> CameraInterface & = delete;
  virtual ~CameraInterface() = default;
  /**
   * @brief Render the scene.
   *
   * @param scene scene to render
   * @param image image vector to store the rendered image in
   */
  virtual void render(const std::shared_ptr<Scene> &scene,
                      thrust::universal_host_pinned_vector<uchar4> &image) = 0;
};

/**
 * @brief The camera class is responsible for rendering the scene.
 *
 * @tparam Params the hyper parameters for the camera.
 */
template <typename Params = HighQuality> class Camera : public CameraInterface {
  static constexpr dim3 BLOCK_SIZE = Params::block_size;
  static constexpr uint16_t NUM_SAMPLES = Params::num_samples;
  static constexpr uint16_t NUM_IMAGES = Params::num_images;
  static constexpr uint16_t DEPTH = Params::depth;
  static constexpr bool AVG_WITH_THRUST = Params::avg_with_thrust;
  using State = typename Params::default_state;

public:
  template <typename P> friend class CameraBuilder;

  __host__ Camera();

  __host__ void
  render(const std::shared_ptr<Scene> &scene,
         thrust::universal_host_pinned_vector<uchar4> &image) override;

  /**
   * @brief Average the rendered images together.
   *
   * @param output output image
   * @param images images to average
   * @param width image width
   * @param height image height
   * @param padded_width padded image width
   */
  __host__ static void
  averageRenderedImages(thrust::universal_host_pinned_vector<uchar4> &output,
                        const cuda::std::span<Vec3> &images,
                        const uint16_t width, const uint16_t height,
                        const uint16_t padded_width);

private:
  /**
   * @brief Initialize the camera, by calculating and setting the correct
   * parameters with the given scene.
   *
   * @param scene scene to initialize the camera with
   */
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

  /**
   * @brief The defocus disk is a disk that is used to simulate the bokeh
   * effect. It's a disk that is perpendicular to the camera's view direction
   * and is centered at the focus plane. The defocus disk is used to sample the
   * focus plane.
   */
  Vec3 defocusDiskU, defocusDiskV;

  /**
   * @brief The defocus angle is the angle of the cone of the defocus disk from
   * the camera center to the focus plane. It's similar to the aperture of a
   * camera, but, contrary to aperture, an higher value offers more bokeh
   * effect. The focus distance indicates the distance to the focus plane.
   */
  float defocusAngle, focusDistance;

  /**
   * @brief The vertical field of view of the camera.
   */
  float verticalFov;

  /**
   * @brief Background color of the scene.
   */
  Color background;
};

/**
 * @brief Camera builder to create a camera with a fluent interface.
 *
 * @tparam Params the hyper parameters for the camera.
 */
template <typename Params = HighQuality> class CameraBuilder {
public:
  __host__ CameraBuilder();
  __host__ auto origin(const Vec3 &origin) -> CameraBuilder &;
  __host__ auto lookAt(const Vec3 &lookAt) -> CameraBuilder &;
  __host__ auto up(const Vec3 &up) -> CameraBuilder &;
  __host__ auto verticalFov(float verticalFov) -> CameraBuilder &;
  __host__ auto focusDistance(float focusDistance) -> CameraBuilder &;
  __host__ auto defocusAngle(float defocusAngle) -> CameraBuilder &;
  __host__ auto background(const Color &background) -> CameraBuilder &;
  __host__ auto build() -> Camera<Params>;

private:
  Camera<Params> camera;
};

/**
 * @brief Sample the defocus disk.
 *
 * @param state curand state
 * @param center center of the disk
 * @param u first vector of the disk
 * @param v second vector of the disk
 * @return Vec3 sampled point on the disk
 */
__device__ auto defocusDiskSample(curandStatePhilox4_32_10_t &state,
                                  const Vec3 &center, const Vec3 &u,
                                  const Vec3 &v) -> Vec3;

/**
 * @brief Sample the defocus disk.
 *
 * @param state curand state
 * @param center center of the disk
 * @param u first vector of the disk
 * @param v second vector of the disk
 * @return Vec3 sampled point on the disk
 */
__device__ auto defocusDiskSample(curandState_t &state, const Vec3 &center,
                                  const Vec3 &u, const Vec3 &v) -> Vec3;

/**
 * @brief Sample the defocus disk with 4 samples.
 *
 * @param state curand state
 * @param center center of the disk
 * @param u first vector of the disk
 * @param v second vector of the disk
 * @return cuda::std::tuple<Vec3, Vec3, Vec3, Vec3> sampled points on the disk
 */
__device__ auto
defocusDisk4Samples(curandStatePhilox4_32_10_t &state, const Vec3 &center,
                    const Vec3 &u,
                    const Vec3 &v) -> cuda::std::tuple<Vec3, Vec3, Vec3, Vec3>;

#include "camera.inl"