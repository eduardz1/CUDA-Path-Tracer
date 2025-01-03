#include <climits>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#include "cuda_path_tracer/camera.cuh"
#include "cuda_path_tracer/error.cuh"
#include "cuda_path_tracer/hit_info.cuh"
#include "cuda_path_tracer/image.cuh"
#include "cuda_path_tracer/lambertian.cuh"
#include "cuda_path_tracer/ray.cuh"
#include "cuda_path_tracer/shape.cuh"
#include "cuda_path_tracer/vec3.cuh"

namespace {
constexpr unsigned long long SEED = 0xba0bab;
constexpr dim3 BLOCK_SIZE(16, 16);
constexpr int DEPTH = 10;

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
  return center + p.getX() * u + p.getY() * v;
}

__device__ auto getRay(const Vec3 &origin, const Vec3 &pixel00,
                       const Vec3 &deltaU, const Vec3 &deltaV,
                       const Vec3 &defocusDiskU, const Vec3 &defocusDiskV,
                       const float defocusAngle, const uint16_t x,
                       const uint16_t y, curandState &state) -> Ray {
  // We sample an area of "half pixel" around the pixel centers
  auto offset =
      Vec3{curand_uniform(&state) - 0.5f, curand_uniform(&state) - 0.5f, 0};

  auto sample = pixel00 + ((float(x) + offset.getX()) * deltaU) +
                ((float(y) + offset.getY()) * deltaV);

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
__device__ auto hitShapes(const Ray &ray, const Shape *shapes,
                          const size_t num_shapes, HitInfo &hi) -> bool {
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
                         const size_t num_shapes, curandState &state,
                         int depth) -> Vec3 {

  Vec3 color = Vec3{1.0f, 1.0f, 1.0f};
  Ray current = ray;

  for (int i = 0; i < depth; i++) {
    auto hi = HitInfo();
    bool hit = hitShapes(current, shapes, num_shapes, hi);

    // could possibly remove the shadow acne problem but this is a little change
    if (hit) {
      Ray scattered;
      Vec3 attenuation;

      Vec3 normal = hi.getNormal();
      Vec3 point = hi.getPoint();
      Material material = hi.getMaterial();

      bool scatter = cuda::std::visit(
          [&current, &normal, &point, &attenuation, &scattered,
           &state](auto &material) {
            return material.scatter(current, normal, point, attenuation,
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
      auto t = 0.5f * (unit_direction.getY() + 1.0f);
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
 * @param shapes Array of shapes to check for hits
 * @param num_shapes Number of shapes in the array
 * @param states Random number generator states for each pixel
 */
__global__ void renderImage(const uint16_t width, const uint16_t height,
                            uchar4 *image, const Vec3 origin,
                            const Vec3 pixel00, const Vec3 deltaU,
                            const Vec3 deltaV, const Vec3 defocusDiskU,
                            const Vec3 defocusDiskV, const float defocusAngle,
                            const Shape *shapes, const size_t num_shapes,
                            curandState *states) {
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const auto index = y * width + x;
  curand_init(SEED, index, 0, &states[index]);

  curandState state = states[index];
  auto color = Vec3{};
  for (auto s = 0; s < NUM_SAMPLES; s++) {
    const auto ray = getRay(origin, pixel00, deltaU, deltaV, defocusDiskU,
                            defocusDiskV, defocusAngle, x, y, state);
    color += getColor(ray, shapes, num_shapes, state, DEPTH);
  }

  image[index] = convertColorTo8Bit(color / float(NUM_SAMPLES));
}
} // namespace

__host__ Camera::Camera() : origin() {}
__host__ Camera::Camera(const Vec3 &origin) : origin(origin) {}

__host__ void Camera::init(const std::shared_ptr<Scene> &scene) {
  const auto width = scene->getWidth();
  const auto height = scene->getHeight();

  const auto theta = this->verticalFov * M_PIf32 / 180;
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
      this->focusDistance * std::tan(this->defocusAngle * M_PIf32 / 360);
  this->defocusDiskU = u * defocusRadius;
  this->defocusDiskV = v * defocusRadius;
}

__host__ void Camera::render(const std::shared_ptr<Scene> &scene,
                             uchar4 *image) {
  this->init(scene);

  const auto width = scene->getWidth();
  const auto height = scene->getHeight();
  curandState *states;
  CUDA_ERROR_CHECK(
      cudaMalloc((void **)&states, width * height * sizeof(curandState)));

  std::vector<Shape> &h_shapes = scene->getShapes();
  // Dummy shape introduced because the last shape always fails to hit, cannot
  // figure out why so this is a easy workaround
  h_shapes.emplace_back(Sphere{});
  const size_t num_shapes = h_shapes.size();

  Shape *d_shapes;
  CUDA_ERROR_CHECK(cudaMalloc((void **)&d_shapes, num_shapes * sizeof(Shape)));
  CUDA_ERROR_CHECK(cudaMemcpy(d_shapes, h_shapes.data(),
                              num_shapes * sizeof(Sphere),
                              cudaMemcpyHostToDevice));

  uchar4 *image_device;

  const auto size = static_cast<long>(width) * height * sizeof(uchar4);

  CUDA_ERROR_CHECK(cudaMalloc((void **)&image_device, size));

  dim3 grid((width + BLOCK_SIZE.x - 1) / BLOCK_SIZE.x,
            (height + BLOCK_SIZE.y - 1) / BLOCK_SIZE.y);

  renderImage<<<grid, BLOCK_SIZE>>>(
      width, height, image_device, origin, pixel00, deltaU, deltaV,
      defocusDiskU, defocusDiskV, defocusAngle, d_shapes, num_shapes, states);
  cudaDeviceSynchronize();
  CUDA_ERROR_CHECK(cudaGetLastError());

  CUDA_ERROR_CHECK(
      cudaMemcpy(image, image_device, size, cudaMemcpyDeviceToHost));

  cudaFree(d_shapes);
  CUDA_ERROR_CHECK(cudaGetLastError());
}
