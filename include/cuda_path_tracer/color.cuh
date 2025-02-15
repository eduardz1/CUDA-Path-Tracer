#pragma once

#include "cuda_path_tracer/vec3.cuh"
#include <algorithm>
#include <cstdint>

class Color {
public:
  __host__ __device__ constexpr Color() : vec(0) {}

  /**
   * @brief Initialize a new Color object from a Vec3, pay attention that the
   * resulting color will not necessarily be a valid normalized color
   *
   * @param vec Vec3 to initialize the color with
   * @return __host__ constexpr Color
   */
  __host__ __device__ constexpr Color(Vec3 vec) : vec(vec) {}

  /**
   * @brief Create a new Color object from RGB values in the [0-255] range
   */
  __host__ __device__ constexpr static auto RGB(uint8_t r, uint8_t g, uint8_t b)
      -> Color {
    return Vec3{static_cast<float>(r) / UINT8_MAX,
                static_cast<float>(g) / UINT8_MAX,
                static_cast<float>(b) / UINT8_MAX};
  }

  /**
   * @brief Create a new Color object from RGB values in the [0-1] range by
   * clamping the values
   */
  __host__ __device__ constexpr static auto Normalized(float r, float g,
                                                       float b) -> Color {
    return Vec3{std::clamp(r, 0.0F, 1.0F), std::clamp(g, 0.0F, 1.0F),
                std::clamp(b, 0.0F, 1.0F)};
  }

  /**
   * @brief Create a new Color object from a Vec3 by clamping the values
   */
  __host__ __device__ constexpr static auto Normalized(Vec3 vec) -> Color {
    return Vec3{std::clamp(vec.x, 0.0F, 1.0F), std::clamp(vec.y, 0.0F, 1.0F),
                std::clamp(vec.z, 0.0F, 1.0F)};
  }

  /**
   * @brief Convert the color to an 8 bit color (integer values in the [0-255]
   * range)
   *
   * @return uchar4 8 bit color
   */
  __host__ __device__ auto to8Bit() const -> uchar4 {
    return make_uchar4(static_cast<unsigned char>(vec.x * UINT8_MAX),
                       static_cast<unsigned char>(vec.y * UINT8_MAX),
                       static_cast<unsigned char>(vec.z * UINT8_MAX),
                       UINT8_MAX);
  }

  /**
   * @brief Conert the color to a gamma corrected color, this is useful because
   * most image viewers expect gamma corrected colors
   *
   * @return Color gamma corrected color
   */
  __host__ __device__ constexpr auto correctGamma() const -> Color {
    return Color::Normalized(vec.x > 0 ? sqrtf(vec.x) : 0.0F,
                             vec.y > 0 ? sqrtf(vec.y) : 0.0F,
                             vec.z > 0 ? sqrtf(vec.z) : 0.0F);
  }

  __host__ __device__ constexpr operator Vec3() const { return vec; }

private:
  Vec3 vec;
};

// Some predefined colors
namespace Colors {
constexpr auto White = Color::RGB(255, 255, 255);
constexpr auto Black = Color::RGB(0, 0, 0);
constexpr auto Red = Color::RGB(255, 0, 0);
constexpr auto Green = Color::RGB(0, 255, 0);
constexpr auto Blue = Color::RGB(0, 0, 255);
constexpr auto Yellow = Color::RGB(255, 255, 0);
constexpr auto Cyan = Color::RGB(0, 255, 255);
constexpr auto Magenta = Color::RGB(255, 0, 255);
} // namespace Colors

// Captuccin Latte colors, taken from https://catppuccin.com/palette
namespace Catpuccin::Latte {
constexpr auto Rosewater = Color::RGB(220, 138, 120);
constexpr auto Flamingo = Color::RGB(221, 120, 120);
constexpr auto Pink = Color::RGB(234, 118, 203);
constexpr auto Mauve = Color::RGB(136, 57, 239);
constexpr auto Red = Color::RGB(210, 15, 57);
constexpr auto Maroon = Color::RGB(230, 69, 83);
constexpr auto Peach = Color::RGB(254, 100, 11);
constexpr auto Yellow = Color::RGB(223, 142, 29);
constexpr auto Green = Color::RGB(64, 160, 43);
constexpr auto Teal = Color::RGB(23, 146, 153);
constexpr auto Sky = Color::RGB(4, 165, 229);
constexpr auto Sapphire = Color::RGB(32, 159, 181);
constexpr auto Blue = Color::RGB(30, 102, 245);
constexpr auto Lavander = Color::RGB(114, 135, 253);
constexpr auto Text = Color::RGB(76, 79, 105);
constexpr auto Base = Color::RGB(239, 241, 245);
constexpr auto Mantle = Color::RGB(230, 233, 239);
constexpr auto Crust = Color::RGB(220, 224, 232);
} // namespace Catpuccin::Latte
