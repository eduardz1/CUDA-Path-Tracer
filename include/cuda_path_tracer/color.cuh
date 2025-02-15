#pragma once

#include "cuda_path_tracer/vec3.cuh"
#include <cstdint>

class Color {
public:
  __host__ __device__ constexpr Color() : vec(0) {}

  /**
   * @brief Create a new Color object from RGB values in the [0-255] range
   */
  __host__ __device__ constexpr static auto RGB(uint8_t r, uint8_t g,
                                                uint8_t b) -> Color {
    return Vec3{static_cast<float>(r) / UINT8_MAX,
                static_cast<float>(g) / UINT8_MAX,
                static_cast<float>(b) / UINT8_MAX};
  }

  /**
   * @brief Create a new Color object from RGB values in the [0-1] range
   */
  __host__ __device__ constexpr static auto Normalized(float r, float g,
                                                       float b) -> Color {
    return Vec3{r, g, b};
  }
  __host__ __device__ constexpr static auto
  Normalized(const Vec3 color) -> Color {
    return {color};
  }

  __host__ __device__ constexpr auto r() const -> uint8_t {
    return static_cast<uint8_t>(vec.x * UINT8_MAX);
  }
  __host__ __device__ constexpr auto g() const -> uint8_t {
    return static_cast<uint8_t>(vec.y * UINT8_MAX);
  }
  __host__ __device__ constexpr auto b() const -> uint8_t {
    return static_cast<uint8_t>(vec.z * UINT8_MAX);
  }

  __host__ __device__ constexpr auto r_normalized() const -> float {
    return vec.x;
  }
  __host__ __device__ constexpr auto g_normalized() const -> float {
    return vec.y;
  }
  __host__ __device__ constexpr auto b_normalized() const -> float {
    return vec.z;
  }

  __host__ __device__ constexpr operator Vec3() const { return vec; }

private:
  __host__ __device__ constexpr Color(Vec3 vec) : vec(vec) {}
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
