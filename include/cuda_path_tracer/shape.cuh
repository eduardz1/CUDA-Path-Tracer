/**
 * @file shape.cuh
 * @author Eduard Occhipinti (occhipinti.eduard@icloud.com)
 * @brief Abstract class for a shape in the scene
 * @version 0.1
 * @date 2024-10-30
 *
 * @copyright Copyright (c) 2024
 *
 */

#pragma once

#include <variant>
#include "sphere.cuh"

using Shape = std::variant<Sphere>;
