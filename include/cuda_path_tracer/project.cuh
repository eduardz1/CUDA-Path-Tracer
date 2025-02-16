#pragma once

#include "cuda_path_tracer/camera.cuh"
#include "cuda_path_tracer/scene.cuh"
#include <memory>
#include <nlohmann/json-schema.hpp>
#include <string>

const static nlohmann::json schema = R"(
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["name", "image", "camera", "shapes"],
    "properties": {
      "name": {
        "type": "string",
        "description": "Name of the scene, will be used as the output file name"
      },
      "image": {
        "type": "object",
        "required": ["width", "height"],
        "properties": {
          "width": {
            "type": "integer",
            "minimum": 1,
            "description": "Image width in pixels"
          },
          "height": {
            "type": "integer",
            "minimum": 1,
            "description": "Image height in pixels"
          }
        }
      },
      "camera": {
        "type": "object",
        "properties": {
          "origin": {
            "type": "array",
            "items": { "type": "number" },
            "minItems": 3,
            "maxItems": 3
          },
          "lookAt": {
            "type": "array",
            "items": { "type": "number" },
            "minItems": 3,
            "maxItems": 3
          },
          "up": {
            "type": "array",
            "items": { "type": "number" },
            "minItems": 3,
            "maxItems": 3
          },
          "verticalFov": { "type": "number", "minimum": 0, "maximum": 180 },
          "defocusAngle": { "type": "number", "minimum": 0 },
          "focusDistance": { "type": "number", "minimum": 0 },
          "background": { 
            "oneOf": [
              { "type": "string" },
              { 
                "type": "array",
                "items": { "type": "number" },
                "minItems": 3,
                "maxItems": 3
              }
            ]
          }
        }
      },
      "shapes": {
        "type": "array",
        "items": {
          "type": "object",
          "required": ["type", "material"],
          "properties": {
            "type": {
              "type": "string",
              "enum": ["sphere", "rectangular_cuboid", "parallelogram"]
            },
            "material": {
              "type": "object",
              "required": ["type"],
              "properties": {
                "type": {
                  "type": "string",
                  "enum": ["lambertian", "dielectric", "metal", "light"]
                }
              },
              "allOf": [
                {
                  "if": {
                    "properties": { "type": { "const": "lambertian" } }
                  },
                  "then": {
                    "oneOf": [
                      {
                        "required": ["color"],
                        "properties": {
                          "color": { 
                            "oneOf": [
                              { "type": "string" },
                              {
                                "type": "array",
                                "items": { "type": "number" },
                                "minItems": 3,
                                "maxItems": 3
                              }
                            ]
                          }
                        }
                      },
                      {
                        "required": ["texture"],
                        "properties": {
                          "texture": {
                            "oneOf": [
                              { "type": "string" },
                              {
                                "type": "object",
                                "required": ["checker"],
                                "properties": {
                                  "checker": {
                                    "type": "object",
                                    "required": ["scale", "even", "odd"],
                                    "properties": {
                                      "scale": { "type": "number" },
                                      "even": { "type": "string" },
                                      "odd": { "type": "string" }
                                    }
                                  }
                                }
                              }
                            ]
                          }
                        }
                      }
                    ]
                  }
                },
                {
                  "if": {
                    "properties": { "type": { "const": "dielectric" } }
                  },
                  "then": {
                    "required": ["refraction_index"],
                    "properties": {
                      "refraction_index": { "type": "number" }
                    }
                  }
                },
                {
                  "if": {
                    "properties": { "type": { "const": "metal" } }
                  },
                  "then": {
                    "required": ["color", "fuzz"],
                    "properties": {
                      "color": { 
                        "oneOf": [
                          { "type": "string" },
                          {
                            "type": "array",
                            "items": { "type": "number" },
                            "minItems": 3,
                            "maxItems": 3
                          }
                        ]
                      },
                      "fuzz": { "type": "number" }
                    }
                  }
                },
                {
                  "if": {
                    "properties": { "type": { "const": "light" } }
                  },
                  "then": {
                    "required": ["color"],
                    "properties": {
                      "color": { 
                        "oneOf": [
                          { "type": "string" },
                          {
                            "type": "array",
                            "items": { "type": "number" },
                            "minItems": 3,
                            "maxItems": 3
                          }
                        ]
                      }
                    }
                  }
                }
              ]
            },
            "center": {
              "type": "array",
              "items": { "type": "number" },
              "minItems": 3,
              "maxItems": 3
            },
            "radius": {
              "type": "number",
              "minimum": 0
            },
            "vertices": {
              "type": "array",
              "items": {
                "type": "array",
                "items": { "type": "number" },
                "minItems": 3,
                "maxItems": 3
              },
              "minItems": 2,
              "maxItems": 2
            },
            "rotation": {
              "type": "array",
              "items": { "type": "number" },
              "minItems": 3,
              "maxItems": 3
            },
            "translation": {
              "type": "array",
              "items": { "type": "number" },
              "minItems": 3,
              "maxItems": 3
            },
            "origin": {
              "type": "array",
              "items": { "type": "number" },
              "minItems": 3,
              "maxItems": 3
            },
            "u": {
              "type": "array",
              "items": { "type": "number" },
              "minItems": 3,
              "maxItems": 3
            },
            "v": {
              "type": "array",
              "items": { "type": "number" },
              "minItems": 3,
              "maxItems": 3
            }
          },
          "allOf": [
            {
              "if": {
                "properties": { "type": { "const": "sphere" } }
              },
              "then": {
                "required": ["center", "radius"]
              }
            },
            {
              "if": {
                "properties": { "type": { "const": "rectangular_cuboid" } }
              },
              "then": {
                "required": ["vertices"]
              }
            },
            {
              "if": {
                "properties": { "type": { "const": "parallelogram" } }
              },
              "then": {
                "required": ["origin", "u", "v"]
              }
            }
          ]
        }
      }
    }
  }

)"_json;

class Project {
public:
  __host__ static auto
  load(const std::string &filename,
       const std::string &quality) -> std::shared_ptr<Project>;
  __host__ auto render() -> void;

private:
  std::shared_ptr<Scene> scene;
  std::shared_ptr<CameraInterface> camera;
  std::string name;
};