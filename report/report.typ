#import "template.typ": template, balance

#show: template.with(
  title: [CUDA Path Tracer],
  authors: ("Eduard Occhipinti", "Dominika Bochenczyk"),
)

= Introduction

#balance(
  columns(2)[
    #lorem(500)
  ],
)

#place(bottom + center, scope: "parent", float: true)[
  ```cpp
  __device__ auto getColor(const Ray &ray, const Shape *shapes,
                           const size_t num_shapes) -> uchar4 {
    for (size_t i = 0; i < num_shapes; i++) {
      bool hit = cuda::std::visit(
          [&ray](const auto &shape) { return shape.hit(ray); }, shapes[i]);

      if (hit) {
        return make_uchar4(1, 0, 0, UCHAR_MAX);
      }
    }
    return make_uchar4(0, 0, 1, UCHAR_MAX);
  }
  ```
]