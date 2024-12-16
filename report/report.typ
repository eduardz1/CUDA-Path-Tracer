#import "template.typ": template, eqcolumns

#show: template.with(
  title: [CUDA Path Tracer],
  subtitle: [Report for the course of GP-GPU Programming],
  authors: ("Eduard Occhipinti", "Dominika Bochenczyk"),
)


#align(
  center,
  box(width: 75%)[
    #heading(level: 1, numbering: none)[Abstract]

    In this project we explored an implementation of a path tracer heavily inspired by the series of books by #cite(<Shirley2024RTW1>, form: "prose") @Shirley2024RTW2 @Shirley2024RTW3 and by the books published by NVIDIA on the topic @Haines2019 @Marrs2021. Inspiration was also drawn from the presentation of #cite(<Nikolaus>, form: "prose") and the article published by #cite(<BibEntry2020Jun>, form: "prose"). Iterating on these ideas we built our path tracer in a way that makes it easy to extend and easy to use. Most importantly our focus was on implementing the various algorithms that make it work on the GPU side and particular care was taken in the CUDA integration.
  ],
)

#v(4em)

#eqcolumns(2)[
  = Design Methodology

  #lorem(300)

  = Results

  = Conclusion
]

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