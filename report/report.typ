#import "template.typ": template, eqcolumns

#show: template.with(
  title: [CUDA Path Tracer],
  subtitle: [Report for the course of GP-GPU Programming],
  authors: ("Eduard Occhipinti", "Dominika Bochenczyk"),
)

// Proofread your reports. Up to 10% of your report grade can be deducted for poor grammar and related errors. Label and properly introduce all figures that appear in your report

#align(
  center,
  box(width: 75%)[
    #heading(level: 1, numbering: none)[Abstract]

    // Start with a statement regarding the purpose or objective of the assignment. Summarize your solution to the problem your procedure/design methodology. Summarize your results. Briefly draw some conclusions based on your results. In general, anyone should be able to read your abstract and get a general idea of what appears in the rest of the report. It might be a good idea to write this section last

    In this project we explored an implementation of a path tracer heavily inspired by the series of books by #cite(<Shirley2024RTW1>, form: "prose") @Shirley2024RTW2 @Shirley2024RTW3 and by the books published by NVIDIA on the topic @Haines2019 @Marrs2021. Inspiration was also drawn from the presentation of #cite(<Nikolaus>, form: "prose") and the article published by #cite(<BibEntry2020Jun>, form: "prose"). Iterating on these ideas we built our path tracer in a way that makes it easy to extend and easy to use. Most importantly our focus was on implementing the various algorithms that make it work on the GPU side and particular care was taken in the CUDA integration.
  ],
)

#v(4em)

#eqcolumns(2)[
  = Design Methodology

  - Dynamic parallelism for large amount of shapes
  - CUDA Occupacy API for optimal block size

  // Give a brief overview of the overall algorithm (what is the algorithm you are parallelizing?). Identify where parallelism can be introduced to the algorithm. Discuss the code executed by the master and the slaves. Discuss how the master and slaves communicate and why they communicate. Provide figures and equations to support your explanations. There should be no results in this section. After reading this section, we should have a complete understanding of how you solved the problem without having to read your code for further details. At the same time, there should be little to no code in your report.

  #lorem(300)

  = Results

  // Include all necessary tables (and make sure they are completely filled out). Include all relevant figures. Introduce all tables and figures in text BEFORE they appear in the report. When answering questions, always provide explanations and reasoning for your answers. If you don’t know what a question or requirement is asking for, please ask us in advance! We are here to help you learn.

  = Conclusion

  // Restate the purpose or objective of the assignment. Was the exercise successful in fulfilling its intended purpose? Why was it (or wasn’t it)? Summarize your results. Draw general conclusions based on your results (what did you learn?)
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