#import "template.typ": template, eqcolumns, pseudocode-list


#show: template.with(
  title: [CUDA Path Tracer],
  subtitle: [Report for the course of GP-GPU Programming],
  authors: ("Eduard Occhipinti", "Dominika Bocheńczyk"),
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
  = Introduction

  // Give a brief overview of the overall algorithm (what is the algorithm you are parallelizing?). Identify where parallelism can be introduced to the algorithm. Discuss the code executed by the master and the slaves. Discuss how the master and slaves communicate and why they communicate. Provide figures and equations to support your explanations. There should be no results in this section. After reading this section, we should have a complete understanding of how you solved the problem without having to read your code for further details. At the same time, there should be little to no code in your report.

  The code we are parallelizing is that of a path tracer. This algorithm has the advantage of being embarrassingly parallel due to its nature. Each pixel is sampled $N$ times and each sample is independent of the other.

  The algorithm can be visualized as a map-reduce operation, where each ray is mapped to a color and then the group of rays corresponding to each pixel is reduced to a single color.

  = Design Methodology

  == Rendering

  The rendering of an image in our path tracer can be summarized in the following pseudocode:

  #figure(
    kind: "algorithm",
    supplement: [Algorithm],

    pseudocode-list(
      booktabs: true,
      numbered-title: [Rendering algorithm for a generic path tracer],
    )[
      + *function* #smallcaps[Render]\(image)
        + *for each* pixel in image:
          + *for each* sample in samples:
            + ray = #smallcaps[GetRay]\(pixel, sample)
            + color = #smallcaps[GetColor]\(ray)
            + pixel.color += color
          + *end for*
          + pixel.color /= samples.size()
        + *end for*
      + *end function*

    ],
  )

  When parallelizing the rendering code, we considered multiple approaches.

  - On one hand spectrum we could parallelize only the rendering of each pixel, by invoking a thread for each pixel and then maintaining the loop over the samples in the kernel. This is the first approach we explored, it's the most straight-forward but becomes inefficient when the number of samples is high. In this configuration, the reduce operation is done at each step, making it more efficient.

  - On the other hand, we could effectively consider the image as having size $("WIDTH" times "SAMPLES" / 2) times ("HEIGHT" times "SAMPLES" / 2)$ and parallelize the rendering of each pixel-sample pair. The reduce operation is then done in parallel and corresponds to a subsampling of the image to the original size. This approach is more efficient and provides the highest possible parallelism, however, it requires a lot of memory, consider for example a 1080p image with 1024 samples per pixel, the expanded image array, considering that each pixel is represented by 16 bytes (during the computations each pixel is represented as a 3D vector of type `Vec3` composed of three floats for a total of 12 bytes plus 4 bytes of padding for alignment purposes), would occupy $1920 times 1080 times 16 times 1024 = 33973862400 "bytes" approx 34 "GB"$ of memory.

  - Another approach we considered was to parallelize the sampling directly by launching a nested kernel inside each pixel kernel. This approach utilizes a niche feature of CUDA, dynamic parallelism @DynParallelism, which allows a kernel to launch another kernel. While we explored initially this approach, we wanted to keep the code compilable with clang, which does not support this feature.

  - Finally, we could consider a hybrid approach, where we maintain the loop over the samples in the kernel, but we diminish the need to have a high number of samples by averaging multiple images. This is the approach that we decided to present as the final implementation and is summarized in @camera-rendering-model.

  #figure(
    image("imgs/camera.drawio.svg"),
    caption: [Camera rendering model],
  ) <camera-rendering-model>

  == Querying for Rays

  // talk about getRay (pseudocode), show image with the querying in the circle inscribed in each pixel
  For the ray querying we used the Get2Rays function. In order to achieve anti-aliasing, we sample an area of half-pixel around the center of each pixel. This results in smoother edges in the rendered picture and in the same way more realistic pictures.

  #figure(
    kind: "algorithm",
    supplement: [Algorithm],

    pseudocode-list(
      booktabs: true,
      numbered-title: [Get2Rays funtion],
    )[
      + *function* #smallcaps[Get2Rays]\(camera)
        + sampleA = center + offsetA
        + sampleB = center + offsetB
      + *if defocusAngle > 0*:
        + originA = defocusDiskSample
        + originB = defocusDiskSample
      + directionA = sampleA - originA
      + directionB = sampleB - originB
      + *end if*
      + *end function*
    ],
  )
  The function uses two half-pixel offsets A and B. Based on that, we calculate sampleA and sampleB. If defocusAngle of the camera is more than 0 then we have a new origin if not, the origin remains as default. We calculate the directionA and directionB and in the result we get two rays from origins to directions for A and B, respectively.


  === Defocus Blur <defocus-blur>

  Defocus blur is a photographic effect that simulates the out-of-focus areas of an image produced by a camera lens with an arbitrarily large aperture (and thus, arbitrarly small depth of field). It is also known as bokeh.

  #figure(
    image("imgs/defocus_blur.drawio.svg"),
    caption: [When we want to simulate the defocus blur effect, each sample from the lens comes from a disk instead of a single point, shapes not aligned with the focus plane will appear blurred],
  )

  // TODO: add image with defocus blur effect applied

  == Tracing Rays

  // talk about getColor (pseudocode)
  To get the appropriate color for the pixel we use the function GetColor, which can be described in pseudocode below:
  #figure(
    kind: "algorithm",
    supplement: [Algorithm],

    pseudocode-list(
      booktabs: true,
      numbered-title: [GetColor function],
    )[
      + *function* #smallcaps[GetColor]\(camera)
        + *for each* i < depth:
          + *if no object hit* return background color
          + *if any object hit*:
            + scatter = #smallcaps[scatter]\(object material, attenuation)
            + emitted = #smallcaps[emitted]\(object material, attenuation)
          + *if scatter*:
            pixel.color = pixel.color \* attenuation + emitted
          + *else*:
            return emitted
          + *end if*
        + *end for*
      + *end function*
    ],
  )
  For an arbitrarily set up rendering depth, which is most commonly between 10 and 50, we do the following: firstly, check if any shape was hit. If not, then we return the background color. If so, we save the information about the hitting point, including the material of the hit object. Based on this information, we update the scatter and emmitted values which are then combined as a result color. The function was inspired mainly by the RayColor function @Shirley2024RTW2. However, our approach focused on iterative calculations in order to omit recursive calls. We decided to set our depth value on [...] resulting in the best quality / processing time ratio.

  === Materials

  // TODO: talk about materials

  == CUDA Features

  === Streams

  To parallelize the rendering of multiple images we utilized CUDA streams @CUDAStream. This allowed us to render multiple images in parallel, by launching multiple kernels in different streams.

  === Pinned Memory

  To eliminate completely the need for copying data between the host and the device, we used pinned memory to allocate the image buffer @PinnedMemory. This allows us to directly access the memory from the device without the need to copy it. This proves useful also when reducing multiple images into the final one, as we can directly write the result in the final image buffer without the need to allocate a temporary buffer.

  === Thrust

  In our project we used the Thrust library @ThrustLib to abstract away some operations, like the array allocations. This makes it explicit where we are using device allocated arrays, and where we are using host allocated arrays.

  === Polymorphism

  CUDA does not fully support all features of abstract classes, in particular for our use case we wanted to support polymorphic access to the `Material`, `Texture` and `Shape` classes, making the API of our library flexible and easy to use. To achieve this while maintaining type safety, we defined the `Material`, `Texture` and `Shape` classes as unions of types instead, this feature is supported natively from CUDA Toolkit V12.4 onwards with the `cuda::std::variant` class. This has the disadvantage of making the two classes less easily extensible, requiring redefinition of the union type and recompilation of the library.

  = Experiments
  // the experiments set up
  The experiments were conducted ... // hardware requirements
  We used two computationally demanding scenes to render: [3 spheres] and [cornellBox]. Both of them included various shapes of different materials and textures as well as rotations and movement. The scenes are presented in figures 2. and 3.
  #figure(
    image("imgs/spheres.png"),
    caption: [Spheres scene],
  ) <spheres>

  #figure(
    image("imgs/boxes.png"),
    caption: [Boxes scene],
  ) <boxes>

  For each scene we measured... // types of tests


  === Random <random>

  Random number generation is not an easy problem, to tackle it, we decided to use the cuRAND library @curand. A lot of considerations were made in how to write a performant kernel, given that we have to use random states in various parts of our program.

  For starters, random states are used for sub-pixel sampling, an antialiasing technique which allows us to eliminate jagged edges from the objects. We also use random states to simulate the defocus blur effect @defocus-blur. Random states are also used for some materials, for example to scatter light in a diffuse material.

  In chapter 3.6 of #cite(<curand>, form: "prose") the authors suggests using a setup kernel to inizialize all the random states, in our case we couldn't measure any improvements and we, on the contrary found it more efficient to inizialize the random states directly in the kernel. It's clear though that a setup kernel might be beneficial in the case we wanted to focus on supporting more efficiently multiple image rendering, allowing us to memeoize the random states.

  The other possible improvement is the usage of the `curandStatePhilox4_32_10_t` random state type instead of the `curandState` type. At the cost of a slightly higher memory usage (64 bytes instead of 48 bytes), we are able to generate four random numbers at once through the usage of the `curand_uniform4` function.

  = Results

  // Include all necessary tables (and make sure they are completely filled out). Include all relevant figures. Introduce all tables and figures in text BEFORE they appear in the report. When answering questions, always provide explanations and reasoning for your answers. If you don’t know what a question or requirement is asking for, please ask us in advance! We are here to help you learn.

  == Different compiler and Link Time Optimization (LTO)

  CUDA code can be compiled with different compilers, we tested our code with `nvcc` and `clang`. While we had a preference with clang due to the great integration with the LLVM suite of tools (like `clang-format`, `clang-tidy` and in particular `clangd`, which provides modern LSP features), we found it lacking in some features, notably the aforementioned dynamic parallelism but more importantly the lack of support for LTO (also known as _Interprocedural Optimization_ or IPO).

  While the code compiled with clang was generally faster without LTO, this feature caused `nvcc` to completely overtake the performance of clang. In our CMake configuration LTO support is checked and enabled automatically with the `checkIPOSupported` function of CMake.

  ```cmake
  include(CheckIPOSupported)
  check_ipo_supported(RESULT supported OUTPUT error)

  if (supported)
      set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
  else()
      message(WARNING "IPO is not supported: ${error}")
  endif()
  ```

  For this reason, all the benchmarks presented in this report were done with code compiled using `nvcc` with LTO enabled.


  == Benchmarks

  Benchmarks were performed in two ways, a first one using #cite(<Catch2>, form: "prose") built-in tooling, meaning that we rely on CPU timers instead of `cudaEvent` timers. This is a potential future improvement of our benchmarking system but requires integration upstream. We believe this tradeoff is acceptable, given that on #cite(<CUDAGuide>, form: "prose"), in the section dedicated to benchmarking and timings, the authors present both methods as valid (provided we explicitly synchronize the device after the kernel launch for the CPU timers).

  The second method consists of using the Nsight Systems profiler by NVIDIA, this enables us to have a more granular view of the perfomance of our program and, by analyzing the `.sqlite` file generated by the profiler, we can better identify potential bottlnecks in our code.

  The hyperparameters that are tunable in our path tracer are the following:
  / BlockSize: The `dim3` size of the block for launching the kernel
  / NumSamples: The number of samples for each pixel, to obtain a clean image, each pixel needs to be sampled randomly a high number of times to eliminate the noise
  / NumImages: The number of images to average to obtain the final image, this provides the same effect as increasing the number of samples, but can be used to avoid kernels with too high of a block size
  / Depth: The maximum depth of the ray tracing algorithm, this is the number of times a ray can bounce before it is considered lost. We fix this at 50, which we consider a good tradeoff between quality and performance
  / AverageWithThrust: A boolean flag that allows us to choose between averaging the images with a custom kernel or with the `thrust::transform` function, this was done mainly to explore NVIDIA's `thrust` library and see if our naive implementation is competitive with it
  / State: The random state generator, we explored the default generator and the Philox generator

  // TODO: Add a table with the benchmarks when the code is ready

  === Random in Unit Disk

  As mentioned in @random, changing the random state generator can have a big impact on the performance of the program. For the defocus blur effect we need to generate random points in a unit disk, to do so we explore three methods: two that use rejection sampling and one that generates random points directly by calculating the density function.

  In general, the Philox generators performs slightly worse than the default generator using rejection sampling, even though only $pi / 4$ of the points are accepted.
  == Notes

  - Talk about changing block size
  - talk about LTO
  - talk about reducing `poll` and `ioctl` calls
  - talk about reducing in parallel with `thrust`
  - custom kernel vs `thrust::transform_reduce`, talk about it, benchmark it
  - talk about cudaOccupacyAPI.

  == Limitations and future research

  = Conclusions
  To sum up,
  // Restate the purpose or objective of the assignment. Was the exercise successful in fulfilling its intended purpose? Why was it (or wasn’t it)? Summarize your results. Draw general conclusions based on your results (what did you learn?)
]
