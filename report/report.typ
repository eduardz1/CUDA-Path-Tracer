#import "template.typ": template, eqcolumns, pseudocode-list


#show: template.with(
  title: [CUDA Path Tracer],
  subtitle: [Report for the course of GP-GPU Programming],
  authors: (
    "Dominika Bocheńczyk",
    "Eduard Occhipinti",
  ), //for the alphabetical order
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

#columns(2)[
  = Introduction

  // Give a brief overview of the overall algorithm (what is the algorithm you are parallelizing?). Identify where parallelism can be introduced to the algorithm. Discuss the code executed by the master and the slaves. Discuss how the master and slaves communicate and why they communicate. Provide figures and equations to support your explanations. There should be no results in this section. After reading this section, we should have a complete understanding of how you solved the problem without having to read your code for further details. At the same time, there should be little to no code in your report.

  The code we are parallelizing is that of a path tracer. This algorithm has the advantage of being embarrassingly parallel due to its nature. Each pixel is sampled $N$ times and each sample is independent of the other.

  The algorithm can be visualized as a map-reduce operation, where each ray is mapped to a color and then the group of rays corresponding to each pixel is reduced to a single color.

  Path tracing is fundamentally a Monte Carlo method, where we sample the light transport equation to estimate the radiance at a point in the scene. The algorithm is recursive, as each ray can spawn new rays when it hits a surface, this is done until a maximum depth is reached.

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
          + pixel.color /= samples.size()
    ],
  )

  When parallelizing the rendering code, we considered multiple approaches.

  - On one hand of the spectrum, we could parallelize only the rendering of each pixel, by invoking a thread for each pixel and then maintaining the loop over the samples in the kernel. This is the first approach we explored, it's the most straight-forward but becomes inefficient when the number of samples is high. In this configuration, the reduce operation is done at each step, making it more efficient.

  - On the other hand, we could effectively consider the image as having size $("WIDTH" times "SAMPLES" / 2) times ("HEIGHT" times "SAMPLES" / 2)$ and parallelize the rendering of each pixel-sample pair. The reduce operation is then done in parallel and corresponds to a subsampling of the image to the original size. This approach is more efficient and provides the highest possible parallelism, however, it requires a lot of memory, consider for example a 1080p image with 1024 samples per pixel, the expanded image array, considering that each pixel is represented by 16 bytes (during the computations each pixel is represented as a 3D vector of type `Vec3` composed of three floats for a total of 12 bytes plus 4 bytes of padding for alignment purposes), would occupy $1920 times 1080 times 16 times 1024 = 33973862400 "bytes" approx 34 "GB"$ of memory.

  - Another approach we considered was to parallelize the sampling directly by launching a nested kernel inside each pixel kernel. This approach utilizes a niche feature of CUDA, dynamic parallelism @DynParallelism, which allows a kernel to launch another kernel. While we explored initially this approach, we wanted to keep the code compilable with the Clang compiler, which does not support this feature.

  - Finally, we could consider a hybrid approach, where we maintain the loop over the samples in the kernel, but we diminish the need to have a high number of samples by averaging multiple images. This is the approach that we decided to present as the final implementation and is summarized in @camera-rendering-model.

  #figure(
    image("imgs/camera.drawio.svg"),
    caption: [Camera rendering model],
  ) <camera-rendering-model>

  == Querying for Rays

  For querying rays we used the function described in @get-ray. In order to achieve anti-aliasing, we sample an area of half-pixel around the center of each pixel. This results in smoother edges in the rendered picture and in the same way more realistic pictures.

  #figure(
    kind: "algorithm",
    supplement: [Algorithm],

    pseudocode-list(
      booktabs: true,
      numbered-title: [Function to generate rays],
    )[
      + *function* #smallcaps[GetRay]\(camera, x, y)
        + a = random number between 0 and 1
        + b = random number between 0 and 1
        + offset = #smallcaps[Vec3]\(a, b, 0) - #smallcaps[Vec3]\(0.5, 0.5, 0)
        + sample = coordinate of pixel + offset
        + *if camera.defocusAngle > 0*:
          + origin = #smallcaps[DefocusDiskSample]\(camera)
        + *else*:
          + origin = camera.origin
        + direction = sample - origin
        + *return* #smallcaps[Ray]\(origin, direction)
    ],
  ) <get-ray>

  #colbreak()

  === Defocus Blur <defocus-blur>

  Defocus blur is a photographic effect that simulates the out-of-focus areas of an image produced by a camera lens with an arbitrarily large aperture (and thus, arbitrarly small depth of field). It is also known as bokeh.

  #figure(
    grid(
      columns: 1,
      row-gutter: 1em,
      image("imgs/defocus_blur.png"),
      image("imgs/defocus_blur.drawio.svg"),
    ),
    caption: [When we want to simulate the defocus blur effect, each sample from the lens comes from a disk instead of a single point, shapes not aligned with the focus plane will appear blurred],
  )

  #colbreak()

  == Tracing Rays

  To get the appropriate color for the pixel we use the function `GetColor`, which can be described in pseudocode as follows:

  #figure(
    kind: "algorithm",
    supplement: [Algorithm],

    pseudocode-list(
      booktabs: true,
      numbered-title: [Function to get the color of a ray],
    )[
      + *function* #smallcaps[GetColor]\(ray)
        + *if depth $<=$ 0*:
          + return color black;
        + *if no object hit*:
          + return background color
        + scattered = new ray
        + attenuation = color white
        + scatter = object material.#smallcaps[Scatter]\(ray, attenuation)
        + emitted = object material.#smallcaps[Emitted]\(ray, attenuation, scattered)
        + *if not scatter*:
          + return emitted
        + *return* emitted + attenuation \* #smallcaps[GetColor]\(scattered, depth - 1)
    ],
  ) <get-color>

  For a given render depth we do the following: first we check if we exceeded the maximum depth, if so we return black. If not, we check if any object was hit by the ray. If not, we return the background color. At this point we can compute the information about the emitted light produced by the material and calculate the scattered ray. If the material is not scattering the light, we return the emitted light. Otherwise, we return the sum of the emitted light and the attenuation of the scattered ray.

  The algorithm is inspired by #cite(<Shirley2024RTW2>, form: "prose") but ours was converted to an iterative approach, which generally performs better on the GPU. Our algorithm also includes an early stopping criteria, called "Russian Roulette" which randomly stops the recursion of the ray based on its contribution to the final color, meaning that ray carrying very little information are likely to be stopped early.

  #figure(
    image("imgs/cornell_box.drawio.svg"),
    caption: [Contrary to real world physics, in ray tracers, normally, the rays are not emitted from the light sources],
  )

  === Materials

  In our work we used different materials and textures for the generated shapes inspired by a subset of materials proposed by #cite(<Shirley2024RTW2>, form: "prose"). These included solid color and checkered texture as well as the following materials: lambertian, dielectric (particularly glass), metal and light (treating the shape as a light source). As in regular ray tracing the materials were calculated based on the hitting points and their physical properties, such as reflection, refraction, fuzziness or emission.

  #figure(
    scope: "parent",
    placement: auto,
    image("imgs/reflection_showcase.png"),
    caption: [On the sides, we have two metal spheres of different fuzziness, in the middle a glass sphere, a light sphere is placed on the top, out of frame. Behind we see a rectanguar cuboid with a lambertian texture, it's interesting seeing how the light gets reflected and refracted by the glass sphere],
  ) <boxes>

  == CUDA Features

  === Streams

  To parallelize the rendering of multiple images we utilized CUDA streams @CUDAStream. This allowed us to render multiple images in parallel, by launching multiple kernels in different streams.

  === Pinned Memory

  To eliminate completely the need for copying data between the host and the device, we used pinned memory to allocate the image buffer @PinnedMemory. This allows us to directly access the memory from the device without the need to copy it. This proves useful also when reducing multiple images into the final one, as we can directly write the result in the final image buffer without the need to allocate a temporary buffer.

  === Thrust

  In our project we used the Thrust library @ThrustLib to abstract away some operations, like the array allocations. This makes it explicit where we are using device allocated arrays, and where we are using host allocated arrays.

  === Polymorphism

  CUDA does not fully support all features of abstract classes, in particular for our use case we wanted to support polymorphic access to the `Material`, `Texture` and `Shape` classes, making the API of our library flexible and easy to use. To achieve this while maintaining type safety, we defined the `Material`, `Texture` and `Shape` classes as unions of types instead, this feature is supported natively from CUDA Toolkit V12.4 onwards with the `cuda::std::variant` class. This has the disadvantage of making the two classes less easily extensible, requiring redefinition of the union type and recompilation of the library.

  === Shared Memory

  We opted to avoid using shared memory due to the fact that the only real benefit in our algorithm would result from the caching of the `Shape` objects. Given the standard limit of 48KB, we estimate that we could cache only around 60 `Shape` objects, which is enough to show some performance improvements in our simple scenes but is clearly not scalable, we also tried decoupling shapes from their materials but, given that they are dependant from each other, the advanges of shared memory were lost. Regarding the averaging kernel, it's already optimal, given that we reduce the expanded image of $N$ pixels to the original size $M$ with $N$ global reads and $M$ global writes.

  === Random <random>

  Random number generation is not an easy problem, to tackle it, we decided to use the cuRAND library @curand. A lot of considerations were made in how to write a performant kernel, given that we have to use random states in various parts of our program.

  For starters, random states are used for sub-pixel sampling, an antialiasing technique which allows us to eliminate jagged edges from the objects. We also use random states to simulate the defocus blur effect @defocus-blur. Random states are also used for some materials, for example to scatter light in a diffuse material.

  In chapter 3.6 of #cite(<curand>, form: "prose") the authors suggests using a setup kernel to inizialize all the random states, in our case we couldn't measure any improvements and we, on the contrary found it more efficient to initialize the random states directly in the kernel. It's clear though that a setup kernel might be beneficial in the case we wanted to focus on supporting more efficiently multiple image rendering, allowing us to memeoize the random states.

  The other possible improvement is the usage of the `curandStatePhilox4_32_10_t` random state type instead of the `curandState` type. At the cost of a slightly higher memory usage (64 bytes instead of 48 bytes), we are able to generate four random numbers at once through the usage of the `curand_uniform4` function.

  == Floating point numbers

  Particular care was taken in avoiding the usage of double precision floating point numbers, through explicit casts and declarations. We know that in general GPUs are not optimized for double precision floating point numbers, and we wanted to avoid any potential performance hit.

  = Experiments and Results

  // Include all necessary tables (and make sure they are completely filled out). Include all relevant figures. Introduce all tables and figures in text BEFORE they appear in the report. When answering questions, always provide explanations and reasoning for your answers. If you don’t know what a question or requirement is asking for, please ask us in advance! We are here to help you learn.

  #figure(
    scope: "parent",
    placement: top,
    table(
      columns: 6,
      align: horizon,
      stroke: (x: none, y: 0.5pt + black),
      row-gutter: (2.2pt, auto),
      table.header(
        [*GPU*],
        [*CPU*],
        [*RAM*],
        [*OS & Kernel*],
        [*Compilers*],
        [*CUDA*],
      ),

      [NVIDIA GeForce\ RTX 3060Ti LHR],
      [Intel i5-11400F \ (12) \@ 4.400GHz],
      [46763MiB],
      [Ubuntu 24.10 with kernel 6.11.0-8-generic],
      [GCC 14.2.0 \ NVCC 12.8.61],
      [v2.7.0],
    ),
    caption: [System specifications used for the benchmarks],
  )

  == Scene

  For the benchmarks we used one scene presented in @benchmark. It includes various shapes of different materials and textures as well as rotations and transaltions. Although int the picture seems to only include 4 spheres, the scene actually includes also additional sphere, parallelogram and rectangular-cuboid.

  #figure(
    image("imgs/benchmark.png", height: 25%),
    caption: [Benchmark scene],
  ) <benchmark>

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

  == Reducing control divergence

  Control divergence in a path tracer is an inherent problem, as each ray can hit a different object and thus follow a different path. One way we can reduce control divergence is by expanding the image array with the goal of aligning it with the warp size.

  == Checking for memory leaks

  We used the `compute-sanitizer` tool, a standard tool bundled with the NVIDIA Toolkit, to check for memory leaks in our code. The command can be run with the Just command runner as `just memcheck -s examples/defocus_blur.json`, where we pass the defocus blur demo scene as the scene argument.

  == Benchmarks

  Benchmarks were performed in two ways, a first one using #cite(<Catch2>, form: "prose") built-in tooling, meaning that we rely on CPU timers instead of `cudaEvent` timers. This is a potential future improvement of our benchmarking system but requires integration upstream. We believe this tradeoff is acceptable, given that on #cite(<CUDAGuide>, form: "prose"), in the section dedicated to benchmarking and timings, the authors present both methods as valid (provided we explicitly synchronize the device after the kernel launch for the CPU timers).

  The second method consists of using the Nsight Systems profiler by NVIDIA, this enables us to have a more granular view of the perfomance of our program and, by analyzing the `.sqlite` file generated by the profiler, we can better identify potential bottlnecks in our code.

  The hyperparameters that are tunable in our path tracer are the following:

  / `BLOCK_SIZE`: The `dim3` size of the block for launching the kernel

  / `NUM_SAMPLES`: The number of samples for each pixel, to obtain a clean image, each pixel needs to be sampled randomly a high number of times to eliminate the noise
  / `NUM_IMAGES`: The number of images to average to obtain the final image, this provides the same effect as increasing the number of samples, but can be used to avoid kernels with too high of a block size
  / `DEPTH`: The maximum depth of the ray tracing algorithm, this is the number of times a ray can bounce before it is considered lost. We fix this at 50, which we consider a good tradeoff between quality and performance
  / `AVG_WITH_THRUST`: A boolean flag that allows us to choose between averaging the images with a custom kernel or with the `thrust::transform` function, this was done mainly to explore NVIDIA's `thrust` library and see if our naive implementation is competitive with it
  / `STATE`: The random state generator, we explored the default generator and the Philox generator

  To allow for maximum flexibility without any perfomance overhead, the hyperparameters are defined as template arguments of the `Camera` class and of the kernel responsible for the rendering of the image.

  === `curandState` vs `curandStatePhilox4_32_10`

  As anticipated in @random, we explored the performance of two different random state generators, the default `curandState` and the `curandStatePhilox4_32_10`. Often times the Philox generator allows us to perform less function calls and sometimes to generate more results at once. We perfomed some isolated benchmarks to understand the impact in two of the "hottest" functions in our path tracer.

  ==== Random in Unit Disk/Sphere

  As mentioned in @random, changing the random state generator can have a big impact on the performance of the program. For the defocus blur effect we need to generate random points in a unit disk, to do so we explore three methods: two that use rejection sampling and one that generates random points directly by calculating the density function.

  In general, the Philox generators performs slightly worse than the default generator using rejection sampling, even though only $pi / 4$ of the points are accepted.

  #figure(
    table(
      columns: (auto, auto),
      align: (x, y) => (
        if x > 0 {
          center
        } else {
          left
        }
      ),
      stroke: (x: none, y: 0.5pt + black),
      row-gutter: (2.2pt, auto),
      table.header(
        [*Benchmark name*],
        [*mean ($mu s$)*],
      ),

      "Rejection Sampling", $90.4922$,
      "Rejection Sampling (Philox)", $103.621$,
      "Direct Generation (Philox)", $111.112$,
    ),
    caption: [Different strategies to generate random points in a unit disk],
  ) <c2-rudisk>

  A similar algorithm can be applied to generate random points in a unit sphere, an operation that is needed to scatter light in a diffuse material.

  ==== Ray generation

  The generation of rays is a fundamental operation in a path tracer, as it is the first step in the rendering pipeline. We explored different strategies to generate rays, from the simple generation of a single ray to the generation of multiple rays at once, using the Philox generator. Again, the Philox generator performs slightly worse than the default generator, even though it generates four random numbers at once. This is slightly surprising because earlier in our optimization journey the Philox generator was performing consistently better than the default generator.

  #figure(
    table(
      columns: (auto, auto),
      align: (x, y) => (
        if x > 0 {
          center
        } else {
          left
        }
      ),
      stroke: (x: none, y: 0.5pt + black),
      row-gutter: (2.2pt, auto),
      table.header(
        [*Benchmark name*],
        [*mean ($mu s$)*],
      ),

      "Single Ray Generation", $128.486$,
      "2 Rays at once (Philox)", $140.26$,
      "4 Rays at once (Philox)", $177.945$,
    ),
    caption: [Different strategies to generate rays benchmarked against each other, the times are normalized to account for the amount of rays generated],
  ) <c2-raygen>

  ==== Impact on the whole pipeline

  The impact of the random state generator on the whole pipeline is not as big as we expected.

  #figure(
    image("imgs/rng_difference.svg"),
    caption: [As we can see the difference betwen the two is negligible],
  ) <rng-difference>

  === Averaging with `thrust`

  The perfomance is extremely similar to our custom kernel and, given that averaging only accounts for less than 1% of the total time, picking one over the other is a matter of personal preference.

  #figure(
    scope: "parent",
    placement: top,
    grid(
      columns: 3,
      column-gutter: -2em,
      image("imgs/b_high.svg"),
      image("imgs/b_mid.svg"),
      image("imgs/b_low.svg"),
    ),
    caption: [The two most important hyperparameters to track are the number of samples compared with the number of images to average (keep in mind that the product of the two remains constant) and the block size. Here we present the two as heatmaps, for the three quality presets we provide in our application],
  ) <all-benches>

  === Block Size

  With our benchmark we concluded that the `4x4` kernel is measurably worse than all the others, while the other choices are all very similar.

  === Number of Samples vs Number of Images tradeoff

  From the heatmaps in @all-benches we can see that the general trend sees a higher sample count performing better than a high image count consistently. It's interesting seeing that the higher the quality preset the lower the impact of this tradeoff.

  === Final configurations

  #table(
    columns: (auto, auto),
    align: (x, y) => (
      if x > 0 {
        center
      } else {
        left
      }
    ),
    stroke: (x: none, y: none),
    table.hline(stroke: 2pt),
    table.cell(colspan: 2, align: center)[*High Quality*],
    table.hline(stroke: 0.5pt),
    [*`BLOCK_SIZE`*], [8 #sym.times 8],
    [*`NUM_SAMPLES`*], [2048],
    [*`NUM_IMAGES`*], [8],
    [*`DEPTH`*], [64],
    [*`AVG_WITH_THRUST`*], [true],
    [*`STATE`*], [curandStatePhilox4_32_10],

    table.hline(stroke: 2pt),
    table.cell(colspan: 2, align: center)[*Medium Quality*],
    table.hline(stroke: 0.5pt),
    [*`BLOCK_SIZE`*], [8 #sym.times 8],
    [*`NUM_SAMPLES`*], [256],
    [*`NUM_IMAGES`*], [8],
    [*`DEPTH`*], [16],
    [*`AVG_WITH_THRUST`*], [false],
    [*`STATE`*], [curandState],

    table.hline(stroke: 2pt),
    table.cell(colspan: 2, align: center)[*Low Quality*],
    table.hline(stroke: 0.5pt),
    [*`BLOCK_SIZE`*], [8 #sym.times 8],
    [*`NUM_SAMPLES`*], [64],
    [*`NUM_IMAGES`*], [4],
    [*`DEPTH`*], [4],
    [*`AVG_WITH_THRUST`*], [true],
    [*`STATE`*], [curandState],
    table.hline(stroke: 2pt),
  )



  // Restate the purpose or objective of the assignment. Was the exercise successful in fulfilling its intended purpose? Why was it (or wasn’t it)? Summarize your results. Draw general conclusions based on your results (what did you learn?)
]

= Conclusions

In summary, in our work we parallelized a classical path tracer. Using different CUDA tools and solutions, we were able to generate high-quality images in a reasonable time. The results look promising and we developed a solid base for potential extension of this work in the future.

We learned a lot about the CUDA programming model and the various tools that NVIDIA provides to developers. We also developed a better understanding of the path tracing algorithm and the various optimizations that can be applied to it. We came to understand the limitations that even embarrassingly parallel algorithms can have (see the ray bouncing algorithm).


