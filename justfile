set positional-arguments

export CMAKE_BUILD_TYPE := "Release"

build_dir := "build"
bench_dir := "bench"
target    := build_dir / "apps/cuda_path_tracer"

# Lists all available targets
@default:
    just --list

# Builds the application with the specified CMake arguments
@build *CMAKE_ARGS:
    cmake -S . -B {{build_dir}} $@
    cmake --build {{build_dir}}

# Builds the application with testing enabled
@test *TEST_ARGS: (build "-DBUILD_TESTING=ON")
    ./{{build_dir}}/tests/tests $@

# Runs the application
@run *RUN_ARGS: (build)
    ./{{target}} $@

# Benchmarks the application using NVIDIA Nsight Systems, run with `sudo` for better results
@bench *RUN_ARGS: (build)
    mkdir -p {{bench_dir}}
    rm -rf {{bench_dir}}/*
    nsys profile --stats=true -o {{bench_dir}}/bench ./{{target}} $@
    nsys analyze {{bench_dir}}/bench.sqlite

# Uses the NVIDIA Compute Sanitizer to check for memory leaks
@memcheck *RUN_ARGS: (build)
    compute-sanitizer --show-backtrace yes --tool memcheck --leak-check full ./{{target}} $@

# Cleans the build and benchmark directories, as well as any generated images
@clean:
    rm -rf {{build_dir}} {{bench_dir}} *.ppm
