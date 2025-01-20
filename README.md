# CUDA-Path-Tracer

## Commands

If you have the [just](https://github.com/casey/just) command runner installed you can run the command `just` to see the list of available commands. In particular, to run the application use:

```bash
just run
```

Each argument passed to each `just` command is forwarded to `cmake`, for example, to build the project for all major CUDA architectures use:

```bash
just build -DCMAKE_CUDA_ARCHITECTURES=all-major
```
