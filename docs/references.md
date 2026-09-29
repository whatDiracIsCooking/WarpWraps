# References

Related work in CUDA/HIP portability. Each of these solves a real problem well,
and several were instructive while building `wwr`. They span a range of
altitudes — from source-level porting to full performance-portability
frameworks — and this project owes a debt to the ideas they explored first.

| Project | Link | Description |
|---|---|---|
| HIP | https://github.com/ROCm/HIP | AMD's C++ heterogeneous-compute interface — the foundation this whole space is built on. A single kernel language and runtime that compiles for AMD natively and NVIDIA via the CUDA backend. |
| HIPIFY | https://github.com/ROCm/HIPIFY | AMD's source-to-source translators (`hipify-perl`, `hipify-clang`) that convert CUDA to portable HIP. The pragmatic, well-maintained first step for most codebases moving to AMD. |
| hipBLAS / hipFFT / hipRAND / hipSPARSE / hipSOLVER | https://github.com/ROCm/rocm-libraries | AMD's marshalling libraries that expose one API and dispatch to rocBLAS/cuBLAS (etc.) depending on the hardware — vendor-neutral math libraries that already run on both backends. |
| hop | https://github.com/cschpc/hop | Header-Only Porting (CSC). A lightweight, header-only layer that maps runtime, BLAS, FFT, RAND and SPARSE calls to neutral `gpu*` names, works both directions, and reaches C and Fortran as well as C++. The closest neighbour to `wwr` in spirit. |
| cudawrappers | https://github.com/nlesc-recruit/cudawrappers | Ergonomic C++ RAII wrappers (exceptions, resource management) over the CUDA driver API, NVRTC, cuFFT, NVML and NVTX, with a HIP backend. A lovely way to write safer, more compact GPU host code. |
| cuda-api-wrappers | https://github.com/eyalroz/cuda-api-wrappers | Thin, modern-C++ wrappers over the CUDA runtime and driver APIs — idiomatic, header-only, and a pleasure to read. An excellent reference for API ergonomics. |
| Kokkos | https://github.com/kokkos/kokkos | A mature C++ performance-portability programming model (CUDA, HIP, SYCL, OpenMP, …), with Kokkos Kernels providing portable BLAS and sparse routines. The reference point for portable HPC. |
| RAJA | https://github.com/LLNL/RAJA | LLNL's loop-abstraction portability layer — a clean, well-supported way to write kernels once and target many backends. |
| Alpaka | https://github.com/alpaka-group/alpaka | A header-only, single-source C++ abstraction over CUDA, HIP, SYCL and CPU backends. Elegant and widely used in the particle-physics community. |
| SYCL / AdaptiveCpp | https://github.com/AdaptiveCpp/AdaptiveCpp | An open, standards-based single-source C++ model targeting NVIDIA, AMD, Intel and CPU. One of the broadest vendor-agnostic stories available today. |
| oneMath (formerly oneMKL Interfaces) | https://github.com/uxlfoundation/oneMath | An open, backend-neutral math-library interface (BLAS, FFT, RNG, …) with backends for cuBLAS, rocBLAS and MKL — vendor-agnostic numerics with a clean runtime-dispatch design. |
| Ginkgo | https://github.com/ginkgo-project/ginkgo | A modern sparse-linear-algebra library with CUDA, HIP, SYCL and OpenMP executors — a great example of a batteries-included, backend-agnostic numerical library. |
| Thrust (CCCL) | https://github.com/NVIDIA/cccl | NVIDIA's parallel-algorithms library (part of CCCL, alongside CUB and libcu++). CUDA-only, but AMD's rocThrust (below) mirrors its interface, so the two together give a familiar API on both vendors. |
| rocThrust | https://github.com/ROCm/rocm-libraries | AMD's source-compatible port of Thrust to HIP — the same high-level parallel-algorithm interface, running on ROCm. |
