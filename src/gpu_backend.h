/**
 * @file gpu_backend.h
 * @brief The CUDA-or-HIP switch point for src modules
 *
 * Included ONLY in the global module fragment of a src module. Every such
 * module exports backend-neutral gpu* names into namespace gpumod; this header
 * picks which backend's raw module those names refer to.
 *
 * Exactly one of GPUMOD_GPU_BACKEND_CUDA / GPUMOD_GPU_BACKEND_HIP is defined by
 * the gpumod_backend CMake target, PRIVATE to the src targets. Macros
 * defined in a module unit do not leak to importers, so none of these are
 * visible outside src.
 *
 *   GPUMOD_TYPE(gpu, cuda, hip)      using gpu = <backend type>;
 *   GPUMOD_VALUE(gpu, cuda, hip)     inline constexpr auto gpu = <backend constant>;
 *   GPUMOD_FUNCTION(gpu, cuda, hip)  inline constexpr auto& gpu = <backend function>;
 *
 * GPUMOD_FUNCTION binds a reference to the backend's own function, so the
 * signature is never restated and cannot drift. A function reference carries no
 * default arguments and cannot name an overload set -- write an explicit
 * forwarding function for those. See docs/architecture.md, section 4.
 */

#pragma once

#if defined(GPUMOD_GPU_BACKEND_CUDA) == defined(GPUMOD_GPU_BACKEND_HIP)
#error                                                                                             \
    "Define exactly one of GPUMOD_GPU_BACKEND_CUDA or GPUMOD_GPU_BACKEND_HIP (link gpumod_backend)"
#endif

// The raw CUDA modules (src/cuda) export into gpumod::cuda, the raw HIP
// modules (src/hip) into gpumod::hip.
#if defined(GPUMOD_GPU_BACKEND_CUDA)
#define GPUMOD_SELECT(cuda_name, hip_name) ::gpumod::cuda::cuda_name
#else
#define GPUMOD_SELECT(cuda_name, hip_name) ::gpumod::hip::hip_name
#endif

#define GPUMOD_TYPE(gpu_name, cuda_name, hip_name)                                                 \
  using gpu_name = GPUMOD_SELECT(cuda_name, hip_name);

#define GPUMOD_VALUE(gpu_name, cuda_name, hip_name)                                                \
  inline constexpr auto gpu_name = GPUMOD_SELECT(cuda_name, hip_name);

#define GPUMOD_FUNCTION(gpu_name, cuda_name, hip_name)                                             \
  inline constexpr auto &gpu_name = GPUMOD_SELECT(cuda_name, hip_name);
