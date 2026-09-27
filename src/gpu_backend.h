/**
 * @file gpu_backend.h
 * @brief The CUDA-or-HIP switch point for src modules
 *
 * Included ONLY in the global module fragment of a src module. Every such
 * module exports backend-neutral gpu* names into namespace wwr; this header
 * picks which backend's raw module those names refer to.
 *
 * Exactly one of WWR_GPU_BACKEND_CUDA / WWR_GPU_BACKEND_HIP is defined by
 * the gpumod_backend CMake target, PRIVATE to the src targets. Macros
 * defined in a module unit do not leak to importers, so none of these are
 * visible outside src.
 *
 *   WWR_TYPE(gpu, cuda, hip)      using gpu = <backend type>;
 *   WWR_VALUE(gpu, cuda, hip)     inline constexpr auto gpu = <backend constant>;
 *   WWR_FUNCTION(gpu, cuda, hip)  inline constexpr auto& gpu = <backend function>;
 *
 * WWR_FUNCTION binds a reference to the backend's own function, so the
 * signature is never restated and cannot drift. A function reference carries no
 * default arguments and cannot name an overload set -- write an explicit
 * forwarding function for those. See docs/architecture.md, section 4.
 */

#pragma once

#if defined(WWR_GPU_BACKEND_CUDA) == defined(WWR_GPU_BACKEND_HIP)
#error                                                                                             \
    "Define exactly one of WWR_GPU_BACKEND_CUDA or WWR_GPU_BACKEND_HIP (link gpumod_backend)"
#endif

// The raw CUDA modules (src/cuda) export into wwr::cuda, the raw HIP
// modules (src/hip) into wwr::hip.
#if defined(WWR_GPU_BACKEND_CUDA)
#define WWR_SELECT(cuda_name, hip_name) ::wwr::cuda::cuda_name
#else
#define WWR_SELECT(cuda_name, hip_name) ::wwr::hip::hip_name
#endif

#define WWR_TYPE(gpu_name, cuda_name, hip_name)                                                 \
  using gpu_name = WWR_SELECT(cuda_name, hip_name);

#define WWR_VALUE(gpu_name, cuda_name, hip_name)                                                \
  inline constexpr auto gpu_name = WWR_SELECT(cuda_name, hip_name);

#define WWR_FUNCTION(gpu_name, cuda_name, hip_name)                                             \
  inline constexpr auto &gpu_name = WWR_SELECT(cuda_name, hip_name);
