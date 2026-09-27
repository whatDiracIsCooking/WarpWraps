/**
 * @file gpu_backend.h
 * @brief The CUDA-or-HIP switch point for src modules
 *
 * Included ONLY in the global module fragment of a src module. Every such
 * module exports backend-neutral wwr* names into namespace wwr; this header
 * picks which backend's raw module those names refer to.
 *
 * Exactly one of WWR_GPU_BACKEND_CUDA / WWR_GPU_BACKEND_HIP is defined by
 * the wwr_backend CMake target, PRIVATE to the src targets. Macros
 * defined in a module unit do not leak to importers, so none of these are
 * visible outside src.
 *
 *   WWR_TYPE(wwr, cuda, hip)      using wwr = <backend type>;
 *   WWR_VALUE(wwr, cuda, hip)     inline constexpr auto wwr = <backend constant>;
 *   WWR_FUNCTION(wwr, cuda, hip)  inline constexpr auto& wwr = <backend function>;
 *
 * WWR_FUNCTION binds a reference to the backend's own function, so the
 * signature is never restated and cannot drift. A function reference carries no
 * default arguments and cannot name an overload set -- write an explicit
 * forwarding function for those. See docs/architecture.md, section 4.
 */

#pragma once

#if defined(WWR_GPU_BACKEND_CUDA) == defined(WWR_GPU_BACKEND_HIP)
#error                                                                                             \
    "Define exactly one of WWR_GPU_BACKEND_CUDA or WWR_GPU_BACKEND_HIP (link wwr_backend)"
#endif

// The raw CUDA modules (src/cuda) export into wwr::cuda, the raw HIP
// modules (src/hip) into wwr::hip.
#if defined(WWR_GPU_BACKEND_CUDA)
#define WWR_SELECT(cuda_name, hip_name) ::wwr::cuda::cuda_name
#else
#define WWR_SELECT(cuda_name, hip_name) ::wwr::hip::hip_name
#endif

#define WWR_TYPE(wwr_name, cuda_name, hip_name)                                                 \
  using wwr_name = WWR_SELECT(cuda_name, hip_name);

#define WWR_VALUE(wwr_name, cuda_name, hip_name)                                                \
  inline constexpr auto wwr_name = WWR_SELECT(cuda_name, hip_name);

#define WWR_FUNCTION(wwr_name, cuda_name, hip_name)                                             \
  inline constexpr auto &wwr_name = WWR_SELECT(cuda_name, hip_name);
