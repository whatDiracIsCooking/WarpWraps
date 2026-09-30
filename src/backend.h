/**
 * @file backend.h
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
 *
 * The _RAW variants map onto the vendor's OWN global names (::curand*,
 * ::hiprand*) rather than the ::wwr::cuda / ::wwr::hip module re-exports. They
 * are for a neutral module whose vendor header it reaches by #include -- through
 * a src/-root sibling .h -- instead of by importing the raw module. That is only
 * safe when the vendor host API has external linkage, because a reference or a
 * type alias needs no more than the declaration the header supplies; wwr.rand is
 * the sole such module (cuRAND / hipRAND are real libraries). A module wrapping
 * static-inline vendor math (complex, fp16, bf16) cannot use these: an exported
 * inline naming a TU-local static-inline function is ill-formed, so it must
 * import its raw module for that module's external-linkage wrappers. See
 * src/rand.h and docs/architecture.md, section 12.
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

// Raw-name variants -- bind to the vendor's own global names, for a module that
// #includes the vendor header directly rather than importing its raw module.
// See this file's header for when (and only when) these apply.
#if defined(WWR_GPU_BACKEND_CUDA)
#define WWR_SELECT_RAW(cuda_name, hip_name) ::cuda_name
#else
#define WWR_SELECT_RAW(cuda_name, hip_name) ::hip_name
#endif

#define WWR_TYPE_RAW(wwr_name, cuda_name, hip_name)                                             \
  using wwr_name = WWR_SELECT_RAW(cuda_name, hip_name);

#define WWR_VALUE_RAW(wwr_name, cuda_name, hip_name)                                            \
  inline constexpr auto wwr_name = WWR_SELECT_RAW(cuda_name, hip_name);

#define WWR_FUNCTION_RAW(wwr_name, cuda_name, hip_name)                                         \
  inline constexpr auto &wwr_name = WWR_SELECT_RAW(cuda_name, hip_name);
