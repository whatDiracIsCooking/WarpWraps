/**
 * @file dispatch_macros.h
 * @brief Preprocessor macros for dispatching GPU BLAS function calls based on type
 *
 * The prefix-agnostic WWR_REAL_DISPATCH / WWR_COMPLEX_DISPATCH cores live in
 * wrappers/common/dispatch_sdcz.h; this header binds them to the gpublas* family and
 * adds the _64 index variants and the WWR_USUAL_DISPATCH convenience. Basenames
 * carry no cuBLAS "_v2" suffix: gpublas<X> already stands for cublas<X>_v2 or
 * hipblas<X>.
 *
 * Macros with _64 suffix support both 32-bit and 64-bit integer parameters:
 * - If IntT is int, calls the 32-bit version (no suffix)
 * - If IntT is int64_t, calls the 64-bit version (_64 suffix)
 *
 * Prerequisites (must be provided by including file):
 * - std::is_same_v (via std import or equivalent)
 * - gpuComplex, gpuDoubleComplex and the gpublas* functions
 *   (wwr.complex, wwr.blas)
 */

#pragma once

#include "wrappers/common/dispatch_sdcz.h"

/// @brief Macro for dispatching to float and double with 32-bit or 64-bit integer support.
#define WWR_REAL_DISPATCH_64(T, IntT, prefix, S, D, basename, ...)                              \
  if constexpr (std::is_same_v<IntT, int>) {                                                       \
    WWR_REAL_DISPATCH(T, prefix, S, D, basename, __VA_ARGS__);                                  \
  } else if constexpr (std::is_same_v<IntT, int64_t>) {                                            \
    WWR_REAL_DISPATCH(T, prefix, S, D, basename##_64, __VA_ARGS__);                             \
  }

/// @brief Macro for dispatching to gpuComplex and gpuDoubleComplex with 32-bit or 64-bit integer support.
#define WWR_COMPLEX_DISPATCH_64(T, IntT, prefix, C, Z, basename, ...)                           \
  if constexpr (std::is_same_v<IntT, int>) {                                                       \
    WWR_COMPLEX_DISPATCH(T, prefix, C, Z, basename, __VA_ARGS__);                               \
  } else if constexpr (std::is_same_v<IntT, int64_t>) {                                            \
    WWR_COMPLEX_DISPATCH(T, prefix, C, Z, basename##_64, __VA_ARGS__);                          \
  }

/// @brief Macro for dispatching to usual types (float, double, gpuComplex, gpuDoubleComplex).
#define WWR_USUAL_DISPATCH(T, basename, ...)                                                    \
  WWR_REAL_DISPATCH(T, gpublas, S, D, basename, __VA_ARGS__);                                   \
  WWR_COMPLEX_DISPATCH(T, gpublas, C, Z, basename, __VA_ARGS__);

/// @brief Macro for dispatching to usual types with 32-bit or 64-bit integer support.
#define WWR_USUAL_DISPATCH_64(T, IntT, basename, ...)                                           \
  WWR_REAL_DISPATCH_64(T, IntT, gpublas, S, D, basename, __VA_ARGS__);                          \
  WWR_COMPLEX_DISPATCH_64(T, IntT, gpublas, C, Z, basename, __VA_ARGS__);
