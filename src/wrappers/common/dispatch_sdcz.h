/**
 * @file dispatch_sdcz.h
 * @brief Type-based dispatch to S/D/C/Z-prefixed entry points, prefix-agnostic
 *
 * WWR_REAL_DISPATCH and WWR_COMPLEX_DISPATCH pick a typed vendor entry point by the C++
 * element type: float/double select the S/D real letters, gpuComplex/
 * gpuDoubleComplex the C/Z complex letters. The prefix and basename are
 * caller-supplied, so the same two macros serve any library whose typed API
 * spells its functions <prefix><letter><basename> -- gpublas* and gpusolverDn*
 * alike. Families that need more build on these in their own dispatch_macros.h
 * (BLAS's _64 index variants and WWR_USUAL_DISPATCH convenience); FFT pastes the
 * type letter as a suffix rather than the middle, so it does not use these.
 *
 * Prerequisites (must be provided by the including file):
 * - std::is_same_v (via `import std;` or equivalent)
 * - gpuComplex, gpuDoubleComplex (gpumod.complex) for WWR_COMPLEX_DISPATCH
 */

#pragma once

/// @brief Macro for dispatching to float and double.
#define WWR_REAL_DISPATCH(T, prefix, S, D, basename, ...)                                       \
  if constexpr (std::is_same_v<T, float>) {                                                        \
    return prefix##S##basename(__VA_ARGS__);                                                       \
  } else if constexpr (std::is_same_v<T, double>) {                                                \
    return prefix##D##basename(__VA_ARGS__);                                                       \
  }

/// @brief Macro for dispatching to gpuComplex and gpuDoubleComplex.
#define WWR_COMPLEX_DISPATCH(T, prefix, C, Z, basename, ...)                                    \
  if constexpr (std::is_same_v<T, gpuComplex>) {                                                   \
    return prefix##C##basename(__VA_ARGS__);                                                       \
  } else if constexpr (std::is_same_v<T, gpuDoubleComplex>) {                                      \
    return prefix##Z##basename(__VA_ARGS__);                                                       \
  }
