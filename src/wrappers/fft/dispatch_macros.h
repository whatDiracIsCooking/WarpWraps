/**
 * @file dispatch_macros.h
 * @brief Preprocessor macro for dispatching GPU FFT execution by precision
 *
 * The six gpufftExec* entry points split by precision (single: C2C/R2C/C2R,
 * double: Z2Z/D2Z/Z2D) and by transform kind. A wrapper templated on the real
 * precision T (float or double) picks its entry point by token-pasting the
 * transform-kind suffix onto gpufftExec. Unlike the BLAS/solver dispatch there
 * is no _64 variant -- FFT sizes are passed as long long to the *Many64 plan
 * functions, not selected by an index-type suffix on the exec call.
 *
 * Prerequisites (must be provided by the including file):
 * - std::is_same_v (via `import std;` or equivalent)
 * - the gpufftExec* functions (wwr.fft)
 */

#pragma once

/// @brief Dispatch to gpufftExec<single> for float and gpufftExec<dbl> for
/// double, where <single>/<dbl> are transform-kind suffixes (e.g. C2C, Z2Z).
#define WWR_FFT_EXEC_DISPATCH(T, single, dbl, ...)                                              \
  if constexpr (std::is_same_v<T, float>) {                                                        \
    return gpufftExec##single(__VA_ARGS__);                                                        \
  } else if constexpr (std::is_same_v<T, double>) {                                                \
    return gpufftExec##dbl(__VA_ARGS__);                                                           \
  }
