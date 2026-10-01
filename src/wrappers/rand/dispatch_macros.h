/**
 * @file dispatch_macros.h
 * @brief Preprocessor macro for dispatching GPU RNG host generation by precision
 *
 * Each of cuRAND / hipRAND's host generators splits by precision into a float
 * entry point and a *Double sibling (wwrrandGenerateUniform /
 * wwrrandGenerateUniformDouble, and so on). A wrapper templated on the real
 * precision T (float or double) picks its entry point by token-pasting the
 * precision-specific suffix onto wwrrandGenerate. Unlike the BLAS/solver
 * dispatch there is no _64 variant -- the count is a size_t argument, not an
 * index-type suffix on the call.
 *
 * Prerequisites (must be provided by the including file):
 * - std::is_same_v (via `import std;` or equivalent)
 * - the wwrrandGenerate* functions (wwr.rand)
 */

#pragma once

/// @brief Dispatch to wwrrandGenerate<single> for float and
/// wwrrandGenerate<dbl> for double, where <single>/<dbl> are the name suffixes
/// (e.g. Uniform, UniformDouble).
#define WWR_RAND_GENERATE_DISPATCH(T, single, dbl, ...)                                            \
  if constexpr (std::is_same_v<T, float>) {                                                         \
    return wwrrandGenerate##single(__VA_ARGS__);                                                    \
  } else if constexpr (std::is_same_v<T, double>) {                                                 \
    return wwrrandGenerate##dbl(__VA_ARGS__);                                                       \
  }
