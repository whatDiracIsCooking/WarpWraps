/**
 * @file dispatch_macros.h
 * @brief Preprocessor macros for dispatching GPU sparse function calls based on type
 *
 * The prefix-agnostic WWR_REAL_DISPATCH / WWR_COMPLEX_DISPATCH cores live in
 * wrappers/common/dispatch_sdcz.h; this header binds them to the gpusparse* family.
 * Unlike BLAS there is no _64 index variant: the legacy typed cuSPARSE/hipSPARSE
 * functions wrapped here take a single index width, so a wrapper is templated on
 * the element type T alone.
 *
 * Prerequisites (must be provided by including file):
 * - std::is_same_v (via std import or equivalent)
 * - gpuComplex, gpuDoubleComplex and the gpusparse* functions
 *   (gpumod.complex, gpumod.sparse)
 */

#pragma once

#include "wrappers/common/dispatch_sdcz.h"

/// @brief Macro for dispatching to usual types (float, double, gpuComplex, gpuDoubleComplex).
#define WWR_USUAL_DISPATCH(T, basename, ...)                                                    \
  WWR_REAL_DISPATCH(T, gpusparse, S, D, basename, __VA_ARGS__);                                 \
  WWR_COMPLEX_DISPATCH(T, gpusparse, C, Z, basename, __VA_ARGS__);
