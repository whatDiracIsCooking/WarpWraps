/**
 * @file dispatch_macros.h
 * @brief Preprocessor macros for dispatching GPU sparse function calls based on type
 *
 * The prefix-agnostic WWR_REAL_DISPATCH / WWR_COMPLEX_DISPATCH cores live in
 * wrappers/common/dispatch_sdcz.h; this header binds them to the wwrsparse* family.
 * Unlike BLAS there is no _64 index variant: the legacy typed cuSPARSE/hipSPARSE
 * functions wrapped here take a single index width, so a wrapper is templated on
 * the element type T alone.
 *
 * Prerequisites (must be provided by including file):
 * - std::is_same_v (via std import or equivalent)
 * - wwrComplex, wwrDoubleComplex and the wwrsparse* functions
 *   (wwr.complex, wwr.sparse)
 */

#pragma once

#include "wrappers/common/dispatch_sdcz.h"

/// @brief Macro for dispatching to usual types (float, double, wwrComplex, wwrDoubleComplex).
#define WWR_USUAL_DISPATCH(T, basename, ...)                                                    \
  WWR_REAL_DISPATCH(T, wwrsparse, S, D, basename, __VA_ARGS__);                                 \
  WWR_COMPLEX_DISPATCH(T, wwrsparse, C, Z, basename, __VA_ARGS__);
