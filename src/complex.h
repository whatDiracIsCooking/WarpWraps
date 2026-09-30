/**
 * @file complex.h
 * @brief The complex types, plus the device wrappers, for the complex layer
 *
 * wwrFloatComplex / wwrDoubleComplex / wwrComplex, aliased to the vendor's
 * cuComplex / hipComplex for whichever backend was selected -- the one type
 * definition a host-allocated buffer and a kernel parameter must agree on. It
 * carries two things:
 *   - the type aliases (always), which complex.cppm re-exports from its global
 *     module fragment and any TU naming the type includes;
 *   - the make_wwr* / wwrC* / wwrComplex*To* wrappers as __device__
 *     __forceinline__, in a section gated behind the device-pass macros -- the
 *     device half that once lived in the separate complex.cuh, so a device .cu
 *     includes this one neutral header.
 *
 * The HOST wrappers are NOT here and cannot be: the vendors' own make_* / cuC*
 * functions are static inline, and an exported inline in complex.cppm cannot
 * expose a TU-local static-inline function, so complex.cppm forwards to the raw
 * cuComplex / hip_complex module's external-linkage host wrappers via its import.
 * That is why complex, unlike rand, keeps importing its raw module. This is the
 * `.h` + `.cppm` shape for a vendor header with both host and device symbols; a
 * purely device-only wrapper stays a `.cuh`. See docs/architecture.md, section 3.
 */

#pragma once

// WWR_SELECTED_CUDA / WWR_SELECTED_HIP, from the compiler's device macro in a
// device pass or from WWR_GPU_BACKEND_* in complex.cppm's host compile.
#include "selected_backend.h"

#if defined(WWR_SELECTED_CUDA)

#include <cuComplex.h>

#else

// Load-bearing, and must stay before the HIP header: host_defines.h (pulled in
// transitively by the HIP vendor headers) poisons __noinline__ for libc++'s
// __config, so a HIP header reached before <array> makes __config fail to
// compile. Same pre-include the src/hip global module fragments carry.
// docs/architecture.md, section 9.
#include <array>

#include <hip/hip_complex.h>

#endif

namespace wwr {

// ========================================================================
// Types -- the same ones complex.cppm exports and the device section below exposes
// ========================================================================

#if defined(WWR_SELECTED_CUDA)

using wwrFloatComplex = ::cuFloatComplex;
using wwrDoubleComplex = ::cuDoubleComplex;
using wwrComplex = ::cuComplex;

#else

using wwrFloatComplex = ::hipFloatComplex;
using wwrDoubleComplex = ::hipDoubleComplex;
using wwrComplex = ::hipComplex;

#endif

} // namespace wwr

// ========================================================================
// Device construction, arithmetic, and conversions -- present only in a
// device-compile pass
//
// The __device__ __forceinline__ wrappers, gated behind the compiler's own
// device-pass macros so the rest of this header still compiles in a host TU
// (where __device__ is not a keyword, so these must be ABSENT rather than
// #error). A .cu that #includes complex.h gets them; complex.cppm, compiled as
// host C++, does not -- it uses only the types above and reaches the same
// construction and arithmetic through the raw module's external-linkage host
// wrappers, because the vendors' own functions are static inline and an exported
// inline cannot expose them. This is the device half that once lived in the
// separate complex.cuh, folded in so a device consumer includes one neutral
// header. Link wwr.device.
//
// Construction and arithmetic go through wwr* functions rather than operators
// because cuComplex is an operator-less float2 aggregate where hipComplex is a
// class -- so `a * b` and brace-initialisation are not portable, and the
// vendors' C-style functions are the only spelling that exists on both. Half and
// bfloat16 diverge the other way and live in fp16 / bf16. See
// docs/architecture.md, section 3.
// ========================================================================

#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)

namespace wwr {

// ------------------------------------------------------------------------
// Construction
// ------------------------------------------------------------------------

/// @brief Build a single-precision complex value from its two components
__device__ __forceinline__ wwrFloatComplex make_wwrFloatComplex(const float re, const float im) {
#if defined(WWR_SELECTED_CUDA)
  return ::make_cuFloatComplex(re, im);
#else
  return ::make_hipFloatComplex(re, im);
#endif
}

/// @brief Build a double-precision complex value from its two components
__device__ __forceinline__ wwrDoubleComplex make_wwrDoubleComplex(const double re,
                                                                  const double im) {
#if defined(WWR_SELECTED_CUDA)
  return ::make_cuDoubleComplex(re, im);
#else
  return ::make_hipDoubleComplex(re, im);
#endif
}

/// @brief Build a single-precision complex value (the vendors' make_*Complex
///        alias for make_wwrFloatComplex -- wwrComplex is wwrFloatComplex)
__device__ __forceinline__ wwrComplex make_wwrComplex(const float re, const float im) {
#if defined(WWR_SELECTED_CUDA)
  return ::make_cuComplex(re, im);
#else
  return ::make_hipComplex(re, im);
#endif
}

// ------------------------------------------------------------------------
// Arithmetic and accessors -- taken by value, as the vendors declare them
// (wwrFloatComplex is 8 bytes, wwrDoubleComplex 16).
// ------------------------------------------------------------------------

/// @brief Real part of a single-precision complex value
__device__ __forceinline__ float wwrCrealf(const wwrFloatComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCrealf(z);
#else
  return ::hipCrealf(z);
#endif
}

/// @brief Imaginary part of a single-precision complex value
__device__ __forceinline__ float wwrCimagf(const wwrFloatComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCimagf(z);
#else
  return ::hipCimagf(z);
#endif
}

/// @brief Real part of a double-precision complex value
__device__ __forceinline__ double wwrCreal(const wwrDoubleComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCreal(z);
#else
  return ::hipCreal(z);
#endif
}

/// @brief Imaginary part of a double-precision complex value
__device__ __forceinline__ double wwrCimag(const wwrDoubleComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCimag(z);
#else
  return ::hipCimag(z);
#endif
}

/// @brief Magnitude (absolute value) of a single-precision complex value
__device__ __forceinline__ float wwrCabsf(const wwrFloatComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCabsf(z);
#else
  return ::hipCabsf(z);
#endif
}

/// @brief Magnitude (absolute value) of a double-precision complex value
__device__ __forceinline__ double wwrCabs(const wwrDoubleComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCabs(z);
#else
  return ::hipCabs(z);
#endif
}

/// @brief Complex conjugate of a single-precision complex value
__device__ __forceinline__ wwrFloatComplex wwrConjf(const wwrFloatComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuConjf(z);
#else
  return ::hipConjf(z);
#endif
}

/// @brief Complex conjugate of a double-precision complex value
__device__ __forceinline__ wwrDoubleComplex wwrConj(const wwrDoubleComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuConj(z);
#else
  return ::hipConj(z);
#endif
}

/// @brief Sum of two single-precision complex values
__device__ __forceinline__ wwrFloatComplex wwrCaddf(const wwrFloatComplex a,
                                                    const wwrFloatComplex b) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCaddf(a, b);
#else
  return ::hipCaddf(a, b);
#endif
}

/// @brief Difference of two single-precision complex values
__device__ __forceinline__ wwrFloatComplex wwrCsubf(const wwrFloatComplex a,
                                                    const wwrFloatComplex b) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCsubf(a, b);
#else
  return ::hipCsubf(a, b);
#endif
}

/// @brief Product of two single-precision complex values
__device__ __forceinline__ wwrFloatComplex wwrCmulf(const wwrFloatComplex a,
                                                    const wwrFloatComplex b) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCmulf(a, b);
#else
  return ::hipCmulf(a, b);
#endif
}

/// @brief Quotient of two single-precision complex values
__device__ __forceinline__ wwrFloatComplex wwrCdivf(const wwrFloatComplex a,
                                                    const wwrFloatComplex b) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCdivf(a, b);
#else
  return ::hipCdivf(a, b);
#endif
}

/// @brief Sum of two double-precision complex values
__device__ __forceinline__ wwrDoubleComplex wwrCadd(const wwrDoubleComplex a,
                                                    const wwrDoubleComplex b) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCadd(a, b);
#else
  return ::hipCadd(a, b);
#endif
}

/// @brief Difference of two double-precision complex values
__device__ __forceinline__ wwrDoubleComplex wwrCsub(const wwrDoubleComplex a,
                                                    const wwrDoubleComplex b) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCsub(a, b);
#else
  return ::hipCsub(a, b);
#endif
}

/// @brief Product of two double-precision complex values
__device__ __forceinline__ wwrDoubleComplex wwrCmul(const wwrDoubleComplex a,
                                                    const wwrDoubleComplex b) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCmul(a, b);
#else
  return ::hipCmul(a, b);
#endif
}

/// @brief Quotient of two double-precision complex values
__device__ __forceinline__ wwrDoubleComplex wwrCdiv(const wwrDoubleComplex a,
                                                    const wwrDoubleComplex b) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCdiv(a, b);
#else
  return ::hipCdiv(a, b);
#endif
}

// ------------------------------------------------------------------------
// Precision conversion -- widen a single-precision value to double, or narrow it
// back. HIP's Complex-named extras (hipCsqabs*, hipCfma*) have no cuComplex.h
// counterpart and are not part of the wwr* surface.
// ------------------------------------------------------------------------

/// @brief Widen a single-precision complex value to double precision
__device__ __forceinline__ wwrDoubleComplex wwrComplexFloatToDouble(const wwrFloatComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuComplexFloatToDouble(z);
#else
  return ::hipComplexFloatToDouble(z);
#endif
}

/// @brief Narrow a double-precision complex value to single precision
__device__ __forceinline__ wwrFloatComplex wwrComplexDoubleToFloat(const wwrDoubleComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuComplexDoubleToFloat(z);
#else
  return ::hipComplexDoubleToFloat(z);
#endif
}

} // namespace wwr

#endif // device-compile pass
