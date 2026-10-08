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
 *     __forceinline__ (construction and the real/imag accessors also constexpr),
 *     in a section gated behind the device-pass macros -- the device half that
 *     once lived in the separate complex.cuh, so a device .cu includes this one
 *     neutral header.
 *
 * Construction and the real/imag accessors build and read the value directly
 * (`wwrFloatComplex{re, im}`, `z.x`, `z.y`), needing no vendor function, so
 * complex.cppm spells them the same way. The HOST arithmetic wrappers, by
 * contrast, are NOT here and cannot be: the vendors' own cuC* functions are
 * static inline, and an exported inline in complex.cppm cannot expose a TU-local
 * static-inline function, so complex.cppm forwards to the raw cuComplex /
 * hip_complex module's external-linkage host wrappers via its import. That is why
 * complex, unlike rand, keeps importing its raw module. This is the `.h` + `.cppm`
 * shape for a vendor header with both host and device symbols; a purely
 * device-only wrapper stays a `.cuh`. See docs/architecture.md, section 3.
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
// host C++, does not -- it uses only the types above, builds and reads values the
// same direct way, and reaches the arithmetic through the raw module's
// external-linkage host wrappers, because the vendors' own arithmetic functions
// are static inline and an exported inline cannot expose them. This is the device
// half that once lived in the separate complex.cuh, folded in so a device
// consumer includes one neutral header. Link wwr.device.
//
// Construction and the real/imag accessors build and read the value directly
// (`wwrFloatComplex{re, im}`, `z.x`, `z.y`) and are constexpr: brace-init is a
// different operation on each backend (aggregate init of CUDA's plain-struct
// float2, a constexpr ctor call on HIP's HIP_vector_type class) but valid and
// constant-evaluable on both. The arithmetic goes through wwr* functions rather
// than operators because cuComplex defines none where hipComplex's are class
// members -- so `a * b` is not portable, and the vendors' C-style functions are
// the only spelling on both. Half and bfloat16 diverge the other way and live in
// fp16 / bf16. See docs/architecture.md, section 3.
// ========================================================================

#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)

namespace wwr {

// ------------------------------------------------------------------------
// Construction
// ------------------------------------------------------------------------

/// @brief Build a single-precision complex value from its two components
__device__ __forceinline__ constexpr wwrFloatComplex make_wwrFloatComplex(const float re,
                                                                          const float im) {
  return wwrFloatComplex{re, im};
}

/// @brief Build a double-precision complex value from its two components
__device__ __forceinline__ constexpr wwrDoubleComplex make_wwrDoubleComplex(const double re,
                                                                            const double im) {
  return wwrDoubleComplex{re, im};
}

/// @brief Build a single-precision complex value (the vendors' make_*Complex
///        alias for make_wwrFloatComplex -- wwrComplex is wwrFloatComplex)
__device__ __forceinline__ constexpr wwrComplex make_wwrComplex(const float re, const float im) {
  return wwrComplex{re, im};
}

// ------------------------------------------------------------------------
// Arithmetic and accessors -- taken by value, as the vendors declare them
// (wwrFloatComplex is 8 bytes, wwrDoubleComplex 16).
// ------------------------------------------------------------------------

/// @brief Real part of a single-precision complex value
__device__ __forceinline__ constexpr float wwrCrealf(const wwrFloatComplex z) {
  return z.x;
}

/// @brief Imaginary part of a single-precision complex value
__device__ __forceinline__ constexpr float wwrCimagf(const wwrFloatComplex z) {
  return z.y;
}

/// @brief Real part of a double-precision complex value
__device__ __forceinline__ constexpr double wwrCreal(const wwrDoubleComplex z) {
  return z.x;
}

/// @brief Imaginary part of a double-precision complex value
__device__ __forceinline__ constexpr double wwrCimag(const wwrDoubleComplex z) {
  return z.y;
}

// ROCm's hipCabs* and hipCdiv* square their argument unscaled, so they overflow
// (to inf/NaN) for |z| beyond ~sqrt(max) and lose range below ~sqrt(min), where
// cuComplex.h scales. The HIP branches below are scaled instead, so both
// backends cover the full exponent range.

/// @brief Magnitude (absolute value) of a single-precision complex value
__device__ __forceinline__ float wwrCabsf(const wwrFloatComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCabsf(z);
#else
  return ::hypotf(z.x, z.y);
#endif
}

/// @brief Magnitude (absolute value) of a double-precision complex value
__device__ __forceinline__ double wwrCabs(const wwrDoubleComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCabs(z);
#else
  return ::hypot(z.x, z.y);
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
  // cuCdivf's algorithm: scale both operands by 1 / (|re b| + |im b|).
  const float oos = 1.0f / (::fabsf(b.x) + ::fabsf(b.y));
  const float ars = a.x * oos;
  const float ais = a.y * oos;
  const float brs = b.x * oos;
  const float bis = b.y * oos;
  const float oon = 1.0f / ((brs * brs) + (bis * bis));
  return wwrFloatComplex{((ars * brs) + (ais * bis)) * oon,
                         ((ais * brs) - (ars * bis)) * oon};
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
  // cuCdiv's algorithm: scale both operands by 1 / (|re b| + |im b|).
  const double oos = 1.0 / (::fabs(b.x) + ::fabs(b.y));
  const double ars = a.x * oos;
  const double ais = a.y * oos;
  const double brs = b.x * oos;
  const double bis = b.y * oos;
  const double oon = 1.0 / ((brs * brs) + (bis * bis));
  return wwrDoubleComplex{((ars * brs) + (ais * bis)) * oon,
                          ((ais * brs) - (ars * bis)) * oon};
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
