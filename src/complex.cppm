/**
 * @file complex.cppm
 * @brief Backend-neutral complex types and arithmetic: gpu* names for
 *        cuComplex / hipComplex
 *
 * The host-module counterpart to complex.cuh: wwrFloatComplex /
 * wwrDoubleComplex / wwrComplex, make_gpu*Complex, the wwrC* arithmetic and
 * accessors, and the wwrComplexFloatToDouble / wwrComplexDoubleToFloat precision
 * conversions, whichever backend this build is configured for. See
 * backend.h. Companion to fp16.cppm and bf16.cppm.
 *
 * Usage:
 *   import wwr.complex;
 *
 *   wwrDoubleComplex z = make_gpuDoubleComplex(1.0, 2.0);
 *
 * The wrappers below duplicate complex.cuh's surface on purpose, the same way
 * fp16.cppm duplicates fp16.cuh: a host TU reaches this construction and
 * arithmetic by importing the module, a device TU reaches the same names by
 * including the header, and neither can use the other's -- complex.cuh is
 * device-only (device_guard.h) and a module cannot be #included into a kernel.
 * The two copies name the SAME vendor types, so a host-allocated buffer and a
 * kernel parameter agree.
 *
 * Construction and arithmetic go through gpu* functions rather than operators
 * because cuComplex is an operator-less float2 aggregate where hipComplex is a
 * class -- so `a * b` and brace-initialisation are not portable, and the
 * vendors' C-style functions are the only spelling that exists on both. Half
 * and bfloat16 diverge the other way and live in fp16.cppm / bf16.cppm. See
 * docs/architecture.md, section 3.
 *
 * Unlike fp16.cppm, this module includes no vendor header in its GMF: the raw
 * cuComplex / hip_complex modules already export host wrappers for these
 * functions (their static-inline vendor originals are wrapped there, section
 * 12), so the gpu* forwarders below reach them through the import via
 * WWR_SELECT, exactly as the types do.
 */

module;

#include "backend.h"

// Complex types: wwrX -> cuX / hipX
#define WWR_COMPLEX_TYPE(x) WWR_TYPE(wwr##x, cu##x, hip##x)

export module wwr.complex;

#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.cuComplex;
#else
import wwr.hip.hip_complex;
#endif

export namespace wwr {

// ========================================================================
// Types -- the same ones complex.cuh exposes
// ========================================================================

WWR_COMPLEX_TYPE(FloatComplex)
WWR_COMPLEX_TYPE(DoubleComplex)
WWR_COMPLEX_TYPE(Complex)

// ========================================================================
// Construction
//
// Forwarding functions, not WWR_FUNCTION reference bindings: the host wrappers
// duplicate complex.cuh's surface (see the file header) and route through the
// raw module's own host wrappers, which WWR_SELECT names.
// ========================================================================

/// @brief Build a single-precision complex value from its two components
inline wwrFloatComplex make_gpuFloatComplex(const float re, const float im) {
  return WWR_SELECT(make_cuFloatComplex, make_hipFloatComplex)(re, im);
}

/// @brief Build a double-precision complex value from its two components
inline wwrDoubleComplex make_gpuDoubleComplex(const double re, const double im) {
  return WWR_SELECT(make_cuDoubleComplex, make_hipDoubleComplex)(re, im);
}

/// @brief Build a single-precision complex value (the vendors' make_*Complex
///        alias for make_gpuFloatComplex -- wwrComplex is wwrFloatComplex)
inline wwrComplex make_gpuComplex(const float re, const float im) {
  return WWR_SELECT(make_cuComplex, make_hipComplex)(re, im);
}

// ========================================================================
// Arithmetic and accessors
//
// Through gpu* functions rather than operators, for the same reason
// construction is (docs/architecture.md §3): cuFloatComplex is a plain float2
// with no arithmetic operators, so `a * b` compiles under HIP -- whose
// hipComplex is a class that defines them -- and fails under CUDA with no
// operator match. The vendors' C-style functions exist on both and are the
// portable spelling; the gpu* names carry the divergent cu*/hip* spellings.
//
// The arguments are taken by value, as the vendors declare them --
// wwrFloatComplex is 8 bytes, wwrDoubleComplex 16.
// ========================================================================

/// @brief Real part of a single-precision complex value
inline float wwrCrealf(const wwrFloatComplex z) {
  return WWR_SELECT(cuCrealf, hipCrealf)(z);
}

/// @brief Imaginary part of a single-precision complex value
inline float wwrCimagf(const wwrFloatComplex z) {
  return WWR_SELECT(cuCimagf, hipCimagf)(z);
}

/// @brief Real part of a double-precision complex value
inline double wwrCreal(const wwrDoubleComplex z) {
  return WWR_SELECT(cuCreal, hipCreal)(z);
}

/// @brief Imaginary part of a double-precision complex value
inline double wwrCimag(const wwrDoubleComplex z) {
  return WWR_SELECT(cuCimag, hipCimag)(z);
}

/// @brief Magnitude (absolute value) of a single-precision complex value
inline float wwrCabsf(const wwrFloatComplex z) {
  return WWR_SELECT(cuCabsf, hipCabsf)(z);
}

/// @brief Magnitude (absolute value) of a double-precision complex value
inline double wwrCabs(const wwrDoubleComplex z) {
  return WWR_SELECT(cuCabs, hipCabs)(z);
}

/// @brief Complex conjugate of a single-precision complex value
inline wwrFloatComplex wwrConjf(const wwrFloatComplex z) {
  return WWR_SELECT(cuConjf, hipConjf)(z);
}

/// @brief Complex conjugate of a double-precision complex value
inline wwrDoubleComplex wwrConj(const wwrDoubleComplex z) {
  return WWR_SELECT(cuConj, hipConj)(z);
}

/// @brief Sum of two single-precision complex values
inline wwrFloatComplex wwrCaddf(const wwrFloatComplex a, const wwrFloatComplex b) {
  return WWR_SELECT(cuCaddf, hipCaddf)(a, b);
}

/// @brief Difference of two single-precision complex values
inline wwrFloatComplex wwrCsubf(const wwrFloatComplex a, const wwrFloatComplex b) {
  return WWR_SELECT(cuCsubf, hipCsubf)(a, b);
}

/// @brief Product of two single-precision complex values
inline wwrFloatComplex wwrCmulf(const wwrFloatComplex a, const wwrFloatComplex b) {
  return WWR_SELECT(cuCmulf, hipCmulf)(a, b);
}

/// @brief Quotient of two single-precision complex values
inline wwrFloatComplex wwrCdivf(const wwrFloatComplex a, const wwrFloatComplex b) {
  return WWR_SELECT(cuCdivf, hipCdivf)(a, b);
}

/// @brief Sum of two double-precision complex values
inline wwrDoubleComplex wwrCadd(const wwrDoubleComplex a, const wwrDoubleComplex b) {
  return WWR_SELECT(cuCadd, hipCadd)(a, b);
}

/// @brief Difference of two double-precision complex values
inline wwrDoubleComplex wwrCsub(const wwrDoubleComplex a, const wwrDoubleComplex b) {
  return WWR_SELECT(cuCsub, hipCsub)(a, b);
}

/// @brief Product of two double-precision complex values
inline wwrDoubleComplex wwrCmul(const wwrDoubleComplex a, const wwrDoubleComplex b) {
  return WWR_SELECT(cuCmul, hipCmul)(a, b);
}

/// @brief Quotient of two double-precision complex values
inline wwrDoubleComplex wwrCdiv(const wwrDoubleComplex a, const wwrDoubleComplex b) {
  return WWR_SELECT(cuCdiv, hipCdiv)(a, b);
}

// ========================================================================
// Precision conversion
//
// The other portable functions the vendors name with the bare Complex token
// (alongside make_gpuComplex above): widen a single-precision value to double,
// or narrow it back. HIP's Complex-named extras -- hipCsqabs*, hipCfma* -- have
// no cuComplex.h counterpart, so they are not part of the gpu* surface.
// ========================================================================

/// @brief Widen a single-precision complex value to double precision
inline wwrDoubleComplex wwrComplexFloatToDouble(const wwrFloatComplex z) {
  return WWR_SELECT(cuComplexFloatToDouble, hipComplexFloatToDouble)(z);
}

/// @brief Narrow a double-precision complex value to single precision
inline wwrFloatComplex wwrComplexDoubleToFloat(const wwrDoubleComplex z) {
  return WWR_SELECT(cuComplexDoubleToFloat, hipComplexDoubleToFloat)(z);
}

} // namespace wwr
