/**
 * @file complex.cppm
 * @brief Backend-neutral complex types and arithmetic: wwr* names for
 *        cuComplex / hipComplex
 *
 * The host-module counterpart to complex.h's device-pass-gated section:
 * wwrFloatComplex / wwrDoubleComplex / wwrComplex, make_wwr*Complex, the wwrC* arithmetic and
 * accessors, and the wwrComplexFloatToDouble / wwrComplexDoubleToFloat precision
 * conversions, whichever backend this build is configured for. See
 * backend.h. Companion to fp16.cppm and bf16.cppm.
 *
 * Usage:
 *   import wwr.complex;
 *
 *   wwrDoubleComplex z = make_wwrDoubleComplex(1.0, 2.0);
 *
 * The wrappers below duplicate complex.h's device section on purpose, the same
 * way fp16.cppm duplicates fp16.h's device section: a host TU reaches this construction and
 * arithmetic by importing the module, a device TU reaches the same names by
 * including complex.h, and neither can use the other's -- the device wrappers are
 * gated to a device pass and a module cannot be #included into a kernel. The two
 * copies name the SAME vendor types, so a host-allocated buffer and a kernel
 * parameter agree.
 *
 * Construction and the real/imag accessors build and read the value directly --
 * `wwrFloatComplex{re, im}`, `z.x`, `z.y` -- so they are `constexpr`. Brace-init
 * is a different operation on each backend (aggregate init of CUDA's plain-struct
 * float2, a constexpr ctor call on HIP's HIP_vector_type class) but is valid and
 * constant-evaluable on both, so it needs no vendor function and no forwarding.
 *
 * Arithmetic, conjugate, magnitude and the precision conversions still go through
 * the vendors' C-style functions rather than operators: cuComplex defines no
 * operators (hipComplex's are members of its class type), so `a * b` is not
 * portable, and the vendor math -- notably cuCdiv's overflow-avoiding scaling --
 * is not worth re-deriving. Half and bfloat16 diverge the other way and live in
 * fp16.cppm / bf16.cppm. See docs/architecture.md, section 3.
 *
 * The GMF #includes complex.h for the types (the one definition it shares with a
 * device .cu that includes complex.h) and re-exports the three names below. The
 * forwarding functions are not shared: the vendor's own cuC* functions are
 * static inline and an exported inline cannot expose them, so they route through
 * the raw cuComplex / hip_complex module's external-linkage host wrappers (their
 * static-inline originals wrapped there, section 12) via WWR_SELECT -- reached
 * through the import, not the GMF header.
 */

module;

#include "backend.h"

// The types (wwrFloatComplex / wwrDoubleComplex / wwrComplex), from complex.h.
// They land in the global module here and are re-exported below; the forwarders'
// host wrappers still come from the import.
#include "complex.h"

export module wwr.complex;

#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.cuComplex;
#else
import wwr.hip.hip_complex;
#endif

export namespace wwr {

// ========================================================================
// Types -- re-exported from complex.h (the global-module aliases the GMF
// #include brought in), so importers of wwr.complex see them
// ========================================================================

using wwr::wwrFloatComplex;
using wwr::wwrDoubleComplex;
using wwr::wwrComplex;

// ========================================================================
// Construction
//
// Direct brace-init, not a forward: constexpr, and the same spelling the device
// section in complex.h uses (see the file header). No vendor function, so no
// WWR_SELECT and no raw-module host wrapper.
// ========================================================================

/// @brief Build a single-precision complex value from its two components
constexpr wwrFloatComplex make_wwrFloatComplex(const float re, const float im) {
  return wwrFloatComplex{re, im};
}

/// @brief Build a double-precision complex value from its two components
constexpr wwrDoubleComplex make_wwrDoubleComplex(const double re, const double im) {
  return wwrDoubleComplex{re, im};
}

/// @brief Build a single-precision complex value (the vendors' make_*Complex
///        alias for make_wwrFloatComplex -- wwrComplex is wwrFloatComplex)
constexpr wwrComplex make_wwrComplex(const float re, const float im) {
  return wwrComplex{re, im};
}

// ========================================================================
// Arithmetic and accessors
//
// The real/imag accessors read `.x`/`.y` directly and are constexpr. The rest
// forward through wwr* functions rather than operators (docs/architecture.md
// §3): cuFloatComplex is a plain float2 with no arithmetic operators, so `a * b`
// compiles under HIP -- whose hipComplex is a class that defines them -- and
// fails under CUDA with no operator match. The vendors' C-style functions exist
// on both and are the portable spelling; the wwr* names carry the divergent
// cu*/hip* spellings.
//
// The arguments are taken by value, as the vendors declare them --
// wwrFloatComplex is 8 bytes, wwrDoubleComplex 16.
// ========================================================================

/// @brief Real part of a single-precision complex value
constexpr float wwrCrealf(const wwrFloatComplex z) {
  return z.x;
}

/// @brief Imaginary part of a single-precision complex value
constexpr float wwrCimagf(const wwrFloatComplex z) {
  return z.y;
}

/// @brief Real part of a double-precision complex value
constexpr double wwrCreal(const wwrDoubleComplex z) {
  return z.x;
}

/// @brief Imaginary part of a double-precision complex value
constexpr double wwrCimag(const wwrDoubleComplex z) {
  return z.y;
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
// (alongside make_wwrComplex above): widen a single-precision value to double,
// or narrow it back. HIP's Complex-named extras -- hipCsqabs*, hipCfma* -- have
// no cuComplex.h counterpart, so they are not part of the wwr* surface.
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
