/**
 * @file hip_complex.cppm
 * @brief HIP Complex Number API module wrapper for gpumod project
 *
 * Wraps hip/hip_complex.h for C++23 module-based code: types and host-side
 * functions for complex number manipulation. CUDA counterpart:
 * gpumod.cuda.cuComplex.
 *
 * Every function in amd_detail/amd_hip_complex.h is `static inline` in the
 * global namespace and so cannot be re-exported by a `using` declaration.
 * This module provides thin forwarding functions with the same names, calling
 * straight through to the ::-qualified original, exactly as cuComplex.cppm
 * does. See docs/architecture.md, section 12.
 *
 * <array> is pre-included and must stay first -- docs/architecture.md,
 * section 9.
 *
 * Usage:
 *   import gpumod.hip.hip_complex;
 */

module;

// Load-bearing, and must stay before the HIP header: host_defines.h poisons
// __noinline__ for libc++'s __config. docs/architecture.md, section 9.
#include <array>
#include <hip/hip_complex.h>

export module gpumod.hip.hip_complex;

// ========================================================================
// Export all hip_complex types and functions in gpumod::hip
// (NOT bare gpumod -- see src/hip/README.md "Design decisions")
// ========================================================================

export namespace gpumod::hip {

// ========================================================================
// Complex Number Types
// ========================================================================

using ::hipComplex;
using ::hipDoubleComplex;
using ::hipFloatComplex;

// ========================================================================
// Constructors - Create Complex Numbers
// ========================================================================

hipFloatComplex make_hipFloatComplex(float a, float b) {
  return ::make_hipFloatComplex(a, b);
}

hipDoubleComplex make_hipDoubleComplex(double a, double b) {
  return ::make_hipDoubleComplex(a, b);
}

// Alias for make_hipFloatComplex
hipComplex make_hipComplex(float x, float y) {
  return ::make_hipComplex(x, y);
}

// ========================================================================
// Component Access - Real and Imaginary Parts
// ========================================================================

float hipCrealf(hipFloatComplex z) {
  return ::hipCrealf(z);
}

float hipCimagf(hipFloatComplex z) {
  return ::hipCimagf(z);
}

double hipCreal(hipDoubleComplex z) {
  return ::hipCreal(z);
}

double hipCimag(hipDoubleComplex z) {
  return ::hipCimag(z);
}

// ========================================================================
// Arithmetic Operations - Single Precision
// ========================================================================

hipFloatComplex hipCaddf(hipFloatComplex p, hipFloatComplex q) {
  return ::hipCaddf(p, q);
}

hipFloatComplex hipCsubf(hipFloatComplex p, hipFloatComplex q) {
  return ::hipCsubf(p, q);
}

hipFloatComplex hipCmulf(hipFloatComplex p, hipFloatComplex q) {
  return ::hipCmulf(p, q);
}

hipFloatComplex hipCdivf(hipFloatComplex p, hipFloatComplex q) {
  return ::hipCdivf(p, q);
}

// Squared magnitude (single precision) -- HIP has no direct cuComplex.h
// counterpart for this one; kept since it is part of the public header.
float hipCsqabsf(hipFloatComplex z) {
  return ::hipCsqabsf(z);
}

float hipCabsf(hipFloatComplex z) {
  return ::hipCabsf(z);
}

hipFloatComplex hipConjf(hipFloatComplex z) {
  return ::hipConjf(z);
}

// ========================================================================
// Arithmetic Operations - Double Precision
// ========================================================================

hipDoubleComplex hipCadd(hipDoubleComplex p, hipDoubleComplex q) {
  return ::hipCadd(p, q);
}

hipDoubleComplex hipCsub(hipDoubleComplex p, hipDoubleComplex q) {
  return ::hipCsub(p, q);
}

hipDoubleComplex hipCmul(hipDoubleComplex p, hipDoubleComplex q) {
  return ::hipCmul(p, q);
}

hipDoubleComplex hipCdiv(hipDoubleComplex p, hipDoubleComplex q) {
  return ::hipCdiv(p, q);
}

// Squared magnitude (double precision) -- see hipCsqabsf note above.
double hipCsqabs(hipDoubleComplex z) {
  return ::hipCsqabs(z);
}

double hipCabs(hipDoubleComplex z) {
  return ::hipCabs(z);
}

hipDoubleComplex hipConj(hipDoubleComplex z) {
  return ::hipConj(z);
}

// ========================================================================
// Type Conversion
// ========================================================================

hipFloatComplex hipComplexDoubleToFloat(hipDoubleComplex z) {
  return ::hipComplexDoubleToFloat(z);
}

hipDoubleComplex hipComplexFloatToDouble(hipFloatComplex z) {
  return ::hipComplexFloatToDouble(z);
}

// ========================================================================
// Fused Multiply-Add
// ========================================================================
// HIP's amd_hip_complex.h has no cuComplex.h counterpart to these two, but
// they are part of the public header surface, so they are wrapped too.

hipComplex hipCfmaf(hipComplex p, hipComplex q, hipComplex r) {
  return ::hipCfmaf(p, q, r);
}

hipDoubleComplex hipCfma(hipDoubleComplex p, hipDoubleComplex q, hipDoubleComplex r) {
  return ::hipCfma(p, q, r);
}

} // namespace gpumod::hip
