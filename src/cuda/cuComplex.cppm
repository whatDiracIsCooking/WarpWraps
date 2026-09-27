/**
 * @file cuComplex.cppm
 * @brief CUDA Complex Number API module wrapper
 *
 * This module wraps the cuComplex API for use in C++20/C++23 module-based code.
 * It exports types and host-side functions for complex number manipulation.
 *
 * Usage:
 *   import wwr.cuda.cuComplex;
 *
 * Note: This module exports host-side complex number functions. Device-side
 * functions are available in device code via the native <cuComplex.h> header.
 *
 * Note: cuComplex.h functions have internal linkage (static inline), so we
 * must provide wrapper functions to export them from the module.
 */

module;

#include <cuComplex.h>

export module wwr.cuda.cuComplex;

// ========================================================================
// Export all cuComplex types and functions in wwr namespace
// ========================================================================

export namespace wwr::cuda {

// ========================================================================
// Complex Number Types
// ========================================================================

using ::cuComplex;
using ::cuDoubleComplex;
using ::cuFloatComplex;

// ========================================================================
// Constructors - Create Complex Numbers
// ========================================================================

// Create single precision complex number from real and imaginary parts
cuFloatComplex make_cuFloatComplex(float r, float i) {
  return ::make_cuFloatComplex(r, i);
}

// Create double precision complex number from real and imaginary parts
cuDoubleComplex make_cuDoubleComplex(double r, double i) {
  return ::make_cuDoubleComplex(r, i);
}

// Alias for make_cuFloatComplex
cuComplex make_cuComplex(float r, float i) {
  return ::make_cuComplex(r, i);
}

// ========================================================================
// Component Access - Real and Imaginary Parts
// ========================================================================

// Get real part of single precision complex number
float cuCrealf(cuFloatComplex x) {
  return ::cuCrealf(x);
}

// Get imaginary part of single precision complex number
float cuCimagf(cuFloatComplex x) {
  return ::cuCimagf(x);
}

// Get real part of double precision complex number
double cuCreal(cuDoubleComplex x) {
  return ::cuCreal(x);
}

// Get imaginary part of double precision complex number
double cuCimag(cuDoubleComplex x) {
  return ::cuCimag(x);
}

// ========================================================================
// Arithmetic Operations - Single Precision
// ========================================================================

// Complex addition (single precision)
cuFloatComplex cuCaddf(cuFloatComplex x, cuFloatComplex y) {
  return ::cuCaddf(x, y);
}

// Complex subtraction (single precision)
cuFloatComplex cuCsubf(cuFloatComplex x, cuFloatComplex y) {
  return ::cuCsubf(x, y);
}

// Complex multiplication (single precision)
cuFloatComplex cuCmulf(cuFloatComplex x, cuFloatComplex y) {
  return ::cuCmulf(x, y);
}

// Complex division (single precision)
cuFloatComplex cuCdivf(cuFloatComplex x, cuFloatComplex y) {
  return ::cuCdivf(x, y);
}

// Complex absolute value / magnitude (single precision)
float cuCabsf(cuFloatComplex x) {
  return ::cuCabsf(x);
}

// Complex conjugate (single precision)
cuFloatComplex cuConjf(cuFloatComplex x) {
  return ::cuConjf(x);
}

// ========================================================================
// Arithmetic Operations - Double Precision
// ========================================================================

// Complex addition (double precision)
cuDoubleComplex cuCadd(cuDoubleComplex x, cuDoubleComplex y) {
  return ::cuCadd(x, y);
}

// Complex subtraction (double precision)
cuDoubleComplex cuCsub(cuDoubleComplex x, cuDoubleComplex y) {
  return ::cuCsub(x, y);
}

// Complex multiplication (double precision)
cuDoubleComplex cuCmul(cuDoubleComplex x, cuDoubleComplex y) {
  return ::cuCmul(x, y);
}

// Complex division (double precision)
cuDoubleComplex cuCdiv(cuDoubleComplex x, cuDoubleComplex y) {
  return ::cuCdiv(x, y);
}

// Complex absolute value / magnitude (double precision)
double cuCabs(cuDoubleComplex x) {
  return ::cuCabs(x);
}

// Complex conjugate (double precision)
cuDoubleComplex cuConj(cuDoubleComplex x) {
  return ::cuConj(x);
}

// ========================================================================
// Type Conversion
// ========================================================================

// Convert double precision complex to single precision
cuFloatComplex cuComplexDoubleToFloat(cuDoubleComplex x) {
  return ::cuComplexDoubleToFloat(x);
}

// Convert single precision complex to double precision
cuDoubleComplex cuComplexFloatToDouble(cuFloatComplex x) {
  return ::cuComplexFloatToDouble(x);
}

} // namespace wwr::cuda
