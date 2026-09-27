/**
 * @file cuda_fp16.cppm
 * @brief CUDA Half-Precision Floating-Point API module wrapper
 *
 * This module wraps the cuda_fp16.h header for use in C++20/C++23 module-based code.
 * It exports half-precision types and their operators for host-side code.
 *
 * Usage:
 *   import wwr.cuda.cuda_fp16;
 *
 * Note: This module exports types and host-side operators. The full set of
 * device-side intrinsics is available in device code via the native <cuda_fp16.h> header.
 *
 * Note: cuda_fp16.h's arithmetic/comparison operators are free functions with
 * internal-to-the-header linkage in the global namespace. They are NOT
 * reachable via Argument-Dependent Lookup from a translation unit that only
 * imports this module (entities declared in a module's global module
 * fragment are not guaranteed reachable that way), so — like cuComplex.cppm —
 * this module provides thin forwarding operators to re-export them.
 */

module;

#include <cuda_fp16.h>

export module wwr.cuda.cuda_fp16;

// ========================================================================
// Export all cuda_fp16 types in wwr namespace
// ========================================================================

export namespace wwr::cuda {

// ========================================================================
// Core Half-Precision Types
// ========================================================================

// IEEE 754 half-precision floating-point type
using ::__half;

// Vector of two half-precision values (SIMD type)
using ::__half2;

// ========================================================================
// Type Aliases
// ========================================================================

// Standard type aliases for half-precision
using ::half;  // Alias for __half
using ::half2; // Alias for __half2

// NVIDIA-prefixed aliases
using ::__nv_half;
using ::__nv_half2;
using ::nv_half;
using ::nv_half2;

// Raw storage types (without constructors/operators)
using ::__half2_raw;
using ::__half_raw;
using ::__nv_half2_raw;
using ::__nv_half_raw;

// ========================================================================
// Operators — __half
// ========================================================================
// Thin forwarding wrappers over the global-namespace free functions in
// <cuda_fp16.h>, needed because those functions are not reachable via ADL
// from a TU that only imports this module (see file header note above).

__half operator+(const __half &lh, const __half &rh) {
  return ::operator+(lh, rh);
}
__half operator-(const __half &lh, const __half &rh) {
  return ::operator-(lh, rh);
}
__half operator*(const __half &lh, const __half &rh) {
  return ::operator*(lh, rh);
}
__half operator/(const __half &lh, const __half &rh) {
  return ::operator/(lh, rh);
}

__half &operator+=(__half &lh, const __half &rh) {
  return ::operator+=(lh, rh);
}
__half &operator-=(__half &lh, const __half &rh) {
  return ::operator-=(lh, rh);
}
__half &operator*=(__half &lh, const __half &rh) {
  return ::operator*=(lh, rh);
}
__half &operator/=(__half &lh, const __half &rh) {
  return ::operator/=(lh, rh);
}

__half &operator++(__half &h) {
  return ::operator++(h);
}
__half &operator--(__half &h) {
  return ::operator--(h);
}
__half operator++(__half &h, int ignored) {
  return ::operator++(h, ignored);
}
__half operator--(__half &h, int ignored) {
  return ::operator--(h, ignored);
}

__half operator+(const __half &h) {
  return ::operator+(h);
}
__half operator-(const __half &h) {
  return ::operator-(h);
}

bool operator==(const __half &lh, const __half &rh) {
  return ::operator==(lh, rh);
}
bool operator!=(const __half &lh, const __half &rh) {
  return ::operator!=(lh, rh);
}
bool operator<(const __half &lh, const __half &rh) {
  return ::operator<(lh, rh);
}
bool operator>(const __half &lh, const __half &rh) {
  return ::operator>(lh, rh);
}
bool operator<=(const __half &lh, const __half &rh) {
  return ::operator<=(lh, rh);
}
bool operator>=(const __half &lh, const __half &rh) {
  return ::operator>=(lh, rh);
}

// ========================================================================
// Operators — __half2
// ========================================================================

__half2 operator+(const __half2 &lh, const __half2 &rh) {
  return ::operator+(lh, rh);
}
__half2 operator-(const __half2 &lh, const __half2 &rh) {
  return ::operator-(lh, rh);
}
__half2 operator*(const __half2 &lh, const __half2 &rh) {
  return ::operator*(lh, rh);
}
__half2 operator/(const __half2 &lh, const __half2 &rh) {
  return ::operator/(lh, rh);
}

__half2 &operator+=(__half2 &lh, const __half2 &rh) {
  return ::operator+=(lh, rh);
}
__half2 &operator-=(__half2 &lh, const __half2 &rh) {
  return ::operator-=(lh, rh);
}
__half2 &operator*=(__half2 &lh, const __half2 &rh) {
  return ::operator*=(lh, rh);
}
__half2 &operator/=(__half2 &lh, const __half2 &rh) {
  return ::operator/=(lh, rh);
}

__half2 &operator++(__half2 &h) {
  return ::operator++(h);
}
__half2 &operator--(__half2 &h) {
  return ::operator--(h);
}
__half2 operator++(__half2 &h, int ignored) {
  return ::operator++(h, ignored);
}
__half2 operator--(__half2 &h, int ignored) {
  return ::operator--(h, ignored);
}

__half2 operator+(const __half2 &h) {
  return ::operator+(h);
}
__half2 operator-(const __half2 &h) {
  return ::operator-(h);
}

bool operator==(const __half2 &lh, const __half2 &rh) {
  return ::operator==(lh, rh);
}
bool operator!=(const __half2 &lh, const __half2 &rh) {
  return ::operator!=(lh, rh);
}
bool operator<(const __half2 &lh, const __half2 &rh) {
  return ::operator<(lh, rh);
}
bool operator>(const __half2 &lh, const __half2 &rh) {
  return ::operator>(lh, rh);
}
bool operator<=(const __half2 &lh, const __half2 &rh) {
  return ::operator<=(lh, rh);
}
bool operator>=(const __half2 &lh, const __half2 &rh) {
  return ::operator>=(lh, rh);
}

} // namespace wwr::cuda
