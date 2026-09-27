/**
 * @file cuda_bf16.cppm
 * @brief CUDA BFloat16 Floating-Point API module wrapper
 *
 * This module wraps the cuda_bf16.h header for use in C++20/C++23 module-based code.
 * It exports bfloat16 types and their operators for host-side code.
 *
 * Usage:
 *   import gpumod.cuda.cuda_bf16;
 *
 * Note: BFloat16 is a brain floating-point format with the same exponent range as
 * IEEE 754 single-precision (8 bits) but reduced mantissa precision (7 bits vs 23 bits).
 * This format is particularly useful for deep learning applications.
 *
 * Note: cuda_bf16.h's arithmetic/comparison operators are free functions with
 * internal-to-the-header linkage in the global namespace. They are NOT
 * reachable via Argument-Dependent Lookup from a translation unit that only
 * imports this module (entities declared in a module's global module
 * fragment are not guaranteed reachable that way), so — like cuComplex.cppm —
 * this module provides thin forwarding operators to re-export them.
 */

module;

#include <cuda_bf16.h>

export module gpumod.cuda.cuda_bf16;

// ========================================================================
// Export all cuda_bf16 types in wwr namespace
// ========================================================================

export namespace wwr::cuda {

// ========================================================================
// Core BFloat16 Types
// ========================================================================

// NVIDIA BFloat16 floating-point type
using ::__nv_bfloat16;

// Vector of two bfloat16 values (SIMD type)
using ::__nv_bfloat162;

// ========================================================================
// Type Aliases
// ========================================================================

// Standard type aliases for bfloat16
using ::nv_bfloat16;  // Alias for __nv_bfloat16
using ::nv_bfloat162; // Alias for __nv_bfloat162

// Raw storage types (without constructors/operators)
using ::__nv_bfloat162_raw;
using ::__nv_bfloat16_raw;

// ========================================================================
// Operators — __nv_bfloat16
// ========================================================================
// Thin forwarding wrappers over the global-namespace free functions in
// <cuda_bf16.h>, needed because those functions are not reachable via ADL
// from a TU that only imports this module (see file header note above).

__nv_bfloat16 operator+(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) {
  return ::operator+(lh, rh);
}
__nv_bfloat16 operator-(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) {
  return ::operator-(lh, rh);
}
__nv_bfloat16 operator*(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) {
  return ::operator*(lh, rh);
}
__nv_bfloat16 operator/(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) {
  return ::operator/(lh, rh);
}

__nv_bfloat16 &operator+=(__nv_bfloat16 &lh, const __nv_bfloat16 &rh) {
  return ::operator+=(lh, rh);
}
__nv_bfloat16 &operator-=(__nv_bfloat16 &lh, const __nv_bfloat16 &rh) {
  return ::operator-=(lh, rh);
}
__nv_bfloat16 &operator*=(__nv_bfloat16 &lh, const __nv_bfloat16 &rh) {
  return ::operator*=(lh, rh);
}
__nv_bfloat16 &operator/=(__nv_bfloat16 &lh, const __nv_bfloat16 &rh) {
  return ::operator/=(lh, rh);
}

__nv_bfloat16 &operator++(__nv_bfloat16 &h) {
  return ::operator++(h);
}
__nv_bfloat16 &operator--(__nv_bfloat16 &h) {
  return ::operator--(h);
}
__nv_bfloat16 operator++(__nv_bfloat16 &h, int ignored) {
  return ::operator++(h, ignored);
}
__nv_bfloat16 operator--(__nv_bfloat16 &h, int ignored) {
  return ::operator--(h, ignored);
}

__nv_bfloat16 operator+(const __nv_bfloat16 &h) {
  return ::operator+(h);
}
__nv_bfloat16 operator-(const __nv_bfloat16 &h) {
  return ::operator-(h);
}

bool operator==(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) {
  return ::operator==(lh, rh);
}
bool operator!=(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) {
  return ::operator!=(lh, rh);
}
bool operator<(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) {
  return ::operator<(lh, rh);
}
bool operator>(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) {
  return ::operator>(lh, rh);
}
bool operator<=(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) {
  return ::operator<=(lh, rh);
}
bool operator>=(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) {
  return ::operator>=(lh, rh);
}

// ========================================================================
// Operators — __nv_bfloat162
// ========================================================================

__nv_bfloat162 operator+(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh) {
  return ::operator+(lh, rh);
}
__nv_bfloat162 operator-(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh) {
  return ::operator-(lh, rh);
}
__nv_bfloat162 operator*(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh) {
  return ::operator*(lh, rh);
}
__nv_bfloat162 operator/(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh) {
  return ::operator/(lh, rh);
}

__nv_bfloat162 &operator+=(__nv_bfloat162 &lh, const __nv_bfloat162 &rh) {
  return ::operator+=(lh, rh);
}
__nv_bfloat162 &operator-=(__nv_bfloat162 &lh, const __nv_bfloat162 &rh) {
  return ::operator-=(lh, rh);
}
__nv_bfloat162 &operator*=(__nv_bfloat162 &lh, const __nv_bfloat162 &rh) {
  return ::operator*=(lh, rh);
}
__nv_bfloat162 &operator/=(__nv_bfloat162 &lh, const __nv_bfloat162 &rh) {
  return ::operator/=(lh, rh);
}

__nv_bfloat162 &operator++(__nv_bfloat162 &h) {
  return ::operator++(h);
}
__nv_bfloat162 &operator--(__nv_bfloat162 &h) {
  return ::operator--(h);
}
__nv_bfloat162 operator++(__nv_bfloat162 &h, int ignored) {
  return ::operator++(h, ignored);
}
__nv_bfloat162 operator--(__nv_bfloat162 &h, int ignored) {
  return ::operator--(h, ignored);
}

__nv_bfloat162 operator+(const __nv_bfloat162 &h) {
  return ::operator+(h);
}
__nv_bfloat162 operator-(const __nv_bfloat162 &h) {
  return ::operator-(h);
}

bool operator==(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh) {
  return ::operator==(lh, rh);
}
bool operator!=(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh) {
  return ::operator!=(lh, rh);
}
bool operator<(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh) {
  return ::operator<(lh, rh);
}
bool operator>(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh) {
  return ::operator>(lh, rh);
}
bool operator<=(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh) {
  return ::operator<=(lh, rh);
}
bool operator>=(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh) {
  return ::operator>=(lh, rh);
}

} // namespace wwr::cuda
