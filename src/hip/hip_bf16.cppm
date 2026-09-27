/**
 * @file hip_bf16.cppm
 * @brief HIP BFloat16 Floating-Point API module wrapper
 *
 * Wraps hip/hip_bf16.h for C++23 module-based code, exporting bfloat16 types
 * and their operators for host-side code. CUDA counterpart:
 * gpumod.cuda.cuda_bf16.
 *
 * HIP spells the types __hip_bfloat16 / __hip_bfloat162, distinct from CUDA's
 * __nv_bfloat16 / __nv_bfloat162, so there is no collision here the way
 * hip_fp16's `half` has one. Everything still exports into wwr::hip
 * regardless -- see src/hip/README.md.
 *
 * Unlike hip_fp16's hidden friends, amd_hip_bf16.h defines the arithmetic and
 * comparison operators as `static inline` free functions in the global
 * namespace. Internal linkage means they cannot be re-exported by `using`, so
 * this module provides thin forwarding operators, as cuda_bf16.cppm does. See
 * docs/architecture.md, section 12.
 *
 * <array> is pre-included and must stay first -- docs/architecture.md,
 * section 9.
 *
 * Usage:
 *   import gpumod.hip.hip_bf16;
 */

module;

// Load-bearing, and must stay before the HIP header: host_defines.h poisons
// __noinline__ for libc++'s __config. docs/architecture.md, section 9.
#include <array>
#include <hip/hip_bf16.h>

export module gpumod.hip.hip_bf16;

// ========================================================================
// Export all hip_bf16 types in wwr::hip
// ========================================================================

export namespace wwr::hip {

// ========================================================================
// Core BFloat16 Types
// ========================================================================

// HIP BFloat16 floating-point type
using ::__hip_bfloat16;

// Vector of two bfloat16 values (SIMD type)
using ::__hip_bfloat162;

// Raw storage types (without constructors/operators)
using ::__hip_bfloat162_raw;
using ::__hip_bfloat16_raw;

// ========================================================================
// Operators -- __hip_bfloat16
// ========================================================================
// Thin forwarding wrappers over the global-namespace static-inline free
// functions in <hip/amd_detail/amd_hip_bf16.h>, needed because those
// functions have internal linkage and cannot be re-exported via a `using`
// declaration (see file header note above).

__hip_bfloat16 operator+(const __hip_bfloat16 &lh, const __hip_bfloat16 &rh) {
  return ::operator+(lh, rh);
}
__hip_bfloat16 operator-(const __hip_bfloat16 &lh, const __hip_bfloat16 &rh) {
  return ::operator-(lh, rh);
}
__hip_bfloat16 operator*(const __hip_bfloat16 &lh, const __hip_bfloat16 &rh) {
  return ::operator*(lh, rh);
}
__hip_bfloat16 operator/(const __hip_bfloat16 &lh, const __hip_bfloat16 &rh) {
  return ::operator/(lh, rh);
}

__hip_bfloat16 &operator+=(__hip_bfloat16 &lh, const __hip_bfloat16 &rh) {
  return ::operator+=(lh, rh);
}
__hip_bfloat16 &operator-=(__hip_bfloat16 &lh, const __hip_bfloat16 &rh) {
  return ::operator-=(lh, rh);
}
__hip_bfloat16 &operator*=(__hip_bfloat16 &lh, const __hip_bfloat16 &rh) {
  return ::operator*=(lh, rh);
}
__hip_bfloat16 &operator/=(__hip_bfloat16 &lh, const __hip_bfloat16 &rh) {
  return ::operator/=(lh, rh);
}

__hip_bfloat16 &operator++(__hip_bfloat16 &h) {
  return ::operator++(h);
}
__hip_bfloat16 &operator--(__hip_bfloat16 &h) {
  return ::operator--(h);
}
__hip_bfloat16 operator++(__hip_bfloat16 &h, int ignored) {
  return ::operator++(h, ignored);
}
__hip_bfloat16 operator--(__hip_bfloat16 &h, int ignored) {
  return ::operator--(h, ignored);
}

__hip_bfloat16 operator+(const __hip_bfloat16 &h) {
  return ::operator+(h);
}
__hip_bfloat16 operator-(const __hip_bfloat16 &h) {
  return ::operator-(h);
}

bool operator==(const __hip_bfloat16 &lh, const __hip_bfloat16 &rh) {
  return ::operator==(lh, rh);
}
bool operator!=(const __hip_bfloat16 &lh, const __hip_bfloat16 &rh) {
  return ::operator!=(lh, rh);
}
bool operator<(const __hip_bfloat16 &lh, const __hip_bfloat16 &rh) {
  return ::operator<(lh, rh);
}
bool operator>(const __hip_bfloat16 &lh, const __hip_bfloat16 &rh) {
  return ::operator>(lh, rh);
}
bool operator<=(const __hip_bfloat16 &lh, const __hip_bfloat16 &rh) {
  return ::operator<=(lh, rh);
}
bool operator>=(const __hip_bfloat16 &lh, const __hip_bfloat16 &rh) {
  return ::operator>=(lh, rh);
}

// ========================================================================
// Operators -- __hip_bfloat162
// ========================================================================

__hip_bfloat162 operator+(const __hip_bfloat162 &lh, const __hip_bfloat162 &rh) {
  return ::operator+(lh, rh);
}
__hip_bfloat162 operator-(const __hip_bfloat162 &lh, const __hip_bfloat162 &rh) {
  return ::operator-(lh, rh);
}
__hip_bfloat162 operator*(const __hip_bfloat162 &lh, const __hip_bfloat162 &rh) {
  return ::operator*(lh, rh);
}
__hip_bfloat162 operator/(const __hip_bfloat162 &lh, const __hip_bfloat162 &rh) {
  return ::operator/(lh, rh);
}

__hip_bfloat162 &operator+=(__hip_bfloat162 &lh, const __hip_bfloat162 &rh) {
  return ::operator+=(lh, rh);
}
__hip_bfloat162 &operator-=(__hip_bfloat162 &lh, const __hip_bfloat162 &rh) {
  return ::operator-=(lh, rh);
}
__hip_bfloat162 &operator*=(__hip_bfloat162 &lh, const __hip_bfloat162 &rh) {
  return ::operator*=(lh, rh);
}
__hip_bfloat162 &operator/=(__hip_bfloat162 &lh, const __hip_bfloat162 &rh) {
  return ::operator/=(lh, rh);
}

__hip_bfloat162 &operator++(__hip_bfloat162 &h) {
  return ::operator++(h);
}
__hip_bfloat162 &operator--(__hip_bfloat162 &h) {
  return ::operator--(h);
}
__hip_bfloat162 operator++(__hip_bfloat162 &h, int ignored) {
  return ::operator++(h, ignored);
}
__hip_bfloat162 operator--(__hip_bfloat162 &h, int ignored) {
  return ::operator--(h, ignored);
}

__hip_bfloat162 operator+(const __hip_bfloat162 &h) {
  return ::operator+(h);
}
__hip_bfloat162 operator-(const __hip_bfloat162 &h) {
  return ::operator-(h);
}

bool operator==(const __hip_bfloat162 &lh, const __hip_bfloat162 &rh) {
  return ::operator==(lh, rh);
}
bool operator!=(const __hip_bfloat162 &lh, const __hip_bfloat162 &rh) {
  return ::operator!=(lh, rh);
}
bool operator<(const __hip_bfloat162 &lh, const __hip_bfloat162 &rh) {
  return ::operator<(lh, rh);
}
bool operator<=(const __hip_bfloat162 &lh, const __hip_bfloat162 &rh) {
  return ::operator<=(lh, rh);
}
bool operator>(const __hip_bfloat162 &lh, const __hip_bfloat162 &rh) {
  return ::operator>(lh, rh);
}
bool operator>=(const __hip_bfloat162 &lh, const __hip_bfloat162 &rh) {
  return ::operator>=(lh, rh);
}

} // namespace wwr::hip
