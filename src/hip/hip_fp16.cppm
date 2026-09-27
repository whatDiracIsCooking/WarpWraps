/**
 * @file hip_fp16.cppm
 * @brief HIP Half-Precision Floating-Point API module wrapper
 *
 * Wraps hip/hip_fp16.h for C++23 module-based code, exporting the
 * half-precision types for host-side code. CUDA counterpart:
 * wwr.cuda.cuda_fp16.
 *
 * IMPORTANT -- __half here has NO arithmetic or comparison operators.
 * amd_hip_fp16.h selects its struct definition on __HIP__ vs __GNUC__, and a
 * plain C++23 module compile (clang++, no -x hip) never defines __HIP__ while
 * clang always predefines __GNUC__. So the compiled branch is the portable
 * hip_fp16_gcc.h fallback; the hidden-friend operators exist only under a real
 * -x hip device compile and are simply not present in the code this module
 * sees.
 *
 * The consequence is silent: `__half a, b; a + b;` compiles, but
 * decltype(a + b) is float -- addition goes through __half's operator float()
 * and drops back to full precision. This module exports the types as they are
 * and does not paper over the missing native arithmetic.
 *
 * <array> is pre-included and must stay first -- docs/architecture.md,
 * section 9.
 *
 * Usage:
 *   import wwr.hip.hip_fp16;
 */

module;

// Load-bearing, and must stay before the HIP header: host_defines.h poisons
// __noinline__ for libc++'s __config. docs/architecture.md, section 9.
#include <array>
#include <hip/hip_fp16.h>

export module wwr.hip.hip_fp16;

// ========================================================================
// Export all hip_fp16 types in wwr::hip
// (NOT bare wwr -- see src/hip/README.md "Design decisions": this is
// the real `half` collision the README calls out against wwr.cuda.cuda_fp16)
// ========================================================================

export namespace wwr::hip {

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

using ::half;  // Alias for __half
using ::half2; // Alias for __half2

// ========================================================================
// Raw Storage Types (without constructors/operators)
// ========================================================================

using ::__half2_raw;
using ::__half_raw;

// No operator exports: on the code path this module actually compiles
// against (see file header note above), __half/__half2 have no
// arithmetic/comparison operators of their own -- only an implicit
// `operator float()`/conversion from `__half2_raw`. There is nothing to
// `using`-declare or forward.

} // namespace wwr::hip
