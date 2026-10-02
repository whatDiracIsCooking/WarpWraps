/**
 * @file execution_policy.cuh
 * @brief Backend-neutral stream-bound Thrust execution policy, for device TUs
 *
 * The one genuine wwr*-layer divergence Thrust has: the stream-bound execution
 * policy is `thrust::cuda::par.on(stream)` on CUDA and `thrust::hip::par.on(stream)`
 * on HIP. `wwr::par_on(stream)` is the one-line shim over that -- it names the
 * backend's `par` and returns the policy its `.on(stream)` yields, so a wrapper
 * writes `thrust::<algo>(wwr::par_on(stream), ...)` once and the right backend
 * policy is selected from the compiler's own device-pass macro. Every in-scope
 * algorithm's execution-policy-FIRST overload is identical across both backends,
 * so the policy this returns is usable as that first argument throughout.
 *
 * `#include`d directly into a `.cu` (CUDA) or `-x hip` device-compiled (HIP)
 * translation unit, not a module: the Thrust headers assume a device pass and a
 * device TU cannot `import` (docs/architecture.md §8). A consumer links
 * `wwr::thrust` for the backend's Thrust include dirs and reaches the src/-root
 * headers below (device_guard.h, runtime.h) through the same src/ include root
 * `wwr::thrust` carries -- the spelling "thrust/execution_policy.cuh" is
 * resolved off that root.
 *
 * The policy type is a backend-specific implementation detail, so `par_on`
 * returns `auto`; callers name it only as an algorithm's first argument.
 */

#pragma once

// WWR_SELECTED_CUDA / WWR_SELECTED_HIP, and #errors outside a device pass: the
// Thrust system headers below are only meaningful in a device compile.
#include "device_guard.h"

// wwrStream_t -- the vendor stream handle par.on() takes. The always-on part of
// runtime.h carries just the type; kept before the Thrust headers for the same
// libc++/__config ordering reason math.cuh documents (a HIP header reached first
// poisons __noinline__).
#include "runtime.h"

// thrust::<backend>::par lives behind the backend's own execution_policy.h.
#if defined(WWR_SELECTED_CUDA)
#include <thrust/system/cuda/execution_policy.h>
#else

// Two rocThrust-vs-toolchain gaps this layer owns, both bridged for every
// device TU through the one header that pulls rocThrust. rocThrust 2.8.5 was
// written against an older libc++ and a pristine HIP runtime; neither holds in
// this toolchain, and the Thrust headers below hit both unless we restore what
// they expect FIRST:
//   - _VSTD: rocThrust's HIP path still spells the std namespace `_VSTD`, a
//     libc++ internal macro libc++ 20 removed. Define it back to `std`.
//   - hipStreamDefault: runtime.h (above) ran runtime_api.h's flag-macro
//     #undef-to-constexpr dance, so the name is now a `const unsigned int`, not
//     the `0x00` null-pointer macro rocPRIM uses as a `hipStream_t` default
//     argument (`hipStream_t s = hipStreamDefault`), which no longer converts.
//     Re-#define it to the null-pointer constant so those defaults compile; the
//     value is unchanged (the constexpr is still 0x00 for any wwr* use).
#ifndef _VSTD
#define _VSTD std
#endif
#ifdef hipStreamDefault
#undef hipStreamDefault
#endif
#define hipStreamDefault 0

#include <thrust/system/hip/execution_policy.h>
#endif

namespace wwr {

/// @brief The stream-bound Thrust execution policy for the selected backend
///
/// `thrust::cuda::par.on(stream)` on CUDA, `thrust::hip::par.on(stream)` on HIP
/// -- the one spelling that differs between the two. Pass the result as the
/// first argument to any in-scope `thrust::` algorithm to run it, and enqueue
/// its work, on `stream`.
///
/// @param stream GPU stream the algorithm's work is enqueued on
/// @return A backend-specific stream-bound execution policy (type is an
///         implementation detail; name it only in the algorithm call)
inline auto par_on(const wwrStream_t stream) {
#if defined(WWR_SELECTED_CUDA)
  return ::thrust::cuda::par.on(stream);
#else
  return ::thrust::hip::par.on(stream);
#endif
}

} // namespace wwr
