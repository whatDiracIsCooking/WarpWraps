// scan.cu
//
// The device half of wwr.wrappers.thrust.scan (issue #240, A-scan). A plain
// device TU -- it imports nothing (docs/architecture.md §8) -- that #includes
// the F2 execution-policy shim and routes every scan through
// wwr::par_on(stream), so the work lands on the caller's stream. Shared
// unchanged between both backends: .cu means "device pass, whichever backend",
// and the audit (src/wrappers/thrust/README.md) confirms every in-scope scan
// overload is byte-identical across CCCL Thrust 3.0.1 and rocThrust 2.8.5, so
// no per-backend #if is needed here.
//
// execution_policy.cuh first: it pulls runtime.h, which defines wwrStream_t for
// the bridge declarations that follow (and establishes the libc++/__config
// include ordering the Thrust headers below depend on -- see the shim's header).
#include "wrappers/thrust/execution_policy.cuh"

#include "wrappers/thrust/scan/scan_bridge.h"

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/transform_scan.h>

#include <cstddef>
#include <cstdint>

namespace wwr::thrust::device {

// The free functions are defined here and explicitly instantiated at the
// bottom; the host module (interface.cppm) declares them extern and binds to
// these definitions across the host/device link boundary (§14). Each wraps the
// raw device pointers with thrust::device_pointer_cast so Thrust treats them as
// device iterators, and passes wwr::par_on(stream) as the execution-policy-first
// argument the audit guarantees is portable.

template<typename T>
void inclusive_scan(const wwrStream_t stream, const T *in, const std::size_t n, T *out) {
  const auto first = ::thrust::device_pointer_cast(in);
  const auto result = ::thrust::device_pointer_cast(out);
  ::thrust::inclusive_scan(wwr::par_on(stream), first, first + n, result);
}

template<typename T>
void inclusive_scan_max(const wwrStream_t stream, const T *in, const std::size_t n, T *out) {
  const auto first = ::thrust::device_pointer_cast(in);
  const auto result = ::thrust::device_pointer_cast(out);
  // The acceptance's custom associative operator: a running maximum rather than
  // the default sum. thrust::maximum<T> is a function OBJECT (portable in both
  // trees), not the scalar thrust::max the audit flags as 2.8-only.
  ::thrust::inclusive_scan(wwr::par_on(stream), first, first + n, result,
                           ::thrust::maximum<T>{});
}

template<typename T>
void exclusive_scan(const wwrStream_t stream, const T *in, const std::size_t n, T *out,
                    const T init) {
  const auto first = ::thrust::device_pointer_cast(in);
  const auto result = ::thrust::device_pointer_cast(out);
  ::thrust::exclusive_scan(wwr::par_on(stream), first, first + n, result, init);
}

template<typename T>
void transform_inclusive_scan_negate(const wwrStream_t stream, const T *in, const std::size_t n,
                                     T *out) {
  const auto first = ::thrust::device_pointer_cast(in);
  const auto result = ::thrust::device_pointer_cast(out);
  // Unary op applied to each element before the inclusive sum: thrust::negate<T>
  // is the concrete representative the typed wrapper fixes (both the unary and
  // the binary op must be concrete functors so the device TU can instantiate).
  ::thrust::transform_inclusive_scan(wwr::par_on(stream), first, first + n, result,
                                     ::thrust::negate<T>{}, ::thrust::plus<T>{});
}

template<typename T>
void transform_exclusive_scan_negate(const wwrStream_t stream, const T *in, const std::size_t n,
                                     T *out, const T init) {
  const auto first = ::thrust::device_pointer_cast(in);
  const auto result = ::thrust::device_pointer_cast(out);
  ::thrust::transform_exclusive_scan(wwr::par_on(stream), first, first + n, result,
                                     ::thrust::negate<T>{}, init, ::thrust::plus<T>{});
}

// ========================================================================
// Explicit instantiations -- the portable type set for this family
// ========================================================================
//
// Arithmetic element types only (issue #240: "Type coverage: arithmetic
// types"). The wwr* complex types are deliberately excluded: cuComplex is an
// operator-less float2 aggregate on CUDA (src/complex.h), so thrust::plus over
// it does not compile there -- a complex scan would need a hand-written add
// functor, which is out of this issue's arithmetic scope. float/double/int/
// int64_t each satisfy thrust::plus, thrust::negate and (for _max) a sensible
// ordering, and are byte-identical across both backends.
#define WWR_SCAN_INSTANTIATE(T)                                                                    \
  template void inclusive_scan<T>(wwrStream_t, const T *, std::size_t, T *);                        \
  template void exclusive_scan<T>(wwrStream_t, const T *, std::size_t, T *, T);                     \
  template void transform_inclusive_scan_negate<T>(wwrStream_t, const T *, std::size_t, T *);       \
  template void transform_exclusive_scan_negate<T>(wwrStream_t, const T *, std::size_t, T *, T);    \
  template void inclusive_scan_max<T>(wwrStream_t, const T *, std::size_t, T *);

WWR_SCAN_INSTANTIATE(float)
WWR_SCAN_INSTANTIATE(double)
WWR_SCAN_INSTANTIATE(int)
WWR_SCAN_INSTANTIATE(std::int64_t)

#undef WWR_SCAN_INSTANTIATE

} // namespace wwr::thrust::device
