/**
 * @file interface.cppm
 * @brief Primary interface for wwr.wrappers.thrust.reduce
 *
 * Portable, typed host wrappers for the Thrust reduction family -- reduce,
 * transform_reduce, count/count_if, inner_product and min_element/max_element --
 * each dispatched through the F2 stream-bound policy (wwr::par_on) so its work
 * lands on the caller's stream. The device calls themselves live in reduce.cu
 * (a device TU, since Thrust's headers assume a device pass and a device TU
 * cannot `import` -- docs/architecture.md §8); this module is ordinary host
 * C++23 that forwards to them.
 *
 * The forwarded functions are declared in reduce_bridge.h, included in the
 * global module fragment below so they keep external (not module) linkage and
 * bind to reduce.cu's explicit instantiations (§14). The matching
 * `extern template` declarations keep THIS translation unit from instantiating
 * any of them -- there is no Thrust in a host compile to instantiate against;
 * the one definition is the device TU's. Each supported element type gets one
 * `extern template` block, kept in lock-step with reduce.cu's instantiation
 * list the way wwr.wrappers.blas keeps its declarations and instantiations.cpp
 * in step.
 *
 * Type set: float, double, int32_t, int64_t -- see reduce.cu for why (native
 * device arithmetic + total order for every algorithm including the extrema;
 * no dependence on the unaudited thrust::complex). Complex is excluded from the
 * whole family.
 *
 * Usage:
 *   import wwr.wrappers.thrust.reduce;
 *   using namespace wwr::thrust;
 *
 *   auto total = reduce_sum(stream, d_values, n, 0.0f);
 *   auto n_nonzero = count_if_nonzero(stream, d_values, n);
 *   auto lo = min_element_value(stream, d_values, n);
 */

module;

#include "wrappers/thrust/reduce/reduce_bridge.h"

export module wwr.wrappers.thrust.reduce;

import std;
import wwr.runtime_api;

// This TU must not instantiate the family templates: the one definition is
// reduce.cu's (host code has no Thrust to compile them against). One block per
// supported type; keep it in step with reduce.cu's WWR_REDUCE_INSTANTIATE list.
namespace wwr::thrust::device {

#define WWR_REDUCE_EXTERN(T)                                                                       \
  extern template T reduce_sum<T>(wwrStream_t, const T *, std::size_t, T);                          \
  extern template T transform_reduce_square_sum<T>(wwrStream_t, const T *, std::size_t, T);         \
  extern template std::int64_t count<T>(wwrStream_t, const T *, std::size_t, T);                    \
  extern template std::int64_t count_if_nonzero<T>(wwrStream_t, const T *, std::size_t);            \
  extern template T inner_product<T>(wwrStream_t, const T *, const T *, std::size_t, T);            \
  extern template T min_element_value<T>(wwrStream_t, const T *, std::size_t);                      \
  extern template T max_element_value<T>(wwrStream_t, const T *, std::size_t)

WWR_REDUCE_EXTERN(float);
WWR_REDUCE_EXTERN(double);
WWR_REDUCE_EXTERN(std::int32_t);
WWR_REDUCE_EXTERN(std::int64_t);

#undef WWR_REDUCE_EXTERN

} // namespace wwr::thrust::device

export namespace wwr::thrust {

/// @brief Sum of `[first, first + n)` on the GPU, seeded with `init`
///
/// thrust::reduce with thrust::plus, run on `stream`. Returns `init` when
/// `n == 0`.
///
/// @param stream Stream the reduction is enqueued on
/// @param first  Device pointer to the first element
/// @param n      Number of elements
/// @param init   Value the sum is seeded with (identity is `T{}`)
/// @return The sum, copied back to the host
template<typename T>
T reduce_sum(const wwrStream_t stream, const T *first, const std::size_t n, const T init = T{}) {
  // device:: is load-bearing -- without it this names itself.
  return device::reduce_sum(stream, first, n, init);
}

/// @brief Sum of squares of `[first, first + n)` on the GPU, seeded with `init`
///
/// thrust::transform_reduce with a `x*x` unary op and thrust::plus, run on
/// `stream` -- the squared-L2-norm building block. Returns `init` when
/// `n == 0`.
template<typename T>
T transform_reduce_square_sum(const wwrStream_t stream, const T *first, const std::size_t n,
                              const T init = T{}) {
  return device::transform_reduce_square_sum(stream, first, n, init);
}

/// @brief Count of elements in `[first, first + n)` equal to `value`
///
/// thrust::count, run on `stream`. Returns 0 when `n == 0`.
template<typename T>
std::int64_t count(const wwrStream_t stream, const T *first, const std::size_t n, const T value) {
  return device::count(stream, first, n, value);
}

/// @brief Count of non-zero elements in `[first, first + n)`
///
/// thrust::count_if with a `!= T{}` predicate, run on `stream`. Returns 0 when
/// `n == 0`.
template<typename T>
std::int64_t count_if_nonzero(const wwrStream_t stream, const T *first, const std::size_t n) {
  return device::count_if_nonzero(stream, first, n);
}

/// @brief Dot product of two length-`n` device ranges, seeded with `init`
///
/// thrust::inner_product, run on `stream`. Returns `init` when `n == 0`.
template<typename T>
T inner_product(const wwrStream_t stream, const T *first1, const T *first2, const std::size_t n,
                const T init = T{}) {
  return device::inner_product(stream, first1, first2, n, init);
}

/// @brief Value of the smallest element in `[first, first + n)`
///
/// thrust::min_element (the iterator form, never the scalar thrust::min the
/// F2 audit bans), dereferenced on the device and copied back. `n` must be
/// `>= 1`; the minimum of an empty range is undefined.
template<typename T>
T min_element_value(const wwrStream_t stream, const T *first, const std::size_t n) {
  return device::min_element_value(stream, first, n);
}

/// @brief Value of the largest element in `[first, first + n)`
///
/// thrust::max_element, dereferenced on the device and copied back. `n` must
/// be `>= 1`; the maximum of an empty range is undefined.
template<typename T>
T max_element_value(const wwrStream_t stream, const T *first, const std::size_t n) {
  return device::max_element_value(stream, first, n);
}

} // namespace wwr::thrust
