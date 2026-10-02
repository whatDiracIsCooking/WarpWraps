// reduce.cu
//
// The device half of wwr.wrappers.thrust.reduce: the reduction family's typed
// entry points, defined once against the portable thrust:: names and dispatched
// through wwr::par_on(stream) so each lands on the caller's stream. Shared
// unchanged between both backends -- .cu means "device pass, whichever
// backend" here, as in src/extension/init_state/init_state.cu, and the F2
// audit (src/wrappers/thrust/README.md) confirms every in-scope overload is
// byte-identical across CCCL Thrust 3.0.1 and rocThrust 2.8.5, so no per-backend
// #if is needed.
//
// execution_policy.cuh first: it pulls runtime.h (wwrStream_t for the bridge
// declarations below) and establishes the libc++/HIP header ordering the shim
// documents, before any other Thrust header is seen.
#include "wrappers/thrust/execution_policy.cuh"

#include "wrappers/thrust/reduce/reduce_bridge.h"

#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

#include <cstddef>
#include <cstdint>

namespace wwr::thrust::device {

namespace {

// A raw device pointer wrapped so Thrust treats it as device memory, as the F2
// stream_ops.cu does for sort. thrust::device_pointer_cast is the one portable
// spelling for this across both backends.
template<typename T>
auto as_device(const T *p) {
  return ::thrust::device_pointer_cast(p);
}

// x*x as a device-callable functor -- the transform_reduce unary op. Kept
// local (rather than thrust::square) so the family owns the one spelling it
// needs and depends on no functor whose availability differs across the two
// Thrust trees.
template<typename T>
struct square {
  __host__ __device__ T operator()(const T &x) const { return x * x; }
};

// "!= zero" as a device-callable functor rather than a lambda: a __device__
// lambda needs nvcc's --extended-lambda, which this build does not pass, so
// the family keeps its functors as named structs (and the meaning is explicit
// for every T, where thrust::identity would truthiness-test instead).
template<typename T>
struct is_nonzero {
  __host__ __device__ bool operator()(const T &x) const { return x != T{}; }
};

} // namespace

template<typename T>
T reduce_sum(const wwrStream_t stream, const T *first, const std::size_t n, const T init) {
  const auto begin = as_device(first);
  return ::thrust::reduce(wwr::par_on(stream), begin, begin + n, init, ::thrust::plus<T>{});
}

template<typename T>
T transform_reduce_square_sum(const wwrStream_t stream, const T *first, const std::size_t n,
                              const T init) {
  const auto begin = as_device(first);
  return ::thrust::transform_reduce(wwr::par_on(stream), begin, begin + n, square<T>{}, init,
                                    ::thrust::plus<T>{});
}

template<typename T>
std::int64_t count(const wwrStream_t stream, const T *first, const std::size_t n, const T value) {
  const auto begin = as_device(first);
  return static_cast<std::int64_t>(::thrust::count(wwr::par_on(stream), begin, begin + n, value));
}

template<typename T>
std::int64_t count_if_nonzero(const wwrStream_t stream, const T *first, const std::size_t n) {
  const auto begin = as_device(first);
  return static_cast<std::int64_t>(
      ::thrust::count_if(wwr::par_on(stream), begin, begin + n, is_nonzero<T>{}));
}

template<typename T>
T inner_product(const wwrStream_t stream, const T *first1, const T *first2, const std::size_t n,
                const T init) {
  const auto begin1 = as_device(first1);
  const auto begin2 = as_device(first2);
  return ::thrust::inner_product(wwr::par_on(stream), begin1, begin1 + n, begin2, init);
}

template<typename T>
T min_element_value(const wwrStream_t stream, const T *first, const std::size_t n) {
  const auto begin = as_device(first);
  // min_element (the iterator form), never the scalar thrust::min the audit
  // bans: CCCL 3.0 removed the scalar comparator from extrema.h, min_element is
  // portable in both trees. Dereferenced on the device side, then copied back.
  const auto it = ::thrust::min_element(wwr::par_on(stream), begin, begin + n);
  return *it;
}

template<typename T>
T max_element_value(const wwrStream_t stream, const T *first, const std::size_t n) {
  const auto begin = as_device(first);
  const auto it = ::thrust::max_element(wwr::par_on(stream), begin, begin + n);
  return *it;
}

// Explicit instantiations -- the typed surface the module links against.
//
// Type set: float, double, int32_t, int64_t. These are the element types that
// (a) have native device arithmetic and a total order, so every algorithm in
// the family -- including min_element/max_element, which NEED ordering -- is
// well defined, and (b) work with Thrust's default-constructed thrust::plus /
// thrust::less and require no thrust::complex, which the F2 audit explicitly
// did NOT verify (README.md "Scope boundary"). Complex is therefore excluded
// from the whole family: it is unordered (so the extrema are undefined on it)
// and its device sum would depend on the unaudited thrust::complex surface.
#define WWR_REDUCE_INSTANTIATE(T)                                                                  \
  template T reduce_sum<T>(wwrStream_t, const T *, std::size_t, T);                                 \
  template T transform_reduce_square_sum<T>(wwrStream_t, const T *, std::size_t, T);                \
  template std::int64_t count<T>(wwrStream_t, const T *, std::size_t, T);                           \
  template std::int64_t count_if_nonzero<T>(wwrStream_t, const T *, std::size_t);                   \
  template T inner_product<T>(wwrStream_t, const T *, const T *, std::size_t, T);                   \
  template T min_element_value<T>(wwrStream_t, const T *, std::size_t);                             \
  template T max_element_value<T>(wwrStream_t, const T *, std::size_t)

WWR_REDUCE_INSTANTIATE(float);
WWR_REDUCE_INSTANTIATE(double);
WWR_REDUCE_INSTANTIATE(std::int32_t);
WWR_REDUCE_INSTANTIATE(std::int64_t);

#undef WWR_REDUCE_INSTANTIATE

} // namespace wwr::thrust::device
