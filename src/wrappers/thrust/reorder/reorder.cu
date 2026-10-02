// reorder.cu
//
// The device half of wwr.wrappers.thrust.reorder: the portable reorder family
// (sort, unique, partition, remove, copy_if incl. the stencil form, reverse),
// each dispatched to Thrust/rocThrust through the F2 stream-bound policy so the
// work lands on the caller's stream. A device TU #includes the shim and calls
// thrust::<algo>(wwr::par_on(stream), ...) -- it cannot `import` (§8). Shared
// unchanged between both backends: README.md's audit confirms every in-scope
// policy-first overload is byte-identical across CCCL 3.0.1 and rocThrust 2.8.5,
// so there is no per-backend #if here.
//
// execution_policy.cuh first: it pulls runtime.h, which defines wwrStream_t for
// the bridge declarations that follow.
#include "wrappers/thrust/execution_policy.cuh"

#include "wrappers/thrust/reorder/reorder_bridge.h"

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/partition.h>
#include <thrust/remove.h>
#include <thrust/reverse.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <cstddef>
#include <cstdint>

namespace wwr::reorder::device {

namespace {

/// @brief The predicate the compaction/partition algorithms test with: x != 0
///
/// A fixed, portable predicate keeps the host/device boundary typed -- a host
/// lambda cannot cross into a device TU. Nonzero is the natural "mask" meaning
/// for numeric device data (and exactly what the stencil form tests on each
/// stencil element). It compiles for every instantiated type.
template<typename T>
struct is_nonzero {
  __host__ __device__ bool operator()(const T x) const { return x != T{0}; }
};

} // namespace

template<typename T>
void sort(const wwrStream_t stream, T *d, const std::size_t n) {
  if (n < 1) {
    return;
  }
  const auto first = ::thrust::device_pointer_cast(d);
  ::thrust::sort(wwr::par_on(stream), first, first + n);
}

template<typename T>
std::size_t unique(const wwrStream_t stream, T *d, const std::size_t n) {
  if (n < 1) {
    return 0;
  }
  const auto first = ::thrust::device_pointer_cast(d);
  const auto last = ::thrust::unique(wwr::par_on(stream), first, first + n);
  return static_cast<std::size_t>(last - first);
}

template<typename T>
std::size_t partition(const wwrStream_t stream, T *d, const std::size_t n) {
  if (n < 1) {
    return 0;
  }
  const auto first = ::thrust::device_pointer_cast(d);
  const auto point = ::thrust::partition(wwr::par_on(stream), first, first + n, is_nonzero<T>{});
  return static_cast<std::size_t>(point - first);
}

template<typename T>
std::size_t remove(const wwrStream_t stream, T *d, const std::size_t n, const T value) {
  if (n < 1) {
    return 0;
  }
  const auto first = ::thrust::device_pointer_cast(d);
  const auto last = ::thrust::remove(wwr::par_on(stream), first, first + n, value);
  return static_cast<std::size_t>(last - first);
}

template<typename T>
std::size_t copy_if(const wwrStream_t stream, const T *src, const std::size_t n, T *dst) {
  if (n < 1) {
    return 0;
  }
  const auto in = ::thrust::device_pointer_cast(src);
  const auto out = ::thrust::device_pointer_cast(dst);
  const auto end = ::thrust::copy_if(wwr::par_on(stream), in, in + n, out, is_nonzero<T>{});
  return static_cast<std::size_t>(end - out);
}

template<typename T>
std::size_t copy_if_stencil(const wwrStream_t stream, const T *src, const T *stencil,
                            const std::size_t n, T *dst) {
  if (n < 1) {
    return 0;
  }
  const auto in = ::thrust::device_pointer_cast(src);
  const auto mask = ::thrust::device_pointer_cast(stencil);
  const auto out = ::thrust::device_pointer_cast(dst);
  const auto end = ::thrust::copy_if(wwr::par_on(stream), in, in + n, mask, out, is_nonzero<T>{});
  return static_cast<std::size_t>(end - out);
}

template<typename T>
void reverse(const wwrStream_t stream, T *d, const std::size_t n) {
  if (n < 1) {
    return;
  }
  const auto first = ::thrust::device_pointer_cast(d);
  ::thrust::reverse(wwr::par_on(stream), first, first + n);
}

// One instantiation per supported type, matching reorder.cppm's extern template
// list and instantiations.cpp's. All three lists cover the same orderable-real
// set (sort needs operator<, so complex is excluded -- README.md's audit). A
// type added here without being added there links against nothing.
#define WWR_REORDER_INSTANTIATE(T)                                                                 \
  template void sort<T>(wwrStream_t, T *, std::size_t);                                             \
  template std::size_t unique<T>(wwrStream_t, T *, std::size_t);                                    \
  template std::size_t partition<T>(wwrStream_t, T *, std::size_t);                                 \
  template std::size_t remove<T>(wwrStream_t, T *, std::size_t, T);                                 \
  template std::size_t copy_if<T>(wwrStream_t, const T *, std::size_t, T *);                        \
  template std::size_t copy_if_stencil<T>(wwrStream_t, const T *, const T *, std::size_t, T *);     \
  template void reverse<T>(wwrStream_t, T *, std::size_t)

WWR_REORDER_INSTANTIATE(float);
WWR_REORDER_INSTANTIATE(double);
WWR_REORDER_INSTANTIATE(std::int32_t);
WWR_REORDER_INSTANTIATE(std::int64_t);
WWR_REORDER_INSTANTIATE(std::uint32_t);
WWR_REORDER_INSTANTIATE(std::uint64_t);

#undef WWR_REORDER_INSTANTIATE

} // namespace wwr::reorder::device
