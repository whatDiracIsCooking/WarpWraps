/**
 * @file reorder.cppm
 * @brief Primary interface for wwr.wrappers.thrust.reorder
 *
 * Portable, typed wrappers for the Thrust reorder family -- sort, unique,
 * partition, remove, copy_if (incl. the stencil form) and reverse -- over raw
 * device pointers. Each forwards to a device-compiled entry (reorder.cu) that
 * dispatches to Thrust/rocThrust through the F2 stream-bound policy, so the work
 * is enqueued on the caller's stream. README.md's audit confirms every in-scope
 * policy-first overload is identical across CCCL 3.0.1 and rocThrust 2.8.5, so
 * no backend guard is needed.
 *
 * The wrappers are unconstrained templates; the supported types are exactly the
 * ones explicitly instantiated below -- the orderable reals (float, double, and
 * the 32/64-bit signed and unsigned integers). sort/unique/partition need an
 * ordering or equality, which complex does not have, so complex is out of scope
 * for this family (README.md's audit). Any other type fails to link.
 *
 * The compaction wrappers (partition, copy_if, copy_if_stencil) test elements
 * for nonzero: a fixed predicate, because a host functor cannot cross into the
 * device TU. remove drops a given value. unique drops only CONSECUTIVE
 * duplicates, so sort first for a global dedup.
 *
 * Usage:
 *   import wwr.wrappers.thrust.reorder;
 *   using namespace wwr::reorder;
 *
 *   sort(stream, d_keys, n);                 // ascending, in place
 *   const std::size_t kept = unique(stream, d_keys, n);
 */

module;

#include "wrappers/thrust/reorder/reorder_bridge.h"

export module wwr.wrappers.thrust.reorder;

import std;

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so each template carries its own
// `export` and the declarations below sit in the plain namespace -- the shape
// wwr.extension.random_normal uses.
namespace wwr::reorder {

/// @brief Sort [d, d+n) ascending in place; work enqueued on @p stream
export template<typename T>
void sort(const wwrStream_t stream, T *d, const std::size_t n) {
  device::sort(stream, d, n); // device:: is load-bearing -- else this names itself
}

/// @brief Drop consecutive duplicates in place; sort first for a global dedup
/// @return the new logical length (kept elements, at the front)
export template<typename T>
std::size_t unique(const wwrStream_t stream, T *d, const std::size_t n) {
  return device::unique(stream, d, n);
}

/// @brief Move the nonzero elements before the zero ones
/// @return the partition point: the count of nonzero elements, now at the front
export template<typename T>
std::size_t partition(const wwrStream_t stream, T *d, const std::size_t n) {
  return device::partition(stream, d, n);
}

/// @brief Remove every element equal to @p value in place
/// @return the new logical length (kept elements, at the front)
export template<typename T>
std::size_t remove(const wwrStream_t stream, T *d, const std::size_t n, const T value) {
  return device::remove(stream, d, n, value);
}

/// @brief Copy the nonzero elements of [src, src+n) to dst
/// @return the number of elements written to dst
export template<typename T>
std::size_t copy_if(const wwrStream_t stream, const T *src, const std::size_t n, T *dst) {
  return device::copy_if(stream, src, n, dst);
}

/// @brief Copy src[i] where stencil[i] is nonzero to dst (the stencil form)
/// @return the number of elements written to dst
export template<typename T>
std::size_t copy_if_stencil(const wwrStream_t stream, const T *src, const T *stencil,
                            const std::size_t n, T *dst) {
  return device::copy_if_stencil(stream, src, stencil, n, dst);
}

/// @brief Reverse [d, d+n) in place; work enqueued on @p stream
export template<typename T>
void reverse(const wwrStream_t stream, T *d, const std::size_t n) {
  device::reverse(stream, d, n);
}

// Instantiated once in instantiations.cpp, not at every call site. The
// definitions these resolve to are in reorder.cu, compiled as device code.
#define WWR_REORDER_EXTERN(T)                                                                      \
  extern template void sort<T>(wwrStream_t, T *, std::size_t);                                     \
  extern template std::size_t unique<T>(wwrStream_t, T *, std::size_t);                            \
  extern template std::size_t partition<T>(wwrStream_t, T *, std::size_t);                         \
  extern template std::size_t remove<T>(wwrStream_t, T *, std::size_t, T);                         \
  extern template std::size_t copy_if<T>(wwrStream_t, const T *, std::size_t, T *);                \
  extern template std::size_t copy_if_stencil<T>(wwrStream_t, const T *, const T *, std::size_t,   \
                                                 T *);                                             \
  extern template void reverse<T>(wwrStream_t, T *, std::size_t)

WWR_REORDER_EXTERN(float);
WWR_REORDER_EXTERN(double);
WWR_REORDER_EXTERN(std::int32_t);
WWR_REORDER_EXTERN(std::int64_t);
WWR_REORDER_EXTERN(std::uint32_t);
WWR_REORDER_EXTERN(std::uint64_t);

#undef WWR_REORDER_EXTERN

} // namespace wwr::reorder
