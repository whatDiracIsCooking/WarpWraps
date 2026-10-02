/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of each reorder wrapper per type
 *
 * Implementation unit of wwr.wrappers.thrust.reorder. Pairs with the
 * `extern template` declarations in reorder.cppm: together they keep every
 * importer from instantiating the templates again at each call site.
 *
 * These instantiate the exported WRAPPERS. The device-side work they call
 * (wwr::reorder::device::*) is instantiated separately, in reorder.cu, because
 * that one has to be compiled as device code. Both lists cover the same
 * orderable-real set and have to stay in step -- a type added here without
 * being added there links against nothing.
 */

module wwr.wrappers.thrust.reorder;

// An implementation unit implicitly imports its primary interface, but neither
// an import nor a GMF #include is re-exported through it -- std::size_t / the
// fixed-width ints come from std, and wwrStream_t from wwr.runtime_api (the same
// type the interface's GMF runtime.h declares).
import std;
import wwr.runtime_api;

namespace wwr::reorder {

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

} // namespace wwr::reorder
