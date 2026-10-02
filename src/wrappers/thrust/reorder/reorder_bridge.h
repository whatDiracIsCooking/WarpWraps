/**
 * @file reorder_bridge.h
 * @brief Declarations shared between the reorder module's interface unit and its
 *        device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by reorder.cu
 * directly. The declarations must live in the GMF, not the module purview: a
 * purview name gets module linkage and can never bind to a definition compiled
 * in a plain TU, which is what reorder.cu is. See docs/architecture.md §14.
 *
 * The `_bridge` suffix marks a declaration carried across the host/device
 * boundary, so it must compile in BOTH modes -- the extension alone cannot say
 * it (a `.cuh` is device-pass-only). See src/README.md.
 *
 * wwrStream_t comes from the wwr* layer's include-only runtime.h rather than an
 * `import`, since a GMF cannot import; it is the SAME type wwr.runtime_api
 * exports, so the wrapper passes its argument straight through. Reading the
 * backend define runtime.h depends on is why this module links wwr::backend
 * PRIVATE -- see this directory's CMakeLists.txt.
 */

#pragma once

#include "runtime.h"

#include <cstddef>

namespace wwr::reorder::device {

/// @brief Sort [d, d+n) ascending in place via thrust::sort
template<typename T>
void sort(wwrStream_t stream, T *d, std::size_t n);

/// @brief Drop consecutive duplicates in place via thrust::unique
/// @return the new logical length (elements kept, at the front)
/// @pre the range is sorted, or "consecutive" is not "all"
template<typename T>
std::size_t unique(wwrStream_t stream, T *d, std::size_t n);

/// @brief Move the nonzero elements before the zero ones via thrust::partition
/// @return the partition point: the count of nonzero elements, now at the front
template<typename T>
std::size_t partition(wwrStream_t stream, T *d, std::size_t n);

/// @brief Remove every element equal to @p value in place via thrust::remove
/// @return the new logical length (elements kept, at the front)
template<typename T>
std::size_t remove(wwrStream_t stream, T *d, std::size_t n, T value);

/// @brief Copy the nonzero elements of [src, src+n) to dst via thrust::copy_if
/// @return the number of elements written to dst
template<typename T>
std::size_t copy_if(wwrStream_t stream, const T *src, std::size_t n, T *dst);

/// @brief Copy src[i] where stencil[i] is nonzero via thrust::copy_if (stencil form)
/// @return the number of elements written to dst
template<typename T>
std::size_t copy_if_stencil(wwrStream_t stream, const T *src, const T *stencil, std::size_t n,
                            T *dst);

/// @brief Reverse [d, d+n) in place via thrust::reverse
template<typename T>
void reverse(wwrStream_t stream, T *d, std::size_t n);

} // namespace wwr::reorder::device
