// comp.cppm - Compile-time tests for wwr.comp
//
// wwr.comp is the one wwr* module whose functions are hand-written forwarding
// shims rather than WWR_FUNCTION aliases (nvCOMP 5.3 and hipCOMP 2.2 have
// divergent batched signatures -- see comp.cppm and issue #110), so there is no
// WWR_SAME_FUNCTION to restate: the shims are proved by the device round-trip in
// test/extension/comp, and their neutral signatures are pinned here.
//
// What this file locks:
//   - the three type aliases resolve to the selected backend's type;
//   - every exported enumerator has the backend's type/value (WWR_SAME_VALUE)
//     AND a fixed integer (static_assert) -- matching names never guarantee
//     matching values (cf. WWRRAND_RNG_*), and here they agree, so pin it;
//   - each algorithm's compress/decompress shim has the expected signature.

module;

#include "gpu_check_macros.h"

export module wwr.test.gpu.comp;

import std;
import wwr.comp;
#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.nvcomp;
#else
import wwr.hip.hipcomp;
#endif

namespace wwr::test {

using namespace wwr;
#if defined(WWR_GPU_BACKEND_CUDA)
using namespace wwr::cuda;
#else
using namespace wwr::hip;
#endif

// ────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────

#if defined(WWR_GPU_BACKEND_CUDA)
WWR_SAME_TYPE(wwrcompStatus_t, nvcompStatus_t)
WWR_SAME_TYPE(wwrcompType_t, nvcompType_t)
WWR_SAME_TYPE(wwrcompStream_t, cudaStream_t)
#else
WWR_SAME_TYPE(wwrcompStatus_t, hipcompStatus_t)
WWR_SAME_TYPE(wwrcompType_t, hipcompType_t)
WWR_SAME_TYPE(wwrcompStream_t, hipStream_t)
#endif

// ────────────────────────────────────────────────────────────────────────
// Status codes -- backend identity + fixed integer
// ────────────────────────────────────────────────────────────────────────

#if defined(WWR_GPU_BACKEND_CUDA)
WWR_SAME_VALUE(WWRCOMP_SUCCESS, nvcompSuccess)
WWR_SAME_VALUE(WWRCOMP_ERROR_INVALID_VALUE, nvcompErrorInvalidValue)
WWR_SAME_VALUE(WWRCOMP_ERROR_NOT_SUPPORTED, nvcompErrorNotSupported)
WWR_SAME_VALUE(WWRCOMP_ERROR_CANNOT_DECOMPRESS, nvcompErrorCannotDecompress)
WWR_SAME_VALUE(WWRCOMP_ERROR_CUDA_ERROR, nvcompErrorCudaError)
WWR_SAME_VALUE(WWRCOMP_ERROR_INTERNAL, nvcompErrorInternal)
#else
WWR_SAME_VALUE(WWRCOMP_SUCCESS, hipcompSuccess)
WWR_SAME_VALUE(WWRCOMP_ERROR_INVALID_VALUE, hipcompErrorInvalidValue)
WWR_SAME_VALUE(WWRCOMP_ERROR_NOT_SUPPORTED, hipcompErrorNotSupported)
WWR_SAME_VALUE(WWRCOMP_ERROR_CANNOT_DECOMPRESS, hipcompErrorCannotDecompress)
WWR_SAME_VALUE(WWRCOMP_ERROR_CUDA_ERROR, hipcompErrorCudaError)
WWR_SAME_VALUE(WWRCOMP_ERROR_INTERNAL, hipcompErrorInternal)
#endif

static_assert(static_cast<int>(WWRCOMP_SUCCESS) == 0);
static_assert(static_cast<int>(WWRCOMP_ERROR_INVALID_VALUE) == 10);
static_assert(static_cast<int>(WWRCOMP_ERROR_NOT_SUPPORTED) == 11);
static_assert(static_cast<int>(WWRCOMP_ERROR_CANNOT_DECOMPRESS) == 12);
static_assert(static_cast<int>(WWRCOMP_ERROR_CUDA_ERROR) == 1000);
static_assert(static_cast<int>(WWRCOMP_ERROR_INTERNAL) == 10000);

// ────────────────────────────────────────────────────────────────────────
// Data types -- backend identity + fixed integer
// ────────────────────────────────────────────────────────────────────────

#if defined(WWR_GPU_BACKEND_CUDA)
WWR_SAME_VALUE(WWRCOMP_TYPE_CHAR, NVCOMP_TYPE_CHAR)
WWR_SAME_VALUE(WWRCOMP_TYPE_UCHAR, NVCOMP_TYPE_UCHAR)
WWR_SAME_VALUE(WWRCOMP_TYPE_SHORT, NVCOMP_TYPE_SHORT)
WWR_SAME_VALUE(WWRCOMP_TYPE_USHORT, NVCOMP_TYPE_USHORT)
WWR_SAME_VALUE(WWRCOMP_TYPE_INT, NVCOMP_TYPE_INT)
WWR_SAME_VALUE(WWRCOMP_TYPE_UINT, NVCOMP_TYPE_UINT)
WWR_SAME_VALUE(WWRCOMP_TYPE_LONGLONG, NVCOMP_TYPE_LONGLONG)
WWR_SAME_VALUE(WWRCOMP_TYPE_ULONGLONG, NVCOMP_TYPE_ULONGLONG)
WWR_SAME_VALUE(WWRCOMP_TYPE_BITS, NVCOMP_TYPE_BITS)
#else
WWR_SAME_VALUE(WWRCOMP_TYPE_CHAR, HIPCOMP_TYPE_CHAR)
WWR_SAME_VALUE(WWRCOMP_TYPE_UCHAR, HIPCOMP_TYPE_UCHAR)
WWR_SAME_VALUE(WWRCOMP_TYPE_SHORT, HIPCOMP_TYPE_SHORT)
WWR_SAME_VALUE(WWRCOMP_TYPE_USHORT, HIPCOMP_TYPE_USHORT)
WWR_SAME_VALUE(WWRCOMP_TYPE_INT, HIPCOMP_TYPE_INT)
WWR_SAME_VALUE(WWRCOMP_TYPE_UINT, HIPCOMP_TYPE_UINT)
WWR_SAME_VALUE(WWRCOMP_TYPE_LONGLONG, HIPCOMP_TYPE_LONGLONG)
WWR_SAME_VALUE(WWRCOMP_TYPE_ULONGLONG, HIPCOMP_TYPE_ULONGLONG)
WWR_SAME_VALUE(WWRCOMP_TYPE_BITS, HIPCOMP_TYPE_BITS)
#endif

static_assert(static_cast<int>(WWRCOMP_TYPE_CHAR) == 0);
static_assert(static_cast<int>(WWRCOMP_TYPE_UCHAR) == 1);
static_assert(static_cast<int>(WWRCOMP_TYPE_SHORT) == 2);
static_assert(static_cast<int>(WWRCOMP_TYPE_USHORT) == 3);
static_assert(static_cast<int>(WWRCOMP_TYPE_INT) == 4);
static_assert(static_cast<int>(WWRCOMP_TYPE_UINT) == 5);
static_assert(static_cast<int>(WWRCOMP_TYPE_LONGLONG) == 6);
static_assert(static_cast<int>(WWRCOMP_TYPE_ULONGLONG) == 7);
static_assert(static_cast<int>(WWRCOMP_TYPE_BITS) == 0xff);

// ────────────────────────────────────────────────────────────────────────
// Shim signatures -- one compress + one decompress per algorithm, locking the
// neutral (hipCOMP-2.2-shaped) parameter list the round-trip test relies on.
// ────────────────────────────────────────────────────────────────────────

template <class Opts>
using compress_async_t = wwrcompStatus_t (*)(
    const void *const *, const std::size_t *, std::size_t, std::size_t, void *, std::size_t,
    void *const *, std::size_t *, Opts, wwrcompStream_t);

using decompress_async_t = wwrcompStatus_t (*)(
    const void *const *, const std::size_t *, const std::size_t *, std::size_t *, std::size_t,
    void *, std::size_t, void *const *, wwrcompStatus_t *, wwrcompStream_t);

static_assert(std::is_same_v<decltype(&wwrcompBatchedLZ4CompressAsync),
                             compress_async_t<wwrcompBatchedLZ4Opts>>);
static_assert(std::is_same_v<decltype(&wwrcompBatchedSnappyCompressAsync),
                             compress_async_t<wwrcompBatchedSnappyOpts>>);
static_assert(std::is_same_v<decltype(&wwrcompBatchedCascadedCompressAsync),
                             compress_async_t<wwrcompBatchedCascadedOpts>>);

static_assert(std::is_same_v<decltype(&wwrcompBatchedLZ4DecompressAsync), decompress_async_t>);
static_assert(std::is_same_v<decltype(&wwrcompBatchedSnappyDecompressAsync), decompress_async_t>);
static_assert(std::is_same_v<decltype(&wwrcompBatchedCascadedDecompressAsync), decompress_async_t>);

} // namespace wwr::test
