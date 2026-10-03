// comp_host.cpp - the non-module HOST #include path for the comp surface
//
// The counterpart to comp.cppm's module test (test/gpu/comp.cppm), through the
// OTHER path a consumer has -- #include "wwr/comp.h" from a plain, non-module TU
// -- proving that path yields the identical surface bound to the identical
// backend entities. Like the module test, it samples: the full surface (types,
// enumerator values, every shim signature) is pinned name-by-name there, and this
// shares detail/comp_names.h with the module, so what is under test here is the
// include PATH.
//
// comp's functions are hand-written forwarding shims, not WWR_FUNCTION aliases
// (nvCOMP 5.3 vs hipCOMP 2.2 skew -- see comp.cppm), so there is no
// WWR_SAME_FUNCTION to restate; the neutral signature is locked with is_same_v
// and the symbol forced to link via address-of, exactly as the module test does.
//
// A plain .cpp that imports nothing and links wwr::comp::host, the header-only
// target, which carries WWR_GPU_BACKEND_* via wwr::backend.

#include <cstddef>
#include <type_traits>

#include "wwr/comp.h"

#include "gpu_check_macros.h"

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)

WWR_SAME_TYPE(wwrcompStatus_t, nvcompStatus_t)
WWR_SAME_TYPE(wwrcompStream_t, cudaStream_t)

WWR_SAME_VALUE(WWRCOMP_SUCCESS, nvcompSuccess)
WWR_SAME_VALUE(WWRCOMP_TYPE_CHAR, NVCOMP_TYPE_CHAR)

#else

WWR_SAME_TYPE(wwrcompStatus_t, hipcompStatus_t)
WWR_SAME_TYPE(wwrcompStream_t, hipStream_t)

WWR_SAME_VALUE(WWRCOMP_SUCCESS, hipcompSuccess)
WWR_SAME_VALUE(WWRCOMP_TYPE_CHAR, HIPCOMP_TYPE_CHAR)

#endif

// One compress + one decompress shim, locking the neutral (hipCOMP-2.2-shaped)
// parameter list the round-trip test relies on and forcing the forwarder to link
// (which pulls in the vendor global it calls, so the vendor library must resolve).
using lz4_compress_async_t = wwrcompStatus_t (*)(
    const void *const *, const std::size_t *, std::size_t, std::size_t, void *, std::size_t,
    void *const *, std::size_t *, wwrcompBatchedLZ4Opts, wwrcompStream_t);
using decompress_async_t = wwrcompStatus_t (*)(
    const void *const *, const std::size_t *, const std::size_t *, std::size_t *, std::size_t,
    void *, std::size_t, void *const *, wwrcompStatus_t *, wwrcompStream_t);

static_assert(std::is_same_v<decltype(&wwrcompBatchedLZ4CompressAsync), lz4_compress_async_t>);
static_assert(std::is_same_v<decltype(&wwrcompBatchedLZ4DecompressAsync), decompress_async_t>);

namespace {
[[maybe_unused]] auto *const lz4_compress = &wwr::wwrcompBatchedLZ4CompressAsync;
[[maybe_unused]] auto *const lz4_decompress = &wwr::wwrcompBatchedLZ4DecompressAsync;
} // namespace

int main() { return 0; }
