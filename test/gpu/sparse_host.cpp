// sparse_host.cpp - the non-module HOST #include path for the sparse surface
//
// The counterpart to sparse.cppm's module test, through the OTHER path a
// consumer has -- #include "wwr/sparse.h" from a plain, non-module TU -- proving
// that path yields the identical surface bound to the identical backend entities.
// Like the module test it samples rather than restating every typed entry point:
// the full surface is pinned name-by-name by the dispatch check, and this shares
// detail/sparse_names.h with the module, so what is under test here is the
// include PATH.
//
// A plain .cpp that imports nothing and links wwr::sparse::host, the header-only
// target, which carries WWR_GPU_BACKEND_* via wwr::backend.

#include <cstddef>
#include <type_traits>

#include "wwr/sparse.h"

#include "gpu_check_macros.h"

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)

WWR_SAME_TYPE(wwrsparseHandle_t, cusparseHandle_t)
WWR_SAME_TYPE(wwrsparseOperation_t, cusparseOperation_t)

WWR_SAME_VALUE(WWRSPARSE_STATUS_SUCCESS, CUSPARSE_STATUS_SUCCESS)
WWR_SAME_VALUE(WWRSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE)

// A handle call, a typed level-2 entry point, and a descriptor helper -- spanning
// the spellings sparse.h binds.
WWR_SAME_FUNCTION(wwrsparseCreate, cusparseCreate)
WWR_SAME_FUNCTION(wwrsparseSbsrmv, cusparseSbsrmv)
WWR_SAME_FUNCTION(wwrsparseCreateMatDescr, cusparseCreateMatDescr)

#else

WWR_SAME_TYPE(wwrsparseHandle_t, hipsparseHandle_t)
WWR_SAME_TYPE(wwrsparseOperation_t, hipsparseOperation_t)

WWR_SAME_VALUE(WWRSPARSE_STATUS_SUCCESS, HIPSPARSE_STATUS_SUCCESS)
WWR_SAME_VALUE(WWRSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE)

WWR_SAME_FUNCTION(wwrsparseCreate, hipsparseCreate)
WWR_SAME_FUNCTION(wwrsparseSbsrmv, hipsparseSbsrmv)
WWR_SAME_FUNCTION(wwrsparseCreateMatDescr, hipsparseCreateMatDescr)

#endif

// csr2gebsr_bufferSize is a WWR_FUNCTION_RAW reference on HIP and a hand-written
// int-forwarding shim on CUDA (see detail/sparse_names.h); its out parameter is
// hipSPARSE's std::size_t* on both, and the complex overload names
// wwrFloatComplex. Checking the signature exercises the shim through the #include
// path; taking its address forces it to link.
WWR_SAME_TYPE(std::remove_cvref_t<decltype(wwrsparseCcsr2gebsr_bufferSize)>,
              wwrsparseStatus_t(wwrsparseHandle_t, wwrsparseDirection_t, int, int,
                                const wwrsparseMatDescr_t, const wwrFloatComplex *, const int *,
                                const int *, int, int, std::size_t *))

namespace {
[[maybe_unused]] auto *const buffer_size = &wwr::wwrsparseCcsr2gebsr_bufferSize;
} // namespace

int main() { return 0; }
