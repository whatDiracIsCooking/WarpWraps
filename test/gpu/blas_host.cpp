// blas_host.cpp - the non-module HOST #include path for the blas surface
//
// The counterpart to blas.cppm's module test, through the OTHER path a consumer
// has -- #include "wwr/blas.h" from a plain, non-module TU -- proving that path
// yields the identical surface bound to the identical backend entities. Like the
// module test (and unlike rand), it samples rather than restating every typed
// entry point: the full surface is pinned name-by-name by the dispatch check and
// alias_coverage.py, and this shares detail/blas_names.h with the module, so what
// is under test here is the include PATH.
//
// A plain .cpp that imports nothing and links wwr::blas::host, the header-only
// target, which carries WWR_GPU_BACKEND_* via wwr::backend.

#include <type_traits>

#include "wwr/blas.h"

#include "gpu_check_macros.h"

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)

WWR_SAME_TYPE(wwrblasHandle_t, cublasHandle_t)
WWR_SAME_TYPE(wwrblasOperation_t, cublasOperation_t)

WWR_SAME_VALUE(WWRBLAS_STATUS_SUCCESS, CUBLAS_STATUS_SUCCESS)
WWR_SAME_VALUE(WWRBLAS_OP_N, CUBLAS_OP_N)

// A _v2-paired name, a non-_v2 extension name, and a handle call -- spanning the
// spellings blas.h binds.
WWR_SAME_FUNCTION(wwrblasCreate, cublasCreate_v2)
WWR_SAME_FUNCTION(wwrblasSgemm, cublasSgemm_v2)
WWR_SAME_FUNCTION(wwrblasSgeam, cublasSgeam)
WWR_SAME_FUNCTION(wwrblasGetStatusName, cublasGetStatusName)

#else

WWR_SAME_TYPE(wwrblasHandle_t, hipblasHandle_t)
WWR_SAME_TYPE(wwrblasOperation_t, hipblasOperation_t)

WWR_SAME_VALUE(WWRBLAS_STATUS_SUCCESS, HIPBLAS_STATUS_SUCCESS)
WWR_SAME_VALUE(WWRBLAS_OP_N, HIPBLAS_OP_N)

WWR_SAME_FUNCTION(wwrblasCreate, hipblasCreate)
WWR_SAME_FUNCTION(wwrblasSgemm, hipblasSgemm)
WWR_SAME_FUNCTION(wwrblasSgeam, hipblasSgeam)
// hipBLAS has one status-to-string function; both neutral names map to it.
WWR_SAME_FUNCTION(wwrblasGetStatusName, hipblasStatusToString)

#endif

// getrsBatched is a WWR_FUNCTION_RAW reference on CUDA and a hand-written
// const-correct forwarder on HIP (see detail/blas_names.h); its signature is
// cuBLAS's const-correct one on both. Checking the signature exercises the
// forwarder through the #include path; taking its address forces it to link.
WWR_SAME_TYPE(std::remove_cvref_t<decltype(wwrblasSgetrsBatched)>,
              wwrblasStatus_t(wwrblasHandle_t, wwrblasOperation_t, int, int, const float *const[],
                              int, const int *, float *const[], int, int *, int))

namespace {
[[maybe_unused]] auto *const getrs = &wwr::wwrblasSgetrsBatched;
} // namespace

int main() { return 0; }
