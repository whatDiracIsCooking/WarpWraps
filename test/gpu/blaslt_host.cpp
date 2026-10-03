// blaslt_host.cpp - the non-module HOST #include path for the blaslt surface
//
// The counterpart to blaslt.cppm's module test, through the OTHER path a
// consumer has -- #include "wwr/blaslt.h" from a plain, non-module TU -- proving
// that path yields the identical surface bound to the identical backend
// entities. Like the module test it samples rather than restating every entry
// point: the full surface is pinned name-by-name by the module test
// (test/gpu/blaslt.cppm), and this shares detail/blaslt_names.h with the module,
// so what is under test here is the include PATH.
//
// A plain .cpp that imports nothing and links wwr::blaslt::host, the header-only
// target, which carries WWR_GPU_BACKEND_* via wwr::backend.

#include "wwr/blaslt.h"

#include "gpu_check_macros.h"

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)

WWR_SAME_TYPE(wwrblasLtHandle_t, cublasLtHandle_t)
WWR_SAME_TYPE(wwrblasLtStatus_t, cublasStatus_t)

WWR_SAME_VALUE(WWRBLASLT_EPILOGUE_DEFAULT, CUBLASLT_EPILOGUE_DEFAULT)
WWR_SAME_VALUE(WWRBLASLT_ORDER_COL, CUBLASLT_ORDER_COL)

WWR_SAME_FUNCTION(wwrblasLtCreate, cublasLtCreate)
WWR_SAME_FUNCTION(wwrblasLtMatmul, cublasLtMatmul)

#else

WWR_SAME_TYPE(wwrblasLtHandle_t, hipblasLtHandle_t)
WWR_SAME_TYPE(wwrblasLtStatus_t, hipblasStatus_t)

WWR_SAME_VALUE(WWRBLASLT_EPILOGUE_DEFAULT, HIPBLASLT_EPILOGUE_DEFAULT)
WWR_SAME_VALUE(WWRBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL)

WWR_SAME_FUNCTION(wwrblasLtCreate, hipblasLtCreate)
WWR_SAME_FUNCTION(wwrblasLtMatmul, hipblasLtMatmul)

#endif

int main() { return 0; }
