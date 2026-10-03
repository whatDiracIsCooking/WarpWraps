// solver_host.cpp - the non-module HOST #include path for the solver surface
//
// The counterpart to solver.cppm's module test, through the OTHER path a
// consumer has -- #include "wwr/solver.h" from a plain, non-module TU -- proving
// that path yields the identical surface bound to the identical backend entities.
// Like the module test it samples rather than restating every typed entry point:
// the full surface is pinned name-by-name by the dispatch check, and this shares
// detail/solver_names.h with the module, so what is under test here is the
// include PATH.
//
// A plain .cpp that imports nothing and links wwr::solver::host, the header-only
// target, which carries WWR_GPU_BACKEND_* via wwr::backend.

#include "wwr/solver.h"

#include "gpu_check_macros.h"

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)

WWR_SAME_TYPE(wwrsolverDnHandle_t, cusolverDnHandle_t)
WWR_SAME_TYPE(wwrsolverDataType_t, cudaDataType)

WWR_SAME_VALUE(WWRSOLVER_STATUS_SUCCESS, CUSOLVER_STATUS_SUCCESS)
WWR_SAME_VALUE(WWRSOLVER_R_32F, CUDA_R_32F)

// A legacy typed entry point and a modern X-prefixed one -- spanning the two APIs
// solver.h binds.
WWR_SAME_FUNCTION(wwrsolverDnSpotrf, cusolverDnSpotrf)
WWR_SAME_FUNCTION(wwrsolverDnXpotrf, cusolverDnXpotrf)

#else

WWR_SAME_TYPE(wwrsolverDnHandle_t, hipsolverDnHandle_t)
WWR_SAME_TYPE(wwrsolverDataType_t, hipDataType)

WWR_SAME_VALUE(WWRSOLVER_STATUS_SUCCESS, HIPSOLVER_STATUS_SUCCESS)
WWR_SAME_VALUE(WWRSOLVER_R_32F, HIP_R_32F)

WWR_SAME_FUNCTION(wwrsolverDnSpotrf, hipsolverDnSpotrf)
WWR_SAME_FUNCTION(wwrsolverDnXpotrf, hipsolverDnXpotrf)

#endif

// wwrsolverGetStatusName/String are hand-written per-backend switches (see
// detail/solver_names.h), not WWR_FUNCTION_RAW aliases, so they cannot be
// WWR_SAME_FUNCTION'd; taking their address exercises the switch through the
// #include path and forces it to link.
namespace {
[[maybe_unused]] auto *const status_name = &wwr::wwrsolverGetStatusName;
[[maybe_unused]] auto *const status_string = &wwr::wwrsolverGetStatusString;
} // namespace

int main() { return 0; }
