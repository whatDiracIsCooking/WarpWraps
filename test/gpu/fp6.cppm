// fp6.cppm - Compile-time tests for wwr.fp6
//
// Every wwrFp6* name must be exactly the backend entity it stands for. The
// expected backend name is spelled out in full under one #if switch, so a
// mistake in the wwr* layer's WWR_SELECT macros cannot be mirrored here and
// pass. See gpu_check_macros.h.

module;

#include "gpu_check_macros.h"

export module wwr.test.gpu.fp6;

import std;
import wwr.fp6;
#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.cuda_fp6;
#else
import wwr.hip.hip_fp6;
#endif

namespace wwr::test {

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)

// Storage typedefs
WWR_SAME_TYPE(wwr::wwrFp6Storage, wwr::cuda::__nv_fp6_storage_t)
WWR_SAME_TYPE(wwr::wwrFp6x2Storage, wwr::cuda::__nv_fp6x2_storage_t)
WWR_SAME_TYPE(wwr::wwrFp6x4Storage, wwr::cuda::__nv_fp6x4_storage_t)

// Interpretation enum (type and value)
WWR_SAME_TYPE(wwr::wwrFp6Interpretation, wwr::cuda::__nv_fp6_interpretation_t)
WWR_SAME_VALUE(wwr::wwrE3m2, wwr::cuda::__NV_E3M2)
WWR_SAME_VALUE(wwr::wwrE2m3, wwr::cuda::__NV_E2M3)

// Rounding mode (type and value)
WWR_SAME_TYPE(wwr::wwrFp6RoundMode, wwr::cuda::cudaRoundMode)
WWR_SAME_VALUE(wwr::wwrFp6RoundNearest, wwr::cuda::cudaRoundNearest)
WWR_SAME_VALUE(wwr::wwrFp6RoundZero, wwr::cuda::cudaRoundZero)
WWR_SAME_VALUE(wwr::wwrFp6RoundPosInf, wwr::cuda::cudaRoundPosInf)
WWR_SAME_VALUE(wwr::wwrFp6RoundMinInf, wwr::cuda::cudaRoundMinInf)

// Scalar struct types
WWR_SAME_TYPE(wwr::wwrFp6E3m2, wwr::cuda::__nv_fp6_e3m2)
WWR_SAME_TYPE(wwr::wwrFp6x2E3m2, wwr::cuda::__nv_fp6x2_e3m2)
WWR_SAME_TYPE(wwr::wwrFp6x4E3m2, wwr::cuda::__nv_fp6x4_e3m2)
WWR_SAME_TYPE(wwr::wwrFp6E2m3, wwr::cuda::__nv_fp6_e2m3)
WWR_SAME_TYPE(wwr::wwrFp6x2E2m3, wwr::cuda::__nv_fp6x2_e2m3)
WWR_SAME_TYPE(wwr::wwrFp6x4E2m3, wwr::cuda::__nv_fp6x4_e2m3)

#else

WWR_SAME_TYPE(wwr::wwrFp6Storage, wwr::hip::__hip_fp6_storage_t)
WWR_SAME_TYPE(wwr::wwrFp6x2Storage, wwr::hip::__hip_fp6x2_storage_t)
WWR_SAME_TYPE(wwr::wwrFp6x4Storage, wwr::hip::__hip_fp6x4_storage_t)

WWR_SAME_TYPE(wwr::wwrFp6Interpretation, wwr::hip::__hip_fp6_interpretation_t)
WWR_SAME_VALUE(wwr::wwrE3m2, wwr::hip::__HIP_E3M2)
WWR_SAME_VALUE(wwr::wwrE2m3, wwr::hip::__HIP_E2M3)

WWR_SAME_TYPE(wwr::wwrFp6RoundMode, wwr::hip::hipRoundMode)
WWR_SAME_VALUE(wwr::wwrFp6RoundNearest, wwr::hip::hipRoundNearest)
WWR_SAME_VALUE(wwr::wwrFp6RoundZero, wwr::hip::hipRoundZero)
WWR_SAME_VALUE(wwr::wwrFp6RoundPosInf, wwr::hip::hipRoundPosInf)
WWR_SAME_VALUE(wwr::wwrFp6RoundMinInf, wwr::hip::hipRoundMinInf)

WWR_SAME_TYPE(wwr::wwrFp6E3m2, wwr::hip::__hip_fp6_e3m2)
WWR_SAME_TYPE(wwr::wwrFp6x2E3m2, wwr::hip::__hip_fp6x2_e3m2)
WWR_SAME_TYPE(wwr::wwrFp6x4E3m2, wwr::hip::__hip_fp6x4_e3m2)
WWR_SAME_TYPE(wwr::wwrFp6E2m3, wwr::hip::__hip_fp6_e2m3)
WWR_SAME_TYPE(wwr::wwrFp6x2E2m3, wwr::hip::__hip_fp6x2_e2m3)
WWR_SAME_TYPE(wwr::wwrFp6x4E2m3, wwr::hip::__hip_fp6x4_e2m3)

#endif

static_assert(sizeof(wwrFp6Storage) == 1);
static_assert(sizeof(wwrFp6x2Storage) == 2);
static_assert(sizeof(wwrFp6x4Storage) == 4);
static_assert(sizeof(wwrFp6E3m2) == 1);
static_assert(sizeof(wwrFp6E2m3) == 1);

// Narrowing forwarders -- see fp8.cppm on why WWR_LINK_CHECK, not
// WWR_SAME_FUNCTION.
WWR_LINK_CHECK(wwrFloat2Fp6)
WWR_LINK_CHECK(wwrDouble2Fp6)

static_assert(std::is_invocable_r_v<wwrFp6Storage, decltype(wwrFloat2Fp6), float,
                                    wwrFp6Interpretation, wwrFp6RoundMode>);
static_assert(std::is_invocable_r_v<wwrFp6Storage, decltype(wwrDouble2Fp6), double,
                                    wwrFp6Interpretation, wwrFp6RoundMode>);

} // namespace wwr::test
