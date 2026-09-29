// fp4.cppm - Compile-time tests for wwr.fp4
//
// Every wwrFp4* name must be exactly the backend entity it stands for. The
// expected backend name is spelled out in full under one #if switch, so a
// mistake in the gpu* layer's WWR_SELECT macros cannot be mirrored here and
// pass. See gpu_check_macros.h.
//
// This test imports only the fp4 raw module, never hip_fp6 -- keeping the two
// apart is the point of docs/architecture.md section 11.

module;

#include "gpu_check_macros.h"

export module wwr.test.gpu.fp4;

import std;
import wwr.fp4;
#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.cuda_fp4;
#else
import wwr.hip.hip_fp4;
#endif

namespace wwr::test {

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)

// Storage typedefs
WWR_SAME_TYPE(wwr::wwrFp4Storage, wwr::cuda::__nv_fp4_storage_t)
WWR_SAME_TYPE(wwr::wwrFp4x2Storage, wwr::cuda::__nv_fp4x2_storage_t)
WWR_SAME_TYPE(wwr::wwrFp4x4Storage, wwr::cuda::__nv_fp4x4_storage_t)

// Interpretation enum (type and value)
WWR_SAME_TYPE(wwr::wwrFp4Interpretation, wwr::cuda::__nv_fp4_interpretation_t)
WWR_SAME_VALUE(wwr::wwrE2m1, wwr::cuda::__NV_E2M1)

// Rounding mode (type and value)
WWR_SAME_TYPE(wwr::wwrFp4RoundMode, wwr::cuda::cudaRoundMode)
WWR_SAME_VALUE(wwr::wwrFp4RoundNearest, wwr::cuda::cudaRoundNearest)
WWR_SAME_VALUE(wwr::wwrFp4RoundZero, wwr::cuda::cudaRoundZero)
WWR_SAME_VALUE(wwr::wwrFp4RoundPosInf, wwr::cuda::cudaRoundPosInf)
WWR_SAME_VALUE(wwr::wwrFp4RoundMinInf, wwr::cuda::cudaRoundMinInf)

// Scalar struct types
WWR_SAME_TYPE(wwr::wwrFp4E2m1, wwr::cuda::__nv_fp4_e2m1)
WWR_SAME_TYPE(wwr::wwrFp4x2E2m1, wwr::cuda::__nv_fp4x2_e2m1)
WWR_SAME_TYPE(wwr::wwrFp4x4E2m1, wwr::cuda::__nv_fp4x4_e2m1)

#else

WWR_SAME_TYPE(wwr::wwrFp4Storage, wwr::hip::__hip_fp4_storage_t)
WWR_SAME_TYPE(wwr::wwrFp4x2Storage, wwr::hip::__hip_fp4x2_storage_t)
WWR_SAME_TYPE(wwr::wwrFp4x4Storage, wwr::hip::__hip_fp4x4_storage_t)

WWR_SAME_TYPE(wwr::wwrFp4Interpretation, wwr::hip::__hip_fp4_interpretation_t)
WWR_SAME_VALUE(wwr::wwrE2m1, wwr::hip::__HIP_E2M1)

WWR_SAME_TYPE(wwr::wwrFp4RoundMode, wwr::hip::hipRoundMode)
WWR_SAME_VALUE(wwr::wwrFp4RoundNearest, wwr::hip::hipRoundNearest)
WWR_SAME_VALUE(wwr::wwrFp4RoundZero, wwr::hip::hipRoundZero)
WWR_SAME_VALUE(wwr::wwrFp4RoundPosInf, wwr::hip::hipRoundPosInf)
WWR_SAME_VALUE(wwr::wwrFp4RoundMinInf, wwr::hip::hipRoundMinInf)

WWR_SAME_TYPE(wwr::wwrFp4E2m1, wwr::hip::__hip_fp4_e2m1)
WWR_SAME_TYPE(wwr::wwrFp4x2E2m1, wwr::hip::__hip_fp4x2_e2m1)
WWR_SAME_TYPE(wwr::wwrFp4x4E2m1, wwr::hip::__hip_fp4x4_e2m1)

#endif

// fp4 packs two values per byte, so the scalar and x2 storage are both 1 byte
// and x4 is 2 bytes (see cuda_fp4.cppm / hip_fp4.cppm).
static_assert(sizeof(wwrFp4Storage) == 1);
static_assert(sizeof(wwrFp4x2Storage) == 1);
static_assert(sizeof(wwrFp4x4Storage) == 2);

// Narrowing forwarders -- see fp8.cppm on why WWR_LINK_CHECK, not
// WWR_SAME_FUNCTION.
WWR_LINK_CHECK(wwrFloat2Fp4)
WWR_LINK_CHECK(wwrDouble2Fp4)

static_assert(std::is_invocable_r_v<wwrFp4Storage, decltype(wwrFloat2Fp4), float,
                                    wwrFp4Interpretation, wwrFp4RoundMode>);
static_assert(std::is_invocable_r_v<wwrFp4Storage, decltype(wwrDouble2Fp4), double,
                                    wwrFp4Interpretation, wwrFp4RoundMode>);

} // namespace wwr::test
