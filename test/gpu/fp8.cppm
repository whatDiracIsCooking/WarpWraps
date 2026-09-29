// fp8.cppm - Compile-time tests for wwr.fp8
//
// Every wwrFp8* name must be exactly the backend entity it stands for: the same
// storage/scalar type, the same enum constant (type and value). The expected
// backend name is spelled out in full under one #if switch, so a mistake in the
// wwr* layer's WWR_SELECT macros cannot be mirrored here and pass. See
// gpu_check_macros.h.

module;

#include "gpu_check_macros.h"

export module wwr.test.gpu.fp8;

import std;
import wwr.fp8;
#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.cuda_fp8;
#else
import wwr.hip.hip_fp8;
#endif

namespace wwr::test {

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)

// Storage typedefs
WWR_SAME_TYPE(wwr::wwrFp8Storage, wwr::cuda::__nv_fp8_storage_t)
WWR_SAME_TYPE(wwr::wwrFp8x2Storage, wwr::cuda::__nv_fp8x2_storage_t)
WWR_SAME_TYPE(wwr::wwrFp8x4Storage, wwr::cuda::__nv_fp8x4_storage_t)

// Saturation and interpretation enums (type and value)
WWR_SAME_TYPE(wwr::wwrSaturation, wwr::cuda::__nv_saturation_t)
WWR_SAME_VALUE(wwr::wwrNosat, wwr::cuda::__NV_NOSAT)
WWR_SAME_VALUE(wwr::wwrSatfinite, wwr::cuda::__NV_SATFINITE)
WWR_SAME_TYPE(wwr::wwrFp8Interpretation, wwr::cuda::__nv_fp8_interpretation_t)
WWR_SAME_VALUE(wwr::wwrE4m3, wwr::cuda::__NV_E4M3)
WWR_SAME_VALUE(wwr::wwrE5m2, wwr::cuda::__NV_E5M2)

// Scalar struct types
WWR_SAME_TYPE(wwr::wwrFp8E4m3, wwr::cuda::__nv_fp8_e4m3)
WWR_SAME_TYPE(wwr::wwrFp8x2E4m3, wwr::cuda::__nv_fp8x2_e4m3)
WWR_SAME_TYPE(wwr::wwrFp8x4E4m3, wwr::cuda::__nv_fp8x4_e4m3)
WWR_SAME_TYPE(wwr::wwrFp8E5m2, wwr::cuda::__nv_fp8_e5m2)
WWR_SAME_TYPE(wwr::wwrFp8x2E5m2, wwr::cuda::__nv_fp8x2_e5m2)
WWR_SAME_TYPE(wwr::wwrFp8x4E5m2, wwr::cuda::__nv_fp8x4_e5m2)

#else

WWR_SAME_TYPE(wwr::wwrFp8Storage, wwr::hip::__hip_fp8_storage_t)
WWR_SAME_TYPE(wwr::wwrFp8x2Storage, wwr::hip::__hip_fp8x2_storage_t)
WWR_SAME_TYPE(wwr::wwrFp8x4Storage, wwr::hip::__hip_fp8x4_storage_t)

WWR_SAME_TYPE(wwr::wwrSaturation, wwr::hip::__hip_saturation_t)
WWR_SAME_VALUE(wwr::wwrNosat, wwr::hip::__HIP_NOSAT)
WWR_SAME_VALUE(wwr::wwrSatfinite, wwr::hip::__HIP_SATFINITE)
WWR_SAME_TYPE(wwr::wwrFp8Interpretation, wwr::hip::__hip_fp8_interpretation_t)
WWR_SAME_VALUE(wwr::wwrE4m3, wwr::hip::__HIP_E4M3)
WWR_SAME_VALUE(wwr::wwrE5m2, wwr::hip::__HIP_E5M2)

WWR_SAME_TYPE(wwr::wwrFp8E4m3, wwr::hip::__hip_fp8_e4m3)
WWR_SAME_TYPE(wwr::wwrFp8x2E4m3, wwr::hip::__hip_fp8x2_e4m3)
WWR_SAME_TYPE(wwr::wwrFp8x4E4m3, wwr::hip::__hip_fp8x4_e4m3)
WWR_SAME_TYPE(wwr::wwrFp8E5m2, wwr::hip::__hip_fp8_e5m2)
WWR_SAME_TYPE(wwr::wwrFp8x2E5m2, wwr::hip::__hip_fp8x2_e5m2)
WWR_SAME_TYPE(wwr::wwrFp8x4E5m2, wwr::hip::__hip_fp8x4_e5m2)

#endif

// Storage widths the neutral scalar types agree on with any caller and buffer.
static_assert(sizeof(wwrFp8Storage) == 1);
static_assert(sizeof(wwrFp8x2Storage) == 2);
static_assert(sizeof(wwrFp8x4Storage) == 4);
static_assert(sizeof(wwrFp8E4m3) == 1);
static_assert(sizeof(wwrFp8E5m2) == 1);

// The narrowing conversions are forwarding functions, not WWR_FUNCTION
// reference bindings, so &gpu != &backend and WWR_SAME_FUNCTION cannot apply --
// exactly as fp16.cppm's host wrappers. A bare WWR_LINK_CHECK is the build-time
// claim that the exported inline wrapper is reachable by name across the import
// and links; the is_invocable_v checks pin its neutral signature.
WWR_LINK_CHECK(wwrFloat2Fp8)
WWR_LINK_CHECK(wwrDouble2Fp8)

static_assert(std::is_invocable_r_v<wwrFp8Storage, decltype(wwrFloat2Fp8), float, wwrSaturation,
                                    wwrFp8Interpretation>);
static_assert(std::is_invocable_r_v<wwrFp8Storage, decltype(wwrDouble2Fp8), double, wwrSaturation,
                                    wwrFp8Interpretation>);

} // namespace wwr::test
